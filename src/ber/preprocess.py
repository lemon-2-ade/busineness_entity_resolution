"""Stage 1: normalise every record of every source once and cache as parquet.

    python -m ber.preprocess --data data --out artifacts [--workers 4]

Train dictionaries (native-script -> English) are learned from the *training*
ground truth only and then applied identically to train and test.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import gc
from multiprocessing import get_context

import polars as pl

from .io import read_ground_truth, read_source
from .normalize import Normalizer, build_dicts_from_train

NAME_COLS = ['name_norm', 'name_core', 'name_alt', 'name_compact', 'legal', 'nflags']
ADDR_COLS = ['addr_norm', 'addr_core', 'addr_nums', 'state']

_N: Normalizer | None = None


def _init(translit, seg):
    global _N
    _N = Normalizer(translit, seg)


def _work(args):
    names, addrs = args
    out_n = [_N.name(x) for x in names]
    out_a = [_N.address(x) for x in addrs]
    return out_n, out_a


def _to_frame(rn, ra):
    cols = {c: [r[i] for r in rn] for i, c in enumerate(NAME_COLS)}
    cols.update({c: [r[i] for r in ra] for i, c in enumerate(ADDR_COLS)})
    return pl.DataFrame(cols, schema_overrides={'nflags': pl.Int8})


def normalise_frame(df: pl.DataFrame, translit, seg, workers=2, chunk=100_000) -> pl.DataFrame:
    """Normalise in chunks; each chunk is turned into a small frame right away
    so peak memory stays low on 10M-row sources."""
    def jobs():
        for i in range(0, df.height, chunk):
            sl = df.slice(i, chunk)
            yield sl['business_name'].to_list(), sl['business_address'].to_list()
    parts = []
    if workers > 1:
        with get_context('spawn').Pool(workers, initializer=_init, initargs=(translit, seg)) as p:
            for rn, ra in p.imap(_work, jobs()):
                parts.append(_to_frame(rn, ra))
    else:
        _init(translit, seg)
        for j in jobs():
            parts.append(_to_frame(*_work(j)))
    extra = pl.concat(parts)
    return pl.concat([df.select('entity_id', 'country', 'business_name', 'business_address'), extra], how='horizontal_extend')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--out', default='artifacts')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument('--splits', default='train,test')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dict_path = os.path.join(a.out, 'dicts.json')
    if os.path.exists(dict_path):
        d = json.load(open(dict_path)); translit, seg = d['translit'], d['seg']
    else:
        t = time.time()
        s1 = read_source(f'{a.data}/train/train_source1.tsv')
        s23 = pl.concat([read_source(f'{a.data}/train/train_source2.tsv'), read_source(f'{a.data}/train/train_source3.tsv')])
        _, pairs = read_ground_truth(f'{a.data}/train/train_ground_truth.tsv')
        translit, seg = build_dicts_from_train(s1, s23, pairs)
        json.dump({'translit': translit, 'seg': seg}, open(dict_path, 'w'), ensure_ascii=False)
        print(f'learned {len(translit)} token / {len(seg)} segment mappings in {time.time()-t:.0f}s', flush=True)
        del s1, s23, pairs
    for split in a.splits.split(','):
        for src in (1, 2, 3):
            outp = os.path.join(a.out, f'{split}_s{src}.parquet')
            if os.path.exists(outp):
                continue
            t = time.time()
            df = read_source(f'{a.data}/{split}/{split}_source{src}.tsv')
            nd = normalise_frame(df, translit, seg, a.workers)
            nd.write_parquet(outp)
            print(f'{split} s{src}: {nd.height} rows in {time.time()-t:.0f}s', flush=True)
            del df, nd
            gc.collect()


if __name__ == '__main__':
    main()
