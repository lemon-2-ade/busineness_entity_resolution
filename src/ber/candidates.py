"""Stage 2 driver: run blocking for a split and cache the candidate table.

    python -m ber.candidates --art artifacts --split train|test
Output: artifacts/cands_<split>.parquet with columns
    i1, i23 (row indices into <split>_s1 / concat(<split>_s2, <split>_s3)),
    blk_name, blk_addr, blk_joint, rk_joint, rk_name, rk_addr

Countries are processed one at a time and only the blocking columns of that
country are loaded (lazy parquet scan), which keeps peak memory ~4-5 GB on
the 10M-record sources.  The country label is treated as an opaque string.
"""
import argparse
import gc
import os
import time

import polars as pl

from .blocking import BLOCK_COLS, DEFAULTS, generate_candidates


def scan_split(art, split):
    """Lazy frames with global row ids (_i1 / _i23)."""
    s1 = pl.scan_parquet(f'{art}/{split}_s1.parquet').select(BLOCK_COLS).with_row_index('_i1')
    s23 = pl.concat([pl.scan_parquet(f'{art}/{split}_s{i}.parquet').select(BLOCK_COLS) for i in (2, 3)]) \
        .with_row_index('_i23')
    return s1, s23


def run(art, split, threads, s1_filter=None, **kw):
    s1, s23 = scan_split(art, split)
    if s1_filter is not None:
        s1 = s1.filter(s1_filter)
    countries = sorted(s1.select('country').unique().collect()['country'].to_list())
    out = []
    for cn in countries:
        a = s1.filter(pl.col('country') == cn).collect()
        b = s23.filter(pl.col('country') == cn).collect()
        u = generate_candidates(a, b, threads=threads, **{**DEFAULTS, **kw})
        del a, b
        gc.collect()
        if u is not None:
            out.append(u)
    return pl.concat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--art', default='artifacts')
    ap.add_argument('--split', default='test')
    ap.add_argument('--threads', type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    t = time.time()
    c = run(a.art, a.split, a.threads)
    c.write_parquet(f'{a.art}/cands_{a.split}.parquet')
    n1 = pl.scan_parquet(f'{a.art}/{a.split}_s1.parquet').select(pl.len()).collect().item()
    print(f'{a.split}: {c.height} candidate pairs for {n1} S1 ({c.height / n1:.1f}/S1) '
          f'in {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main()
