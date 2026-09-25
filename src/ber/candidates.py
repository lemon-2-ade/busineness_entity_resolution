"""Stage 2 driver: run blocking for a split and cache the candidate table.

    python -m ber.candidates --art artifacts --split train|test
Output: artifacts/cands_<split>.parquet with columns
    i1, i23 (row indices into <split>_s1 / concat(<split>_s2, <split>_s3)),
    blk_name, blk_addr, blk_joint
"""
import argparse
import os
import time

import polars as pl

from .blocking import BLOCK_COLS, DEFAULTS, generate_candidates


def load_split(art, split, cols=None):
    s1 = pl.read_parquet(f'{art}/{split}_s1.parquet', columns=cols)
    s23 = pl.concat([pl.read_parquet(f'{art}/{split}_s{i}.parquet', columns=cols) for i in (2, 3)])
    return s1, s23


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--art', default='artifacts')
    ap.add_argument('--split', default='test')
    ap.add_argument('--threads', type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    t = time.time()
    s1, s23 = load_split(a.art, a.split, BLOCK_COLS)
    c = generate_candidates(s1, s23, threads=a.threads, **DEFAULTS)
    c.write_parquet(f'{a.art}/cands_{a.split}.parquet')
    print(f'{a.split}: {c.height} candidate pairs for {s1.height} S1 ({c.height / s1.height:.1f}/S1) '
          f'in {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main()
