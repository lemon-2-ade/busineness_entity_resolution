"""Measure blocking recall ceiling / candidate volume on a random sample of
train S1 (S1 rows are retrieved independently, so a sample gives an unbiased
recall estimate).       python scripts/eval_blocking.py --n 50000"""
import argparse, sys, time
sys.path.insert(0, 'src')
import polars as pl
from ber.candidates import run
from ber.io import read_ground_truth

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=50000); ap.add_argument('--art', default='artifacts')
ap.add_argument('--capn', type=int, default=5000); ap.add_argument('--capa', type=int, default=5000)
a = ap.parse_args()
ids = pl.read_parquet(f'{a.art}/train_s1.parquet', columns=['entity_id']).sample(a.n, seed=0)['entity_id']
t = time.time()
c = run(a.art, 'train', 2, s1_filter=pl.col('entity_id').is_in(ids.implode()), cap_name=a.capn, cap_addr=a.capa)
print('time', time.time() - t)
e1 = pl.read_parquet(f'{a.art}/train_s1.parquet', columns=['entity_id'])['entity_id'].to_numpy()
e23 = pl.concat([pl.read_parquet(f'{a.art}/train_s{i}.parquet', columns=['entity_id']) for i in (2, 3)])['entity_id'].to_numpy()
c = c.with_columns(pl.Series('s1', e1[c['i1'].to_numpy()]), pl.Series('s23', e23[c['i23'].to_numpy()]))
_, pairs = read_ground_truth('data/train/train_ground_truth.tsv')
tp = pairs.filter(pl.col('source1_entity_id').is_in(ids.implode()))
hit = tp.join(c, left_on=['source1_entity_id', 'matched_entity_id'], right_on=['s1', 's23'], how='left')
print('true pairs', tp.height, 'pair recall', round(hit['blk_joint'].is_not_null().mean(), 4), 'cands/S1', round(c.height / a.n, 1))
