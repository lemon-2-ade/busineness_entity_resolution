"""Measure blocking recall ceiling / candidate volume on a sample of train S1.
python scripts/eval_blocking.py --n 50000"""
import argparse, sys, time
sys.path.insert(0, 'src')
import polars as pl
from ber.blocking import generate_candidates
from ber.io import read_ground_truth

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=50000); ap.add_argument('--art', default='artifacts')
ap.add_argument('--kn', type=int, default=15); ap.add_argument('--ka', type=int, default=15); ap.add_argument('--kj', type=int, default=25)
ap.add_argument('--maxc', type=int, default=30)
a = ap.parse_args()
s1 = pl.read_parquet(f'{a.art}/train_s1.parquet').sample(a.n, seed=0)
s23 = pl.concat([pl.read_parquet(f'{a.art}/train_s2.parquet'), pl.read_parquet(f'{a.art}/train_s3.parquet')])
_, pairs = read_ground_truth('data/train/train_ground_truth.tsv')
t = time.time()
c = generate_candidates(s1, s23, a.kn, a.ka, a.kj, a.maxc)
print('time', time.time() - t)
c = c.with_columns(pl.Series('s1', s1['entity_id'].to_numpy()[c['i1'].to_numpy()]), pl.Series('s23', s23['entity_id'].to_numpy()[c['i23'].to_numpy()]))
tp = pairs.filter(pl.col('source1_entity_id').is_in(s1['entity_id'].implode()))
hit = tp.join(c, left_on=['source1_entity_id', 'matched_entity_id'], right_on=['s1', 's23'], how='left')
print('true pairs', tp.height, 'recall', hit['blk_joint'].is_not_null().mean(), 'cands/S1', c.height / a.n)
c = c.join(tp.with_columns(pl.lit(1).alias('y')), left_on=['s1', 's23'], right_on=['source1_entity_id', 'matched_entity_id'], how='left').fill_null(0)
for m in (5, 10, 15, 20, 25, 30):
    cc = c.sort(['i1', 'blk_joint'], descending=[False, True]).group_by('i1', maintain_order=True).head(m)
    print(f' top{m} by joint: recall {cc["y"].sum()/tp.height:.4f}')
miss = hit.filter(pl.col('blk_joint').is_null()).head(3000)
miss.write_parquet('/tmp/claude-0/miss.parquet')
