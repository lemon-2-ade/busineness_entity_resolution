"""Stage 4: train the pairwise matcher and tune post-processing on a held-out
validation split of the training data.

    python -m ber.train --art artifacts --n-train 300000

Validation design: S1 entities are split *by state* (whole geo-blocks are
held out).  Blocking only compares records inside the same state block (+ the
small no-state residual pool), so a held-out state is an almost closed world:
all S1 entities competing for its S2/S3 records are also in validation,
which makes the exclusivity step and the macro-F0.5 estimate honest.  It is
also a harder, "new geography" test, which is what the unseen-country
(France) part of the test set needs.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from .features import REC_COLS, context_features, pair_features, raw_flags
from .io import read_ground_truth
from .metrics import breakdown
from .postprocess import exclusive, select_expected_f, select_threshold, to_lists

NON_FEATURES = {'i1', 'i23', 'y', 'prob', 'fold'}


def load_records(art, split):
    cols = [c for c in REC_COLS if c != 'rflags'] + ['business_name']
    s1 = raw_flags(pl.read_parquet(f'{art}/{split}_s1.parquet', columns=cols + ['country']))
    s23 = pl.concat([raw_flags(pl.read_parquet(f'{art}/{split}_s{i}.parquet', columns=cols)) for i in (2, 3)])
    return s1, s23


def featurise(c, s1, s23, workers, chunk=1_000_000, log=True, emb=None):
    parts = []
    t = time.time()
    for i in range(0, c.height, chunk):
        part = pair_features(c.slice(i, chunk), s1, s23, workers)
        if emb is not None:
            from .embed import pair_cos
            part = part.with_columns(pl.Series('emb_name_cos', pair_cos(emb, part['i1'].to_numpy(),
                                                                         part['i23'].to_numpy())))
        parts.append(part)
        if log:
            print(f'    features {min(i + chunk, c.height)}/{c.height} {time.time() - t:.0f}s', flush=True)
    return pl.concat(parts)


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--art', default='artifacts')
    ap.add_argument('--data', default='data')
    ap.add_argument('--n-train', type=int, default=300_000, help='#train S1 entities to featurise')
    ap.add_argument('--valid-frac', type=float, default=0.12, help='fraction of S1 (by state) held out')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--emb', action='store_true', help='add the optional GPU name-embedding feature')
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)

    s1, s23 = load_records(a.art, 'train')
    c = pl.read_parquet(f'{a.art}/cands_train.parquet')
    c = context_features(c)
    print(f'candidates {c.height} ({time.time() - t0:.0f}s)', flush=True)

    # labels
    _, pairs = read_ground_truth(f'{a.data}/train/train_ground_truth.tsv')
    e1 = s1['entity_id'].to_numpy(); e23 = s23['entity_id'].to_numpy()
    idx1 = pl.DataFrame({'entity_id': e1, 'i1': np.arange(len(e1), dtype=np.int32)})
    idx23 = pl.DataFrame({'entity_id': e23, 'i23': np.arange(len(e23), dtype=np.int32)})
    tp = (pairs.join(idx1, left_on='source1_entity_id', right_on='entity_id')
          .join(idx23, left_on='matched_entity_id', right_on='entity_id').select('i1', 'i23'))
    c = c.join(tp.with_columns(pl.lit(1, dtype=pl.Int8).alias('y')), on=['i1', 'i23'], how='left') \
         .with_columns(pl.col('y').fill_null(0))
    truth = {}
    for x, y in zip(pairs['source1_entity_id'], pairs['matched_entity_id']):
        truth.setdefault(x, set()).add(y)
    print(f'candidate pair recall (ceiling): {c["y"].sum() / tp.height:.4f}', flush=True)

    # ---- split by state (geo blocks)
    st = s1.select(pl.col('state').str.split(' ').list.first().fill_null('').alias('g'), 'country')
    groups = st.group_by('g', 'country').len().sort('g')
    groups = groups.sample(fraction=1.0, shuffle=True, seed=a.seed)
    tot = s1.height; acc = 0; valid_groups = set()
    for g, cn, n in groups.iter_rows():
        if g == '':
            continue
        if acc + n <= a.valid_frac * tot:
            valid_groups.add(g); acc += n
    is_valid = st['g'].is_in(list(valid_groups)).to_numpy()
    print(f'valid: {is_valid.sum()} S1 in {len(valid_groups)} states: {sorted(valid_groups)}', flush=True)
    tr_ids = np.nonzero(~is_valid)[0]
    tr_ids = rng.choice(tr_ids, size=min(a.n_train, len(tr_ids)), replace=False)
    va_ids = np.nonzero(is_valid)[0]

    ctr = c.filter(pl.col('i1').is_in(tr_ids))
    cva = c.filter(pl.col('i1').is_in(va_ids))
    del c
    print(f'train pairs {ctr.height}, valid pairs {cva.height}', flush=True)
    emb = None
    if a.emb:
        from .embed import load as load_emb
        emb = load_emb(a.art, 'train')
        assert emb is not None, 'run `python -m ber.embed --split train` first'
    ctr = featurise(ctr, s1, s23, a.workers, emb=emb)
    cva = featurise(cva, s1, s23, a.workers, emb=emb)
    feats = feature_cols(ctr)
    print(f'{len(feats)} features', flush=True)

    params = dict(objective='binary', learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, num_threads=a.workers, verbose=-1, seed=a.seed)
    dtr = lgb.Dataset(ctr.select(feats).to_numpy(), ctr['y'].to_numpy(), feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(cva.select(feats).to_numpy(), cva['y'].to_numpy(), reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], valid_names=['valid'],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    os.makedirs(f'{a.art}/model', exist_ok=True)
    model.save_model(f'{a.art}/model/lgb.txt')
    imp = sorted(zip(feats, model.feature_importance('gain')), key=lambda x: -x[1])
    print('top features:', [(f, round(g)) for f, g in imp[:25]], flush=True)

    # ---- tune post-processing on validation
    cva = cva.with_columns(pl.Series('prob', model.predict(cva.select(feats).to_numpy()).astype(np.float32)))
    cva.select('i1', 'i23', 'y', 'prob').write_parquet(f'{a.art}/valid_scored.parquet')
    va_s1 = e1[va_ids]
    ex = exclusive(cva)
    results = []
    for thr in np.arange(0.3, 0.95, 0.05):
        pred = to_lists(select_threshold(ex, thr), e1, e23)
        results.append(('thr', round(float(thr), 2), breakdown(pred, truth, va_s1)))
    for mp in (0.0, 0.05, 0.2, 0.5):
        for mn in (0.05, 0.2, 0.35):
            pred = to_lists(select_expected_f(ex, miss_prior=mp, min_prob=mn), e1, e23)
            results.append(('ef', (mp, mn), breakdown(pred, truth, va_s1)))
    for r in results:
        print(r[0], r[1], {k: round(v, 4) if isinstance(v, float) else v for k, v in r[2].items()})
    best = max(results, key=lambda r: r[2]['macro_f05'])
    print('BEST', best, flush=True)
    cfg = {'features': feats, 'best_iteration': model.best_iteration,
           'post': {'method': best[0], 'param': best[1]}, 'valid': best[2]}
    json.dump(cfg, open(f'{a.art}/model/config.json', 'w'), indent=1, default=str)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
