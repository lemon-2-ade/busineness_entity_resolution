"""Stage 4: train the pairwise matcher and tune post-processing on a held-out
validation split of the training data.

    python -m ber.train --art artifacts --n-train 250000

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
import gc
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


def entity_ids(art, split):
    """Entity-id Series (Arrow strings - far lighter than numpy object arrays)."""
    e1 = pl.read_parquet(f'{art}/{split}_s1.parquet', columns=['entity_id'])['entity_id']
    e23 = pl.concat([pl.read_parquet(f'{art}/{split}_s{i}.parquet', columns=['entity_id'])
                     for i in (2, 3)])['entity_id']
    return e1, e23


class RecordStore:
    """Feature columns of every record of a source, stored as uncompressed
    Arrow IPC and memory-mapped: gathering the rows referenced by a chunk of
    candidates touches only those pages, so RAM stays flat even for the
    10M-record S2+S3 tables (built once from the parquet cache)."""

    def __init__(self, art, split, sources):
        self.frames, self.offsets = [], [0]
        cols = [x for x in REC_COLS if x != 'rflags'] + ['business_name']
        for src in sources:
            path = f'{art}/{split}_s{src}.rec.arrow'
            if not os.path.exists(path):
                raw_flags(pl.read_parquet(f'{art}/{split}_s{src}.parquet', columns=cols)) \
                    .write_ipc(path, compression='uncompressed')
                gc.collect()
            f = pl.read_ipc(path, memory_map=True)
            self.frames.append(f)
            self.offsets.append(self.offsets[-1] + f.height)

    def take(self, rows: np.ndarray) -> pl.DataFrame:
        """Rows (global numbering over the sources) in the given order."""
        rows = np.asarray(rows, dtype=np.int64)
        if len(self.frames) == 1:
            return self.frames[0][rows]
        which = np.searchsorted(self.offsets, rows, side='right') - 1
        parts, pos = [], []
        for k, f in enumerate(self.frames):
            m = np.nonzero(which == k)[0]
            if len(m):
                parts.append(f[rows[m] - self.offsets[k]]); pos.append(m)
        out = pl.concat(parts)
        return out[np.argsort(np.concatenate(pos), kind='stable')]


def featurise(c: pl.DataFrame, art, split, workers, chunk=1_000_000, log=True, emb=None, keep=('y',)):
    """Pair features for candidate frame `c` (global i1/i23 indices), computed
    chunk by chunk; for each chunk only the records it references are loaded.
    Returns (meta frame [i1, i23, *keep], float32 feature matrix, names)."""
    st1 = RecordStore(art, split, (1,)); st23 = RecordStore(art, split, (2, 3))
    metas, mats, names = [], [], None
    t = time.time()
    for i in range(0, c.height, chunk):
        ch = c.slice(i, chunk)
        u1 = np.unique(ch['i1'].to_numpy()); u23 = np.unique(ch['i23'].to_numpy())
        s1 = st1.take(u1); s23 = st23.take(u23)
        loc = ch.with_columns(pl.Series('i1', np.searchsorted(u1, ch['i1'].to_numpy()).astype(np.int32)),
                              pl.Series('i23', np.searchsorted(u23, ch['i23'].to_numpy()).astype(np.int32)))
        part = pair_features(loc, s1, s23, workers)
        del s1, s23, loc
        part = part.with_columns(ch['i1'], ch['i23'])   # back to global ids
        if emb is not None:
            from .embed import pair_cos
            part = part.with_columns(pl.Series('emb_name_cos', pair_cos(emb, part['i1'].to_numpy(),
                                                                         part['i23'].to_numpy())))
        if names is None:
            names = feature_cols(part)
        metas.append(part.select(['i1', 'i23'] + [k for k in keep if k in part.columns]))
        mats.append(part.select(names).to_numpy().astype(np.float32))
        del part
        gc.collect()
        if log:
            print(f'    features {min(i + chunk, c.height)}/{c.height} {time.time() - t:.0f}s', flush=True)
    return pl.concat(metas), np.concatenate(mats), names


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def candidates_with_context(art, split, keep_i1):
    """Candidate rows of the S1 entities `keep_i1`, with context features.
    S2/S3-side competition is measured against the *full* candidate graph
    (all S1 that retrieved the same record), so train/test semantics agree.
    Memory: only a lean (i1, i23, blk_joint) view of the full graph is read."""
    path = f'{art}/cands_{split}.parquet'
    lean = pl.read_parquet(path, columns=['i1', 'i23', 'blk_joint'])
    i1 = lean['i1'].to_numpy(); i23 = lean['i23'].to_numpy(); sc = lean['blk_joint'].to_numpy()
    del lean
    want = np.zeros(int(i1.max()) + 1, dtype=bool); want[np.asarray(keep_i1)] = True
    kmask = want[i1]
    want23 = np.zeros(int(i23.max()) + 1, dtype=bool); want23[i23[kmask]] = True
    cmask = want23[i23]                     # rows competing for the same S2/S3 records
    c = context_features(i1[cmask], i23[cmask], sc[cmask], keep_mask=kmask[cmask])
    del i1, i23, sc
    k = pl.Series(np.asarray(keep_i1, dtype=np.int32)).implode()
    rest = pl.scan_parquet(path).filter(pl.col('i1').is_in(k)).drop('blk_joint').collect()
    return c.join(rest, on=['i1', 'i23'], how='left')


def labelled_candidates(art, data, split='train', keep_i1=None, truth_ids=None):
    """Candidate table of the split with context features and labels."""
    e1, e23 = entity_ids(art, split)
    c = candidates_with_context(art, split, keep_i1)
    _, pairs = read_ground_truth(f'{data}/{split}/{split}_ground_truth.tsv')
    idx1 = pl.DataFrame({'entity_id': e1, 'i1': np.arange(len(e1), dtype=np.int32)})
    idx23 = pl.DataFrame({'entity_id': e23, 'i23': np.arange(len(e23), dtype=np.int32)})
    tp = (pairs.join(idx1, left_on='source1_entity_id', right_on='entity_id')
          .join(idx23, left_on='matched_entity_id', right_on='entity_id').select('i1', 'i23'))
    if keep_i1 is not None:
        tp = tp.filter(pl.col('i1').is_in(pl.Series(np.asarray(keep_i1, dtype=np.int32)).implode()))
    c = c.join(tp.with_columns(pl.lit(1, dtype=pl.Int8).alias('y')), on=['i1', 'i23'], how='left') \
         .with_columns(pl.col('y').fill_null(0))
    truth = {}
    if truth_ids is not None:
        pt = pairs.filter(pl.col('source1_entity_id').is_in(pl.Series(truth_ids).implode()))
        for x, y in zip(pt['source1_entity_id'].to_list(), pt['matched_entity_id'].to_list()):
            truth.setdefault(x, set()).add(y)
    del pairs
    print(f'{c.height} candidates; pair recall ceiling {c["y"].sum() / tp.height:.4f}', flush=True)
    return c, truth, e1, e23


def geo_split(art, split, valid_frac, seed):
    """Boolean mask over S1 rows: True = validation (whole states held out).
    The key is country|state because state codes collide across countries."""
    s1 = pl.read_parquet(f'{art}/{split}_s1.parquet', columns=['state', 'country'])
    key = (s1['country'] + '|' + s1['state'].str.split(' ').list.first().fill_null('')).alias('g')
    groups = key.value_counts().sort('g').sample(fraction=1.0, shuffle=True, seed=seed)
    acc = 0; valid = set()
    for g, n in groups.iter_rows():
        if not g.endswith('|') and acc + n <= valid_frac * s1.height:
            valid.add(g); acc += n
    return key.is_in(list(valid)).to_numpy(), sorted(valid)


def tune_postprocessing(cva, truth, e1, e23, va_s1):
    ex = exclusive(cva)
    results = []
    for thr in np.arange(0.3, 0.96, 0.05):
        pred = to_lists(select_threshold(ex, thr), e1, e23)
        results.append(('thr', round(float(thr), 2), breakdown(pred, truth, va_s1)))
    for mp in (0.0, 0.1, 0.3):
        for mn in (0.1, 0.25, 0.4, 0.5):
            pred = to_lists(select_expected_f(ex, miss_prior=mp, min_prob=mn), e1, e23)
            results.append(('ef', (mp, mn), breakdown(pred, truth, va_s1)))
    pred = to_lists(select_threshold(cva, 0.5), e1, e23)
    print('reference: thr 0.5 without exclusivity', round(breakdown(pred, truth, va_s1)['macro_f05'], 4))
    for r in results:
        print(r[0], r[1], {k: round(v, 4) if isinstance(v, float) else v for k, v in r[2].items()})
    return max(results, key=lambda r: r[2]['macro_f05'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--art', default='artifacts')
    ap.add_argument('--data', default='data')
    ap.add_argument('--n-train', type=int, default=250_000, help='#train S1 entities to featurise')
    ap.add_argument('--valid-frac', type=float, default=0.06, help='fraction of S1 (by state) held out')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--emb', action='store_true', help='add the optional GPU name-embedding feature')
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)

    is_valid, vgroups = geo_split(a.art, 'train', a.valid_frac, a.seed)
    print(f'valid: {is_valid.sum()} S1 in {len(vgroups)} held-out states {vgroups}', flush=True)
    tr_ids = rng.choice(np.nonzero(~is_valid)[0], size=min(a.n_train, int((~is_valid).sum())), replace=False)
    va_ids = np.nonzero(is_valid)[0]
    e1 = pl.read_parquet(f'{a.art}/train_s1.parquet', columns=['entity_id'])['entity_id']
    c, truth, e1, e23 = labelled_candidates(a.art, a.data, keep_i1=np.concatenate([tr_ids, va_ids]),
                                            truth_ids=e1.gather(va_ids))
    ctr = c.filter(pl.col('i1').is_in(pl.Series(tr_ids.astype(np.int32)).implode()))
    cva = c.filter(pl.col('i1').is_in(pl.Series(va_ids.astype(np.int32)).implode()))
    del c; gc.collect()
    print(f'train pairs {ctr.height} (pos {ctr["y"].sum()}), valid pairs {cva.height}', flush=True)

    emb = None
    if a.emb:
        from .embed import load as load_emb
        emb = load_emb(a.art, 'train')
        assert emb is not None, 'run `python -m ber.embed --split train` first'
    mtr, Xtr, feats = featurise(ctr, a.art, 'train', a.workers, emb=emb)
    ytr = mtr['y'].to_numpy()
    print(f'{len(feats)} features: {feats}', flush=True)
    del ctr, mtr; gc.collect()
    cva, Xva, _ = featurise(cva, a.art, 'train', a.workers, emb=emb)

    params = dict(objective='binary', learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, num_threads=a.workers, verbose=-1, seed=a.seed)
    dtr = lgb.Dataset(Xtr, ytr, feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(Xva, cva['y'].to_numpy(), reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva], valid_names=['valid'],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    os.makedirs(f'{a.art}/model', exist_ok=True)
    model.save_model(f'{a.art}/model/lgb.txt')
    imp = sorted(zip(feats, model.feature_importance('gain')), key=lambda x: -x[1])
    print('feature importance (gain):', [(f, round(g)) for f, g in imp], flush=True)

    cva = cva.with_columns(pl.Series('prob', model.predict(Xva, num_threads=a.workers).astype(np.float32)))
    cva.write_parquet(f'{a.art}/valid_scored.parquet')
    best = tune_postprocessing(cva, truth, e1, e23, e1.gather(va_ids).to_list())
    print('BEST', best, flush=True)
    cfg = {'features': feats, 'best_iteration': model.best_iteration, 'params': params,
           'post': {'method': best[0], 'param': best[1]}, 'valid': best[2], 'valid_states': vgroups,
           'n_train_s1': int(len(tr_ids))}
    json.dump(cfg, open(f'{a.art}/model/config.json', 'w'), indent=1, default=str)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
