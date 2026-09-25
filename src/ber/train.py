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

from .features import REC_COLS, context_features, pair_features, prob_context, raw_flags
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


def name_freq_table(art, split):
    """How many records (S1+S2+S3 of the split, same country) share each core
    name.  Unsupervised statistic of the split itself - no labels used."""
    path = f'{art}/{split}_namefreq.parquet'
    if not os.path.exists(path):
        lf = pl.concat([pl.scan_parquet(f'{art}/{split}_s{i}.parquet').select('country', 'name_core')
                        for i in (1, 2, 3)])
        lf.group_by('country', 'name_core').agg(pl.len().cast(pl.Float32).alias('name_freq')) \
            .collect().write_parquet(path)
    return pl.read_parquet(path)


class RecordStore:
    """Feature columns of every record of a source, stored as uncompressed
    Arrow IPC and memory-mapped: gathering the rows referenced by a chunk of
    candidates touches only those pages, so RAM stays flat even for the
    10M-record S2+S3 tables (built once from the parquet cache)."""

    def __init__(self, art, split, sources):
        self.frames, self.offsets = [], [0]
        cols = [x for x in REC_COLS if x not in ('rflags', 'name_freq')] + ['business_name']
        for src in sources:
            path = f'{art}/{split}_s{src}.rec.arrow'
            if not os.path.exists(path):
                freq = name_freq_table(art, split)
                df = raw_flags(pl.read_parquet(f'{art}/{split}_s{src}.parquet', columns=cols + ['country']))
                df = df.join(freq, on=['country', 'name_core'], how='left').drop('country')
                df.write_ipc(path, compression='uncompressed')
                del df
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


def geo_split(art, split, fracs, seed):
    """Assign whole geo blocks (country|state) to held-out folds.
    Returns (fold array over S1 rows: 0 = training pool, k = k-th held-out
    fold, list of held-out keys per fold).  The key is country|state because
    state codes collide across countries."""
    s1 = pl.read_parquet(f'{art}/{split}_s1.parquet', columns=['state', 'country'])
    key = (s1['country'] + '|' + s1['state'].str.split(' ').list.first().fill_null('')).alias('g')
    groups = key.value_counts().sort('g').sample(fraction=1.0, shuffle=True, seed=seed)
    fold_of = {}; folds = []
    for k, frac in enumerate(fracs, start=1):
        acc = 0; chosen = []
        for g, n in groups.iter_rows():
            if g in fold_of or g.endswith('|'):
                continue
            if acc + n <= frac * s1.height:
                fold_of[g] = k; chosen.append(g); acc += n
        folds.append(sorted(chosen))
    fold = key.replace_strict(fold_of, default=0, return_dtype=pl.Int8).to_numpy()
    return fold, folds


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


LGB_PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, verbose=-1)
STAGE2_PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                     max_bin=255, verbose=-1)


def binned(X, y, names, workers, seed, params=None):
    """Construct the LightGBM training Dataset right away so the float matrix
    can be freed (the binned copy is ~4x smaller)."""
    p = dict(params or LGB_PARAMS, num_threads=workers, seed=seed)
    return lgb.Dataset(X, y, feature_name=names, free_raw_data=True, params=p).construct()


def fit(params, X, y, Xv, yv, names, workers, seed):
    """X may be a raw matrix or an already constructed lgb.Dataset."""
    p = dict(params, num_threads=workers, seed=seed)
    dtr = X if isinstance(X, lgb.Dataset) else lgb.Dataset(X, y, feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    return lgb.train(p, dtr, num_boost_round=4000, valid_sets=[dva], valid_names=['valid'],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])


def stage2_matrix(meta, X, prob):
    """Stage-2 design matrix = stage-1 features + probability-context features."""
    pc = prob_context(meta['i1'].to_numpy(), meta['i23'].to_numpy(), prob)
    return np.hstack([X, np.column_stack(list(pc.values())).astype(np.float32)]), list(pc.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--art', default='artifacts')
    ap.add_argument('--data', default='data')
    ap.add_argument('--n-train', type=int, default=250_000, help='#train S1 entities to featurise (stage 1)')
    ap.add_argument('--valid-frac', type=float, default=0.06, help='fraction of S1 (whole states) for validation')
    ap.add_argument('--stage2-frac', type=float, default=0.08,
                    help='fraction of S1 (whole states) to train the stage-2 re-scorer; 0 disables stage 2')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--emb', action='store_true', help='add the optional GPU name-embedding feature')
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    use2 = a.stage2_frac > 0

    fold, folds = geo_split(a.art, 'train', [a.valid_frac] + ([a.stage2_frac] if use2 else []), a.seed)
    print(f'valid fold: {(fold == 1).sum()} S1 in {folds[0]}', flush=True)
    if use2:
        print(f'stage-2 fold: {(fold == 2).sum()} S1 in {folds[1]}', flush=True)
    tr_ids = rng.choice(np.nonzero(fold == 0)[0], size=min(a.n_train, int((fold == 0).sum())), replace=False)
    va_ids = np.nonzero(fold == 1)[0]
    s2_ids = np.nonzero(fold == 2)[0]
    e1 = pl.read_parquet(f'{a.art}/train_s1.parquet', columns=['entity_id'])['entity_id']
    c, truth, e1, e23 = labelled_candidates(a.art, a.data, keep_i1=np.concatenate([tr_ids, va_ids, s2_ids]),
                                            truth_ids=e1.gather(va_ids))
    sel = lambda ids: c.filter(pl.col('i1').is_in(pl.Series(ids.astype(np.int32)).implode()))
    ctr, cva, cs2 = sel(tr_ids), sel(va_ids), sel(s2_ids)
    del c; gc.collect()
    print(f'pairs: stage-1 train {ctr.height} (pos {ctr["y"].sum()}), valid {cva.height}, stage-2 {cs2.height}',
          flush=True)

    emb = None
    if a.emb:
        from .embed import load as load_emb
        emb = load_emb(a.art, 'train')
        assert emb is not None, 'run `python -m ber.embed --split train` first'
    mtr, Xtr, feats = featurise(ctr, a.art, 'train', a.workers, emb=emb)
    ytr = mtr['y'].to_numpy()
    print(f'{len(feats)} features: {feats}', flush=True)
    dtr = binned(Xtr, ytr, feats, a.workers, a.seed)
    del ctr, mtr, Xtr; gc.collect()
    mva, Xva, _ = featurise(cva, a.art, 'train', a.workers, emb=emb)
    del cva
    # stage 1 early-stops on the stage-2 fold when it exists (keeps the validation fold untouched)
    if use2:
        ms2, Xs2, _ = featurise(cs2, a.art, 'train', a.workers, emb=emb)
        del cs2
        m1 = fit(LGB_PARAMS, dtr, ytr, Xs2, ms2['y'].to_numpy(), feats, a.workers, a.seed)
    else:
        m1 = fit(LGB_PARAMS, dtr, ytr, Xva, mva['y'].to_numpy(), feats, a.workers, a.seed)
    del dtr, ytr; gc.collect()
    os.makedirs(f'{a.art}/model', exist_ok=True)
    m1.save_model(f'{a.art}/model/lgb.txt')
    imp = sorted(zip(feats, m1.feature_importance('gain')), key=lambda x: -x[1])
    print('stage-1 feature importance (gain):', [(f, round(g)) for f, g in imp], flush=True)

    pva = m1.predict(Xva, num_threads=a.workers).astype(np.float32)
    cfg = {'features': feats, 'best_iteration': m1.best_iteration, 'params': LGB_PARAMS,
           'valid_states': folds[0], 'n_train_s1': int(len(tr_ids)), 'stage2': None}
    va_s1 = e1.gather(va_ids).to_list()
    print('--- stage-1 only', flush=True)
    best1 = tune_postprocessing(mva.select('i1', 'i23', 'y').with_columns(pl.Series('prob', pva)),
                                truth, e1, e23, va_s1)
    print('BEST stage-1', best1, flush=True)
    best, pfinal = best1, pva
    if use2:
        ps2 = m1.predict(Xs2, num_threads=a.workers).astype(np.float32)
        X2, names2 = stage2_matrix(ms2, Xs2, ps2); del Xs2
        X2v, _ = stage2_matrix(mva, Xva, pva)
        # stage 2 is fitted on the stage-2 fold; early stopping on 20% of that
        # fold's S1 entities, so the validation fold stays untouched.
        es = (ms2['i1'].to_numpy() % 5) == 0
        y2 = ms2['y'].to_numpy()
        d2 = binned(X2[~es], y2[~es], feats + names2, a.workers, a.seed, STAGE2_PARAMS)
        X2es = X2[es]; del X2; gc.collect()
        m2 = fit(STAGE2_PARAMS, d2, None, X2es, y2[es], feats + names2, a.workers, a.seed)
        m2.save_model(f'{a.art}/model/lgb_stage2.txt')
        imp2 = sorted(zip(feats + names2, m2.feature_importance('gain')), key=lambda x: -x[1])
        print('stage-2 feature importance (gain):', [(f, round(g)) for f, g in imp2[:20]], flush=True)
        p2 = m2.predict(X2v, num_threads=a.workers).astype(np.float32)
        print('--- stage-2', flush=True)
        best2 = tune_postprocessing(mva.select('i1', 'i23', 'y').with_columns(pl.Series('prob', p2)),
                                    truth, e1, e23, va_s1)
        print('BEST stage-2', best2, flush=True)
        if best2[2]['macro_f05'] > best1[2]['macro_f05']:
            best, pfinal = best2, p2
            cfg['stage2'] = {'features': names2, 'best_iteration': m2.best_iteration, 'params': STAGE2_PARAMS}
    mva.select('i1', 'i23', 'y').with_columns(pl.Series('prob', pfinal)).write_parquet(
        f'{a.art}/valid_scored.parquet')
    cfg.update({'post': {'method': best[0], 'param': best[1]}, 'valid': best[2],
                'valid_stage1': best1[2]})
    json.dump(cfg, open(f'{a.art}/model/config.json', 'w'), indent=1, default=str)
    print('FINAL', cfg['post'], cfg['valid'], flush=True)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
