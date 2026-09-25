"""Stage 6: score the test candidates and write the two submission files.

    python -m ber.predict --art artifacts --out output
Writes output/candidate_pairs.tsv (exactly the pairs the model scores) and
output/matching_results.tsv (final matches, a subset of the candidates).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from .io import write_id_lists
from .postprocess import exclusive, select_expected_f, select_threshold
from .features import prob_context
from .train import candidates_with_context, entity_ids, featurise


def group_lists(sel: pl.DataFrame, e1, e23):
    g = (sel.with_columns(e23.gather(sel['i23']).alias('m'))
         .sort('m').group_by('i1').agg(pl.col('m')))
    lists = [[] for _ in range(len(e1))]
    for i, m in zip(g['i1'].to_list(), g['m'].to_list()):
        lists[i] = m
    return lists


def score_all(a, cfg, feats, model, model2, emb, e1, e23, t0):
    country = pl.read_parquet(f'{a.art}/{a.split}_s1.parquet', columns=['country'])['country']
    cand_lists = []
    scored = []
    for cn in sorted(country.unique().to_list()):
        # countries are disjoint in both S1 and S2/S3, so per-country context is exact
        keep = np.nonzero((country == cn).to_numpy())[0]
        cache = f'{a.art}/scored_{a.split}_{cn}.parquet'      # per-country checkpoint (resumable)
        if os.path.exists(cache):
            done = pl.read_parquet(cache)
            scored.append(done); cand_lists.append(done.select('i1', 'i23'))
            print(f'[{cn}] loaded {done.height} scored pairs from checkpoint', flush=True)
            continue
        c = candidates_with_context(a.art, a.split, keep)
        print(f'[{cn}] {c.height} candidate pairs for {len(keep)} S1 ({time.time() - t0:.0f}s)', flush=True)
        cand_lists.append(c.select('i1', 'i23'))
        metas, ps = [], []
        spills = []   # stage-1 features (float32, exactly as in training) spilled to disk for stage 2
        for i in range(0, c.height, a.chunk):
            meta, X, names = featurise(c.slice(i, a.chunk), a.art, a.split, a.workers, log=False, emb=emb)
            assert names == feats, 'feature mismatch between training and inference'
            ps.append(model.predict(X, num_threads=a.workers).astype(np.float32))
            metas.append(meta.select('i1', 'i23'))
            if model2 is not None:
                spills.append(f'{a.art}/_stage1_X_{len(spills)}.npy')
                np.save(spills[-1], X)
            del meta, X
            print(f'  stage-1 scored {min(i + a.chunk, c.height)}/{c.height} ({time.time() - t0:.0f}s)', flush=True)
        del c
        meta = pl.concat(metas); p = np.concatenate(ps)
        if model2 is not None:
            # stage 2: probability context over the whole country (a closed candidate set)
            pc = np.column_stack(list(prob_context(meta['i1'].to_numpy(), meta['i23'].to_numpy(), p).values()))
            p2 = np.empty_like(p)
            i = 0
            for path in spills:
                X = np.load(path)
                X2 = np.hstack([X, pc[i:i + len(X)]])
                p2[i:i + len(X)] = model2.predict(X2, num_threads=a.workers)
                i += len(X)
                del X, X2
                os.remove(path)
            p = p2
            del pc
            print(f'  stage-2 scored ({time.time() - t0:.0f}s)', flush=True)
        scored.append(meta.with_columns(pl.Series('prob', p)))
        scored[-1].write_parquet(cache)
        del meta, p
    write_id_lists(f'{a.out}/candidate_pairs.tsv', e1, group_lists(pl.concat(cand_lists), e1, e23),
                   'candidate_entity_ids')
    scored = pl.concat(scored)
    scored.write_parquet(f'{a.art}/scored_{a.split}.parquet')

    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--art', default='artifacts')
    ap.add_argument('--out', default='output')
    ap.add_argument('--split', default='test')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 2)
    ap.add_argument('--chunk', type=int, default=3_000_000)
    ap.add_argument('--post-only', action='store_true',
                    help='re-run only the selection step from artifacts/scored_<split>.parquet')
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(f'{a.art}/model/config.json'))
    feats = cfg['features']
    model = lgb.Booster(model_file=f'{a.art}/model/lgb.txt')
    model2 = lgb.Booster(model_file=f'{a.art}/model/lgb_stage2.txt') if cfg.get('stage2') else None

    emb = None
    if 'emb_name_cos' in feats:
        from .embed import load as load_emb
        emb = load_emb(a.art, a.split)
        assert emb is not None, 'model uses embeddings: run `python -m ber.embed --split test` first'
    e1, e23 = entity_ids(a.art, a.split)
    if a.post_only:
        scored = pl.read_parquet(f'{a.art}/scored_{a.split}.parquet')
    else:
        scored = score_all(a, cfg, feats, model, model2, emb, e1, e23, t0)

    ex = exclusive(scored)
    method, param = cfg['post']['method'], cfg['post']['param']
    if method == 'thr':
        sel = select_threshold(ex, float(param))
    else:
        sel = select_expected_f(ex, miss_prior=float(param[0]), min_prob=float(param[1]))
    write_id_lists(f'{a.out}/matching_results.tsv', e1, group_lists(sel, e1, e23), 'matched_entity_ids')
    n_match = sel['i1'].n_unique()
    print(f'wrote {sel.height} matches; {n_match}/{len(e1)} S1 entities matched '
          f'({1 - n_match / len(e1):.3f} predicted singletons) in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
