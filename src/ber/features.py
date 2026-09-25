"""Stage 3: pairwise + context features for candidate pairs.

All features are country-agnostic (no country one-hot): string similarities
on the normalised name/address, numeric-token agreement, legal-form
agreement, blocking cosines/ranks and "competition" context features that
describe how a candidate compares with the other candidates of the same S1
entity and with the other S1 entities competing for the same S2/S3 record.
"""
from __future__ import annotations

import re

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

REC_COLS = ['entity_id', 'rflags', 'name_norm', 'name_core', 'name_alt', 'name_compact', 'legal', 'nflags',
            'addr_core', 'addr_nums', 'state']

_VOWELS = re.compile(r'[aeiouyh\s]')
_REPEAT = re.compile(r'(.)\1+')
_INDIC = r'[ऀ-෿]'


def skeleton(s: str) -> str:
    """Consonant skeleton: robust to transliteration vowel noise
    (lakshmi / laxmi / lksmi -> lksm-ish)."""
    return _REPEAT.sub(r'\1', _VOWELS.sub('', s.replace('x', 'ks').replace('w', 'v').replace('ph', 'f')))


def _cp(a, b, scorer, workers, **kw):
    return cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32, **kw)


def _set_feats(a_list, b_list):
    """Token-set statistics between two lists of space-joined strings."""
    n = len(a_list)
    jac = np.zeros(n, np.float32); inter = np.zeros(n, np.float32)
    cov1 = np.zeros(n, np.float32); cov2 = np.zeros(n, np.float32)
    for i, (x, y) in enumerate(zip(a_list, b_list)):
        if not x or not y:
            continue
        sx = set(x.split()); sy = set(y.split())
        k = len(sx & sy)
        inter[i] = k
        jac[i] = k / len(sx | sy)
        cov1[i] = k / len(sx)
        cov2[i] = k / len(sy)
    return jac, inter, cov1, cov2


def _num_feats(n1, n2):
    """Numeric-token agreement (house numbers, plot/flat numbers)."""
    n = len(n1)
    anyc = np.zeros(n, np.float32); first_eq = np.zeros(n, np.float32)
    jac = np.zeros(n, np.float32); sub = np.zeros(n, np.float32); conflict = np.zeros(n, np.float32)
    both = np.zeros(n, np.float32); maxlen_common = np.zeros(n, np.float32)
    for i, (x, y) in enumerate(zip(n1, n2)):
        if not x or not y:
            continue
        lx = x.split(); ly = y.split()
        sx = set(lx); sy = set(ly)
        c = sx & sy
        both[i] = 1
        anyc[i] = 1.0 if c else 0.0
        conflict[i] = 0.0 if c else 1.0
        first_eq[i] = 1.0 if lx[0] == ly[0] else 0.0
        jac[i] = len(c) / len(sx | sy)
        sub[i] = 1.0 if sy <= sx else 0.0
        maxlen_common[i] = max((len(t) for t in c), default=0)
    return anyc, first_eq, jac, sub, conflict, both, maxlen_common


def _legal_feats(l1, l2):
    n = len(l1)
    both = np.zeros(n, np.float32); agree = np.zeros(n, np.float32); conflict = np.zeros(n, np.float32)
    for i, (x, y) in enumerate(zip(l1, l2)):
        if not x or not y:
            continue
        sx = set(x.split()); sy = set(y.split())
        both[i] = 1
        agree[i] = 1.0 if sx & sy else 0.0
        conflict[i] = 0.0 if sx & sy else 1.0
    return both, agree, conflict


def raw_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Replace the raw name by two bit flags (native script, ALL-CAPS) to save memory."""
    n = pl.col('business_name')
    return df.with_columns((n.str.contains(_INDIC).cast(pl.Int8) +
                            (n == n.str.to_uppercase()).cast(pl.Int8) * 2).alias('rflags')).drop('business_name')


def pair_features(p: pl.DataFrame, s1: pl.DataFrame, s23: pl.DataFrame, workers=2) -> pl.DataFrame:
    """p: candidate frame with i1, i23 (+ blocking columns).  s1/s23: normalised
    record frames (REC_COLS).  Returns p with feature columns appended."""
    i1 = p['i1'].to_numpy(); i23 = p['i23'].to_numpy()
    A = s1.select(REC_COLS)[i1]
    B = s23.select(REC_COLS)[i23]
    f = {}
    # ------------------------------------------------------------ names
    n1, n2 = A['name_norm'].to_list(), B['name_norm'].to_list()
    c1, c2 = A['name_core'].to_list(), B['name_core'].to_list()
    alt2 = B['name_alt'].to_list()
    k1, k2 = A['name_compact'].to_list(), B['name_compact'].to_list()
    f['nm_ratio'] = _cp(n1, n2, fuzz.ratio, workers)
    f['nm_tsort'] = _cp(c1, c2, fuzz.token_sort_ratio, workers)
    f['nm_tset'] = _cp(c1, c2, fuzz.token_set_ratio, workers)
    f['nm_partial'] = _cp(c1, c2, fuzz.partial_ratio, workers)
    f['nm_core_ratio'] = _cp(c1, c2, fuzz.ratio, workers)
    f['nm_compact_ratio'] = _cp(k1, k2, fuzz.ratio, workers)
    f['nm_compact_jw'] = _cp(k1, k2, JaroWinkler.normalized_similarity, workers)
    f['nm_compact_partial'] = _cp(k1, k2, fuzz.partial_ratio, workers)
    f['nm_lev'] = _cp(k1, k2, Levenshtein.distance, workers)
    has_alt = np.array([bool(x) for x in alt2], dtype=np.float32)
    alt_ts = _cp(c1, alt2, fuzz.token_set_ratio, workers) * has_alt
    f['nm_alt_tset'] = alt_ts
    f['nm_best_tset'] = np.maximum(f['nm_tset'], alt_ts)
    sk1 = [skeleton(x) for x in k1]; sk2 = [skeleton(x) for x in k2]
    f['nm_skel_ratio'] = _cp(sk1, sk2, fuzz.ratio, workers)
    f['nm_skel_partial'] = _cp(sk1, sk2, fuzz.partial_ratio, workers)
    jac, inter, cov1, cov2 = _set_feats(c1, c2)
    f['nm_jac'] = jac; f['nm_inter'] = inter; f['nm_cov1'] = cov1; f['nm_cov2'] = cov2
    f['nm_len1'] = np.array([len(x) for x in k1], np.float32)
    f['nm_len2'] = np.array([len(x) for x in k2], np.float32)
    f['nm_ntok2'] = np.array([len(x.split()) for x in c2], np.float32)
    f['nm_first_eq'] = np.array([(x.split()[:1] == y.split()[:1]) and bool(x) for x, y in zip(c1, c2)], np.float32)
    f['nm_prefix4'] = np.array([x[:4] == y[:4] and len(x) >= 4 for x, y in zip(k1, k2)], np.float32)
    lb, la, lc = _legal_feats(A['legal'].to_list(), B['legal'].to_list())
    f['lg_both'] = lb; f['lg_agree'] = la; f['lg_conflict'] = lc
    fl = B['nflags'].to_numpy()
    f['nm2_domain'] = (fl & 1).astype(np.float32)
    f['nm2_alias'] = ((fl & 2) > 0).astype(np.float32)
    rf = B['rflags'].to_numpy()
    f['nm2_native'] = (rf & 1).astype(np.float32)
    f['nm2_upper'] = ((rf & 2) > 0).astype(np.float32)
    # ------------------------------------------------------------ addresses
    a1, a2 = A['addr_core'].to_list(), B['addr_core'].to_list()
    f['ad_empty2'] = np.array([not x for x in a2], np.float32)
    f['ad_ratio'] = _cp(a1, a2, fuzz.ratio, workers)
    f['ad_tsort'] = _cp(a1, a2, fuzz.token_sort_ratio, workers)
    f['ad_tset'] = _cp(a1, a2, fuzz.token_set_ratio, workers)
    f['ad_partial'] = _cp(a1, a2, fuzz.partial_token_set_ratio, workers)
    jac, inter, cov1, cov2 = _set_feats(a1, a2)
    f['ad_jac'] = jac; f['ad_inter'] = inter; f['ad_cov1'] = cov1; f['ad_cov2'] = cov2
    f['ad_ntok1'] = np.array([len(x.split()) for x in a1], np.float32)
    f['ad_ntok2'] = np.array([len(x.split()) for x in a2], np.float32)
    # word-only (street/locality) similarity, numbers removed
    w1 = [' '.join(t for t in x.split() if not t.isdigit()) for x in a1]
    w2 = [' '.join(t for t in x.split() if not t.isdigit()) for x in a2]
    f['ad_words_tset'] = _cp(w1, w2, fuzz.token_set_ratio, workers)
    anyc, feq, njac, nsub, ncon, nboth, nlen = _num_feats(A['addr_nums'].to_list(), B['addr_nums'].to_list())
    f['num_any'] = anyc; f['num_first_eq'] = feq; f['num_jac'] = njac; f['num_sub'] = nsub
    f['num_conflict'] = ncon; f['num_both'] = nboth; f['num_maxlen'] = nlen
    st1, st2 = A['state'].to_list(), B['state'].to_list()
    f['st_eq'] = np.array([bool(x) and x == y for x, y in zip(st1, st2)], np.float32)
    f['st_missing2'] = np.array([not y for y in st2], np.float32)
    f['src3'] = B['entity_id'].str.starts_with('S3').cast(pl.Float32).to_numpy()
    # interaction: strong name AND strong address
    f['nm_x_ad'] = f['nm_best_tset'] * f['ad_tset'] / 100.0
    return p.with_columns([pl.Series(k, v.astype(np.float32)) for k, v in f.items()])


def _group_stats(key, score):
    """For each row: group size, rank of score within its key-group (1 = best),
    best score of the group, second-best score (0 if none).  Pure numpy with a
    single lexsort, so it scales to 50M+ rows in bounded memory."""
    order = np.lexsort((-score, key))
    k = key[order]; sc = score[order]
    start = np.r_[0, np.nonzero(k[1:] != k[:-1])[0] + 1]
    size = np.diff(np.r_[start, len(k)])
    gid = np.repeat(np.arange(len(start)), size)
    rank = np.arange(len(k)) - start[gid] + 1
    best = sc[start]
    second = np.where(size > 1, sc[np.minimum(start + 1, len(k) - 1)], 0).astype(np.float32)
    out_size = np.empty(len(k), np.float32); out_rank = np.empty(len(k), np.float32)
    out_best = np.empty(len(k), np.float32); out_second = np.empty(len(k), np.float32)
    out_size[order] = size[gid]; out_rank[order] = rank
    out_best[order] = best[gid]; out_second[order] = second[gid]
    return out_size, out_rank, out_best, out_second


def context_features(i1, i23, score, keep_mask=None, prefix='cx') -> pl.DataFrame:
    """Competition features computed on the *full* candidate graph, so they
    mean the same thing at train and test time.

    i1, i23, score : arrays over all candidate rows that compete for the
                     S2/S3 records of interest (the whole graph, or a subset
                     closed under "same i23")
    keep_mask      : rows to return (all candidates of the S1 entities of
                     interest); default all.

    S1 side : #candidates, gap to the best score, score / best.
    S23 side: #S1 entities that retrieved this record, rank of this S1 among
              them, gap to the best competing S1 and margin over the best
              *other* S1 (positive only for the top-ranked S1).
    """
    score = score.astype(np.float32)
    n23, rk23, best23, second23 = _group_stats(i23, score)
    if keep_mask is not None:
        i1, i23, score = i1[keep_mask], i23[keep_mask], score[keep_mask]
        n23, rk23, best23, second23 = n23[keep_mask], rk23[keep_mask], best23[keep_mask], second23[keep_mask]
    n1, _, best1, _ = _group_stats(i1, score)
    gap23 = best23 - score
    margin23 = np.where(rk23 == 1, score - second23, -gap23)
    return pl.DataFrame({
        'i1': i1, 'i23': i23, 'blk_joint': score,
        f'{prefix}_n1': n1, f'{prefix}_gap1': best1 - score, f'{prefix}_rel1': score / (best1 + 1e-6),
        f'{prefix}_n23': n23, f'{prefix}_rk23': rk23, f'{prefix}_gap23': gap23, f'{prefix}_margin23': margin23,
    })
