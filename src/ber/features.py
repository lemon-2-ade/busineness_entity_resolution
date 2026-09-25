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
            'addr_norm', 'addr_core', 'addr_nums', 'state']

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


def context_features(c: pl.DataFrame, score='blk_joint', prefix='cx') -> pl.DataFrame:
    """Competition features computed on the *full* candidate graph of a split
    (all S1 entities), so they mean the same thing at train and test time.

    S1 side : #candidates, best score, gap to best, score / best, rank.
    S23 side: #S1 entities that retrieved this record, rank of this S1 among
              them, gap to the best competing S1, and the 2nd-best score.
    """
    s = pl.col(score)
    c = c.with_columns(
        pl.len().over('i1').cast(pl.Float32).alias(f'{prefix}_n1'),
        (s.max().over('i1') - s).alias(f'{prefix}_gap1'),
        (s / (s.max().over('i1') + 1e-6)).alias(f'{prefix}_rel1'),
        pl.len().over('i23').cast(pl.Float32).alias(f'{prefix}_n23'),
        s.rank('ordinal', descending=True).over('i23').cast(pl.Float32).alias(f'{prefix}_rk23'),
        (s.max().over('i23') - s).alias(f'{prefix}_gap23'),
    )
    # margin of this S1 over the best *other* S1 competing for the same record
    second = s.filter(pl.col(f'{prefix}_rk23') > 1).max().over('i23').fill_null(0)
    c = c.with_columns(
        pl.when(pl.col(f'{prefix}_rk23') == 1).then(s - second)
        .otherwise(-pl.col(f'{prefix}_gap23')).alias(f'{prefix}_margin23')
    )
    return c
