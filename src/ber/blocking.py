"""Stage 2: candidate generation (blocking).

Three complementary sparse "views" of every record are retrieved with an
IDF-weighted cosine top-k search, restricted to records with the same
country label (true pairs are 100% country-consistent in training; the label
is used as an opaque string, so France is handled like any other value):

  * NAME view    - name word tokens + char 3-grams of the compact core name
                   (typo / word-split / domain-name robust)
  * ADDRESS view - address core tokens + adjacent token bigrams
                   (catches pure-alias names such as "Lumquo" at the S1 address)

The per-view top-k lists are unioned; each candidate keeps both blocking
cosines (features for the matcher).  The union is ranked by the mean of the
two cosines ("joint") and truncated to at most `max_cands` per S1 entity.
"""
from __future__ import annotations

import math
from array import array
from collections import defaultdict

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn


# ------------------------------------------------------------------ tokenisers
def name_tokens(core: str, alt: str, compact: str, ngram=3):
    toks = ['w' + t for t in (core + ' ' + alt).split() if len(t) > 1]
    c = compact
    if c:
        c = '^' + c + '$'
        toks += ['c' + c[i:i + ngram] for i in range(max(1, len(c) - ngram + 1))]
    ac = alt.replace(' ', '')
    if ac:
        ac = '^' + ac + '$'
        toks += ['c' + ac[i:i + ngram] for i in range(max(1, len(ac) - ngram + 1))]
    return toks


def addr_tokens(core: str):
    t = core.split()
    toks = ['a' + x for x in t]
    toks += ['b' + t[i] + '_' + t[i + 1] for i in range(len(t) - 1)]
    return toks


class Vocab:
    """Token -> column index with document frequencies, built on the fly."""

    def __init__(self):
        self.idx = {}

    def matrix(self, token_lists):
        indptr = array('q', [0]); indices = array('i')
        idx = self.idx
        for toks in token_lists:
            seen = set()
            for t in toks:
                j = idx.get(t)
                if j is None:
                    j = idx[t] = len(idx)
                if j not in seen:
                    seen.add(j); indices.append(j)
            indptr.append(len(indices))
        return indptr, indices


def _tfidf(indptr, indices, ncols, idf):
    indptr = np.frombuffer(indptr, dtype=np.int64); indices = np.frombuffer(indices, dtype=np.int32)
    data = idf[indices].astype(np.float32)
    m = sp.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, ncols))
    norms = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel()); norms[norms == 0] = 1
    return sp.diags((1 / norms).astype(np.float32)) @ m


def build_view(s1: pl.DataFrame, s23: pl.DataFrame, view: str, max_df_frac=0.02):
    """Returns (A (n1 x V), B (n23 x V)) L2-normalised tf-idf matrices for one view.
    IDF is computed on the target side (S2+S3); very common tokens
    (df > max_df_frac * n23) are dropped from blocking - they carry little
    identity signal and dominate the cost of the sparse product."""
    if view == 'name':
        fn = lambda df: (name_tokens(a, b, c) for a, b, c in zip(df['name_core'], df['name_alt'], df['name_compact']))
    else:
        fn = lambda df: (addr_tokens(a) for a in df['addr_core'])
    voc = Vocab()
    ip2, ix2 = voc.matrix(fn(s23))
    ip1, ix1 = voc.matrix(fn(s1))
    V = len(voc.idx); n23 = len(ip2) - 1
    del voc
    df_ = np.bincount(np.frombuffer(ix2, dtype=np.int32), minlength=V).astype(np.float64)
    idf = np.log((n23 + 1) / (df_ + 1)) + 1
    idf[df_ > max_df_frac * n23] = 0.0          # prune stop-tokens
    idf[df_ == 0] = 0.0                         # tokens absent from target side
    A = _tfidf(ip1, ix1, V, idf); del ip1, ix1
    B = _tfidf(ip2, ix2, V, idf); del ip2, ix2
    A.eliminate_zeros(); B.eliminate_zeros()
    return A, B


def _topn(A, B, k, threads, chunk=20000):
    """Row-wise top-k of A @ B.T -> (rows, cols, vals)."""
    BT = B.T.tocsr()
    rs, cs, vs = [], [], []
    for i in range(0, A.shape[0], chunk):
        C = sp_matmul_topn(A[i:i + chunk], BT, top_n=k, threshold=0.05, sort=False, n_threads=threads)
        C = C.tocoo()
        rs.append(C.row.astype(np.int64) + i); cs.append(C.col.astype(np.int64)); vs.append(C.data.astype(np.float32))
    if not rs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(rs), np.concatenate(cs), np.concatenate(vs)


BLOCK_COLS = ['entity_id', 'country', 'name_core', 'name_alt', 'name_compact', 'addr_core']


def generate_candidates(s1: pl.DataFrame, s23: pl.DataFrame, k_name=20, k_addr=20, max_cands=30,
                        threads=2, verbose=True):
    """Returns a polars frame (i1, i23, blk_name, blk_addr, blk_joint) with row
    indices into s1 / s23 plus the blocking cosines.  blk_joint is the mean of
    the name and address cosines and is used to rank / truncate the union."""
    out = []
    s1 = s1.select([c for c in BLOCK_COLS if c in s1.columns]).with_row_index('_i1')
    s23 = s23.select([c for c in BLOCK_COLS if c in s23.columns]).with_row_index('_i23')
    for country in sorted(set(s1['country'].unique().to_list())):
        a = s1.filter(pl.col('country') == country)
        b = s23.filter(pl.col('country') == country)
        if a.height == 0 or b.height == 0:
            continue
        mats, lists = {}, []
        for v, k in (('name', k_name), ('addr', k_addr)):
            A, B = build_view(a, b, v)
            r, c, _ = _topn(A, B, k, threads)
            lists.append(pl.DataFrame({'r': r, 'c': c}))
            mats[v] = (A, B)
        u = pl.concat(lists).unique()
        r, c = u['r'].to_numpy(), u['c'].to_numpy()
        for v in ('name', 'addr'):
            u = u.with_columns(pl.Series(v, _pair_cos(*mats[v], r, c)))
        del mats
        u = u.with_columns(((pl.col('name') + pl.col('addr')) / 2).alias('joint'))
        u = (u.sort(['r', 'joint'], descending=[False, True])
             .with_columns(pl.int_range(pl.len()).over('r').alias('rk'))
             .filter(pl.col('rk') < max_cands).drop('rk'))
        ia = a['_i1'].to_numpy(); ib = b['_i23'].to_numpy()
        u = u.select(pl.Series('i1', ia[u['r'].to_numpy()]), pl.Series('i23', ib[u['c'].to_numpy()]),
                     pl.col('name').alias('blk_name'), pl.col('addr').alias('blk_addr'),
                     pl.col('joint').alias('blk_joint'))
        if verbose:
            print(f'  blocking {country}: {a.height} x {b.height} -> {u.height} pairs', flush=True)
        out.append(u)
    return pl.concat(out)


def _pair_cos(A, B, r, c, chunk=2_000_000):
    """Exact cosine for given (row of A, row of B) pairs."""
    out = np.empty(len(r), dtype=np.float32)
    for i in range(0, len(r), chunk):
        a = A[r[i:i + chunk]]; b = B[c[i:i + chunk]]
        out[i:i + chunk] = np.asarray(a.multiply(b).sum(1)).ravel()
    return out
