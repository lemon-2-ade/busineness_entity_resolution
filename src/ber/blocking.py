"""Stage 2: candidate generation (blocking).

Three complementary sparse "views" of every record are retrieved with an
IDF-weighted cosine top-k search, restricted to records with the same
country label (true pairs are 100% country-consistent in training; the label
is used as an opaque string, so France is handled like any other value):

  * NAME view    - name word tokens + char 3-grams of the compact core name
                   (typo / word-split / domain-name robust)
  * ADDRESS view - address core tokens + adjacent token bigrams
                   (catches pure-alias names such as "Lumquo" at the S1 address)
  * JOINT view   - both views concatenated (cos_joint = mean of the two); by
                   far the best single ranking because names are highly
                   non-unique (~50% of S1 core names repeat) while addresses
                   are nearly unique (3-4% repeat).

Geo-partitioning: retrieval is done inside (country, state) partitions
(95% of true pairs share the canonical state; 4.7% of S2/S3 records have no
state - those form a per-country "residual" pool that every S1 of the country
is also searched against).  S1 records without a single canonical state
(e.g. France, whose regions are not in the state dictionary) are searched
against the whole country.  This cuts the sparse-product work by ~20x and
makes IDF local (a street name that is rare *in the state* is informative).

The per-view top-k lists are unioned and every pair keeps its name/addr/joint
cosines and per-S1 ranks as matcher features.  The union is truncated by rank
(see `generate_candidates`); what survives is exactly the set the matcher
scores (candidate_pairs.tsv).

Measured on a 50k-S1 train sample: without geo-partitioning (global IDF,
country-wide search) union recall was 96.1% at ~42/S1 and ~5 ms/S1;
with geo-partitioning 99.0% at ~54/S1, truncated to 98.5% at ~24/S1.
"""
from __future__ import annotations

import math
import time
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
    """Token -> column index, built on the fly."""

    def __init__(self):
        self.idx = {}

    def matrix(self, token_lists):
        """Binary CSR (as indptr/indices arrays) for an iterable of token lists."""
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


def _binary(ip, ix, ncols):
    """Memory-lean binary CSR: bool data + int32 index arrays (no copies)."""
    ip = np.frombuffer(ip, dtype=np.int64)
    if ip[-1] < 2**31 - 1:
        ip = ip.astype(np.int32)
    ix = np.frombuffer(ix, dtype=np.int32).copy()
    m = sp.csr_matrix((np.ones(len(ix), dtype=np.bool_), ix, ip), shape=(len(ip) - 1, ncols), copy=False)
    return m


def tokenize_views(s1: pl.DataFrame, s23: pl.DataFrame):
    """Binary token matrices for both views: {view: (X1, X2)}."""
    out = {}
    for view in ('name', 'addr'):
        if view == 'name':
            fn = lambda df: (name_tokens(a, b, c) for a, b, c in
                             zip(df['name_core'], df['name_alt'], df['name_compact']))
        else:
            fn = lambda df: (addr_tokens(a) for a in df['addr_core'])
        voc = Vocab()
        ip2, ix2 = voc.matrix(fn(s23))
        ip1, ix1 = voc.matrix(fn(s1))
        V = len(voc.idx)
        out[view] = (_binary(ip1, ix1, V), _binary(ip2, ix2, V))
    return out


def _weighted(X, idf):
    """rows of binary CSR X -> idf-weighted, L2-normalised float32 CSR (one copy)."""
    data = idf[X.indices]
    sq = data * data
    rn = np.zeros(X.shape[0], dtype=np.float32)
    nz = np.diff(X.indptr) > 0
    rn[nz] = np.add.reduceat(sq, X.indptr[:-1][nz]) if len(sq) else 0
    rn = np.sqrt(rn); rn[rn == 0] = 1
    data /= np.repeat(rn, np.diff(X.indptr))
    m = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)
    m.eliminate_zeros()
    return m


def weight(X1, X2, cap):
    """IDF (computed on the target block X2) weighting + L2 norm.  Tokens with
    df > cap inside the block are dropped (stop-tokens for blocking)."""
    df_ = np.bincount(X2.indices, minlength=X2.shape[1])
    n = X2.shape[0]
    idf = (np.log((n + 1) / (df_ + 1)) + 1).astype(np.float32)
    idf[(df_ > cap) | (df_ == 0)] = 0
    return _weighted(X1, idf), _weighted(X2, idf)


def _pair_cos(A, B, r, c, chunk=2_000_000):
    """Exact cosine for given (row of A, row of B) pairs."""
    out = np.empty(len(r), dtype=np.float32)
    for i in range(0, len(r), chunk):
        a = A[r[i:i + chunk]]; b = B[c[i:i + chunk]]
        out[i:i + chunk] = np.asarray(a.multiply(b).sum(1)).ravel()
    return out


def _truncate(u: pl.DataFrame, keep) -> pl.DataFrame:
    """Rank candidates of each S1 row (r) by the three cosines and keep the
    union of joint-top keep[0], name-top keep[1], addr-top keep[2]."""
    u = u.with_columns(((pl.col('blk_name') + pl.col('blk_addr')) / 2).alias('blk_joint'))
    u = u.with_columns([pl.col(f'blk_{v}').rank('ordinal', descending=True).over('r').cast(pl.Int16)
                        .alias(f'rk_{v}') for v in ('joint', 'name', 'addr')])
    return u.filter((pl.col('rk_joint') <= keep[0]) | (pl.col('rk_name') <= keep[1]) | (pl.col('rk_addr') <= keep[2]))


def search_block(X, rows1, rows23, ks, caps, threads, keep, chunk=50_000):
    """Top-k retrieval of S1 rows `rows1` against S23 rows `rows23` for the
    three views.  Returns frame (r, c, blk_name, blk_addr) with *global* row
    numbers (within the country frame)."""
    if len(rows1) == 0 or len(rows23) == 0:
        return None
    An, Bn = weight(X['name'][0][rows1], X['name'][1][rows23], caps[0])
    Aa, Ba = weight(X['addr'][0][rows1], X['addr'][1][rows23], caps[1])
    w = np.float32(1 / math.sqrt(2))
    Aj = (sp.hstack([An, Aa], format='csr') * w).tocsr()
    BjT = (sp.hstack([Bn, Ba], format='csr') * w).T.tocsr()
    BnT, BaT = Bn.T.tocsr(), Ba.T.tocsr()
    out = []
    for i in range(0, len(rows1), chunk):
        sl = slice(i, i + chunk)
        lists = []
        for A, BT, k in ((An, BnT, ks[0]), (Aa, BaT, ks[1]), (Aj, BjT, ks[2])):
            C = sp_matmul_topn(A[sl], BT, top_n=k, threshold=0.02, sort=False, n_threads=threads).tocoo()
            lists.append(pl.DataFrame({'r': C.row.astype(np.int64) + i, 'c': C.col.astype(np.int64)}))
        u = pl.concat(lists).unique()
        r, c = u['r'].to_numpy(), u['c'].to_numpy()
        u = pl.DataFrame({'r': rows1[r].astype(np.int32), 'c': rows23[c].astype(np.int32),
                          'blk_name': _pair_cos(An, Bn, r, c), 'blk_addr': _pair_cos(Aa, Ba, r, c)})
        out.append(_truncate(u, keep).select('r', 'c', 'blk_name', 'blk_addr'))
    return pl.concat(out)


BLOCK_COLS = ['entity_id', 'country', 'name_core', 'name_alt', 'name_compact', 'addr_core', 'state']
DEFAULTS = dict(k_name=15, k_addr=15, k_joint=30, cap_name=5000, cap_addr=5000, resid_ks=(5, 5, 10), keep=(20, 5, 10))
# Telangana was carved out of Andhra Pradesh; vendors still mix the two.
STATE_ALIAS = {'state_tg': 'state_ap'}


def generate_candidates(s1: pl.DataFrame, s23: pl.DataFrame, k_name=15, k_addr=15, k_joint=30,
                        cap_name=5000, cap_addr=5000, resid_ks=(5, 5, 10), keep=(20, 5, 10), threads=2,
                        verbose=True):
    """Candidate generation for records of ONE country (callers loop over
    countries so that only one country's records are in memory).

    s1 / s23 must carry BLOCK_COLS plus global row ids `_i1` / `_i23`.
    Returns a polars frame (i1, i23, blk_name, blk_addr, blk_joint,
    rk_joint, rk_name, rk_addr): global row indices, the blocking cosines and
    the per-S1 rank of the candidate under each cosine.

    Per geo block the raw top-k union is truncated to
    joint-rank <= keep[0] OR name-rank <= keep[1] OR addr-rank <= keep[2],
    and the same rule is re-applied to the union over blocks
    (~24 candidates / S1, 98% pair recall on a 50k-S1 train sample)."""
    ks, caps = (k_name, k_addr, k_joint), (cap_name, cap_addr)
    t0 = time.time()
    country = s1['country'][0] if s1.height else ''
    if s1.height == 0 or s23.height == 0:
        return None
    X = tokenize_views(s1, s23)
    ia = s1['_i1'].to_numpy(); ib = s23['_i23'].to_numpy()
    st1 = s1['state'].to_list(); st2 = s23['state'].to_list()
    # partition keys: S1 with several states is searched in each of them;
    # S1 with no known state (e.g. France) is searched against the whole country
    groups1 = {}
    for i, st in enumerate(st1):
        for g in (set(STATE_ALIAS.get(x, x) for x in st.split()) or {''}):
            groups1.setdefault(g, []).append(i)
    g2 = np.array([STATE_ALIAS.get(x, x) if x and ' ' not in x else '' for x in st2])
    resid = np.nonzero(g2 == '')[0]
    order2 = np.argsort(g2, kind='stable'); sg2 = g2[order2]
    parts = []
    multi = np.array([len(set(x.split())) > 1 for x in st1])
    for g, rows in groups1.items():
        rows1 = np.asarray(rows, dtype=np.int64)
        tb = time.time()
        if g == '':
            if len(rows1) > 0.5 * len(st1):
                # country without canonical states (e.g. France): search everything
                gp = [search_block(X, rows1, np.arange(len(g2), dtype=np.int64), ks, caps, threads, keep)]
            else:
                # a handful of stateless S1 in a stateful country: residual pool only
                gp = [search_block(X, rows1, resid, ks, caps, threads, keep)]
        else:
            lo, hi = np.searchsorted(sg2, g, 'left'), np.searchsorted(sg2, g, 'right')
            gp = [search_block(X, rows1, np.sort(order2[lo:hi]), ks, caps, threads, keep),
                  search_block(X, rows1, resid, resid_ks, caps, threads, keep)]
        gp = [p for p in gp if p is not None]
        if gp:   # union of the state block and the residual pool, truncated per S1 now (bounded memory)
            parts.append(_truncate(pl.concat(gp), keep).select('r', 'c', 'blk_name', 'blk_addr'))
        if verbose and len(rows1) > 20000:
            print(f'    block {g}: {len(rows1)} S1 {time.time() - tb:.0f}s', flush=True)
    del X
    u = pl.concat(parts); del parts
    # S1 rows searched in several state blocks: merge their lists
    mrows = np.nonzero(multi)[0]
    if len(mrows):
        is_m = pl.col('r').is_in(pl.Series(mrows.astype(np.int32)).implode())
        u = pl.concat([u.filter(~is_m), u.filter(is_m).unique(['r', 'c'])])
    u = _truncate(u, keep)
    u = u.select(pl.Series('i1', ia[u['r'].to_numpy()].astype(np.int32)),
                 pl.Series('i23', ib[u['c'].to_numpy()].astype(np.int32)),
                 'blk_name', 'blk_addr', 'blk_joint', 'rk_joint', 'rk_name', 'rk_addr')
    if verbose:
        print(f'  blocking {country}: {len(ia)} x {len(ib)} -> {u.height} pairs '
              f'({u.height / len(ia):.1f}/S1), {len(groups1)} geo blocks, {time.time()-t0:.0f}s', flush=True)
    return u
