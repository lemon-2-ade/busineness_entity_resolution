"""Stage 5: turn pair probabilities into per-S1 match lists that maximise the
macro F0.5 metric.

1. Exclusivity - in the training data every S2/S3 record belongs to at most
   one S1 entity, so each record is only kept for the S1 entity that gives it
   the highest probability.
2. Per-entity subset choice - for each S1 entity with candidate probabilities
   p1 >= p2 >= ... we choose k (0..n) maximising the expected F0.5:
       E[F | top-k] ~= (1+b^2) * sum_{i<=k} p_i / (b^2 * (sum_i p_i + m) + k)
       E[F | empty] =  prod_i (1 - p_i)          (the singleton case)
   where m is a small prior for true matches outside the candidate set.
   A plain probability threshold is also supported; both are tuned on the
   validation split.
"""
from __future__ import annotations

import numpy as np
import polars as pl

B2 = 0.25


def exclusive(scored: pl.DataFrame, prob='prob') -> pl.DataFrame:
    """Keep, for each S2/S3 record, only its best S1 entity."""
    return scored.filter(pl.col(prob) == pl.col(prob).max().over('i23')).unique('i23', keep='first')


def select_threshold(scored: pl.DataFrame, thr: float, prob='prob') -> pl.DataFrame:
    return scored.filter(pl.col(prob) >= thr).select('i1', 'i23')


def select_expected_f(scored: pl.DataFrame, prob='prob', miss_prior=0.05, min_prob=0.05, gain=1.0) -> pl.DataFrame:
    """Expected-F0.5-optimal top-k per S1 (see module docstring)."""
    d = (scored.filter(pl.col(prob) >= 1e-4)
         .sort(['i1', prob], descending=[False, True])
         .with_columns(
             pl.col(prob).cum_sum().over('i1').alias('_cum'),
             pl.int_range(1, pl.len() + 1).over('i1').alias('_k'),
             pl.col(prob).sum().over('i1').alias('_et'),
             (1 - pl.col(prob)).clip(1e-9, 1).log().sum().over('i1').exp().alias('_pempty'),
         ))
    d = d.with_columns(((1 + B2) * pl.col('_cum') / (B2 * (pl.col('_et') + miss_prior) + pl.col('_k'))).alias('_ef'))
    best = d.group_by('i1').agg(pl.col('_ef').max().alias('_best'), pl.col('_ef').arg_max().alias('_argk'),
                                pl.col('_pempty').first())
    d = d.join(best, on='i1')
    keep = d.filter((pl.col('_best') * gain > pl.col('_pempty')) & (pl.col('_k') <= pl.col('_argk') + 1)
                    & (pl.col(prob) >= min_prob))
    return keep.select('i1', 'i23')


def to_lists(sel: pl.DataFrame, s1_ids: pl.Series, s23_ids: pl.Series) -> dict:
    """{s1_entity_id: set(matched ids)}"""
    out = {}
    if sel.height == 0:
        return out
    a = s1_ids.gather(sel['i1']).to_list(); b = s23_ids.gather(sel['i23']).to_list()
    for x, y in zip(a, b):
        out.setdefault(x, set()).add(y)
    return out
