"""Macro F0.5 exactly as the leaderboard computes it (per S1 entity, then mean;
singletons score 1 for an empty prediction and 0 otherwise)."""
from __future__ import annotations


def f_beta(pred: set, true: set, beta=0.5) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred); r = tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f05(pred: dict, truth: dict, ids) -> float:
    """pred/truth: {s1_id: set(ids)}; ids: the S1 entities to average over."""
    tot = 0.0; n = 0
    for s in ids:
        tot += f_beta(pred.get(s, set()), truth.get(s, set()))
        n += 1
    return tot / max(n, 1)


def breakdown(pred: dict, truth: dict, ids):
    """Score split into singleton / non-singleton entities + P/R summary."""
    sing = [s for s in ids if not truth.get(s)]
    non = [s for s in ids if truth.get(s)]
    tp = sum(len(pred.get(s, set()) & truth.get(s, set())) for s in ids)
    npred = sum(len(pred.get(s, set())) for s in ids)
    ntrue = sum(len(truth.get(s, set())) for s in ids)
    return {
        'macro_f05': macro_f05(pred, truth, ids),
        'singleton_f05': macro_f05(pred, truth, sing), 'n_singleton': len(sing),
        'nonsingleton_f05': macro_f05(pred, truth, non), 'n_nonsingleton': len(non),
        'micro_precision': tp / max(npred, 1), 'micro_recall': tp / max(ntrue, 1),
    }
