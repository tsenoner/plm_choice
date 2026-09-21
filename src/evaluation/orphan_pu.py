"""Positive-unlabelled-aware statistics for the orphan arm (Prabakaran & Bromberg 2025).

The orphan pair table is POSITIVE / UNLABELLED, not positive / negative: a pair is
labelled ``sibling`` only when TM >= 0.7 AND SNN >= 0.98, a rule the authors themselves
measure at ~88% precision but only ~2.4% RECALL against experimental EC. The 303,330
non-sibling pairs are therefore *unlabelled* and contain an unknown number of true
functional siblings. Plain AUROC treats every one of them as a negative and so flatters
any ranker whose false positives are concentrated on the missed positives.

This module implements the two PU-aware read-outs the source paper reports, verbatim
from its equations (Section 2, Bioinformatics 41(2):btaf035):

* **DeltaS** -- Eq. (9)::

      DeltaS(tau_p) = Recall * [ mean_{sibling pairs} S  -  mean_{unlabelled pairs} S ]

  the gap between the two score distributions, *weighted by Recall* "to penalize methods
  for failing to identify test set siblings". Eq. (10) takes the max over prediction
  thresholds ``tau_p``; an embedding similarity has no ``tau_p`` (it scores 100% of pairs
  at 100% coverage), so ``DeltaS_max == DeltaS`` for every arm here.

* **Recall_maxPPf50** -- Eqs. (11)-(12)::

      PPf(tau_s) = #pairs predicted sibling / total pairs
      Recall_maxPPf50 = max_{tau_s} Recall(tau_s)   subject to   PPf(tau_s) < 0.5

  "the best possible recall for each method, without encouraging trivial ... positive
  overprediction". Because an embedding gives ONE monotone score per pair, the recall is
  maximised at the loosest admissible threshold, i.e. the largest predicted-positive set
  with PPf < 0.5. A random ranker scores exactly 0.5 here, which is the null this
  statistic is built around.

The similarity score is the paper's own Eq. (5) for embeddings,
``S_cosine(P1, P2) = (E1 . E2 + 1) / 2`` on L2-normalised vectors -- i.e. cosine rescaled
to [0, 1]. DeltaS lives in the LINEAR score space, so (the authors say so explicitly) it
is only comparable between methods with similar score distributions.

Balanced protocol: for F1max / Precision / Recall / PR-AUC the paper under-samples the
unlabelled pairs to match the 6,219 siblings and averages over 100 iterations. That is
reproduced here (:func:`balanced_threshold_metrics`) so the ``Recall`` that weights
DeltaS is the paper's Recall, and so our AUROC can be compared against their Table 1
numbers on their own footing.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "cosine_to_similarity",
    "delta_s_components",
    "recall_max_ppf50",
    "balanced_threshold_metrics",
    "random_classifier_null",
]


def cosine_to_similarity(cos: np.ndarray) -> np.ndarray:
    """Eq. (5): map a cosine in [-1, 1] to the paper's similarity score in [0, 1]."""
    return (np.asarray(cos, dtype=np.float64) + 1.0) / 2.0


# ── DeltaS, Eq. (9)/(10) ──────────────────────────────────────────────────────────────
def delta_s_components(score: np.ndarray, sibling: np.ndarray) -> dict:
    """Return the unweighted score gap and its two means (the Eq. 9 bracket).

    ``delta_s_raw = mean(S | sibling) - mean(S | unlabelled)``. Multiply by the paper's
    Recall to obtain DeltaS itself; both are returned by the caller's assembly so the
    weighting is visible rather than baked in.
    """
    score = np.asarray(score, dtype=np.float64)
    sibling = np.asarray(sibling, dtype=bool)
    if not sibling.any() or sibling.all():
        return {
            "mean_S_sibling": float("nan"),
            "mean_S_unlabelled": float("nan"),
            "delta_s_raw": float("nan"),
        }
    m_sib = float(score[sibling].mean())
    m_unl = float(score[~sibling].mean())
    return {
        "mean_S_sibling": m_sib,
        "mean_S_unlabelled": m_unl,
        "delta_s_raw": m_sib - m_unl,
    }


# ── Recall_maxPPf50, Eqs. (11)/(12) ───────────────────────────────────────────────────
def recall_max_ppf50(
    score: np.ndarray, sibling: np.ndarray, *, ppf_cap: float = 0.5
) -> dict:
    """Max Recall over similarity thresholds subject to ``PPf < ppf_cap`` (Eq. 12).

    Thresholds are the distinct score values (a threshold predicts positive every pair
    with ``S >= tau_s``), so ties are handled exactly as a real threshold would handle
    them -- a tied block is admitted whole or not at all. Returns the achieved recall,
    the achieved PPf, the threshold, and the enrichment over the ``ppf_cap`` null
    (a random ranker achieves ``recall == PPf``).
    """
    score = np.asarray(score, dtype=np.float64)
    sibling = np.asarray(sibling, dtype=bool)
    n = score.size
    n_pos = int(sibling.sum())
    if n == 0 or n_pos == 0:
        return {
            "recall_max_ppf50": float("nan"),
            "ppf_achieved": float("nan"),
            "tau_s": float("nan"),
            "n_predicted_positive": 0,
            "enrichment_vs_ppf": float("nan"),
        }
    order = np.argsort(-score, kind="mergesort")  # descending
    s_sorted = score[order]
    tp_cum = np.cumsum(sibling[order].astype(np.int64))
    # Last index of each tied block == an admissible threshold boundary.
    block_end = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])
    k = block_end + 1  # number of predicted positives at that threshold
    ok = (k / n) < ppf_cap
    if not ok.any():
        return {
            "recall_max_ppf50": 0.0,
            "ppf_achieved": 0.0,
            "tau_s": float("nan"),
            "n_predicted_positive": 0,
            "enrichment_vs_ppf": float("nan"),
        }
    j = int(np.flatnonzero(ok)[-1])  # loosest admissible threshold == max recall
    k_star = int(k[j])
    recall = float(tp_cum[block_end[j]] / n_pos)
    ppf = k_star / n
    return {
        "recall_max_ppf50": recall,
        "ppf_achieved": float(ppf),
        "tau_s": float(s_sorted[block_end[j]]),
        "n_predicted_positive": k_star,
        "enrichment_vs_ppf": float(recall / ppf) if ppf > 0 else float("nan"),
    }


# ── the paper's balanced under-sampling protocol ──────────────────────────────────────
def balanced_threshold_metrics(
    score: np.ndarray,
    sibling: np.ndarray,
    *,
    n_iter: int = 100,
    seed: int = 42,
) -> dict:
    """F1max / Precision / Recall / ROC-AUC / PR-AUC under the paper's balancing.

    "To balance the number of sibling (positives) versus unlabeled (mostly non-sibling)
    pairs, we under-sampled the latter to match the number of sibling pairs; we repeated
    the under-sampling 100 times and computed the average and standard deviation of all
    measures." Each iteration keeps all ``n_pos`` siblings and draws ``n_pos`` unlabelled
    pairs WITHOUT replacement; the threshold sweep is exact (every distinct score is a
    candidate ``tau_s``).

    ``recall_at_f1max`` is the ``Recall`` that Eq. (9) uses to weight DeltaS.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    score = np.asarray(score, dtype=np.float64)
    sibling = np.asarray(sibling, dtype=bool)
    pos_idx = np.flatnonzero(sibling)
    neg_idx = np.flatnonzero(~sibling)
    n_pos = pos_idx.size
    if n_pos == 0 or neg_idx.size < n_pos:
        return {k: float("nan") for k in (
            "f1max", "precision_at_f1max", "recall_at_f1max", "tau_s_at_f1max",
            "roc_auc_balanced", "pr_auc_balanced",
            "f1max_sd", "precision_at_f1max_sd", "recall_at_f1max_sd",
            "roc_auc_balanced_sd", "pr_auc_balanced_sd",
        )} | {"n_iter": 0}

    rng = np.random.default_rng(seed)
    f1s, precs, recs, taus, rocs, prs = [], [], [], [], [], []
    s_pos = score[pos_idx]
    for _ in range(n_iter):
        sel = rng.choice(neg_idx, size=n_pos, replace=False)
        s = np.concatenate([s_pos, score[sel]])
        y = np.concatenate([np.ones(n_pos, bool), np.zeros(n_pos, bool)])
        order = np.argsort(-s, kind="mergesort")
        s_sorted = s[order]
        tp_cum = np.cumsum(y[order].astype(np.int64))
        block_end = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])
        k = (block_end + 1).astype(np.float64)
        tp = tp_cum[block_end].astype(np.float64)
        f1 = 2.0 * tp / (k + n_pos)
        j = int(np.argmax(f1))
        f1s.append(float(f1[j]))
        precs.append(float(tp[j] / k[j]))
        recs.append(float(tp[j] / n_pos))
        taus.append(float(s_sorted[block_end[j]]))
        rocs.append(float(roc_auc_score(y, s)))
        prs.append(float(average_precision_score(y, s)))

    def ms(a):
        a = np.asarray(a, dtype=np.float64)
        return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0

    f1m, f1sd = ms(f1s)
    pm, psd = ms(precs)
    rm, rsd = ms(recs)
    rocm, rocsd = ms(rocs)
    prm, prsd = ms(prs)
    return {
        "f1max": f1m, "f1max_sd": f1sd,
        "precision_at_f1max": pm, "precision_at_f1max_sd": psd,
        "recall_at_f1max": rm, "recall_at_f1max_sd": rsd,
        "tau_s_at_f1max": float(np.mean(taus)),
        "roc_auc_balanced": rocm, "roc_auc_balanced_sd": rocsd,
        "pr_auc_balanced": prm, "pr_auc_balanced_sd": prsd,
        "n_iter": int(n_iter),
    }


def random_classifier_null(
    sibling: np.ndarray, *, n_iter: int = 100, seed: int = 0
) -> dict:
    """The paper's 'random classifier': S ~ Uniform(0, 1) per pair, repeated ``n_iter``.

    Gives the empirical null band for AUROC, DeltaS_raw and Recall_maxPPf50 on THIS
    pair table (same n, same sibling count), so a measured value can be read against a
    measured, not an assumed, null.
    """
    from sklearn.metrics import roc_auc_score

    sibling = np.asarray(sibling, dtype=bool)
    rng = np.random.default_rng(seed)
    au, ds, rc = [], [], []
    for _ in range(n_iter):
        s = rng.random(sibling.size)
        au.append(float(roc_auc_score(sibling, s)))
        ds.append(delta_s_components(s, sibling)["delta_s_raw"])
        rc.append(recall_max_ppf50(s, sibling)["recall_max_ppf50"])
    out = {}
    for name, arr in (("auroc", au), ("delta_s_raw", ds), ("recall_max_ppf50", rc)):
        a = np.asarray(arr, dtype=np.float64)
        out[f"{name}_mean"] = float(a.mean())
        out[f"{name}_sd"] = float(a.std(ddof=1))
        out[f"{name}_p2.5"] = float(np.quantile(a, 0.025))
        out[f"{name}_p97.5"] = float(np.quantile(a, 0.975))
    out["n_iter"] = int(n_iter)
    return out
