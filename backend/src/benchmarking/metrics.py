"""
Retrieval Metrics — Precision@K, Recall@K, MRR, Hit Rate, Semantic Score.

All metric functions follow a consistent signature:

    metric(retrieved_ids, relevant_ids, k) -> float

where:
* ``retrieved_ids`` — ordered list of retrieved chunk IDs (index 0 = rank 1)
* ``relevant_ids``  — set/list of ground-truth relevant chunk IDs
* ``k``             — cut-off rank (e.g., 1, 3, 5)

Returned values are all in [0.0, 1.0] unless stated otherwise.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Union


# ── Individual metric functions ───────────────────────────────────────────────

def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
    k: int,
) -> float:
    """
    Precision@K — fraction of top-K retrieved results that are relevant.

    .. math::

        P@K = \\frac{|\\text{retrieved}_{1..K} \\cap \\text{relevant}|}{K}
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    rel_set = set(relevant_ids)
    hits = sum(1 for r in top_k if r in rel_set)
    return hits / k


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
    k: int,
) -> float:
    """
    Recall@K — fraction of all relevant docs found within the top-K results.

    .. math::

        R@K = \\frac{|\\text{retrieved}_{1..K} \\cap \\text{relevant}|}{|\\text{relevant}|}
    """
    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in rel_set)
    return hits / len(rel_set)


def hit_rate_at_k(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
    k: int,
) -> float:
    """
    Hit Rate@K — 1.0 if *any* relevant document appears in top-K, else 0.0.

    Also called ``Recall@K`` when there is exactly one relevant document.
    """
    rel_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    return 1.0 if any(r in rel_set for r in top_k) else 0.0


def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
) -> float:
    """
    Reciprocal Rank (RR) for a single query — the reciprocal of the rank of
    the first relevant document.

    Returns 0.0 if no relevant document is retrieved.

    .. math::

        RR = \\frac{1}{\\text{rank of first relevant}}
    """
    rel_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in rel_set:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
    k: int,
) -> float:
    """
    Average Precision@K (AP@K).

    .. math::

        AP@K = \\frac{\\sum_{i=1}^{K} P@i \\cdot \\text{rel}(i)}{|\\text{relevant}|}

    where :math:`\\text{rel}(i) = 1` if the item at rank *i* is relevant.
    """
    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0

    score = 0.0
    hits = 0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in rel_set:
            hits += 1
            score += hits / i
    return score / min(len(rel_set), k)


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
    k: int,
) -> float:
    """
    Normalised Discounted Cumulative Gain @ K (nDCG@K).

    Assumes binary relevance (1 if relevant, 0 otherwise).

    .. math::

        nDCG@K = \\frac{DCG@K}{IDCG@K}
    """
    rel_set = set(relevant_ids)

    def dcg(ranked: List[str]) -> float:
        return sum(
            1.0 / math.log2(i + 2)  # i+2 because i is 0-based
            for i, doc_id in enumerate(ranked[:k])
            if doc_id in rel_set
        )

    actual_dcg = dcg(retrieved_ids)
    # Ideal DCG: all relevant docs at the top
    ideal_ranking = list(rel_set) + [f"__placeholder_{j}" for j in range(k)]
    ideal_dcg = dcg(ideal_ranking)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def semantic_relevance_score(
    retrieved_scores: List[float],
    k: int,
) -> float:
    """
    Average cosine similarity of the top-K retrieved results.

    This is a proxy metric that doesn't require ground truth labels —
    it measures how semantically close the retrieved chunks are to the query.

    Parameters
    ----------
    retrieved_scores:
        Cosine similarity scores for the retrieved results (ordered).
    k:
        Cut-off rank.
    """
    if not retrieved_scores:
        return 0.0
    return sum(retrieved_scores[:k]) / min(len(retrieved_scores), k)


# ── Aggregate metric computation ──────────────────────────────────────────────

def compute_all_metrics(
    retrieved_ids: List[str],
    retrieved_scores: List[float],
    relevant_ids: Union[List[str], Set[str]],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute the full suite of retrieval metrics at each value of K.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved chunk IDs (rank 1 first).
    retrieved_scores:
        Cosine similarity scores aligned with *retrieved_ids*.
    relevant_ids:
        Ground-truth relevant chunk IDs.
    k_values:
        List of K values to evaluate (default: [1, 3, 5]).

    Returns
    -------
    dict
        Keys like ``"precision@3"``, ``"recall@5"``, ``"mrr"``, etc.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    results: Dict[str, float] = {}

    for k in k_values:
        results[f"precision@{k}"] = round(precision_at_k(retrieved_ids, relevant_ids, k), 4)
        results[f"recall@{k}"] = round(recall_at_k(retrieved_ids, relevant_ids, k), 4)
        results[f"hit_rate@{k}"] = round(hit_rate_at_k(retrieved_ids, relevant_ids, k), 4)
        results[f"ap@{k}"] = round(average_precision_at_k(retrieved_ids, relevant_ids, k), 4)
        results[f"ndcg@{k}"] = round(ndcg_at_k(retrieved_ids, relevant_ids, k), 4)
        results[f"semantic_score@{k}"] = round(semantic_relevance_score(retrieved_scores, k), 4)

    results["mrr"] = round(reciprocal_rank(retrieved_ids, relevant_ids), 4)

    return results


def compare_strategies(
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
) -> Dict[str, dict]:
    """
    Produce a side-by-side comparison showing absolute and relative deltas.

    Returns
    -------
    dict
        Each key maps to ``{"strategy_a": v_a, "strategy_b": v_b, "delta": v_b - v_a}``.
    """
    all_keys = set(metrics_a) | set(metrics_b)
    comparison: Dict[str, dict] = {}
    for key in sorted(all_keys):
        v_a = metrics_a.get(key, 0.0)
        v_b = metrics_b.get(key, 0.0)
        comparison[key] = {
            "strategy_a": v_a,
            "strategy_b": v_b,
            "delta": round(v_b - v_a, 4),
            "relative_improvement_%": (
                round((v_b - v_a) / v_a * 100, 1) if v_a > 0 else None
            ),
        }
    return comparison
