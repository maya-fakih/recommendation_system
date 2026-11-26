"""Shared candidate rules, ranking metrics, and reference recommenders."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .data import Dataset


Scorer = Callable[[int, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RankingMetrics:
    recall_at_10: float
    ndcg_at_10: float
    n_users: int


def top_ten(product_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Sort by decreasing score, breaking ties by increasing product ID."""
    if not np.isfinite(scores).all():
        raise FloatingPointError("Non-finite recommendation score")
    return product_ids[np.lexsort((product_ids, -scores))[:10]]


def evaluate(data: Dataset, task: str, split: str, score: Scorer) -> RankingMetrics:
    """Evaluate every supplied user against the entire eligible catalog."""
    relevant = data.relevant[task, split]
    catalog = data.catalogs["ordinary" if task == "ordinary" else f"cold_{split}"]
    discounts = 1 / np.log2(np.arange(2, 12))
    recall, ndcg = [], []
    for user, targets in relevant.items():
        # Remove training reviews from the eligible catalog before ranking.
        seen = data.reviewed.get(user, set())
        candidates = catalog[~np.isin(catalog, list(seen))]
        ranking = top_ten(candidates, score(user, candidates))
        hits = np.isin(ranking, list(targets))
        recall.append(hits.sum() / len(targets))
        # Normalize discounted hits by the best ranking possible for this user.
        ndcg.append(hits @ discounts[:len(hits)] / discounts[:min(10, len(targets))].sum())
    return RankingMetrics(float(np.mean(recall)), float(np.mean(ndcg)), len(relevant))


def popularity(data: Dataset) -> Scorer:
    """Score ordinary items by their number of positive training reviews."""
    counts = np.zeros(len(data.Z), dtype=float)
    for products in data.positives.values():
        counts[list(products)] += 1
    return lambda user, candidates: counts[candidates]


def description_similarity(data: Dataset) -> Scorer:
    """Score cold items by cosine similarity to each user's mean positive text vector."""
    normalized = {}
    for user, products in data.positives.items():
        profile = data.Z[sorted(products)].mean(axis=0)
        norm = np.linalg.norm(profile)
        if norm > 1e-12:
            normalized[user] = profile / norm
    unit_text = data.Z / np.linalg.norm(data.Z, axis=1, keepdims=True)

    def score(user, candidates):
        if user in normalized:
            return unit_text[candidates] @ normalized[user]
        return np.zeros(len(candidates))

    return score


def random_reference(data: Dataset, split: str, *, seed: int, repetitions: int) -> RankingMetrics:
    """Average uniformly shuffled ordinary rankings with a reported seed and repeat count."""
    if repetitions < 1:
        raise ValueError("Expected at least one repetition")
    rng = np.random.default_rng(seed)
    results = [
        evaluate(data, "ordinary", split, lambda user, ids: rng.permutation(len(ids)))
        for _ in range(repetitions)
    ]
    return RankingMetrics(
        float(np.mean([result.recall_at_10 for result in results])),
        float(np.mean([result.ndcg_at_10 for result in results])),
        results[0].n_users,
    )
