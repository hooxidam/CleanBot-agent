"""Binary-relevance metrics used by the retrieval evaluator."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


ChunkKey = tuple[str, int]


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    mrr_at_k: float
    ndcg_at_k: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def make_chunk_key(source_name: object, chunk_index: object) -> ChunkKey:
    """Normalize dataset labels and Chroma metadata into one comparable key."""
    return str(source_name), int(chunk_index)


def calculate_metrics(
    retrieved: Sequence[ChunkKey],
    relevant: Iterable[ChunkKey],
    k: int,
) -> RetrievalMetrics:
    """Calculate binary Precision/Recall/HitRate/MRR/NDCG at K.

    A duplicate retrieved key only receives credit at its first rank. Missing
    results count as non-relevant, so Precision@K always divides by K.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("relevant chunks must not be empty")

    gains: list[int] = []
    seen: set[ChunkKey] = set()
    for key in list(retrieved[:k]):
        gain = int(key in relevant_set and key not in seen)
        gains.append(gain)
        seen.add(key)

    hit_count = sum(gains)
    precision = hit_count / k
    recall = hit_count / len(relevant_set)
    hit_rate = float(hit_count > 0)

    reciprocal_rank = 0.0
    for rank, gain in enumerate(gains, start=1):
        if gain:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_count = min(k, len(relevant_set))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    return RetrievalMetrics(
        precision_at_k=precision,
        recall_at_k=recall,
        hit_rate_at_k=hit_rate,
        mrr_at_k=reciprocal_rank,
        ndcg_at_k=ndcg,
    )


def mean_metrics(items: Sequence[RetrievalMetrics]) -> dict[str, float]:
    if not items:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "mrr_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }

    keys = items[0].to_dict()
    return {
        key: sum(item.to_dict()[key] for item in items) / len(items)
        for key in keys
    }

