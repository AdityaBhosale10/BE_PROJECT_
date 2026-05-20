from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, List, Sequence, Tuple


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for rid in top if rid in relevant_ids)
    return hits / float(k)


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0

    def dcg(ids: Sequence[str]) -> float:
        score = 0.0
        for i, rid in enumerate(ids[:k], start=1):
            rel = 1.0 if rid in relevant_ids else 0.0
            score += (2.0**rel - 1.0) / math.log2(i + 1)
        return score

    ideal = list(relevant_ids)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(retrieved_ids) / idcg


@dataclass(frozen=True)
class TimedResult:
    elapsed_ms: float
    retrieved_ids: List[str]


def timed_retrieval(fn) -> TimedResult:
    start = perf_counter()
    retrieved_ids = fn()
    elapsed = (perf_counter() - start) * 1000.0
    return TimedResult(elapsed_ms=elapsed, retrieved_ids=list(retrieved_ids))

