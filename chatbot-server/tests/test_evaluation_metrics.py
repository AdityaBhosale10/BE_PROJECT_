from src.services.evaluation import ndcg_at_k, precision_at_k, timed_retrieval


def test_precision_at_k() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, 1) == 1.0
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k([], relevant, 5) == 0.0
    assert precision_at_k(retrieved, relevant, 0) == 0.0


def test_ndcg_at_k_basic() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c"}
    score = ndcg_at_k(retrieved, relevant, 3)
    assert 0.0 <= score <= 1.0
    assert ndcg_at_k(retrieved, set(), 3) == 0.0
    assert ndcg_at_k(retrieved, relevant, 0) == 0.0


def test_timed_retrieval() -> None:
    tr = timed_retrieval(lambda: ["x", "y"])
    assert tr.retrieved_ids == ["x", "y"]
    assert tr.elapsed_ms >= 0.0

