import pytest

from evaluation.metrics import calculate_metrics, mean_metrics


A = ("a.txt", 0)
B = ("a.txt", 1)
C = ("b.txt", 0)
D = ("b.txt", 1)


def test_perfect_ranking():
    metrics = calculate_metrics([A, B, C], [A, B], 3)
    assert metrics.precision_at_k == pytest.approx(2 / 3)
    assert metrics.recall_at_k == 1.0
    assert metrics.hit_rate_at_k == 1.0
    assert metrics.mrr_at_k == 1.0
    assert metrics.ndcg_at_k == 1.0


def test_partial_hit_uses_first_relevant_rank():
    metrics = calculate_metrics([C, A, D], [A, B], 3)
    assert metrics.precision_at_k == pytest.approx(1 / 3)
    assert metrics.recall_at_k == 0.5
    assert metrics.hit_rate_at_k == 1.0
    assert metrics.mrr_at_k == 0.5
    assert 0.0 < metrics.ndcg_at_k < 1.0


def test_no_hit_returns_zero_metrics():
    metrics = calculate_metrics([C, D], [A, B], 3)
    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.hit_rate_at_k == 0.0
    assert metrics.mrr_at_k == 0.0
    assert metrics.ndcg_at_k == 0.0


def test_duplicate_result_only_receives_credit_once():
    metrics = calculate_metrics([A, A, C], [A], 3)
    assert metrics.precision_at_k == pytest.approx(1 / 3)
    assert metrics.recall_at_k == 1.0


def test_missing_results_still_divide_precision_by_k():
    metrics = calculate_metrics([A], [A], 3)
    assert metrics.precision_at_k == pytest.approx(1 / 3)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        calculate_metrics([A], [A], 0)
    with pytest.raises(ValueError):
        calculate_metrics([A], [], 1)


def test_mean_metrics():
    first = calculate_metrics([A], [A], 1)
    second = calculate_metrics([B], [A], 1)
    result = mean_metrics([first, second])
    assert all(value == pytest.approx(0.5) for value in result.values())

