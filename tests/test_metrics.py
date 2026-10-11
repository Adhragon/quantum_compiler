import pytest
from qubit_allocation.metrics import regression_metrics, percentage_reduction, ranking_metrics


def test_regression_metrics():
    m = regression_metrics([1, 2, 3], [1, 2, 4])
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["mse"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx((1 / 3) ** 0.5)


def test_zero_baseline_percent_is_undefined():
    assert percentage_reduction(0, 1) is None
    assert percentage_reduction(10, 8) == pytest.approx(20)


def test_ranking_metrics():
    m = ranking_metrics([1, 2, 3, 4], [1, 2, 4, 3])
    assert m["top_5_percent_overlap"] == pytest.approx(1.0)
    assert m["top_k"] == 1
