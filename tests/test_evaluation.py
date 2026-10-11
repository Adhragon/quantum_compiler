import pytest
from qubit_allocation.metrics import percentage_reduction


def test_swap_reduction_signs():
    assert percentage_reduction(10, 7) == pytest.approx(30.0)
    assert percentage_reduction(7, 10) == pytest.approx(-100.0 * 3 / 7)
    assert percentage_reduction(0, 0) is None
