import pytest
from decision_policy import CostMatrix, BinaryExpectedLossPolicy


@pytest.fixture
def defi_policy():
    """Provides a standard Web3 cost matrix for all tests."""
    matrix = CostMatrix(loss_tp=50, loss_fp=15000, loss_fn=5000000, loss_tn=0)
    return BinaryExpectedLossPolicy(matrix)


def test_optimal_threshold_calculation(defi_policy):
    """Verifies the mathematical tipping point is correctly calculated."""
    threshold = defi_policy.optimal_threshold
    # (15000 - 0) / ((5000000 - 50) + (15000 - 0)) ≈ 0.002991
    assert 0.0029 < threshold < 0.0030


def test_evaluate_above_threshold_blocks(defi_policy):
    """Verifies that a 0.5% risk triggers a block."""
    result = defi_policy.evaluate(0.005)
    assert result.action == "TRIGGER_PAUSE"
    assert result.cost_basis["TRIGGER_PAUSE"] < result.cost_basis["ALLOW_TRANSACTION"]


def test_evaluate_below_threshold_allows(defi_policy):
    """Verifies that a 0.1% risk allows the transaction."""
    result = defi_policy.evaluate(0.001)
    assert result.action == "ALLOW_TRANSACTION"


def test_invalid_probability(defi_policy):
    """Ensures the wrapper crashes safely if fed impossible probabilities."""
    with pytest.raises(ValueError):
        defi_policy.evaluate(1.5)
    with pytest.raises(ValueError):
        defi_policy.evaluate(-0.1)


def test_batch_evaluation(defi_policy):
    """Tests evaluating an array of transactions at once."""
    results = defi_policy.evaluate_batch([0.001, 0.005, 0.99])
    assert len(results) == 3
    assert results[0].action == "ALLOW_TRANSACTION"
    assert results[1].action == "TRIGGER_PAUSE"
    assert results[2].action == "TRIGGER_PAUSE"
