"""Expected-loss decision wrapper for binary classifiers."""

from typing import Union, List
import numpy as np
from decision_policy.models import CostMatrix, DecisionResult


class BinaryExpectedLossPolicy:
    def __init__(
        self,
        cost_matrix: CostMatrix,
        positive_action: str = "TRIGGER_PAUSE",
        negative_action: str = "ALLOW_TRANSACTION",
    ):
        self.costs = cost_matrix
        self.positive_action = positive_action
        self.negative_action = negative_action

    @property
    def optimal_threshold(self) -> float:
        """Computes the exact probability threshold where the decision flips."""
        numerator = self.costs.loss_fp - self.costs.loss_tn
        denominator = (self.costs.loss_fn - self.costs.loss_tp) + (
            self.costs.loss_fp - self.costs.loss_tn
        )

        if denominator == 0:
            return 0.5
        return float(np.clip(numerator / denominator, 0.0, 1.0))

    def evaluate(self, probability: float) -> DecisionResult:
        """Evaluates a single transaction's probability."""
        if not (0.0 <= probability <= 1.0):
            raise ValueError(
                f"Probability must be between 0.0 and 1.0, got {probability}"
            )

        p = float(probability)
        expected_cost_pos = (p * self.costs.loss_tp) + ((1.0 - p) * self.costs.loss_fp)
        expected_cost_neg = (p * self.costs.loss_fn) + ((1.0 - p) * self.costs.loss_tn)

        if expected_cost_pos <= expected_cost_neg:
            selected_action = self.positive_action
        else:
            selected_action = self.negative_action

        return DecisionResult(
            action=selected_action,
            confidence=p,
            cost_basis={
                self.positive_action: round(expected_cost_pos, 4),
                self.negative_action: round(expected_cost_neg, 4),
                "threshold_equivalent": round(self.optimal_threshold, 6),
            },
        )

    def evaluate_batch(
        self, probabilities: Union[List[float], np.ndarray]
    ) -> List[DecisionResult]:
        """Evaluates multiple probabilities at once using numpy for speed."""
        probs = np.asarray(probabilities, dtype=float)
        if np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError("All probabilities must be within [0.0, 1.0]")

        return [self.evaluate(p) for p in probs]
