"""Data models for decision policy evaluation."""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass(frozen=True)
class CostMatrix:
    """Defines the financial loss for each possible outcome.
    Higher values represent worse financial outcomes.
    """

    loss_tp: float  # Cost of blocking an actual attack (e.g., gas fees)
    loss_fp: float  # Cost of blocking a normal user (e.g., lost reputation/fees)
    loss_fn: float  # Cost of allowing an attack (e.g., stolen pool funds)
    loss_tn: float = 0.0  # Cost of allowing a normal user (usually 0.0)


@dataclass
class DecisionResult:
    """The auditable output returned to the application."""

    action: str
    confidence: float
    cost_basis: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Converts the result to a dictionary for JSON logging or APIs."""
        return asdict(self)
