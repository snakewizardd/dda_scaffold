"""
DDA State Management

Core state representation for the Dynamic Decision Algorithm.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional


@dataclass
class DDAState:
    """The agent's internal state in decision-space."""

    # Core state vector
    x: np.ndarray                          # Current position in ℝ^d

    # Identity
    x_star: np.ndarray                     # Identity attractor
    gamma: float = 1.0                     # Identity stiffness

    # Rigidity dynamics
    rho: float = 0.0                       # Rigidity ∈ [0, 1]
    epsilon_0: float = 0.3                 # Surprise threshold
    alpha: float = 0.1                     # Rigidity learning rate
    s: float = 0.1                         # Sigmoid sensitivity

    # Effective parameters
    k_base: float = 0.5                    # Base step size
    m: float = 1.0                         # External pressure/gain

    # History for prediction error
    x_pred: Optional[np.ndarray] = None

    @property
    def k_eff(self) -> float:
        """Effective openness = base × (1 - rigidity)"""
        return self.k_base * (1 - self.rho)

    @property
    def d(self) -> int:
        """Dimensionality of state space."""
        return len(self.x)

    def update_rigidity(self, prediction_error: float) -> None:
        """
        Update rigidity based on prediction error (surprise).

        ρ_{t+1} = clip(ρ_t + α[σ((ε - ε₀)/s) - 0.5], 0, 1)
        """
        z = (prediction_error - self.epsilon_0) / self.s
        sigmoid = 1 / (1 + np.exp(-z))
        delta = self.alpha * (sigmoid - 0.5)
        self.rho = np.clip(self.rho + delta, 0.0, 1.0)

    def compute_prediction_error(self, x_actual: np.ndarray) -> float:
        """ε = ||x_pred - x_actual||₂"""
        if self.x_pred is None:
            return 0.0
        return np.linalg.norm(self.x_pred - x_actual)

    def copy(self) -> "DDAState":
        """Create a deep copy of the state."""
        return DDAState(
            x=self.x.copy(),
            x_star=self.x_star.copy(),
            gamma=self.gamma,
            rho=self.rho,
            epsilon_0=self.epsilon_0,
            alpha=self.alpha,
            s=self.s,
            k_base=self.k_base,
            m=self.m,
            x_pred=self.x_pred.copy() if self.x_pred is not None else None
        )

    @classmethod
    def from_identity_config(cls, config: dict, dim: int = 64) -> "DDAState":
        """Initialize state from identity configuration."""
        identity_vector = config.get("identity_vector")
        if identity_vector is None:
            x_star = np.zeros(dim)
        else:
            x_star = np.array(identity_vector)

        return cls(
            x=x_star.copy(),  # Start at identity
            x_star=x_star,
            gamma=config.get("gamma", 1.0),
            epsilon_0=config.get("epsilon_0", 0.3),
            alpha=config.get("alpha", 0.1),
            s=config.get("s", 0.1),
            k_base=config.get("k_base", 0.5),
            m=config.get("m", 1.0),
            rho=config.get("initial_rho", 0.0)
        )


@dataclass
class ActionDirection:
    """An action's representation in decision-space."""

    action_id: str                         # Unique identifier
    raw_action: dict                       # Original action data
    direction: np.ndarray                  # d̂(a) — unit vector in ℝ^d
    prior_prob: float = 0.0                # P(a|s) from LLM sampling

    # MCTS statistics
    Q: float = 0.0                         # Action value estimate
    N: int = 0                             # Visit count

    def __hash__(self):
        """Hash by action_id for use as dict key."""
        return hash(self.action_id)

    def __eq__(self, other):
        """Equality by action_id."""
        if not isinstance(other, ActionDirection):
            return False
        return self.action_id == other.action_id

    @property
    def d_hat(self) -> np.ndarray:
        """Normalized direction vector."""
        norm = np.linalg.norm(self.direction)
        if norm < 1e-8:
            return self.direction
        return self.direction / norm