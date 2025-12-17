"""
Force Channels for DDA

Implementation of the three force channels:
- Identity Pull
- Truth Channel
- Reflection Channel
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Optional


class ForceChannel(ABC):
    """Abstract base for force channels."""

    @abstractmethod
    def compute(self, state: "DDAState", *args, **kwargs) -> np.ndarray:
        """Compute force vector F ∈ ℝ^d"""
        pass


class IdentityPull(ForceChannel):
    """F_id = γ(x* - x_t) — Pull toward identity attractor."""

    def compute(self, state: "DDAState", observation: Any = None) -> np.ndarray:
        return state.gamma * (state.x_star - state.x)


class TruthChannel(ForceChannel):
    """
    F_T = T(I, IΔ) - x_t

    Maps observations to a target state in decision-space.
    """

    def __init__(self, encoder: Optional["ObservationEncoder"] = None):
        self.encoder = encoder
        self.prev_embedding = None

    def compute(self, state: "DDAState", observation: Any) -> np.ndarray:
        if self.encoder is None:
            # Placeholder implementation
            return np.zeros_like(state.x)

        # Get base observation embedding
        obs_embedding = self.encoder.encode(observation)

        # Compute change sensitivity (IΔ component)
        if self.prev_embedding is not None:
            delta = obs_embedding - self.prev_embedding
            delta_magnitude = np.linalg.norm(delta)
        else:
            delta = np.zeros_like(obs_embedding)
            delta_magnitude = 0.0

        self.prev_embedding = obs_embedding.copy()

        # Target state: x^T = f_parse(I) + λ × f_delta(IΔ)
        lambda_delta = 0.3  # Sensitivity to change
        x_T = obs_embedding + lambda_delta * delta

        # Force toward target
        return x_T - state.x


class ReflectionChannel(ForceChannel):
    """
    F_R = R(D, FM) - x_t

    Maps available actions + assessments to a target state.
    """

    def __init__(self, scorer: Optional["ActionScorer"] = None):
        self.scorer = scorer

    def compute(
        self,
        state: "DDAState",
        actions: List["ActionDirection"],
        context: dict
    ) -> np.ndarray:
        if not actions:
            return np.zeros_like(state.x)

        if self.scorer is None:
            # Simple placeholder: average of action directions
            weighted_direction = sum(a.d_hat for a in actions) / len(actions)
        else:
            # Score each action (objective + subjective)
            scores = self.scorer.score_actions(actions, context)

            # Softmax to get preference distribution
            tau = 2.0  # Temperature
            exp_scores = np.exp(tau * np.array(scores))
            probs = exp_scores / exp_scores.sum()

            # Target = current + weighted sum of action directions
            weighted_direction = sum(
                p * a.d_hat for p, a in zip(probs, actions)
            )

        x_R = state.x + weighted_direction
        return x_R - state.x


class ForceAggregator:
    """Combines all forces into state update."""

    def __init__(
        self,
        identity_pull: IdentityPull,
        truth_channel: TruthChannel,
        reflection_channel: ReflectionChannel
    ):
        self.F_id = identity_pull
        self.F_T = truth_channel
        self.F_R = reflection_channel

    def compute_delta_x(
        self,
        state: "DDAState",
        observation: Any,
        actions: List["ActionDirection"],
        context: dict
    ) -> np.ndarray:
        """
        Δx = k_eff × [F_id + m × (F_T + F_R)]
        """
        f_id = self.F_id.compute(state, observation)
        f_t = self.F_T.compute(state, observation)
        f_r = self.F_R.compute(state, actions, context)

        return state.k_eff * (f_id + state.m * (f_t + f_r))

    def apply_update(
        self,
        state: "DDAState",
        observation: Any,
        actions: List["ActionDirection"],
        context: dict
    ) -> np.ndarray:
        """
        Update state and return new x.

        x_{t+1} = x_t + Δx
        """
        delta_x = self.compute_delta_x(state, observation, actions, context)

        # Store prediction for later error computation
        state.x_pred = state.x + delta_x

        return delta_x