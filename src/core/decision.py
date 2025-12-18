"""
Decision Making for DDA-X

Implements the DDA-X action selection algorithm combining:
- DDA alignment
- UCT exploration
- Rigidity dampening
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DecisionConfig:
    """Hyperparameters for action selection."""
    c_explore: float = 1.0                 # Exploration constant
    use_rigidity_damping: bool = True      # Apply rigidity to exploration
    min_alignment_threshold: float = -0.5  # Reject actions misaligned with Δx


class DDADecisionMaker:
    """
    Selects actions using DDA-X scoring:

    Score(a) = cos(Δx, d̂(a)) + c × P(a|s) × √N(s)/(1+N(s,a)) × (1 - ρ)
    """

    def __init__(self, config: DecisionConfig):
        self.config = config

    def compute_scores(
        self,
        delta_x: np.ndarray,
        actions: List["ActionDirection"],
        state: "DDAState",
        total_state_visits: int
    ) -> List[float]:
        """Compute DDA-X score for each action."""

        scores = []
        delta_x_norm = np.linalg.norm(delta_x)

        for action in actions:
            # Component 1: DDA alignment (cosine similarity)
            if delta_x_norm < 1e-8:
                alignment = 0.0
            else:
                alignment = np.dot(delta_x, action.d_hat) / delta_x_norm

            # Component 2: Exploration bonus (UCT-style)
            if total_state_visits == 0:
                exploration = self.config.c_explore * action.prior_prob
            else:
                exploration = (
                    self.config.c_explore
                    * action.prior_prob
                    * np.sqrt(total_state_visits)
                    / (1 + action.N)
                )

            # Component 3: Rigidity dampening (DDA signature!)
            if self.config.use_rigidity_damping:
                rigidity_factor = 1 - state.rho
            else:
                rigidity_factor = 1.0

            # Final score
            score = alignment + exploration * rigidity_factor
            scores.append(score)

        return scores

    def select_action(
        self,
        delta_x: np.ndarray,
        actions: List["ActionDirection"],
        state: "DDAState",
        total_state_visits: int
    ) -> "ActionDirection":
        """Select the highest-scoring action."""

        if not actions:
            raise ValueError("No actions available for selection")

        scores = self.compute_scores(delta_x, actions, state, total_state_visits)
        best_idx = np.argmax(scores)
        return actions[best_idx]

    def select_with_threshold(
        self,
        delta_x: np.ndarray,
        actions: List["ActionDirection"],
        state: "DDAState",
        total_state_visits: int
    ) -> Optional["ActionDirection"]:
        """
        Select action only if it meets alignment threshold.
        Returns None if all actions are too misaligned (protect mode).
        """
        if not actions:
            return None

        scores = self.compute_scores(delta_x, actions, state, total_state_visits)

        # Check alignment component for each action
        delta_x_norm = np.linalg.norm(delta_x)
        alignments = []

        for action in actions:
            if delta_x_norm < 1e-8:
                alignment = 0.0
            else:
                alignment = np.dot(delta_x, action.d_hat) / delta_x_norm
            alignments.append(alignment)

        # Check if any action meets threshold
        valid_actions = [
            (a, s) for a, s, align in zip(actions, scores, alignments)
            if align >= self.config.min_alignment_threshold
        ]

        if not valid_actions:
            return None  # Trigger protect mode

        best_action = max(valid_actions, key=lambda x: x[1])[0]
        return best_action


def dda_x_select(
    state: "DDAState",
    actions: List["ActionDirection"],
    delta_x: np.ndarray,
    total_visits: int,
    c: float = 1.0
) -> "ActionDirection":
    """
    Standalone DDA-X action selection function.

    Score(a) = cos(Δx, d̂(a)) + c × P(a|s) × √N(s)/(1+N(s,a)) × (1 - ρ)
    """
    best_score = float('-inf')
    best_action = None

    delta_x_norm = np.linalg.norm(delta_x)
    rigidity_factor = 1 - state.rho  # DDA signature!

    for action in actions:
        # Component 1: DDA alignment
        if delta_x_norm > 1e-8:
            alignment = np.dot(delta_x, action.d_hat) / delta_x_norm
        else:
            alignment = 0.0

        # Component 2: UCT exploration
        if total_visits == 0:
            exploration = c * action.prior_prob
        else:
            exploration = c * action.prior_prob * np.sqrt(total_visits) / (1 + action.N)

        # Component 3: Rigidity dampening
        score = alignment + exploration * rigidity_factor

        if score > best_score:
            best_score = score
            best_action = action

    return best_action