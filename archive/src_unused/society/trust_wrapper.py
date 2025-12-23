"""
Trust Matrix Wrapper for String-based Agent IDs
"""

from typing import Dict, List, Optional
import numpy as np


class TrustMatrix:
    """
    Simplified Trust Matrix for string-based agent IDs.
    Trust = 1 / (1 + cumulative_prediction_error)
    """

    def __init__(self, agent_ids: List[str]):
        """Initialize trust matrix with string IDs."""
        self.agent_ids = agent_ids
        self.id_to_index = {aid: i for i, aid in enumerate(agent_ids)}
        self.n_agents = len(agent_ids)

        # Initialize trust matrix (all start at 0.5)
        self.trust = np.ones((self.n_agents, self.n_agents)) * 0.5
        np.fill_diagonal(self.trust, 1.0)  # Perfect self-trust

        # Track cumulative errors
        self.cumulative_errors = np.zeros((self.n_agents, self.n_agents))

    def update(self, observer_id: str, observed_id: str, prediction_error: float):
        """Update trust based on prediction error."""
        if observer_id == observed_id:
            return

        i = self.id_to_index.get(observer_id)
        j = self.id_to_index.get(observed_id)

        if i is None or j is None:
            return

        # Accumulate error
        self.cumulative_errors[i, j] += prediction_error

        # Update trust: T = 1 / (1 + Σε)
        self.trust[i, j] = 1.0 / (1.0 + self.cumulative_errors[i, j])

    def get_trust(self, observer_id: str, observed_id: str) -> float:
        """Get trust value between two agents."""
        if observer_id == observed_id:
            return 1.0

        i = self.id_to_index.get(observer_id)
        j = self.id_to_index.get(observed_id)

        if i is None or j is None:
            return 0.5  # Default neutral trust

        return float(self.trust[i, j])

    def get_consensus(self) -> float:
        """Get overall consensus level (average mutual trust)."""
        if self.n_agents <= 1:
            return 1.0

        # Get off-diagonal elements (excluding self-trust)
        mask = ~np.eye(self.n_agents, dtype=bool)
        mutual_trust = self.trust[mask]

        return float(np.mean(mutual_trust))