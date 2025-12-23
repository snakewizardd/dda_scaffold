"""
Search Tree for DDA-X

Manages the tree structure for Monte Carlo Tree Search with DDA enhancements.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict
import numpy as np
from collections import defaultdict


@dataclass
class DDANode:
    """Node in the DDA-X search tree."""

    # Observation at this node
    observation: Any

    # DDA state at this node
    dda_state: "DDAState"

    # Action that led to this node (None for root)
    parent_action: Optional["ActionDirection"] = None

    # Parent node
    parent: Optional["DDANode"] = None

    # Children indexed by action
    children: Dict["ActionDirection", "DDANode"] = field(default_factory=dict)

    # Value estimate V(s)
    value: float = 0.0

    # Visit count N(s)
    visits: int = 0

    # Depth in tree
    depth: int = 0

    # Is this a terminal state?
    is_terminal: bool = False

    # Prediction error at this node
    prediction_error: float = 0.0

    # Additional metadata
    metadata: Dict = field(default_factory=dict)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def get_trajectory(self) -> List["ActionDirection"]:
        """Get sequence of actions from root to this node."""
        actions = []
        node = self
        while node.parent is not None:
            actions.append(node.parent_action)
            node = node.parent
        return list(reversed(actions))

    def get_observation_trajectory(self) -> List[Any]:
        """Get sequence of observations from root to this node."""
        observations = []
        node = self
        while node is not None:
            observations.append(node.observation)
            node = node.parent
        return list(reversed(observations))


class DDASearchTree:
    """Manages the search tree for DDA-X."""

    def __init__(self, root_observation: Any, initial_state: "DDAState"):
        self.root = DDANode(
            observation=root_observation,
            dda_state=initial_state.copy()
        )

        # Global statistics indexed by node hash
        self.Q: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.N: Dict[str, int] = defaultdict(int)
        self.Na: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Cache node hashes
        self._node_hashes = {}

    def get_node_hash(self, node: DDANode) -> str:
        """Create hashable identifier for a node."""
        # Use id() for hashability since DDANode contains mutable fields
        node_id = id(node)
        if node_id not in self._node_hashes:
            trajectory = node.get_trajectory()
            self._node_hashes[node_id] = " -> ".join(a.action_id for a in trajectory) or "ROOT"
        return self._node_hashes[node_id]

    def expand(self, node: DDANode, actions: List["ActionDirection"]) -> None:
        """Add children for all available actions."""
        if node.is_terminal:
            return

        for action in actions:
            if action not in node.children:
                child = DDANode(
                    observation=None,  # Will be filled by simulation
                    dda_state=node.dda_state.copy(),
                    parent_action=action,
                    parent=node,
                    depth=node.depth + 1,
                )
                node.children[action] = child

    def backpropagate(self, leaf: DDANode, value: float) -> None:
        """
        Backpropagate value up the tree.

        Q(s,a) ← [N(s,a) × Q(s,a) + v] / [N(s,a) + 1]
        """
        node = leaf

        # Update leaf node value
        node.value = value
        node.visits += 1

        # Propagate up the tree
        while node.parent is not None:
            parent = node.parent
            action = node.parent_action

            parent_hash = self.get_node_hash(parent)
            action_id = action.action_id

            # Incremental mean update
            old_q = self.Q[parent_hash][action_id]
            old_n = self.Na[parent_hash][action_id]

            new_q = (old_n * old_q + value) / (old_n + 1)

            self.Q[parent_hash][action_id] = new_q
            self.Na[parent_hash][action_id] = old_n + 1
            self.N[parent_hash] += 1

            # Update action object
            action.Q = new_q
            action.N = old_n + 1

            # Update parent value (average of children values)
            parent.visits += 1
            parent.value = self._compute_node_value(parent)

            node = parent

    def get_best_action(self, node: DDANode, criterion: str = "visits") -> Optional["ActionDirection"]:
        """
        Get best action from node.

        Args:
            criterion: "visits" (robust), "value" (greedy), or "ucb" (exploratory)
        """
        if not node.children:
            return None

        node_hash = self.get_node_hash(node)

        if criterion == "visits":
            # Most visited (robust child)
            best_action = max(
                node.children.keys(),
                key=lambda a: self.Na[node_hash][a.action_id]
            )
        elif criterion == "value":
            # Highest Q value
            best_action = max(
                node.children.keys(),
                key=lambda a: self.Q[node_hash][a.action_id]
            )
        else:  # ucb
            # Highest UCB score
            best_action = max(
                node.children.keys(),
                key=lambda a: self._compute_ucb(node_hash, a)
            )

        return best_action

    def get_statistics(self, node: DDANode) -> Dict:
        """Get statistics for a node."""
        node_hash = self.get_node_hash(node)

        stats = {
            "visits": node.visits,
            "value": node.value,
            "depth": node.depth,
            "is_terminal": node.is_terminal,
            "prediction_error": node.prediction_error,
            "rigidity": node.dda_state.rho,
            "num_children": len(node.children),
            "total_state_visits": self.N[node_hash]
        }

        # Add action statistics
        if node.children:
            action_stats = {}
            for action, child in node.children.items():
                action_stats[action.action_id] = {
                    "Q": self.Q[node_hash][action.action_id],
                    "N": self.Na[node_hash][action.action_id],
                    "child_value": child.value
                }
            stats["actions"] = action_stats

        return stats

    def _compute_node_value(self, node: DDANode) -> float:
        """Compute value of a node based on children."""
        if not node.children:
            return node.value

        node_hash = self.get_node_hash(node)
        total_visits = sum(self.Na[node_hash].values())

        if total_visits == 0:
            return node.value

        # Weighted average of child values
        weighted_sum = sum(
            self.Q[node_hash][a.action_id] * self.Na[node_hash][a.action_id]
            for a in node.children.keys()
        )

        return weighted_sum / total_visits

    def _compute_ucb(self, node_hash: str, action: "ActionDirection", c: float = 1.0) -> float:
        """Compute UCB score for an action."""
        q = self.Q[node_hash][action.action_id]
        n = self.Na[node_hash][action.action_id]
        n_total = self.N[node_hash]

        if n == 0:
            return float('inf')

        return q + c * np.sqrt(np.log(n_total) / n)