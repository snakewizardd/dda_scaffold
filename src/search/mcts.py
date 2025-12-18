"""
Monte Carlo Tree Search with DDA-X enhancements.
"""

import numpy as np
from typing import Optional, List, Any, Dict, Callable
from dataclasses import dataclass

from ..core.state import DDAState, ActionDirection
from ..core.decision import DDADecisionMaker
from .tree import DDASearchTree, DDANode


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    max_iterations: int = 50           # Search budget
    max_depth: int = 20                # Maximum tree depth
    c_explore: float = 1.0             # Exploration constant
    use_dda_selection: bool = True     # Use DDA-X selection vs standard UCT
    expansion_threshold: int = 1       # Visits before expansion


class DDAMCTS:
    """DDA-enhanced Monte Carlo Tree Search."""

    def __init__(
        self,
        config: MCTSConfig,
        decision_maker: DDADecisionMaker,
        value_function: Callable[[Any], float],
        simulator: Optional[Callable] = None
    ):
        self.config = config
        self.decision_maker = decision_maker
        self.value_function = value_function
        self.simulator = simulator

        # Statistics
        self.iterations_run = 0
        self.nodes_expanded = 0

    def search(
        self,
        tree: DDASearchTree,
        delta_x: np.ndarray,
        available_actions: List[ActionDirection],
        context: Dict
    ) -> ActionDirection:
        """
        Run MCTS to find best action.

        Args:
            tree: Search tree with root node
            delta_x: Force-based state update direction
            available_actions: Actions available at root
            context: Additional context for decision making

        Returns:
            Best action to take
        """
        self.iterations_run = 0
        self.nodes_expanded = 0

        # CRITICAL FIX: Initialize root visit count so expansion threshold triggers
        root_hash = tree.get_node_hash(tree.root)
        if tree.N[root_hash] == 0:
            tree.N[root_hash] = 1
        
        # CRITICAL FIX: Expand root immediately with available actions
        if tree.root.is_leaf() and available_actions:
            self._expand(tree, tree.root, available_actions)

        for iteration in range(self.config.max_iterations):
            # Phase 1: Selection
            leaf = self._select(tree, tree.root, delta_x)

            # Phase 2: Expansion
            if self._should_expand(leaf, tree):
                self._expand(tree, leaf, available_actions)
                # Select a child to evaluate
                if leaf.children:
                    leaf = self._select_child_for_evaluation(tree, leaf, delta_x)

            # Phase 3: Simulation (rollout)
            value = self._simulate(leaf, context)

            # Phase 4: Backpropagation
            tree.backpropagate(leaf, value)

            self.iterations_run += 1

        # Return best action from root
        best_action = tree.get_best_action(tree.root, criterion="visits")
        
        # CRITICAL FIX: Fallback if tree search produced no result
        if best_action is None and available_actions:
            # Use DDA-X scoring directly on available actions
            best_action = self.decision_maker.select_action(
                delta_x,
                available_actions,
                tree.root.dda_state,
                tree.N[root_hash]
            )
        
        return best_action

    def _select(
        self,
        tree: DDASearchTree,
        node: DDANode,
        delta_x: np.ndarray
    ) -> DDANode:
        """
        Traverse tree from node to leaf using DDA-X selection.
        """
        current = node

        while not current.is_leaf() and not current.is_terminal:
            if self.config.use_dda_selection:
                # Use DDA-X selection
                current = self._select_child_dda(tree, current, delta_x)
            else:
                # Use standard UCT
                current = self._select_child_uct(tree, current)

            if current.depth >= self.config.max_depth:
                break

        return current

    def _select_child_dda(
        self,
        tree: DDASearchTree,
        node: DDANode,
        delta_x: np.ndarray
    ) -> DDANode:
        """Select child using DDA-X scoring."""
        node_hash = tree.get_node_hash(node)
        total_visits = tree.N[node_hash]

        # Get actions from children
        actions = list(node.children.keys())

        # Use decision maker to select
        best_action = self.decision_maker.select_action(
            delta_x,
            actions,
            node.dda_state,
            total_visits
        )

        return node.children[best_action]

    def _select_child_uct(
        self,
        tree: DDASearchTree,
        node: DDANode
    ) -> DDANode:
        """Select child using standard UCT."""
        best_action = tree.get_best_action(node, criterion="ucb")
        return node.children[best_action]

    def _should_expand(self, node: DDANode, tree: DDASearchTree) -> bool:
        """Check if node should be expanded."""
        if node.is_terminal or node.is_leaf() == False:
            return False

        node_hash = tree.get_node_hash(node)
        visits = tree.N[node_hash]

        return visits >= self.config.expansion_threshold

    def _expand(
        self,
        tree: DDASearchTree,
        node: DDANode,
        actions: List[ActionDirection]
    ) -> None:
        """Expand node with available actions."""
        tree.expand(node, actions)
        self.nodes_expanded += 1

    def _select_child_for_evaluation(
        self,
        tree: DDASearchTree,
        parent: DDANode,
        delta_x: np.ndarray
    ) -> DDANode:
        """Select which newly expanded child to evaluate first."""
        if not parent.children:
            return parent

        # Select using DDA-X scoring with exploration bonus
        if self.config.use_dda_selection:
            node_hash = tree.get_node_hash(parent)
            actions = list(parent.children.keys())

            best_action = self.decision_maker.select_action(
                delta_x,
                actions,
                parent.dda_state,
                tree.N[node_hash]
            )
            return parent.children[best_action]
        else:
            # Random selection for first visit
            import random
            return random.choice(list(parent.children.values()))

    def _simulate(self, node: DDANode, context: Dict) -> float:
        """
        Simulate from node to estimate value.

        This is where we'd run a rollout or call the value function.
        """
        if self.simulator:
            # Run actual simulation
            return self.simulator(node, context)
        else:
            # Use value function directly
            return self.value_function(node.observation)

    def get_statistics(self) -> Dict:
        """Get search statistics."""
        return {
            "iterations": self.iterations_run,
            "nodes_expanded": self.nodes_expanded,
            "avg_expansions_per_iteration": (
                self.nodes_expanded / max(1, self.iterations_run)
            )
        }