"""
DDA-X Agent Implementation

Main agent class that coordinates the Dynamic Decision Algorithm (DDA-X)
components, integrating Hierarchical Identity, Metacognitive awareness,
and Multi-timescale dynamics.

=============================================================================
ARCHITECTURAL NOTE: Cognitive Architecture
=============================================================================
This agent is constructed as a triad:
1. THE CORE (Constraints): Hierarchical Identity (x*)
2. THE SHIELD (Adaptation): Multi-timescale Rigidity (ρ)
3. THE MONITOR (Metacognition): Self-Correction Layer
=============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, List, Dict, Callable
import asyncio
import time

from .core.state import DDAState, ActionDirection
from .core.forces import ForceAggregator, IdentityPull, TruthChannel, ReflectionChannel
from .core.decision import DDADecisionMaker, DecisionConfig
from .core.dynamics import update_rigidity, check_protect_mode, MultiTimescaleRigidity
from .core.hierarchy import HierarchicalIdentity, create_aligned_identity
from .core.metacognition import MetacognitiveState, create_default_metacognition
from .search.tree import DDASearchTree, DDANode
from .search.mcts import DDAMCTS, MCTSConfig
from .search.simulation import ValueEstimator
from .memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry


@dataclass
class DDAXConfig:
    """Agent configuration for DDA-X Iteration 3."""

    # DDA parameters
    gamma: float = 1.0                     # Identity stiffness
    k_base: float = 0.5                    # Base step size
    m: float = 1.0                         # External pressure
    epsilon_0: float = 0.3                 # Surprise threshold
    alpha: float = 0.1                     # Rigidity learning rate
    s: float = 0.1                         # Sigmoid sensitivity

    # Search parameters
    c_explore: float = 1.0                 # Exploration constant
    max_iterations: int = 50               # Search budget
    branching_factor: int = 5              # Actions to consider per state
    max_depth: int = 10                    # Maximum search depth

    # State space
    state_dim: int = 64                    # Dimension of x ∈ ℝ^d

    # Protect mode
    protect_threshold: float = 0.7         # Enter protect if ρ > this

    # Memory
    max_reflections_retrieved: int = 3     # Reflections to retrieve per decision


class DDAXAgent:
    """
    DDA-X: Dynamic Decision Algorithm with Exploration (Iteration 3).

    A holistic agent combining:
    - Hierarchical Identity: Core values, Personas, and Roles.
    - Metacognition: Self-awareness of cognitive state/rigidity.
    - Multi-timescale Dynamics: Modeling startle, stress, and trauma.
    - Trust-Based Society (via society module).
    """

    def __init__(
        self,
        config: DDAXConfig,
        observation_encoder: Optional[Callable] = None,
        outcome_encoder: Optional[Callable] = None,
        action_generator: Optional[Callable] = None,
        value_estimator: Optional[ValueEstimator] = None,
        ledger: Optional[ExperienceLedger] = None,
        identity_config: Optional[Dict] = None,
        metacognition: Optional[MetacognitiveState] = None,
        hierarchy: Optional[HierarchicalIdentity] = None,
    ):
        self.config = config
        self.obs_encoder = observation_encoder
        self.outcome_encoder = outcome_encoder
        self.action_generator = action_generator
        self.ledger = ledger

        # Initialize Hierarchical Identity
        if hierarchy:
            self.identity = hierarchy
        elif identity_config:
            self.identity = HierarchicalIdentity.from_config(identity_config, dim=config.state_dim)
        else:
            self.identity = create_aligned_identity(dim=config.state_dim)

        # Initialize DDA state
        self.state = DDAState(
            x=self.identity.core.x_star.copy() if self.identity.core else np.zeros(config.state_dim),
            x_star=self.identity.get_effective_attractor(np.zeros(config.state_dim)),
            gamma=config.gamma,
            k_base=config.k_base,
            m=config.m,
            epsilon_0=config.epsilon_0,
            alpha=config.alpha
        )

        # Initialize Metacognition
        self.metacognition = metacognition or create_default_metacognition()

        # Initialize Multi-timescale Rigidity
        self.dynamics = MultiTimescaleRigidity(
            epsilon_0=config.epsilon_0,
            s=config.s,
            alpha_fast=config.alpha,
            alpha_slow=0.01,
            alpha_trauma=0.0001
        )

        # Force channels
        self.forces = ForceAggregator(
            identity_pull=IdentityPull(),
            truth_channel=TruthChannel(observation_encoder),
            reflection_channel=ReflectionChannel(None)  # Scorer created on demand
        )

        # Decision maker
        self.decision_maker = DDADecisionMaker(
            DecisionConfig(c_explore=config.c_explore)
        )

        # Value estimator
        self.value_estimator = value_estimator or ValueEstimator(method="heuristic")

        # MCTS engine
        self.mcts = DDAMCTS(
            config=MCTSConfig(
                max_iterations=config.max_iterations,
                c_explore=config.c_explore,
                use_dda_selection=True
            ),
            decision_maker=self.decision_maker,
            value_function=self._evaluate_state
        )

        # Search tree (initialized per task)
        self.tree: Optional[DDASearchTree] = None

        # Current task context
        self.current_task: Optional[str] = None
        self.task_trajectory: List[LedgerEntry] = []

    async def decide(
        self,
        observation: Any,
        available_actions: Optional[List[Dict]] = None,
        task_intent: str = "unknown"
    ) -> Dict:
        """
        Main decision method using DDA Iteration 3 dynamics.
        """

        # Initialize tree if new task
        if self.tree is None or self.current_task != task_intent:
            self.tree = DDASearchTree(observation, self.state)
            self.current_task = task_intent
            self.task_trajectory = []

        # Generate action directions
        if self.action_generator:
            actions = await self._generate_action_directions(observation, available_actions)
        else:
            actions = self._create_dummy_actions(available_actions or [])

        # ENHANCED: Update x_star based on identity hierarchy
        self.state.x_star = self.identity.get_effective_attractor(self.state.x)

        # Encode observation
        if self.obs_encoder:
            res = self.obs_encoder.encode(observation) if hasattr(self.obs_encoder, 'encode') else self.obs_encoder(observation)
            if asyncio.iscoroutine(res):
                obs_embedding = await res
            else:
                obs_embedding = res
        else:
            obs_embedding = np.random.randn(self.config.state_dim)

        # Retrieve relevant reflections
        reflections = []
        if self.ledger:
            reflections = self.ledger.retrieve_reflections(
                obs_embedding,
                k=self.config.max_reflections_retrieved
            )

        # Build context
        context = {
            "intent": task_intent,
            "reflections": [r.reflection_text for r in reflections],
            "rigidity": self.state.rho,
            "observation": observation,
            "identity_layers": self.identity.get_layer_states(self.state.x)
        }

        # Compute force-based delta using Hierarchical Forces
        # We override the identity force in the aggregator by injecting the hierarchical force
        h_force = self.identity.compute_total_force(self.state.x)
        
        # Original compute_delta_x logic with hierarchical override
        delta_x = self.forces.compute_delta_x(
            self.state, observation, actions, context
        )
        
        # Override identity contribution with hierarchy
        delta_x = delta_x - self.state.gamma * (self.state.x_star - self.state.x) + h_force

        # Store predicted next state for prediction error calculation
        self.state.x_pred = self.state.x + delta_x

        # METACOGNITION: Introspect on current state
        report = self.metacognition.introspect(self.state.rho)
        if report:
            print(f"[DDA-X METALOG] {report}")

        # Check for protect mode or help request
        if self.metacognition.should_request_help(self.state.rho) or check_protect_mode(self.state, self.config.protect_threshold):
            return await self._protect_mode_action(observation, task_intent)

        # Run tree search
        best_action = self.mcts.search(
            self.tree,
            delta_x,
            actions,
            context
        )

        if best_action:
            return best_action.raw_action
        else:
            return {"action_type": "wait", "message": "No action selected"}

    async def observe_outcome(self, outcome: Any) -> None:
        """
        Process outcome and update Multi-Timescale Rigidity.
        """
        if self.outcome_encoder:
            x_actual = self.outcome_encoder.encode(outcome) if hasattr(self.outcome_encoder, 'encode') else self.outcome_encoder(outcome)
        else:
            x_actual = self.state.x + np.random.randn(self.config.state_dim) * 0.1

        # Compute prediction error
        epsilon = self.state.compute_prediction_error(x_actual)

        # Update Multi-timescale Dynamics
        pre_rho = self.state.rho
        dyn_results = self.dynamics.update(epsilon)
        self.state.rho = dyn_results["rho_effective"]
        post_rho = self.state.rho
        
        # Log rigidity results
        if abs(post_rho - pre_rho) > 0.0001 or epsilon > 0.1:
            print(f"[DDA-X] epsilon={epsilon:.3f}, ρ_eff: {pre_rho:.3f} → {post_rho:.3f} (Δ={post_rho-pre_rho:+.4f})")
            print(f"       Timescales: fast={dyn_results['rho_fast']:.3f}, slow={dyn_results['rho_slow']:.3f}, trauma={dyn_results['rho_trauma']:.4f}")

        # Store experience
        if self.ledger and self.current_task:
            entry = LedgerEntry(
                timestamp=time.time(),
                state_vector=self.state.x.copy(),
                action_id="last_action",
                observation_embedding=self.state.x.copy(),
                outcome_embedding=x_actual,
                prediction_error=epsilon,
                context_embedding=self.state.x.copy(),
                task_id=self.current_task,
                rigidity_at_time=self.state.rho,
                was_successful=None
            )
            self.ledger.add_entry(entry)
            self.task_trajectory.append(entry)

        # Update state
        self.state.x = x_actual

    async def end_task(self, success: bool) -> None:
        """End of task processing and reflection."""
        if not self.task_trajectory:
            return

        for entry in self.task_trajectory:
            entry.was_successful = success

        max_error = 0
        surprising_entry = None
        for entry in self.task_trajectory:
            if entry.prediction_error > max_error:
                max_error = entry.prediction_error
                surprising_entry = entry

        if surprising_entry and max_error > self.config.epsilon_0:
            reflection_text = await self._generate_reflection(surprising_entry, success)
            if self.ledger:
                self.ledger.add_reflection(ReflectionEntry(
                    timestamp=time.time(),
                    task_intent=self.current_task,
                    situation_embedding=surprising_entry.context_embedding,
                    reflection_text=reflection_text,
                    prediction_error=max_error,
                    outcome_success=success,
                ))

        # Reset search tree
        self.tree = None
        self.task_trajectory = []

    def get_state_info(self) -> Dict:
        """Get current agent state information, including diagnostics."""
        diag = self.dynamics.get_diagnostic()
        layer_states = self.identity.get_layer_states(self.state.x)
        
        return {
            "rigidity": self.state.rho,
            "rigidity_diagnostics": diag,
            "identity_layers": layer_states,
            "identity_distance": np.linalg.norm(self.state.x - (self.identity.core.x_star if self.identity.core else self.state.x)),
            "k_eff": self.state.k_eff,
            "in_protect_mode": check_protect_mode(self.state, self.config.protect_threshold),
            "current_task": self.current_task,
            "trajectory_length": len(self.task_trajectory)
        }

    async def _generate_action_directions(
        self,
        observation: Any,
        available_actions: List[Dict]
    ) -> List[ActionDirection]:
        if self.action_generator:
            return await self.action_generator(observation, available_actions)
        return self._create_dummy_actions(available_actions)

    def _create_dummy_actions(self, available_actions: List[Dict]) -> List[ActionDirection]:
        actions = []
        for i, action_dict in enumerate(available_actions[:self.config.branching_factor]):
            actions.append(ActionDirection(
                action_id=f"action_{i}",
                raw_action=action_dict,
                direction=np.random.randn(self.config.state_dim),
                prior_prob=1.0 / (len(available_actions) or 1)
            ))
        return actions

    async def _generate_reflection(self, entry: LedgerEntry, success: bool) -> str:
        return (
            f"Cognitive Insight: Task {'succeeded' if success else 'failed'}. "
            f"Extreme surprise detected (ε={entry.prediction_error:.2f}) at ρ={entry.rigidity_at_time:.2f}. "
            f"This traumatic event has left a permanent trace in my dynamics."
        )

    def _evaluate_state(self, observation: Any) -> float:
        return self.value_estimator.estimate(self.state, observation, {"task": self.current_task})

    async def _protect_mode_action(self, observation: Any, intent: str) -> Dict:
        """Metacognitive protect mode action."""
        print(f"[DDA-X] PROTECT MODE/HELP REQUEST: rho={self.state.rho:.3f}")
        
        # Metacognitive help request
        help_request = self.metacognition.generate_help_request(self.state.rho, context=f"Task: {intent}")
        
        return {
            "action_type": "clarify",
            "action": "wait",
            "protect_mode": True,
            "metacognitive_report": self.metacognition.introspect(self.state.rho),
            "help_request": help_request,
            "rigidity": self.state.rho,
            "observation_summary": str(observation)[:200]
        }