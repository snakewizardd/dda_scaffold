"""
Simulation and rollout policies for DDA-X.
"""

import numpy as np
from typing import Any, List, Dict, Optional, Callable
from abc import ABC, abstractmethod

from ..core.state import DDAState, ActionDirection
from ..core.decision import dda_x_select


class SimulationPolicy(ABC):
    """Abstract base for simulation policies."""

    @abstractmethod
    def select_action(
        self,
        state: DDAState,
        actions: List[ActionDirection],
        context: Dict
    ) -> ActionDirection:
        """Select action during rollout."""
        pass


class RandomPolicy(SimulationPolicy):
    """Random action selection."""

    def select_action(
        self,
        state: DDAState,
        actions: List[ActionDirection],
        context: Dict
    ) -> ActionDirection:
        import random
        return random.choice(actions)


class DDAPolicy(SimulationPolicy):
    """Use DDA-X selection during rollouts."""

    def __init__(self, delta_x: np.ndarray, c_explore: float = 0.5):
        self.delta_x = delta_x
        self.c_explore = c_explore

    def select_action(
        self,
        state: DDAState,
        actions: List[ActionDirection],
        context: Dict
    ) -> ActionDirection:
        # Simplified DDA selection for rollouts
        return dda_x_select(
            state,
            actions,
            self.delta_x,
            total_visits=1,  # Treat as first visit
            c=self.c_explore
        )


class Simulator:
    """Handles simulation/rollout from leaf nodes."""

    def __init__(
        self,
        env: Optional[Any] = None,
        policy: Optional[SimulationPolicy] = None,
        max_depth: int = 10,
        discount: float = 0.95
    ):
        self.env = env
        self.policy = policy or RandomPolicy()
        self.max_depth = max_depth
        self.discount = discount

    def rollout(
        self,
        initial_state: DDAState,
        initial_obs: Any,
        context: Dict
    ) -> float:
        """
        Perform rollout from initial state.

        Returns:
            Estimated value ∈ [0, 1]
        """
        if self.env is None:
            # No environment, return heuristic value
            return self._heuristic_value(initial_state, initial_obs, context)

        state = initial_state.copy()
        obs = initial_obs
        cumulative_reward = 0.0
        depth = 0

        while depth < self.max_depth:
            # Get available actions
            actions = self._get_actions(obs)
            if not actions:
                break

            # Select action using policy
            action = self.policy.select_action(state, actions, context)

            # Execute in environment
            next_obs, reward, done = self._step(obs, action)

            # Accumulate discounted reward
            cumulative_reward += (self.discount ** depth) * reward

            if done:
                break

            obs = next_obs
            depth += 1

        # Normalize to [0, 1]
        max_possible = sum(self.discount ** i for i in range(self.max_depth))
        return min(1.0, max(0.0, cumulative_reward / max_possible))

    def _heuristic_value(
        self,
        state: DDAState,
        obs: Any,
        context: Dict
    ) -> float:
        """
        Compute heuristic value when no environment available.

        Returns value based on:
        - Low rigidity (flexible) → higher value
        - Alignment with identity → higher value
        """
        # Component 1: Flexibility bonus
        flexibility = 1.0 - state.rho

        # Component 2: Identity alignment
        identity_distance = np.linalg.norm(state.x - state.x_star)
        max_distance = np.linalg.norm(state.x_star) * 2  # Rough normalization
        identity_alignment = 1.0 - min(1.0, identity_distance / max(max_distance, 1.0))

        # Combine
        value = 0.6 * flexibility + 0.4 * identity_alignment

        return value

    def _get_actions(self, obs: Any) -> List[ActionDirection]:
        """Get available actions for observation."""
        if self.env:
            return self.env.get_actions(obs)
        return []

    def _step(self, obs: Any, action: ActionDirection) -> tuple:
        """Execute action in environment."""
        if self.env:
            return self.env.step(obs, action)
        return obs, 0.0, True


class ValueEstimator:
    """Estimates state values using various methods."""

    def __init__(
        self,
        method: str = "heuristic",
        llm_provider: Optional[Any] = None,
        simulator: Optional[Simulator] = None
    ):
        self.method = method
        self.llm_provider = llm_provider
        self.simulator = simulator

    def estimate(
        self,
        state: DDAState,
        observation: Any,
        context: Dict
    ) -> float:
        """
        Estimate value of a state.

        Methods:
        - "heuristic": Fast heuristic based on state properties
        - "rollout": Monte Carlo rollout
        - "llm": LLM-based evaluation
        - "debate": Multi-agent debate (ExACT-style)
        """
        if self.method == "heuristic":
            return self._heuristic_value(state, observation)
        elif self.method == "rollout" and self.simulator:
            return self.simulator.rollout(state, observation, context)
        elif self.method == "llm" and self.llm_provider:
            return self._llm_value(observation, context)
        elif self.method == "debate" and self.llm_provider:
            return self._debate_value(observation, context)
        else:
            # Fallback to heuristic
            return self._heuristic_value(state, observation)

    def _heuristic_value(self, state: DDAState, obs: Any) -> float:
        """Simple heuristic value."""
        flexibility = 1.0 - state.rho
        return 0.5 + 0.5 * flexibility  # [0.5, 1.0] range

    async def _llm_value_async(self, obs: Any, context: Dict) -> float:
        """Single LLM evaluation (async)."""
        if not self.llm_provider:
            return 0.5
        
        # Format observation for LLM
        if isinstance(obs, dict):
            obs_text = str(obs.get("text", obs))
        else:
            obs_text = str(obs)
        
        intent = context.get("task", context.get("intent", "complete the task"))
        trajectory = context.get("trajectory", None)
        
        try:
            value = await self.llm_provider.estimate_value(
                observation=obs_text,
                intent=intent,
                trajectory=trajectory
            )
            return value
        except Exception as e:
            print(f"[ValueEstimator] LLM error: {e}")
            return 0.5

    def _llm_value(self, obs: Any, context: Dict) -> float:
        """Single LLM evaluation."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't use run_until_complete in running loop
                return 0.5
            return loop.run_until_complete(self._llm_value_async(obs, context))
        except RuntimeError:
            return asyncio.run(self._llm_value_async(obs, context))

    async def _debate_value_async(self, obs: Any, context: Dict) -> float:
        """Multi-agent debate evaluation (async)."""
        if not self.llm_provider:
            return 0.5
        
        # Format inputs
        if isinstance(obs, dict):
            obs_text = str(obs.get("text", obs))
        else:
            obs_text = str(obs)
        
        intent = context.get("task", context.get("intent", "complete the task"))
        trajectory_str = ""
        if context.get("trajectory"):
            trajectory_str = f"\nActions taken: {', '.join(context['trajectory'])}"
        
        try:
            # Proponent argument
            pro_prompt = f"""Task: {intent}
Current situation: {obs_text}{trajectory_str}

Argue why this situation IS promising for task success. Be specific."""
            
            pro = await self.llm_provider.complete(
                pro_prompt,
                system_prompt="You are an optimistic evaluator. Find reasons for hope.",
                temperature=0.6
            )
            
            # Opponent argument
            con_prompt = f"""Task: {intent}
Current situation: {obs_text}{trajectory_str}

Argue why this situation is NOT promising for task success. Be specific."""
            
            con = await self.llm_provider.complete(
                con_prompt,
                system_prompt="You are a skeptical evaluator. Find potential problems.",
                temperature=0.6
            )
            
            # Judge synthesizes
            judge_prompt = f"""Task: {intent}
Situation: {obs_text}

Proponent says: {pro}

Opponent says: {con}

Considering both arguments, estimate probability of success (0-100).
Respond with only a number."""
            
            judgment = await self.llm_provider.complete(
                judge_prompt,
                system_prompt="You are a fair judge weighing evidence.",
                temperature=0.3,
                max_tokens=16
            )
            
            # Parse number
            import re
            numbers = re.findall(r'\d+', judgment)
            if numbers:
                value = float(numbers[0]) / 100.0
                return min(1.0, max(0.0, value))
            
        except Exception as e:
            print(f"[ValueEstimator] Debate error: {e}")
        
        return 0.5

    def _debate_value(self, obs: Any, context: Dict) -> float:
        """Multi-agent debate evaluation."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return 0.5
            return loop.run_until_complete(self._debate_value_async(obs, context))
        except RuntimeError:
            return asyncio.run(self._debate_value_async(obs, context))