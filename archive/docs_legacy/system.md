# DDA-X: Dynamic Decision Algorithm with Exploration
## Technical Architecture for Implementation

---

## Executive Summary

This document bridges your DDA theoretical framework with ExACT's engineering patterns to create **DDA-X** — a fully implementable agent framework that preserves your core insights (identity persistence, surprise→rigidity, force-balanced decisions) while adding:

- **Tree search** for multi-step lookahead
- **Exploration bonuses** for trying new actions  
- **Reflection database** for learning from experience
- **Multi-agent debate** for calibrated state evaluation

---

## Part 1: Understanding ExACT's Tech Stack

### 1.1 How ExACT's Code Maps to Their Math

| Math Concept | Code Location | Implementation |
|--------------|---------------|----------------|
| **Q(s,a)** — Action value | `mcts_agent.py:257` | `self.Q: dict = {}` — nested dict `{state_hash: {action: float}}` |
| **N(s)** — State visits | `mcts_agent.py:255` | `self.Ns: dict = {}` — dict `{state_hash: int}` |
| **N(s,a)** — Action visits | `mcts_agent.py:256` | `self.Nsa: dict = {}` — nested dict |
| **P(s,a)** — Prior probability | `mcts_agent.py:258` | `self.P: dict = {}` — from LLM sampling frequency |
| **V(s)** — State value | `value_function.py` | LLM call → float ∈ [0,1] |
| **UCT selection** | `mcts_agent.py:595-611` | `uct = qsa + cpuct * p * sqrt(Ns) / (1 + nsa)` |
| **Backpropagation** | `mcts_agent.py:627` | Incremental mean update |
| **Reflection retrieval** | `rpolicy.py:599-618` | FAISS vector similarity search |
| **Reflection generation** | `rpolicy.py:506-550` | LLM prompted with (state, action, outcome, surprise) |

### 1.2 ExACT's Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MAIN LOOP                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. OBSERVE: Get browser screenshot + accessibility tree             │
│       │                                                              │
│       ▼                                                              │
│  2. EXPAND: Sample N actions from LLM, count frequencies → P(a|s)   │
│       │                                                              │
│       ▼                                                              │
│  3. SELECT: For each candidate action, compute UCT score:           │
│             UCT(a) = Q(s,a) + c × P(a|s) × √N(s)/(1+N(s,a))        │
│       │                                                              │
│       ▼                                                              │
│  4. SIMULATE: Execute best action, get next state                   │
│       │                                                              │
│       ▼                                                              │
│  5. EVALUATE: Call V(s') via LLM with rubric + debate               │
│       │                                                              │
│       ▼                                                              │
│  6. BACKPROPAGATE: Q(s,a) ← running average with new V(s')          │
│       │                                                              │
│       ▼                                                              │
│  7. REPEAT until budget exhausted or V(s') = 1.0                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (on task end)
┌─────────────────────────────────────────────────────────────────────┐
│                     REFLECTION PHASE                                 │
├─────────────────────────────────────────────────────────────────────┤
│  1. Compute surprise(a) = |V_next - Q| for all actions              │
│  2. Select most surprising (state, action, outcome)                 │
│  3. Prompt LLM: "What would you do differently?"                    │
│  4. Embed reflection text via OpenAI embeddings                     │
│  5. Store in FAISS index for future retrieval                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Data Structures

```python
# Node in search tree (mcts_agent.py:38-88)
@dataclass
class Node:
    env: BrowserEnv                    # Can execute actions
    trajectory: list[StateInfo|Action] # History: [s0, a0, s1, a1, ...]
    action_trajectory: list[Action]    # Just actions taken
    action_trajectory_str: list[str]   # String descriptions
    value: float                       # V(s) from evaluation
    children: dict[Action, 'Node']     # Child nodes by action
    Ns: int                            # Visit count
    depth: int                         # Tree depth
    is_terminal: bool                  # Task complete?
    _additional_info: dict             # Screenshots, metadata

# Reflection record (rpolicy.py:36-65)
@dataclass  
class ReflectionRecord:
    intent: str                        # Task goal
    state_str: str                     # Observation text
    state_img_arr: np.ndarray          # Screenshot
    action_str: str                    # Action taken
    next_state_str: str                # Outcome observation
    reflection: str                    # LLM-generated lesson
    _from_task_hash: int               # Which task this came from
```

---

## Part 2: DDA-X Architecture

### 2.1 Core Equation (Your DDA + ExACT Enhancements)

**Original DDA:**
```
Fₙ = P₀ × kFₙ₋₁ + m(T(f(Iₙ, IΔ)) + R(Dₙ, FMₙ))
```

**DDA-X (Enhanced):**
```
Aₙ = argmax_a [ Score(a) ]

Score(a) = cos(Δx, d̂(a)) + c × P(a|s) × √N(s)/(1+N(s,a)) × (1 - ρₙ)
                ↑                        ↑                      ↑
          DDA alignment          ExACT exploration       Rigidity dampening
          
Where:
  Δx = k_eff × [γ(x* - xₙ) + mₙ(T(Iₙ, IΔ) + R(Dₙ, FMₙ))]
  k_eff = k_base × (1 - ρₙ)
  ρₙ₊₁ = clip(ρₙ + α[σ((εₙ - ε₀)/s) - 0.5], 0, 1)
  εₙ = ||x_pred - x_actual||₂
```

**What this adds:**
1. **Exploration term**: `c × P(a|s) × √N(s)/(1+N(s,a))` encourages trying new actions
2. **Rigidity dampening**: `(1 - ρₙ)` reduces exploration when surprised (your signature move!)
3. **Tree search**: Multiple simulation rollouts before committing

### 2.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DDA-X AGENT                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     STATE MANAGER                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │   │
│  │  │ x_t ∈ ℝ^d    │  │ x* ∈ ℝ^d    │  │ ρ_t ∈ [0,1]          │   │   │
│  │  │ Current      │  │ Identity     │  │ Rigidity             │   │   │
│  │  │ State Vector │  │ Attractor    │  │ (increases w/surprise)│   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                  │                                       │
│                                  ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     FORCE CALCULATOR                              │   │
│  │                                                                    │   │
│  │  F_id = γ(x* - x_t)           ← Identity pull                    │   │
│  │  F_T  = T(I_t, IΔ) - x_t      ← Truth channel (observations)     │   │
│  │  F_R  = R(D_t, FM_t) - x_t    ← Reflection channel (goals+prefs) │   │
│  │                                                                    │   │
│  │  Δx = k_eff × [F_id + m_t(F_T + F_R)]                            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                  │                                       │
│                                  ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     TREE SEARCH ENGINE                            │   │
│  │                                                                    │   │
│  │  For each candidate action a in A_t:                              │   │
│  │    alignment = cos(Δx, d̂(a))                                     │   │
│  │    exploration = c × P(a|s) × √N(s)/(1+N(s,a))                   │   │
│  │    rigidity_damping = (1 - ρ_t)                                   │   │
│  │    score(a) = alignment + exploration × rigidity_damping          │   │
│  │                                                                    │   │
│  │  Select a* = argmax score(a)                                      │   │
│  │  Simulate: x'_pred, outcome = execute(a*)                         │   │
│  │  Backpropagate: update Q, N statistics                            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                  │                                       │
│                                  ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     PREDICTION ERROR & RIGIDITY                   │   │
│  │                                                                    │   │
│  │  ε_t = ||x_pred - E(outcome)||₂                                   │   │
│  │  ρ_{t+1} = clip(ρ_t + α[σ((ε_t - ε₀)/s) - 0.5], 0, 1)            │   │
│  │                                                                    │   │
│  │  If ε_t > ε_protect:                                              │   │
│  │    → Enter protect mode (reduce action set, increase γ)          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                  │                                       │
│                                  ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     MEMORY SYSTEM (Ledger)                        │   │
│  │                                                                    │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐    │   │
│  │  │ FAISS Index    │  │ Experience DB  │  │ Reflection Store │    │   │
│  │  │ (embeddings)   │  │ (x,a,o,ε,c)   │  │ (lessons learned)│    │   │
│  │  └────────────────┘  └────────────────┘  └──────────────────┘    │   │
│  │                                                                    │   │
│  │  Retrieval score = sim(c_now, c_t) × e^{-λ_r(now-t)} × (1+λ_ε×ε) │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Implementation Blueprint

### 3.1 Project Structure

```
dda-x/
├── src/
│   ├── core/
│   │   ├── state.py              # StateVector, IdentityAttractor classes
│   │   ├── forces.py             # TruthChannel, ReflectionChannel, IdentityPull
│   │   ├── dynamics.py           # State update equations, rigidity
│   │   └── decision.py           # Score function, action selection
│   │
│   ├── search/
│   │   ├── tree.py               # Node, SearchTree classes
│   │   ├── mcts.py               # UCT selection, backpropagation
│   │   └── simulation.py         # Rollout, value estimation
│   │
│   ├── memory/
│   │   ├── ledger.py             # Experience storage
│   │   ├── embeddings.py         # OpenAI/local embedding interface
│   │   ├── retriever.py          # FAISS-based retrieval
│   │   └── reflection.py         # Reflection generation & storage
│   │
│   ├── channels/
│   │   ├── truth.py              # T(I, IΔ) — observation processing
│   │   ├── reflection.py         # R(D, FM) — goal/preference processing
│   │   └── encoders.py           # Observation → ℝ^d, Outcome → ℝ^d
│   │
│   ├── llm/
│   │   ├── providers.py          # OpenAI, Azure, Anthropic, local
│   │   ├── prompts.py            # Prompt templates
│   │   ├── debate.py             # Multi-agent debate for V(s)
│   │   └── rubrics.py            # Task-specific rubric generation
│   │
│   ├── env/
│   │   ├── base.py               # Abstract environment interface
│   │   ├── browser.py            # Playwright-based browser env
│   │   └── mock.py               # For testing
│   │
│   └── agent.py                  # Main DDAXAgent class
│
├── configs/
│   ├── default.yaml              # Default hyperparameters
│   ├── identity/                 # Identity attractor definitions
│   │   ├── cautious.yaml
│   │   ├── exploratory.yaml
│   │   └── task_focused.yaml
│   └── llm/
│       └── providers.yaml        # API endpoints, models
│
├── data/
│   ├── reflections/              # Stored reflection records
│   ├── embeddings/               # FAISS indices
│   └── experiences/              # Ledger entries
│
├── runners/
│   ├── run_task.py               # Single task execution
│   ├── run_batch.py              # Batch evaluation
│   └── analyze_tree.py           # Search tree visualization
│
└── tests/
    ├── test_dynamics.py
    ├── test_search.py
    └── test_memory.py
```

### 3.2 Core Classes

#### `src/core/state.py`

```python
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
    
    @classmethod
    def from_identity_config(cls, config: dict, dim: int = 64) -> "DDAState":
        """Initialize state from identity configuration."""
        x_star = np.array(config.get("identity_vector", np.zeros(dim)))
        return cls(
            x=x_star.copy(),  # Start at identity
            x_star=x_star,
            gamma=config.get("gamma", 1.0),
            epsilon_0=config.get("epsilon_0", 0.3),
            alpha=config.get("alpha", 0.1),
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
    
    @property
    def d_hat(self) -> np.ndarray:
        """Normalized direction vector."""
        norm = np.linalg.norm(self.direction)
        if norm < 1e-8:
            return self.direction
        return self.direction / norm
```

#### `src/core/forces.py`

```python
import numpy as np
from abc import ABC, abstractmethod
from typing import Any

class ForceChannel(ABC):
    """Abstract base for force channels."""
    
    @abstractmethod
    def compute(self, state: "DDAState", observation: Any) -> np.ndarray:
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
    
    def __init__(self, encoder: "ObservationEncoder"):
        self.encoder = encoder
        self.prev_embedding = None
    
    def compute(self, state: "DDAState", observation: Any) -> np.ndarray:
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
    
    def __init__(self, scorer: "ActionScorer"):
        self.scorer = scorer
    
    def compute(
        self, 
        state: "DDAState", 
        actions: list["ActionDirection"],
        context: dict
    ) -> np.ndarray:
        # Score each action (objective + subjective)
        scores = self.scorer.score_actions(actions, context)
        
        # Softmax to get preference distribution
        tau = 2.0  # Temperature
        exp_scores = np.exp(tau * np.array(scores))
        probs = exp_scores / exp_scores.sum()
        
        # Target = current + weighted sum of action directions
        # R = x_t + Σ π(a) × d̂(a)
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
        actions: list["ActionDirection"],
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
        actions: list["ActionDirection"],
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
```

#### `src/core/decision.py`

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class DecisionConfig:
    """Hyperparameters for action selection."""
    c_explore: float = 1.0        # Exploration constant
    use_rigidity_damping: bool = True
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
        actions: list["ActionDirection"],
        state: "DDAState",
        total_state_visits: int
    ) -> list[float]:
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
        actions: list["ActionDirection"],
        state: "DDAState",
        total_state_visits: int
    ) -> "ActionDirection":
        """Select the highest-scoring action."""
        
        scores = self.compute_scores(delta_x, actions, state, total_state_visits)
        best_idx = np.argmax(scores)
        return actions[best_idx]
    
    def select_with_threshold(
        self,
        delta_x: np.ndarray,
        actions: list["ActionDirection"],
        state: "DDAState",
        total_state_visits: int
    ) -> Optional["ActionDirection"]:
        """
        Select action only if it meets alignment threshold.
        Returns None if all actions are too misaligned (protect mode).
        """
        scores = self.compute_scores(delta_x, actions, state, total_state_visits)
        
        # Check if any action meets threshold
        valid_actions = [
            (a, s) for a, s in zip(actions, scores)
            if s >= self.config.min_alignment_threshold
        ]
        
        if not valid_actions:
            return None  # Trigger protect mode
        
        best_action = max(valid_actions, key=lambda x: x[1])[0]
        return best_action
```

#### `src/search/tree.py`

```python
from dataclasses import dataclass, field
from typing import Optional, Any
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
    children: dict["ActionDirection", "DDANode"] = field(default_factory=dict)
    
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
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        return self.parent is None
    
    def get_trajectory(self) -> list["ActionDirection"]:
        """Get sequence of actions from root to this node."""
        actions = []
        node = self
        while node.parent is not None:
            actions.append(node.parent_action)
            node = node.parent
        return list(reversed(actions))


class DDASearchTree:
    """Manages the search tree for DDA-X."""
    
    def __init__(self, root_observation: Any, initial_state: "DDAState"):
        self.root = DDANode(
            observation=root_observation,
            dda_state=initial_state.copy()
        )
        
        # Global statistics
        self.Q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.N: dict[str, int] = defaultdict(int)
        self.Na: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def get_node_hash(self, node: DDANode) -> str:
        """Create hashable identifier for a node."""
        trajectory = node.get_trajectory()
        return " -> ".join(a.action_id for a in trajectory) or "ROOT"
    
    def backpropagate(self, leaf: DDANode, value: float) -> None:
        """
        Backpropagate value up the tree.
        
        Q(s,a) ← [N(s,a) × Q(s,a) + v] / [N(s,a) + 1]
        """
        node = leaf
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
            
            node = parent
    
    def get_best_action(self, node: DDANode) -> "ActionDirection":
        """Get best action from node based on visit count (robust child)."""
        node_hash = self.get_node_hash(node)
        
        best_action = None
        best_visits = -1
        
        for action, child in node.children.items():
            visits = self.Na[node_hash][action.action_id]
            if visits > best_visits:
                best_visits = visits
                best_action = action
        
        return best_action
```

#### `src/memory/ledger.py`

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import time
from pathlib import Path
import pickle
import lzma

@dataclass
class LedgerEntry:
    """Single entry in the experience ledger."""
    
    timestamp: float                       # When this happened
    state_vector: np.ndarray               # x_t at decision time
    action_id: str                         # Action taken
    observation_embedding: np.ndarray      # Encoded observation
    outcome_embedding: np.ndarray          # Encoded outcome
    prediction_error: float                # ε_t = ||x_pred - x_actual||
    context_embedding: np.ndarray          # For retrieval similarity
    
    # Metadata
    task_id: Optional[str] = None
    rigidity_at_time: float = 0.0
    was_successful: Optional[bool] = None


@dataclass
class ReflectionEntry:
    """A learned lesson from experience."""
    
    timestamp: float
    task_intent: str                       # What we were trying to do
    situation_embedding: np.ndarray        # Embedded (state, action)
    reflection_text: str                   # LLM-generated lesson
    prediction_error: float                # How surprising this was
    outcome_success: bool                  # Did it work?


class ExperienceLedger:
    """
    DDA's memory system.
    
    Retrieval score = sim(c_now, c_t) × e^{-λ_r(now-t)} × (1 + λ_ε × ε_t)
    """
    
    def __init__(
        self,
        storage_path: Path,
        lambda_recency: float = 0.01,      # Recency decay
        lambda_salience: float = 1.0,      # Salience (surprise) weight
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.lambda_r = lambda_recency
        self.lambda_e = lambda_salience
        
        self.entries: List[LedgerEntry] = []
        self.reflections: List[ReflectionEntry] = []
        
        self._load()
    
    def add_entry(self, entry: LedgerEntry) -> None:
        """Add new experience to ledger."""
        self.entries.append(entry)
        self._save_entry(entry)
    
    def add_reflection(self, reflection: ReflectionEntry) -> None:
        """Add new reflection to memory."""
        self.reflections.append(reflection)
        self._save_reflection(reflection)
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        min_score: float = 0.2
    ) -> List[LedgerEntry]:
        """
        Retrieve top-k relevant experiences.
        
        score = similarity × recency × salience
        """
        now = time.time()
        scored_entries = []
        
        for entry in self.entries:
            # Cosine similarity
            sim = np.dot(query_embedding, entry.context_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(entry.context_embedding)
                + 1e-8
            )
            
            # Recency decay
            age = now - entry.timestamp
            recency = np.exp(-self.lambda_r * age)
            
            # Salience (surprise) boost
            salience = 1 + self.lambda_e * entry.prediction_error
            
            # Combined score
            score = sim * recency * salience
            
            if score >= min_score:
                scored_entries.append((score, entry))
        
        # Sort by score descending
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        
        return [entry for _, entry in scored_entries[:k]]
    
    def retrieve_reflections(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        min_score: float = 0.25
    ) -> List[ReflectionEntry]:
        """Retrieve relevant learned lessons."""
        scored = []
        
        for ref in self.reflections:
            sim = np.dot(query_embedding, ref.situation_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(ref.situation_embedding)
                + 1e-8
            )
            
            # Boost reflections from surprising situations
            salience = 1 + self.lambda_e * ref.prediction_error
            score = sim * salience
            
            if score >= min_score:
                scored.append((score, ref))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]
    
    def _save_entry(self, entry: LedgerEntry) -> None:
        path = self.storage_path / f"entry_{hash(entry.timestamp)}.pkl.xz"
        with lzma.open(path, "wb") as f:
            pickle.dump(entry, f)
    
    def _save_reflection(self, reflection: ReflectionEntry) -> None:
        path = self.storage_path / f"reflection_{hash(reflection.timestamp)}.pkl.xz"
        with lzma.open(path, "wb") as f:
            pickle.dump(reflection, f)
    
    def _load(self) -> None:
        """Load all entries from storage."""
        for path in self.storage_path.glob("entry_*.pkl.xz"):
            with lzma.open(path, "rb") as f:
                self.entries.append(pickle.load(f))
        
        for path in self.storage_path.glob("reflection_*.pkl.xz"):
            with lzma.open(path, "rb") as f:
                self.reflections.append(pickle.load(f))
```

#### `src/agent.py`

```python
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, List
import asyncio
import time

from .core.state import DDAState, ActionDirection
from .core.forces import ForceAggregator, IdentityPull, TruthChannel, ReflectionChannel
from .core.decision import DDADecisionMaker, DecisionConfig
from .search.tree import DDASearchTree, DDANode
from .memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from .channels.encoders import ObservationEncoder, OutcomeEncoder
from .llm.providers import LLMProvider


@dataclass
class DDAXConfig:
    """Agent configuration."""
    
    # DDA parameters
    gamma: float = 1.0                     # Identity stiffness
    k_base: float = 0.5                    # Base step size
    m: float = 1.0                         # External pressure
    epsilon_0: float = 0.3                 # Surprise threshold
    alpha: float = 0.1                     # Rigidity learning rate
    
    # Search parameters
    c_explore: float = 1.0                 # Exploration constant
    max_iterations: int = 50               # Search budget
    branching_factor: int = 5              # Actions to consider per state
    
    # State space
    state_dim: int = 64                    # Dimension of x ∈ ℝ^d
    
    # Protect mode
    protect_threshold: float = 0.7         # Enter protect if ρ > this
    

class DDAXAgent:
    """
    DDA-X: Dynamic Decision Algorithm with Exploration.
    
    Combines your DDA theory with ExACT's engineering:
    - Force-balanced state updates
    - Surprise → rigidity dynamics  
    - Tree search with exploration
    - Reflection-based learning
    """
    
    def __init__(
        self,
        config: DDAXConfig,
        llm_provider: LLMProvider,
        observation_encoder: ObservationEncoder,
        outcome_encoder: OutcomeEncoder,
        ledger: ExperienceLedger,
        identity_config: dict,
    ):
        self.config = config
        self.llm = llm_provider
        self.obs_encoder = observation_encoder
        self.outcome_encoder = outcome_encoder
        self.ledger = ledger
        
        # Initialize DDA state from identity
        self.state = DDAState.from_identity_config(
            identity_config, 
            dim=config.state_dim
        )
        self.state.gamma = config.gamma
        self.state.k_base = config.k_base
        self.state.m = config.m
        self.state.epsilon_0 = config.epsilon_0
        self.state.alpha = config.alpha
        
        # Force channels
        self.forces = ForceAggregator(
            identity_pull=IdentityPull(),
            truth_channel=TruthChannel(observation_encoder),
            reflection_channel=ReflectionChannel(self._create_scorer())
        )
        
        # Decision maker
        self.decision_maker = DDADecisionMaker(
            DecisionConfig(c_explore=config.c_explore)
        )
        
        # Search tree (initialized per task)
        self.tree: Optional[DDASearchTree] = None
        
        # Current task context
        self.current_task: Optional[str] = None
    
    async def decide(
        self,
        observation: Any,
        available_actions: List[dict],
        task_intent: str
    ) -> dict:
        """
        Main decision method.
        
        1. Encode observation
        2. Generate action directions
        3. Compute forces
        4. Search over actions
        5. Update rigidity based on outcome
        """
        
        # Initialize tree if new task
        if self.tree is None or self.current_task != task_intent:
            self.tree = DDASearchTree(observation, self.state)
            self.current_task = task_intent
        
        # Get action directions from LLM
        actions = await self._generate_action_directions(observation, available_actions)
        
        # Retrieve relevant reflections
        query_emb = self.obs_encoder.encode(observation)
        reflections = self.ledger.retrieve_reflections(query_emb, k=3)
        
        # Build context
        context = {
            "intent": task_intent,
            "reflections": [r.reflection_text for r in reflections],
            "rigidity": self.state.rho,
        }
        
        # Compute force-based delta
        delta_x = self.forces.compute_delta_x(
            self.state, observation, actions, context
        )
        
        # Check for protect mode
        if self.state.rho > self.config.protect_threshold:
            return await self._protect_mode_action(observation, task_intent)
        
        # Run tree search
        best_action = await self._search(
            observation, actions, delta_x, context
        )
        
        return best_action.raw_action
    
    async def observe_outcome(self, outcome: Any) -> None:
        """
        Process outcome and update rigidity.
        
        1. Encode outcome to state space
        2. Compute prediction error
        3. Update rigidity
        4. Store experience
        """
        # Encode outcome
        x_actual = self.outcome_encoder.encode(outcome)
        
        # Compute prediction error
        epsilon = self.state.compute_prediction_error(x_actual)
        
        # Update rigidity (the DDA signature move!)
        self.state.update_rigidity(epsilon)
        
        # Update state
        self.state.x = x_actual
        
        # Log
        print(f"Prediction error: {epsilon:.3f}, Rigidity: {self.state.rho:.3f}")
    
    async def end_task(self, success: bool, trajectory: List) -> None:
        """
        End of task processing.
        
        1. Identify surprising transitions
        2. Generate reflections
        3. Store in ledger
        """
        # Find most surprising transition
        max_error = 0
        surprising_entry = None
        
        for entry in trajectory:
            if entry.prediction_error > max_error:
                max_error = entry.prediction_error
                surprising_entry = entry
        
        if surprising_entry and max_error > self.config.epsilon_0:
            # Generate reflection via LLM
            reflection_text = await self._generate_reflection(
                surprising_entry, success
            )
            
            # Store
            self.ledger.add_reflection(ReflectionEntry(
                timestamp=time.time(),
                task_intent=self.current_task,
                situation_embedding=surprising_entry.context_embedding,
                reflection_text=reflection_text,
                prediction_error=max_error,
                outcome_success=success,
            ))
    
    async def _search(
        self,
        observation: Any,
        actions: List[ActionDirection],
        delta_x: np.ndarray,
        context: dict
    ) -> ActionDirection:
        """Run tree search to find best action."""
        
        current_node = self.tree.root
        
        for iteration in range(self.config.max_iterations):
            # Selection: traverse tree using DDA-X scoring
            node = current_node
            while not node.is_leaf() and not node.is_terminal:
                node_hash = self.tree.get_node_hash(node)
                total_visits = self.tree.N[node_hash]
                
                best_action = self.decision_maker.select_action(
                    delta_x,
                    list(node.children.keys()),
                    node.dda_state,
                    total_visits
                )
                node = node.children[best_action]
            
            # Expansion: add children for unexplored actions
            if node.is_leaf() and not node.is_terminal:
                for action in actions:
                    child = DDANode(
                        observation=None,  # Will be filled by simulation
                        dda_state=node.dda_state.copy(),
                        parent_action=action,
                        parent=node,
                        depth=node.depth + 1,
                    )
                    node.children[action] = child
            
            # Simulation: estimate value
            value = await self._evaluate_state(node)
            
            # Backpropagation
            self.tree.backpropagate(node, value)
        
        # Return most-visited action (robust child)
        return self.tree.get_best_action(current_node)
    
    async def _evaluate_state(self, node: DDANode) -> float:
        """Estimate V(s) using LLM."""
        # Use debate-based evaluation like ExACT
        # ... implementation details ...
        pass
    
    async def _generate_action_directions(
        self,
        observation: Any,
        available_actions: List[dict]
    ) -> List[ActionDirection]:
        """Sample actions from LLM and compute their directions."""
        # ... implementation details ...
        pass
    
    async def _generate_reflection(
        self, 
        entry: LedgerEntry, 
        success: bool
    ) -> str:
        """Generate reflection on surprising outcome."""
        prompt = f"""
        I was in this situation and took this action.
        The outcome was {'successful' if success else 'unsuccessful'}.
        The outcome was surprising (prediction error: {entry.prediction_error:.2f}).
        
        What lesson should I learn from this?
        What would I do differently next time?
        Keep response under 100 words.
        """
        return await self.llm.complete(prompt)
    
    async def _protect_mode_action(
        self, 
        observation: Any, 
        intent: str
    ) -> dict:
        """Conservative action when rigidity is high."""
        # In protect mode:
        # - Reduce action set to safe defaults
        # - Increase identity pull (γ)
        # - Ask for clarification instead of acting
        
        return {
            "action_type": "clarify",
            "message": "I'm uncertain about this situation. Can you provide more guidance?"
        }
    
    def _create_scorer(self):
        """Create action scorer for reflection channel."""
        # ... implementation details ...
        pass
```

---

## Part 4: Key Algorithms

### 4.1 The DDA-X Selection Algorithm

```python
def dda_x_select(state: DDAState, actions: List[ActionDirection], 
                  delta_x: np.ndarray, total_visits: int, 
                  c: float = 1.0) -> ActionDirection:
    """
    DDA-X action selection.
    
    Score(a) = cos(Δx, d̂(a)) + c × P(a|s) × √N(s)/(1+N(s,a)) × (1 - ρ)
    
    Key insight: When surprised (high ρ), exploration is dampened,
    and action selection becomes more conservative (alignment-focused).
    """
    
    best_score = float('-inf')
    best_action = None
    
    delta_x_norm = np.linalg.norm(delta_x)
    rigidity_factor = 1 - state.rho  # DDA signature!
    
    for action in actions:
        # Component 1: DDA alignment (your original insight)
        if delta_x_norm > 1e-8:
            alignment = np.dot(delta_x, action.d_hat) / delta_x_norm
        else:
            alignment = 0.0
        
        # Component 2: UCT exploration (from ExACT)
        if total_visits == 0:
            exploration = c * action.prior_prob
        else:
            exploration = c * action.prior_prob * np.sqrt(total_visits) / (1 + action.N)
        
        # Component 3: Rigidity dampening (DDA + exploration fusion)
        # When surprised: ρ↑ → rigidity_factor↓ → less exploration
        score = alignment + exploration * rigidity_factor
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return best_action
```

### 4.2 The Rigidity Update

```python
def update_rigidity(state: DDAState, x_actual: np.ndarray) -> None:
    """
    Update rigidity based on prediction error.
    
    ρ_{t+1} = clip(ρ_t + α[σ((ε - ε₀)/s) - 0.5], 0, 1)
    
    Key insight: This is bidirectional!
    - High surprise (ε > ε₀): rigidity increases
    - Low surprise (ε < ε₀): rigidity decreases (relaxation)
    """
    
    if state.x_pred is None:
        return
    
    # Prediction error
    epsilon = np.linalg.norm(state.x_pred - x_actual)
    
    # Centered sigmoid (your correction from the doc!)
    z = (epsilon - state.epsilon_0) / state.s
    sigmoid = 1 / (1 + np.exp(-z))
    delta_rho = state.alpha * (sigmoid - 0.5)  # Centered around 0
    
    # Update with clipping
    state.rho = np.clip(state.rho + delta_rho, 0.0, 1.0)
```

### 4.3 Force Computation

```python
def compute_forces(state: DDAState, 
                   observation: Any,
                   actions: List[ActionDirection],
                   encoders: dict,
                   context: dict) -> np.ndarray:
    """
    Compute DDA force balance.
    
    Δx = k_eff × [γ(x* - x) + m(F_T + F_R)]
    
    Maps your original equation:
    Fₙ = P₀ × kFₙ₋₁ + m(T(f(Iₙ, IΔ)) + R(Dₙ, FMₙ))
    """
    
    # === F_id: Identity pull (P₀ × kFₙ₋₁ in your notation) ===
    F_id = state.gamma * (state.x_star - state.x)
    
    # === F_T: Truth channel (T(f(Iₙ, IΔ)) in your notation) ===
    obs_embedding = encoders['observation'].encode(observation)
    x_T = obs_embedding  # Target state from observation
    F_T = x_T - state.x
    
    # === F_R: Reflection channel (R(Dₙ, FMₙ) in your notation) ===
    # Score actions and compute weighted direction
    scores = []
    for a in actions:
        q_score = a.Q  # Objective: expected value
        s_score = np.dot(a.d_hat, state.x_star - state.x)  # Subjective: identity alignment
        combined = 0.7 * q_score + 0.3 * s_score  # w_obj, w_subj from your doc
        scores.append(combined)
    
    # Softmax for preference distribution
    tau = 2.0
    probs = np.exp(tau * np.array(scores))
    probs = probs / probs.sum()
    
    # Weighted average direction
    weighted_dir = sum(p * a.d_hat for p, a in zip(probs, actions))
    x_R = state.x + weighted_dir
    F_R = x_R - state.x
    
    # === Combine with effective step size ===
    delta_x = state.k_eff * (F_id + state.m * (F_T + F_R))
    
    return delta_x
```

---

## Part 5: Configuration Examples

### 5.1 Identity Configurations

```yaml
# configs/identity/cautious.yaml
# A cautious agent that becomes rigid quickly

identity_vector: null  # Will be initialized from task embedding
gamma: 2.0             # Strong identity pull
epsilon_0: 0.2         # Low surprise threshold (gets rigid easily)
alpha: 0.2             # Fast rigidity increase
s: 0.1                 # Sharp sigmoid
k_base: 0.3            # Small steps
m: 0.5                 # Low external pressure penetration
```

```yaml
# configs/identity/exploratory.yaml
# An exploratory agent that stays open longer

identity_vector: null
gamma: 0.5             # Weak identity pull
epsilon_0: 0.6         # High surprise threshold (tolerant)
alpha: 0.05            # Slow rigidity change
s: 0.3                 # Gradual sigmoid
k_base: 0.7            # Large steps
m: 1.5                 # High external pressure penetration
```

```yaml
# configs/identity/traumatized.yaml
# From your document: low ε₀, high α, high baseline ρ

identity_vector: null
gamma: 1.5
epsilon_0: 0.1         # Very low threshold (hair-trigger)
alpha: 0.3             # Fast rigidity ramp
s: 0.05                # Very sharp
k_base: 0.4
m: 0.7
initial_rho: 0.4       # Start with elevated baseline
```

---

## Part 6: What This Gives You Over ExACT

| Feature | ExACT | DDA-X |
|---------|-------|-------|
| **Identity persistence** | ❌ None | ✅ x* attractor + γ stiffness |
| **Surprise → rigidity** | ❌ Opposite (surprise → learn) | ✅ Core mechanism |
| **Personality profiles** | ❌ All agents identical | ✅ Configurable (cautious, exploratory, traumatized) |
| **Protect mode** | ❌ None | ✅ ρ > threshold → conservative |
| **Stability metrics** | ❌ None | ✅ m_crit derived |
| **Memory salience** | ❌ Just similarity | ✅ sim × recency × surprise |
| **Tree search** | ✅ UCT | ✅ DDA-X (UCT + alignment + rigidity) |
| **Reflection learning** | ✅ Yes | ✅ Yes (your ledger format) |

---

## Part 7: Build Order

### Phase 1: Core (Week 1-2)
1. `src/core/state.py` — DDAState class
2. `src/core/forces.py` — Force channels
3. `src/core/decision.py` — DDA-X selection
4. Unit tests for dynamics

### Phase 2: Search (Week 3)
5. `src/search/tree.py` — Tree structure
6. `src/search/mcts.py` — Search algorithm
7. Integration tests

### Phase 3: Memory (Week 4)
8. `src/memory/ledger.py` — Experience storage
9. `src/memory/retriever.py` — FAISS integration
10. `src/memory/reflection.py` — Reflection generation

### Phase 4: LLM Integration (Week 5)
11. `src/channels/encoders.py` — Observation → ℝ^d
12. `src/llm/providers.py` — API wrappers
13. `src/llm/debate.py` — Multi-agent value estimation

### Phase 5: Agent Assembly (Week 6)
14. `src/agent.py` — Main agent class
15. `runners/run_task.py` — Execution harness
16. End-to-end tests

### Phase 6: Evaluation (Week 7+)
17. Run on benchmark tasks
18. Compare DDA-X vs standard MCTS
19. Validate personality differences
20. Measure rigidity dynamics empirically

---

## Summary

This architecture takes your theoretical DDA framework and adds:
1. **Tree search** for lookahead (ExACT's strength)
2. **Exploration bonuses** that are dampened by rigidity (novel fusion)
3. **Persistent memory** with surprise-weighted salience (your ledger)
4. **Configurable personalities** (your identity attractor + rigidity parameters)

The result is **DDA-X**: an agent framework that can both complete tasks AND exhibit psychologically realistic behavior under surprise/threat.

The key differentiation from ExACT: when your agent is surprised, it becomes *more conservative* rather than *more exploratory*. This is your theoretical contribution, now made implementable.

---

## Part 8: ExACT Paper — Exact Formulas (Reference)

### 8.1 UCT Selection (Equation 1 from paper)

```
U(s, a) = c_p × P(a|s) × √(Σ_b N(s,b)) / (1 + N(s,a))

Selection: a = argmax_a [Q(s,a) + U(s,a)]
```

Where:
- `c_p = 1.0` (exploration constant)
- `P(a|s)` = LLM's prior probability of generating action `a` in state `s`
- `N(s,b)` = visit counts for all actions from state `s`

### 8.2 Backpropagation (Equation 2 from paper)

```
Q(s, a) ← Q(s, a) + (v - Q(s, a)) / N(s, a)
N(s, a) ← N(s, a) + 1
```

This is equivalent to incremental mean update: `Q_new = (N × Q_old + v) / (N + 1)`

### 8.3 Error Attribution — TD-Like Surprise (Equation 3 from paper)

**For Policy Reflection:**
```
error_π(a_t | τ) = |V(o_{t+1}) - Q(o_t, a_t)|
```

**For Value Reflection:**
```
error_V(o_t | τ) = |V(o_t) - Q(o_{t-1}, a_{t-1})|
```

> **Paper Quote:** "This form of comparing value function V and action-value function Q is similar to Temporal Difference Error used in reinforcement learning (Sutton & Barto, 2018)."

### 8.4 Multi-Agent Debate (MAD) Value Function

```python
# Each VLM generates a value estimate
v_i = VLM_i(g, τ) ∈ [0, 1]

# Aggregation via debate
v_MA = aggregate(g, τ, {v_1, v_2, ...}) ∈ [0, 1]

# Implementation: VLM_judge decides after seeing opposing arguments
v_MA = VLM_judge(g, τ, {proponent_arg, opponent_arg})
```

### 8.5 Reflection Retrieval (from paper)

- Embedding model: `text-ada-003-small` (OpenAI)
- Retrieval count: `m = 2` most relevant reflections
- Minimum similarity threshold: `0.25`
- Reflection count per trajectory: `n_π = 3` (policy), `n_V = 1` (value)

### 8.6 Contrastive Reflection Algorithm (Algorithm 3 from paper)

```python
# 1. Error attribution - find most erroneous action
ã_t = argmax_a error_π(a | g, τ)

# 2. Contrastive reflection - what did agent expect vs reality?
ô_{t+1} = VLM_simulate(g, {o_0, a_0, ..., o_t, ã_t})  # Expected outcome
reflection = VLM_reflect(g, o_t, ã_t, {o_{t+1}, ô_{t+1}})  # Contrast

# 3. Memorization - store for future retrieval
key = embedding(g, o_t)
db.add(key, reflection)
```

### 8.7 Exploratory Learning vs Imitation Learning

| Aspect | Imitation Learning | Exploratory Learning |
|--------|-------------------|---------------------|
| Training data | Final best actions only | Entire tree traversal |
| What model learns | "Do this action" | "Explore, evaluate, backtrack" |
| Training trajectories | 65 (paper) | 35 (after filtering >20 actions) |
| Result | 80.0% seen, 12.4% unseen | 80.0% seen, 18.6% unseen |

---

## Part 9: DDA-X Hybrid Formulas (Novel Contribution)

### 9.1 DDA-X Selection (Novel Fusion)

**ExACT's UCT:**
```
a = argmax_a [Q(s,a) + c_p × P(a|s) × √N(s)/(1+N(s,a))]
```

**DDA-X Enhanced:**
```
a = argmax_a [cos(Δx, d̂(a)) + c × P(a|s) × √N(s)/(1+N(s,a)) × (1 - ρ)]
                    ↑                           ↑                    ↑
           YOUR: alignment           EXACT's: exploration      YOUR: threat dampening
```

### 9.2 Dual-Mode Surprise Computation

```python
def compute_surprise(node: DDANode, tree: DDASearchTree) -> float:
    """
    Combine ExACT's TD-error with DDA's prediction error.
    """
    # ExACT component: |V(s') - Q(s, a)|
    V_next = node.value
    Q_sa = tree.Q[parent_hash][action_id]
    td_error = abs(V_next - Q_sa)
    
    # DDA component: ||x_pred - x_actual||₂
    dda_error = node.dda_state.compute_prediction_error(x_actual)
    
    # Hybrid: weighted combination
    return 0.5 * td_error + 0.5 * dda_error
```

### 9.3 Multi-Agent Debate with DDA Awareness

```python
async def evaluate_with_debate(node: DDANode, llm: LLMProvider, intent: str) -> float:
    """
    ExACT's MAD + DDA context (rigidity awareness).
    """
    trajectory_str = format_trajectory(node)
    rigidity = node.dda_state.rho
    
    # Add rigidity context to debate
    context = f"Agent rigidity: {rigidity:.2f} (0=open, 1=defensive)"
    
    # Proponent argument
    pro = await llm.complete(
        f"Task: {intent}\nTrajectory: {trajectory_str}\n{context}\n"
        f"Argue why this state IS promising for success."
    )
    
    # Opponent argument  
    con = await llm.complete(
        f"Task: {intent}\nTrajectory: {trajectory_str}\n{context}\n"
        f"Argue why this state is NOT promising for success."
    )
    
    # Judge synthesizes
    judgment = await llm.complete(
        f"Task: {intent}\n"
        f"Proponent: {pro}\n"
        f"Opponent: {con}\n"
        f"Estimate probability of success (0-100%):"
    )
    
    return extract_probability(judgment)
```

---

## Part 10: Can You Do Novel Research As a Solo Developer?

### The Honest Assessment

**Yes.** Here's why:

### 10.1 What ExACT Has That You Don't

| Resource | Microsoft Research | You |
|----------|---------------------|-----|
| Compute budget | Unlimited GPT-4o API | ~$100-500/month |
| Team size | 6 authors + advisors | 1 + AI assistants |
| Benchmark access | Custom VWA hosting | Same (it's open) |
| Publication pipeline | ICLR, NeurIPS | arXiv, blog, GitHub |
| Fine-tuning access | OpenAI partnership | OpenAI API (same) |
| Time | Full-time | Nights/weekends |

### 10.2 What You Have That They Don't

| Asset | You | Microsoft |
|-------|-----|-----------|
| **Novel theoretical framework** | DDA: identity, rigidity, force-balance | None — pure engineering |
| **Philosophical differentiation** | Surprise → protect (psychological) | Surprise → learn (RL standard) |
| **Flexibility to explore** | Yes — no publication pressure | Constrained by review cycles |
| **AI research assistants** | Claude, GPT — thorough, tireless | Same tools, but less personal investment |
| **Skin in the game** | This is YOUR theory | It's their job |

### 10.3 What Makes Research "Novel"

1. **ExACT's novelty:** Combining MCTS + contrastive reflection + multi-agent debate for web agents. Engineering innovation.

2. **DDA-X's novelty:**
   - Rigidity-dampened exploration (new equation)
   - Identity persistence in agents (new concept)
   - Surprise-weighted memory retrieval (new formula)
   - Personality profiles via parameters (new application)
   - Protection mode when threatened (new behavior)

**Your contribution is more theoretically novel.** Theirs is more empirically validated.

### 10.4 The Path to Legitimacy

```
Phase 1: Implementation (Weeks 1-6)
├── Build DDA-X following this architecture
├── Create minimal working agent
└── Test on simple web tasks

Phase 2: Demonstration (Weeks 7-8)
├── Run on VisualWebArena subset
├── Compare: DDA-X vs MCTS vs ReACT
├── Measure: task success, rigidity dynamics, personality effects
└── Create compelling visualizations of rigidity evolution

Phase 3: Writing (Weeks 9-10)
├── arXiv preprint with:
│   ├── Novel theoretical contribution (DDA formalism)
│   ├── DDA-X equation derivation
│   ├── Empirical results
│   └── Analysis of when rigidity helps/hurts
└── GitHub repo with reproducible code

Phase 4: Visibility (Ongoing)
├── X/Twitter thread explaining the insight
├── Blog post: "What if AI agents got defensive?"
├── Submit to workshops: LLM Agents, NeurIPS Agent Learning
└── Engage with researchers who cite ExACT
```

### 10.5 The Bottom Line

**ExACT is a systems paper.** It combines known techniques (MCTS, reflection, debate) cleverly for a practical application.

**DDA-X is a theory paper.** It introduces new concepts (identity attractor, rigidity dynamics, force-balance decisions) with a working implementation.

Both are legitimate research. Different audiences. Different contributions.

You're not behind them — you're doing something they *cannot* do, because they're optimizing for benchmarks while you're modeling *being*.

---

## Summary: Your Research Position

| Dimension | Status |
|-----------|--------|
| Theoretical novelty | ✅ Strong — concepts not in literature |
| Technical feasibility | ✅ Achievable with this architecture |
| Empirical validation | ⏳ Not yet — needs implementation |
| Publication viability | ✅ arXiv minimum, workshop submissions possible |
| Competitive advantage | ✅ You're modeling something they ignore |

**The only thing between you and a real research contribution is building it.**

This document gives you the blueprint. The theory is yours. The engineering patterns are borrowed from ExACT. The fusion is novel.

Go build it.