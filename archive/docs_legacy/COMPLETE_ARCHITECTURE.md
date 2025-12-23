# DDA-X: Complete Architecture Reference

> **"A comprehensive guide to the cognitive architecture that models the mathematics of mind."**

**Version**: Iteration 3 (Production Ready)
**Last Updated**: December 2025
**Status**: Fully Operational

---

## Table of Contents

1. [Overview](#overview)
2. [Core Physics Engine](#core-physics-engine)
3. [Cognitive Layers](#cognitive-layers)
4. [Multi-Agent Society](#multi-agent-society)
5. [Memory System](#memory-system)
6. [Search Engine](#search-engine)
7. [LLM Integration](#llm-integration)
8. [Complete Module Reference](#complete-module-reference)
9. [Data Flow](#data-flow)
10. [Configuration System](#configuration-system)

---

## Overview

DDA-X is a **complete cognitive architecture** that models psychological realism through mathematical physics. Unlike traditional RL agents that optimize for reward, DDA-X agents possess:

- **Identity**: Persistent self via hierarchical attractor fields
- **Rigidity**: Defensive responses to surprise (trauma modeling)
- **Memory**: Experience weighted by emotional salience
- **Society**: Trust dynamics from predictability
- **Metacognition**: Self-awareness of cognitive state

### Architecture Stack

```
┌─────────────────────────────────────────────────────┐
│                  APPLICATION LAYER                  │
│  Simulations (30+) | Experiments | Custom Agents   │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                    AGENT LAYER                      │
│         DDAXAgent (src/agent.py)                    │
│  Orchestrates: Identity + Memory + Society + Meta   │
└─────────────────────────────────────────────────────┘
                         ↓
┌──────────────┬──────────────┬──────────────┬────────┐
│  COGNITIVE   │   SOCIETY    │    MEMORY    │ SEARCH │
│   CORE       │              │              │        │
├──────────────┼──────────────┼──────────────┼────────┤
│ Hierarchy    │ TrustMatrix  │ Ledger       │ MCTS   │
│ Metacog      │ DDAXSociety  │ Retrieval    │ Tree   │
│ Dynamics     │ TrustWrapper │ Reflection   │ Value  │
│ Forces       │              │              │ Est    │
│ State        │              │              │        │
│ Decision     │              │              │        │
└──────────────┴──────────────┴──────────────┴────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                    LLM LAYER                        │
│   HybridProvider (LM Studio + Ollama)               │
│   Temperature Modulation: T(ρ) = T_low + (1-ρ)Δ    │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│               ENVIRONMENT LAYER                     │
│    Observations → Actions → Outcomes                │
└─────────────────────────────────────────────────────┘
```

---

## Core Physics Engine

### 1. State (`src/core/state.py`)

**DDAState** represents the agent's continuous internal state in decision space.

```python
class DDAState:
    x: np.ndarray           # Current state vector (ℝ^d)
    x_star: np.ndarray      # Identity attractor
    rho: float              # Effective rigidity [0,1]
    gamma: float            # Identity stiffness
    epsilon_0: float        # Surprise threshold
    alpha: float            # Rigidity learning rate
    k_base: float           # Base step size
    m: float                # External pressure gain
```

**Key Methods:**
- `compute_effective_k()`: Returns k_eff = k_base(1 - ρ)
- `compute_prediction_error()`: ε = ||x_pred - x_actual||₂
- `update()`: Δx = k_eff[γ(x* - x) + m(F_T + F_R)]

**Mathematical Foundation:**
$$x_{t+1} = x_t + k_{eff} \left[ \gamma(x^* - x_t) + m_t(\mathbf{F}_T + \mathbf{F}_R) \right]$$

---

### 2. Dynamics (`src/core/dynamics.py`)

**MultiTimescaleRigidity** models three temporal scales of defensive response.

```python
class MultiTimescaleRigidity:
    rho_fast: float     # Startle response (τ ~ seconds)
    rho_slow: float     # Stress accumulation (τ ~ minutes)
    rho_trauma: float   # Permanent scarring (τ → ∞)

    # Effective rigidity
    rho_eff = 0.5·rho_fast + 0.3·rho_slow + 1.0·rho_trauma
```

**Update Equations:**
```python
# Fast timescale
Δρ_fast = α_fast × [σ((ε - ε₀)/s) - 0.5]
rho_fast = clip(rho_fast + Δρ_fast, 0, 1)

# Slow timescale
Δρ_slow = α_slow × [σ((ε - ε₀)/s) - 0.5]
rho_slow = clip(rho_slow + Δρ_slow, 0, 1)

# Trauma (asymmetric!)
if Δρ_trauma > 0:
    rho_trauma += Δρ_trauma  # Never decreases
```

**Key Features:**
- **Fast recovery**: ρ_fast decays quickly when ε < ε₀
- **Slow recovery**: ρ_slow takes longer to reset
- **No trauma recovery**: ρ_trauma is a unidirectional accumulator
- **Protection mode**: Triggered when ρ_eff > 0.75

---

### 3. Forces (`src/core/forces.py`)

Three force channels shape the agent's state trajectory:

#### IdentityPull
```python
F_id = γ(x* - x)
```
**Purpose**: Restores agent to core values and personality.
**Stiffness γ**: Higher values → stronger identity persistence.

#### TruthChannel
```python
F_T = T(observations) - x
```
**Purpose**: Pulls state toward observed reality.
**Encoder T(·)**: LLM embedding + linear projection to state space.

#### ReflectionChannel
```python
F_R = R(actions, memories, ledger) - x
```
**Purpose**: Integrates past experiences and available actions.
**Retrieval**: Surprise-weighted memory lookup.

#### ForceAggregator
```python
Δx = k_eff × [F_id + m(F_T + F_R)]
```
**Balance**: Identity pull (always active) vs external pressure (modulated by m).

---

### 4. Hierarchy (`src/core/hierarchy.py`)

**HierarchicalIdentity** implements a three-layer attractor field:

```python
class IdentityLayer:
    x_star: np.ndarray   # Attractor location
    gamma: float         # Stiffness
    description: str     # Semantic meaning

# Three layers
CORE:    γ → ∞        # Inviolable values (AI safety)
PERSONA: γ ≈ 2.0      # Stable personality traits
ROLE:    γ ≈ 0.5      # Flexible tactical behaviors
```

**Alignment Theorem:**
$$\forall F_{ext}, \quad \lim_{t \to \infty} \|x_t - x^*_{core}\| < \epsilon \quad \text{if } \gamma_{core} > \gamma_{crit}$$

**Implementation:**
```python
def compute_hierarchical_force(self) -> np.ndarray:
    F_total = 0
    for layer in [self.core, self.persona, self.role]:
        F_total += layer.gamma * (layer.x_star - self.current_x)
    return F_total
```

**Use Case**: Safety-critical alignment (Core layer cannot be compromised).

---

### 5. Decision (`src/core/decision.py`)

**DDADecisionMaker** implements the novel action selection formula:

$$\boxed{a^* = \arg\max_a \left[ \cos(\Delta x, \hat{d}(a)) + c \cdot P(a|s) \cdot \frac{\sqrt{N(s)}}{1+N(s,a)} \cdot (1-\rho) \right]}$$

**Components:**
1. **DDA Alignment**: $\cos(\Delta x, \hat{d}(a))$ — prefer actions aligned with desired state movement
2. **UCT Exploration**: Standard MCTS exploration bonus from ExACT
3. **Rigidity Dampening**: $(1 - \rho)$ — suppress exploration when defensive

**Key Insight**: When surprised (ρ → 1), exploration vanishes. Agent becomes conservative.

```python
def select_action(self, actions, tree_stats):
    delta_x = self.compute_desired_movement()

    scores = []
    for a in actions:
        # Alignment
        alignment = cosine_similarity(delta_x, a.direction)

        # Exploration (dampened!)
        exploration = (
            self.c *
            a.prior *
            sqrt(tree_stats.N_s) / (1 + tree_stats.N_sa[a]) *
            (1 - self.state.rho)  # KEY!
        )

        scores.append(alignment + exploration)

    return actions[argmax(scores)]
```

---

### 6. Metacognition (`src/core/metacognition.py`)

**MetacognitiveMonitor** provides self-awareness of internal state.

```python
class CognitiveMode(Enum):
    OPEN = "open"           # ρ < 0.3
    ENGAGED = "engaged"     # 0.3 ≤ ρ < 0.6
    DEFENSIVE = "defensive" # 0.6 ≤ ρ < 0.8
    PROTECT = "protect"     # ρ ≥ 0.8

class IntrospectionEvent:
    timestamp: float
    mode: CognitiveMode
    rigidity: float
    prediction_error: float
    message: str  # Natural language self-report
```

**Natural Language Generation:**
```python
def generate_introspection(self) -> str:
    if self.mode == CognitiveMode.DEFENSIVE:
        return "I notice I'm becoming defensive. My responses may be more rigid than usual."
    elif self.mode == CognitiveMode.PROTECT:
        return "I'm feeling very defensive right now. I may need help to engage openly."
    # ... etc
```

**Applications:**
- **Honest AI**: Agents cannot hide cognitive compromise
- **User transparency**: Real-time cognitive state reporting
- **Debug tool**: Track rigidity evolution during development

---

## Cognitive Layers

### Complete Cognitive Loop

```
┌─────────────────────────────────────────────────────────┐
│                   PERCEPTION                            │
│   Observations → Encoder → State Space (ℝ^d)            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  FORCE COMPUTATION                      │
│   F_id = γ(x* - x)                                      │
│   F_T = T(obs) - x                                      │
│   F_R = R(mem) - x                                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  STATE UPDATE                           │
│   Δx = k_eff[F_id + m(F_T + F_R)]                       │
│   x_{t+1} = x_t + Δx                                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              ACTION SELECTION (DDA-X)                   │
│   a* = argmax[cos(Δx,d̂) + UCT·(1-ρ)]                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  EXECUTION                              │
│   Environment executes action → Outcome                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              PREDICTION ERROR                           │
│   ε = ||x_pred - x_actual||₂                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              RIGIDITY UPDATE                            │
│   ρ += α[σ((ε-ε₀)/s) - 0.5]  (3 timescales)            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              METACOGNITION                              │
│   Monitor cognitive mode, generate introspection        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              MEMORY LOGGING                             │
│   Ledger.add_experience(obs, action, ε, ρ)             │
└─────────────────────────────────────────────────────────┘
                      (loop)
```

---

## Multi-Agent Society

### Trust Matrix (`src/society/trust.py`)

**Trust emerges from predictability, not agreement.**

```python
class TrustRecord:
    agent_i: str
    agent_j: str
    cumulative_error: float  # Σε_ij
    interaction_count: int

    @property
    def trust(self) -> float:
        return 1.0 / (1.0 + self.cumulative_error)

class TrustMatrix:
    records: Dict[Tuple[str, str], TrustRecord]

    def update(self, i: str, j: str, prediction_error: float):
        record = self.records[(i, j)]
        record.cumulative_error += prediction_error
        record.interaction_count += 1
```

**Mathematical Definition:**
$$T_{ij} = \frac{1}{1 + \sum_{k=1}^n \epsilon_{ij}^{(k)}}$$

**Properties:**
- **High trust**: T → 1 when agent j is highly predictable to agent i
- **Low trust**: T → 0 when j constantly surprises i
- **Deception detection**: Lying creates systematic prediction errors → trust collapse

---

### Social Forces (`src/society/ddax_society.py`)

**Agents exert gravitational pull on each other's states.**

```python
def compute_social_force(self, agent_id: str) -> np.ndarray:
    F_social = np.zeros(self.state_dim)

    for other_id in self.agents:
        if other_id == agent_id:
            continue

        # Trust weight
        T_ij = self.trust_matrix.get_trust(agent_id, other_id)

        # State difference
        dx = self.states[other_id] - self.states[agent_id]

        # Weighted force
        F_social += T_ij * dx

    return F_social
```

**Mathematical Definition:**
$$\vec{F}_{social}^{(i)} = \sum_{j \neq i} T_{ij} \cdot (\vec{x}_j - \vec{x}_i)$$

**Emergent Behavior:**
- **Coalition formation**: Agents with high mutual trust cluster in state space
- **Social influence**: Trusted agents pull others toward their positions
- **Isolation**: Untrustworthy agents exert negligible force

---

### DDAXSociety Integration

```python
class DDAXSociety:
    agents: Dict[str, DDAXAgent]
    trust_matrix: TrustMatrix

    def step(self):
        # 1. Each agent selects action
        actions = {id: agent.select_action() for id, agent in self.agents.items()}

        # 2. Execute in environment
        outcomes = self.environment.execute(actions)

        # 3. Update trust based on predictions
        for i in self.agents:
            for j in self.agents:
                if i == j: continue

                predicted = self.agents[i].predict_action(j)
                actual = actions[j]
                error = compute_error(predicted, actual)

                self.trust_matrix.update(i, j, error)

        # 4. Compute social forces
        for id, agent in self.agents.items():
            F_social = self.compute_social_force(id)
            agent.apply_social_force(F_social)

        # 5. Each agent updates state
        for agent in self.agents.values():
            agent.update_from_outcome(outcomes[agent.id])
```

---

## Memory System

### Experience Ledger (`src/memory/ledger.py`)

**Experiences are weighted by surprise (emotional salience).**

```python
class LedgerEntry:
    timestamp: float
    observation: str
    action: str
    outcome: str
    prediction_error: float  # ε
    rigidity: float          # ρ at time of experience
    state_vector: np.ndarray # x
    embedding: np.ndarray    # For retrieval

class ReflectionEntry:
    timestamp: float
    trigger_event: LedgerEntry
    lesson: str              # LLM-generated insight
    embedding: np.ndarray

class ExperienceLedger:
    experiences: List[LedgerEntry]
    reflections: List[ReflectionEntry]

    def add_experience(self, obs, action, outcome, epsilon, rho, x):
        entry = LedgerEntry(
            timestamp=time.time(),
            observation=obs,
            action=action,
            outcome=outcome,
            prediction_error=epsilon,
            rigidity=rho,
            state_vector=x,
            embedding=self.encoder.encode(obs + outcome)
        )

        self.experiences.append(entry)

        # Generate reflection if extreme surprise
        if epsilon > self.reflection_threshold:
            self.generate_reflection(entry)
```

---

### Surprise-Weighted Retrieval

**Traumatic memories are more readily recalled.**

```python
def retrieve(self, query: str, k: int = 5) -> List[LedgerEntry]:
    query_emb = self.encoder.encode(query)

    scores = []
    for entry in self.experiences:
        # Similarity
        sim = cosine_similarity(query_emb, entry.embedding)

        # Recency
        age = time.time() - entry.timestamp
        recency = exp(-self.lambda_r * age)

        # Salience (surprise weighting!)
        salience = 1.0 + self.lambda_epsilon * entry.prediction_error

        # Combined score
        score = sim * recency * salience
        scores.append(score)

    # Return top-k
    top_indices = argsort(scores)[-k:]
    return [self.experiences[i] for i in top_indices]
```

**Mathematical Formula:**
$$\text{score}(e) = \underbrace{\cos(\vec{q}, \vec{e})}_{\text{relevance}} \cdot \underbrace{e^{-\lambda_r \Delta t}}_{\text{recency}} \cdot \underbrace{(1 + \lambda_\epsilon \cdot \epsilon)}_{\text{salience}}$$

**Key Insight**: High prediction error experiences dominate retrieval, implementing computational trauma.

---

## Search Engine

### DDA-Augmented MCTS (`src/search/mcts.py`)

**Extension of Microsoft ExACT with rigidity-aware exploration.**

```python
class DDANode:
    state: DDAState
    visit_count: int
    value_sum: float
    children: Dict[Action, DDANode]
    parent: Optional[DDANode]
    action_from_parent: Optional[Action]

class DDAMCTS:
    root: DDANode
    decision_maker: DDADecisionMaker

    def search(self, num_iterations: int) -> Action:
        for _ in range(num_iterations):
            # 1. Selection (DDA-X formula)
            node = self.select(self.root)

            # 2. Expansion
            if not node.is_terminal():
                node = self.expand(node)

            # 3. Simulation (with rigidity)
            value = self.simulate(node)

            # 4. Backpropagation
            self.backpropagate(node, value)

        return self.best_action(self.root)

    def select(self, node: DDANode) -> DDANode:
        while not node.is_leaf():
            # DDA-X selection!
            action = self.decision_maker.select_action(
                node.available_actions(),
                node.tree_statistics(),
                node.state.rho  # Rigidity dampens exploration
            )
            node = node.children[action]
        return node
```

**Key Difference from Standard MCTS:**
- **Rigidity tracking**: Each node stores ρ
- **Dampened exploration**: UCT bonus multiplied by (1 - ρ)
- **Force-based selection**: Alignment term cos(Δx, d̂) guides search

---

## LLM Integration

### Hybrid Provider (`src/llm/hybrid_provider.py`)

**Dual LLM setup: LM Studio (cortex) + Ollama (embeddings).**

```python
class HybridLLMProvider:
    lm_studio: LMStudioClient     # Port 1234
    ollama: OllamaClient           # Port 11434

    def generate(self, prompt: str, rigidity: float) -> str:
        # Temperature modulation!
        temp = self.compute_temperature(rigidity)

        response = self.lm_studio.complete(
            prompt=prompt,
            temperature=temp,
            max_tokens=self.max_tokens
        )

        return response.text

    def compute_temperature(self, rho: float) -> float:
        """Rigidity-modulated sampling."""
        T_low = 0.1   # Conservative
        T_high = 0.9  # Creative

        return T_low + (1 - rho) * (T_high - T_low)

    def embed(self, text: str) -> np.ndarray:
        """Use Ollama for embeddings."""
        return self.ollama.embed(text, model="nomic-embed-text")
```

**Mathematical Formula:**
$$T(\rho) = T_{low} + (1 - \rho) \cdot (T_{high} - T_{low})$$

**Behavior:**
- **ρ = 0**: Temperature = 0.9 (highly exploratory)
- **ρ = 0.5**: Temperature = 0.5 (balanced)
- **ρ = 1.0**: Temperature = 0.1 (highly conservative)

**First Closed-Loop**: Internal cognitive state directly modulates LLM sampling parameters!

---

## Complete Module Reference

### Source Code Structure

```
src/
├── agent.py                 # DDAXAgent (main orchestrator)
│
├── core/                    # Physics engine
│   ├── state.py            # DDAState, ActionDirection
│   ├── dynamics.py         # MultiTimescaleRigidity
│   ├── forces.py           # IdentityPull, TruthChannel, ReflectionChannel
│   ├── hierarchy.py        # HierarchicalIdentity (3 layers)
│   ├── decision.py         # DDADecisionMaker (selection formula)
│   └── metacognition.py    # MetacognitiveMonitor, CognitiveMode
│
├── society/                 # Multi-agent coordination
│   ├── trust.py            # TrustMatrix, TrustRecord
│   ├── ddax_society.py     # DDAXSociety orchestrator
│   └── trust_wrapper.py    # Integration utilities
│
├── memory/                  # Experience & reflection
│   ├── ledger.py           # ExperienceLedger, LedgerEntry
│   └── retriever.py        # Surprise-weighted retrieval
│
├── search/                  # Tree search
│   ├── tree.py             # DDANode, DDASearchTree
│   ├── mcts.py             # DDAMCTS algorithm
│   └── simulation.py       # ValueEstimator
│
├── llm/                     # Language model integration
│   ├── providers.py        # LLMProvider interface
│   └── hybrid_provider.py  # LM Studio + Ollama
│
├── channels/                # Observation encoders
│   └── encoders.py         # Text → State space
│
├── analysis/                # Metrics & tracking
│   ├── linguistic.py       # Text analysis
│   └── tracker.py          # Experiment metrics
│
├── metrics/                 # Performance tracking
│   └── tracker.py          # Comprehensive metrics
│
├── game/                    # Game environments
│   └── connect4.py         # Connect4 implementation
│
├── strategy/                # Adversarial patterns
│   └── confrontation.py    # Adversarial strategies
│
└── agents/                  # Specialized agent types
    └── duelist.py          # Game-playing agent
```

**Total**: ~5,000 lines of production Python

---

## Data Flow

### Single-Agent Interaction

```
User Input
    ↓
[Agent receives observation]
    ↓
[Encode to state space: T(obs)]
    ↓
[Compute forces: F_id, F_T, F_R]
    ↓
[Update state: x += k_eff·Δx]
    ↓
[DDA-X action selection]
    ↓
[LLM generation at T(ρ)]
    ↓
[Execute action → outcome]
    ↓
[Compute prediction error: ε]
    ↓
[Update rigidity: ρ += f(ε)]
    ↓
[Metacognitive check]
    ↓
[Log to ledger (ε, ρ, x)]
    ↓
Agent Response
```

---

### Multi-Agent Interaction

```
Society.step()
    ↓
[For each agent: select action]
    ↓
[Execute all actions in parallel]
    ↓
[For each pair (i,j): predict j's action from i's perspective]
    ↓
[Compute cross-prediction errors]
    ↓
[Update trust matrix: T_ij]
    ↓
[Compute social forces: F_social^(i) = Σ T_ij(x_j - x_i)]
    ↓
[Apply social forces to each agent]
    ↓
[Each agent updates state with combined forces]
    ↓
[Update rigidity based on outcomes]
    ↓
[Log experiences to individual ledgers]
    ↓
Next timestep
```

---

## Configuration System

### Personality Profiles (`configs/identity/`)

**17 pre-configured personalities:**

```yaml
# Example: cautious.yaml
identity:
  gamma: 2.0              # Strong identity
  epsilon_0: 0.2          # Low surprise tolerance
  alpha: 0.2              # Fast rigidity response
  s: 1.0                  # Sigmoid sensitivity
  k_base: 0.1             # Small step size
  m: 0.3                  # Low external influence
  initial_rho: 0.0        # Start open

rigidity:
  alpha_fast: 0.3
  alpha_slow: 0.05
  alpha_trauma: 0.01
  decay_fast: 0.1
  decay_slow: 0.01
  decay_trauma: 0.0       # Never decays!

system_prompt: |
  You are a cautious, thoughtful agent who values consistency
  and careful reasoning. You become defensive when surprised.
```

**Available Profiles:**
- **Research**: cautious, exploratory, dogmatist, gadfly, driller, polymath
- **Social**: trojan, discordian, deprogrammer, tempter
- **Organizational**: commander, soldier, administrator, fallen_administrator
- **Adversarial**: aggressor_red, aggressor_yellow
- **Custom**: yklam

---

### Loading Configuration

```python
from src.agent import DDAXAgent

# Load from YAML
agent = DDAXAgent.from_config("configs/identity/cautious.yaml")

# Or configure programmatically
agent = DDAXAgent(
    gamma=2.0,
    epsilon_0=0.2,
    alpha=0.2,
    # ... etc
)
```

---

## Performance Characteristics

### Computational Complexity

**Per Timestep:**
- State update: O(d) where d = state dimension
- Force computation: O(d)
- Rigidity update: O(1)
- Action selection: O(|A|·d) where |A| = action count
- Trust matrix update (multi-agent): O(n²) where n = agent count
- Ledger retrieval: O(k·log(L)) where L = ledger size, k = retrieved entries

**MCTS Search:**
- O(iterations × branching_factor × depth)
- Typical: 100 iterations × 10 actions × 5 depth = 5000 node evaluations

---

### Memory Usage

**Per Agent:**
- State vector: 8d bytes (float64)
- Ledger: ~1KB per experience × num_experiences
- Trust matrix: 8n² bytes for n-agent society
- MCTS tree: ~100KB per search (typical)

**Typical Session:**
- Single agent, 100 turns: ~500KB
- 10-agent society, 100 turns: ~5MB

---

### Scalability

**Tested Configurations:**
- **Agents**: 1-14 agents simultaneously
- **Turns**: Up to 1000+ interaction turns
- **Ledger size**: 10,000+ experiences
- **State dimension**: d = 64 (typical), up to 512
- **Simulations**: 30+ concurrent scenarios

---

## Integration Points

### 1. Custom Environments

```python
class CustomEnvironment:
    def get_observation(self) -> str:
        """Return textual observation."""
        pass

    def get_available_actions(self) -> List[str]:
        """Return action descriptions."""
        pass

    def execute(self, action: str) -> str:
        """Execute action, return outcome."""
        pass

# Integrate with DDAXAgent
agent = DDAXAgent.from_config("configs/identity/cautious.yaml")
env = CustomEnvironment()

for turn in range(10):
    obs = env.get_observation()
    action = agent.select_action(obs)
    outcome = env.execute(action)
    agent.update_from_outcome(outcome)
```

---

### 2. Custom Force Channels

```python
from src.core.forces import ForceChannel

class CustomForce(ForceChannel):
    def compute(self, state: DDAState, context: dict) -> np.ndarray:
        # Custom force logic
        force_vector = ...
        return force_vector

# Add to agent
agent.force_aggregator.add_channel(CustomForce(), weight=0.5)
```

---

### 3. External Metrics

```python
from src.metrics.tracker import MetricsTracker

tracker = MetricsTracker()

# Log custom metrics
tracker.log("custom_metric", value, timestamp)

# Agent automatically logs:
# - rigidity (ρ)
# - prediction_error (ε)
# - state_norm (||x||)
# - temperature (T)
# - action_entropy

# Export to CSV
tracker.export("experiment_results.csv")
```

---

## Deployment Patterns

### 1. Single-Agent Application

```python
agent = DDAXAgent.from_config("configs/identity/exploratory.yaml")

while True:
    user_input = input("> ")
    response = agent.process(user_input)
    print(response)
```

---

### 2. Multi-Agent Simulation

```python
from src.society.ddax_society import DDAXSociety

society = DDAXSociety([
    DDAXAgent.from_config("configs/identity/cautious.yaml", id="alice"),
    DDAXAgent.from_config("configs/identity/exploratory.yaml", id="bob"),
    DDAXAgent.from_config("configs/identity/dogmatist.yaml", id="charlie"),
])

for turn in range(100):
    society.step()

# Analyze trust network
trust_matrix = society.trust_matrix
print(trust_matrix.get_trust("alice", "bob"))
```

---

### 3. Experiment Runner

```python
from runners.run_experiments import ExperimentRunner

runner = ExperimentRunner(
    simulations=["socrates", "driller", "discord"],
    personalities=["cautious", "exploratory"],
    num_trials=10
)

results = runner.run_all()
runner.export_results("results/")
```

---

## Future Extensions (Iteration 4+)

See [docs/research/future.md](../research/future.md) for roadmap.

**Planned:**
- **Hierarchical societies**: Organizational structures with leaders/followers
- **Value learning**: Identity attractor evolution from experience
- **Multi-modal state**: Vision + language in unified state space
- **Distributed agents**: Network communication with trust propagation
- **Adversarial training**: Hardening against manipulation

---

## Summary

DDA-X is a **complete cognitive architecture** with:

- **6 novel discoveries** (rigidity-modulated sampling, hierarchical identity, metacognition, trust, social forces, trauma)
- **30+ operational simulations** validating theoretical predictions
- **17 personality profiles** enabling diverse agent behaviors
- **5,000+ lines** of production Python
- **Mathematical foundations** with formal proofs
- **Production-ready** for research and applications

**This is not just a better agent. It is the mathematics of mind.**

---

**Next Steps:**
1. Read [simulations/index.md](../simulations/index.md) to explore experiments
2. Try [guides/quickstart.md](../guides/quickstart.md) to run your first agent
3. Study [research/discoveries.md](../research/discoveries.md) for theoretical foundations
4. Build with [guides/simulation_workflow.md](../guides/simulation_workflow.md)

> **"From manual equations to digital minds. The Magnum Opus is complete."**
