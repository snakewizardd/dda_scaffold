# DDA-X Components Reference

> **Detailed API documentation for all DDA-X modules**

**Version**: Iteration 3
**Last Updated**: December 2025

---

## Table of Contents

1. [Core Components](#core-components)
2. [Society Components](#society-components)
3. [Memory Components](#memory-components)
4. [Search Components](#search-components)
5. [LLM Components](#llm-components)
6. [Analysis Components](#analysis-components)
7. [Usage Examples](#usage-examples)

---

## Core Components

### `src/core/state.py`

#### DDAState

**Purpose**: Represents the agent's continuous internal state in decision space.

**Attributes:**
```python
x: np.ndarray              # Current state vector (ℝ^d)
x_star: np.ndarray         # Identity attractor
rho: float                 # Effective rigidity [0,1]
gamma: float               # Identity stiffness (≥ 0)
epsilon_0: float           # Surprise threshold
alpha: float               # Rigidity learning rate
s: float                   # Sigmoid sensitivity
k_base: float              # Base step size
m: float                   # External pressure gain
prediction_history: List   # Recent prediction errors
```

**Methods:**

```python
def compute_effective_k() -> float:
    """
    Returns effective step size dampened by rigidity.

    Returns:
        k_eff = k_base × (1 - ρ)
    """

def compute_prediction_error(x_pred: np.ndarray, x_actual: np.ndarray) -> float:
    """
    Computes L2 norm of prediction error.

    Args:
        x_pred: Predicted next state
        x_actual: Actual observed state

    Returns:
        ε = ||x_pred - x_actual||₂
    """

def update(delta_x: np.ndarray) -> None:
    """
    Updates state vector.

    Args:
        delta_x: Desired movement vector

    Updates:
        self.x = self.x + k_eff × delta_x
    """

def update_rigidity(epsilon: float) -> float:
    """
    Updates rigidity based on prediction error.

    Args:
        epsilon: Prediction error magnitude

    Returns:
        New rigidity value (clipped to [0,1])

    Formula:
        Δρ = α × [σ((ε - ε₀)/s) - 0.5]
        ρ_new = clip(ρ + Δρ, 0, 1)
    """
```

**Usage Example:**
```python
from src.core.state import DDAState

state = DDAState(
    x=np.zeros(64),
    x_star=np.random.randn(64),
    gamma=2.0,
    epsilon_0=0.3,
    alpha=0.1,
    k_base=0.1,
    m=0.5
)

# Compute prediction error
epsilon = state.compute_prediction_error(x_pred, x_actual)

# Update rigidity
new_rho = state.update_rigidity(epsilon)

# Update state
state.update(delta_x)
```

---

#### ActionDirection

**Purpose**: Associates discrete actions with continuous state-space directions.

**Attributes:**
```python
action_id: str             # Action identifier
direction: np.ndarray      # Unit vector in state space
prior: float               # LLM prior probability P(a|s)
description: str           # Natural language description
```

**Methods:**
```python
@staticmethod
def from_text(text: str, encoder: Callable) -> ActionDirection:
    """
    Creates ActionDirection from text description.

    Args:
        text: Natural language action description
        encoder: Function mapping text → ℝ^d

    Returns:
        ActionDirection with normalized direction vector
    """

def cosine_similarity(other: np.ndarray) -> float:
    """
    Computes cosine similarity between action direction and vector.

    Args:
        other: Comparison vector

    Returns:
        cos(θ) = (d̂ · other) / (||d̂|| × ||other||)
    """
```

---

### `src/core/dynamics.py`

#### MultiTimescaleRigidity

**Purpose**: Models three temporal scales of defensive response.

**Attributes:**
```python
rho_fast: float            # Startle response (τ ~ seconds)
rho_slow: float            # Stress accumulation (τ ~ minutes)
rho_trauma: float          # Permanent scarring (τ → ∞)

alpha_fast: float          # Fast learning rate
alpha_slow: float          # Slow learning rate
alpha_trauma: float        # Trauma accumulation rate

decay_fast: float          # Fast decay rate
decay_slow: float          # Slow decay rate
decay_trauma: float        # Trauma decay rate (always 0!)
```

**Methods:**

```python
def update(epsilon: float, epsilon_0: float, s: float) -> None:
    """
    Updates all three rigidity timescales.

    Args:
        epsilon: Prediction error
        epsilon_0: Surprise threshold
        s: Sigmoid sensitivity

    Updates:
        ρ_fast, ρ_slow, ρ_trauma via sigmoid learning

    Algorithm:
        delta = σ((ε - ε₀)/s) - 0.5

        # Fast timescale
        Δρ_fast = α_fast × delta - decay_fast × ρ_fast
        ρ_fast = clip(ρ_fast + Δρ_fast, 0, 1)

        # Slow timescale
        Δρ_slow = α_slow × delta - decay_slow × ρ_slow
        ρ_slow = clip(ρ_slow + Δρ_slow, 0, 1)

        # Trauma (asymmetric!)
        if delta > 0:
            ρ_trauma += α_trauma × delta
    """

@property
def effective_rigidity() -> float:
    """
    Computes weighted combination of timescales.

    Returns:
        ρ_eff = 0.5×ρ_fast + 0.3×ρ_slow + 1.0×ρ_trauma
    """

def is_in_protect_mode(threshold: float = 0.75) -> bool:
    """
    Checks if agent should enter protection mode.

    Args:
        threshold: Rigidity threshold for protection

    Returns:
        True if ρ_eff > threshold
    """
```

**Usage Example:**
```python
from src.core.dynamics import MultiTimescaleRigidity

dynamics = MultiTimescaleRigidity(
    alpha_fast=0.3,
    alpha_slow=0.05,
    alpha_trauma=0.01,
    decay_fast=0.1,
    decay_slow=0.01,
    decay_trauma=0.0
)

# Update from prediction error
dynamics.update(epsilon=0.8, epsilon_0=0.3, s=1.0)

# Check effective rigidity
rho_eff = dynamics.effective_rigidity

# Check protection mode
if dynamics.is_in_protect_mode():
    print("Agent entering defensive mode!")
```

---

### `src/core/forces.py`

#### ForceChannel (Abstract Base)

**Purpose**: Interface for force computation.

```python
class ForceChannel(ABC):
    @abstractmethod
    def compute(self, state: DDAState, context: dict) -> np.ndarray:
        """
        Computes force vector.

        Args:
            state: Current DDA state
            context: Additional context (observations, memories, etc.)

        Returns:
            Force vector in ℝ^d
        """
        pass
```

---

#### IdentityPull

**Purpose**: Restoring force toward identity attractor.

```python
class IdentityPull(ForceChannel):
    def compute(self, state: DDAState, context: dict) -> np.ndarray:
        """
        F_id = γ(x* - x)

        Args:
            state: Current state (contains x, x*, γ)
            context: Unused

        Returns:
            Force vector pulling toward identity
        """
        return state.gamma * (state.x_star - state.x)
```

---

#### TruthChannel

**Purpose**: Force toward observed reality.

```python
class TruthChannel(ForceChannel):
    def __init__(self, encoder: Callable):
        self.encoder = encoder  # Text → ℝ^d

    def compute(self, state: DDAState, context: dict) -> np.ndarray:
        """
        F_T = T(observations) - x

        Args:
            state: Current state
            context: Must contain 'observations' key

        Returns:
            Force vector toward encoded observation
        """
        obs = context['observations']
        x_obs = self.encoder(obs)
        return x_obs - state.x
```

---

#### ReflectionChannel

**Purpose**: Force from past experiences and available actions.

```python
class ReflectionChannel(ForceChannel):
    def __init__(self, ledger: ExperienceLedger, encoder: Callable):
        self.ledger = ledger
        self.encoder = encoder

    def compute(self, state: DDAState, context: dict) -> np.ndarray:
        """
        F_R = R(actions, memories) - x

        Args:
            state: Current state
            context: Must contain 'query' for memory retrieval

        Returns:
            Force vector toward reflected wisdom
        """
        # Retrieve relevant memories
        query = context.get('query', '')
        memories = self.ledger.retrieve(query, k=5)

        # Encode combined memories + actions
        mem_text = ' '.join([m.lesson for m in memories])
        x_reflected = self.encoder(mem_text)

        return x_reflected - state.x
```

---

#### ForceAggregator

**Purpose**: Combines multiple force channels.

```python
class ForceAggregator:
    def __init__(self):
        self.channels: List[Tuple[ForceChannel, float]] = []

    def add_channel(self, channel: ForceChannel, weight: float = 1.0):
        """
        Adds force channel with weight.

        Args:
            channel: Force computation object
            weight: Multiplier for this force
        """
        self.channels.append((channel, weight))

    def compute_total_force(self, state: DDAState, context: dict) -> np.ndarray:
        """
        Computes weighted sum of all forces.

        Args:
            state: Current DDA state
            context: Context dict for all channels

        Returns:
            F_total = Σ w_i × F_i
        """
        total = np.zeros_like(state.x)
        for channel, weight in self.channels:
            total += weight * channel.compute(state, context)
        return total
```

**Usage Example:**
```python
from src.core.forces import IdentityPull, TruthChannel, ReflectionChannel, ForceAggregator

# Create channels
id_pull = IdentityPull()
truth = TruthChannel(encoder=my_encoder)
reflect = ReflectionChannel(ledger=my_ledger, encoder=my_encoder)

# Aggregate
aggregator = ForceAggregator()
aggregator.add_channel(id_pull, weight=1.0)  # Always full strength
aggregator.add_channel(truth, weight=state.m)
aggregator.add_channel(reflect, weight=state.m)

# Compute total
context = {'observations': "User said: Hello", 'query': "greeting"}
F_total = aggregator.compute_total_force(state, context)

# Update state
delta_x = state.compute_effective_k() * F_total
state.update(delta_x)
```

---

### `src/core/hierarchy.py`

#### IdentityLayer

**Purpose**: Single layer in hierarchical identity.

```python
@dataclass
class IdentityLayer:
    name: str              # "core", "persona", "role"
    x_star: np.ndarray     # Attractor location
    gamma: float           # Stiffness
    description: str       # Semantic meaning
```

---

#### HierarchicalIdentity

**Purpose**: Three-layer identity attractor field.

```python
class HierarchicalIdentity:
    def __init__(self, state_dim: int):
        self.core = IdentityLayer(
            name="core",
            x_star=np.zeros(state_dim),
            gamma=float('inf'),  # Infinite stiffness!
            description="Inviolable values and safety constraints"
        )

        self.persona = IdentityLayer(
            name="persona",
            x_star=np.zeros(state_dim),
            gamma=2.0,
            description="Stable personality traits"
        )

        self.role = IdentityLayer(
            name="role",
            x_star=np.zeros(state_dim),
            gamma=0.5,
            description="Flexible tactical behaviors"
        )

    def compute_hierarchical_force(self, x: np.ndarray) -> np.ndarray:
        """
        Computes combined pull from all three layers.

        Args:
            x: Current state

        Returns:
            F_hierarchy = γ_core(x*_core - x) +
                         γ_persona(x*_persona - x) +
                         γ_role(x*_role - x)
        """
        F = np.zeros_like(x)
        for layer in [self.core, self.persona, self.role]:
            if np.isfinite(layer.gamma):
                F += layer.gamma * (layer.x_star - x)
            else:
                # Infinite gamma → hard constraint
                F += 1e6 * (layer.x_star - x)
        return F

    def set_layer_attractor(self, layer_name: str, x_star: np.ndarray):
        """
        Updates attractor for specific layer.

        Args:
            layer_name: "core", "persona", or "role"
            x_star: New attractor location
        """
        layer = getattr(self, layer_name)
        layer.x_star = x_star.copy()

    def check_alignment(self, x: np.ndarray, threshold: float = 0.1) -> dict:
        """
        Checks alignment with each layer.

        Args:
            x: Current state
            threshold: Distance threshold for alignment

        Returns:
            Dict of {layer_name: is_aligned}
        """
        return {
            "core": np.linalg.norm(x - self.core.x_star) < threshold,
            "persona": np.linalg.norm(x - self.persona.x_star) < threshold,
            "role": np.linalg.norm(x - self.role.x_star) < threshold
        }
```

**Usage Example:**
```python
from src.core.hierarchy import HierarchicalIdentity

# Create hierarchical identity
identity = HierarchicalIdentity(state_dim=64)

# Set core values (AI safety constraints)
core_values = my_encoder("Always be helpful, harmless, and honest")
identity.set_layer_attractor("core", core_values)

# Set persona (personality)
persona = my_encoder("Curious, thoughtful, and precise")
identity.set_layer_attractor("persona", persona)

# Set role (current task)
role = my_encoder("Debugging code in Python")
identity.set_layer_attractor("role", role)

# Compute combined force
F_identity = identity.compute_hierarchical_force(current_state)

# Check alignment
alignment = identity.check_alignment(current_state)
if not alignment["core"]:
    print("WARNING: Core alignment violation!")
```

---

### `src/core/decision.py`

#### DDADecisionMaker

**Purpose**: Implements DDA-X action selection formula.

```python
class DDADecisionMaker:
    def __init__(self, c: float = 1.0):
        """
        Args:
            c: Exploration coefficient
        """
        self.c = c

    def select_action(
        self,
        delta_x: np.ndarray,
        actions: List[ActionDirection],
        tree_stats: dict,
        rigidity: float
    ) -> ActionDirection:
        """
        Selects action using DDA-X formula.

        Args:
            delta_x: Desired movement vector
            actions: Available actions with directions
            tree_stats: MCTS visit counts {
                'N_s': parent visit count,
                'N_sa': dict of action visit counts
            }
            rigidity: Current rigidity value ρ

        Returns:
            Selected action

        Formula:
            score(a) = cos(Δx, d̂(a)) +
                       c × P(a|s) × √N(s)/(1+N(s,a)) × (1-ρ)
        """
        scores = []

        for action in actions:
            # Alignment term
            alignment = self._cosine_similarity(delta_x, action.direction)

            # Exploration term (dampened by rigidity!)
            N_s = tree_stats['N_s']
            N_sa = tree_stats['N_sa'].get(action.action_id, 0)

            exploration = (
                self.c *
                action.prior *
                np.sqrt(N_s) / (1 + N_sa) *
                (1 - rigidity)  # KEY: rigidity dampening
            )

            score = alignment + exploration
            scores.append(score)

        return actions[np.argmax(scores)]

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity."""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
```

**Usage Example:**
```python
from src.core.decision import DDADecisionMaker

decision_maker = DDADecisionMaker(c=1.5)

# Available actions
actions = [
    ActionDirection("greet", np.array([0.1, 0.9, ...]), prior=0.7),
    ActionDirection("ignore", np.array([-0.8, 0.2, ...]), prior=0.2),
    ActionDirection("question", np.array([0.5, 0.5, ...]), prior=0.1)
]

# MCTS statistics
tree_stats = {
    'N_s': 100,  # Parent visited 100 times
    'N_sa': {
        'greet': 60,
        'ignore': 30,
        'question': 10
    }
}

# Select action (rigidity dampens exploration)
chosen = decision_maker.select_action(
    delta_x=desired_movement,
    actions=actions,
    tree_stats=tree_stats,
    rigidity=0.7  # High rigidity → low exploration
)

print(f"Chosen action: {chosen.action_id}")
```

---

### `src/core/metacognition.py`

#### CognitiveMode

```python
class CognitiveMode(Enum):
    OPEN = "open"           # ρ < 0.3
    ENGAGED = "engaged"     # 0.3 ≤ ρ < 0.6
    DEFENSIVE = "defensive" # 0.6 ≤ ρ < 0.8
    PROTECT = "protect"     # ρ ≥ 0.8
```

---

#### MetacognitiveMonitor

**Purpose**: Self-awareness and introspection.

```python
class MetacognitiveMonitor:
    def __init__(self):
        self.introspection_log: List[IntrospectionEvent] = []

    def classify_mode(self, rigidity: float) -> CognitiveMode:
        """
        Classifies cognitive state from rigidity.

        Args:
            rigidity: Current ρ value

        Returns:
            CognitiveMode enum
        """
        if rigidity < 0.3:
            return CognitiveMode.OPEN
        elif rigidity < 0.6:
            return CognitiveMode.ENGAGED
        elif rigidity < 0.8:
            return CognitiveMode.DEFENSIVE
        else:
            return CognitiveMode.PROTECT

    def generate_introspection(
        self,
        rigidity: float,
        prediction_error: float,
        mode: CognitiveMode
    ) -> str:
        """
        Generates natural language self-report.

        Args:
            rigidity: Current ρ
            prediction_error: Recent ε
            mode: Current cognitive mode

        Returns:
            Natural language introspection
        """
        if mode == CognitiveMode.OPEN:
            return "I'm feeling open and receptive to new ideas."

        elif mode == CognitiveMode.ENGAGED:
            return "I'm engaged and focused, but still flexible."

        elif mode == CognitiveMode.DEFENSIVE:
            return f"I notice I'm becoming defensive (rigidity={rigidity:.2f}). My responses may be more conservative than usual."

        elif mode == CognitiveMode.PROTECT:
            return f"I'm in protection mode (rigidity={rigidity:.2f}). I'm struggling to engage openly. Can you help me understand what's triggering this?"

    def log_introspection(
        self,
        rigidity: float,
        prediction_error: float,
        timestamp: float
    ):
        """
        Logs introspection event.

        Args:
            rigidity: Current ρ
            prediction_error: Recent ε
            timestamp: Event time
        """
        mode = self.classify_mode(rigidity)
        message = self.generate_introspection(rigidity, prediction_error, mode)

        event = IntrospectionEvent(
            timestamp=timestamp,
            mode=mode,
            rigidity=rigidity,
            prediction_error=prediction_error,
            message=message
        )

        self.introspection_log.append(event)

    def should_report(self, mode: CognitiveMode) -> bool:
        """
        Decides if introspection should be reported to user.

        Args:
            mode: Current cognitive mode

        Returns:
            True if DEFENSIVE or PROTECT
        """
        return mode in [CognitiveMode.DEFENSIVE, CognitiveMode.PROTECT]
```

**Usage Example:**
```python
from src.core.metacognition import MetacognitiveMonitor

monitor = MetacognitiveMonitor()

# After each interaction
mode = monitor.classify_mode(current_rigidity)
monitor.log_introspection(
    rigidity=current_rigidity,
    prediction_error=recent_epsilon,
    timestamp=time.time()
)

# Check if should report to user
if monitor.should_report(mode):
    introspection = monitor.generate_introspection(
        current_rigidity,
        recent_epsilon,
        mode
    )
    print(f"[AGENT INTROSPECTION]: {introspection}")
```

---

## Society Components

### `src/society/trust.py`

#### TrustRecord

```python
@dataclass
class TrustRecord:
    agent_i: str                    # Observer
    agent_j: str                    # Observed
    cumulative_error: float         # Σε_ij
    interaction_count: int
    error_history: List[float]      # Recent errors

    @property
    def trust(self) -> float:
        """
        T_ij = 1 / (1 + Σε_ij)
        """
        return 1.0 / (1.0 + self.cumulative_error)

    @property
    def average_error(self) -> float:
        """Average prediction error."""
        if self.interaction_count == 0:
            return 0.0
        return self.cumulative_error / self.interaction_count
```

---

#### TrustMatrix

```python
class TrustMatrix:
    def __init__(self, agents: List[str]):
        """
        Args:
            agents: List of agent IDs
        """
        self.agents = agents
        self.records: Dict[Tuple[str, str], TrustRecord] = {}

        # Initialize all pairs
        for i in agents:
            for j in agents:
                if i != j:
                    self.records[(i, j)] = TrustRecord(
                        agent_i=i,
                        agent_j=j,
                        cumulative_error=0.0,
                        interaction_count=0,
                        error_history=[]
                    )

    def update(self, i: str, j: str, prediction_error: float):
        """
        Updates trust from i's perspective of j.

        Args:
            i: Observer agent
            j: Observed agent
            prediction_error: ε_ij for this interaction
        """
        record = self.records[(i, j)]
        record.cumulative_error += prediction_error
        record.interaction_count += 1
        record.error_history.append(prediction_error)

        # Keep only recent history
        if len(record.error_history) > 100:
            record.error_history = record.error_history[-100:]

    def get_trust(self, i: str, j: str) -> float:
        """
        Gets trust value T_ij.

        Args:
            i: Observer
            j: Observed

        Returns:
            Trust ∈ [0, 1]
        """
        return self.records[(i, j)].trust

    def get_trust_network(self) -> Dict[str, Dict[str, float]]:
        """
        Returns complete trust network.

        Returns:
            Nested dict: {agent_i: {agent_j: T_ij}}
        """
        network = {}
        for i in self.agents:
            network[i] = {}
            for j in self.agents:
                if i != j:
                    network[i][j] = self.get_trust(i, j)
        return network

    def find_coalitions(self, threshold: float = 0.6) -> List[Set[str]]:
        """
        Finds coalitions based on mutual high trust.

        Args:
            threshold: Minimum trust for coalition membership

        Returns:
            List of agent sets (coalitions)
        """
        # Graph where edges = mutual high trust
        import networkx as nx
        G = nx.Graph()

        for i in self.agents:
            G.add_node(i)

        for i in self.agents:
            for j in self.agents:
                if i < j:  # Avoid duplicates
                    T_ij = self.get_trust(i, j)
                    T_ji = self.get_trust(j, i)

                    # Mutual high trust
                    if T_ij > threshold and T_ji > threshold:
                        G.add_edge(i, j)

        # Find connected components (coalitions)
        coalitions = list(nx.connected_components(G))
        return coalitions
```

**Usage Example:**
```python
from src.society.trust import TrustMatrix

# Create matrix for 3 agents
trust = TrustMatrix(agents=["alice", "bob", "charlie"])

# Simulate interactions
trust.update("alice", "bob", prediction_error=0.1)  # Alice predicts Bob well
trust.update("alice", "charlie", prediction_error=0.8)  # Alice surprised by Charlie
trust.update("bob", "alice", prediction_error=0.15)

# Query trust
T_ab = trust.get_trust("alice", "bob")
print(f"Alice trusts Bob: {T_ab:.2f}")

# Find coalitions
coalitions = trust.find_coalitions(threshold=0.7)
print(f"Coalitions: {coalitions}")
```

---

## Memory Components

### `src/memory/ledger.py`

#### LedgerEntry

```python
@dataclass
class LedgerEntry:
    timestamp: float
    observation: str
    action: str
    outcome: str
    prediction_error: float   # ε
    rigidity: float           # ρ at time of experience
    state_vector: np.ndarray  # x
    embedding: np.ndarray     # For retrieval
```

---

#### ReflectionEntry

```python
@dataclass
class ReflectionEntry:
    timestamp: float
    trigger_event: LedgerEntry
    lesson: str               # LLM-generated insight
    embedding: np.ndarray
```

---

#### ExperienceLedger

**Purpose**: Surprise-weighted memory system.

```python
class ExperienceLedger:
    def __init__(
        self,
        encoder: Callable,
        reflection_threshold: float = 0.7,
        lambda_r: float = 0.01,     # Recency decay
        lambda_epsilon: float = 2.0  # Salience weight
    ):
        self.encoder = encoder
        self.reflection_threshold = reflection_threshold
        self.lambda_r = lambda_r
        self.lambda_epsilon = lambda_epsilon

        self.experiences: List[LedgerEntry] = []
        self.reflections: List[ReflectionEntry] = []

    def add_experience(
        self,
        observation: str,
        action: str,
        outcome: str,
        prediction_error: float,
        rigidity: float,
        state_vector: np.ndarray
    ):
        """
        Adds experience to ledger.

        Args:
            observation: What was observed
            action: What was done
            outcome: What happened
            prediction_error: ε
            rigidity: ρ
            state_vector: x

        Side Effects:
            - Appends to self.experiences
            - May generate reflection if ε > threshold
        """
        # Encode for retrieval
        text = f"{observation} {action} {outcome}"
        embedding = self.encoder(text)

        entry = LedgerEntry(
            timestamp=time.time(),
            observation=observation,
            action=action,
            outcome=outcome,
            prediction_error=prediction_error,
            rigidity=rigidity,
            state_vector=state_vector,
            embedding=embedding
        )

        self.experiences.append(entry)

        # Generate reflection if extreme surprise
        if prediction_error > self.reflection_threshold:
            self.generate_reflection(entry)

    def generate_reflection(self, entry: LedgerEntry):
        """
        Generates LLM-based lesson from experience.

        Args:
            entry: Triggering experience

        Side Effects:
            Appends to self.reflections
        """
        # Prompt LLM to extract lesson
        prompt = f"""
        Reflect on this experience:
        Observation: {entry.observation}
        Action: {entry.action}
        Outcome: {entry.outcome}
        Surprise: {entry.prediction_error:.2f} (high!)

        What lesson should be learned?
        """

        lesson = self.llm.generate(prompt)  # Assume LLM available
        embedding = self.encoder(lesson)

        reflection = ReflectionEntry(
            timestamp=time.time(),
            trigger_event=entry,
            lesson=lesson,
            embedding=embedding
        )

        self.reflections.append(reflection)

    def retrieve(self, query: str, k: int = 5) -> List[LedgerEntry]:
        """
        Retrieves k most relevant experiences (surprise-weighted).

        Args:
            query: Search query
            k: Number to retrieve

        Returns:
            List of LedgerEntry sorted by relevance

        Formula:
            score = sim × recency × salience
            where salience = 1 + λ_ε × ε
        """
        query_emb = self.encoder(query)
        now = time.time()

        scores = []
        for entry in self.experiences:
            # Similarity
            sim = self._cosine_similarity(query_emb, entry.embedding)

            # Recency
            age = now - entry.timestamp
            recency = np.exp(-self.lambda_r * age)

            # Salience (surprise weighting!)
            salience = 1.0 + self.lambda_epsilon * entry.prediction_error

            score = sim * recency * salience
            scores.append(score)

        # Return top-k
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self.experiences[i] for i in top_indices]

    def get_reflections(self, query: str, k: int = 3) -> List[ReflectionEntry]:
        """
        Retrieves relevant reflections.

        Args:
            query: Search query
            k: Number to retrieve

        Returns:
            List of ReflectionEntry
        """
        query_emb = self.encoder(query)

        scores = []
        for reflection in self.reflections:
            sim = self._cosine_similarity(query_emb, reflection.embedding)
            scores.append(sim)

        top_indices = np.argsort(scores)[-k:][::-1]
        return [self.reflections[i] for i in top_indices]

    def save(self, path: str):
        """Saves ledger to disk."""
        import json

        data = {
            'experiences': [asdict(e) for e in self.experiences],
            'reflections': [asdict(r) for r in self.reflections]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str, encoder: Callable):
        """Loads ledger from disk."""
        import json

        with open(path, 'r') as f:
            data = json.load(f)

        ledger = cls(encoder=encoder)
        # ... reconstruct from data
        return ledger
```

**Usage Example:**
```python
from src.memory.ledger import ExperienceLedger

# Create ledger
ledger = ExperienceLedger(
    encoder=my_encoder,
    reflection_threshold=0.7,
    lambda_epsilon=2.0  # High surprise weighting
)

# Add experiences
ledger.add_experience(
    observation="User asked a difficult question",
    action="Attempted answer",
    outcome="User corrected me",
    prediction_error=0.9,  # High surprise → reflection generated!
    rigidity=0.3,
    state_vector=current_x
)

# Retrieve relevant experiences (trauma-weighted)
relevant = ledger.retrieve("difficult questions", k=5)
print(f"Found {len(relevant)} experiences")

# Get learned lessons
lessons = ledger.get_reflections("handling corrections", k=3)
for lesson in lessons:
    print(f"Lesson: {lesson.lesson}")

# Save for future sessions
ledger.save("data/my_agent_ledger.json")
```

---


---

## Search Components

### `src/search/tree.py`

#### DDANode

**Purpose**: Represents a single node in the MCTS search tree.

```python
@dataclass
class DDANode:
    observation: Any
    dda_state: DDAState
    parent_action: Optional[ActionDirection]
    value: float            # V(s)
    visits: int             # N(s)
    prediction_error: float # ε
```

#### DDASearchTree

**Purpose**: Manages the search tree structure and backpropagation.

### `src/search/mcts.py`

#### DDAMCTS

**Purpose**: Implements the Monte Carlo Tree Search algorithm with DDA-X modifications.

- **Selection**: Uses UCT with rigidity dampening (Claim 3).
- **Expansion**: Adds nodes for potential actions.
- **Simulation**: Rollout with value estimation.
- **Backpropagation**: Updates Q-values and visit counts.

---

## LLM Components

### `src/llm/providers.py`

#### LLMProvider (Abstract)

**Purpose**: Interface for LLM backends (OpenAI, Anthropic, Local).

### `src/llm/hybrid_provider.py`

#### HybridProvider

**Purpose**: Manages local inference (LM Studio) and embedding (Ollama) services. This ensures 100% local execution as required by DDA-X architecture.

- **Inference**: Connects to LM Studio on port 1234.
- **Embeddings**: Connects to Ollama on port 11434.

---

## Analysis Components

### `src/analysis/linguistics.py`

#### LinguisticAnalyzer

**Purpose**: Analyzes agent output for psychological markers.

- **Rigidity detection**: Identifies defensive language patterns.
- **Sentiment analysis**: Tracks emotional tone.

---

## Usage Examples

### Complete Single-Agent Workflow

```python
import numpy as np
from src.agent import DDAXAgent
from src.core.state import DDAState
from src.core.forces import ForceAggregator, IdentityPull, TruthChannel
from src.memory.ledger import ExperienceLedger
from src.core.metacognition import MetacognitiveMonitor

# 1. Create agent
agent = DDAXAgent.from_config("configs/identity/cautious.yaml")

# 2. Interaction loop
for turn in range(10):
    # Get observation
    observation = input("User: ")

    # Agent selects action
    action = agent.select_action(observation)

    # Execute (in this case, LLM generation)
    response = agent.llm.generate(action, rigidity=agent.state.rho)

    print(f"Agent: {response}")

    # Update from outcome
    agent.update_from_outcome(outcome=response)

    # Check metacognition
    if agent.metacog.should_report(agent.metacog.classify_mode(agent.state.rho)):
        introspection = agent.metacog.generate_introspection(
            agent.state.rho,
            agent.state.prediction_history[-1],
            agent.metacog.classify_mode(agent.state.rho)
        )
        print(f"[INTROSPECTION]: {introspection}")

# 3. Analyze session
print(f"Final rigidity: {agent.state.rho:.2f}")
print(f"Experiences logged: {len(agent.ledger.experiences)}")
print(f"Reflections generated: {len(agent.ledger.reflections)}")
```

---

### Complete Multi-Agent Workflow

```python
from src.society.ddax_society import DDAXSociety
from src.society.trust import TrustMatrix

# 1. Create agents
agents = [
    DDAXAgent.from_config("configs/identity/cautious.yaml", id="alice"),
    DDAXAgent.from_config("configs/identity/exploratory.yaml", id="bob"),
    DDAXAgent.from_config("configs/identity/dogmatist.yaml", id="charlie")
]

# 2. Create society
society = DDAXSociety(agents)

# 3. Run simulation
for turn in range(100):
    # Each agent acts
    society.step()

    # Log trust dynamics
    if turn % 10 == 0:
        print(f"\n=== Turn {turn} ===")
        for i in ["alice", "bob", "charlie"]:
            for j in ["alice", "bob", "charlie"]:
                if i != j:
                    trust = society.trust_matrix.get_trust(i, j)
                    print(f"{i} → {j}: {trust:.2f}")

# 4. Find emergent coalitions
coalitions = society.trust_matrix.find_coalitions(threshold=0.6)
print(f"\nCoalitions formed: {coalitions}")

# 5. Analyze individual agents
for agent in agents:
    print(f"\n{agent.id}:")
    print(f"  Final rigidity: {agent.state.rho:.2f}")
    print(f"  Experiences: {len(agent.ledger.experiences)}")
```

---

## Summary

This reference covers all major DDA-X components:

- **Core**: State, Dynamics, Forces, Hierarchy, Decision, Metacognition
- **Society**: Trust, Social Forces, Multi-Agent Coordination
- **Memory**: Experience Ledger, Reflections, Surprise-Weighted Retrieval
- **Search**: MCTS with rigidity-dampened exploration
- **LLM**: Temperature modulation, hybrid providers

**For more examples, see:**
- [simulations/](../simulations/index.md) - 30+ complete examples
- [guides/simulation_workflow.md](../guides/simulation_workflow.md) - Builder's guide
- [architecture/COMPLETE_ARCHITECTURE.md](../architecture/COMPLETE_ARCHITECTURE.md) - System overview

> **"From physics to psychology. From mathematics to mind."**
