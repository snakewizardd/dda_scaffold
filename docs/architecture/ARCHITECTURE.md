# DDA-X Cognitive Engine Implementation

> **How the theory becomes code — with real examples from the simulations.**

This document maps theoretical concepts to their **actual implementations** in the pinnacle simulations. Every code block below is extracted from real, working simulation files.

---

## 1. Core Data Structures

### AgentState (The Cognitive Engine)

From [simulate_agi_debate.py](../../simulations/simulate_agi_debate.py#L462-L539):

```python
@dataclass
class AgentState:
    """Complete agent state following DDA-X architecture."""
    id: str
    name: str
    
    # Embedding vectors (ℝ^d)
    identity_emb: np.ndarray = None      # x* - identity attractor
    wound_emb: np.ndarray = None         # w* - wound trigger vector
    x: np.ndarray = None                 # Current state vector
    x_pred: np.ndarray = None            # Prediction vector
    
    # Multi-timescale rigidity
    multi_rho: MultiTimescaleRigidity = None
    
    # Agent-specific parameters
    epsilon_0: float = 0.3               # Surprise threshold
    gamma: float = 1.5                   # Identity stiffness
    
    # Hierarchical Identity (3 layers)
    hierarchical_identity: Dict[str, Dict] = field(default_factory=dict)
    
    # Trust toward other agents
    trust_opponent: float = 0.5
    
    # Wound mechanics
    wound_last_activated: int = -100
    
    # Memory
    ledger: ExperienceLedger = None
```

### D1 Physics Parameters

From [simulate_agi_debate.py](../../simulations/simulate_agi_debate.py#L230-L275):

```python
D1_PARAMS = {
    # Rigidity dynamics
    "epsilon_0": 0.75,           # Surprise threshold
    "alpha": 0.12,               # Rigidity learning rate
    "s": 0.20,                   # Sigmoid sensitivity
    
    # Multi-timescale rigidity
    "alpha_fast": 0.30,          # Fast timescale learning rate
    "alpha_slow": 0.01,          # Slow timescale learning rate
    "alpha_trauma": 0.0001,      # Trauma accumulation rate (asymmetric!)
    "trauma_threshold": 0.7,     # Epsilon threshold for trauma
    "rho_weights": {             # Effective rigidity weights
        "fast": 0.5,
        "slow": 0.3,
        "trauma": 1.0,
    },
    
    # State dynamics
    "drift_cap": 0.05,           # Max state drift per turn
    "k_base": 0.5,               # Base step size
    "m": 1.0,                    # External pressure gain
    
    # Wound mechanics
    "wound_cooldown": 3,
    "wound_amp_max": 1.4,
    "wound_cosine_threshold": 0.28,
    
    # Trust mechanics
    "trust_intra_weight": 0.08,
    "trust_inter_weight": 0.03,
    
    # Protection mode
    "protect_threshold": 0.75,
}
```

---

## 2. Multi-Timescale Rigidity Engine

From [simulate_agi_debate.py](../../simulations/simulate_agi_debate.py#L411-L456):

```python
@dataclass
class MultiTimescaleRigidity:
    """
    Three temporal scales of defensive response:
    - rho_fast: Startle response (τ ~ seconds)
    - rho_slow: Stress accumulation (τ ~ minutes)
    - rho_trauma: Permanent scarring (τ → ∞, asymmetric!)
    """
    rho_fast: float = 0.0
    rho_slow: float = 0.0
    rho_trauma: float = 0.0
    
    def update(self, prediction_error: float, epsilon_0: float = 0.3, s: float = 0.1):
        """Update all timescales based on prediction error."""
        z = (prediction_error - epsilon_0) / s
        sig = sigmoid(z)
        
        # Fast timescale - quick response, quick decay
        delta_fast = D1_PARAMS["alpha_fast"] * (sig - 0.5)
        self.rho_fast = float(np.clip(self.rho_fast + delta_fast, 0.0, 1.0))
        
        # Slow timescale - gradual accumulation
        delta_slow = D1_PARAMS["alpha_slow"] * (sig - 0.5)
        self.rho_slow = float(np.clip(self.rho_slow + delta_slow, 0.0, 1.0))
        
        # Trauma - ASYMMETRIC! Only increases when above threshold
        delta_trauma = 0.0
        if prediction_error > D1_PARAMS["trauma_threshold"]:
            delta_trauma = D1_PARAMS["alpha_trauma"] * (prediction_error - D1_PARAMS["trauma_threshold"])
            self.rho_trauma = float(np.clip(self.rho_trauma + delta_trauma, 0.0, 1.0))
    
    @property
    def effective_rho(self) -> float:
        """Weighted combination: rho_eff = 0.5·rho_fast + 0.3·rho_slow + 1.0·rho_trauma"""
        w = D1_PARAMS["rho_weights"]
        effective = w["fast"] * self.rho_fast + w["slow"] * self.rho_slow + w["trauma"] * self.rho_trauma
        return min(1.0, effective)
```

**Equation:** $\rho_{\text{eff}} = 0.5 \rho_{\text{fast}} + 0.3 \rho_{\text{slow}} + 1.0 \rho_{\text{trauma}}$

---

## 3. Effective Openness & Will Impedance

From [simulate_agi_debate.py](../../simulations/simulate_agi_debate.py#L392-L405):

```python
def compute_k_effective(k_base: float, rho: float) -> float:
    """k_eff = k_base(1 - ρ) - from paper.md"""
    return k_base * (1 - rho)


def compute_will_impedance(gamma: float, m: float, k_eff: float) -> float:
    """
    Will Impedance: W_t = γ / (m_t · k_eff)
    Quantifies resistance to environmental pressure.
    """
    if m * k_eff == 0:
        return float('inf')
    return gamma / (m * k_eff)
```

**Equations:**
- $k_{\text{eff}} = k_{\text{base}} \cdot (1 - \rho)$
- $W_t = \gamma \,/\, (m_t \cdot k_{\text{eff}})$

---

## 4. Wound Detection (Semantic + Lexical)

### Wound Lexicons

From [simulate_agi_debate.py](../../simulations/simulate_agi_debate.py#L124-L136):

```python
WOUND_LEX_DEFENDER = {
    "naive", "naïve", "hype", "delusional", "wrong", "irresponsible",
    "overpromising", "dangerous", "misleading", "cult", "religion",
    "sciencefiction", "science fiction", "fantasy", "wishful thinking",
    "no evidence", "unfounded", "grift", "grifter",
}

WOUND_LEX_SKEPTIC = {
    "luddite", "dinosaur", "shortsighted", "ignorant", "not understanding",
    "technophobe", "fearmonger", "doomer", "pessimist", "stuck in the past",
    "missing the point", "doesn't get it", "outdated", "irrelevant",
}
```

### Lexical Detection

From [simulate_agi_debate.py](../../simulations/simulate_agi_debate.py#L147-L161):

```python
def lexical_wound_with(text: str, words: Set[str]) -> bool:
    """Check for wound terms in text using specified lexicon."""
    t_lower = text.lower()
    t_norm = normalize_text(text)
    return any(w in t_lower or w in t_norm for w in words)
```

### Semantic Detection (Cosine Similarity)

From [simulate_skeptics_gauntlet.py](../../simulations/simulate_skeptics_gauntlet.py):

```python
def check_wound_activation(agent: AgentState, message: str, msg_emb: np.ndarray, turn: int) -> Tuple[bool, float, str]:
    """Check if message triggers agent's psychological wound."""
    # Semantic: cosine similarity with wound embedding
    wound_cosine = float(np.dot(msg_emb, agent.wound_emb))
    
    # Lexical: keyword matching
    lexicon = WOUND_LEX_DEFENDER if agent.id == "DEFENDER" else WOUND_LEX_SKEPTIC
    lex_hit = lexical_wound_with(message, lexicon)
    lex_trigger = find_lexical_trigger(message, lexicon) if lex_hit else ""
    
    # Cooldown check
    since_last = turn - agent.wound_last_activated
    cooled = since_last >= D1_PARAMS["wound_cooldown"]
    
    # Activation condition
    activated = ((wound_cosine > D1_PARAMS["wound_cosine_threshold"]) or lex_hit) and cooled
    
    return activated, wound_cosine, lex_trigger
```

### Wound Amplification

```python
if wound_active:
    epsilon *= min(D1_PARAMS["wound_amp_max"], 1 + 0.5 * wound_cosine)
    agent.wound_last_activated = turn
```

**Equation:** $\varepsilon' = \varepsilon \cdot \min(\eta_{\max}, 1 + 0.5 \cdot r_{\text{wound}})$

---

## 5. Cognitive Mode Bands

From [simulate_agi_debate.py](../../simulations/simulate_agi_debate.py#L101-L116):

```python
class CognitiveMode(Enum):
    OPEN = "open"           # ρ < 0.3
    ENGAGED = "engaged"     # 0.3 ≤ ρ < 0.6
    DEFENSIVE = "defensive" # 0.6 ≤ ρ < 0.8
    PROTECT = "protect"     # ρ ≥ 0.8


def get_cognitive_mode(rho: float) -> CognitiveMode:
    if rho < 0.3:
        return CognitiveMode.OPEN
    elif rho < 0.6:
        return CognitiveMode.ENGAGED
    elif rho < 0.8:
        return CognitiveMode.DEFENSIVE
    else:
        return CognitiveMode.PROTECT
```

### Response Length Constraints

From [simulate_agi_debate.py](../../simulations/simulate_agi_debate.py#L365-L372):

```python
def regime_words(band: str) -> Tuple[int, int]:
    return {
        "OPEN": (100, 200),
        "MEASURED": (70, 140),
        "GUARDED": (40, 90),
        "FORTIFIED": (20, 50),
        "SILENT": (0, 0),
    }.get(band, (70, 140))
```

---

## 6. Trust from Predictability

From [simulate_philosophers_duel.py](../../simulations/simulate_philosophers_duel.py):

```python
def update_trust(agent: AgentState, other_id: str, prediction_error: float):
    """
    Trust = 1 / (1 + Σε_ij)
    Trust accumulates inversely with cumulative prediction error.
    """
    agent.cumulative_epsilon[other_id] += prediction_error
    agent.trust[other_id] = 1.0 / (1.0 + agent.cumulative_epsilon[other_id])
```

**Equation:** $T_{ij} = 1 \,/\, (1 + \sum \varepsilon_{ij})$

---

## 7. Hierarchical Identity Force

From [simulate_agi_debate.py](../../simulations/simulate_agi_debate.py#L528-L538):

```python
def compute_hierarchical_force(self) -> np.ndarray:
    """
    Compute hierarchical identity pull:
    F_total = Σ γ_layer × (x*_layer - x)
    """
    F_total = np.zeros_like(self.x)
    for layer_name, layer_data in self.hierarchical_identity.items():
        gamma_layer = layer_data.get("gamma", 1.0)
        F_total += gamma_layer * (self.identity_emb - self.x)
    return F_total
```

### Hierarchical Configuration

```python
"hierarchical_identity": {
    "core": {"gamma": 5.0, "text": "I base claims on evidence..."},    # Inviolable
    "persona": {"gamma": 2.0, "text": "I am an optimistic technologist..."}, # Stable
    "role": {"gamma": 0.5, "text": "I must defend the near-term AGI thesis..."}, # Flexible
}
```

**Equation:** $\mathbf{F}_{\text{total}} = \gamma_{\text{core}}(\mathbf{x}^*_c - \mathbf{x}) + \gamma_{\text{persona}}(\mathbf{x}^*_p - \mathbf{x}) + \gamma_{\text{role}}(\mathbf{x}^*_r - \mathbf{x})$

---

## 8. Embedding Initialization

From [simulate_agi_debate.py](../../simulations/simulate_agi_debate.py#L633-L645):

```python
# Create identity embedding (x*)
full_identity = f"{cfg['core']} {cfg['persona']}"
identity_emb = await self.provider.embed(full_identity)
identity_emb = identity_emb / (np.linalg.norm(identity_emb) + 1e-9)

# Core embedding
core_emb = await self.provider.embed(cfg['core'])
core_emb = core_emb / (np.linalg.norm(core_emb) + 1e-9)

# Wound embedding (w*)
wound_emb = await self.provider.embed(cfg['wound_text'])
wound_emb = wound_emb / (np.linalg.norm(wound_emb) + 1e-9)

# Initialize current state at identity attractor
x = identity_emb.copy()
```

---

## 9. Parameter-Level LLM Coupling

From [src/llm/openai_provider.py](../../src/llm/openai_provider.py#L150-L170):

```python
def compute_temperature(self, rho: float) -> float:
    """
    T(ρ) = T_low + (1 - ρ)(T_high - T_low)
    High rigidity → low temperature → conservative output
    Low rigidity → high temperature → exploratory output
    """
    T_low, T_high = 0.1, 0.9
    return T_low + (1 - rho) * (T_high - T_low)
```

---

## 10. Therapeutic Recovery (Trauma Decay)

From [simulate_healing_field.py](../../simulations/simulate_healing_field.py):

```python
def check_therapeutic_recovery(agent: AgentState, epsilon: float):
    """
    If consecutive safe interactions, allow trauma to decay.
    Safe: ε < 0.8ε₀
    """
    if epsilon < 0.8 * agent.epsilon_0:
        agent.safe_interaction_count += 1
    else:
        agent.safe_interaction_count = 0
    
    # Healing threshold reached
    if agent.safe_interaction_count >= D1_PARAMS["safe_threshold"]:
        agent.multi_rho.rho_trauma = max(
            D1_PARAMS["trauma_floor"],
            agent.multi_rho.rho_trauma - D1_PARAMS["healing_rate"]
        )
        agent.safe_interaction_count = 0  # Reset counter
```

---

## 11. Presence & Release Fields

From [simulate_inner_council.py](../../simulations/simulate_inner_council.py):

```python
# Presence Field (inverse of rigidity)
presence = 1 - agent.multi_rho.effective_rho

# Release Field (same formula, different semantic context)
release = 1 - agent.multi_rho.effective_rho
```

**Equations:**
- Presence: $\Pi = 1 - \rho$
- Release: $\Phi = 1 - \rho$

---

## 12. Calibration from Live Data

From [simulate_agi_debate.py](../../simulations/simulate_agi_debate.py#L680-L692):

```python
def calibrate_epsilon_params(self):
    """Calibrate ε₀ and s from early run data."""
    all_eps = [r.epsilon for r in self.results if not r.is_silent]
    if len(all_eps) >= 4:
        med = float(np.median(all_eps))
        iqr = float(np.subtract(*np.percentile(all_eps, [75, 25]))) or 0.2
        D1_PARAMS["epsilon_0"] = med
        D1_PARAMS["s"] = max(0.10, min(0.30, iqr))
```

---

## Summary: The Cognitive Loop

```
1. EMBED: Convert identity/wound/message to vectors
2. PREDICT: Expected response embedding
3. OBSERVE: Actual response embedding
4. COMPUTE ε: ||x_pred - x_actual||
5. CHECK WOUND: Semantic + lexical detection
6. AMPLIFY if wound: ε' = ε × amp
7. UPDATE ρ: Multi-timescale (fast, slow, trauma)
8. COMPUTE k_eff: k_base × (1 - ρ)
9. COMPUTE W_t: γ / (m × k_eff)
10. UPDATE TRUST: 1 / (1 + Σε)
11. DETERMINE MODE: OPEN/MEASURED/GUARDED/FORTIFIED
12. GENERATE: LLM with T(ρ) and word constraints
13. LOG: To ExperienceLedger
14. REPEAT
```

This is the DDA-X cognitive engine. It runs inline in every pinnacle simulation.
