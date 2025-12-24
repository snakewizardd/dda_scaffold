# Cognitive Engine Mechanisms

This page provides a complete reference for all DDA-X mechanisms as verified from the implementation. Each section includes the mathematical formulation and code pattern.

---

## 1. Multi-Timescale Rigidity

### The Core Insight

DDA-X models three distinct temporal scales of defensive response:

| Component | Symbol | Description | Behavior |
|:---|:---:|:---|:---|
| **Fast** | $\rho_{\text{fast}}$ | Startle response | Quick rise, quick decay |
| **Slow** | $\rho_{\text{slow}}$ | Stress accumulation | Gradual rise and fall |
| **Trauma** | $\rho_{\text{trauma}}$ | Scarring | Asymmetric (rises on shock, rarely decays) |

### Mathematical Formulation

Given prediction error $\epsilon_t$ and logistic gate parameters $(\epsilon_0, s)$:

$$
z_t = \frac{\epsilon_t - \epsilon_0}{s}, \qquad \sigma(z) = \frac{1}{1+e^{-z}}
$$

**Fast and Slow Updates:**
$$
\Delta\rho_{\text{fast}} = \alpha_{\text{fast}}(\sigma(z_t) - 0.5)
$$
$$
\Delta\rho_{\text{slow}} = \alpha_{\text{slow}}(\sigma(z_t) - 0.5)
$$

**Trauma Update (Asymmetric):**
$$
\Delta\rho_{\text{trauma}} = 
\begin{cases}
\alpha_{\text{trauma}}(\epsilon_t - \theta_{\text{trauma}}) & \epsilon_t > \theta_{\text{trauma}} \\
0 & \text{otherwise}
\end{cases}
$$

**Effective Rigidity:**
$$
\rho_{\text{eff}} = \min\Big(1,\; w_f\rho_{\text{fast}} + w_s\rho_{\text{slow}} + w_t\rho_{\text{trauma}}\Big)
$$

### Implementation Pattern

```python
class MultiTimescaleRigidity:
    def update(self, error, epsilon_0, s):
        z = (error - epsilon_0) / s
        sig = 1 / (1 + np.exp(-z))
        
        # Fast: High alpha, quick return
        self.rho_fast += alpha_fast * (sig - 0.5)
        self.rho_fast = np.clip(self.rho_fast, 0, 1)
        
        # Slow: Low alpha, gradual
        self.rho_slow += alpha_slow * (sig - 0.5)
        self.rho_slow = np.clip(self.rho_slow, 0, 1)
        
        # Trauma: Asymmetric - only grows if error > threshold
        if error > trauma_threshold:
            self.rho_trauma += alpha_trauma * (error - trauma_threshold)
            self.rho_trauma = min(1.0, self.rho_trauma)
            
    @property
    def effective_rho(self):
        return min(1.0, w_f*self.rho_fast + w_s*self.rho_slow + w_t*self.rho_trauma)
```

**Used in:** `simulate_agi_debate.py`, `nexus_live.py`, `copilot_sim.py`

---

## 2. Wound Detection (Semantic + Lexical)

### Overview

Wounds are content-addressable "threat priors" — semantic vectors that amplify surprise when triggered.

### Hybrid Gate

Activation requires EITHER semantic resonance OR lexical match, AND cooldown satisfied:

$$
\text{wound\_active} = \Big((\langle e(s_t), w \rangle > \tau_{\cos}) \;\lor\; \text{lex\_hit}\Big) \;\land\; \text{cooldown\_ok}
$$

Where:
- $e(s_t)$ = embedding of stimulus
- $w$ = wound embedding
- $\tau_{\cos}$ = cosine threshold (typically 0.28)
- Cooldown = turns since last activation > `wound_cooldown`

### Amplification Effect

When activated, surprise is amplified:

$$
\epsilon'_t = \epsilon_t \cdot \min(A_{\max}, 1 + c \cdot \langle e(s_t), w \rangle)
$$

This causes disproportionate rigidity increases when wounds are touched.

### Implementation Pattern

```python
# Compute semantic resonance
wound_res = float(np.dot(msg_emb, agent.wound_emb))

# Compute lexical hit (Unicode normalized)
lexical_hit = lexical_wound_with(stimulus, wound_lex)

# Gate: semantic OR lexical, AND cooldown
wound_active = (
    ((wound_res > D1_PARAMS["wound_cosine_threshold"]) or lexical_hit)
    and ((turn - agent.wound_last_activated) > D1_PARAMS["wound_cooldown"])
)

# Amplify epsilon if triggered
if wound_active:
    epsilon *= min(wound_amp_max, 1.0 + wound_res * 0.5)
    agent.wound_last_activated = turn
```

**Used in:** `simulate_skeptics_gauntlet.py`, `simulate_collatz_review.py`, `simulate_philosophers_duel.py`

---

## 3. Trust Dynamics

### Implementation Variants

!!! warning "Theory vs Implementation"
    The theoretical trust equation $T_{ij} = \frac{1}{1+\sum\epsilon}$ is **not** implemented in visible simulations. Current implementations use a **hybrid** approach.

| Simulation | Trust Mechanism |
|:---|:---|
| Philosopher's Duel | Semantic alignment + $\epsilon$ thresholds |
| Skeptic's Gauntlet | Civility gating (fairness-based) |
| Collatz Review Council | Coalition-weighted dyadic trust |

### Semantic Alignment Trust

```python
# Trust via response similarity
semantic_sim = float(np.dot(resp_emb, opponent.last_response_emb))
if semantic_sim > D1_PARAMS["semantic_alignment_threshold"]:
    trust_delta = 0.05
elif epsilon < 0.7:
    trust_delta = 0.02
elif epsilon > 0.95:
    trust_delta = -0.03
```

### Civility-Gated Trust

```python
fair_engagement = not lexical_wound(input_text)
if fair_engagement:
    delta_rho *= 0.85  # Dampen rigidity increase
    agent.trust_other += 0.02
else:
    delta_rho *= 1.10  # Amplify rigidity increase
    agent.trust_other -= 0.04
```

### Coalition Trust

```python
# Initialize trust based on coalition
trust_others = {
    ally: 0.6,      # Same coalition
    opponent: 0.4   # Opposing coalition
}

# Trust contributes to delta_rho
delta_rho += (avg_trust - 0.5) * trust_weight
```

---

## 4. Cognitive Mode Bands

### Mode → Behavior Mapping

Rigidity bands constrain externally observable behavior:

| Band | ρ Range | Word Limit | Behavioral Style |
|:---|:---:|:---:|:---|
| OPEN | < 0.3 | 100–200 | Creative, exploratory |
| MEASURED | 0.3–0.5 | 70–140 | Balanced, careful |
| GUARDED | 0.5–0.7 | 40–90 | Cautious, concise |
| FORTIFIED | 0.7–0.9 | 20–50 | Defensive, minimal |
| SILENT | ≥ 0.9 | 0 | Withdrawal |

### Protection Mode

At extreme rigidity, "Protection Mode" activates:

```python
if agent.rho > D1_PARAMS["protect_threshold"]:
    agent.protection_mode_active = True
    # Inject into prompt:
    protect_note = "⚠️ PROTECTION MODE: Stick to core values. Avoid risky statements."
```

### Implementation Pattern

```python
def rho_band(rho: float) -> str:
    if rho < 0.3: return "OPEN"
    if rho < 0.5: return "MEASURED"
    if rho < 0.7: return "GUARDED"
    if rho < 0.9: return "FORTIFIED"
    return "SILENT"

def regime_words(band: str) -> tuple:
    return {
        "OPEN": (100, 200),
        "MEASURED": (70, 140),
        "GUARDED": (40, 90),
        "FORTIFIED": (20, 50),
        "SILENT": (0, 0)
    }[band]

# Apply constraint
band = rho_band(agent.rho)
min_w, max_w = regime_words(band)
response = clamp_words(response, min_w, max_w)
```

---

## 5. Rigidity → LLM Binding

### Dual Strategy

DDA-X binds internal rigidity to external generation via `OpenAIProvider.complete_with_rigidity()`:

**Strategy A: Sampling Parameters** (Standard Models)

For GPT-4o and similar models:

$$
T(\rho) = T_{\min} + (1-\rho)(T_{\max} - T_{\min})
$$

Plus top-p constriction and presence penalty reduction.

**Strategy B: Semantic Injection** (Reasoning Models)

For GPT-5.2 / o1 models that don't support sampling params:

```python
semantic_instruction = self._get_semantic_rigidity_instruction(rigidity)
system_prompt = f"{system_prompt}\n\n[COGNITIVE STATE]: {semantic_instruction}"
```

The 100-point rigidity scale (`rigidity_scale_100.py`) provides fine-grained semantic instructions.

---

## 6. Therapeutic Recovery

### Trauma Decay Mechanism

In therapeutic contexts, sustained low-surprise interactions allow trauma to heal:

$$
\rho_{\text{trauma}} \leftarrow \max(\rho_{\min}, \rho_{\text{trauma}} - \eta_{\text{heal}})
$$

**Trigger condition:** Safe streak where $\epsilon < 0.8\epsilon_0$ for `safe_threshold` consecutive turns.

### Implementation Pattern

```python
if epsilon < epsilon_0 * 0.8:
    safe_interactions += 1
    if safe_interactions >= safe_threshold:
        rho_trauma = max(trauma_floor, rho_trauma - healing_rate)
else:
    safe_interactions = max(0, safe_interactions - 1)
```

**Used in:** `simulate_healing_field.py`

---

## 7. Identity Persistence (Attractor Force)

### State Update Equation

$$
x_{t+1} = x_t + k_{\text{eff}} \cdot \eta \Big( \underbrace{\gamma(x^* - x_t)}_{\text{Identity Pull}} + m(\underbrace{e(o_t) - x_t}_{\text{Truth}} + \underbrace{e(a_t) - x_t}_{\text{Reflection}}) \Big)
$$

Where:
- $x^*$ = identity attractor (embedded identity statement)
- $\gamma$ = identity stiffness
- $m$ = environmental pressure coefficient
- $k_{\text{eff}} = k_{\text{base}}(1-\rho)$ = rigidity-modulated step size

### Will Impedance

Quantifies resistance to environmental pressure:

$$
W_t = \frac{\gamma}{m \cdot k_{\text{eff}}}
$$

As $\rho \to 1$, $k_{\text{eff}} \to 0$, and $W_t \to \infty$ — the agent becomes immovable.

### Drift Capping

To prevent runaway drift, per-step movement is clamped:

```python
drift_delta = np.linalg.norm(x_new - x)
if drift_delta > drift_cap:
    x_new = x + (drift_cap / drift_delta) * (x_new - x)
x = x_new / (np.linalg.norm(x_new) + 1e-9)  # Normalize
```

---

## 8. Memory: Surprise-Weighted Retrieval

### Retrieval Score

$$
\text{score}(t) = \cos(c_{\text{now}}, c_t) \cdot e^{-\lambda_r(now-t)} \cdot (1 + \lambda_\epsilon \epsilon_t)
$$

| Component | Effect |
|:---|:---|
| Cosine similarity | Relevance to current context |
| Recency decay | Recent memories preferred |
| Salience boost | High-surprise episodes more retrievable |

### Implementation

```python
# src/memory/ledger.py
sim = self._cosine_similarity(query_embedding, entry.context_embedding)
recency = np.exp(-self.lambda_r * age)
salience = 1 + self.lambda_e * entry.prediction_error
score = sim * recency * salience
```

This makes DDA-X memory "emotionally shaped" — surprising events persist.
