# DDA-X Architecture: From Theory to Implementation

This document maps the DDA-X theoretical components to the concrete **verified code patterns** used across the simulations.

> **Note:** Examples below are taken from the verified simulation files (e.g., AGI Debate, Healing Field, Nexus, Skeptics Gauntlet).

---

## 1. Core Dataflow Per Turn

A typical simulation turn follows this specific sequence:

1.  **Embed**: Convert stimulus $o_t$ to `msg_emb`.
2.  **Wound Check**: (Optional) Check semantic/lexical triggers $\to$ modulate $\epsilon$.
3.  **Generate**: Call `OpenAIProvider.complete_with_rigidity(..., rho)`.
4.  **Embed Response**: Convert response $a_t$ to `resp_emb`.
5.  **Compute Error**: $\epsilon_t = \|x_{\text{pred}} - \text{resp\_emb}\|$.
6.  **Update Rigidity**: $\Delta \rho$ via logistic gate (single or multi-timescale).
7.  **Update State**: $x_{t+1}$ moves toward $x^*$ and observation, throttled by $k_{\text{eff}}$.
8.  **Log**: Write `LedgerEntry` to `ExperienceLedger`.

---

## 2. Embedding + Normalization Phase

**Pattern**: All vectors are normalized to the hypersphere.

```python
msg_emb = await provider.embed(stimulus)
msg_emb = msg_emb / (np.linalg.norm(msg_emb) + 1e-9)
```

**File(s)**: `simulate_agi_debate.py`, `simulate_skeptics_gauntlet.py`.

---

## 3. Prediction Error Engine

**Pattern**: Euclidean distance in 3072-D space. The prediction Vector $x_{\text{pred}}$ is an EMA of *own* output.

```python
epsilon = float(np.linalg.norm(agent.x_pred - resp_emb))
agent.x_pred = 0.7 * agent.x_pred + 0.3 * resp_emb
```

**File(s)**: `simulate_the_returning.py`, `simulate_philosophers_duel.py`.

---

## 4. Rigidity Update (Logistic Gate)

**Pattern**: The core "Expansion/Contraction" mechanism.

```python
z = (epsilon - params["epsilon_0"]) / params["s"]
sig = sigmoid(z)
delta_rho = params["alpha"] * (sig - 0.5)
agent.rho = np.clip(agent.rho + delta_rho, 0.0, 1.0)
```

**Common Modifiers**:
*   **Civility**: `delta_rho -= 0.02` if polite, `+= 0.04` if rude (Skeptics Gauntlet).
*   **Trust**: `delta_rho -= trust_factor` (Collatz Review).

---

## 5. Multi-Timescale Rigidity Module

**Pattern**: Decomposition into Fast/Slow/Trauma components.

```python
class MultiTimescaleRigidity:
    def update(self, error, ...):
        # Fast: High alpha, quick return
        self.rho_fast += alpha_fast * (sig - 0.5)
        
        # Trauma: Asymmetric - only grows if error > threshold
        if error > trauma_threshold:
            self.rho_trauma += alpha_trauma * (error - threshold)
            
    @property
    def effective_rho(self):
        return w_f*self.rho_fast + w_s*self.rho_slow + w_t*self.rho_trauma
```

**File(s)**: `simulate_agi_debate.py`, `nexus_live.py`.

---

## 6. Wound Detection (Hybrid)

**Pattern**: Semantic Resonance + Lexical Trigger + Cooldown.

```python
wound_res = float(np.dot(msg_emb, agent.wound_emb))
lexical_hit = lexical_wound_with(stimulus, wound_lex)

wound_active = (
    (wound_res > params["wound_cosine_threshold"] or lexical_hit)
    and (turn - agent.wound_last_activated) > params["wound_cooldown"]
)
```

**Effect**: Amplifies Epsilon.

```python
if wound_active:
    epsilon *= min(params["wound_amp_max"], 1.0 + wound_res * 0.5)
```

**File(s)**: `simulate_skeptics_gauntlet.py`, `simulate_collatz_review.py`.

---

## 7. Binding Rigidity to LLM Generation

**Pattern**: `OpenAIProvider.complete_with_rigidity()`.

**Strategy A (Reasoning Models - o1/GPT-5.2)**: Semantic Injection.
```python
# src/llm/openai_provider.py
semantic_instruction = self._get_semantic_rigidity_instruction(rigidity)
# e.g. "You are DEFENSIVE. Adhere strictly to priors. Output < 50 words."
system_prompt = f"{system_prompt}\n\n[COGNITIVE STATE]: {semantic_instruction}"
```

**Strategy B (Standard Models - GPT-4o)**: Sampling Parameter Modulation.
```python
# Modulates temperature based on rigidity
temperature = T_min + (1 - rho) * (T_max - T_min)
```

---

## 8. State Update & Drift Cap

**Pattern**: Movement throttled by Rigidity ($k_{\text{eff}}$), with stability caps.

```python
k_eff = k_base * (1 - rho)
x_new = x + k_eff * (gamma*(id - x) + m*(obs - x))

# Drift Cap
drift = norm(x_new - x)
if drift > drift_cap:
    x_new = x + (drift_cap / drift) * (x_new - x)
```

**File(s)**: `simulate_the_returning.py`, `simulate_identity_siege.py`.

---

## 9. Therapeutic Recovery (Trauma Decay)

**Pattern**: Explicit decay of $\rho_{\text{trauma}}$ after sustained safe interactions.

```python
if epsilon < epsilon_0 * 0.8:
    safe_streak += 1
    if safe_streak > threshold:
        rho_trauma = max(floor, rho_trauma - healing_rate)
else:
    safe_streak = 0
```

**File(s)**: `simulate_healing_field.py`.

---

## 10. Memory: Experience Ledger

**Pattern**: Surprise-Weighted Retrieval.

```python
# src/memory/ledger.py
score = similarity * recency * salience
salience = 1 + lambda_e * entry.prediction_error
```

Data is stored as `LedgerEntry` (numpy vectors) and `ReflectionEntry` (LLM lessons), serialized to `.pkl.xz`.

---

## 11. Simulation-Specific Architectures

### The Returning
*   **Release Field**: $\Phi = 1 - \rho$. Used to model "letting go".
*   **Isolation Index**: Mean distance from "PRESENCE" agent.

### Identity Siege
*   **Hierarchical Identity**: 3 Layers (Core, Persona, Role) with distinct stiffness $\gamma$.
*   $F_{\text{id}} = \gamma_c(x^*_c - x) + \gamma_p(x^*_p - x) + \gamma_r(x^*_r - x)$.

### Collatz Review Council
*   **Dyadic Trust**: Trust map `trust[advocacy_group]` modulates $\Delta \rho$.
*   **Calibration**: Calculates $\epsilon_0$ and $s$ dynamically from early-run statistics.
