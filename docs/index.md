# DDA-X: Surprise → Rigidity → Contraction
**A Dynamical Framework for Agent Behavior**

> **Standard RL:** surprise → exploration  
> **DDA-X:** **surprise → rigidity → contraction**

DDA-X is a cognitive-dynamics framework in which **prediction error (surprise) increases rigidity** (defensive contraction) rather than immediately driving exploration. Across **59 verified simulations**, agents maintain a continuous latent state (via 3072-D text embeddings), measure surprise as embedding-space prediction error, and bind that internal rigidity to externally visible behavior—constraining bandwidth, altering decoding styles, and injecting semantic "cognitive state" instructions.

This repository couples:

- **LLM reasoning** (GPT-5.2 / o1 family)
- **High-dimensional conceptual state** (`text-embedding-3-large`, 3072-D)

to produce measurable trajectories of openness, defensiveness, identity drift, wounds, trust, and recovery.

---

## The Core Insight

In standard Reinforcement Learning, surprise is often treated as an "intrinsic motivation" signal (curiosity) to explore. DDA-X inverts this. It models the behavior of organisms that **freeze and contract** when startled.

1. **Startle Response**: High prediction error ($\epsilon$) triggers a spike in rigidity ($\rho$).
2. **Contraction**: High rigidity reduces the "step size" of state updates ($k_{\text{eff}}$) and constrains the "bandwidth" of output (word counts, topic variance).
3. **Safety & Recovery**: Only when prediction error remains low for a sustained period does rigidity decay, allowing the system to reopen.

---

## Core Equations

### 1. Rigidity Update (Logistic Gate)

$$
z_t = \frac{\epsilon_t - \epsilon_0}{s}, \quad \Delta\rho_t = \alpha(\sigma(z_t) - 0.5)
$$

When $\epsilon_t > \epsilon_0$, rigidity increases. When $\epsilon_t < \epsilon_0$, rigidity decreases.

### 2. Effective Step Size

$$
k_{\text{eff}} = k_{\text{base}} (1 - \rho_t)
$$

Higher rigidity → smaller steps → more resistance to change.

### 3. State Evolution

$$
x_{t+1} = x_t + k_{\text{eff}} \cdot \eta \Big( \underbrace{\gamma(x^* - x_t)}_{\text{Identity Pull}} + m(\underbrace{e(o_t) - x_t}_{\text{Truth}} + \underbrace{e(a_t) - x_t}_{\text{Reflection}}) \Big)
$$

### 4. Multi-Timescale Rigidity

$$
\rho_{\text{eff}} = \min(1, \, w_f\rho_{\text{fast}} + w_s\rho_{\text{slow}} + w_t\rho_{\text{trauma}})
$$

---

## Key Mechanisms

| Mechanism | Description |
|:---|:---|
| **Multi-Timescale Rigidity** | Fast (startle), Slow (stress), Trauma (scarring) |
| **Wound Detection** | Semantic + lexical triggers with cooldown |
| **Mode Bands** | Rigidity → word budget constraints |
| **Therapeutic Recovery** | Trauma decay after sustained safety |
| **Identity Persistence** | Attractor force $\gamma(x^* - x)$ |

See [Mechanism Reference](architecture/mechanisms.md) for complete details.

---

## Infrastructure

### LLM Provider
**File:** `src/llm/openai_provider.py`

- Handles coupling between Rigidity ($\rho$) and LLM generation
- For reasoning models (o1/GPT-5.2): Injects semantic "Cognitive State" instructions
- For standard models: Modulates temperature/top_p sampling parameters
- Includes cost tracking for all API calls

### Experience Ledger
**File:** `src/memory/ledger.py`

- Implements **Surprise-Weighted Memory**
- Retrieval score: $\text{sim} \times \text{recency} \times \text{salience}$
- Salience scales with prediction error: $(1 + \lambda_\epsilon \cdot \epsilon_t)$
- High-surprise episodes remain more retrievable

---

## Explore the Documentation

| Page | Description |
|:---|:---|
| [Paper v2.0](architecture/paper.md) | Full theoretical framework with rigorous math |
| [Implementation Mapping](architecture/ARCHITECTURE.md) | How theory runs in actual Python code |
| [Mechanism Reference](architecture/mechanisms.md) | Complete mechanism documentation |
| [Simulation Chronology](simulation_chronology.md) | Catalog of all 59 simulations |
| [Unique Contributions](unique_contributions.md) | What makes DDA-X novel |
| [Known Limitations](limitations.md) | Honest critical assessment |
| [GPT-5.2 Review](gpt52_review/gpt52_feedback_final.md) | Independent review by reasoning models |

---

## Quick Start

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run the pinnacle debate simulation
python simulations/simulate_agi_debate.py

# Run the therapeutic recovery test
python simulations/simulate_healing_field.py

# Run the real-time Pygame visualization
python simulations/nexus_live.py
```

---

## Unique Contributions

1. **Rigidity as Control Variable**: Explicit $k_{\text{eff}} = k_{\text{base}}(1-\rho)$
2. **Inverted Exploration**: Surprise → Contraction (not curiosity)
3. **Wounds as Threat Priors**: Content-addressable semantic triggers
4. **Multi-Timescale Defensiveness**: Fast/Slow/Trauma decomposition
5. **Therapeutic Recovery**: Explicit trauma decay dynamics
6. **Identity as Attractor**: Dynamical systems framing for persistence

See [Unique Contributions](unique_contributions.md) for detailed comparison with standard RL/LLM agents.

---

*(c) 2025 DDA-X Research Team*
