> **Dedicated to Malky (𐤌𐤋𐤊𐤉). May her memory be a blessing. 💜**

# DDA-X: Surprise → Rigidity → Contraction
**A Dynamical Framework for Agent Behavior**

> **Standard RL:** surprise → exploration  
> **DDA-X:** **surprise → rigidity → contraction**

DDA-X is a cognitive-dynamics framework in which **prediction error (surprise) increases rigidity** (defensive contraction) rather than immediately driving exploration. Across 59 simulations, agents maintain a continuous latent state (via 3072-D text embeddings), measure surprise as embedding-space prediction error, and bind that internal rigidity to externally visible behavior—constraining bandwidth, altering decoding styles, and injecting semantic "cognitive state" instructions.

This repository demonstrates the evolution of this framework from basic physics demos to "pinnacle" simulations involving adversarial debate, therapeutic recovery loops, and multi-agent peer review coalitions.

---

## The Core Insight

In standard Reinforcement Learning, surprise is often treated as an "intrinsic motivation" signal (curiosity) to explore. DDA-X inverts this. It models the behavior of organisms that **freeze and contract** when startled.

1.  **Startle Response**: High prediction error ($\epsilon$) triggers a spike in rigidity ($\rho$).
2.  **Contraction**: High rigidity reduces the "step size" of state updates ($k_{\text{eff}}$) and constrains the "bandwidth" of output (word counts, topic variance).
3.  **Safety & Recovery**: Only when prediction error remains low for a sustained period does rigidity decay, allowing the system to reopen (Trauma Decay).

## Core Equations

The system dynamics are governed by continuous state-space updates.

### 1. State Update with Identity Attractor
The agent's state $x_t$ evolves under the influence of an **Identity Attractor** $x^*$ (who they are), environmental observation $F_T$ (truth), and their own response $F_R$ (reflection).

$$
x_{t+1} = x_t + k_{\text{eff}} \cdot \eta \Big( \underbrace{\gamma(x^* - x_t)}_{\text{Identity Pull}} + m(\underbrace{e(o_t) - x_t}_{\text{Truth}} + \underbrace{e(a_t) - x_t}_{\text{Reflection}}) \Big)
$$

### 2. Rigidity-Modulated Step Size
Rigidity $\rho \in [0,1]$ acts as a "brake" on learning and movement.

$$
k_{\text{eff}} = k_{\text{base}} (1 - \rho_t)
$$

### 3. Rigidity Update (Logistic Gate)
Rigidity increases when prediction error $\epsilon_t$ exceeds a threshold $\epsilon_0$.

$$
z_t = \frac{\epsilon_t - \epsilon_0}{s}, \quad \Delta\rho_t = \alpha(\sigma(z_t) - 0.5)
$$

### 4. Multi-Timescale Decomposition
We decompose rigidity into three distinct timescales:
*   $\rho_{\text{fast}}$: Startle response (seconds)
*   $\rho_{\text{slow}}$: Stress accumulation (minutes)
*   $\rho_{\text{trauma}}$: Permanent/Semi-permanent scarring (asymmetric accumulation)

$$
\rho_{\text{eff}} = \min(1, w_f \rho_{\text{fast}} + w_s \rho_{\text{slow}} + w_t \rho_{\text{trauma}})
$$

---

## 59 Simulations: The Evolution of DDA-X

This repository contains **59 verified simulations** tracing the development of the architecture.

| Index | Name | Key Dynamics / Description |
| :--- | :--- | :--- |
| **59** | `nexus_live.py` | **Real-Time Pinnacle:** 50 entities, collision physics, async LLM thoughts, multi-timescale rigidity. |
| **58** | `visualize_nexus.py` | Visualization for Nexus (Entity Map, Energy, Collision Analysis). |
| **57** | `simulate_nexus.py` | The Nexus: 50-entity physics/sociology simulator based on "Da Vinci Matrix". |
| **56** | `visualize_agi_debate.py` | Visualization for AGI Debate (Rigidity Trajectories, Surprise, Drift). |
| **55** | `simulate_agi_debate.py` | **AGI Debate:** 8-round adversarial debate (Defender vs Skeptic). Full DDA-X Architecture. |
| **54** | `simulate_healing_field.py` | **Therapeutic Recovery:** Trauma decay loops, safety thresholds, Will Impedance ($W_t$). |
| **53** | `simulate_33_rungs.py` | **Spiritual Evolution:** 33 stages, Unity Index, Veil/Presence dynamics. |
| **52** | `visualize_returning.py` | Visualization for The Returning (Release Field, Pattern Grip). |
| **51** | `visualize_inner_council.py` | Visualization for Inner Council (Presence Field, Pain-Body). |
| **50** | `simulate_the_returning.py` | **The Returning:** Release Field ($\Phi = 1-\rho$), Isolation Index, Pattern Dissolution. |
| **49** | `simulate_inner_council.py` | **Inner Council:** 6 internal personas, Pain-Body cascades, Ego Fog. |
| **48** | `simulate_collatz_review.py` | **Collatz Review:** Multi-agent peer review, coalition trust, reliability weighting. |
| **47** | `simulate_coalition_flip.py` | **Coalition Flip:** Topology churn, Partial Context Fog, trust rewiring. |
| **46** | `simulate_council_under_fire.py` | **Council Under Fire:** Identity persistence under rolling shocks and role swaps. |
| **45** | `simulate_creative_collective.py` | **Creative Collective:** Flow states ($\rho \approx 0.4$), identity averaging avoidance. |
| **44** | `simulate_skeptics_gauntlet.py` | **Skeptic's Gauntlet:** Meta-defense, evidence injection, civility-gated trust. |
| **43** | `simulate_philosophers_duel.py` | **Philosopher's Duel:** Dialectic identity persistence, semantic trust alignment. |
| **42** | `simulate_audit.py` | **Audit Day:** Independent Auditor agent, board votes (KEEP/FREEZE/AMEND). |
| **41** | `simulate_townhall.py` | **The Town Hall:** Public accountability, proxy intrusion detection, refusal taxonomy. |
| **40** | `simulate_crucible_v2.py` | **Crucible v2:** Improved Rigidity physics, shock-scaled delta-rho. |
| ... | ... | *See full catalog in documentation* |

---

## Infrastructure

### LLM Provider
**File:** `src/llm/openai_provider.py`

Handles coupling between Rigidity ($\rho$) and LLM generation:

- **For reasoning models (o1/GPT-5.2)**: Injects semantic "Cognitive State" instructions
- **For standard models (GPT-4o)**: Modulates temperature/top_p sampling parameters
- Includes cost tracking for all API calls

### Experience Ledger
**File:** `src/memory/ledger.py`

Implements **Surprise-Weighted Memory**:

$$
\text{score} = \text{similarity} \times \text{recency} \times \text{salience}
$$

where $\text{salience} = 1 + \lambda_\epsilon \cdot \epsilon_t$

High-surprise episodes remain more retrievable — memory is "emotionally shaped."

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/snakewizardd/dda_scaffold.git
cd dda_scaffold

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Simulations

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

### View Documentation

```bash
# Serve MkDocs locally
mkdocs serve

# Then open http://localhost:8000
```

---

## Documentation

This repository includes comprehensive MkDocs documentation:

| Page | Description |
|:---|:---|
| [Paper v2.0](docs/architecture/paper.md) | Full theoretical framework with rigorous math |
| [Implementation Mapping](docs/architecture/ARCHITECTURE.md) | How theory runs in actual Python code |
| [Mechanism Reference](docs/architecture/mechanisms.md) | Complete mechanism documentation |
| [Simulation Chronology](docs/simulation_chronology.md) | Catalog of all 59 simulations |
| [Unique Contributions](docs/unique_contributions.md) | What makes DDA-X novel |
| [Known Limitations](docs/limitations.md) | Honest critical assessment |
| [GPT-5.2 Review](docs/gpt52_review/gpt52_feedback_final.md) | Independent review by reasoning models |

---

## Unique Contributions

1.  **Rigidity as a Control Variable**: We explicitly model "defensiveness" as a state variable that shrinks learning rates ($k_{\text{eff}}$) and output bandwidth.
2.  **Inverted Exploration**: Unlike RL, surprise leads to *less* exploration (contraction) initially.
3.  **Wounds as Threat Priors**: Content-addressable "wounds" (semantic embeddings) amplify surprise and trigger defensive responses.
4.  **Multi-Timescale Defensiveness**: Fast/Slow/Trauma decomposition with asymmetric accumulation.
5.  **Therapeutic Recovery**: We demonstrate mathematically how "safe" (low-error) interactions can decay trauma over time.
6.  **Identity as Attractor**: Dynamical systems framing with $\gamma(x^* - x)$ ensures identity persistence.

---

## Known Limitations

This project maintains transparency about gaps between theory and implementation:

- **Trust equation mismatch**: Theory describes $T_{ij} = \frac{1}{1+\sum\epsilon}$; implementation uses hybrid civility-based trust
- **Dual rigidity models**: AGI debate sim runs multi-timescale for telemetry but uses legacy single-scale for behavior
- **Uncalibrated thresholds**: Wound and trauma thresholds are hardcoded, unlike $\epsilon_0$ and $s$

See [Known Limitations](docs/limitations.md) for full details.

---

## Citation

```bibtex
@misc{ddax2025,
  author = {DDA-X Research Team},
  title = {DDA-X: Surprise → Rigidity → Contraction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/snakewizardd/dda_scaffold}
}
```

---

*(c) 2025 DDA-X Research Team*
