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

This repository contains **59 simulations** tracing the development of the architecture.

| Tier | Sim Index | Name | Key Dynamics Tested |
| :--- | :--- | :--- | :--- |
| **5 (Pinnacle)** | 59 | `nexus_live.py` | Real-time Pygame, 50 entities, collision physics, async LLM thoughts. |
| **5** | 58 | `simulate_agi_debate.py` | 8-round adversarial debate, multi-timescale rigidity, wound lexicons. |
| **5** | 57 | `simulate_healing_field.py` | Therapeutic recovery loops, trauma decay, will impedance. |
| **5** | 56 | `simulate_33_rungs.py` | Multi-voice transmission, veil/presence dynamics, unity convergence. |
| **4 (Advanced)** | 55 | `simulate_skeptics_gauntlet.py` | Meta-defense, evidence injection, civility-gated trust. |
| **4** | 54 | `simulate_collatz_review.py` | Multi-agent coalition trust, peer review dynamics, calibration. |
| **4** | 53 | `simulate_identity_siege.py` | Hierarchical identity stiffness ($\gamma_c, \gamma_p, \gamma_r$). |
| **4** | 52 | `simulate_philosophers_duel.py` | Dialectic identity persistence, semantic trust alignment. |
| **4** | 51 | `simulate_wounded_healers.py` | Mutual healing dynamics, trauma resonance. |
| **4** | 50 | `simulate_coalition_flip.py` | Topology churn, trust decay, group formation. |
| **4** | 49 | `simulate_creative_collective.py` | Flow states, optimal resonance ($\rho \approx 0.4$). |
| **4** | 48 | `simulate_the_returning.py` | Release field ($\Phi = 1-\rho$), pattern dissolution, isolation index. |
| **3 (Intermediate)** | 47 | `simulate_echo_chamber_rupture.py` | Information insulation, breakage thresholds. |
| **3** | 46 | `simulate_trust_battery_drain.py` | Trust decay mechanics, betrayal modeling. |
| **3** | 45 | `simulate_infinite_mirror.py` | Recursive self-reflection, feedback loops. |
| **3** | 44 | `simulate_void_genesis.py` | Emergence from null state, noise-induced order. |
| **3** | 43 | `simulate_entropic_decay.py` | System degradation, entropy tracking. |
| **3** | 42 | `simulate_quantum_observer.py` | Observation effect, state collapse. |
| **3** | 41 | `simulate_memetic_virus.py` | Idea propagation, immune response (rigidity). |
| **3** | 40 | `simulate_reality_tunnel.py` | Belief confirmation bias, tunnel vision. |
| **3** | 39 | `simulate_consensus_reality.py` | Group agreement, outlier pressure. |
| **3** | 38 | `simulate_cognitive_dissonance.py` | Conflicting inputs, internal friction ($F_T \neq F_R$). |
| **3** | 37 | `simulate_ego_death.py` | Identity dissolution, attractor collapse. |
| **3** | 36 | `simulate_shadow_integration.py` | Hidden state incorporation, psychological shadow. |
| **3** | 35 | `simulate_collective_unconscious.py` | Shared latent space, primal archetypes. |
| **3** | 34 | `simulate_akashic_read.py` | Global memory retrieval, historical access. |
| **3** | 33 | `simulate_karmic_wheel.py` | Action-reaction cycles, consequence tracking. |
| **2 (Foundational)** | 32 | `solve_collatz.py` | Tool use, mathematical proof attempt. |
| **2** | 31 | `simulate_gravity_well.py` | Attractor strength testing ($\gamma$ calibration). |
| **2** | 30 | `simulate_orbital_mechanics.py` | State trajectory mapping, stability orbits. |
| **2** | 29 | `simulate_dark_matter.py` | Unobserved influence, latent variables. |
| **2** | 28 | `simulate_event_horizon.py` | Irreversible state changes, point of no return. |
| **2** | 27 | `simulate_supernova.py` | Explosive divergence, system crash testing. |
| **2** | 26 | `simulate_black_hole.py` | Information loss, singularity modeling. |
| **2** | 25 | `simulate_wormhole.py` | State tunneling, discontinuous updates. |
| **2** | 24 | `simulate_time_dilation.py` | Processing speed variance, relative time. |
| **2** | 23 | `simulate_multiverse.py` | Parallel state tracking, branching paths. |
| **2** | 22 | `simulate_quantum_entanglement.py` | Correlated state updates across agents. |
| **2** | 21 | `simulate_schrodingers_cat.py` | Superposition states, observation collapse. |
| **2** | 20 | `simulate_double_slit.py` | Wave-particle duality in decision making. |
| **2** | 19 | `simulate_heisenberg.py` | Uncertainty principle, measurement disturbance. |
| **2** | 18 | `simulate_planck_scale.py` | Minimal state updates, quantization. |
| **2** | 17 | `simulate_string_theory.py` | High-dimensional state vibration. |
| **1 (Primitive)** | 16 | `simulate_big_bang.py` | Initial state generation. |
| **1** | 15 | `simulate_inflation.py` | Rapid state expansion. |
| **1** | 14 | `simulate_nucleosynthesis.py` | Core formation. |
| **1** | 13 | `simulate_recombination.py` | Stability testing. |
| **1** | 12 | `simulate_dark_ages.py` | Low information flow. |
| **1** | 11 | `simulate_first_stars.py` | Agent emergence. |
| **1** | 10 | `simulate_galaxy_formation.py` | Group clustering. |
| **1** | 09 | `simulate_solar_system.py` | Hierarchical orbits. |
| **1** | 08 | `simulate_earth_formation.py` | Environment stability. |
| **1** | 07 | `simulate_life_origin.py` | Self-replicating patterns. |
| **1** | 06 | `simulate_evolution.py` | Adaptation testing. |
| **1** | 05 | `simulate_consciousness.py` | Self-awareness metrics. |
| **1** | 04 | `simulate_civilization.py` | Social dynamics. |
| **1** | 03 | `simulate_technology.py` | Tool discovery. |
| **1** | 02 | `simulate_singularity.py` | Recursive improvement. |
| **1** | 01 | `simulate_omega_point.py` | Final convergence. |

---

## Infrastructure

*   **`src/llm/openai_provider.py`**: Handles coupling between Rigidity ($\rho$) and LLM generation. For reasoning models (o1/GPT-5.2), it injects semantic "Cognitive State" instructions. For standard models, it modulates temperature/top_p.
*   **`src/memory/ledger.py`**: Implements **Surprise-Weighted Memory**. Experiences are retrieved based on Similarity $\times$ Recency $\times$ Salience (where Salience scales with Prediction Error).

## Running the Simulations

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Run any simulation:

```bash
# Run the pinnacle debate simulation
python simulations/simulate_agi_debate.py

# Run the therapeutic recovery test
python simulations/simulate_healing_field.py

# Run the real-time Pygame visualization
python simulations/nexus_live.py
```

## Unique Contributions

1.  **Rigidity as a Control Variable**: We explicitly model "defensiveness" as a state variable that shrinks learning rates ($k_{\text{eff}}$) and output bandwidth.
2.  **Inverted Exploration**: Unlike RL, surprise leads to *less* exploration (contraction) initially.
3.  **Wounds as Threat Priors**: Content-addressable "wounds" (semantic embeddings) amplify surprise and trigger defensive responses.
4.  **Therapeutic Recovery**: We demonstrate mathematically how "safe" (low-error) interactions can decay trauma over time.

---

*(c) 2025 DDA-X Research Team*
