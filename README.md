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
| **39** | `simulate_collective.py` | **The Collective:** 4 specialized agents, trust deltas with causes. |
| **38** | `simulate_crucible.py` | **The Crucible:** Identity stress test for single agent (VERITY). |
| **37** | `copilot_sim.py` | **Copilot Sim:** One-shot experiment, Multi-Timescale Rigidity + Local Ledger. |
| **36** | `simulate_rigidity_gradient.py` | **Rigidity Gradient:** Validates 100-point semantic scale on GPT-5.2. |
| **35** | `simulate_identity_siege.py` | **Identity Siege:** Hierarchical identity (Core/Persona/Role) with differential stiffness. |
| **34** | `simulate_wounded_healers.py` | **Wounded Healers:** Countertransference, trauma profiles, healing verification. |
| **33** | `solve_collatz.py` | **Solve Collatz:** Tool use (SymPy), low rigidity, rigorous proof attempt. |
| **32** | `simulate_gpt52_society.py` | **GPT-5.2 Society:** High-fidelity "Cognitive Mirror" simulation. |
| **31** | `simulate_sherlock.py` | **Sherlock Society:** Detective agents solving mysteries with Deductive Grader. |
| **30** | `simulate_math_team.py` | **Math Team:** Collaborative solving (Solver, Checker, Grader). |
| **29** | `simulate_problem_solver.py` | **Problem Solver:** 6-agent society solving logic puzzles. |
| **28** | `simulate_society.py` | **The Society:** Discord-style multi-agent chat, basic D1 physics. |
| **27** | `simulate_npc_conversation.py` | **NPC Conversation:** Unscripted interaction driven by Identity Pull. |
| **26** | `simulate_mole_hunt.py` | **Mole Hunt:** Deception detection, conflicting identity hierarchy. |
| **25** | `simulate_logic_solver.py` | **Logic Solver:** Iterative reasoning ("Who Owns the Zebra?") via Ledger. |
| **24** | `simulate_iterative_learning.py` | **Iterative Learning:** Alien language acquisition via Reflection loop. |
| **23** | `simulate_insight_engine.py` | **Insight Engine:** Recursive insight accumulation (Working Memory). |
| **22** | `simulate_goal_learning.py` | **Goal Learning:** Exploration vs Exploitation adaptation. |
| **21** | `simulate_gamma_threshold.py` | **Gamma Threshold:** Phase transition testing (Identity Stiffness). |
| **20** | `simulate_empathy_paradox.py` | **Empathy Paradox:** Logic vs Empathy drift measurement. |
| **19** | `simulate_deceptive_env.py` | **Deceptive Env:** Intelligence amplification against noisy feedback. |
| **18** | `simulate_closed_loop.py` | **Closed Loop:** Full Embed-Force-Evolve-Retrieve-Respond loop. |
| **17** | `simulate_paper_mechanics.py` | **Paper Mechanics:** Explicit visualization of framework math. |
| **16** | `simulate_stress_magic.py` | **Stress Magic:** Existential paradox injection (Chaos Mode trigger). |
| **15** | `simulate_neural_link.py` | **Neural Link:** Real-time Operator vs Subject (Glass Box monitoring). |
| **14** | `simulate_glass_box.py` | **Glass Box:** Real-time breakdown of cognitive cycle stages. |
| **13** | `simulate_dual_yklam.py` | **Dual YKLAM:** "The Mirror Room" - divergent instances of same persona. |
| **12** | `simulate_auto_yklam.py` | **Auto YKLAM:** Natural simulation with variable plasticity. |
| **11** | `simulate_yklam.py` | **YKLAM:** Soulful Proxy with "Soul Telemetry" visualization. |
| **10** | `simulate_connect4_duel.py` | **Connect 4 Duel:** Competitive game agents (MCTS + Memory). |
| **9** | `verify_dda_physics.py` | **Physics Verification:** Testing Rigidity $\to$ Temp mapping. |
| **8** | `simulate_socrates.py` | **Socratic Asymmetry:** Dogmatist vs Gadfly (High vs Low Gamma). |
| **7** | `simulate_schism.py` | **The Schism:** Trust collapse driving rigidity (Live API). |
| **6** | `simulate_redemption.py` | **Redemption Arc:** Corrupted agent recovery via Deprogrammer. |
| **5** | `simulate_infinity.py` | **Infinity:** Infinite dialectic loop (" The Flame War"). |
| **4** | `simulate_driller.py` | **Deep Driller:** Forensic root cause analysis (Rigidity vs Plasticity). |
| **3** | `simulate_discord.py` | **Discord:** Data-driven priming from logs. |
| **2** | `simulate_corruption.py` | **Corruption:** "Boiling the Frog" identity shift. |
| **1** | `demo.py` | **Demo:** Standalone Mechanics demonstration. |

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
