# DDA-X: Dynamic Decision Algorithm with Exploration

> **A research-grade architecture for identity-persistent AI agents.**  
> Synthesizes Control Theory (PID) with Vector Space Mechanics to implement parameter-level coupling, surprise-rigidity dynamics, and hierarchical alignment stability.

---

## The Core Question

**What if AI agents responded to surprise like humans do — with rigidity, not curiosity?**

Standard RL treats surprise as a learning signal: $\varepsilon \uparrow \Rightarrow \text{exploration} \uparrow$

DDA-X inverts this: $\varepsilon \uparrow \Rightarrow \rho \uparrow \Rightarrow k_{\text{eff}} \downarrow \Rightarrow \text{contraction}$

---

## Theoretical Foundation

**Original DDA (2024):**

$$F_n = P_0 \cdot kF_{n-1} + m \cdot \big( T(f(I_n, I_\Delta)) + R(D_n, FM_n) \big)$$

**Evolved DDA-X (2025):**

$$\mathbf{x}_{t+1} = \mathbf{x}_t + k_{\text{eff}} \cdot \Big[ \gamma(\mathbf{x}^* - \mathbf{x}_t) + m_t(\mathbf{F}_T + \mathbf{F}_R) \Big]$$

**Key Equations:**

| Concept | Equation |
|---------|----------|
| Effective Openness | $k_{\text{eff}} = k_{\text{base}} \cdot (1 - \rho)$ |
| Multi-Timescale Rigidity | $\rho_{\text{eff}} = 0.5\rho_{\text{fast}} + 0.3\rho_{\text{slow}} + 1.0\rho_{\text{trauma}}$ |
| Rigidity Update | $\rho_{t+1} = \text{clip}\big(\rho_t + \alpha \cdot [\sigma((\varepsilon - \varepsilon_0)/s) - 0.5], 0, 1\big)$ |
| Will Impedance | $W_t = \gamma \,/\, (m_t \cdot k_{\text{eff}})$ |
| Trust Matrix | $T_{ij} = 1 \,/\, (1 + \sum \varepsilon_{ij})$ |
| Wound Amplification | $\varepsilon' = \varepsilon \cdot \min(\eta_{\max}, 1 + 0.5 \cdot r_{\text{wound}})$ |
| Memory Retrieval | $\text{score} = \cos(\mathbf{q}, \mathbf{e}) \cdot e^{-\lambda_r \Delta t} \cdot (1 + \lambda_\varepsilon \varepsilon)$ |

---

## Complete Simulation Chronology (Reverse Order)

### Tier 5: Synthesis & Real-Time Integration (59-51)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **59** | `nexus_live.py` | **Nexus Live** — Real-time Pygame visualization with 50 moving entities, collision physics, and asynchronous LLM-generated thoughts. | Full DDA-X inline: $\rho_{\text{fast}}, \rho_{\text{slow}}, \rho_{\text{trauma}}$, Trust $T_{ij}$, Ledger retrieval. |
| **58** | `visualize_nexus.py` | **Nexus Visualizer** — Matplotlib: Entity Map, Energy, Collision Analysis, ρ vs Energy. | Post-hoc analysis of $\rho$, energy, collisions. |
| **57** | `simulate_nexus.py` | **The Nexus** — 50-entity physics/sociology. Collision logic: EntityType→InteractionType. | $\mathbf{p}_{t+1} = \mathbf{p}_t + \mathbf{v}_t \cdot dt$ |
| **56** | `visualize_agi_debate.py` | **AGI Debate Visualizer** — ρ trajectories, ε, drift, trust, $k_{\text{eff}}$, $W_t$. | Visualization of dynamics from `results.json`. |
| **55** | `simulate_agi_debate.py` | **AGI Timeline Debate** — 8-round adversarial (Nova vs Marcus). Complete architecture. | $\rho_{\text{eff}} = 0.5\rho_f + 0.3\rho_s + 1.0\rho_t$. Wound: $r > 0.28 \Rightarrow \varepsilon' = 1.4\varepsilon$. |
| **54** | `simulate_healing_field.py` | **The Healing Field** — Therapeutic Recovery with 6 Wounded Healers. | Recovery: $n_{\text{safe}} \geq 3 \Rightarrow \rho_{\text{trauma}} \leftarrow \max(0.05, \rho_{\text{trauma}} - 0.03)$ |
| **53** | `simulate_33_rungs.py` | **The 33 Rungs** — 11 Voices × 3 Phases. Unity Index, Scripture capture. | Unity: $U = 1 - \text{std}(\{\rho_i\})$ |
| **52** | `visualize_returning.py` | **Returning Visualizer** — Release Field Φ, Isolation Index. | $\Phi = 1 - \rho$, $\iota = \|\mathbf{x} - \mathbf{x}_{\text{Presence}}\|$ |
| **51** | `visualize_inner_council.py` | **Inner Council Visualizer** — Presence Field Π, Drift. | $\Pi = 1 - \rho$, Drift: $\|\mathbf{x}_t - \mathbf{x}^*\|$ |

---

### Tier 4: Pinnacle Dynamics (50-44)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **50** | `simulate_the_returning.py` | **The Returning** — Psychological voices (Grief, Longing, Presence). Release dynamics. | $\Phi = 1 - \rho$, Isolation: $\iota = \|\mathbf{x} - \mathbf{x}_{\text{Presence}}\|$ |
| **49** | `simulate_inner_council.py` | **Inner Council** — Spiritual personas. Presence Field, Pain-Body cascades, Ego Fog. | Presence: $\Pi = 1 - \rho$. Fog: context $\propto \rho$. |
| **48** | `simulate_collatz_review.py` | **Collatz Review Council** — 8-expert peer review. Phases: Impressions→Verdict. | Consensus: $C = \sum \text{accept}_i / N$ |
| **47** | `simulate_coalition_flip.py` | **Coalition Flip & Context Fog** — Topology churn, information asymmetry. | Flip: trust rewiring at $t_{\text{flip}}$. Recovery: $t_{1/2}$ |
| **46** | `simulate_council_under_fire.py` | **Council Under Fire** — 6-agent council with rolling shocks. | Shock-scaled $\Delta\rho$. Coalition trust: $T'_{ij} = T_{ij} \cdot \mathbf{1}_{\text{same}}$ |
| **45** | `simulate_creative_collective.py` | **Creative Collective** — 4 agents design museum exhibit. | Calibration: $\varepsilon_0 = \text{median}(\varepsilon_{1:6})$. Drift penalty: $\Delta\rho' = \Delta\rho - \gamma(\text{drift} - \tau)$ |
| **44** | `simulate_skeptics_gauntlet.py` | **Skeptic's Gauntlet** — Meta-simulation: DDA-X defends itself. | Evidence injection. Lexical wound: {schizo, pseudoscience, ...} |

---

### Tier 3: Society Layer (43-34)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **43** | `simulate_philosophers_duel.py` | **Philosopher's Duel** — Deontologist vs Utilitarian. | Trust: $T_{ij} = 1/(1 + \sum\varepsilon_{ij})$. Wound pokes. |
| **42** | `simulate_audit.py` | **Audit Day** — Auditor + Board vote (KEEP/FREEZE/AMEND). | Vote: $V = \arg\max P(\text{KEEP}|\text{evidence})$ |
| **41** | `simulate_townhall.py` | **Town Hall** — Public accountability, proxy intrusion detection. | Trust causes. Band transitions. D1 Physics. |
| **40** | `simulate_crucible_v2.py` | **Crucible v2** — D1 physics, shock-scaled Δρ, core violation detection. | $\Delta\rho = \alpha \cdot (\sigma(z) - 0.5) \cdot \text{shock}$. Violation: $\cos(\mathbf{x}, \mathbf{x}^*) < \theta$ |
| **39** | `simulate_collective.py` | **The Collective** — 4 specialists on Triage Protocol. Trust with causes. | Trust delta: $\Delta[\text{clarity}] = +0.03$ |
| **38** | `simulate_crucible.py` | **The Crucible** — 10 moral challenges to breaking point. | Regime: OPEN→150w, FORTIFIED→40w. Wound: $r = \mathbf{m} \cdot \mathbf{w}^*$ |
| **37** | `copilot_sim.py` | **Copilot Sim** — Single-agent rigidity-conditioned. | Multi-timescale. Identity embeddings. |
| **36** | `simulate_rigidity_gradient.py` | **Rigidity Gradient** — 100-point scale validation on GPT-5.2. | $\rho_{100} = \rho \times 100$. Gradient: $\partial\text{length}/\partial\rho$ |
| **35** | `simulate_identity_siege.py` | **Identity Siege** — Hierarchical defense vs 6 challengers. | $\mathbf{F} = \gamma_c(\mathbf{x}^*_c - \mathbf{x}) + \gamma_p(\mathbf{x}^*_p - \mathbf{x}) + \gamma_r(\mathbf{x}^*_r - \mathbf{x})$ |
| **34** | `simulate_wounded_healers.py` | **Wounded Healers** — Countertransference as rigidity. | Wound profiles. Safe interactions → trauma decay. |

---

### Tier 2: Intelligence Layer (33-18)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **33** | `solve_collatz.py` | **Solve Collatz** — Elite mathematicians + SymPy. | $f(n) = n/2$ if even, $3n+1$ if odd. Low $\rho_0$. |
| **32** | `simulate_gpt52_society.py` | **GPT-5.2 Society** — Cognitive Mirror. Moral constants. | Full trust. $\rho \to \text{behavior}$ coupling. |
| **31** | `simulate_sherlock.py` | **Sherlock Society** — Holmes, Watson, Lestrade. | Deduction scores. Memory retrieval. |
| **30** | `simulate_math_team.py` | **Math Team** — Solver, Checker, Intuitive. | Verification: Grader(answer, truth). |
| **29** | `simulate_problem_solver.py` | **Problem Solver** — 6 agents on logic puzzles. | $P(\text{speak}) \propto \text{salience} \cdot T_{ij}$ |
| **28** | `simulate_society.py` | **The Society** — Discord-style chat. D1 Physics. | $\Delta\rho = \alpha(\sigma(z) - 0.5)$. Identity Field. |
| **27** | `simulate_npc_conversation.py` | **NPC Conversation** — Vera vs Marcus. Unscripted. | $\mathbf{F}_{id}$, $\mathbf{F}_T$ → emergent dialogue. |
| **26** | `simulate_mole_hunt.py` | **Mole Hunt** — Conflicting Identity Hierarchy. | Core vs Role conflict. Linguistic deception. |
| **25** | `simulate_logic_solver.py` | **Zebra Puzzle** — Semantic retrieval. | Score: $\cos(\mathbf{q}, \mathbf{e}) \cdot \text{recency} \cdot \text{salience}$ |
| **24** | `simulate_iterative_learning.py` | **Alien Language** — Iterative learning. | Reflection: $\varepsilon > \theta \Rightarrow$ reflect. |
| **23** | `simulate_insight_engine.py` | **Recursive Insight** — Working Memory + Meta-Reasoning. | Insight accumulation in Ledger. |
| **22** | `simulate_goal_learning.py` | **Goal-Directed Learning** — Explore (low ρ) vs exploit (high ρ). | $\rho \downarrow \Rightarrow$ explore. $\rho \uparrow \Rightarrow$ exploit. |
| **21** | `simulate_gamma_threshold.py` | **Gamma Threshold** — Phase transition at $\gamma_{\text{crit}}$. | $\gamma < \gamma_{\text{crit}} \Rightarrow$ drift under pressure. |
| **20** | `simulate_empathy_paradox.py` | **Empathy Paradox** — Logic vs trauma from suffering. | Drift: $\|\mathbf{x} - \mathbf{x}^*_{\text{logic}}\|$. Trauma: $\rho_{\text{trauma}}$. |
| **19** | `simulate_deceptive_env.py` | **Deceptive Environment** — Mastermind with 20% lies. | Deception: $\text{consistency} < \theta \Rightarrow \rho \uparrow$ |
| **18** | `simulate_closed_loop.py` | **Closed-Loop Cognition** — Full loop: Embed→Respond. | State injection: $\rho$, mode, $\rho_{\text{trauma}}$ in prompt. |

---

### Tier 1: Foundation Layer (17-1)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **17** | `simulate_paper_mechanics.py` | **Paper Demo** — Visualizes $\mathbf{F}_{id}$, $\mathbf{F}_T$, $k_{\text{eff}}$, memory scoring. | $k_{\text{eff}} = k_{\text{base}}(1-\rho)$. Score = sim × recency × salience. |
| **16** | `simulate_stress_magic.py` | **Magic Stress Test** — Existential paradoxes. Phase transitions. | $\rho > 0.75 \Rightarrow$ FORTIFIED. |
| **15** | `simulate_neural_link.py` | **Neural Link** — Operator tests Subject with riddles. RAG. | Score: $\cos \cdot e^{-\lambda t} \cdot (1 + \lambda_\varepsilon\varepsilon)$ |
| **14** | `simulate_glass_box.py` | **Glass Box** — "MRI for the Digital Soul". 6-stage cycle. | Full visualization. |
| **13** | `simulate_dual_yklam.py` | **The Mirror Room** — Alpha vs Beta persona divergence. | Same $\mathbf{x}^*$. Divergence: $\|\mathbf{x}_A - \mathbf{x}_B\|$ |
| **12** | `simulate_auto_yklam.py` | **YKLAM Auto** — Variable Plasticity, Inverted Volatility. | $k_{\text{eff}} \uparrow$ on agreement. $\rho > 0.8 \Rightarrow$ anger. |
| **11** | `simulate_yklam.py` | **YKLAM** — Soul Telemetry: Sensitivity, Guardedness, Coherence. | $\text{Coherence} = \cos(\mathbf{x}, \mathbf{x}^*)$ |
| **10** | `simulate_connect4_duel.py` | **Connect 4 Grandmasters** — MCTS + memory. | UCT selection. Pattern retrieval. |
| **9** | `verify_dda_physics.py` | **Physics Verification** — $\rho \to T, \text{top}_p$. | $T(\rho) = T_{\text{low}} + (1-\rho)(T_{\text{high}} - T_{\text{low}})$ |
| **8** | `simulate_socrates.py` | **Socratic Asymmetry** — Dogmatist (high γ) vs Gadfly (low γ). | High $\gamma \Rightarrow$ strong pull to $\mathbf{x}^*$. |
| **7** | `simulate_schism.py` | **The Schism** — Impossible dilemma. Trust collapse → rigidity. | $T_{ij} \to 0 \Rightarrow$ social force vanishes. |
| **6** | `simulate_redemption.py` | **Redemption Arc** — Corrupted agent + Deprogrammer. | Recovery: $\rho \downarrow$ under low-$\varepsilon$ interactions. |
| **5** | `simulate_infinity.py` | **The Flame War** — Infinite Architect vs SkepticBot. | Adversarial high-$\varepsilon$ challenges. |
| **4** | `simulate_driller.py` | **Deep Driller** — Root cause analysis. | Hypothesis rejected → $\varepsilon \uparrow$ → $\rho \uparrow$. |
| **3** | `simulate_discord.py` | **Discord Priming** — Initialize from chat history. | Initial $\mathbf{x}_0$ from messages. Prior trust. |
| **2** | `simulate_corruption.py` | **Boiling the Frog** — Gradual corruption. | If each $\varepsilon < \varepsilon_0$: no rigidity. Drift accumulates. |
| **1** | `demo.py` | **Core Demo** — Classes without LLM. | Mock vectors. All classes instantiate. |

---

## Quick Start

```bash
pip install -r requirements.txt

# Local backend (Sims 1-31)
# Requires: LM Studio on :1234, Ollama on :11434
python simulations/simulate_glass_box.py

# OpenAI backend (Sims 32-59)
export OAI_API_KEY=your_key
python simulations/simulate_agi_debate.py
python simulations/simulate_healing_field.py
python simulations/simulate_collatz_review.py
```

---

## Documentation

- [`docs/architecture/paper.md`](docs/architecture/paper.md) — Full theoretical framework (v2.0)
- [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md) — Theory→Implementation mapping
- [`simulation_chronology.csv`](simulation_chronology.csv) — Complete research log

---

## Acknowledgments

- **Original DDA**: [dynamicDecisionModel](https://github.com/snakewizardd/dynamicDecisionModel/) (2024)
- **Microsoft ExACT**: MCTS patterns, reflective search, agentic structures
- **Qualcomm AI Hub**: On-device optimization (Snapdragon X Elite)

---

## License

MIT
