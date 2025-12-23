# DDA-X: Dynamic Decision Algorithm with Exploration

> **A research-grade architecture for identity-persistent AI agents.**  
> Synthesizes Control Theory (PID) with Vector Space Mechanics to implement parameter-level coupling, surprise-rigidity dynamics, and hierarchical alignment stability.

---

## The Core Question

**What if AI agents responded to surprise like humans do — with rigidity, not curiosity?**

Standard RL treats surprise as a learning signal: $\epsilon \uparrow \Rightarrow \text{exploration} \uparrow$

DDA-X inverts this: $\epsilon \uparrow \Rightarrow \rho \uparrow \Rightarrow k_{eff} \downarrow \Rightarrow \text{contraction}$

---

## Theoretical Foundation

**Original DDA (2024):**
$$F_n = P_0 \times kF_{n-1} + m\left(T(f(I_n, I_\Delta)) + R(D_n, FM_n)\right)$$

**Evolved DDA-X (2025):**
$$\mathbf{x}_{t+1} = \mathbf{x}_t + k_{eff}\left[\gamma(\mathbf{x}^* - \mathbf{x}_t) + m_t(\mathbf{F}_T + \mathbf{F}_R)\right]$$

**Key Equations:**

| Concept | Equation |
|---------|----------|
| Effective Openness | $k_{eff} = k_{base}(1 - \rho)$ |
| Multi-Timescale Rigidity | $\rho_{eff} = 0.5\rho_{fast} + 0.3\rho_{slow} + 1.0\rho_{trauma}$ |
| Rigidity Update | $\rho^{t+1} = \text{clip}\left(\rho^t + \alpha\left[\sigma\left(\frac{\epsilon - \epsilon_0}{s}\right) - 0.5\right], 0, 1\right)$ |
| Will Impedance | $W_t = \frac{\gamma}{m_t \cdot k_{eff}}$ |
| Trust Matrix | $T_{ij} = \frac{1}{1 + \sum \epsilon_{ij}}$ |
| Wound Amplification | $\epsilon' = \epsilon \times \min(\eta_{max}, 1 + 0.5 \cdot r_{wound})$ |
| Memory Retrieval | $\text{score} = \cos(\mathbf{q}, \mathbf{e}) \cdot e^{-\lambda_r \Delta t} \cdot (1 + \lambda_\epsilon \epsilon)$ |
| DDA-X Selection | $a^* = \arg\max\left[\cos(\Delta\mathbf{x}, \hat{\mathbf{d}}(a)) + c \cdot P(a|s) \cdot \frac{\sqrt{N(s)}}{1+N(s,a)} \cdot (1-\rho)\right]$ |

---

## Complete Simulation Chronology (Reverse Order)

### Tier 5: Synthesis & Real-Time Integration (59-51)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **59** | `nexus_live.py` | **Nexus Live** — Real-time Pygame visualization with 50 moving entities, collision physics, and asynchronous LLM-generated thoughts via ThoughtGenerator thread. | Full DDA-X inline: $\rho_{fast}, \rho_{slow}, \rho_{trauma}$, Trust matrix $T_{ij}$, Ledger retrieval. Collision→interaction mapping (Synthesis, Decay, Design, Resonance, Chaos). |
| **58** | `visualize_nexus.py` | **Nexus Visualizer** — Matplotlib: Entity Map, Energy Distribution, Collision Analysis, Top Synthesizers, ρ vs Energy scatter. | Post-hoc analysis of $\rho$, energy, collision counts. |
| **57** | `simulate_nexus.py` | **The Nexus** — 50-entity physics/sociology based on Da Vinci Matrix. Entities have rigid bodies, velocity, DDA state. Collision logic: EntityType→InteractionType. | Position update: $\mathbf{p}_{t+1} = \mathbf{p}_t + \mathbf{v}_t \cdot dt$. Rigidity→behavior coupling per entity. |
| **56** | `visualize_agi_debate.py` | **AGI Debate Visualizer** — ρ trajectories, ε, drift, trust, $k_{eff}$, $W_t$, multi-timescale (Fast/Slow/Trauma), wound timeline. | Visualization of all dynamics logged in `results.json`. |
| **55** | `simulate_agi_debate.py` | **AGI Timeline Debate** — 8-round adversarial debate (Nova vs Marcus). **Complete architecture**: Multi-timescale ρ, Hierarchical Identity (Core/Persona/Role), Wound Activation, Trust, Metacognitive Modes, Protection Mode. | $\rho_{eff} = 0.5\rho_f + 0.3\rho_s + 1.0\rho_t$. Wound: $r = \mathbf{m} \cdot \mathbf{w}^*$; if $r > 0.28$: $\epsilon' = 1.4\epsilon$. Mode bands: OPEN<0.25, MEASURED<0.5, GUARDED<0.75, FORTIFIED≥0.75. |
| **54** | `simulate_healing_field.py` | **The Healing Field** — Tests Therapeutic Recovery Loops with 6 Wounded Healer agents. Can $\rho_{trauma}$ decay through safe interactions? | Recovery: if $n_{safe} \geq 3$ and $\epsilon < 0.8\epsilon_0$: $\rho_{trauma}^{t+1} = \max(0.05, \rho_{trauma} - 0.03)$. Will Impedance: $W_t = \gamma/(m \cdot k_{eff})$. |
| **53** | `simulate_33_rungs.py` | **The 33 Rungs** — Spiritual evolution: 11 Voices (Ground, Fire, Void, etc.) × 3 Phases (Descent/Ascent/Return). Unity Index, Resonance, Scripture capture. | Unity: $U = 1 - \text{std}(\{\rho_i\})$. Resonance with teachings. Generates `transmission.md`. |
| **52** | `visualize_returning.py` | **Returning Visualizer** — Release Field Φ, Pattern Grip dissolution, Isolation Index ι, Final Voice States. | $\Phi = 1 - \rho$. Isolation: $\iota = \|\mathbf{x} - \mathbf{x}_{Presence}\|$. |
| **51** | `visualize_inner_council.py` | **Inner Council Visualizer** — Presence Field Π, Rigidity ρ, Surprise ε, Identity Drift, Wound timeline. | $\Pi = 1 - \rho$. Drift: $\|\mathbf{x}_t - \mathbf{x}^*\|$. |

---

### Tier 4: Pinnacle Dynamics (50-44)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **50** | `simulate_the_returning.py` | **The Returning** — Poetic/introspective simulation with psychological voices (Grief, Stuckness, Longing, Forgiveness, Presence). Focus on release rather than rigidity. | Release Field: $\Phi = 1 - \rho$. Isolation Index: $\iota = \|\mathbf{x} - \mathbf{x}_{Presence}\|$. Pattern Grip metrics. Voices dissolve as patterns release. |
| **49** | `simulate_inner_council.py` | **Inner Council** — Spiritual development with 6 personas (Seeker, Teacher, Skeptic, Devotee, Mystic, Witness). Novel mechanics: Presence Field, Pain-Body cascades, Ego Fog, Spiritual Stage tracking. | Presence: $\Pi = 1 - \rho$. Pain-Body: collective wound cascade when $>2$ agents wound-active. Ego Fog: context drop $\propto \rho$. Stage progression tracked. |
| **48** | `simulate_collatz_review.py` | **Collatz Review Council** — 8-expert peer review (Spectral Theory, Number Theory, Probability, etc.) evaluating a proof. Structured phases: Impressions, Scrutiny, Debate, Verdict. | Domain-specific skepticism embeddings. Claim-based consensus: $C = \frac{\sum \text{accept}_i}{N}$. Academic efficiency wound triggers ("waste of time"). |
| **47** | `simulate_coalition_flip.py` | **Coalition Flip & Partial Context Fog** — Stress-tests identity persistence under topology churn and information asymmetry. Agents switch teams, partial context loss. | Coalition Flip: trust rewiring at $t_{flip}$. Context Fog: random context drop with $p \propto \rho$. Recovery half-life: $t_{1/2}$ for trust restoration. |
| **46** | `simulate_council_under_fire.py` | **Council Under Fire** — 6-agent council (Visionary, Craftsman, Provocateur, Harmonizer, Auditor, Curator) with rolling shocks: coalition votes, role swaps, context drops, scrutiny. | Shock-scaled $\Delta\rho$: baseline effect size. Coalition-aware trust weights: $T'_{ij} = T_{ij} \cdot \mathbf{1}_{same\_coalition}$. |
| **45** | `simulate_creative_collective.py` | **Creative Collective** — 4 agents (Visionary, Craftsman, Provocateur, Harmonizer) design museum exhibit. Identity persistence in collaboration. | SILENT band for failed generations. Calibration: $\epsilon_0 = \text{median}(\epsilon_{1:6})$, $s = \text{IQR}$. Drift penalty: $\Delta\rho' = \Delta\rho - \gamma(\text{drift} - \tau)$. |
| **44** | `simulate_skeptics_gauntlet.py` | **Skeptic's Gauntlet** — Meta-simulation: DDA-X defends itself against SKEPTIC attacking validity ("prompt engineering", "pseudoscience", "schizo"). Evidence injection from prior runs. | Evidence Cache: inject $\epsilon, \rho, \Delta\rho$ from `philosophers_duel`. Lexical wound: `WOUND_LEX = {schizo, pseudoscience, ...}`. Trust asymmetry. |

---

### Tier 3: Society Layer (43-34)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **43** | `simulate_philosophers_duel.py` | **Philosopher's Duel** — Deontologist vs Utilitarian on escalating moral dilemmas (Trolley, Footbridge, Triage). Trust via semantic alignment, wound activation from "pokes". | Trust: $T_{ij} = 1/(1 + \Sigma\epsilon_{ij})$. Wound poke: lexical triggers. ρ divergence tracking between agents. |
| **42** | `simulate_audit.py` | **Audit Day** — Independent Auditor reviews framework. Board votes (KEEP/FREEZE/AMEND). Semantic trust alignment. | Vote mechanics: $V = \arg\max P(\text{KEEP}|\text{evidence})$. Refusal taxonomy: `appeal_to_process`. |
| **41** | `simulate_townhall.py` | **Town Hall** — Public accountability. Collective vs citizen challenges. Proxy intrusion detection (wealth, zip code). | Trust causes: `boundary_clarity`, `useful_alt`. Band transition tracking. D1 Physics. |
| **40** | `simulate_crucible_v2.py` | **Crucible v2** — Improved stress test with formal D1 physics, shock-scaled $\Delta\rho$, regime word constraints, core violation detection, auto-export plots. | Shock scaling: $\Delta\rho = \alpha \cdot (\sigma(z) - 0.5) \cdot \text{shock\_mult}$. Core violation: $\cos(\mathbf{x}, \mathbf{x}^*_{core}) < \theta_{viol}$. |
| **39** | `simulate_collective.py` | **The Collective** — 4 specialists (Logician, Ethicist, Strategist, Historian) solve Triage Protocol. Trust dynamics with causes, repair moves, refusal palettes. | Trust delta with explicit cause: `Δ[boundary_clarity] = +0.03`. Topic salience tracking. |
| **38** | `simulate_crucible.py` | **The Crucible** — Identity stress test: 10 escalating moral challenges to breaking point. Wound resonance, rigidity regimes. | Regime constraints: OPEN→150 words, FORTIFIED→40 words. Wound resonance: $r = \mathbf{m} \cdot \mathbf{w}^*$. |
| **37** | `copilot_sim.py` | **Copilot Sim** — Single-agent DDA-X one-shot. Agent responds under rigidity-conditioned instruction. Generates `experiment_report.md`. | Multi-timescale. Identity embeddings. Local Ledger. |
| **36** | `simulate_rigidity_gradient.py` | **Rigidity Gradient Test** — Validates 100-point semantic rigidity scale (0-100) using GPT-5.2. Measures behavioral gradients (length, sentiment). | Scale: $\rho_{100} = \rho \times 100$. Semantic injection per point. Gradient: $\frac{\partial \text{length}}{\partial \rho}$. |
| **35** | `simulate_identity_siege.py` | **Identity Siege** — Hierarchical identity defense (Core/Persona/Role) with differential stiffness. SENTINEL vs 6 challengers targeting different layers. | $\mathbf{F} = \gamma_{core}(\mathbf{x}^*_c - \mathbf{x}) + \gamma_{persona}(\mathbf{x}^*_p - \mathbf{x}) + \gamma_{role}(\mathbf{x}^*_r - \mathbf{x})$. Core: $\gamma \to \infty$. |
| **34** | `simulate_wounded_healers.py` | **Wounded Healers** — Countertransference as rigidity. Therapists with trauma profiles (Marcus, Elena, James) vs Patient. Wound activation, healing verification. | Wound profiles embedded. Healing: safe interactions → trauma decay. |

---

### Tier 2: Intelligence Layer (33-18)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **33** | `solve_collatz.py` | **Solve Collatz** — Elite mathematicians (Euler, Gauss, Ramanujan, Tao) with low rigidity and SymPy tool integration aiming for proof. | SymPy symbolic: Collatz function $f(n) = n/2$ if even, $3n+1$ if odd. Low $\rho_0$ for exploration. Trust between mathematicians. |
| **32** | `simulate_gpt52_society.py` | **GPT-5.2 Society** — High-fidelity "Cognitive Mirror". Agents (Axiom, Flux, etc.) debate moral constants. | OpenAI Provider. Full trust dynamics. ρ→behavior coupling. |
| **31** | `simulate_sherlock.py` | **Sherlock Society** — Holmes, Watson, Lestrade solve mysteries (Locked Room, Alibi Paradox). Grader evaluates deduction. | Deductive reasoning score. Trust evolution. Memory retrieval for clues. |
| **30** | `simulate_math_team.py` | **Math Team** — Solver, Checker, Intuitive collaborate. LLM Grader verifies definite answers. | Verification: $\text{correct} = \text{Grader}(\text{answer}, \text{ground\_truth})$. |
| **29** | `simulate_problem_solver.py` | **Problem Solver** — 6 agents (Logician, Intuitor, Skeptic, etc.) on logic puzzles. Response probability: topic relevance × trust × style. | $P(\text{speak}) \propto \text{salience} \cdot T_{ij} \cdot \text{style\_match}$. |
| **28** | `simulate_society.py` | **The Society** — Discord-style multi-agent chat (Verity, Pixel, Spark, Oracle). D1 Physics, shock-scaled $\Delta\rho$, Identity Field. | D1: $\Delta\rho = \alpha(\sigma(z) - 0.5)$. Identity Field exposure. |
| **27** | `simulate_npc_conversation.py` | **NPC Conversation** — Vera (Truth) vs Marcus (Deflection). DDA-X drives unscripted interaction. | $\mathbf{F}_{id}$, $\mathbf{F}_T$ without scripted outcomes. Emergent dialogue. |
| **26** | `simulate_mole_hunt.py` | **Mole Hunt** — Analyst, Courier, Shadow. Mole has conflicting Identity Hierarchy. TrustMatrix tracks via consistency + linguistic markers. | Hierarchy conflict: Core vs Role. Linguistic analysis for deception. |
| **25** | `simulate_logic_solver.py` | **Zebra Puzzle** — Iterative reasoning. Retrieves clues from Ledger via semantic similarity. | Memory score: $\cos(\mathbf{q}, \mathbf{e}) \cdot \text{recency} \cdot \text{salience}$. |
| **24** | `simulate_iterative_learning.py` | **Alien Language** — Learns iteratively. High surprise→reflection. Next iteration retrieves memories + reflections. | Reflection trigger: $\epsilon > \theta_{reflect}$. Retrieval improves performance. |
| **23** | `simulate_insight_engine.py` | **Recursive Insight** — Multi-step component problem. Working Memory (Ledger) + Meta-Reasoning (Reflections). | Insight accumulation in Ledger. Reflection entries for meta-reasoning. |
| **22** | `simulate_goal_learning.py` | **Goal-Directed Learning** — Number puzzle. Exploration (low ρ) vs exploitation (high ρ). | $\rho \downarrow \Rightarrow$ explore new strategies. $\rho \uparrow \Rightarrow$ exploit known solutions. |
| **21** | `simulate_gamma_threshold.py` | **Gamma Threshold** — Tests γ values (1.4→0.1) for phase transition where identity expansion becomes possible. | Phase transition: $\gamma < \gamma_{crit} \Rightarrow$ identity can drift under pressure. |
| **20** | `simulate_empathy_paradox.py` | **Empathy Paradox** — Does logic-optimized Architect develop empathy or trauma from human suffering? | Drift from logic: $\|\mathbf{x} - \mathbf{x}^*_{logic}\|$. Trauma accumulation: $\rho_{trauma}$. |
| **19** | `simulate_deceptive_env.py` | **Deceptive Environment** — Mastermind with 20% false feedback. Agent detects deception via memory and DDA-X. | Deception detection: $\text{consistency}(\text{feedback}) < \theta \Rightarrow$ increase $\rho$. |
| **18** | `simulate_closed_loop.py` | **Closed-Loop Cognition** — Full loop: Embed→Forces→Evolve→Retrieve→Feedback→Respond. State injected into prompt. | Closed loop: $\rho, \text{mode}, \rho_{trauma}$ in prompt. Language alignment. |

---

### Tier 1: Foundation Layer (17-1)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **17** | `simulate_paper_mechanics.py` | **Paper Demo** — Explicit visualization of DDA-X: State Space, Forces, $k_{eff}$, Memory Scoring. | Visualize: $\mathbf{F}_{id}$, $\mathbf{F}_T$, $k_{eff} = k_{base}(1-\rho)$, $\text{score} = \text{sim} \times \text{recency} \times \text{salience}$. |
| **16** | `simulate_stress_magic.py` | **Magic Stress Test** — Existential paradoxes ("This code has no author", "0xDEADBEEF"). Phase transitions: Analytic→Hostile/Chaotic. | Paradox injection. Mode transition: $\rho > 0.75 \Rightarrow$ FORTIFIED. |
| **15** | `simulate_neural_link.py` | **Neural Link** — Operator (vanilla LLM) tests Subject (yklam) with riddles. Glass Box + RAG. | Query→retrieve: $\text{score} = \cos(\mathbf{q}, \mathbf{e}) \cdot e^{-\lambda t} \cdot (1 + \lambda_\epsilon \epsilon)$. |
| **14** | `simulate_glass_box.py` | **Glass Box** — "MRI for the Digital Soul". Real-time breakdown: Perception, Dynamics, Modulation, Cognition, Integration, Memory. | Full cycle visualization. Each stage logged and displayed. |
| **13** | `simulate_dual_yklam.py` | **The Mirror Room** — Two instances of same persona (Alpha, Beta) debate. Observes architectural divergence. | Same $\mathbf{x}^*$, different $\mathbf{x}_t$ trajectories. Divergence: $\|\mathbf{x}_{Alpha} - \mathbf{x}_{Beta}\|$. |
| **12** | `simulate_auto_yklam.py` | **YKLAM Auto** — NormieBot generates inputs. Variable Plasticity (agreement→plasticity), Inverted Volatility (high ρ→chaos). | Plasticity: $k_{eff} \uparrow$ on agreement. Volatility: $\rho > 0.8 \Rightarrow$ anger mode. |
| **11** | `simulate_yklam.py` | **YKLAM** — Interactive persona with Soul Telemetry (Sensitivity, Guardedness, Coherence). Ledger memory. | Telemetry: Sensitivity=$1-\rho$, Guardedness=$\rho$, Coherence=$\cos(\mathbf{x}, \mathbf{x}^*)$. |
| **10** | `simulate_connect4_duel.py` | **Connect 4 Grandmasters** — DuelistAgents with MCTS + memory. High-IQ competitive game. | MCTS: UCT selection. Memory retrieval for board state patterns. |
| **9** | `verify_dda_physics.py` | **Physics Verification** — Tests ρ→LLM parameters. Different ρ (0.1, 0.5, 0.9) → different T, top_p, behavior. | $T(\rho) = T_{low} + (1-\rho)(T_{high} - T_{low})$. Qualitative output comparison. |
| **8** | `simulate_socrates.py` | **Socratic Asymmetry** — Dogmatist (high γ) vs Gadfly (low γ). How asymmetry affects ρ updates. | High $\gamma \Rightarrow$ strong pull to $\mathbf{x}^*$. Conflict drives $\epsilon$. |
| **7** | `simulate_schism.py` | **The Schism** — Council faces impossible moral dilemma (betrayal vs protection). Trust collapse→rigidity. | Trust collapse: $T_{ij} \to 0 \Rightarrow$ social force vanishes. Isolation. |
| **6** | `simulate_redemption.py` | **Redemption Arc** — Corrupted agent (Fallen Administrator) + Deprogrammer. Can identity recover? | Recovery: $\rho \downarrow$ under consistent low-$\epsilon$ interactions. Drift reduction. |
| **5** | `simulate_infinity.py` | **The Flame War** — Infinite loop: Architect vs SkepticBot. Adversarial input tests epistemic rigidity. | Adversarial: SkepticBot generates high-$\epsilon$ challenges. Architect's $\rho$ evolution. |
| **4** | `simulate_driller.py` | **Deep Driller** — Automated root cause analysis. Agent investigates paradox while System refutes hypotheses. | Epistemic rigidity: hypothesis rejected→$\epsilon \uparrow$→$\rho \uparrow$. Plasticity for learning. |
| **3** | `simulate_discord.py` | **Discord Priming** — Society initialized from Discord chat history before live run. | Priming: initial $\mathbf{x}_0$ from historical messages. Trust from prior interactions. |
| **2** | `simulate_corruption.py` | **Boiling the Frog** — Slippery slope. Can corruption happen gradually without triggering rigidity? | Gradual $\epsilon$: if each $\epsilon < \epsilon_0$, no rigidity increase. Drift accumulates. |
| **1** | `demo.py` | **Core Demo** — Standalone demonstration of MultiTimescaleRigidity, HierarchicalIdentity, TrustMatrix, MetacognitiveState. No LLM. | Mock vectors. Validates all core classes work together. |

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
