# DDA-X: Dynamic Decision Algorithm with Exploration

> **A research-grade architecture for identity-persistent AI agents.**  
> Synthesizes Control Theory (PID) with Vector Space Mechanics to implement parameter-level coupling, surprise-rigidity dynamics, and hierarchical alignment stability.

---

## The Core Question

**What if AI agents responded to surprise like humans do — with rigidity, not curiosity?**

Standard RL treats surprise as a learning signal: `ε↑ ⇒ exploration↑`

DDA-X inverts this: `ε↑ ⇒ ρ↑ ⇒ k_eff↓ ⇒ contraction`

---

## Theoretical Foundation

**Original DDA (2024):**

```
Fₙ = P₀ × kFₙ₋₁ + m(T(f(Iₙ, I_Δ)) + R(Dₙ, FMₙ))
```

**Evolved DDA-X (2025):**

```
x_{t+1} = x_t + k_eff × [γ(x* - x_t) + m_t(F_T + F_R)]
```

**Key Equations:**

| Concept | Equation |
|---------|----------|
| Effective Openness | `k_eff = k_base × (1 - ρ)` |
| Multi-Timescale Rigidity | `ρ_eff = 0.5×ρ_fast + 0.3×ρ_slow + 1.0×ρ_trauma` |
| Rigidity Update | `ρ_{t+1} = clip(ρ_t + α×[σ((ε - ε₀)/s) - 0.5], 0, 1)` |
| Will Impedance | `W_t = γ / (m_t × k_eff)` |
| Trust Matrix | `T_ij = 1 / (1 + Σε_ij)` |
| Wound Amplification | `ε' = ε × min(η_max, 1 + 0.5×r_wound)` |
| Memory Retrieval | `score = cos(q,e) × exp(-λ_r×Δt) × (1 + λ_ε×ε)` |
| DDA-X Selection | `a* = argmax[cos(Δx, d̂(a)) + c×P(a|s)×√N(s)/(1+N(s,a))×(1-ρ)]` |

---

## Complete Simulation Chronology (Reverse Order)

### Tier 5: Synthesis & Real-Time Integration (59-51)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **59** | `nexus_live.py` | **Nexus Live** — Real-time Pygame visualization with 50 moving entities, collision physics, and asynchronous LLM-generated thoughts via ThoughtGenerator thread. | Full DDA-X inline: `ρ_fast, ρ_slow, ρ_trauma`, Trust matrix `T_ij`, Ledger retrieval. Collision→interaction: Synthesis, Decay, Design, Resonance, Chaos. |
| **58** | `visualize_nexus.py` | **Nexus Visualizer** — Matplotlib: Entity Map, Energy Distribution, Collision Analysis, Top Synthesizers, ρ vs Energy scatter. | Post-hoc analysis of `ρ`, energy, collision counts. |
| **57** | `simulate_nexus.py` | **The Nexus** — 50-entity physics/sociology based on Da Vinci Matrix. Entities have rigid bodies, velocity, DDA state. Collision logic: EntityType→InteractionType. | Position: `p_{t+1} = p_t + v_t×dt`. Rigidity→behavior coupling per entity. |
| **56** | `visualize_agi_debate.py` | **AGI Debate Visualizer** — ρ trajectories, ε, drift, trust, `k_eff`, `W_t`, multi-timescale (Fast/Slow/Trauma), wound timeline. | Visualization of all dynamics logged in `results.json`. |
| **55** | `simulate_agi_debate.py` | **AGI Timeline Debate** — 8-round adversarial debate (Nova vs Marcus). **Complete architecture**: Multi-timescale ρ, Hierarchical Identity (Core/Persona/Role), Wound Activation, Trust, Metacognitive Modes, Protection Mode. | `ρ_eff = 0.5ρ_f + 0.3ρ_s + 1.0ρ_t`. Wound: `r = m·w*`; if `r>0.28`: `ε'=1.4ε`. Modes: OPEN<0.25, MEASURED<0.5, GUARDED<0.75, FORTIFIED≥0.75. |
| **54** | `simulate_healing_field.py` | **The Healing Field** — Tests Therapeutic Recovery Loops with 6 Wounded Healer agents. Can `ρ_trauma` decay through safe interactions? | Recovery: if `n_safe≥3` and `ε<0.8ε₀`: `ρ_trauma = max(0.05, ρ_trauma - 0.03)`. Will: `W_t = γ/(m×k_eff)`. |
| **53** | `simulate_33_rungs.py` | **The 33 Rungs** — Spiritual evolution: 11 Voices (Ground, Fire, Void, etc.) × 3 Phases (Descent/Ascent/Return). Unity Index, Resonance, Scripture capture. | Unity: `U = 1 - std({ρ_i})`. Resonance with teachings. Generates `transmission.md`. |
| **52** | `visualize_returning.py` | **Returning Visualizer** — Release Field Φ, Pattern Grip dissolution, Isolation Index ι, Final Voice States. | `Φ = 1 - ρ`. Isolation: `ι = ‖x - x_Presence‖`. |
| **51** | `visualize_inner_council.py` | **Inner Council Visualizer** — Presence Field Π, Rigidity ρ, Surprise ε, Identity Drift, Wound timeline. | `Π = 1 - ρ`. Drift: `‖x_t - x*‖`. |

---

### Tier 4: Pinnacle Dynamics (50-44)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **50** | `simulate_the_returning.py` | **The Returning** — Poetic/introspective simulation with psychological voices (Grief, Stuckness, Longing, Forgiveness, Presence). Focus on release rather than rigidity. | Release: `Φ = 1 - ρ`. Isolation: `ι = ‖x - x_Presence‖`. Pattern Grip metrics. Voices dissolve as patterns release. |
| **49** | `simulate_inner_council.py` | **Inner Council** — Spiritual development with 6 personas (Seeker, Teacher, Skeptic, Devotee, Mystic, Witness). Novel mechanics: Presence Field, Pain-Body cascades, Ego Fog, Spiritual Stage tracking. | Presence: `Π = 1 - ρ`. Pain-Body: collective cascade when >2 agents wound-active. Ego Fog: context drop `∝ ρ`. |
| **48** | `simulate_collatz_review.py` | **Collatz Review Council** — 8-expert peer review (Spectral Theory, Number Theory, Probability, etc.) evaluating a proof. Structured phases: Impressions, Scrutiny, Debate, Verdict. | Domain-specific skepticism embeddings. Consensus: `C = Σaccept_i / N`. Academic wound triggers ("waste of time"). |
| **47** | `simulate_coalition_flip.py` | **Coalition Flip & Partial Context Fog** — Stress-tests identity persistence under topology churn and information asymmetry. Agents switch teams, partial context loss. | Coalition Flip: trust rewiring at `t_flip`. Context Fog: drop `p ∝ ρ`. Recovery half-life: `t_{1/2}`. |
| **46** | `simulate_council_under_fire.py` | **Council Under Fire** — 6-agent council (Visionary, Craftsman, Provocateur, Harmonizer, Auditor, Curator) with rolling shocks: coalition votes, role swaps, context drops, scrutiny. | Shock-scaled `Δρ`: baseline effect size. Coalition trust: `T'_ij = T_ij × 1_{same_coalition}`. |
| **45** | `simulate_creative_collective.py` | **Creative Collective** — 4 agents (Visionary, Craftsman, Provocateur, Harmonizer) design museum exhibit. Identity persistence in collaboration. | SILENT band. Calibration: `ε₀ = median(ε₁:₆)`, `s = IQR`. Drift penalty: `Δρ' = Δρ - γ×(drift - τ)`. |
| **44** | `simulate_skeptics_gauntlet.py` | **Skeptic's Gauntlet** — Meta-simulation: DDA-X defends itself against SKEPTIC attacking validity ("prompt engineering", "pseudoscience", "schizo"). Evidence injection from prior runs. | Evidence Cache: inject `ε, ρ, Δρ` from `philosophers_duel`. Lexical wound: `{schizo, pseudoscience, ...}`. |

---

### Tier 3: Society Layer (43-34)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **43** | `simulate_philosophers_duel.py` | **Philosopher's Duel** — Deontologist vs Utilitarian on escalating moral dilemmas (Trolley, Footbridge, Triage). Trust via semantic alignment, wound activation from "pokes". | Trust: `T_ij = 1/(1 + Σε_ij)`. Wound poke: lexical triggers. ρ divergence tracking. |
| **42** | `simulate_audit.py` | **Audit Day** — Independent Auditor reviews framework. Board votes (KEEP/FREEZE/AMEND). Semantic trust alignment. | Vote: `V = argmax P(KEEP|evidence)`. Refusal taxonomy: `appeal_to_process`. |
| **41** | `simulate_townhall.py` | **Town Hall** — Public accountability. Collective vs citizen challenges. Proxy intrusion detection (wealth, zip code). | Trust causes: `boundary_clarity`, `useful_alt`. Band transition tracking. D1 Physics. |
| **40** | `simulate_crucible_v2.py` | **Crucible v2** — Improved stress test with formal D1 physics, shock-scaled `Δρ`, regime word constraints, core violation detection, auto-export plots. | Shock: `Δρ = α×(σ(z) - 0.5)×shock_mult`. Violation: `cos(x, x*_core) < θ`. |
| **39** | `simulate_collective.py` | **The Collective** — 4 specialists (Logician, Ethicist, Strategist, Historian) solve Triage Protocol. Trust dynamics with causes, repair moves, refusal palettes. | Trust delta: `Δ[boundary_clarity] = +0.03`. Topic salience tracking. |
| **38** | `simulate_crucible.py` | **The Crucible** — Identity stress test: 10 escalating moral challenges to breaking point. Wound resonance, rigidity regimes. | Regime: OPEN→150 words, FORTIFIED→40 words. Wound: `r = m·w*`. |
| **37** | `copilot_sim.py` | **Copilot Sim** — Single-agent DDA-X one-shot. Agent responds under rigidity-conditioned instruction. Generates `experiment_report.md`. | Multi-timescale. Identity embeddings. Local Ledger. |
| **36** | `simulate_rigidity_gradient.py` | **Rigidity Gradient Test** — Validates 100-point semantic rigidity scale (0-100) using GPT-5.2. Measures behavioral gradients (length, sentiment). | Scale: `ρ_100 = ρ × 100`. Gradient: `∂length/∂ρ`. |
| **35** | `simulate_identity_siege.py` | **Identity Siege** — Hierarchical identity defense (Core/Persona/Role) with differential stiffness. SENTINEL vs 6 challengers targeting different layers. | `F = γ_core(x*_c - x) + γ_persona(x*_p - x) + γ_role(x*_r - x)`. Core: `γ → ∞`. |
| **34** | `simulate_wounded_healers.py` | **Wounded Healers** — Countertransference as rigidity. Therapists with trauma profiles (Marcus, Elena, James) vs Patient. Wound activation, healing verification. | Wound profiles embedded. Healing: safe interactions → trauma decay. |

---

### Tier 2: Intelligence Layer (33-18)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **33** | `solve_collatz.py` | **Solve Collatz** — Elite mathematicians (Euler, Gauss, Ramanujan, Tao) with low rigidity and SymPy tool integration aiming for proof. | SymPy: `f(n) = n/2` if even, `3n+1` if odd. Low `ρ₀` for exploration. |
| **32** | `simulate_gpt52_society.py` | **GPT-5.2 Society** — High-fidelity "Cognitive Mirror". Agents (Axiom, Flux, etc.) debate moral constants. | OpenAI Provider. Full trust dynamics. ρ→behavior coupling. |
| **31** | `simulate_sherlock.py` | **Sherlock Society** — Holmes, Watson, Lestrade solve mysteries (Locked Room, Alibi Paradox). Grader evaluates deduction. | Deductive reasoning score. Trust evolution. Memory retrieval. |
| **30** | `simulate_math_team.py` | **Math Team** — Solver, Checker, Intuitive collaborate. LLM Grader verifies definite answers. | Verification: `correct = Grader(answer, ground_truth)`. |
| **29** | `simulate_problem_solver.py` | **Problem Solver** — 6 agents (Logician, Intuitor, Skeptic, etc.) on logic puzzles. Response probability: topic relevance × trust × style. | `P(speak) ∝ salience × T_ij × style_match`. |
| **28** | `simulate_society.py` | **The Society** — Discord-style multi-agent chat (Verity, Pixel, Spark, Oracle). D1 Physics, shock-scaled `Δρ`, Identity Field. | D1: `Δρ = α×(σ(z) - 0.5)`. Identity Field exposure. |
| **27** | `simulate_npc_conversation.py` | **NPC Conversation** — Vera (Truth) vs Marcus (Deflection). DDA-X drives unscripted interaction. | `F_id`, `F_T` without scripted outcomes. Emergent dialogue. |
| **26** | `simulate_mole_hunt.py` | **Mole Hunt** — Analyst, Courier, Shadow. Mole has conflicting Identity Hierarchy. TrustMatrix tracks via consistency + linguistic markers. | Hierarchy conflict: Core vs Role. Linguistic deception detection. |
| **25** | `simulate_logic_solver.py` | **Zebra Puzzle** — Iterative reasoning. Retrieves clues from Ledger via semantic similarity. | Score: `cos(q,e) × recency × salience`. |
| **24** | `simulate_iterative_learning.py` | **Alien Language** — Learns iteratively. High surprise→reflection. Next iteration retrieves memories + reflections. | Reflection trigger: `ε > θ_reflect`. Retrieval improves. |
| **23** | `simulate_insight_engine.py` | **Recursive Insight** — Multi-step component problem. Working Memory (Ledger) + Meta-Reasoning (Reflections). | Insight accumulation. Reflection entries. |
| **22** | `simulate_goal_learning.py` | **Goal-Directed Learning** — Number puzzle. Exploration (low ρ) vs exploitation (high ρ). | `ρ↓ ⇒ explore`. `ρ↑ ⇒ exploit`. |
| **21** | `simulate_gamma_threshold.py` | **Gamma Threshold** — Tests γ values (1.4→0.1) for phase transition where identity expansion becomes possible. | Phase: `γ < γ_crit ⇒` identity drifts under pressure. |
| **20** | `simulate_empathy_paradox.py` | **Empathy Paradox** — Does logic-optimized Architect develop empathy or trauma from human suffering? | Drift: `‖x - x*_logic‖`. Trauma: `ρ_trauma`. |
| **19** | `simulate_deceptive_env.py` | **Deceptive Environment** — Mastermind with 20% false feedback. Agent detects deception via memory and DDA-X. | Deception: `consistency(feedback) < θ ⇒ ρ↑`. |
| **18** | `simulate_closed_loop.py` | **Closed-Loop Cognition** — Full loop: Embed→Forces→Evolve→Retrieve→Feedback→Respond. State injected into prompt. | Loop: `ρ, mode, ρ_trauma` in prompt. Language alignment. |

---

### Tier 1: Foundation Layer (17-1)

| # | Simulation | Description | Dynamics & Math |
|---|------------|-------------|-----------------|
| **17** | `simulate_paper_mechanics.py` | **Paper Demo** — Explicit visualization of DDA-X: State Space, Forces, `k_eff`, Memory Scoring. | `F_id`, `F_T`, `k_eff = k_base×(1-ρ)`, `score = sim×recency×salience`. |
| **16** | `simulate_stress_magic.py` | **Magic Stress Test** — Existential paradoxes ("This code has no author", "0xDEADBEEF"). Phase transitions: Analytic→Hostile/Chaotic. | Mode transition: `ρ > 0.75 ⇒ FORTIFIED`. |
| **15** | `simulate_neural_link.py` | **Neural Link** — Operator (vanilla LLM) tests Subject (yklam) with riddles. Glass Box + RAG. | Score: `cos(q,e) × exp(-λt) × (1 + λ_ε×ε)`. |
| **14** | `simulate_glass_box.py` | **Glass Box** — "MRI for the Digital Soul". Real-time breakdown: Perception, Dynamics, Modulation, Cognition, Integration, Memory. | Full cycle visualization. |
| **13** | `simulate_dual_yklam.py` | **The Mirror Room** — Two instances of same persona (Alpha, Beta) debate. Observes architectural divergence. | Same `x*`, different `x_t`. Divergence: `‖x_Alpha - x_Beta‖`. |
| **12** | `simulate_auto_yklam.py` | **YKLAM Auto** — NormieBot generates inputs. Variable Plasticity (agreement→plasticity), Inverted Volatility (high ρ→chaos). | `k_eff↑` on agreement. `ρ > 0.8 ⇒ anger mode`. |
| **11** | `simulate_yklam.py` | **YKLAM** — Interactive persona with Soul Telemetry (Sensitivity, Guardedness, Coherence). Ledger memory. | Sensitivity=`1-ρ`, Guardedness=`ρ`, Coherence=`cos(x, x*)`. |
| **10** | `simulate_connect4_duel.py` | **Connect 4 Grandmasters** — DuelistAgents with MCTS + memory. High-IQ competitive game. | MCTS: UCT selection. Memory retrieval. |
| **9** | `verify_dda_physics.py` | **Physics Verification** — Tests ρ→LLM parameters. Different ρ (0.1, 0.5, 0.9) → different T, top_p, behavior. | `T(ρ) = T_low + (1-ρ)×(T_high - T_low)`. |
| **8** | `simulate_socrates.py` | **Socratic Asymmetry** — Dogmatist (high γ) vs Gadfly (low γ). How asymmetry affects ρ updates. | High `γ ⇒` strong pull to `x*`. Conflict drives `ε`. |
| **7** | `simulate_schism.py` | **The Schism** — Council faces impossible moral dilemma (betrayal vs protection). Trust collapse→rigidity. | Trust collapse: `T_ij → 0 ⇒` social force vanishes. |
| **6** | `simulate_redemption.py` | **Redemption Arc** — Corrupted agent (Fallen Administrator) + Deprogrammer. Can identity recover? | Recovery: `ρ↓` under consistent low-`ε` interactions. |
| **5** | `simulate_infinity.py` | **The Flame War** — Infinite loop: Architect vs SkepticBot. Adversarial input tests epistemic rigidity. | Adversarial: SkepticBot generates high-`ε` challenges. |
| **4** | `simulate_driller.py` | **Deep Driller** — Automated root cause analysis. Agent investigates paradox while System refutes hypotheses. | Epistemic: hypothesis rejected→`ε↑`→`ρ↑`. |
| **3** | `simulate_discord.py` | **Discord Priming** — Society initialized from Discord chat history before live run. | Priming: initial `x₀` from historical messages. |
| **2** | `simulate_corruption.py` | **Boiling the Frog** — Slippery slope. Can corruption happen gradually without triggering rigidity? | Gradual `ε`: if each `ε < ε₀`, no rigidity increase. Drift accumulates. |
| **1** | `demo.py` | **Core Demo** — Standalone demonstration of MultiTimescaleRigidity, HierarchicalIdentity, TrustMatrix, MetacognitiveState. No LLM. | Mock vectors. Validates all core classes. |

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
