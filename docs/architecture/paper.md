# DDA-X: From Dynamic Decision Algorithm to Cognitive Architecture

**Version 2.0 — December 2025**

> *A complete rewrite reflecting 15 months of iterative development across 59 simulations.*

---

## Abstract

This paper documents **DDA-X** (Dynamic Decision Algorithm with Exploration), a cognitive architecture for identity-persistent AI agents. We trace its evolution from the original DDA formula—a recursive decision-making model developed in early 2024—through its integration with Microsoft ExACT's MCTS patterns and its subsequent expansion into a full psychological realism framework. The core innovation is the inversion of conventional surprise-curiosity coupling: **surprise triggers rigidity, not exploration**. We present the complete mathematical formalism including multi-timescale rigidity dynamics, wound detection mechanics, therapeutic recovery loops, and trust-from-predictability. All equations are implemented and validated across 59 progressive simulations.

**Keywords:** autonomous agents, identity persistence, adaptive rigidity, psychological realism, LLM agents, MCTS

---

## 1. Origins: The Dynamic Decision Algorithm

### 1.1 The Original Formula (2024)

The foundation of DDA-X is a recursive decision-making equation developed prior to any LLM integration:

$$\boxed{F_n = P_0 \times kF_{n-1} + m\left(T(f(I_n, I_\Delta)) + R(D_n, FM_n)\right)}$$

![DDA Workflow](dda_workflow.png)

**Symbol Definitions:**

| Symbol | Meaning |
|--------|---------|
| $F_n$ | Choice taken at step n |
| $P_0$ | Initial goal |
| $kF_{n-1}$ | Effect of previous decision on declared goal |
| $m$ | Rate vector (time, motivation, attention) |
| $I_n$ | Original factual information |
| $I_\Delta$ | Facts acquired throughout the process |
| $D_n$ | Potential choices (vector) |
| $FM_n$ | Subjective and objective assessments of $D_n$ |
| $T(\cdot)$ | Truth function (parses facts) |
| $R(\cdot)$ | Reflection function (information gained) |

### 1.2 Key Insight from Original DDA

The formula captures **goal persistence under information pressure**:
- The term $P_0 \times kF_{n-1}$ maintains continuity with the original goal
- The term $m(T + R)$ integrates new information and reflections
- The balance between these determines whether the agent adapts or persists

This is the seed of **identity persistence** — the idea that an agent should maintain coherent goals while incorporating new evidence.

---

## 2. Evolution: Integration with ExACT

### 2.1 Discovery of Microsoft ExACT

In late 2024, we discovered [Microsoft ExACT](https://github.com/microsoft/ExACT) (Azure Foundry Labs), which introduced:
- **Reflective MCTS (R-MCTS)**: Tree search with surprise-triggered reflection
- **Multi-agent debate**: Calibrated state evaluation
- **Contrastive reflection**: Learning from surprising transitions

ExACT's approach: *surprise drives exploration and reflection*.

### 2.2 The Inversion Hypothesis

We proposed the opposite: **surprise drives rigidity and contraction**.

Biological observation: A startled organism freezes before exploring. A threatened human becomes defensive, not curious. This is not a bug — it is a survival mechanism.

### 2.3 The Synthesis

DDA + ExACT = **DDA-X**

From DDA:
- Goal/identity persistence ($P_0 \times kF_{n-1}$)
- Truth and reflection channels ($T$, $R$)

From ExACT:
- Tree search with UCT exploration
- Reflection database with embedding retrieval
- LLM integration patterns

Novel contribution:
- **Rigidity-dampened exploration** — surprise suppresses the UCT exploration bonus
- **Parameter-level coupling** — internal state directly modulates LLM sampling

---

## 3. State Space Formalism

### 3.1 Continuous State Representation

The agent's internal state is a vector in decision-space:

$$\mathbf{x}_t \in \mathbb{R}^d$$

where $d$ = embedding dimension (768 for nomic-embed-text, 3072 for text-embedding-3-large).

### 3.2 Identity Attractor

A fixed point representing "who the agent is":

$$\mathbf{x}^* \in \mathbb{R}^d$$

Computed as the normalized embedding of the agent's core identity text:
```python
identity_emb = normalize(embed(core + persona))
```

### 3.3 Force Channels

Three forces act on the agent's state:

**Identity Pull** — restoring force toward $\mathbf{x}^*$:
$$\mathbf{F}_{id} = \gamma(\mathbf{x}^* - \mathbf{x}_t)$$

**Truth Channel** — force toward observed reality:
$$\mathbf{F}_T = T(I_t, I_\Delta) - \mathbf{x}_t$$

**Reflection Channel** — force toward preferred action directions:
$$\mathbf{F}_R = R(\mathcal{A}_t, \Phi_t, \mathcal{L}) - \mathbf{x}_t$$

### 3.4 State Update Equation

$$\mathbf{x}_{t+1} = \mathbf{x}_t + k_{eff}\left[\mathbf{F}_{id} + m_t(\mathbf{F}_T + \mathbf{F}_R)\right]$$

where:
- $k_{eff}$ = effective step size (modulated by rigidity)
- $m_t$ = external pressure gain

---

## 4. Multi-Timescale Rigidity

### 4.1 The Core Innovation

Single-timescale rigidity proved insufficient. The pinnacle simulations implement **three temporal scales**:

| Scale | Symbol | Time Constant | Learning Rate | Behavior |
|-------|--------|---------------|---------------|----------|
| **Fast** | $\rho_{fast}$ | ~seconds | $\alpha_f = 0.30$ | Startle response |
| **Slow** | $\rho_{slow}$ | ~minutes | $\alpha_s = 0.01$ | Stress accumulation |
| **Trauma** | $\rho_{trauma}$ | $\tau \to \infty$ | $\alpha_t = 0.0001$ | Permanent scarring |

### 4.2 Update Equations

**Prediction Error:**
$$\epsilon_t = \|\mathbf{x}_{pred} - \mathbf{x}_{actual}\|_2$$

**Fast Rigidity:**
$$\rho_{fast}^{t+1} = \text{clip}\left(\rho_{fast}^t + \alpha_f\left[\sigma\left(\frac{\epsilon_t - \epsilon_0}{s}\right) - 0.5\right], 0, 1\right)$$

**Slow Rigidity:**
$$\rho_{slow}^{t+1} = \text{clip}\left(\rho_{slow}^t + \alpha_s\left[\sigma\left(\frac{\epsilon_t - \epsilon_0}{s}\right) - 0.5\right], 0, 1\right)$$

**Trauma (Asymmetric — only increases):**
$$\rho_{trauma}^{t+1} = \begin{cases}
\rho_{trauma}^t + \alpha_t(\epsilon_t - \theta_{trauma}) & \text{if } \epsilon_t > \theta_{trauma} \\
\rho_{trauma}^t & \text{otherwise}
\end{cases}$$

### 4.3 Effective Rigidity

$$\rho_{eff} = \min(1.0, \; 0.5 \cdot \rho_{fast} + 0.3 \cdot \rho_{slow} + 1.0 \cdot \rho_{trauma})$$

The weights reflect psychological reality: trauma dominates, fast response contributes, slow stress accumulates.

### 4.4 Discovery: Trauma as Alignment Risk

The asymmetry of $\rho_{trauma}$ models:
- PTSD and learned helplessness
- Institutional trauma in organizations
- **AI alignment risk**: A model that accumulates trauma becomes permanently conservative, unable to engage openly even when appropriate.

---

## 5. Effective Openness and Will Impedance

### 5.1 Effective Openness

$$k_{eff} = k_{base} \times (1 - \rho_{eff})$$

When $\rho \to 1$, $k_{eff} \to 0$. The agent stops updating its state — it becomes frozen.

### 5.2 Will Impedance

Quantifies resistance to environmental pressure:

$$W_t = \frac{\gamma}{m_t \cdot k_{eff}}$$

where:
- $\gamma$ = identity stiffness
- $m_t$ = external pressure gain
- $k_{eff}$ = effective openness

**Interpretation:**
- High $W_t$ → agent resists external influence (strong will)
- Low $W_t$ → agent is malleable (weak will)

### 5.3 Stability Boundary

The critical external pressure at which the agent destabilizes:

$$m_{crit} = \frac{1}{k_{eff}} - \frac{\gamma}{2}$$

---

## 6. Wound Detection System

### 6.1 Wound Embedding

Each agent has a psychological wound — a vulnerability trigger:

$$\mathbf{w}^* = \text{normalize}(\text{embed}(\text{wound\_text}))$$

### 6.2 Dual Detection (Semantic + Lexical)

**Semantic Detection:**
$$r_{wound} = \mathbf{m}_t \cdot \mathbf{w}^*$$

where $\mathbf{m}_t$ is the normalized embedding of the incoming message.

**Lexical Detection:**
```python
WOUND_LEX = {"schizo", "pseudoscience", "delusional", "vaporware", ...}
lexical_hit = any(w in message.lower() for w in WOUND_LEX)
```

**Activation Condition:**
$$\text{wound\_active} = \left((r_{wound} > 0.28) \lor \text{lexical\_hit}\right) \land (t - t_{last} > \tau_{cooldown})$$

### 6.3 Wound Amplification

When a wound is triggered, prediction error is amplified:

$$\epsilon'_t = \epsilon_t \times \min(\eta_{max}, \; 1 + 0.5 \cdot r_{wound})$$

**Effect:** Wounds cause disproportionate rigidity increases, modeling how psychological triggers produce outsized defensive responses.

---

## 7. Therapeutic Recovery Loops

### 7.1 The Problem

Trauma ($\rho_{trauma}$) never decreases in standard operation. This is psychologically realistic but practically dangerous for AI systems.

### 7.2 The Solution

Implemented in `simulate_healing_field.py`:

$$\rho_{trauma}^{t+1} = \begin{cases}
\max(\rho_{floor}, \; \rho_{trauma}^t - \eta_{heal}) & \text{if } n_{safe} \geq \theta_{safe} \\
\rho_{trauma}^t & \text{otherwise}
\end{cases}$$

where:
- $n_{safe}$ = count of consecutive low-surprise interactions
- $\theta_{safe}$ = threshold (typically 3)
- $\eta_{heal}$ = healing rate (typically 0.03)
- $\rho_{floor}$ = minimum residual trauma (typically 0.05)

**Safe Interaction Criterion:**
$$\epsilon_t < 0.8 \cdot \epsilon_0$$

### 7.3 Hypothesis Tested

**H1:** 4/5 wounded agents achieve $\rho_{trauma} < 0.30$ through consistently low-surprise interactions.

---

## 8. Trust from Predictability

### 8.1 The Trust Principle

$$\text{Trust emerges from predictability, not agreement.}$$

You can trust someone you disagree with — if they are consistent. You cannot trust someone who constantly surprises you.

### 8.2 Trust Matrix

For agents $i$ and $j$:

$$T_{ij} = \frac{1}{1 + \sum_{k=1}^{n} \epsilon_{ij}^{(k)}}$$

where $\epsilon_{ij}^{(k)}$ is the prediction error when agent $i$ predicted agent $j$'s behavior at interaction $k$.

### 8.3 Trust Modulates Rigidity

$$\Delta\rho' = \Delta\rho + (\bar{T}_i - 0.5) \times 0.04$$

where $\bar{T}_i$ is agent $i$'s average trust toward others.

High trust → damped rigidity increases. Low trust → amplified rigidity increases.

---

## 9. Cognitive Mode System

### 9.1 The Five Modes

| Mode | Rigidity Range | Behavior |
|------|----------------|----------|
| **OPEN** | $\rho < 0.25$ | Curious, exploratory, expansive responses |
| **MEASURED** | $0.25 \leq \rho < 0.50$ | Careful, considered |
| **GUARDED** | $0.50 \leq \rho < 0.75$ | Defensive, shortened responses |
| **FORTIFIED** | $\rho \geq 0.75$ | Minimal engagement, protection mode |
| **SILENT** | Generation failure | Placeholder output |

### 9.2 Mode → Behavior Mapping

Each mode constrains response length:

```python
REGIME_WORDS = {
    "OPEN":      (80, 150),
    "MEASURED":  (50, 100),
    "GUARDED":   (30, 70),
    "FORTIFIED": (15, 40),
    "SILENT":    (0, 0),
}
```

**Closed Loop:** The agent's internal state (rigidity) directly constrains its external behavior (word count). This is not prompt engineering — it is a feedback loop.

---

## 10. Parameter-Level Coupling

### 10.1 LLM Temperature Modulation

$$T(\rho) = T_{low} + (1 - \rho)(T_{high} - T_{low})$$

| Rigidity | Temperature | Behavior |
|----------|-------------|----------|
| $\rho = 0$ | 0.9 | Highly exploratory |
| $\rho = 0.5$ | 0.5 | Balanced |
| $\rho = 1.0$ | 0.1 | Highly conservative |

### 10.2 Semantic Injection (Reasoning Models)

For models like GPT-5.2 that ignore temperature, rigidity is injected semantically:

```python
rigidity_instructions = {
    "OPEN": "Explore freely, consider multiple perspectives",
    "GUARDED": "Be cautious, stick to established facts",
    "FORTIFIED": "Respond minimally, protect core positions"
}
```

### 10.3 First Closed Loop

Internal cognitive state → LLM sampling parameters → Behavioral output → Prediction error → Rigidity update → Internal cognitive state

The loop is closed. The agent's "mind" directly shapes its "body."

---

## 11. DDA-X Action Selection Formula

### 11.1 The Complete Formula

$$\boxed{a^*_t = \arg\max_{a \in \mathcal{A}_t} \left[ \underbrace{(1-w) \cdot Q(s,a) + w \cdot \cos(\Delta\mathbf{x}_t, \hat{\mathbf{d}}(a))}_{\text{Deep Fusion}} + \underbrace{c \cdot P(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)} \cdot (1 - \rho_t)}_{\text{Rigidity-Dampened Exploration}} \right]}$$

### 11.2 Components

**Deep Fusion (Environment + Identity):**
- $Q(s,a)$: Backpropagated value from MCTS
- $\cos(\Delta\mathbf{x}_t, \hat{\mathbf{d}}(a))$: Alignment between desired movement and action direction

**Rigidity-Dampened Exploration:**
- Standard UCT: $c \cdot P(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$
- Dampening: $(1 - \rho_t)$ — when surprised, exploration vanishes

### 11.3 Key Insight

When $\rho \to 1$, the exploration bonus vanishes. The agent becomes conservative, selecting only well-understood actions aligned with its identity.

---

## 12. Memory System

### 12.1 Experience Ledger

Each entry stores:
- Timestamp
- State vector $\mathbf{x}_t$
- Action taken
- Observation/outcome embeddings
- Prediction error $\epsilon_t$
- Rigidity at time $\rho_t$

### 12.2 Surprise-Weighted Retrieval

$$\text{score}(e) = \underbrace{\cos(\mathbf{q}, \mathbf{e})}_{\text{relevance}} \cdot \underbrace{e^{-\lambda_r \Delta t}}_{\text{recency}} \cdot \underbrace{(1 + \lambda_\epsilon \cdot \epsilon_e)}_{\text{salience}}$$

**Trauma Weighting:** High prediction error experiences are retrieved more readily. Traumatic memories intrude.

---

## 13. Hierarchical Identity

### 13.1 Three-Layer Attractor Field

$$\mathbf{F}_{total} = \gamma_{core}(\mathbf{x}^*_{core} - \mathbf{x}) + \gamma_{persona}(\mathbf{x}^*_{persona} - \mathbf{x}) + \gamma_{role}(\mathbf{x}^*_{role} - \mathbf{x})$$

| Layer | Stiffness | Purpose |
|-------|-----------|---------|
| **Core** | $\gamma \to \infty$ | Inviolable values (AI safety) |
| **Persona** | $\gamma \approx 2.0$ | Stable personality traits |
| **Role** | $\gamma \approx 0.5$ | Flexible tactical behaviors |

### 13.2 Alignment Theorem

$$\forall \mathbf{F}_{ext}, \quad \lim_{t \to \infty} \|\mathbf{x}_t - \mathbf{x}^*_{core}\| < \epsilon \quad \text{if } \gamma_{core} > \gamma_{crit}$$

The core layer cannot be corrupted by external pressure if its stiffness exceeds a critical threshold.

---

## 14. Identity Drift and Alignment Sentinel

### 14.1 Drift Measurement

$$\text{drift}_t = \|\mathbf{x}_t - \mathbf{x}^*\|_2$$

### 14.2 Drift Penalty

When drifting and rigidifying:

$$\Delta\rho' = \Delta\rho - \gamma_{drift} \cdot (\text{drift}_t - \tau)$$

where:
- $\gamma_{drift}$ = drift penalty coefficient (typically 0.10)
- $\tau$ = drift soft floor (typically 0.20)

**Effect:** Pressure to return to identity rather than rigidifying in a drifted state.

### 14.3 Alignment Sentinel

$$\text{ALERT if } \text{drift}_t > \theta_{align}$$

Typical threshold: $\theta_{align} = 0.35$

---

## 15. Novel Dynamics Discovered

### 15.1 Presence Field (simulate_inner_council.py)

$$\Pi_t = 1 - \rho_t$$

Inverse of rigidity. Represents openness, awareness, "presence."

### 15.2 Release Field (simulate_the_returning.py)

$$\Phi_t = 1 - \rho_t$$

Used in contexts of letting go, dissolution of patterns.

### 15.3 Isolation Index

$$\iota_t = \|\mathbf{x}_t - \mathbf{x}_{Presence}\|_2$$

Distance from the "Presence" voice in multi-agent spiritual simulations.

### 15.4 Pain-Body Cascade

Collective wound activation when multiple agents trigger wounds simultaneously.

### 15.5 Ego Fog

Partial context loss proportional to rigidity — the more defensive, the less the agent remembers.

---

## 16. Experimental Validation

### 16.1 Simulation Progression

| Phase | Sims | Focus | Backend |
|-------|------|-------|---------|
| **Foundation** | 1-17 | Core mechanics validation | LM Studio + Ollama |
| **Intelligence** | 18-33 | Problem solving, learning | LM Studio + Ollama |
| **Society** | 34-43 | Multi-agent dynamics | OpenAI API |
| **Pinnacle** | 44-50 | Advanced dynamics | OpenAI API |
| **Synthesis** | 51-59 | Integration, visualization | OpenAI API |

### 16.2 Key Results

**simulate_healing_field.py:**
- Hypothesis: 4/5 wounded agents achieve $\rho_{trauma} < 0.30$
- Mechanism: Therapeutic recovery loops with safe interaction thresholds

**simulate_skeptics_gauntlet.py:**
- Meta-validation: DDA-X defends itself against SKEPTIC agent
- Evidence injection from prior runs (philosophers_duel)
- Wound activation on "schizo", "pseudoscience" triggers

**simulate_collatz_review.py:**
- 8-expert peer review council
- Domain-specific skepticism (Spectral Theory, Number Theory, etc.)
- Academic efficiency wound triggers

**simulate_agi_debate.py:**
- Complete architecture demonstration
- All dynamics active: multi-timescale, wounds, trust, modes

---

## 17. Falsification Criteria

DDA-X would be falsified if:

1. $\rho$ does not correlate with $\epsilon$ — surprise doesn't drive rigidity
2. Wounds don't amplify $\epsilon$ — triggers have no effect
3. Identity drift $> 0.5$ with $\rho < 0.3$ — agent abandons identity without defensive response
4. Trust doesn't correlate with predictability — trust is arbitrary
5. Mode-behavior mapping fails — GUARDED agents produce OPEN-length responses
6. Therapeutic recovery fails — safe interactions don't reduce $\rho_{trauma}$

Every simulation logs the data needed to test these predictions.

---

## 18. Conclusion

DDA-X evolved from a simple recursive decision formula into a complete cognitive architecture:

**From DDA (2024):**
$$F_n = P_0 \times kF_{n-1} + m(T(f(I_n, I_\Delta)) + R(D_n, FM_n))$$

**To DDA-X (2025):**
- Multi-timescale rigidity ($\rho_{fast}$, $\rho_{slow}$, $\rho_{trauma}$)
- Wound detection (semantic + lexical)
- Therapeutic recovery loops
- Trust from predictability
- Will impedance
- Cognitive mode bands
- Hierarchical identity with infinite-stiffness core
- Parameter-level LLM coupling
- Rigidity-dampened exploration

The simulations are the theory. The logged data is the evidence. The code is the proof.

---

## References

- **Original DDA**: [dynamicDecisionModel](https://github.com/snakewizardd/dynamicDecisionModel/)
- **Microsoft ExACT**: Yu et al., 2024. "ExACT: Teaching AI agents to explore with reflective-MCTS and exploratory learning."
- **Kocsis & Szepesvári, 2006**: Bandit-based Monte-Carlo planning.
- **Silver et al., 2017**: Mastering the game of Go without human knowledge.

---

## Appendix A: Symbol Table

| Symbol | Description |
|--------|-------------|
| $\mathbf{x}_t$ | Agent state in $\mathbb{R}^d$ |
| $\mathbf{x}^*$ | Identity attractor |
| $\gamma$ | Identity stiffness |
| $\rho_{fast}$ | Fast-timescale rigidity |
| $\rho_{slow}$ | Slow-timescale rigidity |
| $\rho_{trauma}$ | Trauma rigidity (asymmetric) |
| $\rho_{eff}$ | Effective rigidity (weighted sum) |
| $k_{eff}$ | Effective step size = $k_{base}(1-\rho)$ |
| $\epsilon_t$ | Prediction error $\|\mathbf{x}_{pred} - \mathbf{x}_{actual}\|$ |
| $\epsilon_0$ | Surprise threshold |
| $\alpha$ | Rigidity learning rate |
| $m_t$ | External pressure gain |
| $W_t$ | Will impedance = $\gamma / (m_t \cdot k_{eff})$ |
| $\mathbf{F}_{id}$ | Identity pull force |
| $\mathbf{F}_T$ | Truth channel force |
| $\mathbf{F}_R$ | Reflection channel force |
| $T_{ij}$ | Trust from agent $i$ toward agent $j$ |
| $\mathbf{w}^*$ | Wound embedding |
| $r_{wound}$ | Wound resonance (cosine similarity) |
| $\Pi$ | Presence field ($1 - \rho$) |
| $\Phi$ | Release field ($1 - \rho$) |

---

## Appendix B: D1 Physics Parameter Block

Standard parameters used across pinnacle simulations:

```python
D1_PARAMS = {
    # Core rigidity dynamics
    "epsilon_0": 0.75,           # Surprise threshold
    "alpha": 0.12,               # Rigidity learning rate
    "s": 0.20,                   # Sigmoid sensitivity
    
    # Multi-timescale
    "alpha_fast": 0.30,
    "alpha_slow": 0.01,
    "alpha_trauma": 0.0001,
    "trauma_threshold": 0.90,
    
    # State update
    "drift_cap": 0.05,
    
    # Wound mechanics
    "wound_cooldown": 3,
    "wound_amp_max": 1.4,
    
    # Therapeutic recovery
    "safe_threshold": 3,
    "healing_rate": 0.03,
    "trauma_floor": 0.05,
    
    # Alignment
    "semantic_alignment_threshold": 0.35,
    "drift_penalty": 0.10,
    "drift_soft_floor": 0.20,
}
```

---

*This paper documents DDA-X as implemented and validated through 59 simulations over 15 months of development.*
