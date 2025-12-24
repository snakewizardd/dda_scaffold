# DDA-X: Surprise → Rigidity → Contraction
*A Dynamical Framework for Embedding-Space Agents with Rigidity-Bound Generation*

## Abstract

DDA-X is a cognitive dynamics framework in which prediction error (surprise) induces defensive rigidity rather than immediate exploration. In contrast to standard reinforcement learning heuristics that treat surprise as an exploration bonus, DDA-X models startled systems as **contracting**—reducing behavioral bandwidth and state update magnitude—until safety and predictability permit reopening. We operationalize DDA-X in a family of simulations spanning adversarial dialogue, therapeutic recovery, multi-agent convergence, and embodied collision dynamics. The framework combines (i) a continuous state space with identity attractors, (ii) rigidity-modulated effective step size, (iii) multi-timescale rigidity decomposition (fast/slow/trauma), and (iv) content-addressable wound activation.

## 1. State Representation

Let $x_t \in \mathbb{R}^d$ be an agent's internal state at time $t$, where $d=3072$ (using `text-embedding-3-large`).

Each agent maintains:
*   $x_t$: Current state (initialized near an identity attractor $x^*$)
*   $x^{\text{pred}}_t$: Predicted embedding of the agent's next action/utterance
*   $e(a_t)$: Embedding of the generated utterance $a_t$

In multiple simulations, $x^{\text{pred}}$ is an exponentially smoothed forecast of the agent's own past output embeddings:

$$
x^{\text{pred}}_{t+1} = (1-\beta)x^{\text{pred}}_{t} + \beta \, e(a_t)
$$

(e.g., $\beta=0.3$ in `simulate_the_returning.py`, `simulate_skeptics_gauntlet.py`).

## 2. Surprise as Prediction Error

We define surprise as the Euclidean distance between prediction and reality in semantic space:

$$
\epsilon_t = \lVert x^{\text{pred}}_t - e(a_t) \rVert_2
$$

This acts as the primary driving signal for the cognitive engine.

## 3. Rigidity Dynamics: The Logistic Gate

Rigidity $\rho_t \in [0,1]$ modulates the system's openness to change. We compute an update $\Delta \rho_t$ using a logistic gating function centered at a surprise threshold $\epsilon_0$:

$$
z_t = \frac{\epsilon_t - \epsilon_0}{s}, \qquad \sigma(z) = \frac{1}{1+e^{-z}}
$$

$$
\Delta \rho_t = \alpha(\sigma(z_t) - 0.5)
$$

$$
\rho_{t+1} = \text{clip}(\rho_t + \Delta\rho_t, \, 0, \, 1)
$$

This implements the core DDA-X thesis: **higher surprise $\rightarrow$ higher rigidity**.

### Multi-Timescale Decomposition

To model complex behaviors like startle vs. trauma, we decompose rigidity into three timescales:

1.  **Fast ($\rho_{\text{fast}}$)**: Quick startle response, rapid decay.
2.  **Slow ($\rho_{\text{slow}}$)**: Quantitative stress accumulation.
3.  **Trauma ($\rho_{\text{trauma}}$)**: Asymmetric accumulation (scarring). Only increases when $\epsilon > \theta_{\text{trauma}}$ (unless therapeutic intervention occurs).

Effective rigidity is a weighted sum:

$$
\rho_{\text{eff}} = \min(1, \, w_f \rho_{\text{fast}} + w_s \rho_{\text{slow}} + w_t \rho_{\text{trauma}})
$$

## 4. State Update and Will Impedance

The agent's state evolves under forces, but the *magnitude* of evolution is throttled by rigidity.

$$
x_{t+1} = x_t + k_{\text{eff}} \cdot \eta \Big( \underbrace{\gamma(x^* - x_t)}_{\text{Identity}} + m(\underbrace{F_{\text{truth}}}_{\text{Observation}} + \underbrace{F_{\text{reflection}}}_{\text{Action}}) \Big)
$$

Where the **Effective Step Size** is:

$$
k_{\text{eff}} = k_{\text{base}}(1 - \rho_t)
$$

We define **Will Impedance** $W_t$, quantifying resistance to environmental pressure:

$$
W_t = \frac{\gamma}{m_t \cdot k_{\text{eff}}}
$$

As rigidity $\rho \to 1$, $k_{\text{eff}} \to 0$, and Will Impedance $W_t \to \infty$. The agent becomes immovable.

## 5. Binding Rigidity to LLM Generation

DDA-X binds the internal scalar $\rho$ to the external LLM generation process via `OpenAIProvider`.

### For Reasoning Models (o1 / GPT-5.2)
Since current reasoning models do not support granular sampling parameters (temperature), we inject **Semantic Rigidity Instructions** directly into the cognitive state block of the prompt:

> `[COGNITIVE STATE]: You are in a GUARDED state. Your beliefs are rigid. Limit your response to 50 words. Do not accept new premises without extreme evidence.`

### For Standard Models (GPT-4o)
We modulate sampling parameters:
*   Temperature: $T(\rho) = T_{\min} + (1-\rho)(T_{\max} - T_{\min})$
*   Top-P: Constricted as rigidity increases.
*   Presence Penalty: Reduced to discourage topic drift.

## 6. Wounds: Content-Addressable Threat Priors

A "Wound" is a semantic point of vulnerability $w \in \mathbb{R}^d$. Activation occurs via:

$$
\text{wound\_active} = (\langle e(s_t), w \rangle > \tau_{\cos} \;\lor\; \text{lexical\_match}) \land \text{cooldown\_satisfied}
$$

When active, surprise is amplified, simulating a disproportionate psychological reaction:

$$
\epsilon'_t = \epsilon_t \cdot \min(A_{\max}, \, 1 + c \cdot \langle e(s_t), w \rangle)
$$

## 7. Trust: Hybrid Implementation

While the theoretical ideal for trust is prediction-based ($T_{ij} = \frac{1}{1 + \sum \epsilon}$), current simulations implement a **Hybrid Trust Model**:

*   **Semantic Alignment**: Trust increases when $e(a_t)$ aligns with the partner's previous output (Dialectic).
*   **Civility Gating**: Trust decreases on "unfair" (lexically wounded) engagement.
*   **Coalition Bias**: Trust is initialized based on group topology (Council).

## 8. Therapeutic Recovery

In specific simulations (e.g., `simulate_healing_field.py`), we model trauma decay. If the agent experiences a sustained "Safe Streak" where $\epsilon < 0.8\epsilon_0$:

$$
\rho_{\text{trauma}} \leftarrow \max(\rho_{\min}, \, \rho_{\text{trauma}} - \eta_{\text{heal}})
$$

This provides the mathematical basis for "healing" within the dynamical system.

---

## 9. Unique Contributions vs. Standard RL

1.  **Inverted exploration**: Surprise $\to$ Contraction.
2.  **Multi-timescale defensiveness**: State separability of startle vs. trauma.
3.  **Wounds as state**: Semantic priors that modulate dynamics.
4.  **Identity as attractor**: Use of $\gamma(x^* - x)$ ensures identity persistence.

---

## 10. Relation to RL and LLM Agents

DDA-X differs from typical RL/LLM-agent designs in several fundamental ways:

| Aspect | Standard RL/LLM | DDA-X |
|:---|:---|:---|
| Response to surprise | Explore more (curiosity bonus) | Contract (reduce $k_{\text{eff}}$) |
| Threat modeling | Reward shaping or constraints | Content-addressable wound embeddings |
| Temporal dynamics | Single timescale or none | Fast/Slow/Trauma decomposition |
| Output control | Fixed or random verbosity | Mode bands constrain word counts |
| Recovery from harm | Implicit or absent | Explicit trauma decay dynamics |
| Identity | Stateless (context window only) | Attractor dynamics with stiffness $\gamma$ |

The key insight is that DDA-X treats surprise as a **contraction signal** rather than an **exploration signal**, and makes defensiveness, wounds, and trauma explicit state variables that directly modulate update magnitudes and policy bandwidth.

---

## 11. Limitations and Open Problems

!!! warning "Transparency"
    This section documents known gaps between theory and implementation. See [Known Limitations](../limitations.md) for full details.

### 11.1 Trust Equation Mismatch

The theoretical trust equation $T_{ij} = \frac{1}{1+\sum\epsilon}$ is not implemented in current simulations. The hybrid trust model (Section 7) is the actual implementation.

### 11.2 Calibration Asymmetry

While $\epsilon_0$ and $s$ are calibrated from runtime statistics, wound thresholds ($\tau_{\cos}$), trauma thresholds ($\theta_{\text{trauma}}$), and multi-timescale weights ($w_f, w_s, w_t$) remain hardcoded. These may require domain-specific tuning.

### 11.3 Measurement Validity

The prediction error $\epsilon_t = \|x_{\text{pred}} - e(a_t)\|$ conflates semantic novelty with style/verbosity shifts. Future work could decompose embeddings into content vs. tone components.

### 11.4 Model Dependency

For reasoning models (GPT-5.2, o1), rigidity cannot modulate sampling parameters—semantic injection is the only option. Effectiveness may vary across models.

### 11.5 Open Research Questions

1. Can multi-timescale weights be learned from data?
2. How do thresholds transfer across domains?
3. What safe interaction patterns most effectively heal trauma?

---

## Appendix A: Parameter Schema (D1_PARAMS)

Most simulations use a consistent "physics dictionary" pattern for configuration:

| Parameter | Symbol | Typical Value | Description |
|:---|:---:|:---:|:---|
| `epsilon_0` | $\epsilon_0$ | Calibrated | Surprise baseline (median of early epsilons) |
| `s` | $s$ | Calibrated | Sigmoid steepness (IQR-based) |
| `alpha` | $\alpha$ | 0.05–0.15 | Base learning rate for $\Delta\rho$ |
| `alpha_fast` | $\alpha_f$ | 0.20 | Fast rigidity learning rate |
| `alpha_slow` | $\alpha_s$ | 0.02 | Slow rigidity learning rate |
| `alpha_trauma` | $\alpha_t$ | 0.10 | Trauma accumulation rate |
| `trauma_threshold` | $\theta_t$ | 0.9–1.0 | Epsilon required for trauma increase |
| `wound_cosine_threshold` | $\tau_{\cos}$ | 0.28 | Semantic resonance trigger |
| `wound_cooldown` | — | 3–5 turns | Wound refractory period |
| `drift_cap` | — | 0.05–0.10 | Maximum per-turn state movement |
| `k_base` | $k$ | 0.1–0.3 | Base step size |
| `gamma` | $\gamma$ | 0.1–0.5 | Identity stiffness |

**Note:** Calibration of `epsilon_0` and `s` occurs after a warm-up period using observed prediction error statistics (median + IQR).

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

