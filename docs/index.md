# DDA-X: Surprise → Rigidity → Contraction
**A Dynamical Framework for Agent Behavior**

> **Standard RL:** surprise → exploration  
> **DDA-X:** **surprise → rigidity → contraction**

DDA-X is a cognitive-dynamics framework in which **prediction error (surprise) increases rigidity** (defensive contraction) rather than immediately driving exploration. Across 59 simulations, agents maintain a continuous latent state (via 3072-D text embeddings), measure surprise as embedding-space prediction error, and bind that internal rigidity to externally visible behavior—constraining bandwidth, altering decoding styles, and injecting semantic "cognitive state" instructions.

---

## The Core Insight

In standard Reinforcement Learning, surprise is often treated as an "intrinsic motivation" signal (curiosity) to explore. DDA-X inverts this. It models the behavior of organisms that **freeze and contract** when startled.

1.  **Startle Response**: High prediction error ($\epsilon$) triggers a spike in rigidity ($\rho$).
2.  **Contraction**: High rigidity reduces the "step size" of state updates ($k_{\text{eff}}$) and constrains the "bandwidth" of output (word counts, topic variance).
3.  **Safety & Recovery**: Only when prediction error remains low for a sustained period does rigidity decay, allowing the system to reopen (Trauma Decay).

## Core Equations

### Rigidity Update
$$
z_t = \frac{\epsilon_t - \epsilon_0}{s}, \quad \Delta\rho_t = \alpha(\sigma(z_t) - 0.5)
$$

### Effective Step Size
$$
k_{\text{eff}} = k_{\text{base}} (1 - \rho_t)
$$

### State Evolution
$$
x_{t+1} = x_t + k_{\text{eff}} \cdot \eta \Big( \gamma(x^* - x_t) + m(F_T + F_R) \Big)
$$

---

## Explore the Documentation

*   **[Paper v2.0](architecture/paper.md)**: The full theoretical framework with rigorous math.
*   **[Implementation Mapping](architecture/ARCHITECTURE.md)**: See how the theory runs in actual Python code.
*   **[Simulation Chronology](simulation_chronology.md)**: Catalog of all 59 simulations.
*   **[Research Review](gpt52_review/gpt52_feedback_final.md)**: Independent review by GPT-5.2 reasoning models.

---

*(c) 2025 DDA-X Research Team*
