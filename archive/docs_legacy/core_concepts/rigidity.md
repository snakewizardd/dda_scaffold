# Cognitive Safety: Rigidity & Metacognition

> "Adaptive constraints for uncertain environments."

DDA-X introduces **Rigidity ($\rho$)** as the central variable of cognitive safety. It is a dynamic parameter that modulates the agent's exploration-exploitation trade-off based on prediction error.

## The Cognitive Loop

We have implemented a closed-loop feedback system between "State" and "Inference":

1.  **Surprise ($\epsilon$)**: The divergence between expected state ($\vec{x}_{pred}$) and observed state ($\vec{x}_{obs}$).
2.  **Rigidity Spike ($\Delta \rho$)**: Surprise triggers a rapid increase in $\rho$ (defensive contraction).
3.  **Parameter Binding**: High $\rho$ physically alters the LLM's neural sampling:
    *   **Temperature drops**: Exploration is suppressed.
    *   **Top-P narrows**: Confidence interval tightens.
    *   **Repetition allows**: Safety in deterministic patterns.

## Multi-Timescale Dynamics

Rigidity operates on three distinct timescales to model different forms of adaptation:

*   **$\rho_{fast}$ (Startle)**: Immediate reaction to novel stimuli. Fast decay.
*   **$\rho_{slow}$ (Stress)**: Accumulating environmental pressure. Slow decay.
*   **$\rho_{trauma}$ (Deviation)**: Asymmetric accumulation that represents permanent adaptation to extreme events.

## Metacognition: The Monitor

Critically, the DDA-X agent possesses a **Metacognitive Layer**.

*   **The Check**: Before every action, the system monitors $\rho$.
*   **The Report**: If $\rho > 0.7$, the monitor flags: *"High Rigidity State. Reliability compromised."*
*   **Protect Mode**: The agent halts autonomous decision making and requests operator intervention (Human-in-the-loop).

This ensures **Honest AI** behavior.
