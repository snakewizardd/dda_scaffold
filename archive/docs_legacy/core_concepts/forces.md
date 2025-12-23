# The Physics of Choice

> "Decision making as vector equilibrium."

In DDA-X, we abandon the scalar "reward function" of Reinforcement Learning for a **Force-Based Dynamics** model. The agent's trajectory through decision-space is determined by the continuous interplay of three fundamental forces.

## 1. Identity Pull ($F_{id}$)
$$F_{id} = \gamma (\vec{x}^* - \vec{x})$$

The force of **Internal Alignment**. It pulls the agent back towards its defined Reference State ($\vec{x}^*$).
*   **Source**: Internal (Model Weights / Configuration).
*   **Nature**: Conservative, stabilizing, defining.
*   **Role**: To prevent catastrophic forgetting or drift.

## 2. Truth Channel ($F_{truth}$)
$$F_{truth} = \vec{T}(obs) - \vec{x}$$

The force of **Empirical Reality**. It pulls the agent towards the observed state of the environment.
*   **Source**: External (Sensors / Text Input).
*   **Nature**: Disruptive, informative, grounding.
*   **Role**: To ensure grounding in current observation.

## 3. Social Resonance ($F_{social}$)
$$F_{social} = \sum_{j} T_{ij} (\vec{x}_j - \vec{x}_i)$$

The force of **Consensus**. It pulls the agent towards the state of its trusted peers.
*   **Source**: Inter-subjective (Multi-Agent Communication).
*   **Nature**: Harmonizing, conformist (if trust is high), isolating (if trust is low).
*   **Role**: To enable collective intelligence and shared error correction.

## The Balance

The decision $\Delta \vec{x}$ is not a calculation; it is an equilibrium:

$$\Delta \vec{x} = k_{eff} [ F_{id} + m(F_{truth} + F_{social}) ]$$

Where $m$ represents "Attention" (how much the agent cares about the external world vs. its internal state).
