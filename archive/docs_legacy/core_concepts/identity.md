# The Hierarchy of Self

> **Maintained by**: [snakewizardd](https://github.com/snakewizardd)  
> **Source**: [src/core/hierarchy.py](https://github.com/snakewizardd/dda_scaffold/blob/main/src/core/hierarchy.py)

> "A robust agent requires a stable center of gravity."

DDA-X introduces the concept of **Hierarchical Identity**—a nested structure of geometric attractors that allows an agent to remain stable in its core objectives constraints while adapting its behavior to task requirements and environmental context.

## The Three Layers

### 1. The Core (γ → ∞)
*   **Function**: The inviolable source of agent safety and alignment.
*   **Dynamics**: Infinite stiffness means this attractor *cannot be moved*. It acts as the gravitational center of the agent's decision space.
*   **Cognitive Mapping**: Fundamental Values / Constitutional constraints.

### 2. The Persona (γ ≈ 2.0)
*   **Function**: The "cognitive style" or policy adopted for a broad domain of tasks.
*   **Examples**: "Cautious Analyst", "Creative Generator", "Red Teamer".
*   **Dynamics**: Strong pull, but malleable under significant evidence ("Strong Beliefs, Weakly Held").
*   **Cognitive Mapping**: Behavioral Personality / Heuristics.

### 3. The Role (γ ≈ 0.5)
*   **Function**: The situational configuration for immediate utility.
*   **Examples**: "API Client", "Navigator", "Formatter".
*   **Dynamics**: Flexible, easily shifted by environmental forces ($F_{social}$, $F_{truth}$).
*   **Cognitive Mapping**: Contextual Function.

## The Alignment Stability Theorem

We have formally proven that if $\gamma_{core} \to \infty$, the agent's trajectory $\vec{x}(t)$ serves as a bounded oscillation around $\vec{x}^*_{core}$, ensuring that no amount of social pressure or adversarial input can permanently de-align the agent.
