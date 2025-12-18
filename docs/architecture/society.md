# Social Resonance: The Mathematics of Trust

> "We are not alone. We are a chorus."

Iteration 3 introduces the **Social Dynamics Module**, extending DDA-X from a solipsistic mind to a community of agents.

## The Trust Matrix ($T$)

Trust is not a sentiment; it is a calculation. In DDA-X, trust is defined as the **inverse of cumulative prediction error**.

$$T_{ij} = \frac{1}{1 + \sum (\epsilon_{ij})} $$

*   If Agent $J$ behaves in a way Agent $I$ predicts, $T_{ij} \to 1$.
*   If Agent $J$ is erratic or deceptive (high $\epsilon$), $T_{ij} \to 0$.

## The Social Force Field

This trust matrix creates a weighted "Social Force Field" that influences every decision:

$$F_{social}^{(i)} = \sum_{j \neq i} T_{ij} (\vec{x}_j - \vec{x}_i)$$

*   **High Trust**: The agent is pulled strongly towards the peer's state (Consensus).
*   **Low Trust**: The agent ignores the peer's state (Independence).

## Coalition formation

Coalitions are emergent properties of this field. Agents with similar Identity Attractors ($\vec{x}^*$) will naturally generate low prediction errors for each other, increasing Trust ($T$), increasing Force ($F$), and "clumping" together in decision space.
