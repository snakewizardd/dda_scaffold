# DDA-X: Dynamic Decision Algorithm with Exploration (Iteration 3)

> **Robust Cognitive Architecture for High-Dimensionsal Decision Making.**

**Maintained by**: [snakewizardd](https://github.com/snakewizardd)  
**Repository**: [https://github.com/snakewizardd/dda_scaffold](https://github.com/snakewizardd/dda_scaffold)  
**Documentation**: [https://snakewizardd.github.io/dda_scaffold/](https://snakewizardd.github.io/dda_scaffold/)

> **RADICAL OPEN SOURCE ENTHUSIAST**: This project is open for the world. Contact privately for anything besides basic public github interactions.

DDA-X is a sophisticated agentic framework that bridges the gap between biological cognitive dynamics and machine decision-making. Unlike traditional Reinforcement Learning (RL), which optimizes purely for scalar reward, DDA-X models **Identity, Surprise, and Rigidity** to create agents that behave with psychological realism and inherent safety constraints.

---

## The Vision: Iteration 3 "Maximum Framework Potential"

Iteration 3 marks the transition from a single-agent decision algorithm to a **Multi-Agent Metacognitive Society**. We have implemented the full spectrum of theoretical extensions outlined in the DDA-X roadmap.

### Key Theoretical Contributions

1.  **Hierarchical Identity ($\vec{x}^*$)**: Moving beyond a single attractor to a nested structure:
    *   **Core ($\gamma \to \infty$)**: Inviolable constraints (Alignment/Safety).
    *   **Persona ($\gamma \approx 2.0$)**: Task-specific cognitive style.
    *   **Role ($\gamma \approx 0.5$)**: Situational flexibility.
2.  **Metacognitive Layer**: The agent is now self-aware of its own rigidity ($\rho$). It introspects on its $\rho$-value and reports when it is becoming "defensive" or "protective," enabling human help requests before failure.
3.  **Multi-Agent Society**: Emergent social dynamics driven by a **Trust Matrix**, where trust is the inverse of cumulative prediction error. Agents exert "Social Pressure" on each other, forming coalitions based on identity alignment.
4.  **Temporal Dynamics**: Rigidity now operates on multiple timescales:
    *   $\rho_{fast}$: Immediate startle/surprise.
    *   $\rho_{slow}$: Long-term stress/adaptation.
    *   $\rho_{trauma}$: Permanent, asymmetric deviations from extreme prediction errors.

---

## Enhanced Hybrid Backend

Optimized for **Snapdragon Elite X** performance using:
*   **LM Studio**: Local completion engine running **GPT-OSS-20B**.
*   **Ollama**: Nomic-embed-text for high-dimensional semantic state mapping.

---

## Research Contributions

Our live experiments have validated:
*   **Cognitive Loop Binding**: Rigidity ($\rho$) now directly modulates LLM sampling parameters (Temperature/Top-p), creating a physical link between "stress" and "cognitive focus."
*   **Alignment Stability Theorem**: Formal proof that core identity is invariant if $\gamma_{core} \to \infty$, regardless of social pressure or environment forces.
*   **Trust as Predictability**: Deceptive agents are mathematically identified by the surprise they generate, not just their outcomes.

---

## System Philosophy: The Architecture

This framework represents a paradigm shift in AI alignment. Instead of training via RLHF, we embed alignment as a **Geometric Attractor**.

*   **Identity ($\vec{x}^*$)** is the "Reference State" that provides direction.
*   **Rigidity ($\rho$)** is the "Adaptive Constraint" that protects the core when the world becomes unpredictable.
*   **Society** is the resonance of individual identities finding alignment.

We are building machines that respect the sanctity of "Self" while remaining humble to the "Truth" ($F_{truth}$). This is the path of **Aligned Intelligence**.

---

## Quick Start (Science for the World)

For scientists and enthusiasts wanting to see the "Novel Science" without the "Psycho" overhead:

1.  **Local Test**: `python demo.py` (No LLM required, see the math move).
2.  **Live Experiments**: `python runners/run_experiments.py` (Requires LM Studio + Ollama).

See [QUICKSTART.md](https://github.com/snakewizardd/dda_scaffold/blob/main/QUICKSTART.md) for a 5-minute setup guide.

---

## Project Structure

*   `src/core/hierarchy.py`: Nesting the Self.
*   `src/core/metacognition.py`: The Self observing the Self.
*   `src/society/trust.py`: The mathematics of social cohesion.
*   `src/llm/hybrid_provider.py`: The bridge to the LLM "Brain."

---

## License
MIT - Patentable discoveries are documented in `DISCOVERIES.md`.