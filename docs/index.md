# DDA-X: Dynamic Decision Algorithm with Exploration

> **A research-grade architecture for identity-persistent AI agents.**

Built on original DDA research, evolved through Microsoft ExACT integration.

## Quick Links

- [Paper v2.0](architecture/paper.md) — Full theoretical framework with equations
- [Implementation Mapping](architecture/ARCHITECTURE.md) — Theory→Code bridging
- [Simulation Chronology](simulation_chronology.md) — 59 progressive experiments

## Core Innovation

Standard RL: surprise → exploration

DDA-X: **surprise → rigidity → contraction**

## Key Equations

**Rigidity Update:**
\[
\rho_{t+1} = \text{clip}\left(\rho_t + \alpha\left[\sigma\left(\frac{\epsilon - \epsilon_0}{s}\right) - 0.5\right], 0, 1\right)
\]

**Effective Openness:**
\[
k_{eff} = k_{base} \times (1 - \rho)
\]

**Multi-Timescale Rigidity:**
\[
\rho_{eff} = 0.5 \rho_{fast} + 0.3 \rho_{slow} + 1.0 \rho_{trauma}
\]

## Repository Structure

```
dda_scaffold/
├── simulations/          # 59 progressive experiments
├── docs/architecture/    # Theory and implementation docs
├── src/                  # Core modules (ledger, LLM providers)
└── simulation_chronology.csv
```
