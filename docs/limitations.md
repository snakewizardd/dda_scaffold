# Known Limitations

An honest assessment of the current DDA-X implementation, based on independent review by GPT-5.2 reasoning models.

---

## Overview

This page documents known gaps between theory and implementation, areas requiring further work, and open research questions. Transparency about limitations is essential for research integrity.

---

## 1. Trust Equation Mismatch

### Theory

The paper describes trust as predictability-based:

$$
T_{ij} = \frac{1}{1 + \sum_{\mathcal{W}} \epsilon_{ij}(t)}
$$

Where trust decreases as accumulated prediction errors increase.

### Implementation

Current simulations implement a **hybrid** trust model instead:

| Simulation | Actual Trust Mechanism |
|:---|:---|
| Philosopher's Duel | Semantic alignment + $\epsilon$ thresholds |
| Skeptic's Gauntlet | Civility gating (fairness-based) |
| Collatz Review Council | Coalition-weighted dyadic trust |

!!! note "Not a Bug"
    The hybrid approach may actually be more realistic than pure predictability-based trust. However, the documentation should accurately reflect what is implemented.

### Status
The `paper.md` Section 7 has been updated to clarify this hybrid nature.

---

## 2. Dual Rigidity Models in AGI Debate

### Issue

In `simulate_agi_debate.py`, two rigidity models run in parallel:

1. **Multi-timescale** (`agent.multi_rho`): Fast/Slow/Trauma decomposition
2. **Legacy single-scale** (`agent.rho`): Simple scalar update

The multi-timescale rigidity is computed and logged as **telemetry**, but the legacy `agent.rho` is what actually drives behavior.

### Impact

- Multi-timescale dynamics exist but don't affect generation
- Claims about multi-timescale control are partially aspirational
- Telemetry shows multi-timescale patterns, but behavior uses single-scale

### Status
Documented here. The simulation works correctly — this is an architectural choice, not a bug.

---

## 3. Uncalibrated Thresholds

### Issue

The simulations calibrate some parameters dynamically:

- $\epsilon_0$ (surprise baseline): Calibrated from early-run median
- $s$ (sigmoid steepness): Calibrated from IQR

But other thresholds are **hardcoded**:

- `wound_cosine_threshold` (typically 0.28)
- `trauma_threshold` ($\theta_{\text{trauma}}$)
- Multi-timescale weights ($w_f, w_s, w_t$)

### Impact

- Wound sensitivity varies across domains and embedding models
- Parameters tuned for one simulation may not transfer

### Recommendation
Future work should extend calibration to wound and trauma thresholds, potentially using percentile-based approaches.

---

## 4. Hierarchical Identity Degeneracy

### Issue

In `simulate_identity_siege.py`, hierarchical identity is implemented with three stiffness values:

$$
F = \gamma_c(x^*_c - x) + \gamma_p(x^*_p - x) + \gamma_r(x^*_r - x)
$$

However, the same `identity_emb` is used for all layers:

```python
# Current implementation
core_emb = identity_emb
persona_emb = identity_emb  # Same!
role_emb = identity_emb     # Same!
```

### Impact

- Layers differ only by $\gamma$ magnitude
- Directional differences between Core/Persona/Role are lost
- True hierarchical identity would require separate embeddings

### Status
Documented here. Would require code changes to fix, which is out of scope.

---

## 5. Measurement Validity

### Concern

Prediction error is computed as:

$$
\epsilon_t = \|x_{\text{pred}} - e(a_t)\|
$$

This conflates multiple factors:

- **Semantic novelty**: Genuine new content
- **Style shifts**: Verbosity, formality changes
- **Topic drift**: Moving between subject areas

### Impact

A highly verbose response might register as "surprising" even if semantically predictable, because embedding distance captures style as well as content.

### Recommendation
Consider decomposing embeddings into content vs. tone components, or tracking cosine distance separately from norm distance.

---

## 6. Model-Dependent Behavior

### Issue

For reasoning models (GPT-5.2, o1), DDA-X cannot control sampling parameters:

```python
if "gpt-5.2" in self.model or "o1" in self.model:
    # Cannot set temperature, top_p, penalties
    # Must use semantic injection instead
```

### Impact

- Rigidity → behavior binding is **semantic only** for these models
- The 100-point rigidity scale compensates, but effectiveness varies
- Different models may respond differently to same semantic instructions

### Status
Working as designed. The semantic injection approach is the only option for reasoning models.

---

## Open Research Questions

1. **Optimal weight learning**: Can $w_f, w_s, w_t$ be learned from data?
2. **Cross-domain calibration**: How do thresholds transfer between domains?
3. **Embedding model sensitivity**: How much do dynamics change with different embedders?
4. **Trust convergence**: Under what conditions does hybrid trust stabilize?
5. **Trauma reversibility**: What safe interaction patterns most effectively heal trauma?

---

## Citing This Work

If referencing these limitations in academic work:

> "The DDA-X framework, while novel in its approach to rigidity-based agent dynamics, has acknowledged limitations including hybrid trust implementations and uncalibrated wound thresholds, as documented by the authors in their Known Limitations disclosure."
