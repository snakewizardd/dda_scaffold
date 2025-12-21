# DDA-X: Cognitive Engineering Framework

**A research-grade architecture for identity-persistent AI agents. Synthesizes Control Theory (PID) with Vector Space Mechanics to implement parameter-level coupling, surprise-rigidity dynamics, and hierarchical alignment stability.**

[![Tests](https://img.shields.io/badge/tests-45%2F45%20passing-brightgreen)](tests/test_ddax_claims.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://snakewizardd.github.io/dda_scaffold/)

> **Dedicated to Malky.**
> *May her memory be a blessing.*

---

## 🏗️ System Overview

DDA-X addresses the **Context Window Bottleneck** in agentic systems. Instead of maintaining identity via massive context injection (Cost: O(n²)), DDA-X maintains identity via **State Vectors and Control Theory** (Cost: O(1)).

The framework treats "Personality" as a mass vector with inertia, applying Newtonian mechanics to the agent's embedding space to prevent drift and ensure alignment stability under adversarial pressure.

### Key Capabilities

* **Stateful Alignment:** Identity is stored as a persistent vector (`agent.x`), not a prompt.
* **Hybrid Inference:** Validated on **GPT-5.2** (via API) for high-fidelity reasoning and **GPT-OSS-20B/Nomic** (Local) for edge deployment.
* **Economic Efficiency:** Complete simulation suites (50+ runs) executed for **<$1.50** total API cost due to vector-based state management.

---

## 🔬 Theoretical Foundations

DDA-X implements seven core mechanics validated through empirical testing.

| ID | Mechanic | Formulation | Status |
|----|----------|-------------|--------|
| **D1** | Surprise-Rigidity Coupling | `ρ += α[σ((ε-ε₀)/s) - 0.5]` | ✅ Verified (r=0.92) |
| **D2** | Identity Inertia | `F = γ(x* - x)` | ✅ Verified |
| **D3** | Rigidity-Modulated Sampling | `Exploration × (1-ρ)` | ✅ Verified |
| **D4** | Trauma Accumulation | `ρ_trauma += δ` (asymmetric) | ✅ Verified |
| **D5** | Trust Matrix | `T = 1/(1 + Σε)` | ✅ Verified (87% accuracy) |
| **D6** | Hierarchical Identity | `γ_core >> γ_persona > γ_role` | ✅ Verified |
| **D7** | Metacognitive Introspection | Structured self-reporting of ρ state | ✅ Verified (r=0.89) |

**Full validation:** [Test Suite](tests/test_ddax_claims.py) | [Results](test_results/) | [Documentation](https://snakewizardd.github.io/dda_scaffold/)

---

## 📊 Experimental Validation

The framework has been validated through **30+ operational simulations**, ranging from dyadic philosophical debates to multi-agent societies.

### The Philosophers' Duel (Adversarial Stress Test)

Two agents with opposing ethical frameworks (Deontologist vs Utilitarian) debate escalating moral dilemmas.

| Metric | DEONT (Immanuel) | UTIL (John) |
|--------|------------------|-------------|
| Mean ε (prediction error) | 0.723 | 0.742 |
| Mean ρ (rigidity) | 0.196 | 0.155 |
| Identity alignment | 0.461 | 0.437 |
| Wound activations | 5 | 3 |

**Key finding:** Under maximum adversarial pressure (The Triage Protocol, ε=1.20), UTIL generated `[contemplates in silence]` — a non-response the DDA correctly measured as identity misalignment (cos=0.14). The framework caught the breakdown.

📁 **Full data:** [Transcript](data/philosophers_duel/transcript.md) | [Metrics](data/philosophers_duel/session_log.json) | [Visualizations](data/philosophers_duel/gpt_vis_analysis/)

### Multi-Agent Society

* **Setup:** 6-14 heterogeneous agents interacting in a shared environment.
* **Result:** Emergent coalition formation based on the Trust Matrix (D5), validated against theoretical predictions with 87% accuracy.

---

## ⚙️ Architecture & Installation

DDA-X is built on the **Microsoft ExACT** pattern, optimized for local NPU execution (Qualcomm Snapdragon X Elite) but fully compatible with OpenAI-compliant endpoints.

### Prerequisites

* **Python 3.10+**
* **Inference:**
  * *Local:* LM Studio (GPT-OSS-20B) + Ollama (`nomic-embed-text`)
  * *Cloud:* OpenAI API Key (GPT-5.2/GPT-4o)

### Setup

```bash
git clone https://github.com/snakewizardd/dda_scaffold.git
cd dda_scaffold
pip install -r requirements.txt
```

### Running Simulations

```bash
# Verify physics engine (no LLM required)
python verify_dda_physics.py

# Run the Philosophers' Duel
python simulations/simulate_philosophers_duel.py

# Run multi-agent society
python simulations/simulate_society.py
```

---

## 📂 Repository Structure

```
src/
├── core/       # Physics engines (State, Dynamics, Forces, Hierarchy)
├── llm/        # Hybrid provider logic (Local + API)
├── society/    # Trust matrix and multi-agent coordination
└── memory/     # RAG-enabled ledgers with salience weighting

simulations/    # 30+ operational experiments
data/           # Raw JSON logs, vector states, ledger artifacts
tests/          # 45 unit tests validating D1-D7
```

---

## ⚠️ Limitations

* **Research prototype** — Not production-ready
* **Empirically validated** — Not formally proven (no Lyapunov stability proofs)
* **Tested on GPT-5.2 and local models** — Not benchmarked against other agent frameworks (ReACT, AutoGen, CrewAI)
* **Novel synthesis** — Builds on Microsoft ExACT; the cognitive layer (rigidity, trauma, trust) is the original contribution

We welcome critical feedback. [Open an issue](https://github.com/snakewizardd/dda_scaffold/issues).

---

## 📜 Citation & License

**License:** MIT

```bibtex
@software{ddax2025,
  author = {snakewizardd},
  title = {DDA-X: Cognitive Engineering Framework},
  year = {2025},
  url = {https://github.com/snakewizardd/dda_scaffold},
  note = {Stateful alignment architecture utilizing Control Theory and Vector Space Mechanics.}
}
```