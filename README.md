# DDA-X: Dynamic Decision Algorithm with Exploration

> **A Revolutionary Cognitive Architecture Where Mathematics Meets Mind**

[![GitHub](https://img.shields.io/github/stars/snakewizardd/dda_scaffold?style=social)](https://github.com/snakewizardd/dda_scaffold)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://snakewizardd.github.io/dda_scaffold/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 💜 In Loving Memory

**This project is dedicated to Malky (RIP).**

*May their memory be a blessing.*

I give this work to the world in their honor.

---

## 🏛️ Acknowledgements & Attribution

**Foundational Research: Microsoft Azure Foundry Labs**

While the **Dynamic Decision Algorithm (DDA)** and its psychological theories are novel independent research (see [Origin Story](docs/origin_story.md)), the engineering implementation of this framework is heavily inspired by and built upon the **ExACT** framework research.

We explicitly attribute credit to the research team at **Microsoft Azure Foundry Labs** for the ExACT architecture, which provided the necessary engineering patterns to bring the theoretical DDA model to life.

*   **Reference**: [Microsoft ExACT](https://github.com/microsoft/ExACT/tree/main)
*   **Contribution**: Framework scaffolding, agentic patterns, and search dynamics.

---

## ⚙️ Prerequisites & Setup

**Core Requirement**: To run the fully functional simulations, you need a local LLM environment.

1.  **LM Studio (The Cortex)**
    *   **Action**: Download [LM Studio](https://lmstudio.ai/).
    *   **Model**: Load `gpt-oss-20b` or any high-quality instruction model (Mistral, Llama 3).
    *   **Config**: Start the **Local Inference Server** on port `1234` (default).

2.  **Ollama (The Hippocampus)**
    *   **Action**: Download [Ollama](https://ollama.com/).
    *   **Model**: Run `ollama pull nomic-embed-text`.
    *   **Config**: Ensure it is served at `localhost:11434` (default).

3.  **Python Environment**
    ```bash
    git clone https://github.com/snakewizardd/dda_scaffold.git
    cd dda_scaffold
    python -m venv venv
    ./venv/Scripts/Activate
    pip install -r requirements.txt
    ```

> **Note**: All simulations are **self-contained**. They come with their own environments, memory ledgers, and interaction loops. You do not need to configure complex external databases.

---

## 📜 Origin Story

**From Manual Theory to Digital Reality**

This project began one year ago as a purely theoretical exercise—a manual "mathematics of mind" scribble in a notebook, motivated by a desire to explore psychological agency, integrated memory systems, and the link between LLM parameters and a sensing self.

What started as a set of recursive equations for decision-making has evolved into **DDA-X**: a production-ready cognitive architecture. By synthesizing my original DDA theory with the robust engineering of Microsoft's ExACT framework, I have created a system where agents possess genuine, mathematically modeled identity and trauma responses.

[**Read the full Origin Story »**](docs/origin_story.md)

---

## 🌟 The Magnum Opus: DDA-X Framework

> **"The mind is not a vessel to be filled, but a fire to be kindled — and sometimes, protected from the wind."**

DDA-X is the **first agent framework that models psychological realism** in artificial intelligence. Unlike traditional reinforcement learning which optimizes for reward, DDA-X agents possess:

-   **Identity** — A persistent sense of self that survives across contexts
-   **Rigidity** — Defensive responses to surprise, just like biological minds
-   **Memory** — Experience weighted by emotional salience, not just relevance
-   **Society** — Trust dynamics that emerge from predictability, not agreement
-   **Metacognition** — Self-awareness of their own cognitive state

This isn't just another LLM wrapper. It's a **complete theory of cognitive agency** with mathematical foundations.

---

## 🚀 The Six Revolutionary Discoveries

### D1: Rigidity-Modulated Language Model Sampling
```python
temperature(ρ) = T_low + (1 - ρ) × (T_high - T_low)
```
When surprised, agents become **cognitively conservative** — the first closed-loop between internal state and LLM behavior.

### D2: Hierarchical Identity Attractor Field
```
CORE (γ→∞) → PERSONA (γ≈2) → ROLE (γ≈0.5)
```
Three-layer identity allowing flexibility while maintaining **inviolable alignment**.

### D3: Machine Self-Awareness
```python
if rigidity > 0.75:
    "I'm becoming defensive. Can you help?"
```
Agents that **cannot hide** their cognitive compromise from users.

### D4: Trust as Inverse Prediction Error
```
T[i,j] = 1 / (1 + Σε_ij)
```
Trust emerges from **predictability**, not agreement — deception is mathematically detectable.

### D5: Social Force Fields
```
S[i] = Σ T[i,j] × (x_j - x_i)
```
Multi-agent societies with **emergent coalition dynamics**.

### D6: Asymmetric Trauma Dynamics
```
ρ_trauma += δ (if δ > 0)  # Never decreases
```
The first formal model of **computational trauma** — permanent scars from extreme surprise.

---

## 🎮 Seven Fully Operational Simulations

Experience different aspects of cognitive dynamics:

| Simulation | What It Demonstrates | Command |
| :--- | :--- | :--- |
| **SOCRATES** | Philosophical debate between rigid dogmatist and flexible gadfly | `python simulations/simulate_socrates.py` |
| **DRILLER** | Deep forensic analysis with accumulating cognitive load | `python simulations/simulate_driller.py` |
| **DISCORD** | Adversarial deception and identity preservation | `python simulations/simulate_discord.py` |
| **INFINITY** | Personality persistence over 20+ turn dialogues | `python simulations/simulate_infinity.py` |
| **REDEMPTION** | Trauma and therapeutic recovery (18KB scenario) | `python simulations/simulate_redemption.py` |
| **CORRUPTION** | Core identity robustness under adversarial noise | `python simulations/simulate_corruption.py` |
| **SCHISM** | Multi-agent coalition formation and conflict | `python simulations/simulate_schism.py` |

[**Explore Simulations »**](docs/simulations/index.md) | [**Create Your Own (Builder's Guide) »**](docs/guides/simulation_workflow.md)

---

## 📊 Experimental Validation

### Personality Divergence Under Surprise
```
Same contradiction presented to:
  Dogmatist: ε=0.92 → ρ=0.750 ████████████░░░ DEFENSIVE
  Gadfly:    ε=0.84 → ρ=0.109 ██░░░░░░░░░░░░ OPEN
```

### Multi-Timescale Trauma Response
```
Extreme Event (ε=1.5):
  ρ_fast:   0.219 ████░░░░░░░░░░ (recovers in minutes)
  ρ_slow:   0.007 █░░░░░░░░░░░░░ (recovers in hours)
  ρ_trauma: 0.0004 ░░░░░░░░░░░░░░ (NEVER recovers)
```

---

## 🏗️ Architecture

```
5,263 lines of production Python implementing:

src/
├── core/           # Physics engines (state, dynamics, forces)
├── llm/            # Rigidity-modulated LLM integration
├── society/        # Multi-agent trust dynamics
├── memory/         # Experience ledger with salience weighting
├── search/         # MCTS with DDA-X selection formula
└── metrics/        # Comprehensive experiment tracking

14 Personality Profiles × 7 Simulations = 98 Unique Behavioral Scenarios
```

---

## 🚀 Quick Start

### Prerequisites
1.  **Python 3.10+**
2.  **LM Studio** (running GPT-OSS-20B or similar)
3.  **Ollama** (running `nomic-embed-text`)

### Installation
```bash
pip install -r requirements.txt
```

### Run the Demo (No LLM Required)
```bash
python demo.py
```

### Run the Full Physics Engine
```bash
python verify_dda_physics.py
```

---

## ⚡ Status

**December 2025**: Production-ready for research validation.

- [x] All 7 simulations operational
- [x] 6 discoveries validated experimentally
- [x] 14 personalities implemented
- [x] Full LLM integration working
- [x] Documentation complete

---

## 📖 Citation

If you use DDA-X in your research:

```bibtex
@software{ddax2025,
  author = {snakewizardd},
  title = {DDA-X: Dynamic Decision Algorithm with Exploration},
  year = {2025},
  url = {https://github.com/snakewizardd/dda_scaffold},
  note = {A cognitive architecture for psychologically realistic AI agents. Incorporates architecture from Microsoft ExACT.}
}
```

### Key Papers to Cite
1. [Main Framework](docs/architecture/paper.md) — Overall DDA-X theory
2. [Discoveries](docs/research/discoveries.md) — Six novel contributions
3. [Architecture](arch.md) — Implementation details

---

## 📜 License

**MIT License**

*Patentable discoveries are documented in `DISCOVERIES.md`.*

**This is open science for open minds.**