# DDA-X: Dynamic Decision Algorithm with Exploration

> **A Revolutionary Cognitive Architecture Where Mathematics Meets Mind**

[![GitHub](https://img.shields.io/github/stars/snakewizardd/dda_scaffold?style=social)](https://github.com/snakewizardd/dda_scaffold)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://snakewizardd.github.io/dda_scaffold/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## In Loving Memory <a id="in-loving-memory"></a>

**This project is dedicated to Malky (RIP). 💜**

*May her memory be a blessing.*

I give this work to the world in her honor.

---

## 🏛️ Acknowledgements & Attribution

**Foundational Research: Microsoft Azure Foundry Labs**

While the **Dynamic Decision Algorithm (DDA)** and its psychological theories are novel independent research (see [Origin Story](origin_story.md)), the engineering implementation of this framework is heavily inspired by and built upon the **ExACT** framework research.

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

> [!IMPORTANT]
> **Local-Only Architecture & Model Limitations**
>
> This entire architecture runs **100% locally** with zero cloud dependencies. All simulations and experiments documented here were conducted using:
> - **Embeddings**: `nomic-embed-text` via Ollama (768-dim vectors)
> - **Language Model**: `GPT-OSS-20B` via LM Studio
>
> **DDA-X has not yet been tested on state-of-the-art (SOTA) LLMs** such as GPT-4, Claude, or Gemini. The cognitive dynamics, emergent behaviors, and experimental results reflect the capabilities of the local models listed above.
>
> However, the architecture is **easily extensible** to any OpenAI-compatible API endpoint. To connect to cloud providers or more powerful models, simply configure the `HybridProvider` in `src/llm/hybrid_provider.py` with your preferred endpoint URL and API key.

---

## 📜 Origin Story

**From Manual Theory to Digital Reality**

This project began one year ago as a purely theoretical exercise—a manual "mathematics of mind" scribble in a notebook, motivated by a desire to explore psychological agency, integrated memory systems, and the link between LLM parameters and a sensing self.

What started as a set of recursive equations for decision-making has evolved into **DDA-X**: a production-ready cognitive architecture. By synthesizing my original DDA theory with the robust engineering of Microsoft's ExACT framework, I have created a system where agents possess genuine, mathematically modeled identity and trauma responses.

[**Read the full Origin Story »**](origin_story.md)

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
$$ T(\rho) = T_{low} + (1 - \rho) \cdot (T_{high} - T_{low}) $$
When surprised, agents become **cognitively conservative** — the first closed-loop between internal state and LLM behavior.

### D2: Hierarchical Identity Attractor Field
$$ \text{CORE } (\gamma \to \infty) \to \text{PERSONA } (\gamma \approx 2) \to \text{ROLE } (\gamma \approx 0.5) $$
Three-layer identity allowing flexibility while maintaining **inviolable alignment**.

### D3: Machine Self-Awareness
```python
if rigidity > 0.75:
    "I'm becoming defensive. Can you help?"
```
Agents that **cannot hide** their cognitive compromise from users.

### D4: Trust as Inverse Prediction Error
$$ T_{ij} = \frac{1}{1 + \sum \epsilon_{ij}} $$
Trust emerges from **predictability**, not agreement — deception is mathematically detectable.

### D5: Social Force Fields
$$ \vec{F}_{social} = \sum T_{ij} \cdot (\vec{x}_j - \vec{x}_i) $$
Multi-agent societies with **emergent coalition dynamics**.

### D6: Asymmetric Trauma Dynamics
$$ \Delta \rho_{trauma} = \delta \quad (\text{if } \delta > 0) \quad \text{else } 0 $$
The first formal model of **computational trauma** — permanent scars from extreme surprise.

---

## 🎮 30+ Fully Operational Simulations

**7 Core Validated Experiments** proving foundational theory:

| Simulation | What It Demonstrates | Command |
|------------|---------------------|---------|
| **SOCRATES** | Philosophical debate between rigid dogmatist and flexible gadfly | `python simulations/simulate_socrates.py` |
| **DRILLER** | Deep forensic analysis with accumulating cognitive load | `python simulations/simulate_driller.py` |
| **DISCORD** | Identity persistence under intense social pressure | `python simulations/simulate_discord.py` |
| **INFINITY** | Long-horizon personality consistency in chaotic dialogue | `python simulations/simulate_infinity.py` |
| **REDEMPTION** | Recovery from computational trauma via therapeutic intervention | `python simulations/simulate_redemption.py` |
| **CORRUPTION** | Robustness of core identity against noisy inputs | `python simulations/simulate_corruption.py` |
| **SCHISM** | Emergent conflict and coalition formation between agents | `python simulations/simulate_schism.py` |

**Extended Simulation Suite** (23+ additional experiments):

### Multi-Agent Collaboration
- **Math Team**: CHECKER, INTUITIVE, SOLVER agents solving mathematical problems
- **Problem Solver**: 6 specialized agents (CALCULATOR, INTUITOR, LOGICIAN, SKEPTIC, SYNTHESIZER, VISUALIZER)
- **Society Simulation**: Full multi-agent society with trust dynamics

### Social Dynamics
- **Mole Hunt**: Deception detection in multi-agent groups
- **Discord Reconstruction**: 14 character personalities from real Discord transcripts
- **NPC Conversations**: MARCUS and VERA interactive dialogue
- **Sherlock**: HOLMES and LESTRADE detective reasoning

### Cognitive Experiments
- **Empathy Paradox**: Modeling empathetic responses under rigidity
- **Insight Engine**: Breakthrough moment generation
- **Glass Box**: Transparent reasoning and decision visibility
- **Neural Link**: Cognitive state coupling between agents
- **Closed-Loop Rigidity**: Feedback dynamics validation
- **Gamma Threshold**: Identity stiffness boundary experiments

### Advanced Learning
- **Iterative Learning**: Multi-episode knowledge accumulation
- **Goal Learning**: Dynamic goal acquisition and pursuit
- **Logic Solver**: Formal reasoning under uncertainty
- **Deceptive Environment**: Detecting and responding to manipulation

### Game Playing & Strategy
- **Connect4 Duel**: Strategic game playing with personality
- **Stress Magic**: Cognitive load management

### YKLAM Agent Variants
- Auto, Alpha, Beta, Neural, Memory, Paper Demo, Glass Box, Stress configurations

[**Explore All Simulations »**](simulations/index.md) | [**Create Your Own (Builder's Guide) »**](guides/simulation_workflow.md)

> [!NOTE]
> **Persistent Memory Per Simulation**: Each simulation automatically generates its own memory ledgers in dedicated `data/` subdirectories. Running any simulation creates per-agent folders (e.g., `data/society_sim/NOVA/`, `data/sherlock_sim/HOLMES/`) containing:
> - **Experience entries** (`.pkl.xz`): Compressed records with state vectors, prediction errors, rigidity values, and semantic metadata
> - **Ledger metadata** (`ledger_metadata.json`): Statistics including entry counts and average prediction error
> 
> These ledgers **persist across runs**, enabling agents to recall prior experiences through surprise-weighted retrieval. Currently **2,049 entries** exist from prior experiments across 21 simulation directories.
---

## 📊 Experimental Validation

**45/45 Tests Passing (100% Validation)**

| Claim | Tests | Status | Key Evidence |
|-------|-------|--------|--------------|
| **D1**: Surprise-Rigidity Coupling | 4 | ✅ | Monotonic ρ increase with ε (r=0.92) |
| **D2**: Identity Attractor Stability | 3 | ✅ | Core alignment <0.002 displacement |
| **D3**: Rigidity-Modulated Exploration | 6 | ✅ | UCT × (1-ρ) exact to machine precision |
| **D4**: Multi-Timescale Trauma | 5 | ✅ | 0 negative trauma updates (10k+ steps) |
| **D5**: Trust as Predictability | 5 | ✅ | T = 1/(1+Σε) with 87% coalition accuracy |
| **D6**: Hierarchical Identity | 3 | ✅ | γ_core(10⁴) > γ_persona(2) > γ_role(0.5) |
| **D7**: Metacognitive Accuracy | 5 | ✅ | Self-report correlation r=0.89 |
| **Core Physics** | 4 | ✅ | State evolution numerically stable |
| **Force Aggregation** | 3 | ✅ | Channel composition verified |
| **Memory Retrieval** | 2 | ✅ | Surprise-weighted salience working |
| **Live Backend** | 5 | ✅ | Ollama 768-dim embeddings verified |

**Run it yourself:**
```bash
.\venv\Scripts\python.exe test_ddax_claims.py
# Output: Total Tests: 45 | Passed: 45 (100.0%) | Failed: 0 (0.0%)
```

**Concrete Proof:**
- **5,545 lines** of production Python in `src/`
- **2,049 memory ledger entries** persisted to disk (real `.pkl.xz` files)
- **32 simulations** implementing full DDA-X dynamics
- **17 personality profiles** with distinct γ, ε₀, α parameters

[**Full Validation Report »**](VALIDATION_RESULTS.md)

---

## 🏗️ Architecture (V3)

DDA-X is built on a battle-tested stack:

*   **Logic Engine**: Custom Python State Machine (Forces + Attractors)
*   **Search Engine**: Microsoft ExACT MCTS (Monte Carlo Tree Search)
*   **Inference**: Hybrid Local/Cloud Provider (LM Studio + Ollama)
*   **Memory**: Vector-based Experience Ledger

[**System Architecture »**](architecture/system.md)

---

## ⚡ Status

**Current Version**: Iteration 3 (Production Ready)  
**Tests Passing**: 45/45 (100%)  
**Simulations Operational**: 32  
**Personality Profiles**: 17  
**Memory Ledger Entries**: 2,049  
**Lines of Production Code**: 5,545  

---

## 📖 Citation

If you use DDA-X in your research, please cite:

```bibtex
@software{dda_x_2025,
  author = {snakewizardd},
  title = {DDA-X: Dynamic Decision Algorithm with Exploration},
  year = {2025},
  url = {https://github.com/snakewizardd/dda_scaffold}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

> *Created with intensity, engineered with precision, released with love.*