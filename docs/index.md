# DDA-X: Dynamic Decision Algorithm with Exploration

> **A Mathematically Rigorous Framework for Parameter-Level Cognitive Dynamics**

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

**The Core DDA Hypothesis**: The Dynamic Decision Algorithm **inverts** the standard surprise-exploration relationship:

> **Standard RL**: Surprise → Curiosity → Explore  
> **DDA**: Surprise → Rigidity → Defend

DDA represents the **Original Theory**—over a year of independent research into the mathematical modeling of cognitive dynamics. We delineate:
1.  **DDA Theory**: The original cognitive model (rigidity dynamics, trauma accumulation, identity attractors).
2.  **ExACT Architecture**: The industrial-grade engineering framework (MCTS, reflection loops, search patterns) from Microsoft Azure Foundry Labs.

**What is DDA-X?**  
DDA-X integrates the original **D**ynamic **D**ecision **A**lgorithm (DDA) into the e**X**tensible (**X**) agent patterns of the ExACT architecture. This combination enables agents to maintain behavioral stability and identity persistence.

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

## 📜 Origin Story: From Notebook to Network

**A Year of Independent Theoretical Research**

This project didn't start in a dev environment; it began over a year ago as a purely theoretical exercise—a manual "mathematics of mind" scribble in a notebook. I was motivated by a fundamental question: **Can we mathematically model the sensation of a sensing "Self"?**

What evolved from those original recursive equations is **DDA-X**: a research prototype for cognitive architecture where my original DDA theory (The Mind) inhabits the robust engineering body (The Chassis) of Microsoft's ExACT framework. 

[**Read the "Notebook to Network" Journey »**](origin_story.md)

---

## 🌟 The DDA-X Framework

> **"The mind is not a vessel to be filled, but a fire to be kindled — and sometimes, protected from the wind."**

DDA-X is an agent framework that explores **psychological modeling** in artificial intelligence. Inspired by biological stress responses, DDA-X agents feature:

-   **Identity** — A persistent sense of self that survives across contexts
-   **Rigidity** — Defensive responses to surprise, inspired by biological stress responses
-   **Memory** — Experience weighted by emotional salience, not just relevance
-   **Society** — Trust dynamics that emerge from predictability, not agreement
-   **Metacognition** — Structured self-reporting of internal cognitive state

This isn't just another LLM wrapper. It is a **mathematically rigorous implementation of parameter-level cognitive dynamics**, designed to bridge the gap between internal state and behavioral output.

---

## 🚀 Core Framework Mechanics

### Core Novel Hypotheses
#### D1: Surprise-Rigidity Coupling
$$ \rho_{t+1} = \text{clip}(\rho_t + \alpha[\sigma((\epsilon-\epsilon_0)/s) - 0.5], 0, 1) $$
Inverting the curiosity-exploration axiom: Surprise increases defensive rigidity.

#### D2: Identity Attractor Dynamics
$$ \mathbf{F}_{id} = \gamma(\mathbf{x}^* - \mathbf{x}) $$
Modeling "Self" as a persistent force field in parameter space.

#### D3: Rigidity-Modulated LLM Sampling
$$ T(\rho) = T_{low} + (1 - \rho) \cdot (T_{high} - T_{low}) $$
Directly binding internal state ($\rho$) to sampling parameters (T/Top_P).

### Supporting Formalizations
#### D4: Multi-Timescale Trauma
Formalizing "scars" as non-decreasing baseline variables in a multi-step dynamics model.

#### D5: Trust as Predictability
Operationalizing trust as the inverse of cumulative prediction error ($T = 1/(1 + \sum \epsilon)$).

#### D6: Hierarchical Identity
A multi-layered stiffness model ($\gamma_{core} \gg \gamma_{persona}$) balancing stability and flexibility.

#### D7: Metacognitive Accuracy
Structured self-reporting derived from measured internal states.

---

## 🎮 30+ Fully Operational Simulations

**7 Core Validated Experiments** proving foundational theory:

| Simulation | Formulation Tested | Command |
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

## 📊 Implementation Verification

**45/45 Tests Passing (Implementation Validation)**

| Formulation | Tests | Status | Key Evidence |
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
| **Live Backend Backend** | 5 | ✅ | Ollama 768-dim embeddings verified |

**Run it yourself:**
```bash
.\venv\Scripts\python.exe tests/test_ddax_claims.py
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
**Lines of Code**: 5,545  

---

## ⚠️ Limitations

> [!WARNING]
> **Set appropriate expectations before using this framework.**

- **Research prototype** — Not a production-ready deployment system
- **Empirically validated, not formally proven** — Tests verify code correctness, not theoretical guarantees
- **Local models only** — All experiments used `nomic-embed-text` + `GPT-OSS-20B`. Not tested on GPT-4/Claude/Gemini
- **No benchmark comparisons** — Not evaluated against ReACT, AutoGen, CrewAI, or other agent frameworks
- **Builds on ExACT** — Engineering scaffolding from Microsoft; cognitive layer (rigidity, trauma, trust) is the novel contribution

**We welcome critical feedback.** See [full Limitations section](https://github.com/snakewizardd/dda_scaffold#%EF%B8%8F-limitations--honest-assessment) in the GitHub README.

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