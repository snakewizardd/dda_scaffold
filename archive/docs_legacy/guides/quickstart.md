# Quick Start (DDA-X Iteration 3)

> **Maintained by**: [snakewizardd](https://github.com/snakewizardd)  
> **Repository**: [https://github.com/snakewizardd/dda_scaffold](https://github.com/snakewizardd/dda_scaffold)

For scientists and engineers wanting to evaluate the **Core Dynamics** without the full multi-agent simulation overhead.

## 1. Prerequisites

*   **Python 3.10+**: `python --version`
*   **Git**: To clone the repository.
*   **LM Studio**: Run local LLM server at `http://127.0.0.1:1234` (Verified with **GPT-OSS-20B on Snapdragon Elite X**).
*   **Ollama**: Run embeddings at `http://localhost:11434` (`ollama pull nomic-embed-text`).

## 2. Setup (5 Minutes)

```bash
# 1. Clone the repository
git clone https://github.com/snakewizardd/dda_scaffold.git
cd dda_scaffold

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate

# 3. Install dependencies (Make sure to install ollama and httpx!)
pip install -r requirements.txt
```

## Quick Test (No LLM Required)

Test the core mechanics without any external services. This proves the **Physics Engine** is functional.

```bash
python simulations/demo.py
```

---

## Running Full Experiments

The simulations are **self-contained** and ready to run.

### Step 1: Start LM Studio
1.  Open **LM Studio**.
2.  Load **GPT-OSS-20B** (or Mistral/Llama).
3.  Start the **Local Server** on port `1234` (Green Start Button).

### Step 2: Start Ollama
1.  Open terminal.
2.  Run: `ollama run nomic-embed-text`.
3.  (This serves embeddings on port `11434`).

### Step 3: Run Any Simulation
All specific simulations are in the root directory.

```bash
# Socratic Debate
python simulations/simulate_socrates.py

# Forensic Analysis
python simulations/simulate_driller.py

# Trauma Restoration
python simulations/simulate_redemption.py
```

Results are automatically saved to `data/experiments/dda_x_live_*.jsonl`.

---

## Understanding the Output

Each experiment logs:

```json
{
  "event": "step",
  "observation": "You see paths left and right",
  "action": "move_forward",
  "pre_rho": 0.199,
  "post_rho": 0.297,
  "delta_rho": 0.097,
  "protect_mode": false
}
```

**Key metrics:**
- `rho`: Rigidity (0=open, 1=defensive)
- `delta_rho`: How much the agent became more/less defensive
- `protect_mode`: True when agent pauses for human guidance

---

## Personality Profiles

Edit `configs/identity/` to create custom personalities:

| Personality | epsilon_0 | alpha | k_base | Behavior |
|-------------|-----------|-------|--------|----------|
| Cautious    | 0.2       | 0.2   | 0.3    | Gets defensive quickly |
| Exploratory | 0.6       | 0.05  | 0.7    | Tolerates surprise well |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/core/dynamics.py` | Rigidity math |
| `src/core/hierarchy.py` | Identity layers |
| `src/core/metacognition.py` | Self-awareness |
| `src/llm/hybrid_provider.py` | LLM integration |
| `runners/run_experiments.py` | Experiment runner |

---

## The Science

**Core equation:**
```
rho_new = rho + alpha * sigmoid((epsilon - epsilon_0) / s)
```

- `epsilon`: Prediction error (surprise)
- `epsilon_0`: Surprise threshold
- `alpha`: Learning rate
- `rho`: Rigidity (defensiveness)

**Insight:** High surprise → high rigidity → conservative behavior

---

## Need Help?
- Full architecture: `docs/architecture/system.md`
- Paper draft: `docs/architecture/paper.md`
- Discoveries: `docs/research/discoveries.md`
