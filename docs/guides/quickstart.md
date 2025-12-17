# DDA-X# Quick Start (DDA-X Iteration 3)

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
```

## Quick Test (No LLM Required)

Test the core mechanics without any external services:

```bash
python -c "
from src.core.dynamics import MultiTimescaleRigidity
from src.llm.hybrid_provider import PersonalityParams

# Simulate rigidity dynamics
r = MultiTimescaleRigidity()
for eps in [0.3, 0.5, 0.8, 1.5, 0.2]:
    result = r.update(eps)
    print(f'Surprise={eps:.1f} -> Rigidity={result[\"rho_fast\"]:.3f}')

# See how rigidity affects LLM behavior
print(f'\nHigh rigidity (defensive): {PersonalityParams.from_rigidity(0.8)}')
print(f'Low rigidity (open): {PersonalityParams.from_rigidity(0.2)}')
"
```

---

## Running Full Experiments

### Step 1: Start LM Studio

1. Open LM Studio
2. Load any model (e.g., Mistral, Llama)
3. Start local server (default: http://127.0.0.1:1234)

### Step 2: Start Ollama

```bash
ollama pull nomic-embed-text
ollama serve
```

### Step 3: Run Experiments

```bash
python runners/run_experiments.py
```

Results saved to `data/experiments/dda_x_live_*.jsonl`

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

- Full architecture: `arch.md`
- Paper draft: `paper_v0.md`
- Discoveries: `DISCOVERIES.md`
