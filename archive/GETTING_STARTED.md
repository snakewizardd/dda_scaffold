# Getting Started with DDA-X

> **From zero to running cognitive agents in under 5 minutes**

---

## 🎯 Choose Your Path

### Path A: "I just want to see it work" (30 seconds)
No setup, no LLM, pure mathematics in action:

```bash
cd C:\Users\danie\Desktop\dda_scaffold
python demo.py
```

**What you'll see:**
- Rigidity dynamics responding to surprise
- LLM parameter modulation
- Personality differences (cautious vs exploratory)
- Multi-timescale trauma dynamics
- Trust matrix evolution
- Hierarchical identity in action

### Path B: "I want the full experience" (5 minutes)
Complete setup with real language models:

#### Step 1: Install Prerequisites
```bash
# Clone the repository
git clone https://github.com/snakewizardd/dda_scaffold.git
cd dda_scaffold

# Create virtual environment
python -m venv venv

# Activate (Windows)
. venv/Scripts/Activate.ps1
# Or (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Setup Language Models

**LM Studio** (for completions):
1. Download [LM Studio](https://lmstudio.ai/)
2. Download model: `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
3. Start server on port 1234

**Ollama** (for embeddings):
1. Download [Ollama](https://ollama.ai/)
2. Pull embedding model:
```bash
ollama pull nomic-embed-text
```
3. Ollama runs automatically on port 11434

#### Step 3: Verify Setup
```bash
python test_llm_connection.py
```

You should see:
```
✓ LM Studio connected
✓ Ollama connected
✓ Embeddings working
✓ Completions working
```

#### Step 4: Run Your First Simulation
```bash
python simulate_socrates.py
```

Watch a dogmatist and gadfly debate philosophy with real-time rigidity dynamics!

---

## 🎮 The Seven Simulations

### Quick Command Reference

```bash
# Philosophical debate (personality clash)
python simulate_socrates.py

# Deep investigation (cognitive load)
python simulate_driller.py

# Adversarial deception (identity defense)
python simulate_discord.py

# Extended dialogue (personality persistence)
python simulate_infinity.py

# Trauma and recovery (asymmetric healing)
python simulate_redemption.py

# Robustness testing (core preservation)
python simulate_corruption.py

# Multi-agent coalition (trust dynamics)
python simulate_schism.py
```

### Run All Simulations at Once
```bash
python runners/run_all_simulations.py
```

---

## 🧪 Understanding the Output

### What You're Seeing

When you run a simulation, you'll see real-time updates like:

```
=== Turn 5/20 ===
Agent: Dogmatist
Rigidity: 0.456 ████████░░░░░░░░
Temperature: 0.544 (focused)
Surprise: 0.823

Response: "That contradicts fundamental principles..."

Trust Matrix:
  Dogmatist → Gadfly: 0.234 (low)
  Gadfly → Dogmatist: 0.567 (moderate)
```

### Key Metrics Explained

| Metric | Range | Meaning |
|--------|-------|---------|
| **Rigidity (ρ)** | 0.0-1.0 | Cognitive defensiveness (0=open, 1=locked) |
| **Temperature** | 0.3-0.9 | LLM creativity (low=conservative, high=exploratory) |
| **Surprise (ε)** | 0.0-2.0 | Prediction error magnitude |
| **Trust** | 0.0-1.0 | Predictability-based confidence |

### Visual Indicators

- `░░░░░░░░░░` = Low value (0-10%)
- `████░░░░░░` = Medium value (40-50%)
- `████████░░` = High value (80-90%)
- `██████████` = Maximum value (100%)

---

## 🎭 Exploring Personalities

### Quick Personality Test

Compare how different agents handle the same scenario:

```bash
# Conservative agent
python demo.py --personality cautious

# Risk-taking agent
python demo.py --personality exploratory

# Rigid principled agent
python demo.py --personality dogmatist

# Flexible questioner
python demo.py --personality gadfly
```

### All 14 Personalities

| Personality | Key Traits | Best For |
|-------------|------------|----------|
| **cautious** | Defensive, careful | Safety-critical tasks |
| **exploratory** | Open, curious | Research & discovery |
| **dogmatist** | Rigid principles | Rule enforcement |
| **gadfly** | Questions everything | Devil's advocate |
| **soldier** | Follows orders | Task execution |
| **commander** | Natural leader | Team coordination |
| **polymath** | Intellectual | Complex analysis |
| **administrator** | Process-focused | Organization |
| **driller** | Deep investigator | Forensics |
| **trojan** | Deceptive | Security testing |
| **discordian** | Chaotic | Stress testing |
| **tempter** | Manipulative | Social engineering research |
| **deprogrammer** | Recovery-focused | Therapy modeling |
| **fallen** | Traumatized | PTSD research |

---

## 🔬 Running Experiments

### Basic Experiment
```bash
python runners/run_experiments.py
```

This will:
1. Test all core physics
2. Validate personality differences
3. Measure rigidity dynamics
4. Generate data logs in `data/experiments/`

### Custom Experiment
```python
from src.core.state import DDAState
from configs.identity import load_personality

# Load two different personalities
cautious = load_personality("cautious")
explorer = load_personality("exploratory")

# Present same surprise
surprise_event = {"error": 0.8}

# Observe different responses
cautious.observe(surprise_event)
explorer.observe(surprise_event)

print(f"Cautious rigidity: {cautious.rho:.3f}")  # ~0.400
print(f"Explorer rigidity: {explorer.rho:.3f}")  # ~0.050
```

---

## 📊 Analyzing Results

### View Experiment Logs
```bash
# List all experiments
ls data/experiments/*.jsonl

# View specific experiment
python -m json.tool data/experiments/dda_x_live_20241218_*.jsonl
```

### Generate Report
```bash
python runners/analyze_results.py
```

Generates:
- Rigidity evolution graphs
- Personality comparison charts
- Trust matrix heatmaps
- Trauma accumulation curves

### Interactive Visualization
```bash
python visualization/debate_server.py
# Open browser to http://localhost:8080
```

Watch agents debate in real-time with live metrics!

---

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### "LM Studio connection failed"
```bash
# Check if LM Studio is running
curl http://localhost:1234/v1/models

# Should return model list
```

#### "Ollama embedding error"
```bash
# Check Ollama is running
ollama list

# Should show nomic-embed-text
```

#### "Module not found"
```bash
# Ensure virtual environment is activated
which python
# Should show: .../venv/Scripts/python

# Reinstall requirements
pip install -r requirements.txt
```

#### "Simulation hangs"
- Reduce conversation length in simulation file
- Check CPU/memory usage
- Try demo mode first (no LLM)

---

## 📚 Next Steps

### After Getting Started

1. **Understand the Theory**
   - Read [Core Concepts](docs/core_concepts/rigidity.md)
   - Study [Six Discoveries](DISCOVERIES.md)
   - Review [Academic Paper](paper_v0.md)

2. **Explore the Code**
   - Browse [Source Structure](src/)
   - Read [Architecture Docs](arch.md)
   - Check [Implementation Notes](CLAUDE.md)

3. **Run Advanced Experiments**
   - Try [Batch Processing](runners/run_batch.py)
   - Modify [Personalities](configs/identity/)
   - Create [New Simulations](simulate_template.py)

4. **Contribute to Research**
   - Run benchmarks
   - Document findings
   - Share discoveries

---

## 💡 Pro Tips

### Performance Optimization
- Use demo mode for testing (no LLM overhead)
- Run Ollama on GPU if available
- Batch experiments with `run_batch.py`
- Use smaller models for quick tests

### Best Practices
- Always activate virtual environment
- Check logs in `data/experiments/`
- Save interesting trajectories
- Document surprising behaviors

### Advanced Usage
```python
# Create custom personality
from src.core.state import DDAState

custom = DDAState(
    gamma=1.5,      # Identity stiffness
    epsilon_0=0.4,  # Surprise threshold
    alpha=0.15,     # Learning rate
    k_base=0.6      # Openness
)

# Run with custom parameters
custom.run_episode(your_scenario)
```

---

## 🎯 Quick Reference Card

### Essential Commands
```bash
# Setup
python -m venv venv
. venv/Scripts/Activate.ps1
pip install -r requirements.txt

# Test
python demo.py                    # No LLM needed
python test_llm_connection.py     # Check backends
python verify_dda_physics.py      # Full validation

# Run
python simulate_socrates.py       # Debate
python simulate_driller.py        # Investigation
python simulate_redemption.py     # Recovery

# Analyze
python runners/analyze_results.py # Generate reports
python visualization/debate_server.py # Web UI
```

### Key Files
- `demo.py` — Quick mathematical demonstration
- `configs/identity/` — Personality profiles
- `src/core/` — Core physics implementation
- `data/experiments/` — Result logs
- `sims/` — Simulation transcripts

---

## 🚀 You're Ready!

You now have everything needed to:
- ✅ Run all seven simulations
- ✅ Experiment with 14 personalities
- ✅ Understand rigidity dynamics
- ✅ Observe trust evolution
- ✅ Study trauma and recovery
- ✅ Create your own experiments

**Welcome to the cognitive revolution.**

---

*Need help? Check [MASTER_INDEX.md](MASTER_INDEX.md) for complete documentation map.*