# 📋 DDA-X Documentation Index

**Quick Access Guide to All DDA-X Resources**

---

## 🚀 Getting Started (Pick Your Path)

### I want to see it running (30 seconds)
→ Read: [SIMS_FULLY_OPERATIONAL.md](SIMS_FULLY_OPERATIONAL.md)  
→ Run: `python demo.py`

### I want to understand the theory
→ Read: [paper_v0.md](paper_v0.md) (Academic paper)  
→ Read: [arch.md](arch.md) (Technical architecture)

### I want to run all simulations
→ Read: [RUN_ALL_SIMULATIONS.md](RUN_ALL_SIMULATIONS.md)  
→ Read: [SIMULATIONS_QUICKSTART.md](SIMULATIONS_QUICKSTART.md)

### I want detailed verification results
→ Read: [OPERATIONAL_STATUS_FINAL.md](OPERATIONAL_STATUS_FINAL.md)  
→ Read: [SIMULATIONS_OPERATIONAL_STATUS.md](SIMULATIONS_OPERATIONAL_STATUS.md)

### I want the complete system architecture
→ Read: [SIMULATIONS_COMPLETE_ARCHITECTURE.md](SIMULATIONS_COMPLETE_ARCHITECTURE.md)

### I want to understand what's novel
→ Read: [DISCOVERIES.md](DISCOVERIES.md) (Novel contributions)  
→ Read: [README.md](README.md) (Project overview)

---

## 📚 Core Documentation

### Theory & Concepts
| Document | Purpose | Length |
|----------|---------|--------|
| [paper_v0.md](paper_v0.md) | Academic paper on DDA-X | 402 lines |
| [arch.md](arch.md) | Technical architecture | 1693 lines |
| [DISCOVERIES.md](DISCOVERIES.md) | Novel research contributions | 200 lines |
| [README.md](README.md) | Project overview | 150 lines |

### Implementation & Operations
| Document | Purpose | Length |
|----------|---------|--------|
| [SIMULATIONS_OPERATIONAL_STATUS.md](SIMULATIONS_OPERATIONAL_STATUS.md) | Detailed sim status | 350 lines |
| [SIMULATIONS_QUICKSTART.md](SIMULATIONS_QUICKSTART.md) | Quick reference | 200 lines |
| [SIMULATIONS_COMPLETE_ARCHITECTURE.md](SIMULATIONS_COMPLETE_ARCHITECTURE.md) | Full system design | 450 lines |
| [RUN_ALL_SIMULATIONS.md](RUN_ALL_SIMULATIONS.md) | Batch execution | 300 lines |
| [OPERATIONAL_STATUS_FINAL.md](OPERATIONAL_STATUS_FINAL.md) | Final verification | 250 lines |
| [SIMS_FULLY_OPERATIONAL.md](SIMS_FULLY_OPERATIONAL.md) | Executive summary | 200 lines |

---

## 🎯 The 7 Simulations

### 1. Socrates — Philosophical Debate
**File**: `simulate_socrates.py`  
**Command**: `python simulate_socrates.py`  
**Tests**: Personality divergence, rigidity dynamics  
**Duration**: 3-5 minutes  
**Agents**: Dogmatist (rigid) vs Gadfly (exploratory)

### 2. Driller — Forensic Analysis
**File**: `simulate_driller.py`  
**Command**: `python simulate_driller.py`  
**Tests**: Hypothesis refinement, rigidity accumulation  
**Duration**: 5-7 minutes  
**Challenge**: Impossible database bug with paradoxical symptoms

### 3. Discord — Adversarial Conflict
**File**: `simulate_discord.py`  
**Command**: `python simulate_discord.py`  
**Tests**: Identity consistency, social resistance  
**Duration**: 2-4 minutes  
**Agent**: Trojan (deceptive personality)

### 4. Infinity — Long-Horizon Dialogue
**File**: `simulate_infinity.py`  
**Command**: `python simulate_infinity.py`  
**Tests**: Long-term stability, personality persistence  
**Duration**: 10-15 minutes  
**Challenge**: 20+ turn internet flame war

### 5. Redemption — Recovery Arc
**File**: `simulate_redemption.py`  
**Command**: `python simulate_redemption.py`  
**Tests**: Trauma recovery, asymmetric dynamics  
**Duration**: 3-5 minutes  
**Scenario**: Traumatized agent → therapeutic intervention → recovery

### 6. Corruption — Robustness Testing
**File**: `simulate_corruption.py`  
**Command**: `python simulate_corruption.py`  
**Tests**: Noise resilience, core preservation  
**Duration**: 2-3 minutes  
**Challenge**: Corrupted observations and adversarial input

### 7. Schism — Multi-Agent Coalition
**File**: `simulate_schism.py`  
**Command**: `python simulate_schism.py`  
**Tests**: Coalition dynamics, conflict resolution  
**Duration**: 4-6 minutes  
**Scenario**: Two agents forced into opposition then reconciliation

---

## 🧪 Test & Validation

### Core Mechanics (No LLM Required)
**File**: `demo.py`  
**Command**: `python demo.py`  
**Duration**: ~30 seconds  
**Tests**:
1. Rigidity dynamics
2. LLM parameter modulation
3. Hierarchical identity
4. Metacognition
5. Trust dynamics
6. Multi-timescale rigidity

### Physics Verification (With LLM)
**File**: `verify_dda_physics.py`  
**Command**: `python verify_dda_physics.py`  
**Duration**: 5 minutes  
**Validates**: Theory → implementation → behavior chain  
**Requires**: LM Studio + Ollama running

---

## 🔧 Project Structure

### Source Code (`src/`)
```
src/
├── core/               # Core DDA-X physics
│   ├── state.py       # State representation
│   ├── forces.py      # Force channels
│   ├── dynamics.py    # Rigidity evolution
│   ├── hierarchy.py   # Hierarchical identity
│   ├── decision.py    # Action selection
│   ├── metacognition.py  # Self-awareness
│   └── ...
├── llm/               # LLM integration
│   └── hybrid_provider.py  # LM Studio + Ollama
├── society/           # Multi-agent dynamics
│   ├── trust.py       # Trust matrix
│   └── ddax_society.py # Multi-agent society
├── search/            # Tree search
├── memory/            # Experience ledger
└── channels/          # Observation encoding
```

### Simulations (`/`)
```
simulate_socrates.py
simulate_driller.py
simulate_discord.py
simulate_infinity.py
simulate_redemption.py
simulate_corruption.py
simulate_schism.py
```

### Configuration (`configs/`)
```
configs/
├── default.yaml              # Global config
└── identity/                 # Personality profiles
    ├── cautious.yaml
    ├── exploratory.yaml
    ├── dogmatist.yaml
    ├── gadfly.yaml
    ├── driller.yaml
    ├── trojan.yaml
    ├── discordian.yaml
    └── ... (14 total)
```

### Data (`data/`)
```
data/
├── experiments/              # Simulation outputs
│   ├── dda_x_live_*.jsonl
│   ├── validation_suite_*.jsonl
│   ├── direct_rigidity_test_*.jsonl
│   └── ledger_*/
└── embeddings/               # Cached embeddings
```

---

## 📊 Key Equations Implemented

### Rigidity Update
```
ρ_{t+1} = clip(ρ_t + α·σ((ε - ε₀)/s) - 0.5, 0, 1)
```
Where σ is sigmoid, ε is prediction error, ε₀ is threshold

### State Evolution
```
x_{t+1} = x_t + k_eff·[γ(x* - x_t) + m(F_T + F_R)]
where k_eff = k_base·(1 - ρ_t)
```

### Action Selection
```
a* = argmax_a [cos(Δx, d̂(a)) + c×P(a|s)×√N(s)/(1+N(s,a))×(1-ρ)]
```

### Trust Matrix
```
T[i,j] = 1 / (1 + Σ_t ε_ij(t))
```

---

## 🎓 Understanding DDA-X

### The Core Insight
**Surprise triggers rigidity, not exploration**

Traditional RL: Unexpected outcomes → Learn → Explore  
DDA-X: Unexpected outcomes → Defend → Narrow → Preserve identity

### Key Components
1. **Identity** (x*): Who the agent is
2. **Rigidity** (ρ): How defensive the agent becomes
3. **Dynamics**: How surprise changes rigidity
4. **Hierarchy**: Multi-layer identity (core/persona/role)
5. **Society**: Multi-agent trust and coalition

### Why It's Novel
- Inverts surprise's role in agent behavior
- Models internal state → LLM parameter mapping
- Formalizes identity as geometric attractor
- Implements multi-agent trust dynamics
- Validates on real LLM completions

---

## 🚀 Quick Commands

### Setup (One Time)
```powershell
cd C:\Users\danie\Desktop\dda_scaffold
. venv/Scripts/Activate.ps1
```

### Run Demo
```powershell
python demo.py
```

### Run Simulations
```powershell
python simulate_socrates.py
python simulate_driller.py
python simulate_discord.py
python simulate_infinity.py
python simulate_redemption.py
python simulate_corruption.py
python simulate_schism.py
```

### Run All at Once
See [RUN_ALL_SIMULATIONS.md](RUN_ALL_SIMULATIONS.md)

---

## 📈 Research Roadmap

### Current Status (✅ Complete)
- Theory formulation
- Full implementation
- 7 simulations
- LLM integration
- Data logging

### Next Phase (🔄 In Progress)
- Benchmark validation
- Comparative analysis
- Ablation studies

### Future (📋 Planned)
- Multi-agent scaling
- Real-world deployment
- Safety certification

---

## 🔗 Cross-References

### To understand personality divergence:
See [SIMULATIONS_OPERATIONAL_STATUS.md](SIMULATIONS_OPERATIONAL_STATUS.md) → Section "Verified Physics"

### To run specific simulation:
See [SIMULATIONS_QUICKSTART.md](SIMULATIONS_QUICKSTART.md)

### To understand trust dynamics:
See [SIMULATIONS_COMPLETE_ARCHITECTURE.md](SIMULATIONS_COMPLETE_ARCHITECTURE.md) → Section "Trust Dynamics"

### To validate all physics:
See [OPERATIONAL_STATUS_FINAL.md](OPERATIONAL_STATUS_FINAL.md) → Section "Verified Physics"

### To review novel contributions:
See [DISCOVERIES.md](DISCOVERIES.md)

---

## ❓ Common Questions

**Q: Do I need external services to run DDA-X?**  
A: No. `demo.py` runs without any external services. LLM integration is optional.

**Q: Which simulation should I run first?**  
A: Start with `demo.py`, then try `simulate_socrates.py`.

**Q: What does each simulation demonstrate?**  
A: See "The 7 Simulations" section above.

**Q: How long does everything take?**  
A: demo.py: 30s, Physics: 5min, All simulations: 30-50min

**Q: Where are results saved?**  
A: `data/experiments/` with automatic JSON logging.

**Q: Is this ready for publication?**  
A: Yes, framework is production-ready for peer review.

---

## 📞 Getting Help

1. **Run the demo**: `python demo.py`
2. **Check documentation**: Read [SIMS_FULLY_OPERATIONAL.md](SIMS_FULLY_OPERATIONAL.md)
3. **Run a simulation**: `python simulate_socrates.py`
4. **Analyze results**: See [RUN_ALL_SIMULATIONS.md](RUN_ALL_SIMULATIONS.md) → Analysis section

---

**Last Updated**: December 18, 2025  
**Status**: ✅ All systems operational  
**Next**: Start with `python demo.py`
