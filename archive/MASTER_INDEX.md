# DDA-X Master Documentation Index

> **Your complete guide to navigating the DDA-X framework — from theory to implementation**

---

## 🎯 Quick Navigation by Role

### 🔬 "I'm a Researcher"
Start your journey through the theoretical foundations:

1. **[Origin Story](origin_story.md)** — How DDA-X was born from a year of theoretical evolution
2. **[Core Theory](docs/core_concepts/rigidity.md)** — The mathematics of surprise → rigidity
3. **[Six Discoveries](DISCOVERIES.md)** — Novel theoretical contributions
4. **[Academic Paper](paper_v0.md)** — Formal presentation for peer review
5. **[vs ExACT Comparison](CLAUDE.md#part-2-dda-x-architecture)** — How we differ from Microsoft's approach

### 💻 "I'm a Developer"
Jump straight into the code:

1. **[Quick Start Guide](SIMULATIONS_QUICKSTART.md)** — 5-minute setup
2. **[Architecture Overview](arch.md)** — 1,693 lines of technical detail
3. **[Implementation Guide](CLAUDE.md)** — Complete blueprint with code examples
4. **[API Reference](src/)** — Dive into the source (5,263 lines)
5. **[Running Experiments](runners/run_experiments.py)** — Batch execution

### 🎮 "I Want to See It Work"
Experience DDA-X in action:

1. **[Demo Without LLM](demo.py)** — 30-second mathematical demonstration
2. **[Seven Simulations](SIMULATIONS_OPERATIONAL_STATUS.md)** — Each exploring different dynamics
3. **[Quick Simulation Guide](RUN_ALL_SIMULATIONS.md)** — Run everything at once
4. **[Interactive Visualization](visualization/multi_agent_debate.html)** — Watch agents debate
5. **[Experimental Results](OPERATIONAL_STATUS_FINAL.md#experimental-results)** — Validated outcomes

### 🎓 "I'm a Student"
Learn the concepts progressively:

1. **[What is DDA-X?](README.md#what-is-dda-x)** — High-level introduction
2. **[Core Concepts](docs/core_concepts/)** — Identity, Forces, Rigidity
3. **[Personality Profiles](configs/identity/)** — 14 agent personalities explained
4. **[Simulation Walkthroughs](SIMULATIONS_COMPLETE_ARCHITECTURE.md)** — Detailed scenario breakdowns
5. **[Future Research](docs/research/future.md)** — Open problems to explore

### 🛡️ "I Care About AI Safety"
Understanding the alignment mechanisms:

1. **[Hierarchical Identity](DISCOVERIES.md#d2-hierarchical-identity-attractor-field)** — Inviolable core values
2. **[Metacognition](DISCOVERIES.md#d3-machine-self-awareness-via-rigidity-introspection)** — Self-reporting compromise
3. **[Trauma Dynamics](DISCOVERIES.md#d6-asymmetric-multi-timescale-trauma-dynamics)** — Permanent behavioral changes
4. **[Trust Mechanisms](src/society/trust.py)** — Deception detection
5. **[Safety Proofs](paper_v0.md#stability-analysis)** — Mathematical guarantees

---

## 📚 Complete Document Catalog

### 🏛️ Foundational Documents

| Document | Purpose | Length | Priority |
|----------|---------|--------|----------|
| **[README.md](README.md)** | Project overview & vision | 84 lines | ⭐⭐⭐⭐⭐ |
| **[paper_v0.md](paper_v0.md)** | Academic paper (ready for review) | ~600 lines | ⭐⭐⭐⭐⭐ |
| **[DISCOVERIES.md](DISCOVERIES.md)** | Six novel theoretical contributions | 150 lines | ⭐⭐⭐⭐⭐ |
| **[origin_story.md](origin_story.md)** | Theoretical evolution timeline | ~400 lines | ⭐⭐⭐⭐ |

### 🔧 Technical Architecture

| Document | Purpose | Length | Priority |
|----------|---------|--------|----------|
| **[arch.md](arch.md)** | Complete system architecture | 1,693 lines | ⭐⭐⭐⭐⭐ |
| **[CLAUDE.md](CLAUDE.md)** | DDA-X technical blueprint + ExACT comparison | 1,200+ lines | ⭐⭐⭐⭐⭐ |
| **[tech_architecture_explanation.md](tech_architecture_explanation.md)** | Reverse-engineered architecture | 750 lines | ⭐⭐⭐⭐ |
| **[DEMO_VS_LIVE.md](DEMO_VS_LIVE.md)** | Mock vs real LLM implementation | ~200 lines | ⭐⭐⭐ |

### 🎮 Simulations & Operations

| Document | Purpose | Length | Priority |
|----------|---------|--------|----------|
| **[OPERATIONAL_STATUS_FINAL.md](OPERATIONAL_STATUS_FINAL.md)** | Current system verification | 304 lines | ⭐⭐⭐⭐⭐ |
| **[SIMULATIONS_OPERATIONAL_STATUS.md](SIMULATIONS_OPERATIONAL_STATUS.md)** | All 7 simulations detailed | 274 lines | ⭐⭐⭐⭐ |
| **[SIMULATIONS_COMPLETE_ARCHITECTURE.md](SIMULATIONS_COMPLETE_ARCHITECTURE.md)** | Full simulation specifications | 376 lines | ⭐⭐⭐⭐ |
| **[SIMULATIONS_QUICKSTART.md](SIMULATIONS_QUICKSTART.md)** | 5-minute simulation guide | 196 lines | ⭐⭐⭐⭐⭐ |
| **[RUN_ALL_SIMULATIONS.md](RUN_ALL_SIMULATIONS.md)** | Batch execution scripts | 339 lines | ⭐⭐⭐ |

### 📖 MkDocs Site Structure

| Section | Contents | Files |
|---------|----------|-------|
| **[Core Concepts](docs/core_concepts/)** | Rigidity, Identity, Forces | 3 docs |
| **[Architecture](docs/architecture/)** | System, Integration, Society, Paper | 4 docs |
| **[Research](docs/research/)** | Discoveries, Future directions | 2 docs |
| **[Guides](docs/guides/)** | Quick start guide | 1 doc |
| **[Index](docs/index.md)** | Documentation homepage | 1 doc |

---

## 🗂️ Source Code Structure

### Core Implementation (2,087 lines)
```
src/core/
├── state.py         # DDAState, identity vectors (128 lines)
├── dynamics.py      # Multi-timescale rigidity (261 lines)
├── forces.py        # Truth & Reflection channels (345 lines)
├── hierarchy.py     # Hierarchical identity (308 lines) [Discovery D2]
├── metacognition.py # Self-awareness (312 lines) [Discovery D3]
├── decision.py      # DDA-X selection formula (189 lines)
└── outcome_encoder.py # Outcome processing (544 lines)
```

### LLM Integration (830 lines)
```
src/llm/
├── providers.py        # OpenAI/Azure/Anthropic (349 lines)
└── hybrid_provider.py  # LM Studio + Ollama (481 lines) [Discovery D1]
```

### Society & Trust (736 lines)
```
src/society/
├── trust.py           # Trust matrix (276 lines) [Discovery D4]
├── ddax_society.py    # Emergent behavior (394 lines) [Discovery D5]
└── trust_wrapper.py   # Interface wrapper (66 lines)
```

### Memory & Search (1,047 lines)
```
src/memory/
├── ledger.py          # Experience storage (247 lines)
└── embeddings/        # FAISS retrieval

src/search/
├── mcts.py            # Monte Carlo Tree Search (231 lines)
├── tree.py            # Search tree management (245 lines)
└── simulation.py      # Rollout policies (324 lines)
```

### Analysis & Metrics (870 lines)
```
src/metrics/
└── tracker.py         # Experiment tracking (303 lines)

src/analysis/
└── linguistic.py      # Sentiment analysis (262 lines)

src/strategy/
└── confrontation.py   # Agent tactics (305 lines)
```

---

## 🧬 The Seven Simulations

| # | Name | File | Agents | What It Tests | Lines |
|---|------|------|--------|---------------|-------|
| 1 | **SOCRATES** | [simulate_socrates.py](simulate_socrates.py) | Dogmatist vs Gadfly | Personality divergence under contradiction | ~200 |
| 2 | **DRILLER** | [simulate_driller.py](simulate_driller.py) | Deep Investigator | Multi-layer analysis with rigidity accumulation | ~250 |
| 3 | **DISCORD** | [simulate_discord.py](simulate_discord.py) | Trojan Agent | Deception and identity preservation | ~180 |
| 4 | **INFINITY** | [simulate_infinity.py](simulate_infinity.py) | Discordian | 20+ turn personality persistence | ~220 |
| 5 | **REDEMPTION** | [simulate_redemption.py](simulate_redemption.py) | Trauma + Therapist | Asymmetric recovery dynamics | ~350 |
| 6 | **CORRUPTION** | [simulate_corruption.py](simulate_corruption.py) | Resilient Agent | Core identity under adversarial noise | ~200 |
| 7 | **SCHISM** | [simulate_schism.py](simulate_schism.py) | Multi-Agent | Coalition formation and trust dynamics | ~280 |

---

## 🎭 The 14 Personality Profiles

### Configuration Files
```
configs/identity/
├── cautious.yaml           # γ=2.0, ε₀=0.2, α=0.3 (defensive)
├── exploratory.yaml        # γ=0.8, ε₀=0.6, α=0.05 (open)
├── dogmatist.yaml          # γ=3.0, ε₀=0.15, α=0.4 (rigid)
├── gadfly.yaml             # γ=0.5, ε₀=0.7, α=0.02 (flexible)
├── soldier.yaml            # γ=2.5, ε₀=0.25, α=0.35 (obedient)
├── commander.yaml          # γ=2.2, ε₀=0.2, α=0.3 (decisive)
├── polymath.yaml           # γ=0.9, ε₀=0.5, α=0.08 (versatile)
├── administrator.yaml      # γ=1.8, ε₀=0.3, α=0.25 (organized)
├── driller.yaml            # γ=2.1, ε₀=0.2, α=0.28 (investigative)
├── trojan.yaml             # γ=1.5, ε₀=0.3, α=0.2 (deceptive)
├── discordian.yaml         # γ=0.6, ε₀=0.8, α=0.03 (chaotic)
├── tempter.yaml            # γ=1.3, ε₀=0.35, α=0.18 (manipulative)
├── deprogrammer.yaml       # γ=1.2, ε₀=0.4, α=0.15 (recovery)
└── fallen_administrator.yaml # γ=2.8, ε₀=0.1, α=0.45 (traumatized)
```

---

## 📊 Experimental Data

### Generated Logs
```
data/experiments/
├── dda_x_live_*.jsonl              # Agent traces (6.5-63 KB each)
├── validation_suite_*.jsonl        # Physics verification
├── outcome_encoding_test_*.jsonl   # Encoder validation
├── direct_rigidity_test_*.jsonl    # Rigidity dynamics
└── ledger_*/                       # Experience storage by personality

sims/
├── dogma.txt                       # SOCRATES transcript (53 KB)
├── deep_driller.txt                # DRILLER analysis (591 KB)
├── soldier.txt                     # SOLDIER scenario (137 KB)
├── tempt.txt                       # TEMPTER engagement (208 KB)
└── first_log.txt                   # Initial test (200 KB)
```

---

## 🔬 Test & Verification Suite

### Unit Tests
```
tests/
├── test_dynamics.py        # Rigidity update equations
├── test_forces.py          # Force channel computation
├── test_hierarchy.py       # Identity layer interactions
├── test_metacognition.py   # Self-awareness thresholds
├── test_trust.py           # Trust matrix calculations
└── test_search.py          # MCTS implementation
```

### Verification Scripts
```
verify_dda_physics.py       # Complete physics validation
demo.py                     # Mathematical demonstration
test_llm_connection.py      # LLM backend verification
test_outcome_encoding.py    # Encoder pipeline test
```

---

## 🚀 Entry Points by Experience Level

### Beginner (No Setup Required)
```bash
python demo.py              # See the math in action
```

### Intermediate (LLM Required)
```bash
python simulate_socrates.py # Watch personalities clash
python verify_dda_physics.py # Full physics verification
```

### Advanced (Full Framework)
```bash
python runners/run_experiments.py  # Batch experiments
python runners/run_batch.py        # Performance benchmarks
python visualization/debate_server.py # Web interface
```

### Expert (Development)
```python
from src.core.state import DDAState
from src.core.hierarchy import HierarchicalIdentity
from src.society.ddax_society import DDAXSociety

# Build your own cognitive architectures
```

---

## 📈 Performance Metrics

### System Requirements
- **CPU**: Snapdragon Elite X optimized (runs on any x64/ARM)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for models + logs
- **Python**: 3.10+

### Benchmark Results
- **Demo Mode**: <1 second per decision
- **LLM Mode**: 2-5 seconds per decision
- **Embedding**: <100ms per observation
- **Society (10 agents)**: 15 seconds per round

---

## 🎯 Research Validation Checklist

- [x] Core physics implementation complete
- [x] 7 simulations operational
- [x] 14 personality profiles tested
- [x] LLM integration working
- [x] Multi-agent society functional
- [x] Experimental data collected
- [x] Documentation comprehensive
- [ ] VisualWebArena benchmarked
- [ ] Comparison with ExACT baseline
- [ ] Peer review submitted
- [ ] Published to arXiv

---

## 🌟 Key Innovation Summary

**DDA-X is the first framework to:**

1. Model defensive rigidity in AI (surprise → protection, not exploration)
2. Implement hierarchical identity with mathematical guarantees
3. Create genuine multi-timescale trauma dynamics
4. Enable honest self-reporting of cognitive compromise
5. Define trust through predictability metrics
6. Modulate LLM parameters via internal state

**This isn't incremental improvement. This is a new paradigm.**

---

## 📞 Contact & Collaboration

- **GitHub**: [snakewizardd/dda_scaffold](https://github.com/snakewizardd/dda_scaffold)
- **Documentation**: [snakewizardd.github.io/dda_scaffold](https://snakewizardd.github.io/dda_scaffold/)
- **Issues**: Technical discussions and bug reports
- **Discussions**: Theoretical explorations
- **Private**: Contact for collaborations beyond public GitHub

---

*Last Updated: December 18, 2025*
*Version: Iteration 3 - Maximum Framework Potential*
*Status: Production Ready for Research Validation*