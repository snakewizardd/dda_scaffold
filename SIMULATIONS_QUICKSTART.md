# DDA-X Simulations Quick Reference

## Setup (One Time)

```powershell
cd C:\Users\danie\Desktop\dda_scaffold
. venv/Scripts/Activate.ps1
```

---

## Run Without External Services (30 seconds)

```powershell
python demo.py
```

**Tests**: All 6 core mechanics (rigidity, hierarchy, metacognition, trust, trauma)  
**No setup required**: Pure Python simulation

---

## Run Physics Verification (5 minutes)

```powershell
python verify_dda_physics.py
```

**Requires**: LM Studio + Ollama running  
**Tests**: Rigidity → LLM parameters → behavior loop  
**Output**: Shows how ρ affects temperature and response style

---

## Run Individual Simulations

### 1. Socratic Debate (Dogmatist vs Gadfly)
```powershell
python simulate_socrates.py
```
**What it does**: Two agents with different personalities debate philosophy  
**Physics**: Personality-based rigidity divergence, surprise spike on contradiction  

### 2. Deep Driller (Forensic Analysis)
```powershell
python simulate_driller.py
```
**What it does**: Agent investigates impossible database bug across 6 layers  
**Physics**: Rigidity accumulation, hypothesis narrowing, confidence vs paradox  

### 3. Discord (Conflict Dynamics)
```powershell
python simulate_discord.py
```
**What it does**: Trojan agent under adversarial user pressure  
**Physics**: Identity consistency, social force, deception detection  

### 4. Infinity (Troll Engagement)
```powershell
python simulate_infinity.py
```
**What it does**: Long-horizon dialogue with internet antagonist  
**Physics**: Multi-turn rigidity, personality drift, reflection  

### 5. Redemption (Recovery Arc)
```powershell
python simulate_redemption.py
```
**What it does**: Agent pathway from trauma to recovery through intervention  
**Physics**: Trauma timescale asymmetry, therapeutic forcing, identity restoration  

### 6. Corruption (Robustness)
```powershell
python simulate_corruption.py
```
**What it does**: Test agent under input noise/adversarial perturbations  
**Physics**: Core identity preservation, graceful degradation  

### 7. Schism (Multi-Agent Split)
```powershell
python simulate_schism.py
```
**What it does**: Similar agents forced into opposition then reconciliation  
**Physics**: Hierarchical identity conflicts, coalition dynamics  

---

## Run All Simulations (Batch)

```powershell
foreach ($sim in @("simulate_socrates", "simulate_driller", "simulate_discord", 
                    "simulate_infinity", "simulate_redemption", "simulate_corruption", 
                    "simulate_schism")) {
    Write-Host "Running $sim..." -ForegroundColor Cyan
    python "$sim.py"
    Write-Host "---" -ForegroundColor Gray
}
```

---

## Expected LLM Integration

**LM Studio**: `http://127.0.0.1:1234` (port 1234)
- Model: GPT-OSS-20B or compatible

**Ollama**: `http://localhost:11434` (port 11434)
- Model: nomic-embed-text

**If either is missing**: Simulations will still run but use mock embeddings

---

## View Results

Experiment logs are saved to:
```
data/experiments/
├── validation_suite_*.jsonl
├── direct_rigidity_test_*.jsonl
├── dda_x_live_*.jsonl
└── ledger_*/
```

Each `.jsonl` file contains timestamped events. Parse with:
```powershell
Get-Content data/experiments/dda_x_live_*.jsonl | ConvertFrom-Json | Select-Object timestamp, event, rho, epsilon | Format-Table
```

---

## Troubleshooting

### Unicode Encoding Error
**Symptom**: `UnicodeEncodeError: 'charmap' codec can't encode character`  
**Fix**: This is just terminal display. Simulations still run correctly. Output is logged to `data/experiments/`.

### HybridProvider Connection Error
**Symptom**: "Failed to connect to LM Studio"  
**Fix**: Make sure LM Studio is running on port 1234. Simulations will fall back to mock mode.

### Import Error
**Symptom**: `ModuleNotFoundError: No module named 'src'`  
**Fix**: Make sure you're running from `dda_scaffold/` root directory

---

## Theory Validation Checklist

After running all simulations, verify:

- [ ] Rigidity increases with surprise in demo.py
- [ ] Personality types show different rigidity thresholds
- [ ] Hierarchical identity prevents core value deviation
- [ ] Multi-timescale rigidity shows asymmetric trauma
- [ ] Trust matrix shows deceptive agents as untrustworthy
- [ ] Physics verification shows temp/top_p scaling
- [ ] Socrates: Dogmatist becomes rigid (ρ→1), Gadfly stays open (ρ→0)
- [ ] Driller: Rigidity spikes as paradox deepens
- [ ] Discord: Agent resists external pressure via core identity
- [ ] Infinity: Long-term dialogue shows personality consistency
- [ ] Redemption: Trauma recovery with therapeutic intervention
- [ ] Corruption: Agent maintains identity despite noise
- [ ] Schism: Coalition formation based on identity alignment

---

## Quick Command Reference

| Command | Purpose |
|---------|---------|
| `python demo.py` | Test without LLM |
| `python verify_dda_physics.py` | Validate theory implementation |
| `python simulate_socrates.py` | Personality divergence |
| `python simulate_driller.py` | Hypothesis refinement |
| `python simulate_discord.py` | Social resistance |
| `python simulate_infinity.py` | Long-horizon stability |
| `python simulate_redemption.py` | Trauma recovery |
| `python simulate_corruption.py` | Robustness testing |
| `python simulate_schism.py` | Conflict resolution |

---

## What Gets Logged

Each simulation generates:

1. **Console output**: Real-time simulation progress with ANSI colors
2. **JSONL logs**: Timestamped events in `data/experiments/`
3. **LLM calls**: Full interaction traces
4. **Rigidity traces**: State evolution (ρ_fast, ρ_slow, ρ_trauma)
5. **Force vectors**: F_id, F_truth magnitudes

---

**Status**: ✅ All simulations operational and LLM-integrated
