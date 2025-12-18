# DDA-X Visualization: DEMO Mode vs LIVE Mode

## 🎭 DEMO Mode (No Setup Required)

Simply open `visualization/multi_agent_debate.html` in any browser.

**What you'll see:**
- **Status:** Red indicator showing "DEMO MODE - Using simulated agents"
- **Responses:** Pre-scripted agent responses
- **Rigidity:** Random fluctuations to simulate dynamics
- **Trust:** Simulated trust updates

**This mode demonstrates:**
- The visual interface
- Basic rigidity dynamics
- Trust matrix evolution
- Force field animations

## 🚀 LIVE Mode (Your ACTUAL Architecture)

### Setup:

1. **Start LM Studio**
   - Load your model (GPT-OSS-20B recommended)
   - Ensure it's running on port 1234

2. **Start Ollama**
   ```bash
   ollama serve
   ollama pull nomic-embed-text  # If not already installed
   ```

3. **Run the WebSocket server**
   ```bash
   # From the dda_scaffold directory with venv activated
   python visualization/debate_server.py
   ```

4. **Open the visualization**
   - Open `visualization/multi_agent_debate.html`
   - You'll see a GREEN indicator: "LIVE MODE - Connected to DDA-X Backend"

### What Makes LIVE Mode Special:

**1. Real LLM Responses**
- Actual GPT-OSS-20B generating unique responses
- Not pre-scripted - different every time
- Token-by-token streaming from the model

**2. Personality-Modulated Generation**
```python
# Your exact architecture in action:
# High rigidity → Low temperature (conservative)
# Low rigidity → High temperature (creative)

Cautious Agent (ρ=0.2): temperature=0.40
Exploratory Agent (ρ=0.1): temperature=1.10
```

**3. Real Prediction Errors**
- Actual surprise from unexpected responses
- True rigidity updates using your equations:
  ```
  ρ_{t+1} = clip(ρ_t + α[σ((ε - ε₀)/s) - 0.5], 0, 1)
  ```

**4. Genuine Trust Evolution**
- Trust based on ACTUAL predictability
- Not random - emerges from real interactions
- Asymmetric trust relationships

**5. Your Exact Hybrid Provider**
- LM Studio for completions
- Ollama for embeddings
- Hardware-optimized for Snapdragon Elite X

## 🔍 How to Verify You're in LIVE Mode

### Visual Indicators:
1. **Green status light** at the top
2. Text shows: **"LIVE MODE - Connected to DDA-X Backend"**
3. Responses are **unique** each time you run a debate
4. Token streaming has **natural, variable delays**

### Console Verification:
Open browser console (F12) and look for:
```
Connected to DDA-X backend
WebSocket connection established
```

### Backend Verification:
In the terminal running `debate_server.py`, you'll see:
```
[WS] Client connected from ('127.0.0.1', xxxxx)
[DEBATE] Round on topic: ...
[RIGIDITY] agent1: ρ 0.200 → 0.234 (ε=0.150)
[TRUST] agent1 → agent2: 0.454
```

## 🎯 Testing the Difference

### In DEMO Mode:
- Click "Start Debate" multiple times
- You'll see the SAME responses each time
- Rigidity changes are random

### In LIVE Mode:
- Click "Start Debate" multiple times
- You'll see DIFFERENT responses each time
- Rigidity changes based on ACTUAL prediction errors
- Watch how cautious agent (low ε₀) gets rigid faster than exploratory

## 🧪 Experiment: Surprise Injection

### DEMO Mode:
- Click "Inject Surprise"
- Both agents jump to high rigidity (preset values)

### LIVE Mode:
- Click "Inject Surprise"
- Rigidity increases based on each agent's ACTUAL parameters:
  - Cautious (α=0.2): Rapid increase
  - Exploratory (α=0.05): Gradual increase
- Temperature automatically adjusts
- Next responses are MORE CONSERVATIVE

## 📊 The Architecture in Action

When in LIVE mode, you're seeing:

```
Your DDA-X Framework
        ↓
HybridProvider (LM Studio + Ollama)
        ↓
Rigidity-Modulated Parameters
        ↓
WebSocket Streaming
        ↓
Real-Time Visualization
```

Every token, every rigidity update, every trust change - it's all your ACTUAL mathematics running live!

## 🎨 Summary

- **DEMO Mode**: Beautiful visualization with simulated behavior
- **LIVE Mode**: Your COMPLETE DDA-X architecture with real LLMs

The HTML automatically detects which mode to use. If the WebSocket connects, you get the full experience. If not, it gracefully falls back to demo mode.

This is YOUR year of research - either simulated for easy viewing, or LIVE with actual cognitive dynamics!