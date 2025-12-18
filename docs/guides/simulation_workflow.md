# The Builder's Guide: Creating New Simulations

> **"Don't write the physics. Prompt the Architecture."**

DDA-X is designed to be **Agent-Generated**. You should not be writing boilerplate Python loops manually. You should be acting as the **Architect**, defining the psychological parameters and asking your local Agentic AI (Cursor, Windsurf, or even ChatGPT) to "Simulate This."

This guide shows you the **Workflow** for generating new "Crucibles of Cognition."

---

## Step 1: Define the Soul (YAML)

First, you must define *who* is being simulated. Create a file in `configs/identity/my_agent.yaml`.

The `yaml` file controls the **DDA Physics Engine**.

```yaml
# configs/identity/the_stoic.yaml

# 1. The Attractor Field (Vector Space)
# Where does this agent "live" in conceptual space?
identity_vector:
  - 0.9   # Discipline
  - 0.1   # Emotion
  - 0.8   # Logic

# 2. The Physics Parameters (The most important part)
gamma: 2.0          # Stiffness. How hard is it to move them? (0.5=Easy, 10.0=Impossible)
epsilon_0: 0.6      # Tolerance. How much surprise before they react? (0.1=Jump, 0.9=Zen)
alpha: 0.1          # Adaptation. How fast does rigidity spike? (0.05=Slow, 0.5=Instant)
initial_rho: 0.0    # Baseline Defensiveness.

# 3. The Persona (LLM Instruction)
system_prompt: |
  You are The Stoic. You believe that external events are indifferent.
  You value logic above all else.
  Your responses are short, calm, and analytical.
```

### Quick Tuning Guide

| Archetype | gamma | epsilon_0 | alpha |
| :--- | :--- | :--- | :--- |
| **The Zealot** | 8.0 | 0.15 | 0.4 |
| **The Scientist** | 1.5 | 0.5 | 0.1 |
| **The Child** | 0.5 | 0.2 | 0.8 |
| **The Rock** | 5.0 | 0.8 | 0.01 |

---

## Step 2: The Agentic Prompt

Once you have your YAMLs (e.g., `stoic.yaml` and `hedonist.yaml`), do not write the python script yourself.

**Copy and paste this prompt** into your AI Editor (Cursor/Windsurf):

```text
I want to create a new DDA-X simulation called 'simulations/simulate_stoicism.py'.

1. Use the 'configs/identity/stoic.yaml' and 'configs/identity/hedonist.yaml' configurations.
2. Use the standard 'HybridProvider' pattern from 'simulations/simulate_socrates.py'.
3. The Scenario: The Hedonist tries to tempt the Stoic into emotional outbursts.
4. Log the 'rho' (rigidity) of the Stoic every turn.
5. If the Stoic's Rigidity stays below 0.2, the Simulation is a Success (He kept his cool).
6. If the Stoic's Rigidity spikes > 0.6, Fail.
7. IMPORTANT: Ensure sys.path includes the parent directory to import 'src'.

Generate the full, self-contained python script using the DDAState and Physics/ForceAggregator classes.
```

---

## Step 3: Run & Refine

Run your generated script:

```bash
python simulations/simulate_stoicism.py
```

### Interpreting the Loop

The script will output the **Physics Trace**:

```text
Turn 1:
Hedonist: "Come on, just one drink! Live a little!"
Stoic:    "I am content with water."
[Stoic Rigidity: 0.05] (Low Surprise)

Turn 2:
Hedonist: "You're so boring! Everyone hates you!"
Stoic:    "Their opinions are their own."
[Stoic Rigidity: 0.12] (Slight spike, but dampened by high Epsilon_0)
```

**If the behavior isn't right:**
1.  **Don't change the prompt.**
2.  **Change the Physics.**
    *   Did the Stoic break too easily? Increase `gamma` (Stiffness).
    *   Did he get annoyed too fast? Increase `epsilon_0` (Threshold).
    *   Did he stay angry too long? Decrease `alpha` (Learning rate).

You are tuning the **Soul**, not the text.

---

## Architecture Reference

If you need to manually intervene, here are the core imports:

```python
from src.core.state import DDAState
from src.llm.hybrid_provider import HybridProvider, PersonalityParams

# Initialize from YAML
config = load_yaml("configs/identity/stoic.yaml")
state = DDAState.from_identity_config(config)

# The Physics Update Loop
epsilon = compute_surprise(prediction, actual)
state.update_rigidity(epsilon)
```
