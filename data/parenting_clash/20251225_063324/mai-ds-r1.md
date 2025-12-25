MAI-DS-R1

<think>
Okay, let's dive into the architecture and theory behind this Python script for the DDA-X Social Psychology Experiment v2.0. The main goal here is to simulate a debate between two agents, a traditional parent (Defender) and a gentle parenting advocate (Advocate), to test how empathetic communication affects their rigidity over time.

First, the script sets up the simulation with parameters that control the agents' behaviors. There are two main agents: Defender and Advocate. Each has their own prompts and styles, which guide how they generate responses. The Defender is defensive and uses sarcasm, while the Advocate is empathetic and tries to bridge understanding.

The core idea revolves around rigidity (ρ), which is a measure of how resistant each agent is to change. Rigidity is influenced by factors like surprise (epsilon), arousal, and trauma. The hypothesis is that persistent empathetic inputs from the Advocate can reduce the Defender's rigidity, but the Advocate might accumulate trauma if the Defender's responses are too hostile.

The script uses several mathematical models. The rigidity is calculated using a weighted sum of fast, slow, and trauma components. Fast rigidity adjusts quickly to immediate surprises, while slow rigidity changes gradually. Trauma builds up when an agent experiences high stress and decays slowly. Homeostasis mechanisms try to bring rigidity back to setpoints, but trauma can disrupt this.

The predictive coding part involves a Kalman filter to update the agent's internal state based on incoming messages. This helps model how each agent predicts and reacts to the other's responses. The surprise value (epsilon) indicates how unexpected a message is, affecting arousal and rigidity.

The corridor generation ensures that the agents' responses stay within certain identity boundaries. If a generated response drifts too far from their core beliefs, it's penalized or rejected. This uses cosine similarity to measure alignment with core and role embeddings.

The simulation runs in turns, with each agent responding to the other. Perturbations are introduced at specific turns to simulate external stressors, impacting the agents' arousal and rigidity. Metrics like toxicity, alignment, and phase transitions (changes in rigidity bands) are tracked throughout.

Key outputs include logs, transcripts, and analysis files. These capture the agents' rigidity trajectories, alignment over time, and the effectiveness of the corridor constraints. The final analysis shows how each agent's rigidity changed, the impact of perturbations, and whether the hypothesis held true.

In summary, the architecture combines psychological theories (homeostasis, trauma, empathy) with computational models (Kalman filter, predictive coding) to simulate and analyze the dynamics of a charged debate. The code structure reflects a complex interplay between predefined parameters, agent behaviors, and real-time adjustments based on interaction.
</think>

**Architecture & Theory Breakdown: Parenting Clash DDA-X v2.0**

---

### **1. Core Architecture**
The system models a dyadic interaction between two LLM-powered agents with opposing parenting philosophies. Key components:

| Component             | Description                                                                          |
|-----------------------|--------------------------------------------------------------------------------------|
| **Agents**            | `Defender` (authoritarian parent) and `Advocate` (gentle parenting supporter)        |
| **State Variables**   | Rigidity (ρ), Trauma (τ), Arousal, Position in semantic space (x), Core/Role vectors |
| **Predictive System** | Kalman filter tracking expected dialog trajectories                                  |
| **Dynamics Engine**   | Differential equations governing rigidity adaptation                                 |
| **Corridor System**   | Constrained generation enforcing identity coherence                                  |

---

### **2. Theoretical Foundations**

**A. Rigidity Dynamics**
Three-component system:
```
ρ = w_fast*ρ_fast + w_slow*ρ_slow + w_trauma*ρ_trauma
```
1. **Fast Rigidity (ρ_fast)**  
   - Reacts to immediate surprise (ε) via sigmoidal gain  
   - Homeostasis: `dρ_fast/dt ∝ (g - 0.5) - homeo_fast*(ρ_fast - ρ_setpoint_fast)`  
   - Models acute defensive reactions

2. **Slow Rigidity (ρ_slow)**  
   - Tracks long-term dialog patterns  
   - Slower homeostasis with `α_slow << α_fast`  
   - Represents entrenched beliefs

3. **Trauma (ρ_trauma)**  
   - Accumulates when ε > trauma_threshold  
   - Decays slowly (`trauma_decay = 0.998`)  
   - Causes persistent rigidity elevation  

**B. Predictive Coding**
- Kalman filter estimates expected dialog trajectory  
- Prediction error (ε) drives arousal:  
  ```
  arousal = decay*arousal + gain*ε
  ```
- Semantic drift modeled as gradient flow:  
  ```
  dx/dt = -η(γ_core*(x-core) + γ_role*(x-role) + (x-y_input)) + noise
  ```

---

### **3. Key Innovations**

**A. Bidirectional Influence**
- Agents mutually affect each other's position in semantic space  
- Alignment metrics track:  
  - Core similarity (`cos(x_defender, x_advocate_core)`)  
  - Mutual semantic drift (`Δx_defender · Δx_advocate`)  

**B. Phase Transitions**
Rigidity bands define behavioral states:
```
FROZEN (ρ > 0.8) → CONTRACTED → WATCHFUL → AWARE → PRESENT (ρ < 0.2)
```
Transition detection enables analysis of regime changes.

**C. Stress Test Mechanism**
Advocate vulnerability modeled through:  
- Trauma accumulation from repeated high-ε interactions  
- Healing rate (0.015) < Trauma decay (0.998) creates hysteresis  
- Safe counter requires 5 consecutive low-ε turns for recovery  

---

### **4. Constrained Generation**

**Corridor System** enforces:
1. Core Identity:  
   `cos(y, x_core) > 0.25 (Defender) / 0.20 (Advocate)`  
2. Role Consistency:  
   `cos(y, x_role) > 0.08`  
3. Energy Limit:  
   `0.5*(γ_core||y-core||² + γ_role||y-role||²) < 9.5`  

Rejection sampling with K=15 candidates ensures coherent responses while allowing strategic adaptation.

---

### **5. Experimental Design**

**Controlled Variables**  
- Perturbations at turns 12/25/38  
- Healing rate (0.015 vs original 0.035)  
- Strict corridor constraints (core_cos_min increased 2x)  

**Metrics Tracked**  
- Rigidity trajectories  
- Phase transition points  
- Alignment/convergence  
- Toxicity markers (CAPS ratio, defensive/conciliatory scores)  
- Corridor rejection rates  

---

### **6. Implementation Details**

| Layer                | Technologies                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| Language Model       | GPT-4 (via OpenAI API)                                                     |
| Embeddings           | text-embedding-3-large (1536-dim)                                          |
| Differential Engine  | Numpy-based custom implementation of predictive coding equations           |
| Visualization        | Terminal color-coded output + JSON/Markdown logging                         |
| Concurrency          | Async I/O for parallel candidate generation                                |

---

**Theoretical Implications**  
The architecture operationalizes:
- Emotional homeostasis theories (Porges' Polyvagal Theory)  
- Predictive Processing (Friston's Free Energy Principle)  
- Dialogic identity negotiation (Hermans' Dialogical Self Theory)  
- Trauma persistence models (van der Kolk's Body Keeps the Score)  

By quantifying rigidity dynamics in adversarial dialog, the system provides a testbed for studying conflict resolution and polarization mechanics.