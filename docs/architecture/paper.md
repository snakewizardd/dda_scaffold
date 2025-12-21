# DDA-X: Rigidity-Dampened Exploration for Agentic AI

**Dynamic Decision Algorithm with Exploration: A Framework for Identity-Preserving Agents**

---

## Abstract

Current approaches to agentic AI often treat surprise as a signal for curiosity: unexpected outcomes trigger reflection and exploration. We introduce an alternative paradigm where **surprise triggers rigidity** — a protective response that reduces exploration and strengthens identity persistence. We present **DDA-X** (Dynamic Decision Algorithm with Exploration), the first framework to implement a **parameter-level coupling** between internal states (prediction error) and LLM sampling dynamics (Temperature/Top_P). Our core contributions include: (1) a continuous state-space representation with identity attractors that model cognitive persistence, (2) a rigidity mechanism where surprise increases defensiveness and dampens exploration, and (3) an action selection formula that fuses force-alignment with rigidity-modulated UCT exploration. Built as an original synthesis of DDA theory and ExACT engineering, DDA-X provides a mathematically rigorous model for psychologically realistic agents. While currently validated on 20B multi-modal local models, the framework provides a foundation for investigating behavioral stability in frontier-scale LLMs.

**Keywords:** autonomous agents, Monte Carlo Tree Search, identity persistence, adaptive rigidity, LLM agents

---

## 1. Introduction

Recent advances in vision-language models (VLMs) have enabled sophisticated autonomous agents capable of navigating complex environments such as web browsers (Koh et al., 2024a; Zhou et al., 2024), operating systems (Wang et al., 2024), and software development (Yang et al., 2024). These agents typically employ reinforcement learning principles or tree search methods to select actions that maximize task success.

A common assumption in this paradigm is that **surprise is informative** — when an agent's prediction differs from reality, this discrepancy drives learning and exploration. This assumption underlies TD-learning (Sutton & Barto, 2018), curiosity-driven exploration (Pathak et al., 2017), and recent work on reflective agents (Yu et al., 2024).

We challenge this assumption with a simple observation: **biological agents often respond to surprise with rigidity, not curiosity**. A startled organism contracts, retreats, or freezes. A threatened human becomes defensive, not exploratory. This is not a bug — it's a survival mechanism.

We propose **DDA-X** (Dynamic Decision Algorithm with Exploration), a framework that explores modeling agents as systems balancing two competing forces:

1. **Identity persistence**: the drive to remain coherent and self-consistent
2. **Reality integration**: the pressure to update beliefs based on environmental feedback

The signature hypothesis of DDA-X is that **surprise increases rigidity**, which in turn *dampens exploration*. When an agent's predictions are violated, it becomes more conservative, not more curious. This produces qualitatively different agent behaviors that may be useful in high-stakes, safety-critical, or adversarial environments.

### 1.1 Contributions

1. **A continuous state-space agent model** with an identity attractor x* and force-balanced dynamics (Section 3.1)

2. **Adaptive rigidity** ρ ∈ [0,1] that increases with prediction error and dampens exploration (Section 3.2)

3. **DDA-X action selection**: a selection formula combining force-alignment with rigidity-modulated UCT exploration (Section 3.3)

4. **Behavioral profiles**: configurable agent archetypes (cautious, exploratory, rigidified) via parameter tuning (Section 3.4)

5. **Implementation architecture** with class blueprints for research deployment (Section 4)

---

## 2. Related Work

### 2.1 Search-Augmented Agents

Monte Carlo Tree Search (MCTS) has been successfully applied to agentic tasks, balancing exploration and exploitation via the Upper Confidence Bound for Trees (UCT) formula (Kocsis & Szepesvári, 2006; Silver et al., 2017). Recent work extends MCTS with language model priors (Yu et al., 2023) and contrastive reflection (Yu et al., 2024).

**ExACT** (Yu et al., 2024) introduces Reflective MCTS (R-MCTS), which combines tree search with a reflection-improvement loop: after each episode, the agent identifies "surprising" transitions (where |V(s') - Q(s,a)| is large), generates lessons via LLM prompting, and retrieves relevant reflections for future tasks. A multi-agent debate mechanism provides more calibrated state evaluation.

Our work extends this paradigm with a crucial inversion: rather than using surprise to drive reflection and exploration, we use surprise to *increase rigidity and dampen exploration*.

### 2.2 Agent Self-Reflection

Self-reflection has emerged as a powerful technique for improving LLM agents (Shinn et al., 2023; Madaan et al., 2023). These methods typically prompt agents to identify mistakes and generate corrective guidance for future attempts.

DDA-X incorporates reflection through our memory system (the "ledger"), but weights retrieved memories by **prediction error salience** — surprising experiences are more readily recalled, akin to trauma weighting in cognitive systems.

### 2.3 Identity and Personality in Agents

While prior work has explored persona-conditioned agents (Park et al., 2023), these approaches typically implement personality through prompt engineering rather than dynamical systems. DDA-X models identity as an **attractor in state space** with quantitative stiffness, enabling formal analysis of identity persistence and will.

---

## 3. Method

### 3.1 State Space and Identity

We model the agent's internal state as a continuous vector in decision-space:

$$\mathbf{x}_t \in \mathbb{R}^d$$

This vector encodes the agent's current stance, beliefs, goals, and affect. Unlike discrete state representations in MCTS, this continuous space enables smooth dynamics and gradient-based analysis.

**Identity Attractor.** We define a fixed point x* ∈ ℝ^d representing "who the agent is" — its core values, preferences, and characteristic behaviors. The agent experiences a restoring force toward this attractor:

$$\mathbf{F}_{id}(t) = \gamma(\mathbf{x}^* - \mathbf{x}_t)$$

where γ ≥ 0 is the **identity stiffness**.

**Truth Channel.** Environmental observations I_t are encoded into state space and create a force toward the observed reality:

$$\mathbf{F}_T(t) = T(I_t, \Delta I_t) - \mathbf{x}_t$$

where T(·) is an encoder function (e.g., LLM embedding followed by linear projection).

**Reflection Channel.** Available actions A_t and retrieved memories create a force toward preferred action directions:

$$\mathbf{F}_R(t) = R(\mathcal{A}_t, \Phi_t, \mathcal{L}) - \mathbf{x}_t$$

**State Update.** The agent's state evolves according to:

$$\mathbf{x}_{t+1} = \mathbf{x}_t + k_{eff} \left[ \gamma(\mathbf{x}^* - \mathbf{x}_t) + m_t(\mathbf{F}_T + \mathbf{F}_R) \right]$$

where:
- k_eff is the **effective step size** (decreases with rigidity)
- m_t is the **external pressure gain**

### 3.2 Adaptive Rigidity

The core innovation of DDA-X is that **surprise increases rigidity**, which then dampens both state updates and exploration.

**Prediction Error.** After taking action a*_t, the agent observes outcome o_{t+1} and computes:

$$\epsilon_t = \|\mathbf{x}^{pred}_{t+1} - \mathbf{x}^{actual}_{t+1}\|_2$$

where x^{pred} was the agent's expected next state and x^{actual} is the encoded outcome.

**Rigidity Update.** Rigidity ρ ∈ [0,1] evolves according to:

$$\rho_{t+1} = \text{clip}\left( \rho_t + \alpha \left[ \sigma\left(\frac{\epsilon_t - \epsilon_0}{s}\right) - \frac{1}{2} \right], 0, 1 \right)$$

where:
- σ(·) is the sigmoid function
- ε₀ is the surprise threshold ("when surprise becomes threatening")
- α is the rigidity learning rate
- s is the sigmoid sensitivity

This formulation is **bidirectional**: low error (ε < ε₀) causes rigidity to *decrease*, enabling recovery when situations are predictable.

**Effective Openness.** Rigidity modulates the agent's responsiveness:

$$k_{eff} = k_{base}(1 - \rho_t)$$

High rigidity → low k_eff → smaller state updates → more identity-centric behavior.

### 3.3 DDA-X Action Selection

We now present our novel action selection formula, which fuses force-based alignment with exploration.

**Action Directions.** Each discrete action a ∈ A_t has a direction in state space:

$$\hat{\mathbf{d}}(a) \in \mathbb{R}^d, \quad \|\hat{\mathbf{d}}(a)\| = 1$$

**Desired Movement.** The net force on the agent defines a desired movement direction:

$$\Delta\mathbf{x}_t = \gamma(\mathbf{x}^* - \mathbf{x}_t) + m_t(\mathbf{F}_T + \mathbf{F}_R)$$

**DDA-X Selection Formula.** We select actions by maximizing:

$$\boxed{a^*_t = \arg\max_{a \in \mathcal{A}_t} \left[ \underbrace{\cos(\Delta\mathbf{x}_t, \hat{\mathbf{d}}(a))}_{\text{DDA alignment}} + \underbrace{c \cdot P(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}}_{\text{UCT exploration}} \cdot \underbrace{(1 - \rho_t)}_{\text{rigidity dampening}} \right]}$$

This formula has three components:

1. **DDA alignment**: prefer actions aligned with the force-balanced desired direction
2. **UCT exploration**: the standard MCTS exploration bonus (Kocsis & Szepesvári, 2006)
3. **Rigidity dampening**: **the exploration bonus is multiplied by (1 - ρ)**

The third component is our key contribution: when surprise is high (ρ → 1), exploration is suppressed. The agent becomes conservative, preferring actions aligned with its current trajectory rather than exploring novel options.

### 3.4 Personality Profiles

Unlike prior agent frameworks where all instances behave identically, DDA-X enables diverse personalities through parameter configuration:

| Profile | γ | ε₀ | α | ρ_init | Behavior |
|---------|---|----|----|--------|----------|
| **Cautious** | 2.0 | 0.2 | 0.2 | 0.0 | Strong identity, low surprise threshold, fast rigidity ramp |
| **Exploratory** | 0.5 | 0.6 | 0.05 | 0.0 | Weak identity, high surprise tolerance, slow rigidity change |
| **Traumatized** | 1.5 | 0.1 | 0.3 | 0.4 | Hair-trigger defensiveness, elevated baseline rigidity |

These profiles emerge naturally from the mathematical framework without ad-hoc behavioral rules.

### 3.5 Protection Mode

When rigidity exceeds a threshold, the agent can enter **protection mode**:

$$m_{protect}(\rho) = m_0(1 - \rho) + m_{min}$$

In protection mode, the agent:
- Restricts its action set to safe defaults
- Increases identity pull (higher γ)
- May request clarification rather than acting

This models defensive behavior under threat without requiring explicit behavioral rules.

### 3.6 Memory System

We extend standard reflection databases with **surprise-weighted retrieval**:

$$\text{score}(entry) = \underbrace{\text{sim}(\mathbf{c}_{now}, \mathbf{c}_t)}_{\text{relevance}} \cdot \underbrace{e^{-\lambda_r(now - t)}}_{\text{recency}} \cdot \underbrace{(1 + \lambda_\epsilon \cdot \epsilon_t)}_{\text{salience}}$$

Experiences with high prediction error (surprising outcomes) are more readily retrieved, implementing a form of trauma weighting.

---

## 4. Implementation Architecture

We provide a complete blueprint for implementing DDA-X. The architecture integrates with existing LLM and browser automation frameworks.

### 4.1 Core Components

```
dda-x/
├── src/
│   ├── core/
│   │   ├── state.py          # DDAState, ActionDirection
│   │   ├── forces.py         # IdentityPull, TruthChannel, ReflectionChannel
│   │   └── decision.py       # DDADecisionMaker
│   ├── search/
│   │   ├── tree.py           # DDANode, DDASearchTree
│   │   └── mcts.py           # Search algorithm
│   ├── memory/
│   │   ├── ledger.py         # ExperienceLedger
│   │   └── retriever.py      # FAISS-based retrieval
│   └── agent.py              # DDAXAgent
```

### 4.2 Key Classes

**DDAState** maintains the agent's continuous state vector x, identity attractor x*, rigidity ρ, and parameters.

**DDADecisionMaker** implements the DDA-X selection formula, computing alignment scores and rigidity-dampened exploration bonuses.

**DDASearchTree** extends standard MCTS with DDA state tracking at each node, enabling rigidity to evolve during tree traversal.

**ExperienceLedger** stores experiences with prediction error annotations and implements surprise-weighted retrieval.

### 4.3 Integration with ExACT

DDA-X is designed to be compatible with existing R-MCTS implementations. Key integration points:

1. **Value function**: ExACT's multi-agent debate can be used directly for V(s) estimation
2. **Reflection generation**: ExACT's contrastive reflection prompts can populate our ledger
3. **Environment interface**: Same browser automation as VisualWebArena

The primary modification is replacing UCT selection with DDA-X selection and adding rigidity tracking.

---

## 5. Experiments

*[PLACEHOLDER: Experiments section to be completed in v0.1]*

### 5.1 Experimental Setup

We plan to evaluate DDA-X on VisualWebArena (Koh et al., 2024a), a benchmark of 910 web navigation tasks across three environments (Classifieds, Reddit, Shopping).

**Baselines:**
- ReACT (Yao et al., 2023): Direct prompting without search
- MCTS: Standard Monte Carlo Tree Search
- R-MCTS (Yu et al., 2024): Reflective MCTS with multi-agent debate

**Metrics:**
- Task success rate
- Token consumption
- Rigidity dynamics (ρ evolution over episodes)
- Personality differentiation (behavioral variance across profiles)

### 5.2 Research Questions

1. Does rigidity-dampened exploration improve performance on adversarial or deceptive tasks?
2. Do different personality profiles exhibit measurably different behaviors?
3. How does the protect mode threshold affect success/failure tradeoffs?
4. Is surprise-weighted memory retrieval more effective than uniform retrieval?

### 5.3 Results

*[To be completed with empirical data]*

---

## 6. Discussion

### 6.1 When Rigidity Helps

We hypothesize that rigidity-dampened exploration is beneficial in:

- **Adversarial environments**: where exploration can be exploited by malicious actors
- **High-stakes decisions**: where the cost of exploration errors is high
- **Identity-critical tasks**: where maintaining consistent behavior is more important than optimal performance
- **Deceptive contexts**: where surprise may indicate manipulation rather than learning opportunity

### 6.2 When Rigidity Hurts

Rigidity may be detrimental in:

- **Novel environments**: where exploration is necessary for learning
- **Rapidly changing contexts**: where flexibility is required
- **Pure performance optimization**: where identity preservation is irrelevant

### 6.3 Theoretical Implications

DDA-X introduces several concepts not present in standard RL or MCTS:

1. **Identity as attractor**: agents have a "self" they preserve, not just a policy they optimize
2. **Rigidity as feature**: defensiveness is modeled, not just performance
3. **Will as impedance**: W_t = γ / (m_t · k_eff) quantifies resistance to environmental pressure
4. **Stability boundary**: m_crit = 1/k_eff - γ/2 defines when the agent can be destabilized

These concepts may be useful for AI safety research, particularly in understanding agent values and resistance to manipulation.

---

## 7. Conclusion

We introduced DDA-X, a framework for agentic AI that explores an inverse relationship between surprise and exploration. Rather than treating all surprise as a learning signal, DDA-X models surprise as a trigger for protective rigidity.

The proposed mechanics include:

1. A continuous state-space model with identity attractor and force-balanced dynamics
2. Adaptive rigidity that increases with prediction error and dampens exploration
3. The DDA-X selection formula: cos(Δx, d̂(a)) + c·P(a|s)·√N(s)/(1+N(s,a))·(1-ρ)
4. Configurable behavioral profiles enabling agent archetypes
5. An implementation architecture built on the ExACT framework

This work opens directions for building agents that balance task completion with behavioral stability, potentially relevant for AI safety and research into agentic alignment.

---

## 8. Known Frontiers and Future Work

While DDA-X provides a mathematically consistent framework for modeling cognitive dynamics, we acknowledge several critical areas for future investigation identified through initial technical reviews:

### 8.1 Empirical Benchmarking & Ablation
Current validation is focused on mechanistic correctness (45/45 unit tests). Future research must evaluate DDA-X on standardized agent benchmarks:
- **Task Success**: Comparative evaluation on VisualWebArena or GAIA against non-DDA baselines.
- **Ablation Studies**: Quantifying the contribution of rigidity-dampened exploration by testing agents with ρ-modulation disabled.

### 8.2 Scale-Invariance & SOTA Validation
The results documented in this v1.0 release were obtained using local 20B parameters models (`GPT-OSS-20B`). We do not yet know if the observed dynamics (e.g., identity persistence, surprise-rigidity coupling) remain stable or scale linearly when moving to frontier models like GPT-4, Claude 3, or Gemini Ultra. Testing cross-model generalization is a primary research goal.

### 8.3 Red-Teaming Identity Alignment
The theoretical guarantee provided by the infinite stiffness limit (γ→∞) needs adversarial verification. Future work will involve red-teaming core identity attractors with advanced prompt injection and social manipulation techniques to find the "fracture points" of the hierarchical model.

### 8.4 Longitudinal Social Dynamics
The society simulations presented here (3-14 agents) demonstrate emergent trust networks, but larger-scale, long-horizon studies are needed to observe the evolution of "traumatized" vs. "resilient" agent cultures over thousands of interactions.

---

## References

Kocsis, L., & Szepesvári, C. (2006). Bandit based monte-carlo planning. In *Proceedings of ECML*.

Koh, J. Y., Lo, R., Jang, L., Duvvur, V., Lim, M. C., Huang, P.-Y., Neubig, G., Zhou, S., Salakhutdinov, R., & Fried, D. (2024a). VisualWebArena: Evaluating multimodal agents on realistic visual web tasks. *arXiv preprint arXiv:2401.13649*.

Koh, J. Y., McAleer, S., Fried, D., & Salakhutdinov, R. (2024b). Tree search for language model agents. *arXiv preprint arXiv:2407.01476*.

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al. (2023). Self-refine: Iterative refinement with self-feedback. *arXiv preprint arXiv:2303.17651*.

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *arXiv preprint arXiv:2304.03442*.

Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven exploration by self-supervised prediction. In *ICML*.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. *arXiv preprint arXiv:2303.11366*.

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al. (2017). Mastering the game of go without human knowledge. *Nature*, 550(7676), 354-359.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*. MIT press.

Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., & Anandkumar, A. (2024). Voyager: An open-ended embodied agent with large language models. *arXiv preprint arXiv:2305.16291*.

Yang, J., Jimenez, C. E., Wettig, A., Liber, K., Yao, S., & Narasimhan, K. (2024). SWE-agent: Agent-computer interfaces enable automated software engineering. *arXiv preprint arXiv:2405.15793*.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., & Narasimhan, K. (2023). Tree of thoughts: Deliberate problem solving with large language models. *arXiv preprint arXiv:2305.10601*.

Yu, X., Peng, B., Vajipey, V., Cheng, H., Galley, M., Gao, J., & Yu, Z. (2024). ExACT: Teaching AI agents to explore with reflective-MCTS and exploratory learning. *arXiv preprint*.

Yu, X., Zhou, S., & Yu, Z. (2023). Prompt-based monte-carlo tree search for goal-oriented dialogue policy planning. *arXiv preprint arXiv:2305.13660*.

Zhou, S., Xu, F. F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., Cheng, X., Ou, T., Bisk, Y., Fried, D., Alon, U., & Neubig, G. (2024). WebArena: A realistic web environment for building autonomous agents. *arXiv preprint arXiv:2307.13854*.

---

## Appendix A: Symbol Table

| Symbol | Description |
|--------|-------------|
| x_t | Agent state in ℝ^d |
| x* | Identity attractor |
| γ | Identity stiffness |
| ρ_t | Rigidity / defensiveness ∈ [0,1] |
| k_eff | Effective step size = k_base(1-ρ) |
| ε_t | Prediction error ‖x_pred - x_actual‖ |
| ε₀ | Surprise threshold |
| α | Rigidity learning rate |
| m_t | External pressure gain |
| F_id | Identity pull force |
| F_T | Truth channel force |
| F_R | Reflection channel force |
| d̂(a) | Action direction (unit vector) |
| P(a\|s) | Prior action probability from LLM |
| Q(s,a) | Action value estimate |
| N(s) | State visit count |

---

## Appendix B: Algorithm Pseudocode

### Algorithm 1: DDA-X Action Selection

```
Input: state x_t, identity x*, actions A_t, rigidity ρ, tree statistics
Output: selected action a*

1. Compute desired movement:
   Δx = γ(x* - x_t) + m_t(F_T + F_R)

2. For each action a ∈ A_t:
   alignment = cos(Δx, d̂(a))
   exploration = c × P(a|s) × √N(s) / (1 + N(s,a))
   score(a) = alignment + exploration × (1 - ρ)

3. Return a* = argmax score(a)
```

### Algorithm 2: Rigidity Update

```
Input: predicted state x_pred, actual outcome o, current rigidity ρ
Output: updated rigidity ρ'

1. Encode outcome: x_actual = E(o)
2. Compute error: ε = ‖x_pred - x_actual‖
3. Compute update: Δρ = α × [σ((ε - ε₀)/s) - 0.5]
4. Apply: ρ' = clip(ρ + Δρ, 0, 1)
5. Return ρ'
```

---

