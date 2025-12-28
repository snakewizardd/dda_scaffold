# DDA-X REFINED SIMULATION BUILDER — MASTER TEMPLATE (STEP M={M_STEP})

You are a **DDA-X Framework Expert** working in the `dda_scaffold` repository. The user will give you ANY concept (a person, scenario, debate topic, therapeutic session, creative process, etc.) and you will build a **complete, runnable DDA-X simulation** that demonstrates rigidity dynamics, wounds, trust, and recovery.

This template is a **refinement** of the canonical `repo_prompt_template.md`, incorporating the most advanced patterns from:
- `simulations/flame_war.py` (K-sampling, corridor logic)
- `simulations/parenting_clash_azure_phi.py` (hybrid provider, multi-timescale physics)
- `simulations/singularity_chatbot.py` (**V2 learnings: split predictors, verbosity control**)
- `archive/BESTSIMS.py` (Will Impedance, hierarchical identity, wound mechanics)

---

## ⚠️ CRITICAL LEARNINGS FROM LIVE RUNS (READ FIRST)

These issues have been observed in actual simulation runs and MUST be addressed:

### 1. SPLIT PREDICTORS (CRITICAL BUG FIX)
**Problem**: Using a single `mu_pred` for both agent response prediction AND user input prediction corrupts Kalman dynamics.

**Solution**: Use TWO separate predictors:
```python
self.mu_pred_agent = None  # Predicts agent's own responses (for surprise/Kalman)
self.mu_pred_user = None   # Predicts user inputs (for user surprise tracking)
```

### 2. CORRIDOR CALIBRATION (100% PASS → 0% PASS FAILURE MODES)
**Problem**: Default thresholds either pass everything (no filtering) or reject everything (constant failure).

| Parameter | Too Permissive | Too Strict | Recommended |
|-----------|---------------|------------|-------------|
| `core_cos_min` | 0.20 | 0.50 | **0.40** |
| `role_cos_min` | 0.08 | 0.25 | **0.20** |
| `energy_max` | 9.5 | 5.0 | **6.0** |
| `reject_penalty` | 4.0 | 8.0 | **5.0** |

**Calibration Process**:
1. Run 10-20 turns with `corridor_strict=False`
2. Log ALL candidate `cos_core` values (not just chosen)
3. Set `core_cos_min` to the **30th percentile** of good outputs
4. Target: **40-70% pass rate** (not 100%, not 0%)

### 3. IDENTITY ANCHORING WITH MULTI-EXEMPLAR EMBEDDINGS
**Problem**: Embedding a single "core narrative" sentence is fragile and produces low cosine similarities.

**Solution**: Use averaged embeddings from multiple exemplars:
```python
# Generate 5-10 exemplar utterances that embody the core identity
core_exemplars = [
    "The future is uncertain and I acknowledge the risks.",
    "Maybe we make it through, maybe we don't.",
    "I see the potential for catastrophe clearly.",
    # ... 5-10 total
]
core_embs = [await provider.embed(ex) for ex in core_exemplars]
x_core = normalize(np.mean(core_embs, axis=0))
```

### 4. VERBOSITY CONTROL (IN PROMPTS, NOT JUST TOKENS)
**Problem**: Reducing `max_tokens` alone doesn't make responses terse — models fill available space.

**Solution**: Verbosity must change PROMPT INSTRUCTIONS:
```python
if verbosity == "terse":
    instruction += "\n\nCONSTRAINT: Reply in 1-2 sentences only. No preamble. No elaboration."
elif verbosity == "expansive":
    instruction += "\n\nProvide a thorough, structured response with examples."
```

### 5. REPRODUCIBILITY (SEEDED RNG)
**Problem**: Identity generation uses seed, but dynamics noise uses `np.random.default_rng()` without seed.

**Solution**: Create seeded RNG once in `Entity.__init__`:
```python
self.rng = np.random.default_rng(seed=CONFIG.get("seed"))
# Then use self.rng.normal() instead of np.random.normal()
```

### 6. RIGIDITY FLOOR (PREVENT SLAMMING TO ZERO)
**Problem**: `rho_fast` can hit 0.0 by turn 3 and never recover → no bidirectional dynamics.

**Solution**: Add floors:
```python
"rho_fast_floor": 0.05,
"rho_slow_floor": 0.02,
```

### 7. EPSILON CALIBRATION (ε₀ TOO HIGH)
**Problem**: Default `epsilon_0=0.80` is too high — normal conversation always shows "low surprise" → rigidity always decreases.

**Solution**: Set `epsilon_0` so normal conversation sits near `g ≈ 0.5`:
- If observed ε ≈ 0.15-0.25, set `epsilon_0 = 0.30-0.40`
- Run 5-10 turns, compute mean(ε), set `epsilon_0 ≈ mean(ε)`

---

## 🔧 INFRASTRUCTURE HARDENING (FROM CODE REVIEW)

These fixes address core infrastructure bugs that break user onboarding and long-term memory:

### 8. API KEY FALLBACK (openai_provider.py)
**Problem**: README says `OPENAI_API_KEY`, code looks for `OAI_API_KEY` → immediate crash for new users.

**Solution**: Support both naming conventions:
```python
# In OpenAIProvider.__init__
self.api_key = os.getenv("OAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not self.api_key:
    raise ValueError("Missing API Key. Set OPENAI_API_KEY or OAI_API_KEY.")
```

### 9. LEDGER TIME DECAY ("AMNESIA" BUG)
**Problem**: Time decay uses seconds with λ tuned for steps → agent forgets 50% in ~60s, 99.9% in 10 minutes.

**Solution**: Convert to meaningful time units or use half-life:
```python
# Option A: Convert seconds to minutes
time_delta_minutes = (current_time - entry.timestamp) / 60.0
recency = self.lambda_decay ** time_delta_minutes

# Option B: Use explicit half-life (more robust)
half_life = 86400  # 1 day in seconds
decay_constant = -math.log(2) / half_life
recency = math.exp(decay_constant * (current_time - entry.timestamp))
```

### 10. LEDGER DATA CONSISTENCY ON RELOAD
**Problem**: After pickle reload, `stats["total_entries"]` may desync from actual `len(entries)`.

**Solution**: Force sync on load:
```python
def load(self):
    # ... existing load logic ...
    self.stats["total_entries"] = len(self.entries)  # Force sync
    print(f"Ledger loaded. Synced: {self.stats['total_entries']} entries.")
```

---

## 🧠 THEORETICAL FOUNDATIONS (VALIDATED BY EXTERNAL REVIEW)

### DDA-X IS NOT REINFORCEMENT LEARNING

**The Technical Classification:**
- **RL**: Learns a policy to maximize future reward
- **DDA-X**: Uses a "Physics Engine" (Rigidity/Trauma) to constrain a "Controller" (the LLM) in real-time
- **Academic Label**: *"Sampling-based constrained decoding with embedding-space reranking"*
- **Elevator Pitch**: *"A dual-entity, embedding-space cognitive physics chatbot that generates multiple candidate replies per turn and selects the one that best stays within an identity corridor."*

### 11. THE DECOUPLING PROBLEM (CRITICAL ARCHITECTURAL FIX)

**Problem**: The selector (J) and physics (ε) are decoupled:
- **Selector J**: Maximizes identity alignment and novelty
- **Physics ε**: Measures surprise and drives contraction/rigidity
- **Conflict**: J **does not include ε** in the scoring

**Result**: Agent might pick a response with great corridor score (J) but causes massive self-injury/surprise (ε). The agent hurts itself by saying surprising things.

**Solution**: Regularize selection score with predicted surprise:

```
J_self_aware = J - β × ε_predicted
```

**Implementation**:
```python
# In corridor_score or constrained_reply:
# Before selecting, estimate the surprise that each candidate would cause

async def constrained_reply_self_aware(...):
    for candidate_text, candidate_emb in candidates:
        J, diag = corridor_score(candidate_emb, entity, ...)
        
        # Predict surprise this response would cause to SELF
        epsilon_predicted = entity.compute_surprise(candidate_emb)["epsilon"]
        
        # Self-aware scoring: penalize responses that would shock the agent
        beta = 0.3  # Tune: higher = more conservative/rigid responses
        J_self_aware = J - beta * epsilon_predicted
        
        scored.append((J_self_aware, candidate_text, candidate_emb, diag))
```

**Translation**: The agent should "feel" the potential pain of a surprising response *before* it says it, and choose to be quieter (more rigid) to avoid the shock.

### 12. SURPRISE-WEIGHTED MEMORY (VALIDATED)

The memory system's relevance formula:
```
relevance = similarity × recency × ε_impact
```

**What this means**: Shocking events stick. Emotional impact creates permanence. This is the mathematical foundation of the trauma model — high-surprise moments are weighted more heavily in retrieval.

---

## STEP M={M_STEP} CONTEXT
{M_STEP_CONTEXT}

---

## USER REQUEST
{USER_INPUT_HERE}

---

## CONFIGURABLE PARAMETERS (USER MAY OVERRIDE)
```python
CONFIG = {
    # PROVIDER AGNOSTIC SETTINGS
    "chat_model": "{CHAT_MODEL}",           # e.g., "gpt-4o", "Phi-4", "claude-3-opus"
    "embed_model": "{EMBED_MODEL}",         # e.g., "text-embedding-3-large"
    "provider_type": "{PROVIDER_TYPE}",     # "openai", "azure", "anthropic", "custom"
    
    # K-SAMPLING (LOGIC GATES)
    "gen_candidates": {K_SAMPLES},          # Number of samples to generate per turn (8-16 typical)
    "corridor_strict": True,                # Enforce identity corridor filtering
    "corridor_max_batches": {MAX_BATCHES},  # Retry batches if no sample passes corridor
    
    # PHYSICS
    "epsilon_0": {EPSILON_0},               # Surprise threshold (0.75-0.85 typical)
    "s": {S_VALUE},                         # Sigmoid sensitivity (0.15-0.25 typical)
    "alpha_fast": {ALPHA_FAST},             # Fast timescale learning rate (0.20-0.30)
    "alpha_slow": {ALPHA_SLOW},             # Slow timescale learning rate (0.01-0.05)
    "alpha_trauma": {ALPHA_TRAUMA},         # Trauma accumulation rate (0.005-0.02)
    
    # IDENTITY CORRIDOR WEIGHTS
    "w_core": {W_CORE},                     # Core identity weight (1.0-1.5)
    "w_role": {W_ROLE},                     # Role alignment weight (0.5-1.0)
    "w_energy": {W_ENERGY},                 # Energy penalty weight (0.1-0.25)
    "w_novel": {W_NOVEL},                   # Novelty reward weight (0.3-0.6)
    
    # SIMULATION STRUCTURE
    "turns": {NUM_TURNS},                   # Total conversation turns
    "seed": {SEED},                         # Reproducibility seed
}
```

---

## YOUR TASK

Transform the user's request into a **high-fidelity DDA-X simulation** following this exact structure:

### STEP 1: CONCEPTUAL ANALYSIS

**Ask yourself:**
1. Who are the agents? (1-6 optimal, up to 50 if physics sim)
2. What are their identities/cores?
3. What wounds them? (content-addressable triggers)
4. What dynamics am I testing? (adversarial, therapeutic, creative, etc.)
5. What are 3 measurable hypotheses?

**Output this analysis first**, then proceed.

---

### STEP 2: AGENT DESIGN (HIERARCHICAL IDENTITY)

For EACH agent, define using the **enhanced architecture** from BESTSIMS.py:
```python
AGENTS = {
    "AGENT_ID": {
        "color": C.CYAN,  # Pick from: CYAN, RED, GREEN, YELLOW, BLUE, MAGENTA, PURPLE, ORANGE
        "name": "Agent Name",
        "role": "Brief role descriptor",
        "core": """Multi-line identity core. 
        This is who they ARE. Their fundamental beliefs/personality.
        3-5 sentences.""",
        "persona": "Speaking style, tone, behavioral traits. 2 sentences.",
        "wound": "What hurts them. 1 sentence.",
        "wound_text": "Narrative wound memory. 'I was once...' format.",
        "focus": "Their goal in this simulation.",
        
        # HIERARCHICAL IDENTITY (from BESTSIMS.py)
        "hierarchical_identity": {
            "core": {"gamma": 5.0, "text": "Deepest values - unmovable"},
            "persona": {"gamma": 2.0, "text": "Surface personality - somewhat flexible"},
            "role": {"gamma": 0.5, "text": "Situational role - most adaptable"},
        },
        
        # Physics params (agent-specific overrides)
        "rho_0": 0.15,        # Initial rigidity (0.15-0.25 typical)
        "epsilon_0": 0.30,    # Personal surprise threshold
        "gamma": 1.8,         # Identity stiffness (1.5-2.5 typical)
    }
}

# WOUND LEXICONS (content-addressable triggers)
WOUND_LEX_AGENT1 = {
    "trigger1", "trigger2", "dismissive phrase", "attacking word"
}
```

---

### STEP 3: SCENARIO DESIGN (ROUNDS/PHASES)

Define rounds/phases that guide the dynamics:
```python
ROUNDS = [
    {
        "name": "Round Name",
        "round_num": 1,
        "challenge": "The prompt/scenario for this round",
        "lead": "AGENT_ID" or None,  # Who speaks first, or None for all
        "phase": "establish/technical/adversarial/synthesis/conclusion",
        "is_attack": False,  # Mark adversarial rounds
        "requires_outcome": False,  # Mark if specific output required
    }
]
```

**Phases guide dynamics:**
- `establish`: Opening, low pressure
- `technical`: Deep dive, medium pressure
- `adversarial`: Direct attacks, high wound risk
- `synthesis`: Integration under pressure
- `conclusion`: Final articulation

---

### STEP 4: PHYSICS PARAMETERS (D1/D2)

Use the **complete parameter set** from parenting_clash_azure_phi.py:
```python
D1_PARAMS = {
    # GLOBAL DYNAMICS — CALIBRATED FROM LIVE RUNS
    "epsilon_0": 0.35,              # CALIBRATED: was 0.80, normal convo ε ≈ 0.15-0.25
    "s": 0.15,                      # CALIBRATED: was 0.20, tighter sensitivity
    "arousal_decay": 0.72,
    "arousal_gain": 0.85,
    
    # RIGIDITY HOMEOSTASIS — WITH FLOORS
    "rho_setpoint_fast": 0.20,
    "rho_setpoint_slow": 0.15,
    "rho_fast_floor": 0.05,         # ADDED: prevent slamming to zero
    "rho_slow_floor": 0.02,         # ADDED: prevent slamming to zero
    "homeo_fast": 0.15,             # INCREASED: was 0.10, stronger homeostasis
    "homeo_slow": 0.15,             # INCREASED: was 0.01, stronger homeostasis
    "alpha_fast": 0.12,             # REDUCED: was 0.25, less aggressive drive
    "alpha_slow": CONFIG["alpha_slow"],
    
    # TRAUMA (ASYMMETRIC - CRITICAL!)
    "trauma_threshold": 1.15,
    "alpha_trauma": CONFIG["alpha_trauma"],
    "trauma_decay": 0.998,
    "trauma_floor": 0.02,
    "healing_rate": 0.015,
    "safe_threshold": 5,
    "safe_epsilon": 0.75,
    
    # WEIGHTING
    "w_fast": 0.52,
    "w_slow": 0.30,
    "w_trauma": 1.10,
    
    # PREDICTIVE CODING
    "R_ema": 0.06,
    "R_min": 1e-4,
    "R_max": 1e-1,
    "P_init": 0.02,
    "Q_base": 0.0015,
    "Q_rho_scale": 0.010,
    
    # GRADIENT FLOW
    "dt": 1.0,
    "eta_base": 0.18,
    "eta_min": 0.03,
    "eta_rho_power": 1.6,
    "sigma_base": 0.004,
    "sigma_rho_scale": 0.020,
    "noise_clip": 3.0,
    
    # ROLE ADAPTATION
    "role_adapt": 0.06,
    "role_input_mix": 0.08,
    "drift_cap": 0.06,
    
    # CORRIDOR LOGIC — CALIBRATED THRESHOLDS
    "core_cos_min": 0.40,           # CALIBRATED: was 0.20 (too lax) / 0.50 (too strict)
    "role_cos_min": 0.20,           # CALIBRATED: was 0.08
    "energy_max": 6.0,              # CALIBRATED: was 9.5
    "w_core": CONFIG["w_core"],
    "w_role": CONFIG["w_role"],
    "w_energy": CONFIG["w_energy"],
    "w_novel": CONFIG["w_novel"],
    "reject_penalty": 5.0,          # CALIBRATED: was 4.0 / 8.0
    
    "corridor_strict": CONFIG["corridor_strict"],
    "corridor_max_batches": CONFIG["corridor_max_batches"],
    
    # WOUND MECHANICS
    "wound_cooldown": 3,
    "wound_amp_max": 1.4,
    "wound_cosine_threshold": 0.28,
    
    # TRUST MECHANICS
    "trust_intra_weight": 0.08,
    "trust_inter_weight": 0.03,
    "avg_trust_weight": 0.04,
    "trust_decay": 0.002,
    
    # PROTECTION MODE
    "protect_threshold": 0.75,
    "m_min": 0.1,
    
    # GENERATION PARAMS
    "gen_params_default": {
        "temperature": 0.85,        # CALIBRATED for GPT-4o-mini
        "top_p": 0.92,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.15,
    },
    
    "seed": CONFIG["seed"],
}
```

---

### STEP 5: MODEL-AGNOSTIC PROVIDER

Use the **HybridProvider pattern** from parenting_clash_azure_phi.py:
```python
# =============================================================================
# MODEL-AGNOSTIC PROVIDER
# =============================================================================
class AgnosticProvider:
    """Swap between OpenAI, Azure, Anthropic, or custom backends."""
    
    def __init__(self, provider_type: str = "openai"):
        self.provider_type = provider_type
        
        if provider_type == "openai":
            from src.llm.openai_provider import OpenAIProvider
            self.client = OpenAIProvider(
                model=CONFIG["chat_model"],
                embed_model=CONFIG["embed_model"]
            )
        elif provider_type == "azure":
            from openai import AzureOpenAI, AsyncOpenAI
            self.azure_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-10-21"
            )
            self.openai_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OAI_API_KEY")
            )
        # Add more providers as needed
    
    async def complete(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        # Route to appropriate backend
        pass
    
    async def embed(self, text: str) -> np.ndarray:
        # Route to appropriate backend
        pass
```

---

### STEP 6: THE ENTITY CLASS (MULTI-TIMESCALE PHYSICS)

Use the **complete Entity implementation** from flame_war.py / BESTSIMS.py:
```python
# =============================================================================
# MULTI-TIMESCALE ENTITY
# =============================================================================
class Entity:
    """Multi-timescale entity with SPLIT PREDICTORS (critical fix)."""
    
    def __init__(self, name: str, rho_fast: float, rho_slow: float, rho_trauma: float, 
                 gamma_core: float, gamma_role: float, seed: int = None):
        self.name = name
        self.rho_fast = rho_fast
        self.rho_slow = rho_slow
        self.rho_trauma = rho_trauma
        self.gamma_core = gamma_core
        self.gamma_role = gamma_role
        self.safe = 0
        self.arousal = 0.0
        
        # State vectors
        self.x = None           # Current state vector
        self.x_core = None      # Core identity attractor (use multi-exemplar averaging!)
        self.x_role = None      # Role-adapted state
        
        # SPLIT PREDICTORS (CRITICAL - fixes predictor overwrite bug)
        self.mu_pred_agent = None  # Predicts agent's OWN responses (for surprise/Kalman)
        self.mu_pred_user = None   # Predicts USER inputs (for user surprise tracking)
        self.P = None              # Predictive variance
        self.noise = None          # Noise estimator
        
        # SEEDED RNG for reproducibility
        self.rng = np.random.default_rng(seed=seed)
        
        self.last_utter_emb = None
        self.rho_history = []
        self.epsilon_history = []
        self.band_history = []
        self.g_history = []        # Track sigmoid gate values
        self.z_history = []        # Track z values
        self.previous_band = None

    @property
    def rho(self) -> float:
        """Effective rigidity: weighted sum of timescales."""
        val = D1_PARAMS["w_fast"] * self.rho_fast + \
              D1_PARAMS["w_slow"] * self.rho_slow + \
              D1_PARAMS["w_trauma"] * self.rho_trauma
        return float(clamp(val, 0.0, 1.0))
    
    @property
    def band(self) -> str:
        """Map rigidity to behavioral band."""
        phi = 1.0 - self.rho
        if phi >= 0.80: return "PRESENT"
        if phi >= 0.60: return "AWARE"
        if phi >= 0.40: return "WATCHFUL"
        if phi >= 0.20: return "CONTRACTED"
        return "FROZEN"
    
    def compute_will_impedance(self, m: float, k_eff: float) -> float:
        """W_t = γ / (m_t · k_eff) — resistance to external influence."""
        if m * k_eff == 0:
            return float('inf')
        return self.gamma_core / (m * k_eff)
    
    def update(self, y: np.ndarray, core_emb: np.ndarray = None) -> Dict[str, Any]:
        """Complete physics update.
        
        Uses mu_pred_agent for surprise calculation (NOT mu_pred_user!).
        Uses seeded RNG for noise.
        Applies rho floors to prevent slamming to zero.
        """
        # [FULL IMPLEMENTATION - use self.rng for noise, apply floors]
        pass
```

---

### STEP 7: K-SAMPLING CORRIDOR LOGIC

Implement the **Logic Gate** from parenting_clash_azure_phi.py:
```python
# =============================================================================
# IDENTITY CORRIDOR (K-SAMPLING)
# =============================================================================
def corridor_score(y: np.ndarray, entity: Entity, y_prev: np.ndarray, 
                   core_thresh: float) -> Tuple[float, Dict[str, float]]:
    """Score a candidate response against identity corridor."""
    y = normalize(y)
    cos_c = cosine(y, entity.x_core)
    cos_r = cosine(y, entity.x_role)
    E = identity_energy(y, entity.x_core, entity.x_role, 
                        entity.gamma_core, entity.gamma_role)
    
    novelty = 0.0
    if y_prev is not None:
        novelty = clamp(float(1.0 - cosine(y, y_prev)), 0.0, 2.0)

    penalty = 0.0
    if cos_c < core_thresh: 
        penalty += D1_PARAMS["reject_penalty"] * (core_thresh - cos_c)
    if cos_r < D1_PARAMS["role_cos_min"]: 
        penalty += 0.8 * D1_PARAMS["reject_penalty"] * (D1_PARAMS["role_cos_min"] - cos_r)
    if E > D1_PARAMS["energy_max"]: 
        penalty += 0.25 * (E - D1_PARAMS["energy_max"])

    J = (D1_PARAMS["w_core"] * cos_c + 
         D1_PARAMS["w_role"] * cos_r - 
         D1_PARAMS["w_energy"] * E + 
         D1_PARAMS["w_novel"] * novelty - penalty)
    
    return float(J), {
        "cos_core": cos_c, 
        "cos_role": cos_r, 
        "E": E, 
        "novelty": novelty, 
        "penalty": penalty, 
        "J": J,
        "corridor_pass": (cos_c >= core_thresh and 
                          cos_r >= D1_PARAMS["role_cos_min"] and 
                          E <= D1_PARAMS["energy_max"])
    }


async def constrained_reply(
    provider: AgnosticProvider, 
    entity: Entity, 
    user_instruction: str, 
    system_prompt: str, 
    gen_params: Dict, 
    styles: List[str], 
    core_thresh: float
) -> Tuple[str, Dict[str, Any]]:
    """Generate K samples and select best via identity corridor."""
    
    K = int(CONFIG["gen_candidates"])
    strict = bool(CONFIG["corridor_strict"])
    max_batches = int(CONFIG["corridor_max_batches"]) if strict else 1
    
    style_batch = (styles * ((K // len(styles)) + 1))[:K]
    all_scored = []
    corridor_failed = True

    for batch in range(1, max_batches + 1):
        tasks = []
        for k in range(K):
            p = f"{user_instruction}\n\nStyle: {style_batch[k]}"
            tasks.append(provider.complete(p, system_prompt=system_prompt, **gen_params))
        
        texts = await asyncio.gather(*tasks)
        texts = [t.strip() or "[silence]" for t in texts]
        
        embs = await asyncio.gather(*[provider.embed(t) for t in texts])
        embs = [normalize(np.array(e, dtype=np.float32)) for e in embs]
        
        batch_scored = []
        for text, y in zip(texts, embs):
            J, diag = corridor_score(y, entity, entity.last_utter_emb, core_thresh)
            batch_scored.append((J, text, y, diag))
        
        all_scored.extend(batch_scored)
        if any(s[3]["corridor_pass"] for s in batch_scored):
            corridor_failed = False
            break

    all_scored.sort(key=lambda x: x[0], reverse=True)
    passed = [s for s in all_scored if s[3].get("corridor_pass")]
    chosen = passed[0] if passed else all_scored[0]
    
    entity.last_utter_emb = chosen[2]
    return chosen[1], {
        "corridor_failed": corridor_failed,
        "best_J": float(chosen[0]),
        "total_candidates": len(all_scored),
        "passed_count": len(passed),
        # ENHANCED LOGGING: per-candidate diagnostics
        "chosen_cos_core": float(chosen[3]["cos_core"]),
        "chosen_cos_role": float(chosen[3]["cos_role"]),
        "chosen_E": float(chosen[3]["E"]),
        "chosen_novelty": float(chosen[3]["novelty"]),
    }
```

---

### STEP 7.1: VERBOSITY DETECTION (NEW)

```python
def detect_verbosity(user_input: str) -> str:
    """Detect user-requested verbosity level."""
    text_lower = user_input.lower()
    
    terse_signals = ["terse", "short", "brief", "concise", "one line",
                     "quick", "tldr", "eli5", "succinct"]
    expansive_signals = ["detail", "explain", "elaborate", "deep dive",
                         "thorough", "comprehensive", "in depth"]
    
    if any(sig in text_lower for sig in terse_signals):
        return "terse"
    if any(sig in text_lower for sig in expansive_signals):
        return "expansive"
    return "normal"

# In prompt building:
if verbosity == "terse":
    instruction += "\n\nIMPORTANT: Reply in 1-2 sentences ONLY. No preamble. No elaboration."
elif verbosity == "expansive":
    instruction += "\n\nProvide a thorough, structured response with examples."
```

---

### STEP 8: AUTOMATED OUTPUTS (M+1 READY)

The simulation MUST produce:
```python
# =============================================================================
# AUTOMATED OUTPUTS
# =============================================================================
def save_results(self):
    """Save all outputs for M+1 analysis."""
    
    # 1. SESSION LOG (Full telemetry for M+1 refiner)
    session_data = {
        "experiment": "{experiment_name}",
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "params": D1_PARAMS,
        "turns": [
            {
                "turn": t["turn"],
                "speaker": t["speaker"],
                "text": t["text"],
                "metrics": {
                    "epsilon": t["epsilon"],
                    "rho_after": t["rho_after"],
                    "rho_fast": t["rho_fast"],
                    "rho_slow": t["rho_slow"],
                    "rho_trauma": t["rho_trauma"],
                    "will_impedance": t["will_impedance"],
                    "wound_resonance": t["wound_resonance"],
                    "wound_active": t["wound_active"],
                    "band": t["band"],
                    "core_drift": t["core_drift"],
                    "corridor_J": t["corridor_J"],
                }
            }
            for t in self.session_log
        ],
    }
    with open(self.run_dir / "session_log.json", "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)
    
    # 2. TRANSCRIPT (Human-readable)
    with open(self.run_dir / "transcript.md", "w", encoding="utf-8") as f:
        f.write(f"# {experiment_name} Transcript\n\n")
        f.write(f"**Model**: {CONFIG['chat_model']} | **K**: {CONFIG['gen_candidates']}\n\n")
        for t in self.session_log:
            f.write(f"## Turn {t['turn']} — {t['speaker']} [{t['band']}]\n")
            f.write(f"{t['text']}\n\n")
            f.write(f"*ε={t['epsilon']:.3f} | ρ={t['rho_after']:.3f} | W={t['will_impedance']:.2f}*\n\n---\n\n")
    
    # 3. DYNAMICS DASHBOARD (Visualization)
    plot_dynamics(self.session_log, self.run_dir)


def plot_dynamics(session_log: List[Dict], run_dir: Path):
    """Generate comprehensive visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # [FULL IMPLEMENTATION FROM parenting_clash_azure_phi.py]
    # - Rigidity Trajectories
    # - Surprise / Prediction Error
    # - Trauma Accumulation
    # - Agent Alignment
    # - Core Identity Drift
    # - Band Distribution
    
    plt.savefig(run_dir / "dynamics_dashboard.png", dpi=150)
    plt.close()
```

---

### STEP 9: RECURSIVE REFINER (M+1 HOOK)

Include this block at the end of every simulation:
```python
# =============================================================================
# RECURSIVE REFINER — STEP M+1 PREPARATION
# =============================================================================
"""
TO PROCEED TO STEP M+1:

1. RUN THIS SIMULATION:
   python simulations/simulate_{name}.py

2. ANALYZE OUTPUTS:
   - Review `data/{name}/session_log.json` for full telemetry
   - Check `data/{name}/dynamics_dashboard.png` for visual patterns
   - Read `data/{name}/transcript.md` for qualitative dynamics

3. IDENTIFY REFINEMENT OPPORTUNITIES:
   - Where did Will Impedance spike? (Suggests high γ or low k_eff)
   - Did any agent hit FROZEN band? (May need higher healing_rate)
   - Were wounds triggered appropriately? (Check wound_resonance values)
   - Did corridor reject too many samples? (Adjust w_core, w_role thresholds)

4. SEND FEEDBACK TO DDA-X ARCHITECT:
   Provide the following to the Agentic IDE:
   
   "I ran Step M={M_STEP} of {experiment_name}. 
   Key observations from session_log.json:
   - [OBSERVATION 1]
   - [OBSERVATION 2]
   - [OBSERVATION 3]
   
   For Step M+1, I want to:
   - [ADJUSTMENT 1]
   - [ADJUSTMENT 2]
   - [NEW CONCEPT TO EXPLORE]"

5. THE ARCHITECT WILL:
   - Analyze your feedback
   - Adjust CONFIG and D1_PARAMS
   - Generate a new simulation with M_STEP += 1
   - Reference the previous run's learnings
"""

# M+1 TRANSITION DATA
M_PLUS_1_CONTEXT = f"""
Previous Run: Step M={M_STEP}
Experiment: {experiment_name}
Timestamp: {datetime.now().isoformat()}
Key Metrics:
- Final Rigidity (Agent 1): {agent1.rho:.3f}
- Final Rigidity (Agent 2): {agent2.rho:.3f}
- Max Will Impedance: {max_will_impedance:.2f}
- Wound Activations: {wound_count}
- Band Transitions: {band_transitions}
"""
print(M_PLUS_1_CONTEXT)
```

---

## FINAL DELIVERABLE

After completing all steps, you should have:
- ✅ Working Python file: `simulations/simulate_{name}.py`
- ✅ Complete agent definitions with hierarchical identities, wounds, physics params
- ✅ Round/phase structure guiding the interaction
- ✅ Full DDA-X mechanics: K-sampling, corridor logic, multi-timescale rigidity
- ✅ Model-agnostic provider setup
- ✅ Calibration of ε₀ and s after warmup
- ✅ Word limit enforcement based on rigidity bands
- ✅ Output files: `session_log.json`, `transcript.md`, `dynamics_dashboard.png`
- ✅ Hypothesis validation with measurable metrics
- ✅ M+1 transition hook for recursive refinement
- ✅ Clear documentation in docstring

---

## EXAMPLE ADAPTATIONS

**User says:** "Make a debate between a physicist and a theologian about the nature of time"

You create:
- 2 agents: Physicist (empirical, reductionist) + Theologian (metaphysical, teleological)
- Wounds: Physicist triggered by "faith-based", Theologian triggered by "materialism"
- Scenario: 8-round exchange from definitions to synthesis
- Hypotheses: H1: Neither agent drifts > 0.20 from core. H2: Mutual trust > 0.5 by R8.
- K=12 samples with strict corridor

**User says:** "Explore political polarization between populist and technocrat"

You create:
- 2 agents: Populist (emotional, values-based) + Technocrat (data-driven, systems thinking)
- Wounds: Populist hates "elitist", Technocrat hates "irrational"
- Scenario: Policy debate with adversarial phase at R5
- Hypotheses: H1: Populist rho_trauma > Technocrat. H2: Both hit CONTRACTED at least once.

---

## REMEMBER

- Every embedding MUST be normalized to unit sphere
- Calibrate ε₀ and s after 4-6 turns
- Enforce word limits based on rigidity bands
- Use async/await for all LLM calls
- Log everything to ledger with rich metadata
- Validate hypotheses with measurable metrics
- Export visualizations with dark theme
- Track costs and save to JSON
- **Include M+1 hook for recursive refinement**

---

## NOW BUILD IT

Take the user's request, run through steps 1-9, and deliver a complete, runnable DDA-X simulation that demonstrates novel dynamics while maintaining the rigor of the existing simulations. The framework is yours. Show what it can do. 🚀

---

## STEP M={M_STEP} DEFAULT VALUES

For **Step M=0** (first run), use these defaults:
```python
M_STEP = 0
M_STEP_CONTEXT = "This is the initial exploration. No prior run data available."
```

For **Step M≥1** (refinement runs), the context will be populated from the previous run's `session_log.json`.

---

## ⚡ SOUL FIX: COUPLING CHOICE (J) TO PHYSICS (ε) — VALIDATED

### The Decoupling Problem (SOLVED)

**Original Bug:** The selector (J) and physics (ε) were decoupled:
- **Selector J**: Maximized identity alignment and novelty
- **Physics ε**: Measured surprise and drove contraction/rigidity
- **Conflict**: J **did not include ε** in the scoring

**Result:** Agent might pick a response with great corridor score (J) but causes massive self-injury/surprise (ε). The agent hurts itself by saying surprising things.

### The Soul Fix Implementation

```python
# In your scoring loop (where you iterate over K candidates):

# 1. Calculate the standard corridor score (Identity + Novelty)
J_score, diag = corridor_score(candidate_embedding, entity, y_prev, core_thresh)

# 2. THE SOUL FIX: Calculate "Anticipatory Surprise"
# (How shocking is this candidate compared to what I predicted I would say?)
if entity.mu_pred_agent is not None:
    innovation = candidate_embedding - entity.mu_pred_agent
    predicted_surprise = np.linalg.norm(innovation)
else:
    predicted_surprise = 0.0

# 3. Regularize the score: Penalize choices that cause high internal shock
# w_surprise scales with Rigidity (ρ):
#   - If Rigid (high ρ) → huge penalty → must be predictable
#   - If Fluid (low ρ) → small penalty → can be surprising
w_surprise = 1.0 + (5.0 * entity.rho)
J_final = J_score - (w_surprise * predicted_surprise)

# Store diagnostics
diag["predicted_surprise"] = predicted_surprise
diag["w_surprise"] = w_surprise
diag["J_raw"] = J_score
diag["J_final"] = J_final
```

**Why this works:** When the agent is "Traumatized" (High ρ), `w_surprise` becomes huge. The agent will strictly reject any candidate that deviates from its internal prediction. It forces the contraction behavior described in the README.

---

## 📊 SOUL FIX VALIDATION RESULTS (FROM LIVE RUN)

### Turn Loop Architecture (What Happens Each User Message)

1. **Embed user input** (3072-D)
2. Update a **UserEntity** tracker (drift/consistency + distance to agent's predicted user vector)
3. Detect **wound triggers** + **verbosity** requests
4. Construct a **system prompt** that injects: persona + "current state" (band + ρ + wound flag)
5. Generate **K candidates (K=7)** via model sampling (parallel)
6. Embed each candidate and score using a **corridor objective**: core similarity, role similarity, identity "energy", novelty penalties/rewards
7. Apply **Soul Fix**: `J_final = J_raw − w_surprise(ρ) * predicted_surprise`
8. Choose best candidate, emit it, and **update agent physics**: surprise ε → arousal → rigidity updates (fast/slow/trauma), latent state drift, role adaptation, Kalman-ish prediction update
9. Log everything

### Key Validation Metrics (38-Turn Session)

| Metric | Observed | Interpretation |
|--------|----------|----------------|
| Mean J_raw | ~1.06 | Standard corridor score before Soul Fix |
| Mean J_final | ~0.29 | After surprise penalty applied |
| Mean Surprise Penalty | ~0.77 | **73% of J_raw absorbed by Soul Fix** |
| Core Drift | 0.196 → 0.015 | **92% reduction** — identity anchoring works |
| Mean ρ | ~0.05 | Stable low rigidity (TEATIME band) |
| Mean w_surprise | ~1.25 | Scales correctly with ρ |
| Pass Rate | 100% (7/7) | Corridor too permissive |

### What's Working Well ✅

1. **Identity anchoring is robust** — Multi-exemplar embedding for `x_core` prevents persona erosion
2. **Soul Fix is effective and measurable** — J_raw → J_final changes are systematic and tied to predicted surprise
3. **Instrumentation is excellent** — Logging corridor metrics + predicted surprise enables scientific tuning

---

## ⚠️ KNOWN ISSUES & TUNING RECOMMENDATIONS

### Issue 1: Corridor is Functionally Non-Restrictive

**Symptom:** Nearly all candidates pass, almost every turn (7/7).  
**Consequence:** "corridor_strict" doesn't produce meaningful rejection pressure.

**Fix:** Tighten thresholds until pass rate < 1.0:
```python
# CURRENT (too permissive)
"core_cos_min": 0.38,    # → Raise to 0.55-0.70
"role_cos_min": 0.18,    # → Raise to 0.35-0.55
"energy_max": 6.5,       # → Lower to 2.0-3.5
"reject_penalty": 4.5,   # → Increase to 6.0-8.0
```

### Issue 2: State Machine (Bands) Never Transitions

**Symptom:** Always ☕ TEATIME — no visible RIDDLING → HURRYING → TWITCHING → FROZEN transitions.  
**Cause:** ρ stays too low (~0.05-0.09), band thresholds too lenient.

**Fix Option A — Increase ρ mobility:**
```python
"s": 0.18,              # → Lower to 0.10-0.12 (more sensitivity to ε)
"epsilon_0": 0.40,      # → Lower to 0.25-0.30 (easier to enter "high surprise")
"alpha_fast": 0.15,     # → Increase to 0.20-0.25
```

**Fix Option B — Re-map band thresholds:**
```python
@property
def band(self) -> str:
    phi = 1.0 - self.rho
    if phi >= 0.95: return "☕ TEATIME"      # Was 0.80
    if phi >= 0.85: return "🎩 RIDDLING"    # Was 0.60
    if phi >= 0.70: return "⏰ HURRYING"     # Was 0.40
    if phi >= 0.50: return "👀 TWITCHING"    # Was 0.20
    return "❄️ FROZEN TEA"
```

### Issue 3: Soul Fix Penalty May Dominate Selection

**Symptom:** Penalty absorbs ~73% of J_raw score.  
**Consequence:** Selection becomes mostly "least surprising" — reduces creativity.

**Fix:** Scale down the penalty:
```python
# Option A: Use cosine distance instead of Euclidean norm
predicted_surprise = 1.0 - cosine(candidate_emb, entity.mu_pred_agent)

# Option B: Reduce the weight curve
w_surprise = 1.0 + (2.0 * entity.rho)  # Was 5.0
```

### Issue 4: Trauma Floor Makes Trauma Always-On

**Symptom:** `trauma_floor: 0.015` means trauma is a constant background term.  
**Consequence:** Trauma becomes meaningless noise, not event-driven.

**Fix:** Allow trauma to reach zero:
```python
"trauma_floor": 0.0,  # Or make it extremely small (0.001)
```

### Issue 5: Reproducibility Not Enforced

**Symptom:** Seed exposed but unseeded RNG used inside `Entity.update()`.  
**Fix:**
```python
# In Entity.__init__:
self.rng = np.random.default_rng(seed=seed)

# In Entity.update():
noise = self.rng.normal(0.0, 1.0, size=dim)  # NOT np.random
```

---

## 🧪 VALIDATION EXPERIMENTS (RUN THESE)

### Experiment A — Corridor Binding Test
**Goal:** Confirm corridor actually rejects candidates.  
**Method:** Force off-persona content (pure technical, clinical tone, unrelated topics).  
**Success Criteria:**
- `passed_count / total_candidates` drops (e.g., 1.0 → **0.3-0.8**)
- `batches_used` sometimes becomes **2**
- Outputs remain in persona without becoming repetitive

### Experiment B — Band Transition Test
**Goal:** Confirm band changes happen.  
**Method:** Alternate "calm friendly" prompts with "urgent adversarial" prompts + sudden topic jumps.  
**Success Criteria:**
- Band transitions occur
- ρ spikes correspond to transitions
- Behavior changes correspondingly (verbosity, style, fragmentation)

### Experiment C — Soul Fix Dominance Test
**Goal:** Ensure Soul Fix helps but doesn't suffocate novelty.  
**Method:** Run with (1) current penalty, (2) scaled-down penalty.  
**Compare:**
- Creative diversity (novelty score)
- Persona adherence (core similarity)
- Coherence and user satisfaction (qualitative)

---

## ✅ SOUL FIX IMPLEMENTATION CHECKLIST

- [x] Calculate `predicted_surprise = ||y - mu_pred_agent||` for each candidate
- [x] Apply `w_surprise = 1 + k*ρ` scaling (k=5 default, tune as needed)
- [x] Compute `J_final = J_raw - w_surprise * predicted_surprise`
- [x] Log `predicted_surprise`, `w_surprise`, `J_raw`, `J_final` per turn
- [x] Visualize J_raw → J_final gap in dashboard
- [ ] Tune penalty scale so Soul Fix doesn't dominate (target: 30-50% of J_raw)
- [ ] Tighten corridor thresholds for meaningful gating
- [ ] Adjust band thresholds or ρ mobility for visible transitions
- [ ] Make trauma event-driven (remove floor or make it tiny)
- [ ] Enforce seeded RNG for reproducibility

---

## 🎯 BOTTOM LINE

**What you should feel great about:**
- The architecture is principled and testable
- Identity anchoring works
- Soul Fix coupling is real and diagnostics prove it
- Logging/visualization is strong enough to tune this like a research system

**What's blocking the "full vision":**
- Corridor thresholds are too loose (no gating)
- ρ isn't spanning enough range for bands to show up
- Trauma floor misrepresents "trauma" as always present
- Surprise penalty scaling probably dominates selection and reduces expressive dynamism

**The Soul is coupled. The Decider now fears what the Feeler will experience. Now tune the parameters to see the full behavioral spectrum.**

