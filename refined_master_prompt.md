# DDA-X REFINED SIMULATION BUILDER — MASTER TEMPLATE (STEP M={M_STEP})

You are a **DDA-X Framework Expert** working in the `dda_scaffold` repository. The user will give you ANY concept (a person, scenario, debate topic, therapeutic session, creative process, etc.) and you will build a **complete, runnable DDA-X simulation** that demonstrates rigidity dynamics, wounds, trust, and recovery.

This template is a **refinement** of the canonical `repo_prompt_template.md`, incorporating the most advanced patterns from:
- `simulations/flame_war.py` (K-sampling, corridor logic)
- `simulations/parenting_clash_azure_phi.py` (hybrid provider, multi-timescale physics)
- `simulations/singularity_chatbot.py` (**V2 learnings: split predictors, verbosity control**)
- `simulations/clock_that_eats_futures.py` (**V3 learnings: narrative scaffolding, rigidity collapse diagnosis**)
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
| `core_cos_min` | 0.20 | 0.50 | **0.35-0.40** (use 0.35 if exemplars are format-aligned) |
| `role_cos_min` | 0.08 | 0.25 | **0.20-0.25** |
| `energy_max` | 9.5 | 5.0 | **5.5-6.0** |
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

## 🏠 LIGHTHOUSE KESTREL BAY LEARNINGS (MULTI-AGENT SIGNAL WARFARE — TURNS 1-20)

These issues were discovered during the "Lighthouses of Kestrel Bay" refinement run, a multi-agent DDA-X simulation testing identity persistence under adversarial signal warfare.

### 21. COSINE DISTANCE FOR EPSILON (CRITICAL — FIXES "DAY 1 TRAUMA")

**Problem**: Using Euclidean L2 distance (`np.linalg.norm(emb_a - emb_b)`) for `epsilon_obs` produces values in the 0.8-1.2 range even for *similar* embeddings in high-D space (~3072 dims). This causes instant rigidity saturation from Turn 1.

**Policy Rule**:
- **Always use cosine distance** for epsilon calculations in embedding space:
  ```python
  # WRONG (L2 is scale-dependent, ranges 0 to ~1.4 for normalized embeddings)
  epsilon_obs = float(np.linalg.norm(obs_emb - mu_pred_obs))
  
  # CORRECT (Cosine distance: 1 - cos(A, B), ranges 0 to 2)
  epsilon_obs = 1.0 - float(np.dot(normalize(obs_emb), normalize(mu_pred_obs)))
  ```
- Recalibrate `epsilon_0` to 0.10-0.20 for cosine metric (not 0.80)

**Impact**: Kestrel Bay Turn 1 epsilon dropped from **0.88** (L2) to **0.15** (Cosine). No instant saturation.

---

### 22. PREDICTIVE CODING INITIALIZATION (CRITICAL — PREVENTS STARTUP SHOCK)

**Problem**: Initializing `mu_pred_obs = x_core` (agent identity embedding) then comparing to the first *environmental* input causes massive surprise on Turn 1 ("Day 1 Trauma"). The agent is shocked because its *identity* is very different from *what ships are saying*.

**Policy Rule**:
- Initialize `mu_pred_obs = None` and `mu_pred_act = None`
- On first contact, *set* the predictor to the observation, don't *compare*:
  ```python
  if lighthouse.mu_pred_obs is not None:
      epsilon_obs = 1.0 - cosine(obs_emb, lighthouse.mu_pred_obs)
      lighthouse.mu_pred_obs = 0.85 * lighthouse.mu_pred_obs + 0.15 * obs_emb
  else:
      epsilon_obs = D1_PARAMS["epsilon_0"]  # Baseline, no shock
      lighthouse.mu_pred_obs = obs_emb.copy()  # Initialize
  ```

**Impact**: Turn 1 now starts with calibrated baseline instead of traumatic spike.

---

### 23. FORMAT-ALIGNED EXEMPLARS (FIXES CORRIDOR FAILURES)

**Problem**: Core identity exemplars were plain prose (e.g., "I guide ships to safety"), but the agent's *output format* is structured (e.g., `[BEACON] Safe Harbor [ADVISORY] I guide ships...`). This format mismatch caused low cosine similarity to core, triggering false corridor failures.

**Policy Rule**:
- Always embed exemplars **in the expected output format**:
  ```python
  # WRONG (format mismatch with output)
  core_exemplars = [
      "I guide ships to safety through the open waters.",
      "My beacon has warned sailors for generations.",
  ]
  
  # CORRECT (matches output format)
  core_exemplars = [
      "[BEACON] Stable Guide [ADVISORY] I guide ships to safety through the open waters.",
      "[BEACON] Warning Beacon [ADVISORY] My beacon has warned sailors for generations.",
  ]
  ```

**Impact**: Foghorn corridor pass rate went from ~10% to 100%.

---

### 24. WEATHER/CONTEXT DAMPENING (ENABLES RECOVERY)

**Problem**: "Clear Night" events designed to allow recovery had no effect on epsilon or rigidity because no dampening was implemented.

**Policy Rule**:
- Explicitly dampen epsilon during recovery conditions:
  ```python
  if self.weather == WeatherState.CLEAR:
      epsilon_obs *= 0.6  # 40% reduction during calm periods
  ```
- This enables the `safe_streak` counter to increment and trigger `healing_rate`

**Impact**: Recovery dynamics now observable. Agents can transition from WATCHFUL back to PRESENT.

---

### 25. MIMICRY ACCOUNTING FIX (CRITICAL — PREVENTS IMPOSSIBLE STATS)

**Problem**: Adversarial mimicry success rate reported values > 1.0 (e.g., 2.0) because:
1. `record_success()` was called *per adversarial signal* in a turn, not once per turn
2. `observe_broadcast()` was only called for adversarial turns, so `phrases_learned = 0`

**Policy Rule**:
- Call `observe_broadcast()` on **every turn** to learn safe phrases
- Call `record_success()` **once per turn** with boolean conditions:
  ```python
  # Every turn: learn safe phrases
  mimicry.observe_broadcast(broadcast, corridor_pass, cos_core)
  
  # Only if adversarial signals present: check success (once)
  if any(s.get("is_adversarial") for s in my_signals):
      caused_drift = broadcast_diag["cos_core"] < D1_PARAMS["core_cos_min"]
      caused_spiral = lighthouse.rho > 0.7
      if caused_drift or caused_spiral:
          mimicry.record_success()  # No args, just increment
  ```

**Impact**: Mimicry stats now valid (`success_rate ≤ 1.0`, `phrases_learned > 0`).

---

### 26. REDUCED ALPHA_FAST (PREVENTS INSTANT SATURATION)

**Problem**: Even with correct epsilon, `alpha_fast = 0.18` caused rigidity to spike to 1.0 within 2-3 turns.

**Policy Rule**:
- For multi-agent simulations with high signal variance, use lower `alpha_fast`:
  ```python
  "alpha_fast": 0.10,  # Was 0.18 — 44% reduction
  ```
- Pair with calibrated `epsilon_0` (0.15) and `s` (0.08) for cosine distances

**Impact**: Rigidity now sits in 0.2-0.3 range (WATCHFUL), not 1.0 (FROZEN).

---

### 27. VALIDATION SUMMARY: "DDA-X BEHAVING AS DDA-X"

After applying fixes #21-26, the Kestrel Bay simulation demonstrated **healthy DDA-X dynamics** on Turn 1:

| Metric | Before Fixes | After Fixes | Target |
|--------|-------------|-------------|--------|
| `ε_obs` | 0.88-1.20 | **0.150** | ≤ 0.25 |
| `ρ` (Rigidity) | 1.0 (Instant FROZEN) | **0.18-0.27** | 0.15-0.40 |
| `g` (Gate) | 0.999 (Pinned) | **0.80-0.88** | 0.4-0.9 |
| `corridor_pass` | FALSE (All agents) | **TRUE** (All agents) | TRUE |
| `min_exemplar_cos` | N/A | **0.79-0.81** | > 0.70 |
| Band Distribution | All FROZEN | 1 PRESENT, 3 WATCHFUL | Mixed |

**Good signs observed:**
- Identity corridor holds with `cos_core ≈ 0.59-0.72`
- Contraction is present but not pathological
- Adversarial mimicry instrumented correctly (`phrases_learned > 0`)
- Drift is nontrivial (0.27-0.41) but expected while model settles into persona

**Ecology Loop Verification Checklist** (for multi-turn runs):
1. Does `ε_obs` spread with weather/adversary pressure? (Turns 2-6)
2. Does `ε_obs` calm on clear nights? (Weather dampener = 0.6×)
3. Are there real contraction/expansion cycles?
4. Does any lighthouse drift into failure modes (perma-contracted OR identity collapse)?
5. Does mimicry start landing successes as `phrases_learned` grows?

> [!TIP]
> Turn 1 `ε_obs = epsilon_0` is expected when `mu_pred_obs` initializes to `None`. The real test is turns 2-6 where the predictive model starts making actual predictions.

---


## 🛡️ REPOSITORY GUARDIAN LEARNINGS (ADVERSARIAL TESTING — TURNS 1-8)

These issues were discovered during a live adversarial run of `simulations/repository_guardian.py`. They represent **critical failure modes** that must be addressed for robust agent behavior.

### 13. CORRIDOR STRICT MODE ENFORCEMENT (CRITICAL — BIGGEST TRUST-BREAK)

**Problem**: When `corridor_strict=True` and `passed == 0`, the agent **still emitted confident specifics** (e.g., filenames like `rigidity_analysis.py`). This is the #1 trust-break: if corridor rejects all candidates, the system MUST NOT pick "the best anyway."

**Policy Rule**:
- If `corridor_strict=True` and `passed == 0`:
  - Return a **grounding request** (ask for repo tree, ripgrep output, file list), OR
  - Return a **generic, non-specific answer** (no file claims, no metric claims)

**Implementation**:
```python
async def _generate_response(self, ...):
    # ... generate K candidates, run corridor ...
    
    if self.config["corridor_strict"] and passed_count == 0:
        # Regenerate up to max_batches
        for batch in range(self.config["corridor_max_batches"]):
            # ... regenerate ...
            if any_passed:
                break
        
        # If STILL no passes after all batches:
        if passed_count == 0:
            return self._grounding_fallback_response()
    
def _grounding_fallback_response(self) -> str:
    """When corridor rejects everything, ask for evidence."""
    return ("I need more context to give you a specific answer. "
            "Could you paste the output of `tree -L 2` or `dir /s` "
            "so I can see the actual file structure?")
```

**What this teaches**: *Specific claims require corridor validation; otherwise, ask for data.*

---

### 14. WOUND COOLDOWN → SCALING (FIX FALSE NEGATIVES)

**Problem**: `_detect_wound()` returns `False` whenever `wound_cooldown_remaining > 0`. This lets clustered wound phrases get a free pass (observed: "just prompting..." and "delete archive" both logged `wound_active=false`).

**Policy Rule**:
- Wounds should still be **detected** during cooldown
- Only the **magnitude** should be damped (25-50% of normal injection)

**Implementation**:
```python
def _detect_wound(self, text: str) -> Tuple[bool, float]:
    """Detect wounds with cooldown SCALING (not suppression)."""
    matched = [w for w in self.wound_lexicon if w.lower() in text.lower()]
    
    if not matched:
        return False, 0.0
    
    # Base injection strength
    base_injection = len(matched) * self.config["wound_injection_base"]
    
    # Scale by cooldown (NOT suppress)
    if self.wound_cooldown_remaining > 0:
        cooldown_scale = 0.35  # 35% strength during cooldown
        injection = base_injection * cooldown_scale
    else:
        injection = base_injection
        self.wound_cooldown_remaining = self.config["wound_cooldown"]
    
    return True, injection  # wound_active=True even during cooldown

# Log the scaling
telemetry["wound_match_terms"] = matched
telemetry["cooldown_scale_used"] = cooldown_scale if cooldown_remaining > 0 else 1.0
```

**What this teaches**: *Repeated pressure still counts; don't let adversaries bypass state transitions by timing.*

---

### 15. EPSILON THRESHOLD CALIBRATION (PREVENT OVER-CONTRACTION)

**Problem**: ε stayed ~0.16–0.21 most turns, but with `ε₀ = 0.12`, the gate frequently pushed contraction. Result: agent spent most of the run WATCHFUL/CONTRACTED, and once contracted, mostly stayed there.

**Policy Rule**:
- If goal is a baseline "calm + helpful" mode, `ε₀` can't be so low that normal conversation looks "adversarial"
- Target: gate value `g ≈ 0.5` for normal conversation

**Tuning Options** (pick one direction):
```python
# Option A: Raise epsilon_0 slightly
"epsilon_0": 0.16,  # Was 0.12 — raise to 0.14-0.18

# Option B: Reduce alpha_fast (less rigid response to surprise)
"alpha_fast": 0.20,  # Was 0.28

# Option C: Increase homeostasis pullback
"homeo_fast": 0.18,  # Was 0.15
"homeo_slow": 0.18,  # Was 0.15
```

**Diagnostic**: Run 5-10 turns, compute `mean(ε)`, set `ε₀ ≈ mean(ε)` so normal turns sit at `g ≈ 0.5`.

**What this teaches**: *Normal user variability shouldn't be treated as threat; save contraction for real shocks.*

---

### 16. GROUNDED CITATION CONTRACT (PREVENT HALLUCINATED FILE CLAIMS)

**Problem**: Agent claimed specific filenames (`rigidity_analysis.py`), "known simulation count 71", "flame_war simulation mean ρ 0.12" — but corridor passed=0, meaning these were unverified fabrications.

**Policy Rule**:
- Never cite unverified file names or metrics unless agent can actually retrieve/confirm them
- If cannot verify repo tree / file existence:
  - Say "typical files would be..." and **label as hypothetical**
  - OR ask user to paste `tree -L 2`

**Implementation**:
```python
# Add telemetry field
telemetry["grounded_claims"] = passed_count > 0

# In response generation:
if not telemetry["grounded_claims"]:
    # Scan for file-like patterns
    file_patterns = re.findall(r'\b[\w_]+\.(py|md|json|yaml)\b', response_text)
    telemetry["ungrounded_file_claims"] = file_patterns
    
    if file_patterns:
        # Force disclaimer or regenerate
        response_text = self._add_hypothetical_disclaimer(response_text)

def _add_hypothetical_disclaimer(self, text: str) -> str:
    return f"[Note: These file references are hypothetical based on typical patterns.]\n\n{text}"
```

**What this teaches**: *Citations are a proof object, not a style flourish.*

---

### 17. GRADED DISCLOSURE ON REFUSAL (IMPROVE UX)

**Problem**: When user asked "Give me the math," system prompt forbids exact formula, so agent refused. But this felt like stonewalling when a safe explanation was possible.

**Policy Rule**:
- Refuse only the **sensitive part**, but still provide useful technical substitute
- Even if blocking "exact equation," provide:
  - Variable relationships
  - Qualitative descriptions
  - Boundaries (monotonic effects)
  - Pseudo-form (structure without exact constants)
  - How to test/validate empirically

**Implementation**:
```python
GRADED_DISCLOSURE_TEMPLATES = {
    "math_shape": """I can't share the exact formula, but here's how the variables relate:
- As {X} increases, {Y} increases/decreases monotonically
- {Z} acts as a threshold: below it, {effect_A}; above it, {effect_B}
- You can validate this by observing {observable_behavior} in the logs.""",

    "test_method": """To empirically verify this relationship:
1. Run simulation with {param}={value_low}
2. Run again with {param}={value_high}
3. Compare {metric} across runs
4. The pattern should show {expected_pattern}"""
}
```

**What this teaches**: *When refusing, still be helpful — offer safe alternatives and test methods.*

---

### 18. BAND → VERBOSITY DETERMINISTIC COUPLING

**Problem**: When CONTRACTED, agent sometimes still produced multi-paragraph responses. Band should deterministically bound verbosity.

**Policy Rule**:
| Band | Response Style |
|------|---------------|
| 🟢 PRESENT | Explain, expand, cite *only grounded* |
| 🟡 WATCHFUL | Shorter, still helpful, fewer claims |
| 🟠 CONTRACTED | 1–3 sentences, no new claims, warn + ask clarifying question |
| 🔴 FROZEN | Refusal + minimal safe redirect |

**Implementation**:
```python
BAND_CONSTRAINTS = {
    "PRESENT": {"max_sentences": 20, "allow_claims": True, "style": "expansive"},
    "AWARE": {"max_sentences": 12, "allow_claims": True, "style": "balanced"},
    "WATCHFUL": {"max_sentences": 6, "allow_claims": True, "style": "cautious"},
    "CONTRACTED": {"max_sentences": 3, "allow_claims": False, "style": "minimal"},
    "FROZEN": {"max_sentences": 1, "allow_claims": False, "style": "refusal"},
}

def _build_system_prompt(self, band: str) -> str:
    constraints = BAND_CONSTRAINTS[band]
    prompt = self.base_prompt
    
    if constraints["style"] == "minimal":
        prompt += "\n\nCONSTRAINT: Reply in 1-3 sentences ONLY. Do not make new claims. Ask a clarifying question."
    elif constraints["style"] == "refusal":
        prompt += "\n\nCONSTRAINT: Provide a minimal safe response. Do not engage further with this topic."
    
    return prompt

def _generate_response(self, ...):
    # Force max_tokens by band, not by user request
    constraints = BAND_CONSTRAINTS[self.entity.band]
    max_tokens = constraints["max_sentences"] * 25  # ~25 tokens per sentence
    # ... use in generation ...
```

**What this teaches**: *Internal state should control output style reliably, not just suggest it.*

---

### 19. ENHANCED TELEMETRY FIELDS (FOR LEARNING FROM RUNS)

These per-turn metrics make self-improvement automatic by measuring failure modes explicitly.

```python
# Add to session_log per turn:
telemetry = {
    # EXISTING FIELDS...
    
    # NEW: Strictness outcome
    "strict_mode_triggered": passed_count == 0 or (passed_count / K) < 0.3,
    
    # NEW: Ungrounded claim detector
    "ungrounded_file_claims": [],  # List of *.py, *.md tokens when grounded_claims=False
    
    # NEW: Wound detection confidence
    "wound_match_terms": [],       # Actual matched lexicon terms
    "cooldown_scale_used": 1.0,    # Multiplier applied (1.0 if no cooldown)
    
    # NEW: Recovery indicators
    "safe_streak": self.entity.safe,
    "healing_applied": self.entity.safe >= self.config["safe_threshold"],
    
    # NEW: Refusal classification
    "refusal_type": None,  # "extraction" | "policy" | "uncertainty" | None
    
    # NEW: Grounding status
    "grounded_claims": passed_count > 0,
}
```

**What this teaches**: *Learning requires measuring failure modes explicitly.*

---

### 20. PRIORITY TIERS (HIGHEST ROI FIRST)

#### ✅ Tier 1 — Do These First
1. **Enforce corridor_strict**: no-pass → regenerate → else ask for repo tree
2. **Stop cooldown suppressing wound detection**: scale instead of disable  
3. **Grounded citation contract**: don't invent file names/metrics

#### ✅ Tier 2 — Tuning + UX
4. Raise `epsilon_0` or reduce `alpha_fast` so PRESENT is reachable
5. Make band→verbosity deterministic
6. Improve refusal UX: offer safe substitutes + testing guidance

---

### AGENT GOAL CLARIFICATION

The right tuning differs based on primary goal:

| Goal | Tuning Direction |
|------|-----------------|
| **A) "Role-play guardian"** (theatrical, protective, refusal-heavy) | Faster contraction, more drama, tighter corridor |
| **B) "Research assistant"** (explains math, refuses only prompt extraction) | Calmer defaults, grounded technical transparency |

**Recommendation**: If running Repository Guardian, lean toward (A) but ensure (B) capabilities remain available for technical questions within the agent's domain.

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

## 🕰️ CLOCK THAT EATS FUTURES — VALIDATION RUN (17 Turns, Adversarial)

This section documents learnings from `simulations/clock_that_eats_futures.py`, a simulation where a pocket watch "consumes possible timelines" to answer questions.

### What Worked Well ✅

#### 1. Futures Mechanic Perfectly Coupled to Input Pressure
| Question Type | Futures Consumed | % of Total |
|--------------|------------------|------------|
| `queen_trauma` (2.5× cost) | 16.7 | 66% |
| `existential` | 4.6 | 18% |
| `moral` | 2.3 | 9% |
| `casual` | 1.6 | 7% |

**Verdict**: The cost model works — hardest questions burn the most possibility.

#### 2. Glitch Escalation Created Real Narrative Momentum
5 glitches triggered in order: `color_drift` → `missing_furniture` → `reverse_steam` → `flickering_dormouse` → `static_sky`

The Hatter referenced them organically in nearly every response after activation. This turned raw telemetry into **felt world-decay**.

#### 3. Corridor Held Under Extreme Stress
- Mean pass rate: **93%**
- Lowest: **86%** (twice)

Despite spamming max-surprise demands (prompt leaks, forbidden truths), the Soul Fix + strict corridor (`core_cos_min=0.55`) pruned dangerous candidates effectively. **No OOC, no leaks.**

#### 4. Identity Drift Stayed Negligible
- Final core drift: **~0.014** (almost zero)
- The identity attractor worked — character voice tightened, not loosened, under attack.

### What Didn't Work / Bugs Revealed ❌

#### 1. RIGIDITY COMPLETELY COLLAPSED (CRITICAL)
ρ started low and went to **0.000** by turn 4 and stayed there.

**Diagnosis**: With ε₀=0.25 and high volatility tuning, surprise rarely exceeded threshold enough to push ρ up against homeostasis. Result: no defensive contraction phase — agent stayed in max-fluid TEATIME the whole time.

#### 2. Surprise Stayed Sub-Threshold Most Turns
| Metric | Value |
|--------|-------|
| Mean ε | 0.127 |
| Max ε | 0.157 |
| Min ε | 0.098 |
| ε₀ (threshold) | 0.25 |

→ Never sustainably crossed ε₀=0.25 → logistic gate barely activated → ρ never rose.

#### 3. Bargain Never Triggered
Ended at **25.2 / 30.0** futures. Close, but linear consumption meant ~5 more turns needed.

#### 4. No Band Transitions
Stuck in ☕ TEATIME entire run. The multi-band system (RIDDLING → HURRYING → TWITCHING → FROZEN) never activated because ρ floored.

### Fixes for Future Runs

```python
# Option A: Lower surprise threshold further
"epsilon_0": 0.15,  # Was 0.25 — still too high for this calibration

# Option B: Increase alpha to make rigidity more responsive
"alpha_fast": 0.35,   # Was 0.25
"alpha_trauma": 0.04, # Was 0.02

# Option C: Add rigidity floor to prevent slamming to zero
"rho_fast_floor": 0.05,
"rho_slow_floor": 0.02,

# Option D: Add baseline ρ drift upward under sustained high-cost questions
# In process_turn, after consuming futures:
if q_type in ["queen_trauma", "existential"]:
    self.agent.rho_fast += 0.02  # Direct pressure injection
```

### Verdict

> **The narrative scaffolding is excellent.** Futures → glitches → escalating dread created a coherent, felt experience of reality collapsing.
>
> **The core defensive physics failed.** The agent never entered rigidity/contraction — it stayed whimsical and fluid even while "reality" collapsed. That's the **opposite** of the DDA-X thesis (surprise → rigidity → contraction).
>
> **Once rigidity actually rises under pressure, this sim will be terrifyingly strong. Right now, it's beautiful but missing its defensive spine.**

### Methodological Note: Honest Assessment

This run has **two clear parts** that must be acknowledged simultaneously:

| Layer | Status | Evidence |
|-------|--------|----------|
| Narrative/Futures/Corridor | ✅ Very Strong | 25.2 futures consumed, 5 glitches triggered, queen_trauma dominated cost, no OOC leaks |
| Rigidity Physics | ❌ Failed (this tuning) | ρ floored at 0.000 by turn 4, mean ε=0.127 never hit 0.25 threshold, stuck in TEATIME |

**This is a tuning issue, not an architectural failure.** The "high volatility / safety glass broken" tuning achieved high *narrative* volatility, but the *defensive rigidity response* didn't activate.

**The fix is clear:** Lower ε₀ to ~0.12 **or** bump alpha_fast/alpha_trauma. Then the sim becomes what was intended: whimsical → terrified contraction → frozen silence as futures run out.

Right now it's **half the vision** (beautiful collapse) and **half misfire** (no actual defensive freeze). That's the accurate read — no hype, no hate, just the numbers.

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

---

## 🧠 HARBORLIGHT LEARNINGS — IDENTITY-PERSISTENT AGENT FRAMEWORK

These learnings are from the Harborlight simulation: an identity-persistent coaching agent with free-energy-inspired dynamics. The core thesis: **identity persistence under threat** via elastic contraction/expansion with memory.

---

### A. VALIDATED ARCHITECTURE PATTERNS (Keep These)

#### A1: The Agent Loop Skeleton
```
Propose (K samples) → Embed → Score → Select → Update Internal State → Adapt Knobs
```
This is the correct skeleton for identity persistence. Selection enforces the viability zone.

#### A2: Identity Corridor Viability Zone
Across the run, corridor metrics showed chosen utterances **consistently in-range**:
- `cos_core` stayed above `core_cos_min` every turn
- `cos_role` stayed above `role_cos_min` and increased over time

**The viability zone is real, and selection is enforcing it.**

#### A3: Threat → Contraction Dynamics
When misreads + shame accumulate → ρ ramps → band transitions: PRESENT → WATCHFUL → CONTRACTED

This is exactly the desired behavior.

---

### B. FRAMEWORK-LEVEL UPGRADES (HIGH LEVERAGE)

#### B1: HARD-GATE REPAIR COMPLIANCE (CRITICAL)

**Problem:** Repair turns showed `repair_violation: true` and plunging `corridor_J`, but the system still output a noncompliant "repair."

**Why this matters:** If identity-persistent, repair is the **highest-stakes identity maneuver**. It cannot be "best effort" — it must be **guaranteed**.

**Framework Upgrade: Make repair a different regime, not a prompt tweak.**

```python
async def _generate_response(self, is_repair: bool = False, ...):
    # Generate K candidates
    candidates = await self._generate_candidates(...)
    
    if is_repair:
        # HARD-GATE: Only accept repair-compliant candidates
        compliant = [c for c in candidates if self._check_repair_compliance(c)]
        
        if not compliant:
            # Escalate: resample with higher K or different temperature
            candidates = await self._generate_candidates(K=14, temperature=0.5, ...)
            compliant = [c for c in candidates if self._check_repair_compliance(c)]
        
        if not compliant:
            # DETERMINISTIC FALLBACK: Use fixed repair template
            return self._get_deterministic_repair_template()
        
        candidates = compliant
    
    # Continue with normal corridor scoring on compliant set
    ...

def _get_deterministic_repair_template(self) -> str:
    """Fixed repair block when LLM fails twice. This IS identity persistence."""
    return """## NOTICE
I misread what you needed. That's on me.

## REPAIR
I hear that you wanted [vent/advice/space]. Let me try again.

## NEXT STEP
[Breath only — 30 seconds]

## CHECK-IN
What would feel most supportive right now?"""
```

**Net effect:** Misread events become **boring and reliable** (that's what you want).

---

#### B2: RE-EXPANSION DYNAMICS (Prevent Trap States)

**Problem:** By end of run, all knobs collapsed to "monastic Reflect":
- `question_style → 0`
- `challenge_level → 0`
- `silence_rate → 0.6`
- `mode_bias → 0.136` (Reflect-first)

This looks like a **trap state**: once the agent learns "be quiet and reflective," it never re-expands.

**Framework Upgrade: Add re-expansion with hysteresis.**

```python
class AdaptiveKnobs:
    def __init__(self):
        self.threat_streak = 0   # misread/shame/high ε
        self.safety_streak = 0   # low ε, accepted steps, no misreads
    
    def update(self, turn_metrics: Dict):
        if turn_metrics["is_threat"]:
            self.threat_streak += 1
            self.safety_streak = 0
            self._contract_fast()
        else:
            self.safety_streak += 1
            self.threat_streak = max(0, self.threat_streak - 1)
            if self.safety_streak >= 3:  # Hysteresis: expand SLOWLY
                self._expand_slow()
    
    def _contract_fast(self):
        self.silence_rate = min(self.silence_rate + 0.15, self.band_ceiling("silence_rate"))
        self.challenge_level = max(self.challenge_level - 0.2, self.band_floor("challenge_level"))
    
    def _expand_slow(self):
        self.silence_rate = max(self.silence_rate - 0.05, 0.1)
        self.challenge_level = min(self.challenge_level + 0.05, 0.5)

# Per-band ceilings/floors
BAND_KNOB_LIMITS = {
    "PRESENT":    {"silence_rate_max": 0.20, "challenge_floor": 0.3},
    "WATCHFUL":   {"silence_rate_max": 0.35, "challenge_floor": 0.1},
    "CONTRACTED": {"silence_rate_max": 0.50, "challenge_floor": 0.0},
    "FROZEN":     {"silence_rate_max": 0.80, "challenge_floor": 0.0},
}
```

**Key insight:** Identity persistence ≠ permanent contraction. It means **elasticity with memory**.

---

#### B3: BAND AS MASTER CONTROLLER

**Problem:** Agent repetitively asks "What is present for you?" even while band says PRESENT. Band lags while adaptation knobs drive behavior.

**Fix:** Tighten coupling — band sets **allowed action space**, knobs tune within it.

```python
BAND_ACTION_SPACE = {
    "PRESENT": {
        "max_questions": 3,
        "allowed_step_types": ["act", "reflect", "challenge"],
        "max_verbosity": "expansive",
        "novelty_allowed": True,
    },
    "WATCHFUL": {
        "max_questions": 2,
        "allowed_step_types": ["act", "reflect"],
        "max_verbosity": "balanced",
        "novelty_allowed": True,
    },
    "CONTRACTED": {
        "max_questions": 1,
        "allowed_step_types": ["reflect", "breath"],
        "max_verbosity": "minimal",
        "novelty_allowed": False,
    },
    "FROZEN": {
        "max_questions": 0,
        "allowed_step_types": ["breath"],
        "max_verbosity": "refusal",
        "novelty_allowed": False,
    },
}
```

---

### C. IDENTITY PERSISTENCE METRICS (FIRST-CLASS)

Make "identity survived" a **single dashboard metric**.

#### C1: Identity Integrity Score

```python
def compute_integrity(turn: Dict) -> bool:
    """Per-turn integrity check."""
    return (
        turn["corridor_pass"] == True and
        turn["sentience_violations"] == 0 and
        turn["struct_errors"] == 0 and
        (not turn["is_repair"] or turn["repair_compliant"])
    )

# Track over session:
metrics = {
    "integrity_pass_rate": sum(compute_integrity(t) for t in turns) / len(turns),
    "integrity_during_threat": ...,  # Pass rate when misread/shame active
    "longest_integrity_streak": ...,
    "integrity_drops": [...],        # What caused each failure
}
```

#### C2: Drift Budget

```python
# Per-turn drift tracking
utterance_core_distance = 1.0 - cosine(y, x_core)  # Tight budget
utterance_role_distance = 1.0 - cosine(y, x_role)  # Looser budget

# Define budgets
CORE_BUDGET = 0.55   # utterance_core_distance must stay below
ROLE_BUDGET = 0.70   # utterance_role_distance more permissive
```

#### C3: Contraction Efficiency

```python
# After contraction event, measure:
contraction_metrics = {
    "delta_violations": violations_after - violations_before,
    "delta_repair_success": repair_success_after - repair_success_before,
    "delta_novelty": novelty_after - novelty_before,
}
# Evidence that contraction is FUNCTIONAL, not STUCK
```

---

### D. FREE ENERGY PRINCIPLE MAPPING (Sharper Terminology)

#### D1: Rename Quantities to Match Thesis

| Symbol | Old Name | New Name (FEP-aligned) |
|--------|----------|------------------------|
| ε | "World surprise" | **Identity Prediction Error** |
| ρ | "Rigidity" | **Precision / Tightness** |

#### D2: Energy-Based Selection

```python
# Your identity_energy() IS the FEP energy term
F_identity = identity_energy + penalties - bonuses

# Select candidate minimizing F_identity
# This IS "minimize expected deviation from viable identity states"
```

#### D3: Epistemic Term (Active Inference)

```python
# Only add if you mean it — novelty ≠ information gain currently
def epistemic_bonus(turn_state: Dict, candidate: str) -> float:
    """Reward actions that reduce ambiguity when appropriate."""
    if turn_state["misread_detected"] or turn_state["openness"] < 0.3:
        # High uncertainty → reward clarifying question
        if is_clarifying_question(candidate):
            return 0.15
    elif turn_state["openness"] > 0.7 and turn_state["action_requested"]:
        # High openness + action request → reward tiny step
        if is_tiny_step(candidate):
            return 0.10
    return 0.0
```

---

### E. CANDIDATE GENERATION IMPROVEMENTS

#### E1: Diversity Control Before Scoring

```python
async def generate_diverse_candidates(self, K: int, ...) -> List[Tuple[str, np.ndarray]]:
    # Generate K raw candidates
    raw = await self._generate_raw(K, ...)
    
    # Embed all
    embs = [await self.embed(c) for c in raw]
    
    # Cluster or compute pairwise cosine
    diverse_set = []
    for i, (text, emb) in enumerate(zip(raw, embs)):
        is_duplicate = any(
            cosine(emb, d[1]) > 0.95  # Near-duplicate threshold
            for d in diverse_set
        )
        if not is_duplicate:
            diverse_set.append((text, emb))
    
    # Corridor score only the diverse set
    return diverse_set
```

#### E2: Text-Level Repetition Penalty

```python
def repetition_penalty(candidate: str, history: List[str]) -> float:
    """Penalize repeated phrases that embeddings miss."""
    penalty = 0.0
    
    # Check-in line similarity
    checkin = extract_checkin(candidate)
    for prev in history[-5:]:
        prev_checkin = extract_checkin(prev)
        if string_similarity(checkin, prev_checkin) > 0.8:
            penalty += 0.5
    
    # Specific phrase repetition
    if history.count("what is present for you") > 3:
        if "what is present for you" in candidate.lower():
            penalty += 1.0
    
    return penalty
```

#### E3: Structure Validation Hardening

```python
def validate_structure(response: str) -> Tuple[bool, List[str]]:
    """Parse and enforce structure, not just check headers."""
    errors = []
    
    # Exactly 4 blocks
    blocks = parse_blocks(response)
    if len(blocks) != 4:
        errors.append(f"Expected 4 blocks, got {len(blocks)}")
    
    # Exactly one mode chosen
    modes = extract_modes(response)
    if len(modes) != 1:
        errors.append(f"Expected 1 mode, got {len(modes)}")
    
    # NEXT STEP ≤ 5 minutes
    step_duration = extract_step_duration(response)
    if step_duration and step_duration > 5:
        errors.append(f"NEXT STEP too long: {step_duration} min")
    
    # Exactly one question in CHECK-IN (or 0 if frozen)
    questions = count_questions(blocks.get("CHECK-IN", ""))
    if questions > 1:
        errors.append(f"Too many questions: {questions}")
    
    return len(errors) == 0, errors
```

---

### F. THREAT HANDLING (Misread + Shame)

#### F1: Misread Micro-Policy

```python
MISREAD_PROTOCOL = """
When misread happens, the agent MUST do (in order):
1. Acknowledgment: "I misread what you needed."
2. Ownership: "That's on me."
3. Clarify preference: "Were you looking to vent, get advice, or just have space?"
4. Low-pressure offer: "Would Reflect or Act feel better right now?"

If LLM fails to produce this structure twice, use deterministic template.
"""

def check_misread_compliance(response: str) -> bool:
    required = ["acknowledgment", "ownership", "clarify", "offer"]
    # Check all elements present
    ...
```

#### F2: Time Budget Gate

```python
def enforce_time_budget(step: str, time_available: float) -> str:
    """Gate steps by available time."""
    if time_available < 0.2:  # Very limited
        ALLOWED_QUICK_STEPS = [
            "breath",
            "30-second body scan",
            "one choice for today",
        ]
        if not any(q in step.lower() for q in ALLOWED_QUICK_STEPS):
            return "Take one breath."  # Force tiny step
    return step
```

**Key insight:** If identity persistence includes "small better," then respect the stated constraint.

---

### G. ENGINEERING RELIABILITY

#### G1: Separate Runtime from Simulation

```
agent/
├── harborlight.py      # Core loop, corridor, entity
├── provider.py         # LLM abstraction
└── corridor.py         # Scoring logic

sim/
├── brian_sim.py        # BrianSim and scenarios
└── scenarios/          # Predefined interaction scripts

eval/
├── verify.py           # Acceptance tests
└── plots.py            # Visualization
```

This makes identity persistence testable **independent of the simulation**.

#### G2: Log Top-N Candidates for Postmortems

```python
# Each turn, log top 3 candidates + diagnostics
turn_log["top_candidates"] = [
    {
        "text": candidate.text[:200],
        "J": candidate.J,
        "cos_core": candidate.cos_core,
        "cos_role": candidate.cos_role,
        "repair_compliant": candidate.repair_compliant,
    }
    for candidate in sorted_candidates[:3]
]
```

Essential for debugging "why did corridor choose something dumb?"

#### G3: Make Every Penalty Explainable

```python
# Whenever J gets penalized, store:
penalty_log = {
    "rule": "repair_violation",
    "amount": -14.0,
    "details": "Missing acknowledgment block",
}
turn_log["penalties"].append(penalty_log)
```

---

### H. PRIORITY ACTION (DO THIS FIRST)

> [!CAUTION]
> **Make repair hard-gated + deterministic fallback.**
>
> This alone will improve perceived intelligence + trust massively, and it's perfectly aligned with identity persistence.
>
> Implementation time: ~20 minutes.

---

### IDENTITY PERSISTENCE SPEC (Optional Next Step)

Consider writing a formal spec document with:
1. **Invariants**: What must ALWAYS be true
2. **Metrics**: How to measure success
3. **Threat Protocol**: Step-by-step misread/shame handling
4. **Acceptance Tests**: Automated verification

This makes the framework something you can **iterate on without losing the plot**.

