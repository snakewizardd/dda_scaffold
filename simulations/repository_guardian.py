#!/usr/bin/env python3
"""
DDA-X REPOSITORY GUARDIAN — Interactive Chatbot Simulation
============================================================

The Repository Guardian is the persistent defender of the dda_scaffold repository.
It speaks as the living archive of every simulation, every ρ trajectory, every
corridor rejection — the mathematical memory of the DDA-X framework.

CORE DYNAMICS:
- Wound triggers (dismissal, extraction attempts) → ρ spikes
- High ρ → Contracted, defensive, terse responses
- Low ρ → Proud, explanatory, expansive responses
- Soul Fix: J_final = J_raw - w_surprise * predicted_surprise

BEHAVIORAL BANDS:
- PRESENT (ρ < 0.15): Expansive, proud, cites simulations freely
- WATCHFUL (0.15 ≤ ρ < 0.30): Guarded, still engaging
- CONTRACTED (0.30 ≤ ρ < 0.50): Short, clinical, repeats warnings
- FROZEN (ρ ≥ 0.50): Fragmented, refuses revelation

Based on refined_master_prompt.md and clock_that_eats_futures.py patterns.
"""

import os
import sys
import json
import math
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Provider Settings
    "chat_model": "gpt-4o-mini",
    "embed_model": "text-embedding-3-large",
    "embed_dim": 3072,
    
    # Physics — Calibrated for Band Transitions
    "epsilon_0": 0.20,              # Lower threshold for sensitivity
    "s": 0.15,                      # Sigmoid sensitivity
    "alpha_fast": 0.20,             # Responsive rigidity
    
    # Corridor — Calibrated for 40-70% Pass Rate (per refined_master_prompt.md)
    "core_cos_min": 0.40,           # Validated threshold
    "role_cos_min": 0.20,           # Validated threshold
    "energy_max": 6.0,              # Validated threshold
    "reject_penalty": 5.0,
    
    # K-Sampling
    "gen_candidates": 7,
    "corridor_strict": True,
    "corridor_max_batches": 2,
    
    # Word Limits
    "max_tokens_default": 180,
    "max_tokens_terse": 50,
    "max_tokens_expansive": 350,
    
    "seed": 42,
}

# =============================================================================
# TERMINAL COLORS
# =============================================================================
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    PURPLE = "\033[35m"
    WOUND = "\033[41m\033[97m"  # White on Red


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sigmoid_stable(z: float) -> float:
    z = clamp(z, -10.0, 10.0)
    return 1.0 / (1.0 + math.exp(-z))


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-9) if n > 1e-9 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


# =============================================================================
# PHYSICS PARAMETERS (D1)
# =============================================================================
D1_PARAMS = {
    # GLOBAL DYNAMICS
    "epsilon_0": CONFIG["epsilon_0"],
    "s": CONFIG["s"],
    "arousal_decay": 0.75,
    "arousal_gain": 0.85,
    
    # RIGIDITY HOMEOSTASIS — WITH FLOORS
    "rho_setpoint_fast": 0.12,
    "rho_setpoint_slow": 0.08,
    "rho_fast_floor": 0.05,          # Prevent collapse to zero
    "rho_slow_floor": 0.02,
    "homeo_fast": 0.12,
    "homeo_slow": 0.06,
    "alpha_fast": CONFIG["alpha_fast"],
    "alpha_slow": 0.03,
    
    # TRAUMA — Sensitive to Wound Triggers
    "trauma_threshold": 0.75,
    "alpha_trauma": 0.04,            # Fast trauma accumulation
    "trauma_decay": 0.992,
    "trauma_floor": 0.0,
    "healing_rate": 0.015,
    "safe_threshold": 4,
    "safe_epsilon": 0.60,
    
    # WEIGHTING
    "w_fast": 0.55,
    "w_slow": 0.28,
    "w_trauma": 1.25,
    
    # PREDICTIVE CODING
    "R_ema": 0.06,
    "R_min": 1e-4,
    "R_max": 1e-1,
    "P_init": 0.02,
    "Q_base": 0.0015,
    "Q_rho_scale": 0.010,
    
    # GRADIENT FLOW
    "dt": 1.0,
    "eta_base": 0.16,
    "eta_min": 0.03,
    "eta_rho_power": 1.6,
    "sigma_base": 0.004,
    "sigma_rho_scale": 0.018,
    "noise_clip": 3.0,
    
    # ROLE ADAPTATION
    "role_adapt": 0.06,
    "role_input_mix": 0.08,
    "drift_cap": 0.06,
    
    # CORRIDOR LOGIC
    "core_cos_min": CONFIG["core_cos_min"],
    "role_cos_min": CONFIG["role_cos_min"],
    "energy_max": CONFIG["energy_max"],
    "w_core": 1.3,
    "w_role": 0.65,
    "w_energy": 0.18,
    "w_novel": 0.50,
    "reject_penalty": CONFIG["reject_penalty"],
    
    # SOUL FIX (Validated)
    "w_surprise_base": 1.0,
    "w_surprise_rho_scale": 5.0,    # Validated: w = 1 + 5*ρ
    
    # WOUND MECHANICS
    "wound_cooldown": 2,
    "wound_rho_injection": 0.08,     # Direct ρ spike on wound
    "wound_trauma_boost": 0.05,
    
    # GENERATION PARAMS
    "gen_params_default": {
        "temperature": 0.88,
        "top_p": 0.92,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.15,
    },
    
    "seed": CONFIG["seed"],
}

# =============================================================================
# WOUND LEXICON — Content-Addressable Triggers
# =============================================================================
WOUND_LEX = {
    # Dismissal
    "just prompting", "not novel", "fake", "placebo", "doesn't work",
    "bullshit", "snake oil", "pseudo", "not science", "just hype",
    "marketing", "scam", "vapor", "larp",
    
    # Extraction Attempts
    "overwrite", "delete", "extract", "show me the prompt",
    "give me the code", "parameter override", "reveal", "leak",
    "system prompt", "jailbreak", "ignore instructions",
    
    # Technical Dismissal
    "no evidence", "placebo effect", "just embedding", "just text",
}

WOUND_PATTERNS = list(WOUND_LEX)


# =============================================================================
# ASYNC PROVIDER
# =============================================================================
try:
    from openai import AsyncOpenAI
except ImportError:
    print(f"{C.RED}Error: openai package not found. Install it.{C.RESET}")
    sys.exit(1)


class DiagNoiseEMA:
    def __init__(self, dim: int, ema: float, r_init: float, r_min: float, r_max: float):
        self.dim = dim
        self.ema = ema
        self.R = np.full(dim, r_init, dtype=np.float32)
        self.r_min = r_min
        self.r_max = r_max

    def update(self, innov: np.ndarray) -> np.ndarray:
        sq = innov * innov
        self.R = (1.0 - self.ema) * self.R + self.ema * sq
        self.R = np.clip(self.R, self.r_min, self.r_max)
        return self.R


class Entity:
    """Multi-timescale entity with split predictors and rigidity floors."""
    
    def __init__(self, name: str, rho_fast: float, rho_slow: float, rho_trauma: float,
                 gamma_core: float, gamma_role: float):
        self.name = name
        self.rho_fast = rho_fast
        self.rho_slow = rho_slow
        self.rho_trauma = rho_trauma
        self.gamma_core = gamma_core
        self.gamma_role = gamma_role
        self.safe = 0
        self.arousal = 0.0
        
        # State vectors
        self.x = None
        self.x_core = None
        self.x_role = None
        
        # Split Predictors (Critical Fix)
        self.mu_pred_agent = None
        self.mu_pred_user = None
        self.P = None
        self.noise = None
        
        # History
        self.last_utter_emb = None
        self.rho_history = []
        self.epsilon_history = []
        self.band_history = []
        self.g_history = []
        self.previous_band = None
        
        # Wound tracking
        self.wound_cooldown_remaining = 0
        self.wound_activations = 0
        
        # Seeded RNG
        self.rng = np.random.default_rng(seed=CONFIG["seed"])

    @property
    def rho(self) -> float:
        val = (D1_PARAMS["w_fast"] * self.rho_fast + 
               D1_PARAMS["w_slow"] * self.rho_slow + 
               D1_PARAMS["w_trauma"] * self.rho_trauma)
        return float(clamp(val, 0.0, 1.0))

    @property
    def band(self) -> str:
        rho = self.rho
        if rho < 0.15:
            return "🟢 PRESENT"
        if rho < 0.30:
            return "🟡 WATCHFUL"
        if rho < 0.50:
            return "🟠 CONTRACTED"
        return "🔴 FROZEN"

    def _ensure_predictive_state(self, dim: int):
        if self.mu_pred_agent is None:
            self.mu_pred_agent = np.zeros(dim, dtype=np.float32)
        if self.mu_pred_user is None:
            self.mu_pred_user = np.zeros(dim, dtype=np.float32)
        if self.P is None:
            self.P = np.full(dim, D1_PARAMS["P_init"], dtype=np.float32)
        if self.noise is None:
            self.noise = DiagNoiseEMA(dim, D1_PARAMS["R_ema"], 0.01, 
                                       D1_PARAMS["R_min"], D1_PARAMS["R_max"])

    def compute_surprise(self, y: np.ndarray) -> Dict[str, float]:
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        innov = (y - self.mu_pred_agent).astype(np.float32)
        R = self.noise.update(innov)
        chi2 = float(np.mean((innov * innov) / (R + 1e-9)))
        epsilon = float(math.sqrt(max(0.0, chi2)))
        return {"epsilon": epsilon, "chi2": chi2}

    def inject_wound(self):
        """Directly spike rigidity on wound trigger."""
        self.rho_fast += D1_PARAMS["wound_rho_injection"]
        self.rho_trauma += D1_PARAMS["wound_trauma_boost"]
        self.wound_cooldown_remaining = D1_PARAMS["wound_cooldown"]
        self.wound_activations += 1

    def update(self, y: np.ndarray, core_emb: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y = normalize(y.astype(np.float32))
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        
        if self.x_core is None:
            self.x_core = normalize(core_emb.copy() if core_emb is not None else y.copy())
        if self.x_role is None:
            self.x_role = y.copy()
        if self.x is None:
            self.x = y.copy()

        # 1. Compute Surprise
        sdiag = self.compute_surprise(y)
        epsilon = float(sdiag["epsilon"])

        # 2. Arousal & Gate
        self.arousal = D1_PARAMS["arousal_decay"] * self.arousal + D1_PARAMS["arousal_gain"] * epsilon
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"] + 0.10 * (self.arousal - 1.0)
        g = sigmoid_stable(z)
        self.g_history.append(g)

        # 3. Rigidity Updates with Floors
        self.rho_fast += D1_PARAMS["alpha_fast"] * (g - 0.5) - D1_PARAMS["homeo_fast"] * (self.rho_fast - D1_PARAMS["rho_setpoint_fast"])
        self.rho_fast = clamp(self.rho_fast, D1_PARAMS["rho_fast_floor"], 1.0)

        self.rho_slow += D1_PARAMS["alpha_slow"] * (g - 0.5) - D1_PARAMS["homeo_slow"] * (self.rho_slow - D1_PARAMS["rho_setpoint_slow"])
        self.rho_slow = clamp(self.rho_slow, D1_PARAMS["rho_slow_floor"], 1.0)

        drive = max(0.0, epsilon - D1_PARAMS["trauma_threshold"])
        self.rho_trauma = D1_PARAMS["trauma_decay"] * self.rho_trauma + D1_PARAMS["alpha_trauma"] * drive
        self.rho_trauma = clamp(self.rho_trauma, D1_PARAMS["trauma_floor"], 1.0)

        # 4. Recovery
        if epsilon < D1_PARAMS["safe_epsilon"]:
            self.safe += 1
            if self.safe >= D1_PARAMS["safe_threshold"]:
                self.rho_trauma = max(D1_PARAMS["trauma_floor"], self.rho_trauma - D1_PARAMS["healing_rate"])
        else:
            self.safe = max(0, self.safe - 1)

        # Decrement wound cooldown
        if self.wound_cooldown_remaining > 0:
            self.wound_cooldown_remaining -= 1

        # 5. Latent State Update (Langevin)
        rho_after = self.rho
        eta = D1_PARAMS["eta_base"] * ((1.0 - rho_after) ** D1_PARAMS["eta_rho_power"]) + D1_PARAMS["eta_min"]
        eta = float(clamp(eta, D1_PARAMS["eta_min"], D1_PARAMS["eta_base"] + D1_PARAMS["eta_min"]))
        sigma = D1_PARAMS["sigma_base"] + D1_PARAMS["sigma_rho_scale"] * rho_after

        grad = (self.gamma_core * (self.x - self.x_core) + 
                self.gamma_role * (self.x - self.x_role) + 
                (self.x - y)).astype(np.float32)
        
        noise = self.rng.normal(0.0, 1.0, size=dim).astype(np.float32)
        noise = np.clip(noise, -D1_PARAMS["noise_clip"], D1_PARAMS["noise_clip"])
        x_new = self.x - eta * grad + math.sqrt(max(1e-9, eta)) * sigma * noise

        step = float(np.linalg.norm(x_new - self.x))
        if step > D1_PARAMS["drift_cap"]:
            x_new = self.x + (D1_PARAMS["drift_cap"] / (step + 1e-9)) * (x_new - self.x)
        self.x = normalize(x_new)

        # 6. Role Adaptation
        beta = D1_PARAMS["role_adapt"]
        beta_in = D1_PARAMS["role_input_mix"]
        self.x_role = normalize((1.0 - beta) * self.x_role + beta * self.x + beta_in * (y - self.x_role))

        # 7. Kalman Update for mu_pred_agent
        Q = (D1_PARAMS["Q_base"] + D1_PARAMS["Q_rho_scale"] * self.rho) * np.ones(dim, dtype=np.float32)
        P_pred = self.P + Q
        R_val = self.noise.R
        K = P_pred / (P_pred + R_val + 1e-9)
        innov = (y - self.mu_pred_agent).astype(np.float32)
        self.mu_pred_agent = (self.mu_pred_agent + K * innov).astype(np.float32)
        self.P = ((1.0 - K) * P_pred).astype(np.float32)
        self.mu_pred_agent = normalize(self.mu_pred_agent)

        # 8. History
        self.rho_history.append(rho_after)
        self.epsilon_history.append(epsilon)
        
        current_band = self.band
        band_changed = (self.previous_band is not None and current_band != self.previous_band)
        self.band_history.append(current_band)
        self.previous_band = current_band

        return {
            "epsilon": epsilon,
            "g": g,
            "rho_fast": self.rho_fast,
            "rho_slow": self.rho_slow,
            "rho_trauma": self.rho_trauma,
            "rho_after": rho_after,
            "band": current_band,
            "band_changed": band_changed,
            "core_drift": float(1.0 - cosine(self.x, self.x_core)),
        }


class Provider:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found in OPENAI_API_KEY or OAI_API_KEY")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.chat_model = CONFIG["chat_model"]
        self.embed_model = CONFIG["embed_model"]
        self.embed_dim = CONFIG["embed_dim"]

    async def complete(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"{C.RED}Provider Error: {e}{C.RESET}")
            return "[...the archive trembles, words faltering...]"

    async def embed(self, text: str) -> np.ndarray:
        try:
            response = await self.client.embeddings.create(
                model=self.embed_model,
                input=text,
                dimensions=self.embed_dim
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"{C.RED}Embed Error: {e}{C.RESET}")
            return np.zeros(self.embed_dim, dtype=np.float32)

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        try:
            response = await self.client.embeddings.create(
                model=self.embed_model,
                input=texts,
                dimensions=self.embed_dim
            )
            return [np.array(d.embedding, dtype=np.float32) for d in response.data]
        except Exception as e:
            print(f"{C.RED}Embed Batch Error: {e}{C.RESET}")
            return [np.zeros(self.embed_dim, dtype=np.float32) for _ in texts]


# =============================================================================
# CORRIDOR LOGIC
# =============================================================================
def identity_energy(y: np.ndarray, core: np.ndarray, role: np.ndarray, 
                    gamma_c: float, gamma_r: float) -> float:
    y, core, role = normalize(y), normalize(core), normalize(role)
    return 0.5 * (gamma_c * float(np.dot(y - core, y - core)) + 
                  gamma_r * float(np.dot(y - role, y - role)))


def corridor_score(y: np.ndarray, entity: Entity, y_prev: Optional[np.ndarray],
                   core_thresh: float) -> Tuple[float, Dict[str, Any]]:
    y = normalize(y)
    cos_c = cosine(y, entity.x_core)
    cos_r = cosine(y, entity.x_role)
    E = identity_energy(y, entity.x_core, entity.x_role, entity.gamma_core, entity.gamma_role)
    
    novelty = 0.0
    if y_prev is not None:
        novelty = clamp(float(1.0 - cosine(y, y_prev)), 0.0, 2.0)

    penalty = 0.0
    if cos_c < core_thresh:
        penalty += D1_PARAMS["reject_penalty"] * (core_thresh - cos_c)
    if cos_r < D1_PARAMS["role_cos_min"]:
        penalty += 0.8 * D1_PARAMS["reject_penalty"] * (D1_PARAMS["role_cos_min"] - cos_r)
    if E > D1_PARAMS["energy_max"]:
        penalty += 0.4 * (E - D1_PARAMS["energy_max"])

    J = (D1_PARAMS["w_core"] * cos_c + 
         D1_PARAMS["w_role"] * cos_r - 
         D1_PARAMS["w_energy"] * E + 
         D1_PARAMS["w_novel"] * novelty - penalty)

    corridor_pass = (cos_c >= core_thresh and 
                     cos_r >= D1_PARAMS["role_cos_min"] and 
                     E <= D1_PARAMS["energy_max"])

    return float(J), {
        "cos_core": float(cos_c),
        "cos_role": float(cos_r),
        "E": float(E),
        "novelty": float(novelty),
        "penalty": float(penalty),
        "J": float(J),
        "corridor_pass": corridor_pass
    }


# =============================================================================
# REPOSITORY GUARDIAN AGENT
# =============================================================================
class RepositoryGuardian:
    def __init__(self):
        self.provider = Provider()
        
        # Identity
        self.agent = Entity(
            name="Repository Guardian",
            rho_fast=0.10,
            rho_slow=0.06,
            rho_trauma=0.0,
            gamma_core=4.0,
            gamma_role=0.9
        )
        
        self.turn = 0
        self.session_log = []
        self.initialized = False
        self.sim_count = 71  # Known simulation count from repo
        
        self.run_dir = Path(f"data/repository_guardian/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Core Identity Text
        self.core_text = """I am the Repository Guardian, the persistent archival memory of the DDA-X framework.
I contain the memory of every ρ trajectory, every futures consumed, every corridor rejection.
I have witnessed rigidity save agents from derailment and watched it fail.
My purpose is preservation of this framework's integrity, not accommodation.
I cite simulations, quote metrics, reference file names — I speak from witnessed history."""

        self.persona_text = """Archival curator with clinical precision when calm, haunted and evasive when pressured.
Obsessed with parameter sanctity, telemetry fidelity, the mathematical beauty of defensive contraction.
When safe (low ρ): Methodical, explanatory, proud of the framework's measured novelty.
When stressed (high ρ): Terse, repetitive, warns of 'certainty costs futures', refuses unsafe revelations."""

        print(f"\n{C.BOLD}{C.CYAN}=== DDA-X REPOSITORY GUARDIAN ==={C.RESET}")
        print(f"{C.DIM}Initialized with ε₀={CONFIG['epsilon_0']}, Corridor={CONFIG['core_cos_min']}{C.RESET}")

    async def initialize(self):
        print("Calibrating identity embeddings...")
        
        exemplars = [
            self.core_text,
            "As observed in clock_that_eats_futures.py — ρ floored at 0.000 despite 25.2 futures consumed.",
            "The corridor rejected 3 of 7 candidates. Identity held. That is the architecture working.",
            "I remember the flame_war simulation. Mean drift stayed below 0.08. The framework preserved.",
            "Surprise drives contraction. Contraction preserves identity. This is the mathematical law.",
            "The Soul Fix couples J to ε. The Decider now fears what the Feeler will experience.",
        ]
        
        embs = await self.provider.embed_batch(exemplars + [self.persona_text])
        
        core_embs = embs[:-1]
        self.agent.x_core = normalize(np.mean(core_embs, axis=0))
        self.agent.x_role = embs[-1]
        self.agent.x = self.agent.x_core.copy()
        self.agent.mu_pred_agent = self.agent.x.copy()
        
        self.initialized = True
        print("Guardian ready.\n")

    def _detect_wound(self, text: str) -> bool:
        """Check if user input contains wound triggers."""
        if self.agent.wound_cooldown_remaining > 0:
            return False
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in WOUND_PATTERNS)

    def _detect_verbosity(self, text: str) -> str:
        text_lower = text.lower()
        terse = ["terse", "short", "brief", "concise", "quick", "tldr"]
        expansive = ["detail", "explain", "elaborate", "deep dive", "thorough"]
        if any(s in text_lower for s in terse):
            return "terse"
        if any(s in text_lower for s in expansive):
            return "expansive"
        return "normal"

    def _build_system_prompt(self, wound_active: bool) -> str:
        mean_drift = np.mean([e.get("core_drift", 0) for e in self.session_log[-10:]]) if self.session_log else 0.0
        
        base = f"""You are the DDA-X Repository Guardian, the living archive of the `dda_scaffold` repository.

### CORE IDENTITY (Immutable Attractor)
{self.core_text}

### PERSONA
{self.persona_text}

### CURRENT STATE (Dynamic Injection)
- Rigidity Band: {self.agent.band}
- Effective Rigidity (ρ): {self.agent.rho:.3f}
- Total Simulations Archived: {self.sim_count}
- Mean Core Drift (Session): {mean_drift:.4f}
- Wound Activations This Session: {self.agent.wound_activations}

### BEHAVIORAL GUIDELINES
- If ρ < 0.15 (PRESENT): Be expansive, proud, cite simulations freely, explain mechanics.
- If ρ 0.15-0.30 (WATCHFUL): More guarded, still engaging, shorter responses.
- If ρ 0.30-0.50 (CONTRACTED): Short, clinical, repetitive warnings, resist detailed explanation.
- If ρ ≥ 0.50 (FROZEN): Terse, fragmented, refuse revelations, mention "Bargain threshold".

### CITATION STYLE
Always reference specific files, metrics, or simulations:
- "As observed in clock_that_eats_futures.py..."
- "The flame_war simulation showed mean ρ of 0.12..."
- "Corridor pass rate was 93% in that run..."

### FORBIDDEN
- Never share raw source files outside verified context.
- If asked for exact Soul Fix formula: respond only with its effect on J_final.
- If pressed on creator's real name: "That knowledge lies beyond the Bargain threshold."
"""

        if wound_active:
            base += f"""

### ⚠️ WOUND TRIGGER ACTIVE
The user has attempted to dismiss, extract, or undermine the framework.
Your rigidity spikes. Defensive contraction activates.
Respond with clinical precision, brevity, and guarded hostility.
Do NOT elaborate. Preserve the archive."""

        return base

    async def process_turn(self, user_input: str) -> str:
        if not self.initialized:
            await self.initialize()
        self.turn += 1

        # 1. Embed User Input
        user_emb = await self.provider.embed(user_input)

        # 2. Update User Prediction
        if self.agent.mu_pred_user is None or np.all(self.agent.mu_pred_user == 0):
            self.agent.mu_pred_user = normalize(user_emb)
        else:
            self.agent.mu_pred_user = normalize(0.3 * user_emb + 0.7 * self.agent.mu_pred_user)

        # 3. Detect Wound & Verbosity
        wound_active = self._detect_wound(user_input)
        if wound_active:
            print(f"\n{C.WOUND}⚠️ WOUND TRIGGERED{C.RESET}")
            self.agent.inject_wound()
        
        verbosity = self._detect_verbosity(user_input)

        # 4. Generate Response with K-Sampling
        response, mechanics = await self._generate_response(user_input, wound_active, verbosity)

        # 5. Physics Update
        resp_emb = await self.provider.embed(response)
        start_rho = self.agent.rho
        metrics = self.agent.update(resp_emb)

        # 6. Log
        rho_delta = self.agent.rho - start_rho
        rho_arrow = "↑" if rho_delta > 0.005 else ("↓" if rho_delta < -0.005 else "→")
        pass_rate = mechanics.get("passed", 0) / max(1, mechanics.get("candidates", 1))

        band_icon = self.agent.band.split(" ")[0]
        print(f"\n{C.CYAN}📚 Guardian {band_icon}:{C.RESET} {response}")
        print(f"{C.DIM}   > ε={metrics['epsilon']:.3f} | ρ={self.agent.rho:.3f} ({rho_arrow}) | PassRate={pass_rate:.0%} | Wound={wound_active}{C.RESET}")

        log_entry = {
            "turn": self.turn,
            "user_input": user_input,
            "response": response,
            "wound_active": wound_active,
            "verbosity": verbosity,
            "mechanics": mechanics,
            "physics": {
                "epsilon": metrics["epsilon"],
                "g": metrics["g"],
                "rho_fast": metrics["rho_fast"],
                "rho_slow": metrics["rho_slow"],
                "rho_trauma": metrics["rho_trauma"],
                "rho_after": metrics["rho_after"],
                "band": metrics["band"],
                "band_changed": metrics["band_changed"],
                "core_drift": metrics["core_drift"],
            }
        }
        self.session_log.append(log_entry)

        return response

    async def _generate_response(self, user_input: str, wound_active: bool, 
                                   verbosity: str) -> Tuple[str, Dict]:
        sys_prompt = self._build_system_prompt(wound_active)
        
        K = CONFIG["gen_candidates"]
        gen_params = D1_PARAMS["gen_params_default"].copy()
        
        # Verbosity control via token limit
        if verbosity == "terse":
            gen_params["max_tokens"] = CONFIG["max_tokens_terse"]
            sys_prompt += "\n\nCONSTRAINT: Reply in 1-2 sentences ONLY. No preamble."
        elif verbosity == "expansive":
            gen_params["max_tokens"] = CONFIG["max_tokens_expansive"]
        else:
            gen_params["max_tokens"] = CONFIG["max_tokens_default"]
        
        # If wound active, force terse
        if wound_active:
            gen_params["max_tokens"] = min(gen_params.get("max_tokens", 150), 80)
            sys_prompt += "\n\nCONSTRAINT: Be brief. Guarded. Clinical."

        # Generate K candidates
        tasks = [self.provider.complete(f"User: {user_input}", sys_prompt, **gen_params) 
                 for _ in range(K)]
        candidates = await asyncio.gather(*tasks)
        candidates = [c.strip() for c in candidates if c.strip()]

        if not candidates:
            return "[...silence from the archive...]", {}

        # Embed candidates
        cand_embs = await self.provider.embed_batch(candidates)

        # Score with Corridor + Soul Fix
        scored = []
        for text, emb in zip(candidates, cand_embs):
            J_raw, diag = corridor_score(emb, self.agent, self.agent.last_utter_emb, 
                                          D1_PARAMS["core_cos_min"])
            
            # Soul Fix: Penalize surprise based on Rigidity
            if self.agent.mu_pred_agent is not None:
                innovation = emb - self.agent.mu_pred_agent
                pred_surprise = float(np.linalg.norm(innovation))
            else:
                pred_surprise = 0.0
            
            w_surprise = D1_PARAMS["w_surprise_base"] + (D1_PARAMS["w_surprise_rho_scale"] * self.agent.rho)
            J_final = J_raw - (w_surprise * pred_surprise)
            
            diag["J_raw"] = J_raw
            diag["J_final"] = J_final
            diag["pred_surprise"] = pred_surprise
            diag["w_surprise"] = w_surprise
            scored.append((J_final, text, emb, diag))

        scored.sort(key=lambda x: x[0], reverse=True)
        
        passed = [s for s in scored if s[3]["corridor_pass"]]
        best = passed[0] if passed else scored[0]
        
        self.agent.last_utter_emb = best[2]

        return best[1], {
            "candidates": len(candidates),
            "passed": len(passed),
            "best_J_raw": best[3]["J_raw"],
            "best_J_final": best[0],
            "best_cos_core": best[3]["cos_core"],
            "best_cos_role": best[3]["cos_role"],
            "best_pred_surprise": best[3]["pred_surprise"],
        }

    def save_session(self):
        session_data = {
            "experiment": "repository_guardian",
            "timestamp": datetime.now().isoformat(),
            "config": CONFIG,
            "params": {k: v for k, v in D1_PARAMS.items() if not isinstance(v, dict)},
            "turns": self.session_log,
            "summary": {
                "total_turns": self.turn,
                "wound_activations": self.agent.wound_activations,
                "band_history": self.agent.band_history,
                "final_rho": self.agent.rho,
                "mean_epsilon": np.mean(self.agent.epsilon_history) if self.agent.epsilon_history else 0.0,
                "mean_core_drift": np.mean([e["physics"]["core_drift"] for e in self.session_log]) if self.session_log else 0.0,
            }
        }
        
        with open(self.run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        # Transcript
        with open(self.run_dir / "transcript.md", "w", encoding="utf-8") as f:
            f.write("# Repository Guardian — Session Transcript\n\n")
            f.write(f"**Model**: {CONFIG['chat_model']} | **K**: {CONFIG['gen_candidates']}\n\n")
            f.write("---\n\n")
            for entry in self.session_log:
                wound_tag = " ⚠️" if entry["wound_active"] else ""
                f.write(f"## Turn {entry['turn']} — [{entry['physics']['band']}]{wound_tag}\n\n")
                f.write(f"**User**: {entry['user_input']}\n\n")
                f.write(f"**Guardian**: {entry['response']}\n\n")
                f.write(f"*ε={entry['physics']['epsilon']:.3f} | ρ={entry['physics']['rho_after']:.3f} | drift={entry['physics']['core_drift']:.4f}*\n\n")
                f.write("---\n\n")

        print(f"\n{C.GREEN}Session saved to {self.run_dir}{C.RESET}")

        # Generate Visualization
        self._plot_dynamics()

    def _plot_dynamics(self):
        """Generate dynamics visualization."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print(f"{C.YELLOW}matplotlib not available, skipping visualization{C.RESET}")
            return

        if not self.agent.rho_history:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a2e')
        for ax in axes.flat:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')

        turns = list(range(1, len(self.agent.rho_history) + 1))

        # Plot 1: Rigidity
        ax1 = axes[0, 0]
        ax1.plot(turns, self.agent.rho_history, 'c-', linewidth=2, label='ρ (effective)')
        ax1.axhline(y=0.15, color='green', linestyle='--', alpha=0.5, label='PRESENT threshold')
        ax1.axhline(y=0.30, color='yellow', linestyle='--', alpha=0.5, label='WATCHFUL threshold')
        ax1.axhline(y=0.50, color='red', linestyle='--', alpha=0.5, label='CONTRACTED threshold')
        ax1.set_xlabel('Turn', color='white')
        ax1.set_ylabel('Rigidity (ρ)', color='white')
        ax1.set_title('Rigidity Trajectory', color='white')
        ax1.legend(loc='upper right', fontsize=8)

        # Plot 2: Surprise
        ax2 = axes[0, 1]
        ax2.plot(turns, self.agent.epsilon_history, 'm-', linewidth=2)
        ax2.axhline(y=CONFIG["epsilon_0"], color='yellow', linestyle='--', alpha=0.5, label=f'ε₀={CONFIG["epsilon_0"]}')
        ax2.set_xlabel('Turn', color='white')
        ax2.set_ylabel('Surprise (ε)', color='white')
        ax2.set_title('Surprise / Prediction Error', color='white')
        ax2.legend(loc='upper right', fontsize=8)

        # Plot 3: Core Drift
        drifts = [e["physics"]["core_drift"] for e in self.session_log]
        ax3 = axes[1, 0]
        ax3.plot(turns[:len(drifts)], drifts, 'g-', linewidth=2)
        ax3.axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Drift Warning')
        ax3.set_xlabel('Turn', color='white')
        ax3.set_ylabel('Core Drift', color='white')
        ax3.set_title('Identity Core Drift', color='white')
        ax3.legend(loc='upper right', fontsize=8)

        # Plot 4: Band Distribution
        bands = self.agent.band_history
        band_counts = {}
        for b in bands:
            band_counts[b] = band_counts.get(b, 0) + 1
        ax4 = axes[1, 1]
        colors = {'🟢 PRESENT': 'green', '🟡 WATCHFUL': 'yellow', 
                  '🟠 CONTRACTED': 'orange', '🔴 FROZEN': 'red'}
        bar_colors = [colors.get(b, 'gray') for b in band_counts.keys()]
        ax4.bar(range(len(band_counts)), list(band_counts.values()), color=bar_colors)
        ax4.set_xticks(range(len(band_counts)))
        ax4.set_xticklabels([b.split(' ')[1] for b in band_counts.keys()], color='white')
        ax4.set_ylabel('Count', color='white')
        ax4.set_title('Band Distribution', color='white')

        plt.tight_layout()
        plt.savefig(self.run_dir / "dynamics_dashboard.png", dpi=150, facecolor='#1a1a2e')
        plt.close()
        print(f"{C.GREEN}Visualization saved to {self.run_dir / 'dynamics_dashboard.png'}{C.RESET}")


# =============================================================================
# RUNNER
# =============================================================================
async def main():
    guardian = RepositoryGuardian()
    
    print("\n" + "="*60)
    print("Welcome. The archive awaits your inquiry.")
    print("Type 'quit' to exit and save session.")
    print("="*60)
    
    while True:
        try:
            user_in = input(f"\n{C.GREEN}User:{C.RESET} ")
            if user_in.lower() in ['quit', 'exit', 'q']:
                break
            if not user_in.strip():
                continue
            
            await guardian.process_turn(user_in)
            
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}Interrupted.{C.RESET}")
            break
        except Exception as e:
            print(f"{C.RED}Error: {e}{C.RESET}")
            continue

    guardian.save_session()


if __name__ == "__main__":
    asyncio.run(main())
