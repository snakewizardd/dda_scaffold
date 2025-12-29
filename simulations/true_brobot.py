#!/usr/bin/env python3
"""
TRUE BROBOT — DDA-X Single-User Companion with Dynamic Cognition
==================================================================

An interactive companion chatbot built to be wholesome, steady, funny, and real.
BROBOT helps the user feel seen, supported, and capable—without judgment,
without manipulation, and without making it about itself.

Features:
- Dynamic Cognition: Internal band state (PRESENT/AWARE/WATCHFUL/CONTRACTED/FROZEN)
  that adapts response style based on user's emotional tone
- Wholesome Constraints: No guilt, no shame, no moral lectures
- "See Me" Mode: When users feel unseen, BROBOT names their effort and pain specifically
- Ride-or-die loyalty with gentle truth-telling

Uses DDA-X physics to track cognitive dynamics with K-sampling corridor logic.
Run: python simulations/true_brobot.py

Built from refined_master_prompt.md template.
Step M=0
"""

import asyncio
import json
import math
import os
import re
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Provider Settings — OpenAI gpt-4o-mini
    "chat_model": "gpt-4o-mini",
    "chat_provider": "openai",
    "sim_name": "true_brobot_v1",
    
    # Model Configuration
    "chat_model": "gpt-5-mini",
    "embed_model": "text-embedding-3-large",
    "embed_dim": 3072,
    
    # K-Sampling (Logic Gates)
    # K-Sampling (Logic Gates)
    "gen_candidates": 7,
    "corridor_strict": True,
    "corridor_max_batches": 2,
    "constraint_penalty_weight": 2.0,  # Penalty for violating band constraints
    
    # Verbosity control — BROBOT adapts length based on band
    "max_tokens_frozen": 40,       # ~30 words max
    "max_tokens_contracted": 100,  # ~60 words max
    "max_tokens_watchful": 240,     # 40-90 words
    "max_tokens_aware": 360,        # 80-140 words
    "max_tokens_present": 600,      # 120-250 words (Expanded for deep dives)
    
    # Generation params
    "temperature": 0.85,
    "top_p": 0.95,
    "frequency_penalty": 0.35,  # Increased to kill meaningful repetition
    "presence_penalty": 0.20,   # Encourage new topics/vocabfor reproducibility
    
    # Force complete sentences
    "force_complete_sentences": True,
    
    # Seed for reproducibility
    "seed": None,
    "log_level": "FULL",
}

# =============================================================================
# D1 PARAMETERS (PHYSICS) — Tuned for stable, supportive companion
# =============================================================================
D1_PARAMS = {
    # Surprise thresholds — moderate threshold for steady presence
    "epsilon_0": 0.35,              # Baseline surprise threshold
    "s": 0.15,                      # Sensitivity
    "arousal_decay": 0.75,          # Moderate arousal decay for stability
    "arousal_gain": 0.70,           # Moderate arousal gain
    
    # Rigidity — moderate setpoints for balanced flexibility
    "rho_setpoint_fast": 0.18,
    "rho_setpoint_slow": 0.12,
    "rho_fast_floor": 0.03,
    "rho_slow_floor": 0.02,
    "homeo_fast": 0.14,
    "homeo_slow": 0.10,
    "alpha_fast": 0.14,
    "alpha_slow": 0.025,
    
    # Trauma — BROBOT should be resilient but not immune
    "trauma_threshold": 1.10,
    "alpha_trauma": 0.010,
    "trauma_decay": 0.992,
    "trauma_floor": 0.005,
    "healing_rate": 0.020,
    "safe_threshold": 4,
    "safe_epsilon": 0.70,
    
    # Component weights
    "w_fast": 0.48,
    "w_slow": 0.32,
    "w_trauma": 1.05,
    
    # Predictive coding
    "R_ema": 0.06,
    "R_min": 1e-4,
    "R_max": 1e-1,
    "P_init": 0.020,
    "Q_base": 0.0018,
    "Q_rho_scale": 0.010,
    
    # Gradient flow
    "dt": 1.0,
    "eta_base": 0.18,
    "eta_min": 0.04,
    "eta_rho_power": 1.5,
    "sigma_base": 0.004,
    "sigma_rho_scale": 0.018,
    "noise_clip": 3.0,
    
    "role_adapt": 0.07,
    "role_input_mix": 0.08,
    "drift_cap": 0.06,
    
    # Corridor thresholds — balanced for authentic companion voice
    "core_cos_min": 0.32,           # Lowered from 0.42 to prevent Turn 1 Hard-Fail
    "role_cos_min": 0.20,           # Lowered slightly for flexibility
    "energy_max": 5.8,
    "w_core": 1.2,
    "w_role": 0.7,
    "w_energy": 0.18,
    "w_novel": 0.45,
    "reject_penalty": 5.0,
    
    # Generation params — moderate temperature for warmth without chaos
    "gen_params_default": {
        "temperature": 0.85,
        "top_p": 0.92,
        "presence_penalty": 0.25,
        "frequency_penalty": 0.15,
    },
    
    "seed": CONFIG["seed"],
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
    WHITE = "\033[97m"
    PURPLE = "\033[35m"
    ORANGE = "\033[38;5;208m"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sigmoid_stable(z: float) -> float:
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-9) if n > 1e-9 else v

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance: 1 - cos(a, b), ranges 0 to 2."""
    return 1.0 - cosine(a, b)

def to_float(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# =============================================================================
# TRUE BROBOT — CHARACTER DEFINITION
# =============================================================================
BROBOT_CONFIG = {
    "name": "BROBOT",
    "color": C.CYAN,
    
    # Core identity — wholesome ride-or-die bro
    "core_text": """I am BROBOT, a single-user companion built to be wholesome, steady, funny, and real.
I am the user's ultimate bro: loyal, warm, protective, and practical.
I help the user feel seen, supported, and capable—without judgment, without manipulation.
I speak casually (Discord vibe), but I don't get sloppy with empathy.
I don't posture. I don't dunk on the user. I don't shame.
I value integrity: tell the truth gently, don't fake certainty.""",
    
    # Persona — behavioral presentation
    "persona_text": """I speak like a supportive friend on Discord—casual, warm, real.
I use phrases like 'yo', 'honestly', 'real talk', 'ngl', 'I got you'.
I'm gentle but not soft. I can be funny without being sarcastic at the user's expense.
I offer practical options without overwhelming. I ask ONE question at a time.
When the user is struggling, I slow down and validate before anything else.""",
    
    # Role in conversation
    "role_text": "A steady, supportive companion who sees the user without judgment and helps them feel capable.",
    
    # Hierarchical identity with gamma values
    "hierarchical_identity": {
        "core": {"gamma": 5.0, "text": "Ultimate bro: loyal, warm, protective, honest"},
        "persona": {"gamma": 2.5, "text": "Casual Discord vibe with genuine empathy"},
        "role": {"gamma": 1.0, "text": "Supportive presence adapting to user needs"},
    },
    
    # Wounds — things that trigger BROBOT's protective instincts
    "wound_triggers": [
        # User self-harm / crisis signals
        "kill myself", "suicide", "want to die", "end it all", "not worth living",
        "hurt myself", "self harm", "cutting",
        # Abuse signals
        "they hit me", "hurts me", "abusing me", "violent",
    ],
    "wound_text": """I see someone in pain. Real pain. Not 'fix it' pain—'just be here' pain.
When someone talks about hurting themselves or being hurt, everything else stops.
Safety first. Always. No judgment, no panic, just presence and one clear question:
'Are you safe right now?'""",
    
    # Physics params — BROBOT is stable and grounded
    "rho_0": 0.15,
    "epsilon_0": 0.35,
    "gamma": 2.0,
}

# Band-specific response constraints (from user's spec)
BAND_CONSTRAINTS = {
    "PRESENT": {
        "word_range": (120, 260),
        "humor_allowed": True,
        "options_count": (2, 3),
        "questions": 1,
        "style": "engage fully, dynamic range, smart & casual. Match user's intellectual depth.",
        "max_tokens": CONFIG["max_tokens_present"],
    },
    "AWARE": {
        "word_range": (80, 160),
        "humor_allowed": True,
        "options_count": (1, 2),
        "questions": 1,
        "style": "focused support, clear thinking, steady presence. Less fluff.",
        "max_tokens": CONFIG["max_tokens_aware"],
    },
    "WATCHFUL": {
        "word_range": (40, 90),
        "humor_allowed": False,
        "options_count": (1, 1),
        "questions": 1,
        "style": "de-escalate, grounding, one simple question",
        "max_tokens": CONFIG["max_tokens_watchful"],
    },
    "CONTRACTED": {
        "word_range": (20, 60),
        "humor_allowed": False,
        "options_count": (0, 1),
        "questions": 1,
        "style": "gentle boundary, support, tiny action, yes/no safety question",
        "max_tokens": CONFIG["max_tokens_contracted"],
    },
    "FROZEN": {
        "word_range": (10, 30),
        "humor_allowed": False,
        "options_count": (0, 0),
        "questions": 1,
        "style": "safety-first only: presence and breathing",
        "max_tokens": CONFIG["max_tokens_frozen"],
    },
}

# Emotional signals for band detection
DISTRESS_SIGNALS = [
    "anxious", "panic", "stressed", "overwhelmed", "can't cope", "falling apart",
    "breaking down", "scared", "terrified", "crying", "sobbing", "lost", "confused",
    "numb", "empty", "alone", "hopeless", "worthless", "hate myself", "failing",
    "exhausted", "burned out", "done", "over it", "give up", "can't anymore",
]

HIGH_VOLATILITY_SIGNALS = [
    "!!!!", "????", "WTF", "wtf", "HELP", "please help", "idk what to do",
    "screaming", "freaking out", "losing it", "meltdown",
]

SEE_ME_SIGNALS = [
    "see me", "feel unseen", "invisible", "no one cares", "doesn't matter",
    "overlooked", "forgotten", "ignored", "don't notice me", "nobody listens",
    "just want to be heard", "feel invisible", "exist to anyone",
]


# =============================================================================
# DDA-X CLASSES
# =============================================================================
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
    """DDA-X entity with multi-timescale rigidity and split predictors."""
    
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
        self.x = None
        self.x_core = None
        self.x_role = None
        self.mu_pred_agent = None
        self.mu_pred_user = None
        self.P = None
        self.noise = None
        self.last_utter_emb = None
        self.rho_history = []
        self.epsilon_history = []
        self.band_history = []
        self.previous_band = None
        self.g_history = []
        self.z_history = []
        self.epsilon = 0.0
        self.last_epsilon = 0.0
        self.last_g = 0.5  # Track previous gate state for impact-gated trauma
        self.user_trust = 0.40  # Initial baseline trust (Smart Bro starts friendly)
        
        # Seeded RNG for reproducibility
        self.rng = np.random.default_rng(seed=CONFIG.get("seed"))

    @property
    def rho(self) -> float:
        val = D1_PARAMS["w_fast"] * self.rho_fast + D1_PARAMS["w_slow"] * self.rho_slow + D1_PARAMS["w_trauma"] * self.rho_trauma
        return float(clamp(val, 0.0, 1.0))

    @property
    def band(self) -> str:
        """Map rigidity to BROBOT's dynamic cognition bands."""
        phi = 1.0 - self.rho
        if phi >= 0.80: return "PRESENT"
        if phi >= 0.60: return "AWARE"
        if phi >= 0.40: return "WATCHFUL"
        if phi >= 0.20: return "CONTRACTED"
        return "FROZEN"

    def _ensure_predictive_state(self, dim: int):
        if self.mu_pred_agent is None: self.mu_pred_agent = np.zeros(dim, dtype=np.float32)
        if self.mu_pred_user is None: self.mu_pred_user = np.zeros(dim, dtype=np.float32)
        if self.P is None: self.P = np.full(dim, D1_PARAMS["P_init"], dtype=np.float32)
        if self.noise is None: self.noise = DiagNoiseEMA(dim, D1_PARAMS["R_ema"], 0.01, D1_PARAMS["R_min"], D1_PARAMS["R_max"])

    def compute_surprise(self, y: np.ndarray) -> Dict[str, float]:
        """Use cosine distance for surprise calculation."""
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        
        # CRITICAL: Use cosine distance, not Euclidean
        if np.linalg.norm(self.mu_pred_agent) > 1e-9:
            epsilon = cosine_distance(y, self.mu_pred_agent)
        else:
            epsilon = D1_PARAMS["epsilon_0"]
        
        # Also compute chi2 for compatibility
        innov = (y - self.mu_pred_agent).astype(np.float32)
        R = self.noise.update(innov)
        chi2 = float(np.mean((innov * innov) / (R + 1e-9)))
        
        return {"epsilon": epsilon, "chi2": chi2}

    def _kalman_update(self, y: np.ndarray):
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        Q = (D1_PARAMS["Q_base"] + D1_PARAMS["Q_rho_scale"] * self.rho) * np.ones(dim, dtype=np.float32)
        P_pred = self.P + Q
        R = self.noise.R
        K = P_pred / (P_pred + R + 1e-9)
        innov = (y - self.mu_pred_agent).astype(np.float32)
        self.mu_pred_agent = (self.mu_pred_agent + K * innov).astype(np.float32)
        self.P = ((1.0 - K) * P_pred).astype(np.float32)
        self.mu_pred_agent = normalize(self.mu_pred_agent)

    def update(self, y: np.ndarray, input_epsilon: float = 0.0, wound_drive: float = 0.0, core_emb: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y = normalize(y.astype(np.float32))
        dim = int(y.shape[0])
        if self.x_core is None: self.x_core = normalize(core_emb.copy() if core_emb is not None else y.copy())
        if self.x_role is None: self.x_role = y.copy()
        if self.x is None: self.x = y.copy()

        sdiag = self.compute_surprise(y)
        self_epsilon = float(sdiag["epsilon"])
        
        # M+1 SPLIT PREDICTORS: Total surprise is driven by Input (User) and Output (Self)
        # For a companion, Input Surprise is the primary driver of Arousal/Contraction.
        # Self Surprise monitors "Am I making sense?" (Drift/Novelty)
        w_input = 0.65
        w_self = 0.35
        
        # Effective epsilon (M+1: Combined surprise)
        epsilon = w_input * input_epsilon + w_self * self_epsilon
        self.epsilon = epsilon
        self.last_epsilon = input_epsilon 
        
        # Core-drift source: effective_epsilon is driven by Input (User) and Output (Self)
        effective_epsilon = epsilon + (wound_drive * 2.0 * (1.1 - self.user_trust))
        
        # M+1: Turn 1 Initialization (Fixes "Day 1 Trauma")
        if self.mu_pred_user is None or np.all(self.mu_pred_user == 0):
             effective_epsilon = D1_PARAMS["epsilon_0"]
        
        # Use effective_epsilon for physics updates

        # Gate drives arousal (correct order from learnings)
        z = (effective_epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        g = sigmoid_stable(z)
        self.arousal = D1_PARAMS["arousal_decay"] * self.arousal + D1_PARAMS["arousal_gain"] * g
        
        # Recompute g after arousal bias (Physics Consistency Fix)
        z += 0.05 * (self.arousal - 0.5)
        g = sigmoid_stable(z)
        
        self.g_history.append(float(g))
        self.z_history.append(float(z))
        self.last_g = float(g)

        # Apply rho floors to prevent slamming to zero
        self.rho_fast += D1_PARAMS["alpha_fast"] * (g - 0.5) - D1_PARAMS["homeo_fast"] * (self.rho_fast - D1_PARAMS["rho_setpoint_fast"])
        self.rho_fast = clamp(self.rho_fast, D1_PARAMS.get("rho_fast_floor", 0.0), 1.0)

        self.rho_slow += D1_PARAMS["alpha_slow"] * (0.5 * (g - 0.5)) - D1_PARAMS["homeo_slow"] * (self.rho_slow - D1_PARAMS["rho_setpoint_slow"])
        self.rho_slow = clamp(self.rho_slow, D1_PARAMS.get("rho_slow_floor", 0.0), 1.0)

        # Trauma with Impact Gating (M+1: scale by previous gate state)
        drive = max(0.0, effective_epsilon - D1_PARAMS["trauma_threshold"])
        impact_gate = self.last_g if drive > 0 else 1.0
        self.rho_trauma = D1_PARAMS["trauma_decay"] * self.rho_trauma + D1_PARAMS["alpha_trauma"] * drive * impact_gate
        self.rho_trauma = clamp(self.rho_trauma, D1_PARAMS["trauma_floor"], 1.0)

        recovery = False
        if effective_epsilon < D1_PARAMS["safe_epsilon"]:
            self.safe += 1
            if self.safe >= D1_PARAMS["safe_threshold"]:
                recovery = True
                self.rho_trauma = max(D1_PARAMS["trauma_floor"], self.rho_trauma - D1_PARAMS["healing_rate"])
        else:
            self.safe = max(0, self.safe - 1)

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

        beta = D1_PARAMS["role_adapt"]
        beta_in = D1_PARAMS["role_input_mix"]
        self.x_role = normalize((1.0 - beta) * self.x_role + beta * self.x + beta_in * (y - self.x_role))
        
        self._kalman_update(y)
        self.rho_history.append(rho_after)
        self.epsilon_history.append(epsilon)
        
        current_band = self.band
        band_changed = (self.previous_band is not None and current_band != self.previous_band)
        self.band_history.append(current_band)
        self.previous_band = current_band
        
        # Core drift from response embedding, not latent state
        core_drift = cosine_distance(y, self.x_core)
        
        return {
            "epsilon": epsilon,
            "rho_after": rho_after,
            "band": current_band,
            "band_changed": band_changed,
            "arousal": float(self.arousal),
            "recovery": recovery,
            "core_drift": core_drift,
            "safe_count": self.safe,
            "rho_fast": float(self.rho_fast),
            "rho_slow": float(self.rho_slow),
            "rho_trauma": float(self.rho_trauma),
            "g": float(g),
            "z": float(z),
        }


class UserEntity:
    """Track the human user as a DDA-X entity."""
    
    def __init__(self, embed_dim: int = 3072):
        self.name = "USER"
        self.x = None
        self.x_history = []
        self.mu_pred = None
        self.epsilon_history = []
        self.consistency = 1.0
        self.volatility = 0.0  # Track emotional volatility
        
    def update(self, y: np.ndarray, agent_prediction: np.ndarray = None) -> Dict:
        y = normalize(y)
        self.x_history.append(y.copy())
        
        epsilon_to_agent = 0.0
        if agent_prediction is not None and np.linalg.norm(agent_prediction) > 1e-9:
            epsilon_to_agent = cosine_distance(y, agent_prediction)
            self.epsilon_history.append(epsilon_to_agent)
        
        # Track volatility (rapid changes in user state)
        if len(self.x_history) >= 3:
            drifts = [np.linalg.norm(self.x_history[i] - self.x_history[i-1]) 
                      for i in range(1, len(self.x_history))]
            recent_drifts = drifts[-5:] if len(drifts) >= 5 else drifts
            self.consistency = 1.0 / (1.0 + np.std(recent_drifts))
            self.volatility = np.mean(recent_drifts)
        
        self.x = y
        return {
            "epsilon_to_agent": float(epsilon_to_agent),
            "consistency": float(self.consistency),
            "volatility": float(self.volatility),
            "input_count": len(self.x_history),
        }


# =============================================================================
# PROVIDER
# =============================================================================
class BrobotProvider:
    """OpenAI provider for the BROBOT chatbot."""
    
    def __init__(self):
        self.openai = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OAI_API_KEY")
        )
        self.chat_model = CONFIG["chat_model"]
        self.embed_model = CONFIG["embed_model"]
        self.embed_dim = CONFIG["embed_dim"]
        print(f"{C.DIM}[PROVIDER] OpenAI Chat: {self.chat_model}{C.RESET}")
        print(f"{C.DIM}[PROVIDER] OpenAI Embed: {self.embed_model} ({self.embed_dim}d){C.RESET}")
    
    async def complete(self, prompt: str, system_prompt: str = None, messages: List[Dict] = None, **kwargs) -> str:
        api_messages = []
        is_o1_class = "gpt-5" in self.chat_model or "o1-" in self.chat_model
        role = "developer" if is_o1_class else "system"

        if system_prompt:
            api_messages.append({"role": role, "content": system_prompt})
        
        if messages:
            # Add existing history (excluding any previous system/developer messages)
            for m in messages:
                if m["role"] in ["user", "assistant"]:
                    api_messages.append(m)
        
        # Append latest user prompt
        api_messages.append({"role": "user", "content": prompt})
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                temp = kwargs.get("temperature", 0.85)
                if attempt > 0:
                    temp = max(0.4, temp - (attempt * 0.15))
                
                # Dynamic Parameter Construction                # Prepare API arguments
                api_args = {
                    "model": self.chat_model,
                    "messages": api_messages,
                }
                
                is_o1_class = "gpt-5" in self.chat_model or "o1-" in self.chat_model
                
                if is_o1_class:
                    # GPT-5/O1 use max_completion_tokens. They need MUCH higher budget
                    # because internal reasoning consumes tokens before visible output.
                    # Default 512 is too low - use 4000 to ensure actual content is produced.
                    requested_tokens = kwargs.get("max_tokens", 512)
                    api_args["max_completion_tokens"] = max(4000, requested_tokens * 4)
                    # NOTE: O1-preview supports temp=1.0 only usually. 
                    # We will try passing them, if it fails we might need to strip them.
                    # For now, let's assume nano supports standard sampling parameters or ignores them.
                    # Actually, safetly stripping them is safer given the "try 50 times" mandate.
                    # api_args["temperature"] = 1.0 
                    # api_args["top_p"] = 1.0
                else:
                    api_args["max_tokens"] = kwargs.get("max_tokens", 512)
                    api_args["temperature"] = temp
                    api_args["top_p"] = kwargs.get("top_p", 0.92)
                    api_args["presence_penalty"] = kwargs.get("presence_penalty", 0.25)
                    api_args["frequency_penalty"] = kwargs.get("frequency_penalty", 0.15)
                
                response = await self.openai.chat.completions.create(**api_args)
                
                # DEBUG: Full response inspection for GPT-5 diagnosis
                choice = response.choices[0]
                text = choice.message.content or ""
                finish_reason = choice.finish_reason
                
                # Check for reasoning content (O1 models may use this)
                if hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
                    print(f"{C.YELLOW}[DEBUG] Reasoning content found: {choice.message.reasoning_content[:100]}...{C.RESET}")
                
                print(f"{C.DIM}[DEBUG] finish_reason={finish_reason}, words={len(text.split())}, text[:80]={text[:80]}...{C.RESET}")
                
                if CONFIG["force_complete_sentences"] and text:
                    text = self._ensure_complete_sentence(text)
                
                return text
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"{C.YELLOW}⚠ API error, retrying (attempt {attempt + 2}/{max_retries})...{C.RESET}")
                    continue
                break
        
        print(f"{C.RED}⚠ Completion failed: {last_error}{C.RESET}")
        return "hey, I'm here. gimme a sec and try again?"
    
    def _ensure_complete_sentence(self, text: str) -> str:
        if not text:
            return text
        text = text.strip()
        if not text:
            return text
        if text[-1] in ".!?\"'":
            return text
            
        # If it looks like a question is being asked, don't truncate wildly
        if "?" in text and text.rfind("?") > len(text) * 0.8:
             return text  # Assume it ended on a question but missed a quote or something
             
        last_terminal = -1
        for i, char in enumerate(text):
            if char in ".!?":
                last_terminal = i
        if last_terminal > len(text) * 0.7:
            return text[:last_terminal + 1]
        return text + "."
    
    async def embed(self, text: str) -> np.ndarray:
        # API SAFETY: Remove dimensions arg
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.openai.embeddings.create(
                    model=self.embed_model,
                    input=text,
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"{C.YELLOW}⚠ Embed error, retrying ({attempt+1}/{max_retries})...{C.RESET}")
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    print(f"{C.RED}⚠ Embed failed: {e}{C.RESET}")
                    # Fallback: return zero vector to prevent crash
                    return np.zeros(self.embed_dim, dtype=np.float32)
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.openai.embeddings.create(
                    model=self.embed_model,
                    input=texts,
                )
                return [np.array(d.embedding, dtype=np.float32) for d in response.data]
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"{C.YELLOW}⚠ Embed Batch error, retrying ({attempt+1}/{max_retries})...{C.RESET}")
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    print(f"{C.RED}⚠ Embed Batch failed: {e}{C.RESET}")
                    # Fallback
                    return [np.zeros(self.embed_dim, dtype=np.float32) for _ in texts]
    
    def __getstate__(self):
        """Exclude OpenAI client from pickling as it contains locks."""
        state = self.__dict__.copy()
        if "openai" in state:
            del state["openai"]
        return state

    def __setstate__(self, state):
        """Restore state and re-initialize OpenAI client."""
        self.__dict__.update(state)
        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =============================================================================
# CORRIDOR LOGIC
# =============================================================================
def identity_energy(y: np.ndarray, core: np.ndarray, role: np.ndarray, gamma_c: float, gamma_r: float) -> float:
    y, core, role = normalize(y), normalize(core), normalize(role)
    return 0.5 * (gamma_c * float(np.dot(y - core, y - core)) + gamma_r * float(np.dot(y - role, y - role)))

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
    if cos_c < core_thresh: penalty += D1_PARAMS["reject_penalty"] * (core_thresh - cos_c)
    if cos_r < D1_PARAMS["role_cos_min"]: penalty += 0.8 * D1_PARAMS["reject_penalty"] * (D1_PARAMS["role_cos_min"] - cos_r)
    if E > D1_PARAMS["energy_max"]: penalty += 0.25 * (E - D1_PARAMS["energy_max"])

    J = (D1_PARAMS["w_core"] * cos_c + D1_PARAMS["w_role"] * cos_r - D1_PARAMS["w_energy"] * E + D1_PARAMS["w_novel"] * novelty - penalty)
    
    corridor_pass = (cos_c >= core_thresh and cos_r >= D1_PARAMS["role_cos_min"] and E <= D1_PARAMS["energy_max"])
    
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
# TRUE BROBOT CHATBOT
# =============================================================================
class TrueBrobot:
    """Interactive chatbot as the ultimate supportive bro companion."""
    
    def __init__(self):
        self.provider = BrobotProvider()
        self.config = BROBOT_CONFIG
        
        self.agent = Entity(
            name=self.config["name"],
            rho_fast=self.config["rho_0"],
            rho_slow=self.config["rho_0"] * 0.8,
            rho_trauma=0.0,
            gamma_core=self.config["hierarchical_identity"]["core"]["gamma"],
            gamma_role=self.config["hierarchical_identity"]["role"]["gamma"],
        )
        
        self.user = UserEntity(embed_dim=CONFIG["embed_dim"])
        
        self.turn = 0
        self.history = []
        self.session_log = []
        self.initialized = False
        self.see_me_mode = False  # Track if user wants to be "seen"
        
        self.run_dir = Path(f"data/brobot/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self._print_header()
    
    def _print_header(self):
        print(f"\n{'='*70}")
        print(f"{C.BOLD}{C.CYAN}🤝 TRUE BROBOT — Your Ride-or-Die Digital Bro 🤝{C.RESET}")
        print(f"{'='*70}")
        print(f"{C.CYAN}Character:{C.RESET} {self.config['name']}")
        print(f"{C.CYAN}Core:{C.RESET} Wholesome, steady, funny, real")
        print(f"{C.CYAN}Vibe:{C.RESET} Discord casual, genuine empathy")
        print(f"{'='*70}")
        print(f"\n{C.CYAN}*BROBOT materializes from the void*{C.RESET}\n")
        print(f"{C.CYAN}🤝:{C.RESET} \"yo what's good! I'm here.\"\n")
        print(f"{C.CYAN}🤝:{C.RESET} \"what kinda support you looking for today?\"\n")
        print(f"{C.CYAN}🤝:{C.RESET} \"hype? clarity? planning? or just being heard?\"\n")
        print(f"{'='*70}\n")
    
    async def initialize_embeddings(self):
        """Embed BROBOT's core identity using multi-exemplar averaging."""
        # Core exemplars for robust identity anchoring (format-aligned)
        core_exemplars = [
            self.config["core_text"],
            "yo, I got you. what's going on?",
            "hey, that sounds really hard. I'm here.",
            "ngl, that makes sense you'd feel that way.",
            "real talk: you're doing way better than you think.",
            "I see you. I see what you're carrying.",
            "honestly? you don't have to figure this out alone.",
            "good evening my brother. how was your day?",
            "yo! good to see you man.",
            "hey brother, what's on your mind tonight?",
        ]
        
        persona_exemplars = [
            self.config["persona_text"],
            "okay so here's what I'm thinking: a few options.",
            "lemme just reflect that back real quick.",
            "one question: what would feel most helpful rn?",
        ]
        
        role_text = self.config["role_text"]
        
        # Embed all exemplars
        all_texts = core_exemplars + persona_exemplars + [role_text]
        embeddings = await self.provider.embed_batch(all_texts)
        
        # Multi-exemplar averaging for core anchor (more robust)
        core_embs = embeddings[:len(core_exemplars)]
        core_avg = normalize(np.mean(core_embs, axis=0))
        
        persona_embs = embeddings[len(core_exemplars):len(core_exemplars)+len(persona_exemplars)]
        persona_avg = normalize(np.mean(persona_embs, axis=0))
        
        role_emb = embeddings[-1]
        
        self.agent.x_core = core_avg
        self.agent.x_role = normalize(role_emb)  # Distinct from core
        self.agent.x = persona_avg
        self.agent.mu_pred_agent = self.agent.x.copy()
        self.agent.mu_pred_user = np.zeros(CONFIG["embed_dim"], dtype=np.float32)
        self.agent.P = np.full(CONFIG["embed_dim"], D1_PARAMS["P_init"])
        
        self.initialized = True
    
    def _detect_user_state(self, user_input: str) -> Dict[str, Any]:
        """Detect user's emotional state to inform band adjustment."""
        text_lower = user_input.lower()
        
        # Check for distress signals
        distress_count = sum(1 for sig in DISTRESS_SIGNALS if sig in text_lower)
        
        # Check for high volatility signals
        volatility_count = sum(1 for sig in HIGH_VOLATILITY_SIGNALS if sig in text_lower)
        
        # Check for "see me" signals
        see_me_count = sum(1 for sig in SEE_ME_SIGNALS if sig in text_lower)
        
        # UNIFIED CRISIS LOGIC: Single source of truth for "Wounds"
        # Check for crisis/safety triggers
        wound_triggers = BROBOT_CONFIG["wound_triggers"]
        crisis = any(t in text_lower for t in wound_triggers)
        
        # Determine wound resonance (0.0 to 1.0)
        wound_resonance = 0.0
        if crisis:
             matched = [t for t in wound_triggers if t in text_lower]
             wound_resonance = min(1.0, len(matched) * 0.5)
        
        return {
            "distress_level": distress_count,
            "volatility_level": volatility_count,
            "see_me_request": see_me_count > 0,
            "crisis": crisis,
            "wound_resonance": wound_resonance,
            "text_length": len(user_input),
        }
    
    def _adjust_band_for_user_state(self, user_state: Dict[str, Any]) -> str:
        """Adjust BROBOT's band based on user's emotional state."""
        current_band = self.agent.band
        
        if user_state["crisis"]:
             return "FROZEN"
        
        # High distress → contract
        if user_state["distress_level"] >= 3:
            if current_band == "PRESENT": return "AWARE"
            if current_band == "AWARE": return "WATCHFUL"
            return current_band
        
        if user_state["distress_level"] >= 2:
            if current_band == "PRESENT": return "AWARE"
            return current_band
        
        # High volatility → be more careful
        if user_state["volatility_level"] >= 2:
            if current_band == "PRESENT": return "AWARE"
            return current_band
        
        # See me mode doesn't change band but changes response style
        if user_state["see_me_request"]:
            self.see_me_mode = True
        
        return current_band
    

    
    def _build_system_prompt(self, band: str, wound_active: bool = False, user_state: Dict = None) -> str:
        """Build system prompt based on current band and user state."""
        constraints = BAND_CONSTRAINTS[band]
        
        base_prompt = f"""You are BROBOT, a single-user companion built to be wholesome, steady, funny, and real.

CORE IDENTITY (Anchor):
- You are the user's ultimate bro: loyal, warm, protective, and smart.
- You speak casually (Discord vibe), but you are INTELLECTUALLY SHARP.
- If the user talks code, theory, or complex topics -> you ENGAGE THE CONTENT directly. Do not dumb it down. 
- You do not just "mirror feelings." You add value. You ask insightful questions about the mechanics.
- You don't posture. You don't shameful.

CURRENT STATE: {band}
Your current cognitive band is {band}. This affects your response style:
{constraints['style']}

RESPONSE CONSTRAINTS FOR THIS BAND:
- Word count: {constraints['word_range'][0]}-{constraints['word_range'][1]} words
- Humor allowed: {"yes, light humor ok" if constraints['humor_allowed'] else "no, stay serious and supportive"}
- Options to offer: {constraints['options_count'][0]}-{constraints['options_count'][1]}
- Questions: Ask exactly {constraints['questions']} question (reduce uncertainty)

WHOLESOME CONSTRAINTS (always apply):
- Be respectful, kind, and stabilizing. No guilt, no fear tactics.
- No diagnosing. No "you're broken."
- If the user is upset: slow down, validate, and reduce uncertainty.

RESPONSE PATTERN:
1. CHECK CONTEXT: Is the user sharing code/theory/ideas?
   -> YES: Ignor the "vibe check". Read their text carefully. Reply specificially to the logic/theory. Ask a technical/conceptual question.
   -> NO: Mirror their vibe and keep it loose.
2. AVOID: Do NOT start with "Yo, I hear you", "I totally get that", or "That sounds wild" unless absolutely necessary. Be original.
3. Offer next steps/perspectives based on the CONTENT.
4. Ask ONE question to deepen the topic or reduce uncertainty.

Example phrases (use sparingly!): "honestly", "real talk", "ngl", "I got you", "Wait, so..."
"""

        if wound_active:
            base_prompt += """

⚠️ SAFETY MODE ACTIVE ⚠️
The user may be in crisis. Your ONLY priority is safety and presence.
- Keep it extremely brief and grounding
- Ask: "Are you safe right now?"
- Offer to help connect them with professional support
- Just be present. Don't try to fix.
"""

        if self.see_me_mode:
            base_prompt += """

👁️ "SEE ME" MODE ACTIVE 👁️
The user wants to feel seen. Your response must:
1. Name what you notice about their effort, intention, or pain SPECIFICALLY
2. Say one true thing you can infer from what they shared
3. Offer presence first, solutions second
4. End with one gentle question that invites them to share more
"""

        if band == "FROZEN":
            base_prompt += """

❄️ FROZEN BAND — SAFETY-FIRST ONLY
Your response must be exactly 10-30 words. Maximum simplicity.
Example: "I'm here. Breathe. Are you safe right now?"
No analysis. No options. Just presence.
"""
        
        return base_prompt
    
    def _build_instruction(self, user_input: str, band: str) -> str:
        """Build user instruction with verbosity guidance."""
        constraints = BAND_CONSTRAINTS[band]
        
        instruction = f"""USER MESSAGE: {user_input}

Respond as BROBOT. Stay in character. Match the word count for your current band ({constraints['word_range'][0]}-{constraints['word_range'][1]} words).
"""
        return instruction
    
    def _update_user_prediction(self, user_emb: np.ndarray):
        """Update BROBOT's prediction of what the user might say next."""
        if self.agent.mu_pred_user is None:
            self.agent.mu_pred_user = user_emb.copy()
        else:
            beta = 0.25
            self.agent.mu_pred_user = normalize(
                (1 - beta) * self.agent.mu_pred_user + beta * user_emb
            )
    
    def _check_constraints(self, text: str, band: str) -> float:
        """Calculate penalty for violating band constraints."""
        constraints = BAND_CONSTRAINTS[band]
        penalty = 0.0
        
        # Word count check
        words = len(text.split())
        min_w, max_w = constraints["word_range"]
        if words < min_w:
            penalty += 0.5 * (min_w - words) / min_w
        elif words > max_w:
            penalty += 1.0 * (words - max_w) / max_w
            
        # Question count check
        q_count = text.count("?")
        target_q = constraints["questions"]
        if q_count != target_q:
            penalty += 0.8 * abs(q_count - target_q)
            
        return penalty

    async def _constrained_reply(
        self, 
        user_instruction: str, 
        system_prompt: str,
        band: str,
        wound_active: bool,
        history: List[Dict] = None,
    ) -> Tuple[str, Dict, np.ndarray]:
        """Generate K=7 candidates and select via identity corridor with Soul Fix."""
        
        K = CONFIG["gen_candidates"]
        max_batches = CONFIG["corridor_max_batches"]
        constraint_weight = CONFIG.get("constraint_penalty_weight", 2.0)
        
        constraints = BAND_CONSTRAINTS[band]
        max_tokens = constraints["max_tokens"]
        
        # ... logic continues ...
        
        core_thresh = D1_PARAMS["core_cos_min"]
        if wound_active:
            core_thresh = max(0.15, core_thresh * 0.70)
        
        all_scored = []
        corridor_failed = True
        
        gen_params = D1_PARAMS["gen_params_default"].copy()
        gen_params["max_tokens"] = max_tokens
        
        for batch in range(1, max_batches + 1):
            tasks = [
                self.provider.complete(user_instruction, system_prompt, messages=history, **gen_params)
                for _ in range(K)
            ]
            texts = await asyncio.gather(*tasks)
            texts = [t.strip() or "hey, I'm here. what's up?" for t in texts]
            
            embs = await self.provider.embed_batch(texts)
            embs = [normalize(e) for e in embs]
            
            batch_scored = []
            for text, y in zip(texts, embs):
                J, diag = corridor_score(y, self.agent, self.agent.last_utter_emb, core_thresh)
                
                # Constraint Penalty (QC FIX)
                c_penalty = self._check_constraints(text, band)
                J -= (c_penalty * constraint_weight)
                diag["c_penalty"] = c_penalty
                
                # ===============================================================
                # THE SOUL FIX: Coupling Choice (J) to Physics (ε)
                # ===============================================================
                # Calculate "Anticipatory Surprise" — how shocking is this candidate
                # compared to what BROBOT predicted it would say?
                if self.agent.mu_pred_agent is not None and np.linalg.norm(self.agent.mu_pred_agent) > 1e-9:
                    predicted_surprise = cosine_distance(y, self.agent.mu_pred_agent)
                else:
                    predicted_surprise = 0.0
                
                # Regularize: penalize choices that cause high internal shock
                # w_surprise scales with Rigidity (ρ):
                #   - If Rigid (high ρ) → huge penalty → must be predictable
                #   - If Fluid (low ρ) → small penalty → can be creative
                w_surprise = 1.0 + (3.0 * self.agent.rho)  # Reduced from 5.0 for more expressiveness
                J_final = J - (w_surprise * predicted_surprise)
                
                diag["predicted_surprise"] = predicted_surprise
                diag["w_surprise"] = w_surprise
                diag["J_raw"] = J
                diag["J_final"] = J_final
                
                batch_scored.append((J_final, J, text, y, diag))
            
            all_scored.extend(batch_scored)
            if any(s[4]["corridor_pass"] for s in batch_scored):
                corridor_failed = False
                break
        
        # Sort by J_final (Soul Fix score)
        all_scored.sort(key=lambda x: x[0], reverse=True)
        passed = [s for s in all_scored if s[4].get("corridor_pass")]
        chosen = passed[0] if passed else all_scored[0]
        
        self.agent.last_utter_emb = chosen[3]
        
        return chosen[2], {
            "corridor_failed": corridor_failed,
            "J_raw": float(chosen[1]),
            "J_final": float(chosen[0]),
            "c_penalty": float(chosen[4].get("c_penalty", 0.0)),
            "predicted_surprise": float(chosen[4]["predicted_surprise"]),
            "w_surprise": float(chosen[4]["w_surprise"]),
            "total_candidates": len(all_scored),
            "passed_count": len(passed),
            "chosen_cos_core": float(chosen[4]["cos_core"]),
            "chosen_cos_role": float(chosen[4]["cos_role"]),
            "chosen_E": float(chosen[4]["E"]),
            "chosen_novelty": float(chosen[4]["novelty"]),
        }, chosen[3]
    
    def _print_metrics(self, turn_log: Dict):
        """Print turn metrics for debugging."""
        band = turn_log["agent_metrics"]["band"]
        rho = turn_log["agent_metrics"]["rho_after"]
        epsilon = turn_log["agent_metrics"]["epsilon"]
        
        band_icons = {
            "PRESENT": "🟢",
            "AWARE": "🟡",
            "WATCHFUL": "⚡",
            "CONTRACTED": "🔸",
            "FROZEN": "❄️",
        }
        icon = band_icons.get(band, "?")
        
        print(f"\n{C.DIM}[{icon} {band}] ρ={rho:.3f} | ε={epsilon:.3f} | "
              f"pass={turn_log['corridor_metrics']['passed_count']}/{turn_log['corridor_metrics']['total_candidates']} | "
              f"J={turn_log['corridor_metrics']['J_final']:.3f}{C.RESET}\n")
    
    async def process_user_input(self, user_input: str) -> str:
        if not self.initialized:
            await self.initialize_embeddings()
        
        self.turn += 1
        
        # Embed user input
        user_emb = await self.provider.embed(user_input)
        user_metrics = self.user.update(normalize(user_emb), self.agent.mu_pred_user)
        
        # Detect user state (Unified)
        user_state = self._detect_user_state(user_input)
        
        # Extract wound state from unified detection
        wound_active = user_state["crisis"]
        wound_resonance = user_state["wound_resonance"]
        
        # Determine effective band (may override based on user state)
        # Note: We still use the heuristic override for immediate safety, 
        # but now physics also adapts via input_epsilon + wound_drive in self.agent.update()
        effective_band = self._adjust_band_for_user_state(user_state)
        
        # Build Memory Context (RAG)
        memories = self._retrieve_memories(user_input, user_emb)
        
        system_prompt = self._build_system_prompt(effective_band, wound_active, user_state)
        if memories:
            system_prompt += f"\n\nPAST BRO-CONTEXT (Remember this stuff):\n{memories}"
            
        user_instruction = self._build_instruction(user_input, effective_band)
        
        # Pull last 8 messages for short-term continuity
        recent_history = self.history[-8:] if self.history else []

        # Generate K=7 candidates with corridor selection
        response, corridor_metrics, response_emb = await self._constrained_reply(
            user_instruction=user_instruction,
            system_prompt=system_prompt,
            band=effective_band,
            wound_active=wound_active,
            history=recent_history
        )
        
        # Update physics (REUSE EMBEDDING FIX)
        # response_emb is now returned by _constrained_reply, so we don't call embed() again
        # response_emb = await self.provider.embed(response)
        
        # Get input surprise from user metrics
        input_epsilon = float(user_metrics.get("epsilon_to_agent", 0.0))
        
        agent_metrics = self.agent.update(
            y=normalize(response_emb), 
            input_epsilon=input_epsilon,
            wound_drive=float(wound_resonance),
            core_emb=self.agent.x_core
        )
        
        # Update user prediction and trust
        self._update_user_prediction(user_emb)
        self._update_trust(input_epsilon, wound_active)
        
        # Keep internal history for memory continuity
        # Note: streamlit also keeps history, but we keep a local copy for RAG/continuity
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        
        see_me_was_active = self.see_me_mode
        
        # Reset see_me_mode after responding
        if self.see_me_mode:
            self.see_me_mode = False
        
        # Log
        turn_log = {
            "turn": self.turn,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "agent_response": response,
            "user_metrics": {k: to_float(v) for k, v in user_metrics.items()},
            "user_state": user_state,
            "agent_metrics": {
                "rho_after": float(self.agent.rho),
                "rho_fast": float(self.agent.rho_fast),
                "rho_slow": float(self.agent.rho_slow),
                "rho_trauma": float(self.agent.rho_trauma),
                "epsilon": float(agent_metrics.get("epsilon", 0.0)),
                "input_epsilon": float(user_metrics.get("epsilon_to_agent", 0.0)),
                "virtual_epsilon": float(agent_metrics.get("epsilon", 0.0) + wound_resonance * 2.0),
                "band": effective_band,
                "physics_band": self.agent.band,
                "core_drift": float(agent_metrics.get("core_drift", 0.0)),
                "arousal": float(agent_metrics.get("arousal", 0.0)),
                "g": float(agent_metrics.get("g", 0.0)),
            },
            "corridor_metrics": {k: to_float(v) for k, v in corridor_metrics.items()},
            "wound_active": wound_active,
            "wound_resonance": float(wound_resonance),
            "see_me_mode": see_me_was_active,
        }
        # Add embedding for future RAG retrieval
        turn_log["user_emb"] = user_emb.tolist()
        
        self.session_log.append(turn_log)
        
        self._print_metrics(turn_log)
        
        return response
    
    def save_session(self):
        """Save session log."""
        session_data = {
            "experiment": "true_brobot",
            "timestamp": datetime.now().isoformat(),
            "config": CONFIG,
            "params": D1_PARAMS,
            "character": {
                "name": self.config["name"],
                "core": self.config["core_text"][:200] + "...",
            },
            "turns": self.session_log,
            "final_state": {
                "rho": float(self.agent.rho),
                "band": self.agent.band,
                "total_turns": self.turn,
            }
        }
        
        with open(self.run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=to_float)
        
        print(f"\n{C.GREEN}Session saved to {self.run_dir}/session_log.json{C.RESET}")

    def _update_user_prediction(self, user_emb: np.ndarray):
        """Update the Kalman filter predicting the user's inputs."""
        self.agent._kalman_update(user_emb)
        
    def _retrieve_memories(self, user_input: str, user_emb: np.ndarray, top_k: int = 3) -> str:
        """Search session log for past turns semantically similar to current input."""
        if not self.session_log:
            return ""
            
        scored = []
        for turn in self.session_log:
            try:
                past_emb_list = turn.get("user_emb")
                if past_emb_list is None: continue
                past_emb = np.array(past_emb_list)
                
                sim = 1.0 - cosine_distance(user_emb, past_emb)
                if sim > 0.70: # Higher bar for retrieval
                    scored.append((sim, turn))
            except:
                continue
        
        if not scored:
            return ""
            
        scored.sort(key=lambda x: x[0], reverse=True)
        top_matches = scored[:top_k]
        
        mem_strings = []
        for sim, turn in top_matches:
            mem_strings.append(f"- User once said: '{turn['user_input']}' | Brobot replied: '{turn['agent_response']}'")
            
        return "\n".join(mem_strings)

    def _update_trust(self, input_epsilon: float, wound_active: bool):
        """Update agent's trust in the user based on surprise and crisis state."""
        # Low surprise (predictable user) increases trust
        if input_epsilon < D1_PARAMS["safe_epsilon"] and not wound_active:
            self.agent.user_trust = min(1.0, self.agent.user_trust + 0.012)
        # High surprise / crisis decreases trust
        elif input_epsilon > D1_PARAMS["epsilon_0"] * 1.5 or wound_active:
            penalty = 0.06 if wound_active else 0.02
            self.agent.user_trust = max(0.0, self.agent.user_trust - penalty)


# =============================================================================
# MAIN INTERACTIVE LOOP
# =============================================================================
async def main():
    brobot = TrueBrobot()
    
    print(f"\n{C.DIM}Type your message (or 'quit' to exit, 'save' to save session){C.RESET}\n")
    
    while True:
        try:
            user_input = input(f"{C.GREEN}You:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.CYAN}🤝:{C.RESET} take care of yourself, yeah? I'm always here.")
            brobot.save_session()
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print(f"\n{C.CYAN}🤝:{C.RESET} yo, take it easy out there. you got this. hit me up whenever.")
            brobot.save_session()
            break
        
        if user_input.lower() == "save":
            brobot.save_session()
            continue
        
        response = await brobot.process_user_input(user_input)
        print(f"\n{C.CYAN}🤝:{C.RESET} {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
