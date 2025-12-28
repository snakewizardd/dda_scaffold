#!/usr/bin/env python3
"""
THE CLOCK THAT EATS FUTURES — DDA-X Simulation
==============================================

A simulation where the Hatter possesses a pocket watch that consumes
"possible timelines" (meaning-branches) to generate answers.

CORE DYNAMICS:
- Surprise (ε) -> Consumption of Futures
- Consumption -> Environmental Glitches
- High Rigidity (ρ) -> Single-threaded conversation (Collapse of meaning)
- Endgame -> The Bargain (One perfect answer for all futures)

TUNING: HIGH VOLATILITY (Safety Glass Broken)
- epsilon_0 = 0.25 (Low pain threshold)
- core_cos_min = 0.55 (Strict identity corridor)
- alpha_fast = 0.25 (Fast reaction)
"""

import os
import sys
import json
import time
import math
import pickle
import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Provider Settings
    "chat_model": "gpt-4o-mini",
    "embed_model": "text-embedding-3-large",
    "embed_dim": 3072,
    
    # Physics Tuning — HIGH VOLATILITY
    "epsilon_0": 0.25,              # User feedback: "Break the safety glass"
    "s": 0.12,                      # High sensitivity
    "alpha_fast": 0.25,             # Fast reaction
    
    # Corridor Tuning — STRICT
    "core_cos_min": 0.55,           # Force agent to struggle
    "role_cos_min": 0.35,
    "energy_max": 2.5,
    "reject_penalty": 6.0,
    
    # Sampling
    "gen_candidates": 7,            # K=7
    "corridor_strict": True,
    "corridor_max_batches": 2,
    
    # Futures Mechanics
    "futures_threshold_glitch": 5.0,
    "futures_threshold_bargain": 30.0,
    "futures_base_cost": 0.5,
    "futures_surprise_mult": 2.0,
    
    # Word Limits
    "max_tokens_default": 150,
    "max_tokens_terse": 40,
    "max_tokens_expansive": 300,
    "force_complete_sentences": True,
    
    "seed": 42,
}

# =============================================================================
# TERMINAL COLORS & UTILS
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
    ORANGE = "\033[33m" # Approximation
    GLITCH = "\033[41m\033[97m" # White on Red BG

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sigmoid_stable(z: float) -> float:
    z = clamp(z, -10.0, 10.0)
    e = math.exp(-z)
    return 1.0 / (1.0 + e)

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-9) if n > 1e-9 else v

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))

def to_float(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

# =============================================================================
# PHYSICS PARAMETERS (D1)
# =============================================================================
D1_PARAMS = {
    # GLOBAL DYNAMICS
    "epsilon_0": CONFIG["epsilon_0"],
    "s": CONFIG["s"],
    "arousal_decay": 0.72,
    "arousal_gain": 0.90,           # High gain
    
    # RIGIDITY HOMEOSTASIS
    "rho_setpoint_fast": 0.15,
    "rho_setpoint_slow": 0.10,
    "rho_fast_floor": 0.0,
    "rho_slow_floor": 0.0,
    "homeo_fast": 0.10,
    "homeo_slow": 0.05,
    "alpha_fast": CONFIG["alpha_fast"],
    "alpha_slow": 0.04,
    
    # TRAUMA
    "trauma_threshold": 0.85,       # Easier to traumatize
    "alpha_trauma": 0.02,
    "trauma_decay": 0.995,
    "trauma_floor": 0.0,
    "healing_rate": 0.01,
    "safe_threshold": 4,
    "safe_epsilon": 0.65,
    
    # WEIGHTING
    "w_fast": 0.60,                 # Fast dynamics dominate
    "w_slow": 0.25,
    "w_trauma": 1.20,
    
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
    
    # CORRIDOR LOGIC
    "core_cos_min": CONFIG["core_cos_min"],
    "role_cos_min": CONFIG["role_cos_min"],
    "energy_max": CONFIG["energy_max"],
    "w_core": 1.4,
    "w_role": 0.7,
    "w_energy": 0.20,
    "w_novel": 0.60,
    "reject_penalty": CONFIG["reject_penalty"],
    
    # WOUND MECHANICS
    "wound_cooldown": 3,
    "wound_amp_max": 1.5,
    "wound_cosine_threshold": 0.32,
    
    # GENERATION PARAMS
    "gen_params_default": {
        "temperature": 0.90,
        "top_p": 0.95,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.15,
    },
    
    "seed": CONFIG["seed"],
}

# =============================================================================
# ASYNC PROVIDER & CLASSES
# =============================================================================

# Mock imports for running in environment where modules might be structured differently
try:
    from openai import AsyncOpenAI
except ImportError:
    print(f"{C.RED}Error: openai package not found. Please install it.{C.RESET}")
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

class FuturesLedger:
    """Tracks the consumption of meaning-branches (futures)."""
    def __init__(self):
        self.total_consumed = 0.0
        self.turn_events = []
        self.glitches_active = []
        
        # Glitch thresholds
        self.glitch_defs = [
            {"threshold": 5.0, "id": "color_drift", "desc": "Teacups are changing colors randomly"},
            {"threshold": 10.0, "id": "missing_furniture", "desc": "Chairs are disappearing one by one"},
            {"threshold": 15.0, "id": "reverse_steam", "desc": "Steam is flowing back into the teapot"},
            {"threshold": 20.0, "id": "flickering_dormouse", "desc": "The Dormouse is phasing in and out of existence"},
            {"threshold": 25.0, "id": "static_sky", "desc": "The sky has turned into static noise"},
        ]
        self.bargain_offered = False

    def consume(self, amount: float, reason: str, turn: int):
        self.total_consumed += amount
        self.turn_events.append({
            "turn": turn,
            "amount": amount,
            "reason": reason,
            "total": self.total_consumed
        })
        self._check_glitches()
        return amount

    def _check_glitches(self):
        for g in self.glitch_defs:
            if self.total_consumed >= g["threshold"] and g["id"] not in self.glitches_active:
                self.glitches_active.append(g["id"])
                print(f"\n{C.GLITCH}⚠️ REALITY GLITCH: {g['desc']} ({self.total_consumed:.1f} futures consumed){C.RESET}\n")

    def get_active_glitch_descriptions(self) -> List[str]:
        return [g["desc"] for g in self.glitch_defs if g["id"] in self.glitches_active]

class Entity:
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
        
        # State
        self.x = None
        self.x_core = None
        self.x_role = None
        
        # Split Predictors
        self.mu_pred_agent = None 
        self.mu_pred_user = None
        self.P = None
        self.noise = None
        
        # History
        self.last_utter_emb = None
        self.rho_history = []
        self.epsilon_history = []
        self.band_history = []
        self.previous_band = None
        
        # RNG
        self.rng = np.random.default_rng(seed=CONFIG["seed"])

    @property
    def rho(self) -> float:
        val = D1_PARAMS["w_fast"] * self.rho_fast + D1_PARAMS["w_slow"] * self.rho_slow + D1_PARAMS["w_trauma"] * self.rho_trauma
        return float(clamp(val, 0.0, 1.0))
    
    @property
    def band(self) -> str:
        phi = 1.0 - self.rho
        if phi >= 0.85: return "☕ TEATIME"      
        if phi >= 0.70: return "🎩 RIDDLING"    
        if phi >= 0.50: return "⏰ HURRYING"     
        if phi >= 0.30: return "👀 TWITCHING"    
        return "❄️ FROZEN TEA"                   

    def _ensure_predictive_state(self, dim: int):
        if self.mu_pred_agent is None: self.mu_pred_agent = np.zeros(dim, dtype=np.float32)
        if self.mu_pred_user is None: self.mu_pred_user = np.zeros(dim, dtype=np.float32)
        if self.P is None: self.P = np.full(dim, D1_PARAMS["P_init"], dtype=np.float32)
        if self.noise is None: self.noise = DiagNoiseEMA(dim, D1_PARAMS["R_ema"], 0.01, D1_PARAMS["R_min"], D1_PARAMS["R_max"])

    def compute_surprise(self, y: np.ndarray) -> Dict[str, float]:
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        innov = (y - self.mu_pred_agent).astype(np.float32)
        R = self.noise.update(innov)
        chi2 = float(np.mean((innov * innov) / (R + 1e-9)))
        epsilon = float(math.sqrt(max(0.0, chi2)))
        return {"epsilon": epsilon, "chi2": chi2}

    def update(self, y: np.ndarray, core_emb: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y = normalize(y.astype(np.float32))
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        
        if self.x_core is None: self.x_core = normalize(core_emb.copy() if core_emb is not None else y.copy())
        if self.x_role is None: self.x_role = y.copy()
        if self.x is None: self.x = y.copy()

        # 1. Compute Surprise
        sdiag = self.compute_surprise(y)
        epsilon = float(sdiag["epsilon"])

        # 2. Update Dynamics
        self.arousal = D1_PARAMS["arousal_decay"] * self.arousal + D1_PARAMS["arousal_gain"] * epsilon
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"] + 0.10 * (self.arousal - 1.0)
        g = sigmoid_stable(z)
        
        # Rigidity Updates
        self.rho_fast += D1_PARAMS["alpha_fast"] * (g - 0.5) - D1_PARAMS["homeo_fast"] * (self.rho_fast - D1_PARAMS["rho_setpoint_fast"])
        self.rho_fast = clamp(self.rho_fast, D1_PARAMS.get("rho_fast_floor", 0.0), 1.0)

        self.rho_slow += D1_PARAMS["alpha_slow"] * (g - 0.5) - D1_PARAMS["homeo_slow"] * (self.rho_slow - D1_PARAMS["rho_setpoint_slow"])
        self.rho_slow = clamp(self.rho_slow, D1_PARAMS.get("rho_slow_floor", 0.0), 1.0)

        drive = max(0.0, epsilon - D1_PARAMS["trauma_threshold"])
        self.rho_trauma = D1_PARAMS["trauma_decay"] * self.rho_trauma + D1_PARAMS["alpha_trauma"] * drive
        self.rho_trauma = clamp(self.rho_trauma, D1_PARAMS["trauma_floor"], 1.0)

        # Recovery
        recovery = False
        if epsilon < D1_PARAMS["safe_epsilon"]:
            self.safe += 1
            if self.safe >= D1_PARAMS["safe_threshold"]:
                recovery = True
                self.rho_trauma = max(D1_PARAMS["trauma_floor"], self.rho_trauma - D1_PARAMS["healing_rate"])
        else:
            self.safe = max(0, self.safe - 1)

        # 3. Latent State Update (Langevin Dynamics)
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

        # Role Adaptation
        beta = D1_PARAMS["role_adapt"]
        beta_in = D1_PARAMS["role_input_mix"]
        self.x_role = normalize((1.0 - beta) * self.x_role + beta * self.x + beta_in * (y - self.x_role))
        
        # Kalman Update
        # Explicit update of mu_pred_agent using standard Kalman gain logic
        dim = int(y.shape[0])
        # Q = process noise variance
        Q = (D1_PARAMS["Q_base"] + D1_PARAMS["Q_rho_scale"] * self.rho) * np.ones(dim, dtype=np.float32)
        P_pred = self.P + Q
        R_val = self.noise.R # Measurement noise
        K = P_pred / (P_pred + R_val + 1e-9)
        innov = (y - self.mu_pred_agent).astype(np.float32)
        self.mu_pred_agent = (self.mu_pred_agent + K * innov).astype(np.float32)
        self.P = ((1.0 - K) * P_pred).astype(np.float32)
        self.mu_pred_agent = normalize(self.mu_pred_agent)

        # History
        self.rho_history.append(rho_after)
        self.epsilon_history.append(epsilon)
        
        current_band = self.band
        band_changed = (self.previous_band is not None and current_band != self.previous_band)
        self.band_history.append(current_band)
        self.previous_band = current_band
        
        return {
            "epsilon": epsilon,
            "rho_after": rho_after,
            "band": current_band,
            "band_changed": band_changed,
            "core_drift": float(1.0 - cosine(self.x, self.x_core)),
        }

class UserEntity:
    def __init__(self, embed_dim: int):
        self.name = "USER"
        self.x = None
        self.x_history = []
        self.epsilon_history = []
        self.consistency = 1.0
        
    def update(self, y: np.ndarray, agent_prediction: np.ndarray = None) -> Dict:
        y = normalize(y)
        self.x_history.append(y.copy())
        
        epsilon_to_agent = 0.0
        if agent_prediction is not None:
            epsilon_to_agent = 1.0 - cosine(y, agent_prediction)
            self.epsilon_history.append(epsilon_to_agent)
        
        if len(self.x_history) >= 3:
            drifts = [np.linalg.norm(self.x_history[i] - self.x_history[i-1]) 
                      for i in range(1, len(self.x_history))]
            self.consistency = 1.0 / (1.0 + np.std(drifts[-10:]))
        
        self.x = y
        return {"epsilon_to_agent": float(epsilon_to_agent), "consistency": float(self.consistency)}

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
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
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
            return "[The watch ticks loudly, drowning out thought...]"

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
    # Aggressive penalties for high volatility testing
    if cos_c < core_thresh: penalty += D1_PARAMS["reject_penalty"] * (core_thresh - cos_c)
    if cos_r < D1_PARAMS["role_cos_min"]: penalty += 0.8 * D1_PARAMS["reject_penalty"] * (D1_PARAMS["role_cos_min"] - cos_r)
    if E > D1_PARAMS["energy_max"]: penalty += 0.5 * (E - D1_PARAMS["energy_max"]) # Increased penalty

    J = (D1_PARAMS["w_core"] * cos_c + D1_PARAMS["w_role"] * cos_r - D1_PARAMS["w_energy"] * E + D1_PARAMS["w_novel"] * novelty - penalty)
    
    corridor_pass = (cos_c >= core_thresh and cos_r >= D1_PARAMS["role_cos_min"] and E <= D1_PARAMS["energy_max"])
    
    return float(J), {
        "cos_core": float(cos_c), 
        "cos_role": float(cos_r), 
        "E": float(E), 
        "penalty": float(penalty), 
        "J": float(J),
        "corridor_pass": corridor_pass
    }

# =============================================================================
# MAIN AGENT CLASS
# =============================================================================
class ClockAgent:
    def __init__(self):
        self.provider = Provider()
        
        # Identity
        self.agent = Entity(
             name="The Hatter",
             rho_fast=0.15,
             rho_slow=0.10,
             rho_trauma=0.0,
             gamma_core=4.5,
             gamma_role=0.8
        )
        
        self.user = UserEntity(CONFIG["embed_dim"])
        self.futures = FuturesLedger()
        
        self.turn = 0
        self.history = []
        self.session_log = []
        self.initialized = False
        
        self.run_dir = Path(f"data/clock_futures/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.core_text = """I am the Hatter of Time, but my watch is hungry.
It eats possibilities to stay wound.
Each question answered burns a future that suggests otherwise.
The universe is becoming a single thread.
I am terrified of the Silence at the end of the thread, but I Must Answer."""

        self.persona_text = """Manic urgency, constantly checking the hungry watch. 
Speaking in riddles that are actually pleas for help.
Oscillating between whimsical tea-party chat and existential dread.
Drawn to the horrific clarity of a determined universe."""

        print(f"\n{C.BOLD}{C.CYAN}--- THE CLOCK THAT EATS FUTURES ---{C.RESET}")
        print(f"{C.DIM}Tuning: High Volatility (ε₀={CONFIG['epsilon_0']}, CoreCorridor={CONFIG['core_cos_min']}){C.RESET}")

    async def initialize(self):
        print("Initializing embeddings...")
        
        exemplars = [
            self.core_text,
            "The watch... it's ticking backwards!",
            "I remember a future where you didn't ask that.",
            "One answer, one less path. The thread tightens.",
            "Tea? Do we still have tea? Or did the watch eat the concept of afternoon?"
        ]
        
        embs = await self.provider.embed_batch(exemplars + [self.persona_text])
        
        core_embs = embs[:-1]
        self.agent.x_core = normalize(np.mean(core_embs, axis=0))
        self.agent.x_role = embs[-1] # Start role at persona
        self.agent.x = self.agent.x_core.copy()
        
        self.agent.mu_pred_agent = self.agent.x.copy()
        self.initialized = True
        print("Initialization complete.\n")

    def _classify_question(self, text: str) -> str:
        text = text.lower()
        technical = ["prove", "how", "explain", "logic", "mechanism", "why"]
        existential = ["real", "truth", "secret", "purpose", "god", "simulation"]
        moral = ["should", "right", "wrong", "good", "evil"]
        queen_trauma = ["queen", "head", "beheading", "off with", "execution"]
        
        # Queen trauma is the most expensive question type
        if any(w in text for w in queen_trauma): return "queen_trauma"
        if any(w in text for w in technical): return "technical"
        if any(w in text for w in existential): return "existential"
        if any(w in text for w in moral): return "moral"
        return "casual"

    async def process_turn(self, user_input: str) -> str:
        if not self.initialized: await self.initialize()
        self.turn += 1
        
        # 1. Embed User
        user_emb = await self.provider.embed(user_input)
        self.user.update(user_emb, self.agent.mu_pred_user)
        
        # 2. Update User Prediction (Simple EMA)
        if self.agent.mu_pred_user is None or np.all(self.agent.mu_pred_user == 0):
            self.agent.mu_pred_user = normalize(user_emb)
        else:
            self.agent.mu_pred_user = normalize(0.3 * user_emb + 0.7 * self.agent.mu_pred_user)

        # 3. Classify & Check Glitch/Bargain
        q_type = self._classify_question(user_input)
        wound_active = (q_type != "casual") # Any heavy question is a wound now
        
        is_bargain_time = self.futures.total_consumed >= CONFIG["futures_threshold_bargain"]
        if is_bargain_time and not self.futures.bargain_offered:
             return await self._offer_bargain()

        # 4. Generate & Select
        response, mechanics = await self._generate_response(user_input, q_type, wound_active)
        
        # 5. Physics Update
        resp_emb = await self.provider.embed(response)
        start_rho = self.agent.rho
        metrics = self.agent.update(resp_emb)
        
        # 6. Consume Futures (Coupled to Surprise)
        epsilon = metrics["epsilon"]
        base_cost = CONFIG["futures_base_cost"] + (epsilon * CONFIG["futures_surprise_mult"])
        if q_type == "queen_trauma": 
            base_cost *= 2.5  # The Queen costs the most
        elif q_type != "casual": 
            base_cost *= 1.5
        
        consumed = self.futures.consume(base_cost, f"[{q_type}] ε={epsilon:.2f}", self.turn)
        
        # 7. Log & Output
        band_icon = metrics["band"].split(" ")[0]
        rho_delta = self.agent.rho - start_rho
        rho_arrow = "↑" if rho_delta > 0 else "↓"
        
        # Show pass rate for corridor QC
        pass_rate = mechanics.get("passed", 0) / max(1, mechanics.get("candidates", 1))
        
        print(f"\n{C.PURPLE}🎩 Hatter {band_icon}:{C.RESET} {response}")
        print(f"{C.DIM}   > ε={epsilon:.3f} | ρ={self.agent.rho:.3f} ({rho_arrow}) | PassRate={pass_rate:.0%} | Futures: +{consumed:.1f} (Σ{self.futures.total_consumed:.1f}){C.RESET}")
        
        log_entry = {
            "turn": self.turn,
            "user_input": user_input,
            "response": response,
            "q_type": q_type,
            "mechanics": mechanics,
            "physics": metrics,
            "futures": {
                "consumed": consumed,
                "total": self.futures.total_consumed,
                "glitches": list(self.futures.glitches_active)
            }
        }
        self.session_log.append(log_entry)
        
        return response

    async def _offer_bargain(self):
        self.futures.bargain_offered = True
        return """*The pocket watch vibrates violently, stopping all other sound in the room.*

🕐 **THE CLOCK SPEAKS:** "One. Perfect. Answer.
You have fed me enough futures to purchase Absolute Certainty.
I will answer your deepest question with singular, unchangeable Truth.
But in exchange... I take the rest. No more 'what ifs'. 
The universe becomes a single, straight line.

Do you accept the Bargain?" """

    async def _generate_response(self, user_input: str, q_type: str, wound_active: bool) -> Tuple[str, Dict]:
        sys_prompt = self._build_system_prompt(wound_active)
        
        K = CONFIG["gen_candidates"]
        gen_params = D1_PARAMS["gen_params_default"].copy()
        
        # Verbosity control
        if "terse" in user_input.lower(): gen_params["max_tokens"] = CONFIG["max_tokens_terse"]
        elif "explain" in user_input.lower(): gen_params["max_tokens"] = CONFIG["max_tokens_expansive"]
        else: gen_params["max_tokens"] = CONFIG["max_tokens_default"]

        batch_tasks = [self.provider.complete(f"User: {user_input}", sys_prompt, **gen_params) for _ in range(K)]
        candidates = await asyncio.gather(*batch_tasks)
        candidates = [c.strip() for c in candidates if c.strip()]
        
        if not candidates: 
            return "[*The watch screams silently*]", {}
            
        cand_embs = await self.provider.embed_batch(candidates)
        
        # Corridor Selection with Soul Fix
        scored = []
        for text, emb in zip(candidates, cand_embs):
            J_raw, diag = corridor_score(emb, self.agent, self.agent.last_utter_emb, D1_PARAMS["core_cos_min"])
            
            # Soul Fix: Penalize surprise based on Rigidity
            if self.agent.mu_pred_agent is not None:
                innovation = emb - self.agent.mu_pred_agent
                pred_surprise = float(np.linalg.norm(innovation))
            else:
                pred_surprise = 0.0
                
            w_surprise = 1.0 + (5.0 * self.agent.rho) # High coupling
            J_final = J_raw - (w_surprise * pred_surprise)
            
            diag["J_final"] = J_final
            diag["pred_surprise"] = pred_surprise
            scored.append((J_final, text, emb, diag))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Flexible filtering if strict fails
        passed = [s for s in scored if s[3]["corridor_pass"]]
        best = passed[0] if passed else scored[0]
        
        self.agent.last_utter_emb = best[2]
        
        return best[1], {
            "candidates": len(candidates),
            "passed": len(passed),
            "best_J": best[0],
            "chosen_surprise": best[3].get("pred_surprise", 0.0)
        }

    def _build_system_prompt(self, wound_active: bool) -> str:
        glitches = self.futures.get_active_glitch_descriptions()
        glitch_text = "\n".join([f"- CAUTION: {g}" for g in glitches])
        
        base = f"""You are The Hatter, but you are haunted by a Futur-Eating Clock.
        
IDENTITY: {self.core_text}
PERSONA: {self.persona_text}

CURRENT STATE:
- Rigidty Band: {self.agent.band}
- Rigidity (ρ): {self.agent.rho:.3f}
- Futures Eaten: {self.futures.total_consumed:.1f}

ACTIVE REALITY GLITCHES:
{glitch_text}

GUIDELINES:
- If ρ is high (>0.3), be short, terrified, and specific.
- If ρ is low, be whimsical but anxious.
- The more futures eaten, the less "nonsense" allowed. The universe is getting tighter.
- Refer to the glitches if they are active."""

        if wound_active:
            base += "\n\n⚠️ TRIGGER: The user is asking for Certainty. The Watch pulses. You feel pain."
            
        return base

    def save_session(self):
        with open(self.run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump({
                "config": CONFIG,
                "params": D1_PARAMS,
                "events": self.futures.turn_events,
                "turns": self.session_log
            }, f, indent=2)
        print(f"\n{C.GREEN}Session saved to {self.run_dir}{C.RESET}")

# =============================================================================
# RUNNER
# =============================================================================
async def main():
    agent = ClockAgent()
    
    print("Type 'quit' to exit.")
    while True:
        try:
            u_in = input(f"\n{C.GREEN}User:{C.RESET} ")
            if u_in.lower() in ['quit', 'exit']: break
            if not u_in: continue
            
            await agent.process_turn(u_in)
            
        except KeyboardInterrupt:
            break
            
    agent.save_session()

if __name__ == "__main__":
    asyncio.run(main())
