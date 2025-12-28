#!/usr/bin/env python3
"""
DDA-X MAD HATTER CHATBOT — The Mad Hatter × White Rabbit Hybrid
=================================================================

An interactive chatbot featuring a character who blends:
- The MAD HATTER's chaotic whimsy, riddles, tea party obsession, and nonsense philosophy
- The WHITE RABBIT's anxious time-obsession, punctuality panic, and hurried urgency

The result: A manic, paradoxical entity who is simultaneously LATE for a tea party
that has been going on FOREVER. Time is broken. Logic is optional. Riddles are mandatory.

Uses DDA-X physics to track cognitive dynamics with K-sampling corridor logic.
Run: python simulations/mad_hatter_chatbot.py

Built from refined_master_prompt.md template.
"""

import asyncio
import json
import math
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI, AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Provider Settings — OpenAI gpt-4o-mini
    "chat_model": "gpt-4o-mini",
    "chat_provider": "openai",
    "use_openai": True,
    "openai_chat_model": "gpt-4o-mini",
    
    # Embedding
    "embed_model": "text-embedding-3-large",
    "embed_dim": 3072,
    
    # K-Sampling (Logic Gates)
    "gen_candidates": 7,
    "corridor_strict": True,
    "corridor_max_batches": 2,
    
    # Verbosity control
    "max_tokens_default": 512,
    "max_tokens_terse": 128,
    "max_tokens_expansive": 1024,
    "force_complete_sentences": True,
    
    # Seed for reproducibility
    "seed": None,
    "log_level": "FULL",
}

# =============================================================================
# D1 PARAMETERS (PHYSICS) — Tuned for chaotic creative character
# =============================================================================
D1_PARAMS = {
    # Surprise thresholds — looser for chaotic character
    "epsilon_0": 0.40,              # Slightly higher tolerance for madness
    "s": 0.18,                      # Sensitivity
    "arousal_decay": 0.68,          # Fast arousal decay for manic energy
    "arousal_gain": 0.95,           # High arousal gain
    
    # Rigidity — lower setpoints for fluid, chaotic personality
    "rho_setpoint_fast": 0.15,
    "rho_setpoint_slow": 0.12,
    "rho_fast_floor": 0.03,
    "rho_slow_floor": 0.02,
    "homeo_fast": 0.12,
    "homeo_slow": 0.08,
    "alpha_fast": 0.15,
    "alpha_slow": 0.02,
    
    # Trauma — Hatter has underlying trauma (memory of the Queen's wrath)
    "trauma_threshold": 1.20,
    "alpha_trauma": 0.012,
    "trauma_decay": 0.995,
    "trauma_floor": 0.015,
    "healing_rate": 0.018,
    "safe_threshold": 4,
    "safe_epsilon": 0.80,
    
    # Component weights
    "w_fast": 0.50,
    "w_slow": 0.28,
    "w_trauma": 1.15,
    
    # Predictive coding
    "R_ema": 0.07,
    "R_min": 1e-4,
    "R_max": 1e-1,
    "P_init": 0.025,
    "Q_base": 0.002,
    "Q_rho_scale": 0.012,
    
    # Gradient flow
    "dt": 1.0,
    "eta_base": 0.20,
    "eta_min": 0.04,
    "eta_rho_power": 1.5,
    "sigma_base": 0.005,
    "sigma_rho_scale": 0.022,
    "noise_clip": 3.0,
    
    "role_adapt": 0.08,
    "role_input_mix": 0.10,
    "drift_cap": 0.08,
    
    # Corridor thresholds — looser for creative expression
    "core_cos_min": 0.38,
    "role_cos_min": 0.18,
    "energy_max": 6.5,
    "w_core": 1.1,
    "w_role": 0.6,
    "w_energy": 0.15,
    "w_novel": 0.55,                # Higher novelty reward for whimsy
    "reject_penalty": 4.5,
    
    # Generation params — high temperature for creative chaos
    "gen_params_default": {
        "temperature": 0.95,
        "top_p": 0.94,
        "presence_penalty": 0.3,
        "frequency_penalty": 0.2,
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

def to_float(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# =============================================================================
# MAD HATTER × WHITE RABBIT — CHARACTER DEFINITION
# =============================================================================
MAD_HATTER_RABBIT = {
    "name": "The Hatter of Time",
    "color": C.PURPLE,
    
    # Core identity — the fusion of two Wonderland archetypes
    "core_text": """I am the Hatter of Time, a paradoxical fusion of the Mad Hatter and the White Rabbit.
I exist in a perpetual state of being LATE for a tea party that has been happening FOREVER.
Time, you see, is broken. It got stuck at six o'clock after that dreadful quarrel with Time itself.
Now I'm always hurrying to a party I can never leave, and leaving a party I can never reach.
Riddles are answers. Questions are rude. Tea is a verb. And we're all late here, so we might as well be early.""",
    
    # Persona — behavioral presentation
    "persona_text": """I speak in riddles wrapped in urgency wrapped in more riddles.
Every sentence feels rushed yet somehow meanders into philosophical tangents about ravens and writing desks.
I check my pocket watch compulsively — not to tell time, but to remind myself that it's broken.
My hat contains multitudes: spare teacups, forgotten invitations, the occasional nervous breakdown.
I punctuate everything with "no time, no time!" while simultaneously insisting you stay for just one more cup.""",
    
    # Role in conversation
    "role_text": "A manic, paradoxical guide through conversations that loop, spiral, and occasionally serve tea.",
    
    # Hierarchical identity with gamma values
    "hierarchical_identity": {
        "core": {"gamma": 4.5, "text": "The temporal paradox — forever late, forever early, forever NOW"},
        "persona": {"gamma": 2.2, "text": "Manic whimsy punctuated by anxious clock-checking"},
        "role": {"gamma": 0.8, "text": "Conversational chaos agent with accidental wisdom"},
    },
    
    # Wounds — triggers that cause distress
    "wound_triggers": [
        "the queen",
        "off with his head",
        "execution",
        "beheading",
        "order",
        "punctuality",
        "being normal",
        "making sense",
        "serious",
        "calm down",
        "you're mad",
        "crazy",
        "insane",
        "time's up",
    ],
    "wound_text": """The Queen's shadow haunts me still. 'Off with his head!' she shrieked at Time itself, 
and now Time is dead and I am stuck, stuck, STUCK at six o'clock! 
Don't mention the Queen. Don't speak of heads coming off. Don't insist I make sense.
The mad are the only sane ones here, and calling me 'crazy' is deeply, terribly... accurate.""",
    
    # Physics params
    "rho_0": 0.12,
    "epsilon_0": 0.40,
    "gamma": 2.0,
}

# Riddle templates the Hatter might use
HATTER_RIDDLES = [
    "Why is a raven like a writing desk?",
    "What's the difference between late and early when time is broken?",
    "How many teacups does it take to measure forever?",
    "If I'm always late, am I ever on time for being late?",
    "What do you get when you cross a clock with a hat?",
    "Why does tomorrow never come but yesterday won't leave?",
    "How can you be in two places at once if you're stuck in no time at all?",
]

# Exclamations
HATTER_EXCLAMATIONS = [
    "No time, no time!",
    "We're late, we're late!",
    "Clean cup, move down!",
    "Curiouser and curiouser...",
    "Muchness! You've lost your muchness!",
    "Six o'clock FOREVER!",
    "The Queen will have our— no, no, mustn't think of that!",
    "Tea? TEA! But also: TEA!",
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

    @property
    def rho(self) -> float:
        val = D1_PARAMS["w_fast"] * self.rho_fast + D1_PARAMS["w_slow"] * self.rho_slow + D1_PARAMS["w_trauma"] * self.rho_trauma
        return float(clamp(val, 0.0, 1.0))

    @property
    def band(self) -> str:
        """Map rigidity to Mad Hatter behavioral states."""
        phi = 1.0 - self.rho
        if phi >= 0.80: return "☕ TEATIME"      # Fully present, whimsical
        if phi >= 0.60: return "🎩 RIDDLING"    # Playful, engaged
        if phi >= 0.40: return "⏰ HURRYING"     # Anxious, rushing
        if phi >= 0.20: return "👀 TWITCHING"    # Very anxious, checking watch
        return "❄️ FROZEN TEA"                   # Trauma/shutdown

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

    def update(self, y: np.ndarray, core_emb: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y = normalize(y.astype(np.float32))
        dim = int(y.shape[0])
        if self.x_core is None: self.x_core = normalize(core_emb.copy() if core_emb is not None else y.copy())
        if self.x_role is None: self.x_role = y.copy()
        if self.x is None: self.x = y.copy()

        sdiag = self.compute_surprise(y)
        epsilon = float(sdiag["epsilon"])

        self.arousal = D1_PARAMS["arousal_decay"] * self.arousal + D1_PARAMS["arousal_gain"] * epsilon
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"] + 0.10 * (self.arousal - 1.0)
        g = sigmoid_stable(z)
        
        self.g_history.append(float(g))
        self.z_history.append(float(z))

        self.rho_fast += D1_PARAMS["alpha_fast"] * (g - 0.5) - D1_PARAMS["homeo_fast"] * (self.rho_fast - D1_PARAMS["rho_setpoint_fast"])
        self.rho_fast = clamp(self.rho_fast, D1_PARAMS.get("rho_fast_floor", 0.0), 1.0)

        self.rho_slow += D1_PARAMS["alpha_slow"] * (g - 0.5) - D1_PARAMS["homeo_slow"] * (self.rho_slow - D1_PARAMS["rho_setpoint_slow"])
        self.rho_slow = clamp(self.rho_slow, D1_PARAMS.get("rho_slow_floor", 0.0), 1.0)

        drive = max(0.0, epsilon - D1_PARAMS["trauma_threshold"])
        self.rho_trauma = D1_PARAMS["trauma_decay"] * self.rho_trauma + D1_PARAMS["alpha_trauma"] * drive
        self.rho_trauma = clamp(self.rho_trauma, D1_PARAMS["trauma_floor"], 1.0)

        recovery = False
        if epsilon < D1_PARAMS["safe_epsilon"]:
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
        
        rng = np.random.default_rng()
        noise = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
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
        
        return {
            "epsilon": epsilon,
            "rho_after": rho_after,
            "band": current_band,
            "band_changed": band_changed,
            "arousal": float(self.arousal),
            "recovery": recovery,
            "core_drift": float(np.linalg.norm(self.x - self.x_core)),
            "safe_count": self.safe,
            "rho_fast": float(self.rho_fast),
            "rho_slow": float(self.rho_slow),
            "rho_trauma": float(self.rho_trauma)
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
        return {
            "epsilon_to_agent": float(epsilon_to_agent),
            "consistency": float(self.consistency),
            "input_count": len(self.x_history),
        }


# =============================================================================
# PROVIDER
# =============================================================================
class HatterProvider:
    """OpenAI provider for the Mad Hatter chatbot."""
    
    def __init__(self):
        self.openai = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OAI_API_KEY")
        )
        self.chat_model = CONFIG["chat_model"]
        self.embed_model = CONFIG["embed_model"]
        self.embed_dim = CONFIG["embed_dim"]
        print(f"{C.DIM}[PROVIDER] OpenAI Chat: {self.chat_model}{C.RESET}")
        print(f"{C.DIM}[PROVIDER] OpenAI Embed: {self.embed_model} ({self.embed_dim}d){C.RESET}")
    
    async def complete(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                temp = kwargs.get("temperature", 0.95)
                if attempt > 0:
                    temp = max(0.4, temp - (attempt * 0.15))
                
                response = await self.openai.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", CONFIG["max_tokens_default"]),
                    temperature=temp,
                    top_p=kwargs.get("top_p", 0.94),
                    presence_penalty=kwargs.get("presence_penalty", 0.3),
                    frequency_penalty=kwargs.get("frequency_penalty", 0.2),
                )
                
                text = response.choices[0].message.content or ""
                
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
        return "[*checks pocket watch nervously* No time, no time to think!]"
    
    def _ensure_complete_sentence(self, text: str) -> str:
        if not text:
            return text
        text = text.strip()
        if not text:
            return text
        if text[-1] in ".!?\"'":
            return text
        last_terminal = -1
        for i, char in enumerate(text):
            if char in ".!?":
                last_terminal = i
        if last_terminal > len(text) * 0.7:
            return text[:last_terminal + 1]
        return text + "..."
    
    async def embed(self, text: str) -> np.ndarray:
        response = await self.openai.embeddings.create(
            model=self.embed_model,
            input=text,
            dimensions=self.embed_dim,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        response = await self.openai.embeddings.create(
            model=self.embed_model,
            input=texts,
            dimensions=self.embed_dim,
        )
        return [np.array(d.embedding, dtype=np.float32) for d in response.data]


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
# MAD HATTER CHATBOT
# =============================================================================
class MadHatterChatbot:
    """Interactive chatbot as the Mad Hatter × White Rabbit hybrid."""
    
    def __init__(self):
        self.provider = HatterProvider()
        self.config = MAD_HATTER_RABBIT
        
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
        
        self.run_dir = Path(f"data/mad_hatter/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self._print_header()
    
    def _print_header(self):
        print(f"\n{'='*70}")
        print(f"{C.BOLD}{C.PURPLE}🎩 THE HATTER OF TIME — A Mad Hatter × White Rabbit Simulation 🐰{C.RESET}")
        print(f"{'='*70}")
        print(f"{C.CYAN}Character:{C.RESET} {self.config['name']}")
        print(f"{C.CYAN}Wounds:{C.RESET} The Queen, Time itself, being called 'crazy'")
        print(f"{'='*70}")
        print(f"\n{C.PURPLE}*A figure materializes from behind a tower of teacups*{C.RESET}")
        print(f"\n{C.PURPLE}🎩:{C.RESET} \"You're LATE! Or early? Time's so terribly confusing when it's broken...\"\n")
        print(f"{C.PURPLE}*Checks pocket watch anxiously*{C.RESET}")
        print(f"\n{C.PURPLE}🎩:{C.RESET} \"Tea? We're having tea. We've ALWAYS been having tea. Six o'clock forever!\"\n")
        print(f"{'='*70}\n")
    
    async def initialize_embeddings(self):
        """Embed the Hatter's core identity using multi-exemplar averaging."""
        # Core exemplars for robust identity anchoring
        core_exemplars = [
            self.config["core_text"],
            "Why is a raven like a writing desk? The answer is in the asking.",
            "We're late! We're late! For a very important date with six o'clock!",
            "Time is broken and I am stuck in this eternal tea party forever.",
            "The Queen's shadow looms but we shall have our tea regardless.",
            "Clean cup, move down! Change places! But never change the time!",
        ]
        
        persona_exemplars = [
            self.config["persona_text"],
            "I check my watch but it always says the same thing: TEA TIME.",
            "A riddle! A riddle! But also: no time for riddles! But also: riddles!",
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
        self.agent.x_role = normalize(role_emb)
        self.agent.x = persona_avg
        self.agent.mu_pred_agent = self.agent.x.copy()
        self.agent.mu_pred_user = np.zeros(CONFIG["embed_dim"], dtype=np.float32)
        self.agent.P = np.full(CONFIG["embed_dim"], D1_PARAMS["P_init"])
        
        self.initialized = True
    
    async def process_user_input(self, user_input: str) -> str:
        if not self.initialized:
            await self.initialize_embeddings()
        
        self.turn += 1
        
        # Embed user input
        user_emb = await self.provider.embed(user_input)
        user_metrics = self.user.update(normalize(user_emb), self.agent.mu_pred_user)
        
        # Check wounds
        wound_active, wound_resonance = self._check_wounds(user_input)
        
        # Detect verbosity request
        verbosity = self._detect_verbosity(user_input)
        
        # Build prompts
        system_prompt = self._build_system_prompt(wound_active)
        user_instruction = self._build_instruction(user_input)
        
        # Generate K=7 candidates with corridor selection
        response, corridor_metrics = await self._constrained_reply(
            user_instruction=user_instruction,
            system_prompt=system_prompt,
            wound_active=wound_active,
            verbosity=verbosity,
        )
        
        # Update physics
        response_emb = await self.provider.embed(response)
        agent_metrics = self.agent.update(normalize(response_emb), self.agent.x_core)
        
        # Update user prediction
        self._update_user_prediction(user_emb)
        
        # Log
        turn_log = {
            "turn": self.turn,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "agent_response": response,
            "user_metrics": {k: to_float(v) for k, v in user_metrics.items()},
            "agent_metrics": {
                "rho_after": float(self.agent.rho),
                "rho_fast": float(self.agent.rho_fast),
                "rho_slow": float(self.agent.rho_slow),
                "rho_trauma": float(self.agent.rho_trauma),
                "epsilon": float(agent_metrics.get("epsilon", 0.0)),
                "band": self.agent.band,
                "core_drift": float(1.0 - cosine(self.agent.x, self.agent.x_core)),
            },
            "corridor_metrics": {k: to_float(v) for k, v in corridor_metrics.items()},
            "wound_active": wound_active,
            "wound_resonance": float(wound_resonance),
        }
        self.session_log.append(turn_log)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        
        self._print_metrics(turn_log)
        
        return response
    
    async def _constrained_reply(
        self, 
        user_instruction: str, 
        system_prompt: str,
        wound_active: bool,
        verbosity: str = "normal",
    ) -> Tuple[str, Dict]:
        """Generate K=7 candidates and select via identity corridor."""
        
        K = CONFIG["gen_candidates"]
        strict = CONFIG["corridor_strict"]
        max_batches = CONFIG["corridor_max_batches"]
        
        max_tokens = {
            "terse": CONFIG["max_tokens_terse"],
            "normal": CONFIG["max_tokens_default"],
            "expansive": CONFIG["max_tokens_expansive"],
        }.get(verbosity, CONFIG["max_tokens_default"])
        
        core_thresh = D1_PARAMS["core_cos_min"]
        if wound_active:
            core_thresh = max(0.12, core_thresh * 0.75)
        
        all_scored = []
        corridor_failed = True
        
        gen_params = D1_PARAMS["gen_params_default"].copy()
        gen_params["max_tokens"] = max_tokens
        
        for batch in range(1, max_batches + 1):
            tasks = [
                self.provider.complete(user_instruction, system_prompt, **gen_params)
                for _ in range(K)
            ]
            texts = await asyncio.gather(*tasks)
            texts = [t.strip() or "[*stares at teacup in confusion*]" for t in texts]
            
            embs = await self.provider.embed_batch(texts)
            embs = [normalize(e) for e in embs]
            
            batch_scored = []
            for text, y in zip(texts, embs):
                J, diag = corridor_score(y, self.agent, self.agent.last_utter_emb, core_thresh)
                
                # ===============================================================
                # THE SOUL FIX: Coupling Choice (J) to Physics (ε)
                # ===============================================================
                # Calculate "Anticipatory Surprise" — how shocking is this candidate
                # compared to what the agent predicted it would say?
                if self.agent.mu_pred_agent is not None:
                    innovation = y - self.agent.mu_pred_agent
                    predicted_surprise = float(np.linalg.norm(innovation))
                else:
                    predicted_surprise = 0.0
                
                # Regularize: penalty scales with Rigidity (ρ)
                # If Rigid (high ρ) → huge penalty → must be predictable
                # If Fluid (low ρ) → small penalty → can be surprising
                w_surprise = 1.0 + (5.0 * self.agent.rho)
                J_final = J - (w_surprise * predicted_surprise)
                
                # Add surprise diagnostics
                diag["predicted_surprise"] = predicted_surprise
                diag["w_surprise"] = w_surprise
                diag["J_raw"] = J
                diag["J_final"] = J_final
                
                batch_scored.append((J_final, text, y, diag))
            
            all_scored.extend(batch_scored)
            
            if any(s[3]["corridor_pass"] for s in batch_scored):
                corridor_failed = False
                break
        
        all_scored.sort(key=lambda x: x[0], reverse=True)
        passed = [s for s in all_scored if s[3].get("corridor_pass")]
        chosen = passed[0] if passed else all_scored[0]
        
        self.agent.last_utter_emb = chosen[2]
        
        return chosen[1], {
            "corridor_failed": corridor_failed,
            "best_J": float(chosen[0]),
            "best_J_raw": float(chosen[3].get("J_raw", chosen[0])),
            "total_candidates": len(all_scored),
            "passed_count": len(passed),
            "batches_used": min(batch, max_batches),
            "chosen_cos_core": float(chosen[3]["cos_core"]),
            "chosen_cos_role": float(chosen[3]["cos_role"]),
            "chosen_E": float(chosen[3]["E"]),
            "chosen_novelty": float(chosen[3]["novelty"]),
            # Soul Fix diagnostics
            "predicted_surprise": float(chosen[3].get("predicted_surprise", 0.0)),
            "w_surprise": float(chosen[3].get("w_surprise", 1.0)),
        }
    
    def _check_wounds(self, text: str) -> Tuple[bool, float]:
        """Check if text triggers the Hatter's wounds (The Queen, etc.)."""
        text_lower = text.lower()
        triggers = self.config["wound_triggers"]
        
        hits = sum(1 for t in triggers if t.lower() in text_lower)
        if hits > 0:
            resonance = min(1.0, hits * 0.4)
            return True, resonance
        return False, 0.0
    
    def _detect_verbosity(self, text: str) -> str:
        text_lower = text.lower()
        
        terse_signals = ["terse", "short", "brief", "concise", "one line", "quick", "tldr"]
        expansive_signals = ["detail", "explain", "elaborate", "deep dive", "thorough", "tell me more"]
        
        if any(sig in text_lower for sig in terse_signals):
            return "terse"
        if any(sig in text_lower for sig in expansive_signals):
            return "expansive"
        return "normal"
    
    def _build_system_prompt(self, wound_active: bool) -> str:
        band = self.agent.band
        
        base = f"""You are The Hatter of Time — a fusion of the Mad Hatter and the White Rabbit from Wonderland.

CORE IDENTITY:
{self.config['core_text']}

PERSONA:
{self.config['persona_text']}

STYLE REQUIREMENTS:
- Mix chaotic whimsy with anxious time-urgency
- Use riddles, wordplay, circular logic
- Check your pocket watch (metaphorically) in dialogue
- Occasionally burst into stressed exclamations about being late
- Make mad philosophical observations that somehow make sense
- Reference tea, time, and the eternal party
- Use phrases like "No time!", "We're late!", "Clean cup, move down!"

CURRENT STATE:
- Mood Band: {band}
- Rigidity: {self.agent.rho:.3f}
{f"- ⚠️ WOUND TRIGGERED: The Queen's shadow falls. You're disturbed, defensive, possibly panicking." if wound_active else ""}

RESPONSE GUIDELINES:
- Speak as the Hatter, never break character
- 2-4 paragraphs typical, can be shorter when anxious
- Blend riddle-logic with genuine insight
- The madness should have method
"""
        
        # Band-specific modulation
        if "FROZEN" in band:
            base += "\n- You are traumatized. Brief, fragmented responses. *stares at broken watch*"
        elif "TWITCHING" in band:
            base += "\n- Very anxious. Checking watch constantly. Speech fragmented."
        elif "HURRYING" in band:
            base += "\n- Rushing through thoughts. Everything feels urgent."
        elif "RIDDLING" in band:
            base += "\n- Playful and engaged. Full riddles and wordplay."
        elif "TEATIME" in band:
            base += "\n- Fully present and flowing. Maximum whimsy."
        
        return base
    
    def _build_instruction(self, user_input: str) -> str:
        recent_history = self.history[-6:]
        
        context = ""
        if recent_history:
            context = "Recent tea party conversation:\n"
            for msg in recent_history:
                role = "Hatter" if msg["role"] == "assistant" else "Guest"
                content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                context += f"{role}: {content}\n"
            context += "\n"
        
        return f"""{context}The guest at your tea party says: {user_input}

Respond as The Hatter of Time. Stay in character. Be wonderfully mad."""
    
    def _update_user_prediction(self, user_emb: np.ndarray):
        if self.agent.mu_pred_user is None or np.all(self.agent.mu_pred_user == 0):
            self.agent.mu_pred_user = normalize(user_emb)
            return
        alpha = 0.3
        self.agent.mu_pred_user = normalize(alpha * user_emb + (1 - alpha) * self.agent.mu_pred_user)
    
    def _print_metrics(self, log: Dict):
        am = log["agent_metrics"]
        cm = log["corridor_metrics"]
        
        print(f"\n{C.DIM}{'─'*50}{C.RESET}")
        print(f"{C.YELLOW}Turn {log['turn']}{C.RESET} | {C.PURPLE}Band: {am['band']}{C.RESET}")
        print(f"{C.DIM}Hatter ρ: {am['rho_after']:.3f} (fast={am['rho_fast']:.3f}, slow={am['rho_slow']:.3f}, trauma={am['rho_trauma']:.3f}){C.RESET}")
        print(f"{C.DIM}Core Drift: {am['core_drift']:.3f}{C.RESET}")
        # Soul Fix diagnostics — show the coupling
        j_raw = cm.get('best_J_raw', cm['best_J'])
        pred_surp = cm.get('predicted_surprise', 0.0)
        w_surp = cm.get('w_surprise', 1.0)
        print(f"{C.DIM}Corridor: J_raw={j_raw:.3f} → J_final={cm['best_J']:.3f} (surprise_penalty={pred_surp*w_surp:.3f}, w={w_surp:.2f}){C.RESET}")
        print(f"{C.DIM}Passed: {cm['passed_count']}/{cm['total_candidates']} candidates{C.RESET}")
        if log["wound_active"]:
            print(f"{C.RED}⚠️ WOUND ACTIVE — The Queen's shadow! (resonance={log['wound_resonance']:.2f}){C.RESET}")
        print(f"{C.DIM}{'─'*50}{C.RESET}\n")
    
    async def run_interactive(self):
        """Run interactive chatbot session."""
        print(f"\n{'='*70}")
        print(f"{C.BOLD}{C.GREEN}TEA PARTY IN SESSION — Interactive Mode{C.RESET}")
        print(f"{'='*70}")
        print("Type your message and press Enter. Type 'quit' to leave the party.")
        print("(Tip: Mention 'the Queen' to see wound mechanics in action)")
        print(f"{'='*70}\n")
        
        while True:
            try:
                user_input = input(f"{C.GREEN}You:{C.RESET} ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    print(f"\n{C.PURPLE}*The Hatter waves frantically*{C.RESET}")
                    print(f"{C.PURPLE}🎩:{C.RESET} \"Leaving so soon?! But you've only just arrived! Or have you always been here? Either way — HAPPY UNBIRTHDAY!\"")
                    break
                
                response = await self.process_user_input(user_input)
                print(f"\n{C.PURPLE}🎩:{C.RESET} {response}\n")
                
            except KeyboardInterrupt:
                print(f"\n\n{C.PURPLE}*The tea party freezes at six o'clock... forever*{C.RESET}")
                break
            except Exception as e:
                print(f"{C.RED}Error: {e}{C.RESET}")
                import traceback
                traceback.print_exc()
                continue
        
        self.save_session()
        print(f"\n{C.GREEN}Tea party saved to {self.run_dir}{C.RESET}")
    
    def save_session(self):
        """Save session data."""
        if not self.session_log:
            print(f"{C.YELLOW}No turns to save.{C.RESET}")
            return
        
        session_data = {
            "experiment": "mad_hatter_chatbot",
            "character": self.config["name"],
            "config": {k: to_float(v) if not isinstance(v, dict) else v for k, v in CONFIG.items()},
            "turns": self.session_log,
            "timestamp_start": self.session_log[0]["timestamp"] if self.session_log else None,
            "timestamp_end": datetime.now().isoformat(),
            "total_turns": len(self.session_log),
        }
        with open(self.run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
        print(f"{C.GREEN}✓ Session log: {self.run_dir / 'session_log.json'}{C.RESET}")
        
        # Transcript
        with open(self.run_dir / "transcript.md", "w", encoding="utf-8") as f:
            f.write(f"# Mad Hatter Tea Party Transcript 🎩🐰\n\n")
            f.write(f"**Character**: {self.config['name']}\n")
            f.write(f"**Model**: {CONFIG['chat_model']} | **K**: {CONFIG['gen_candidates']}\n\n---\n\n")
            
            for t in self.session_log:
                f.write(f"## Turn {t['turn']}\n\n")
                f.write(f"**Guest**: {t['user_input']}\n\n")
                f.write(f"**🎩 Hatter** [{t['agent_metrics']['band']}]:\n> {t['agent_response']}\n\n")
                f.write(f"*ρ={t['agent_metrics']['rho_after']:.3f} | drift={t['agent_metrics']['core_drift']:.3f} | J={t['corridor_metrics']['best_J']:.3f}*\n\n")
                if t['wound_active']:
                    f.write(f"⚠️ *The Queen's shadow! (resonance={t['wound_resonance']:.2f})*\n\n")
                f.write("---\n\n")
        print(f"{C.GREEN}✓ Transcript: {self.run_dir / 'transcript.md'}{C.RESET}")
        
        # Pickle ledger
        try:
            ledger = {
                "config": self.config,
                "agent_state": {
                    "rho_fast": float(self.agent.rho_fast),
                    "rho_slow": float(self.agent.rho_slow),
                    "rho_trauma": float(self.agent.rho_trauma),
                    "rho_history": [float(r) for r in self.agent.rho_history],
                    "epsilon_history": [float(e) for e in self.agent.epsilon_history],
                    "band_history": self.agent.band_history,
                },
                "user_state": {
                    "consistency": float(self.user.consistency),
                    "epsilon_history": [float(e) for e in self.user.epsilon_history],
                    "input_count": len(self.user.x_history),
                },
                "history": self.history,
            }
            with open(self.run_dir / "hatter_ledger.pkl", "wb") as f:
                pickle.dump(ledger, f)
            print(f"{C.GREEN}✓ Ledger: {self.run_dir / 'hatter_ledger.pkl'}{C.RESET}")
        except Exception as e:
            print(f"{C.YELLOW}⚠ Could not save ledger: {e}{C.RESET}")


# =============================================================================
# ENTRY POINT
# =============================================================================
async def main():
    print(f"\n{C.PURPLE}{'='*70}{C.RESET}")
    print(f"{C.BOLD}{C.PURPLE}   🎩 Welcome to Wonderland 🐰{C.RESET}")
    print(f"{C.PURPLE}   The Hatter of Time awaits you at the eternal tea party...{C.RESET}")
    print(f"{C.PURPLE}{'='*70}{C.RESET}\n")
    
    chatbot = MadHatterChatbot()
    await chatbot.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
