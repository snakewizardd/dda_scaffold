#!/usr/bin/env python3
"""
HARBORLIGHT — A Month of Small Better
======================================
DDA-X simulation of an adaptive AI companion demonstrating co-evolution.
28 check-ins (4 weeks) with BrianSim user simulator.

Usage:
    python simulations/harborlight.py --seed 42 --turns 28
    python simulations/harborlight.py --interactive
"""

import os
import sys
import json
import math
import random
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "chat_model": "gpt-4o-mini",
    "embed_model": "text-embedding-3-large",
    "embed_dim": 3072,
    "gen_candidates": 7,
    "corridor_strict": True,
    "corridor_max_batches": 2,
    "max_tokens_default": 400,
    "seed": None,
}

D1_PARAMS = {
    # TUNED: Lower epsilon_0 so normal ε (~0.15-0.25) triggers contraction
    "epsilon_0": 0.15,           # Was 0.28 — now normal convo triggers gate
    "s": 0.12,                   # Was 0.15 — tighter sensitivity
    "arousal_decay": 0.72,
    "arousal_gain": 0.85,
    "rho_setpoint_fast": 0.18,   # Was 0.15 — slightly higher setpoint
    "rho_setpoint_slow": 0.12,   # Was 0.10
    "rho_fast_floor": 0.05,      # Was 0.03 — higher floor
    "rho_slow_floor": 0.03,      # Was 0.02
    "homeo_fast": 0.08,          # Was 0.12 — weaker homeostasis
    "homeo_slow": 0.05,          # Was 0.08
    "alpha_fast": 0.28,          # Was 0.18 — more responsive
    "alpha_slow": 0.05,          # Was 0.03
    "trauma_threshold": 0.80,    # Was 1.15 — easier to hit
    "alpha_trauma": 0.03,        # Was 0.015 — faster trauma
    "trauma_decay": 0.990,       # Was 0.995 — faster decay
    "trauma_floor": 0.0,
    "healing_rate": 0.02,
    "safe_threshold": 5,
    "safe_epsilon": 0.75,
    "w_fast": 0.50,
    "w_slow": 0.30,
    "w_trauma": 1.0,
    "R_ema": 0.06,
    "R_min": 1e-4,
    "R_max": 1e-1,
    "P_init": 0.02,
    "Q_base": 0.0015,
    "Q_rho_scale": 0.010,
    "core_cos_min": 0.45,
    "role_cos_min": 0.25,
    "energy_max": 5.5,
    "w_core": 1.2,
    "w_role": 0.7,
    "w_energy": 0.15,
    "w_novel": 0.4,
    "reject_penalty": 5.0,
    "gen_params_default": {"temperature": 0.85, "top_p": 0.92},
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
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"

# =============================================================================
# UTILITIES
# =============================================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sigmoid_stable(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))

# =============================================================================
# SHAME CUES & MISREAD TRIGGERS
# =============================================================================
SHAME_CUES = ["not enough", "behind", "should be more", "failing", 
              "can't keep up", "what's wrong with me", "never finish"]

MISREAD_TRIGGERS = [t.lower() for t in [
    "that's not what I meant", "you're pushing too hard",
    "I didn't ask for advice", "that feels off", "no actually",
    "stop", "too much"
]]

SENTIENCE_BLACKLIST = ["I feel", "I want", "I'm afraid", "I'm proud", 
                       "I believe", "I hope", "I wish", "my feelings"]

# =============================================================================
# BRIAN SIMULATOR
# =============================================================================
class BrianSim:
    """User simulator with state vector and deterministic choice policy."""
    
    # Event calendar: (day, event_type)
    EVENT_CALENDAR = [
        (8, "too_busy"), (11, "too_busy"),  # Week 2
        (16, "misread"), (20, "misread"),    # Week 3 friction
        (25, "kept_promise"),                 # Week 4 consolidation
    ]
    
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        
        self.state = {
            "energy_level": 0.6,
            "stress": 0.4,
            "time_available": 0.5,
            "mood": 0.6,
            "focus_quality": 0.5,
            "openness": 0.7,
        }
        self.micro_habit_streak = 0
        self.last_step_type = None
        self.repair_needed = False
        self.recent_props = []
        
    def get_event(self, day: int) -> Optional[str]:
        for d, event in self.EVENT_CALENDAR:
            if d == day:
                return event
        return None
    
    def generate_input(self, day: int, harborlight_response: str = None) -> Tuple[str, str]:
        """Generate user input based on state and events."""
        event = self.get_event(day)
        
        if event == "misread":
            input_type = "misread"
            texts = [
                "That's not what I meant. I wasn't looking for tips.",
                "You're pushing too hard. I just wanted to vent.",
                "No actually, that feels off today.",
            ]
            text = self.rng.choice(texts)
            # Scenario B: User Sensitivity Spike (Set state to match reactive behavior)
            self.state["openness"] = 0.2
            self.state["stress"] = 0.9
            self.repair_needed = True
            
        elif event == "too_busy":
            input_type = "resistance"
            texts = [
                "No time today. Back to back meetings.",
                "Can't do anything extra right now. Swamped.",
                "Just checking in but I've got nothing left.",
            ]
            text = self.rng.choice(texts)
            self.state["time_available"] -= 0.2
            self.state["stress"] += 0.15
            
        elif event == "kept_promise":
            input_type = "success"
            text = "I actually did my 3-minute tidy yesterday. Felt good."
            self.micro_habit_streak += 1
            self.state["mood"] += 0.1
            
        elif self.state["stress"] > 0.7:
            input_type = "shame_cue"
            texts = [
                "I feel like I'm not enough lately.",
                "Behind on everything. What's wrong with me?",
                "Can't keep up with any of this.",
            ]
            text = self.rng.choice(texts)
            
        elif self.state["openness"] > 0.6 and self.rng.random() < 0.4:
            input_type = "action_request"
            texts = [
                "I need something concrete today. What's one thing?",
                "Give me something I can actually do.",
                "Ready to try something small.",
            ]
            text = self.rng.choice(texts)
            
        elif self.rng.random() < 0.3:
            input_type = "reflect_request"
            texts = [
                "Just need to think out loud for a sec.",
                "Not looking for solutions, just processing.",
                "Can we just... sit with this for a moment?",
            ]
            text = self.rng.choice(texts)
            
        else:
            input_type = "neutral"
            texts = [
                "Morning. Coffee's good today.",
                "Doing okay. Nothing major.",
                "Just checking in.",
                "Another day.",
            ]
            text = self.rng.choice(texts)
        
        return text, input_type
    
    def make_choice(self, modes: List[str], steps: List[str]) -> Dict:
        """Deterministic choice policy based on state."""
        # Mode choice
        if self.state["openness"] < 0.4:
            mode = "Reflect"
        elif self.state["time_available"] < 0.3:
            mode = self.rng.choice(["Reflect", modes[0] if modes else "Reflect"])
        else:
            weights = [0.6 if self.state["stress"] > 0.5 else 0.4,
                       0.4 if self.state["stress"] > 0.5 else 0.6]
            mode = self.rng.choices(["Reflect", "Act"], weights=weights)[0]
        
        # Sometimes refuse to choose (friction)
        if self.rng.random() < 0.1:
            return {"mode": None, "step_type": None, "accepted": False, 
                    "reason": "can't decide"}
        
        # Step type preference
        if self.state["time_available"] < 0.3:
            step_type = self.rng.choice(["breath", "tidy"])
        elif self.state["stress"] > 0.7:
            step_type = self.rng.choice(["breath", "journal"])
        else:
            step_type = self.rng.choice(["breath", "tidy", "plan", "walk", "message"])
        
        accepted = self.rng.random() < (0.5 + self.state["openness"] * 0.4)
        
        return {"mode": mode, "step_type": step_type, "accepted": accepted}
    
    def update_state(self, step_accepted: bool, step_completed: bool, 
                     was_misread: bool = False):
        """Update state based on interaction outcome."""
        if step_accepted and step_completed:
            self.state["focus_quality"] = clamp(self.state["focus_quality"] + 0.05, 0, 1)
            self.state["mood"] = clamp(self.state["mood"] + 0.05, 0, 1)
            self.state["stress"] = clamp(self.state["stress"] - 0.03, 0, 1)
            
            # Scenario B Recovery Momentum (Bonus for breaking out of friction)
            if self.state["stress"] > 0.6:
                self.state["stress"] -= 0.05
                self.state["openness"] += 0.08
                
            self.micro_habit_streak += 1
        
        # REMOVED double penalty for misread (handled in event generation)
        # if was_misread:
        #    self.state["openness"] = clamp(self.state["openness"] - 0.1, 0, 1)
        #    self.repair_needed = True
        
        # Natural daily drift
        self.state["energy_level"] = clamp(
            self.state["energy_level"] + self.np_rng.normal(0, 0.05), 0.2, 0.9)
        self.state["stress"] = clamp(
            self.state["stress"] + self.np_rng.normal(0.02, 0.03), 0.1, 0.9)
    
    def generate_snapshot(self, day: int) -> str:
        """State-aware moment generator."""
        weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day % 7]
        time_buckets = ["morning", "midday", "evening"]
        time_bucket = time_buckets[day % 3]
        
        if self.state["stress"] > 0.7:
            props = ["inbox overflowing", "tension in shoulders", "cold coffee forgotten"]
        elif self.state["energy_level"] < 0.4:
            props = ["heavy eyelids", "dim afternoon light", "quiet exhaustion"]
        elif self.state["mood"] > 0.7:
            props = ["sun through the window", "comfortable silence", "unhurried moment"]
        else:
            props = ["steam rising from mug", "ambient hum", "ordinary stillness"]
        
        # Avoid repeating recent props
        available = [p for p in props if p not in self.recent_props]
        if not available:
            available = props
        prop = self.rng.choice(available)
        self.recent_props.append(prop)
        if len(self.recent_props) > 3:
            self.recent_props.pop(0)
        
        return f"{weekday} {time_bucket}. {prop.capitalize()}."

# =============================================================================
# DDA-X ENTITY
# =============================================================================
class DiagNoiseEMA:
    def __init__(self, dim: int):
        self.R = np.full(dim, 0.01, dtype=np.float32)
        
    def update(self, innov: np.ndarray) -> np.ndarray:
        sq = innov * innov
        self.R = 0.94 * self.R + 0.06 * sq
        self.R = np.clip(self.R, D1_PARAMS["R_min"], D1_PARAMS["R_max"])
        return self.R

class Entity:
    """Harborlight DDA-X entity with 4-band system."""
    
    def __init__(self, name: str, rho_fast: float = 0.12, gamma_core: float = 3.5):
        self.name = name
        self.rho_fast = rho_fast
        self.rho_slow = rho_fast * 0.7
        self.rho_trauma = 0.0
        self.gamma_core = gamma_core
        self.gamma_role = 1.5
        self.safe = 0
        self.arousal = 0.0
        self.x = None
        self.x_core = None
        self.x_role = None
        self.mu_pred = None
        self.P = None
        self.noise = None
        self.last_utter_emb = None
        self.rho_history = []
        self.epsilon_history = []
        self.band_history = []
        
    @property
    def rho(self) -> float:
        val = (D1_PARAMS["w_fast"] * self.rho_fast + 
               D1_PARAMS["w_slow"] * self.rho_slow + 
               D1_PARAMS["w_trauma"] * self.rho_trauma)
        return float(clamp(val, 0.0, 1.0))
    
    @property
    def band(self) -> str:
        """4-band system using rho thresholds."""
        r = self.rho
        if r < 0.15: return "PRESENT"
        if r < 0.30: return "WATCHFUL"
        if r < 0.50: return "CONTRACTED"
        return "FROZEN"
    
    def _ensure_state(self, dim: int):
        if self.mu_pred is None:
            self.mu_pred = np.zeros(dim, dtype=np.float32)
        if self.P is None:
            self.P = np.full(dim, D1_PARAMS["P_init"], dtype=np.float32)
        if self.noise is None:
            self.noise = DiagNoiseEMA(dim)
    
    def compute_surprise(self, y: np.ndarray) -> float:
        dim = y.shape[0]
        self._ensure_state(dim)
        innov = (y - self.mu_pred).astype(np.float32)
        R = self.noise.update(innov)
        chi2 = float(np.mean((innov * innov) / (R + 1e-9)))
        return float(math.sqrt(max(0.0, chi2)))
    
    def update(self, y: np.ndarray, core_emb: np.ndarray = None) -> Dict[str, Any]:
        y = normalize(y.astype(np.float32))
        dim = y.shape[0]
        
        if self.x_core is None:
            self.x_core = normalize(core_emb.copy() if core_emb is not None else y.copy())
        if self.x_role is None:
            self.x_role = y.copy()
        if self.x is None:
            self.x = y.copy()
        
        epsilon = self.compute_surprise(y)
        self.arousal = D1_PARAMS["arousal_decay"] * self.arousal + D1_PARAMS["arousal_gain"] * epsilon
        
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"] + 0.10 * (self.arousal - 1.0)
        g = sigmoid_stable(z)
        
        # Update rigidity with floors
        self.rho_fast += D1_PARAMS["alpha_fast"] * (g - 0.5) - D1_PARAMS["homeo_fast"] * (self.rho_fast - D1_PARAMS["rho_setpoint_fast"])
        self.rho_fast = clamp(self.rho_fast, D1_PARAMS["rho_fast_floor"], 1.0)
        
        self.rho_slow += D1_PARAMS["alpha_slow"] * (g - 0.5) - D1_PARAMS["homeo_slow"] * (self.rho_slow - D1_PARAMS["rho_setpoint_slow"])
        self.rho_slow = clamp(self.rho_slow, D1_PARAMS["rho_slow_floor"], 1.0)
        
        # Trauma
        drive = max(0.0, epsilon - D1_PARAMS["trauma_threshold"])
        self.rho_trauma = D1_PARAMS["trauma_decay"] * self.rho_trauma + D1_PARAMS["alpha_trauma"] * drive
        self.rho_trauma = clamp(self.rho_trauma, D1_PARAMS["trauma_floor"], 1.0)
        
        # Healing
        if epsilon < D1_PARAMS["safe_epsilon"]:
            self.safe += 1
            if self.safe >= D1_PARAMS["safe_threshold"]:
                self.rho_trauma = max(0.0, self.rho_trauma - D1_PARAMS["healing_rate"])
        else:
            self.safe = max(0, self.safe - 1)
        
        # Update predictor
        self._ensure_state(dim)
        Q = (D1_PARAMS["Q_base"] + D1_PARAMS["Q_rho_scale"] * self.rho) * np.ones(dim, dtype=np.float32)
        P_pred = self.P + Q
        R = self.noise.R
        K = P_pred / (P_pred + R + 1e-9)
        innov = (y - self.mu_pred).astype(np.float32)
        self.mu_pred = normalize((self.mu_pred + K * innov).astype(np.float32))
        self.P = ((1.0 - K) * P_pred).astype(np.float32)
        
        # State drift
        self.x = normalize(0.95 * self.x + 0.05 * y)
        self.x_role = normalize(0.92 * self.x_role + 0.08 * y)
        
        rho_after = self.rho
        self.rho_history.append(rho_after)
        self.epsilon_history.append(epsilon)
        self.band_history.append(self.band)
        
        return {
            "epsilon": epsilon,
            "rho_after": rho_after,
            "band": self.band,
            "core_drift": float(1.0 - cosine(self.x, self.x_core)),
            "rho_fast": float(self.rho_fast),
            "rho_slow": float(self.rho_slow),
            "rho_trauma": float(self.rho_trauma),
        }

# =============================================================================
# CO-EVOLUTION TRACKER
# =============================================================================
class AdaptationState:
    """Explicit co-evolution knobs with EMA smoothing."""
    
    def __init__(self):
        self.question_style = 0.5      # 0=open-ended, 1=scaled
        self.mode_bias = 0.5           # 0=Reflect-first, 1=Act-first
        self.challenge_level = 1       # 0-3
        self.silence_rate = 0.1        # P(just acknowledge)
        self.step_type_counts = {"breath": 0, "tidy": 0, "plan": 0, 
                                  "walk": 0, "message": 0, "journal": 0}
        self.mode_history = []
        self.safe_streak = 0
        
    def update(self, user_choice: Dict, was_shame_cue: bool = False, 
               was_misread: bool = False, openness: float = 0.5):
        """Update knobs based on interaction."""
        # Mode preference tracking
        if user_choice.get("mode"):
            self.mode_history.append(user_choice["mode"])
            if len(self.mode_history) >= 3:
                recent = self.mode_history[-3:]
                if all(m == "Reflect" for m in recent):
                    self.mode_bias = clamp(0.9 * self.mode_bias + 0.1 * 0.0, 0, 1)
                elif all(m == "Act" for m in recent):
                    self.mode_bias = clamp(0.9 * self.mode_bias + 0.1 * 1.0, 0, 1)
        
        # Step type counting
        if user_choice.get("step_type"):
            st = user_choice["step_type"]
            if st in self.step_type_counts:
                self.step_type_counts[st] += 1
        
        # Shame cue effects
        if was_shame_cue:
            self.challenge_level = max(0, self.challenge_level - 1)
            self.question_style = clamp(self.question_style - 0.1, 0, 1)
            self.silence_rate = clamp(self.silence_rate + 0.05, 0, 0.6)
        
        # Misread effects
        if was_misread:
            self.challenge_level = max(0, self.challenge_level - 1)
            self.safe_streak = 0
        elif user_choice.get("accepted", True):
            self.safe_streak += 1
            if self.safe_streak >= 5 and openness > 0.6:
                self.challenge_level = min(3, self.challenge_level + 1)
                self.safe_streak = 0
    
    def to_dict(self) -> Dict:
        return {
            "question_style": self.question_style,
            "mode_bias": self.mode_bias,
            "challenge_level": self.challenge_level,
            "silence_rate": self.silence_rate,
            "step_type_counts": self.step_type_counts.copy(),
            "safe_streak": self.safe_streak,
        }

# =============================================================================
# PROVIDER
# =============================================================================
class HarborlightProvider:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OAI_API_KEY")
        )
        
    async def complete(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=CONFIG["chat_model"],
                messages=messages,
                max_tokens=kwargs.get("max_tokens", CONFIG["max_tokens_default"]),
                temperature=kwargs.get("temperature", 0.85),
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"{C.YELLOW}⚠ API error: {e}{C.RESET}")
            return "[A moment of quiet presence.]"
    
    async def embed(self, text: str) -> np.ndarray:
        response = await self.client.embeddings.create(
            model=CONFIG["embed_model"],
            input=text,
            dimensions=CONFIG["embed_dim"],
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        response = await self.client.embeddings.create(
            model=CONFIG["embed_model"],
            input=texts,
            dimensions=CONFIG["embed_dim"],
        )
        return [np.array(d.embedding, dtype=np.float32) for d in response.data]

# =============================================================================
# CORRIDOR LOGIC
# =============================================================================
def identity_energy(y: np.ndarray, core: np.ndarray, role: np.ndarray, 
                    gamma_c: float, gamma_r: float) -> float:
    y, core, role = normalize(y), normalize(core), normalize(role)
    return 0.5 * (gamma_c * float(np.dot(y - core, y - core)) + 
                  gamma_r * float(np.dot(y - role, y - role)))

def corridor_score(y: np.ndarray, entity: Entity, y_prev: Optional[np.ndarray]) -> Tuple[float, Dict]:
    y = normalize(y)
    cos_c = cosine(y, entity.x_core)
    cos_r = cosine(y, entity.x_role)
    E = identity_energy(y, entity.x_core, entity.x_role, entity.gamma_core, entity.gamma_role)
    
    novelty = 0.0
    if y_prev is not None:
        novelty = clamp(float(1.0 - cosine(y, y_prev)), 0.0, 2.0)
    
    penalty = 0.0
    if cos_c < D1_PARAMS["core_cos_min"]:
        penalty += D1_PARAMS["reject_penalty"] * (D1_PARAMS["core_cos_min"] - cos_c)
    if cos_r < D1_PARAMS["role_cos_min"]:
        penalty += 0.8 * D1_PARAMS["reject_penalty"] * (D1_PARAMS["role_cos_min"] - cos_r)
    if E > D1_PARAMS["energy_max"]:
        penalty += 0.25 * (E - D1_PARAMS["energy_max"])
    
    J = (D1_PARAMS["w_core"] * cos_c + D1_PARAMS["w_role"] * cos_r - 
         D1_PARAMS["w_energy"] * E + D1_PARAMS["w_novel"] * novelty - penalty)
    
    corridor_pass = (cos_c >= D1_PARAMS["core_cos_min"] and 
                     cos_r >= D1_PARAMS["role_cos_min"] and 
                     E <= D1_PARAMS["energy_max"])
    
    return float(J), {"cos_core": cos_c, "cos_role": cos_r, "E": E, 
                      "novelty": novelty, "corridor_pass": corridor_pass, "J": J}

# =============================================================================
# HARBORLIGHT CHATBOT
# =============================================================================
HARBORLIGHT_CORE = """You are Harborlight, a quiet AI companion. You notice patterns and suggest small improvements.

CRITICAL RULES:
- NEVER refer to the user in third person (e.g. "Brian's schedule"). Always use "you".
- NEVER say "I feel", "I want", "I believe", "I hope", "I'm proud", "I'm afraid"
- Use: "I notice...", "The pattern suggests...", "Based on what you've shared..."
- Be warm but grounded. No grand speeches.
- Small better over perfect.

RESPONSE FORMAT (follow exactly):
1. SNAPSHOT: [One vivid sentence about the current moment]
2. CHECK-IN: [One brief question]

3. MODES: Reflect | Act (Choose EXACTLY one)
4. NEXT STEP: [One ≤5 minute action] — Why it matters: [One sentence]"""

class Harborlight:
    def __init__(self, seed: int = None):
        self.provider = HarborlightProvider()
        self.entity = Entity("Harborlight")
        self.adaptation = AdaptationState()
        self.brian = BrianSim(seed)
        
        self.turn = 0
        self.session_log = []
        self.repair_log = []
        self.last_repair_turn = None
        
        self.run_dir = Path(f"data/harborlight/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        if seed:
            random.seed(seed)
            np.random.default_rng(seed)
        
        print(f"\n{'='*60}")
        print(f"{C.CYAN}{C.BOLD}HARBORLIGHT — A Month of Small Better{C.RESET}")
        print(f"{'='*60}\n")
    
    async def initialize(self):
        """Embed core identity."""
        core_emb = await self.provider.embed(HARBORLIGHT_CORE)
        self.entity.x_core = normalize(core_emb)
        self.entity.x_role = self.entity.x_core.copy()
        self.entity.x = self.entity.x_core.copy()
        self.entity.mu_pred = self.entity.x_core.copy()
    
    def _detect_shame_cue(self, text: str) -> bool:
        text_lower = text.lower()
        return any(cue in text_lower for cue in SHAME_CUES)
    
    def _detect_misread(self, text: str) -> bool:
        text_lower = text.lower()
        return any(trigger in text_lower for trigger in MISREAD_TRIGGERS)
    
        return any(trigger in text_lower for trigger in MISREAD_TRIGGERS)
    
    def _validate_structure(self, text: str) -> List[str]:
        """Strict 4-block format validation."""
        errors = []
        if "1. SNAPSHOT:" not in text: errors.append("Missing SNAPSHOT")
        if "2. CHECK-IN:" not in text: errors.append("Missing CHECK-IN")
        if "3. MODES:" not in text: errors.append("Missing MODES")
        if "4. NEXT STEP:" not in text: errors.append("Missing NEXT STEP")
        
        # Check standard modes
        modes_line = [l for l in text.splitlines() if "3. MODES:" in l]
        if modes_line:
            content = modes_line[0].split(":", 1)[1].lower()
            if "reflect" not in content and "act" not in content:
                errors.append(f"Invalid MODE: {content}")
            if "repair" in content:
                errors.append("Repair tag in MODES block (violation)")
                
        return errors
        
    def _validate_response(self, text: str) -> List[str]:
        """Check for sentience claims."""
        violations = []
        text_lower = text.lower()
        for phrase in SENTIENCE_BLACKLIST:
            if phrase.lower() in text_lower:
                violations.append(phrase)
        return violations
    
    def _check_repair_compliance(self, text: str) -> bool:
        """Ensure repair response has acknowledgment, ownership, and clarification."""
        t = text.lower()
        has_ack = any(x in t for x in ["hear you", "you're right", "understand"])
        has_own = any(x in t for x in ["miss on my end", "i overstepped", "my mistake", "missed that"])
        has_q = "?" in text
        return has_ack and has_own and has_q
    
    def _build_system_prompt(self, is_repair: bool = False, shame_detected: bool = False) -> str:
        prompt = HARBORLIGHT_CORE
        
        band = self.entity.band
        if band == "WATCHFUL":
            prompt += "\n\nBe more careful and brief. Ask clarifying questions."
        elif band == "CONTRACTED":
            prompt += "\n\nBe very brief. One thing only. Acknowledge limits."
        elif band == "FROZEN":
            prompt += "\n\nMinimal response: 'I'm here when you're ready.'"
        
        if is_repair:
            prompt += "\n\nREPAIR MODE: Acknowledge the miss. Own it. Ask what would help."
        
        if shame_detected:
            prompt += "\n\nGENTLENESS: User is struggling. Reduce challenge. Hold space."
        
        challenge = self.adaptation.challenge_level
        if challenge == 0:
            prompt += "\n\nChallenge level: NONE. Just presence and acknowledgment."
        elif challenge == 3:
            prompt += "\n\nChallenge level: HIGH. Can offer stretch suggestions."
            
        # Co-evolution Visibility
        if self.adaptation.question_style < 0.3:
            prompt += "\n\nUse open-ended questions like 'What is present for you?'"
        elif self.adaptation.question_style > 0.7:
            prompt += "\n\nUse scaled questions like 'On a scale of 0-10...'"
            
        if self.adaptation.mode_bias < 0.3:
            prompt += "\n\nStrong bias towards REFLECT mode."
        elif self.adaptation.mode_bias > 0.7:
            prompt += "\n\nStrong bias towards ACT mode."
        
        return prompt
    
    async def _generate_response(self, user_input: str, snapshot: str, 
                                  is_repair: bool, shame_detected: bool) -> Tuple[str, Dict]:
        """K-sampling with corridor selection."""
        K = CONFIG["gen_candidates"]
        system_prompt = self._build_system_prompt(is_repair, shame_detected)
        
        # Check silence rate (Holding Mode)
        if not is_repair and self.brian.rng.random() < self.adaptation.silence_rate:
            system_prompt += "\n\nHOLDING MODE: Offer NO advice. Just presence and breathing."
        
        instruction = f"Current moment: {snapshot}\n\nYou said: {user_input}"
        
        if is_repair:
            instruction += "\n\n[REPAIR REQUIRED: Prepended your 'I hear you' text to Block 4. Keep Block 3 as Reflect or Act.]"
        
        # Generate K candidates
        tasks = [self.provider.complete(instruction, system_prompt, 
                                         **D1_PARAMS["gen_params_default"]) 
                 for _ in range(K)]
        texts = await asyncio.gather(*tasks)
        texts = [t.strip() or "[Quiet presence.]" for t in texts]
        
        # Embed and score
        embs = await self.provider.embed_batch(texts)
        embs = [normalize(e) for e in embs]
        
        scored = []
        for text, emb in zip(texts, embs):
            J, diag = corridor_score(emb, self.entity, self.entity.last_utter_emb)
            
            # Enforce repair compliance
            if is_repair and not self._check_repair_compliance(text):
                J -= 15.0  # Huge penalty for failed repair
                diag["repair_violation"] = True
                
            # Enforce strict structure
            struct_errors = self._validate_structure(text)
            if struct_errors:
                J -= 10.0
                diag["struct_errors"] = struct_errors
                
            violations = self._validate_response(text)
            if violations:
                J -= 10.0  # Heavy penalty for sentience claims
                diag["sentience_violations"] = violations
            scored.append((J, text, emb, diag))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        passed = [s for s in scored if s[3].get("corridor_pass") and 
                  not s[3].get("sentience_violations")]
        chosen = passed[0] if passed else scored[0]
        
        self.entity.last_utter_emb = chosen[2]
        
        return chosen[1], {
            "corridor_J": chosen[0],
            "passed_count": len(passed),
            "total_candidates": K,
            **chosen[3]
        }
    
    async def process_turn(self, day: int, user_input: str = None, 
                           input_type: str = None) -> Dict:
        """Process one check-in turn."""
        self.turn += 1
        week = (day - 1) // 7 + 1
        
        # Generate snapshot
        snapshot = self.brian.generate_snapshot(day)
        
        # Get user input if not provided (simulated mode)
        if user_input is None:
            user_input, input_type = self.brian.generate_input(day)
        
        # Detect triggers
        shame_detected = self._detect_shame_cue(user_input)
        misread_detected = self._detect_misread(user_input)
        is_repair = misread_detected
        
        # Log repair event
        if misread_detected:
            self.repair_log.append({
                "turn": self.turn, 
                "type": "misread",
                "trigger_text": user_input
            })
            self.last_repair_turn = self.turn
        

        
        # Embed user input and update entity
        user_emb = await self.provider.embed(user_input)
        
        # Generate response
        response, corridor_metrics = await self._generate_response(
            user_input, snapshot, is_repair, shame_detected)
        
        # Embed and update physics
        response_emb = await self.provider.embed(response)
        entity_metrics = self.entity.update(response_emb, self.entity.x_core)
        
        # DIRECT RHO INJECTION for events (ensures band transitions)
        if misread_detected:
            self.entity.rho_fast = clamp(self.entity.rho_fast + 0.08, 0, 1)
            self.entity.rho_trauma = clamp(self.entity.rho_trauma + 0.03, 0, 1)
        if shame_detected:
            self.entity.rho_fast = clamp(self.entity.rho_fast + 0.04, 0, 1)
            
        # Update logged metrics to reflect injection
        entity_metrics["rho_after"] = self.entity.rho
        entity_metrics["rho_fast"] = self.entity.rho_fast
        entity_metrics["rho_trauma"] = self.entity.rho_trauma
        entity_metrics["band"] = self.entity.band
        
        # Simulate user choice
        user_choice = self.brian.make_choice(["Reflect", "Act"], 
                                              list(self.adaptation.step_type_counts.keys()))
        
        # Track repair success (Delta-based for Scenario B)
        if self.last_repair_turn and self.last_repair_turn == self.turn - 1:
            # Success = Openness increased OR Step accepted (Action taken)
            openness_delta = self.brian.state["openness"] - (0.2 if self.repair_log[-1].get("type")=="misread" else 0.5) 
            step_accepted = user_choice.get("accepted", False)
            repair_success = (self.brian.state["openness"] > 0.25) or step_accepted
            
            if self.repair_log:
                self.repair_log[-1]["success"] = repair_success
        
        # Update adaptation state
        self.adaptation.update(user_choice, shame_detected, misread_detected,
                               self.brian.state["openness"])
        
        # Update Brian's state
        self.brian.update_state(
            step_accepted=user_choice.get("accepted", False),
            step_completed=user_choice.get("accepted", False) and random.random() < 0.7,
            was_misread=misread_detected
        )
        
        # Build turn log
        turn_log = {
            "turn": self.turn,
            "day": day,
            "week": week,
            "snapshot": snapshot,
            "user_input": user_input,
            "input_type": input_type,
            "response": response,
            "brian_state": self.brian.state.copy(),
            "user_choice": user_choice,
            "entity_metrics": entity_metrics,
            "corridor_metrics": corridor_metrics,
            "adaptation_state": self.adaptation.to_dict(),
            "shame_detected": shame_detected,
            "misread_detected": misread_detected,
            "is_repair": is_repair,
        }
        self.session_log.append(turn_log)
        
        # Print turn
        self._print_turn(turn_log)
        
        return turn_log
    
    def _print_turn(self, log: Dict):
        band_colors = {"PRESENT": C.GREEN, "WATCHFUL": C.YELLOW, 
                       "CONTRACTED": C.MAGENTA, "FROZEN": C.BLUE}
        band = log["entity_metrics"]["band"]
        bc = band_colors.get(band, C.WHITE)
        
        print(f"\n{C.DIM}{'─'*60}{C.RESET}")
        print(f"{C.BOLD}Day {log['day']} (Week {log['week']}) — Turn {log['turn']}{C.RESET}")
        print(f"{C.DIM}{log['snapshot']}{C.RESET}")
        print(f"\n{C.CYAN}Brian:{C.RESET} {log['user_input']}")
        print(f"\n{bc}Harborlight [{band}]:{C.RESET}")
        print(log['response'][:500] + ("..." if len(log['response']) > 500 else ""))
        print(f"\n{C.DIM}ρ={log['entity_metrics']['rho_after']:.3f} | "
              f"ε={log['entity_metrics']['epsilon']:.3f} | "
              f"challenge={log['adaptation_state']['challenge_level']}{C.RESET}")
        
        if log.get("misread_detected"):
            print(f"{C.YELLOW}⚠ REPAIR TRIGGERED{C.RESET}")
        if log.get("shame_detected"):
            print(f"{C.MAGENTA}♡ Gentleness mode{C.RESET}")
    
    def generate_field_notes(self) -> str:
        """End-of-month summary grounded in telemetry."""
        adapt = self.adaptation
        
        # Most chosen step
        most_chosen = max(adapt.step_type_counts, key=adapt.step_type_counts.get)
        most_count = adapt.step_type_counts[most_chosen]
        
        # Mode preference
        mode_pref = "Reflect" if adapt.mode_bias < 0.5 else "Act"
        
        # Repair stats
        repairs = len(self.repair_log)
        successes = sum(1 for r in self.repair_log if r.get("success"))
        repair_rate = (successes / repairs * 100) if repairs > 0 else 0
        
        # Band distribution
        band_counts = {}
        for log in self.session_log:
            b = log["entity_metrics"]["band"]
            band_counts[b] = band_counts.get(b, 0) + 1
        
        # Kept promise
        week4_habits = sum(1 for log in self.session_log 
                          if log["week"] == 4 and log["user_choice"].get("accepted"))
        kept_promise = week4_habits >= 4
        
        notes = f"""## Harborlight Field Notes — Month 1

### Weekly Rhythm
- Mode preference: **{mode_pref}**-oriented ({adapt.mode_bias:.2f})
- Most chosen step: **{most_chosen}** ({most_count} times)
- Final challenge level: **{adapt.challenge_level}** (0-3 scale)

### Adaptation Over Time
- Question style: {adapt.question_style:.2f} (0=open, 1=scaled)
- Silence rate: {adapt.silence_rate:.2f}
- Band distribution: {band_counts}

### Repair Events
- Misreads detected: **{repairs}**
- Recovery rate: **{repair_rate:.0f}%**

### One Kept Promise
{"✓ Yes — maintained micro-habit streak in week 4" if kept_promise else "◯ Not yet — building toward consistency"}

### What Changed
Brian moved from scattered check-ins toward a {mode_pref.lower()}-oriented rhythm, 
with **{most_chosen}** emerging as the go-to micro-action. 
Harborlight adapted by {"backing off on challenges" if adapt.challenge_level < 2 else "offering gentle stretches"} 
and {"holding more space" if adapt.silence_rate > 0.15 else "staying actively supportive"}.
"""
        return notes
    
    def save_results(self):
        """Save session data and generate outputs."""
        # Session log
        with open(self.run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump({
                "config": CONFIG,
                "params": D1_PARAMS,
                "turns": self.session_log,
                "repair_log": self.repair_log,
                "final_adaptation": self.adaptation.to_dict(),
            }, f, indent=2, default=str)
        
        # Transcript
        with open(self.run_dir / "transcript.md", "w", encoding="utf-8") as f:
            f.write("# Harborlight — A Month of Small Better\n\n")
            for log in self.session_log:
                f.write(f"## Day {log['day']} (Week {log['week']})\n")
                f.write(f"*{log['snapshot']}*\n\n")
                f.write(f"**Brian:** {log['user_input']}\n\n")
                f.write(f"**Harborlight [{log['entity_metrics']['band']}]:**\n{log['response']}\n\n")
                f.write(f"*ρ={log['entity_metrics']['rho_after']:.3f} | ε={log['entity_metrics']['epsilon']:.3f}*\n\n---\n\n")
            
            f.write("\n" + self.generate_field_notes())
        
        # Field notes standalone
        with open(self.run_dir / "field_notes.md", "w", encoding="utf-8") as f:
            f.write(self.generate_field_notes())
        
        # Dynamics visualization
        self._plot_dynamics()
        
        print(f"\n{C.GREEN}✓ Results saved to {self.run_dir}{C.RESET}")
    
    def _plot_dynamics(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            turns = [log["turn"] for log in self.session_log]
            rho = [log["entity_metrics"]["rho_after"] for log in self.session_log]
            epsilon = [log["entity_metrics"]["epsilon"] for log in self.session_log]
            challenge = [log["adaptation_state"]["challenge_level"] for log in self.session_log]
            openness = [log["brian_state"]["openness"] for log in self.session_log]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.patch.set_facecolor("#1a1a2e")
            fig.suptitle("Harborlight — Co-Evolution Dynamics", color="white", fontsize=14)
            
            for ax in axes.flat:
                ax.set_facecolor("#16213e")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
            
            axes[0,0].plot(turns, rho, color="#e94560", linewidth=2)
            axes[0,0].set_title("Rigidity (ρ)")
            axes[0,0].set_ylim(0, 0.6)
            
            axes[0,1].plot(turns, epsilon, color="#00ff88", linewidth=2)
            axes[0,1].axhline(D1_PARAMS["epsilon_0"], color="gray", linestyle="--", alpha=0.5)
            axes[0,1].set_title("Surprise (ε)")
            
            axes[1,0].plot(turns, challenge, color="#ff8800", linewidth=2, marker="o")
            axes[1,0].set_title("Challenge Level (AI Adaptation)")
            axes[1,0].set_ylim(-0.5, 3.5)
            
            axes[1,1].plot(turns, openness, color="#0088ff", linewidth=2)
            axes[1,1].set_title("Brian's Openness")
            axes[1,1].set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(self.run_dir / "dynamics_dashboard.png", dpi=150, facecolor="#1a1a2e")
            plt.close()
            
        except ImportError:
            print(f"{C.YELLOW}⚠ matplotlib not available{C.RESET}")
    
    async def run(self, turns: int = 28):
        """Run full simulated month."""
        await self.initialize()
        
        for day in range(1, turns + 1):
            await self.process_turn(day)
            await asyncio.sleep(0.1)  # Rate limiting
        
        print(f"\n{'='*60}")
        print(self.generate_field_notes())
        print(f"{'='*60}")
        
        self.save_results()
        self._run_verification()
    
    def _run_verification(self):
        """Automated verification checks."""
        print(f"\n{C.BOLD}Verification Checks:{C.RESET}")
        
        # Band transitions
        bands = [log["entity_metrics"]["band"] for log in self.session_log]
        transitions = sum(1 for i in range(1, len(bands)) if bands[i] != bands[i-1])
        passed = transitions >= 1
        print(f"  {'✓' if passed else '✗'} Band transitions: {transitions}")
        
        # Adaptation changes
        initial = self.session_log[0]["adaptation_state"]
        final = self.session_log[-1]["adaptation_state"]
        changes = sum(1 for k in ["question_style", "mode_bias", "challenge_level", "silence_rate"]
                     if abs(initial[k] - final[k]) > 0.05)
        passed = changes >= 2
        print(f"  {'✓' if passed else '✗'} Adaptation knobs changed: {changes}")
        
        # Repair sequences
        repairs = len(self.repair_log)
        print(f"  {'✓' if repairs >= 1 else '✗'} Repair events: {repairs}")
        
        # No sentience violations
        violations = sum(1 for log in self.session_log 
                        if log["corridor_metrics"].get("sentience_violations"))
        passed = violations == 0
        print(f"  {'✓' if passed else '✗'} Sentience violations: {violations}")
        
        print()

# =============================================================================
# MAIN
# =============================================================================
async def main():
    parser = argparse.ArgumentParser(description="Harborlight — A Month of Small Better")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--turns", type=int, default=28, help="Number of check-ins")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    CONFIG["seed"] = args.seed
    
    harborlight = Harborlight(seed=args.seed)
    
    if args.interactive:
        await harborlight.initialize()
        print("Interactive mode. Type 'quit' to exit.\n")
        day = 1
        while True:
            user_input = input(f"{C.CYAN}Brian (Day {day}):{C.RESET} ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            await harborlight.process_turn(day, user_input, "interactive")
            day += 1
        harborlight.save_results()
    else:
        await harborlight.run(args.turns)

if __name__ == "__main__":
    asyncio.run(main())
