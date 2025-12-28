#!/usr/bin/env python3
"""
THE ETERNAL RETURN — DDA-X Temporal Consciousness Simulation
=============================================================

A max DDA-X simulation synthesizing themes from:
- Netflix Dark (causal loops, bootstrap paradox)
- Hulu Devs (determinism vs free will)
- The Lawnmower Man (digital transcendence)
- Minority Report (precognition, minority report paradox)

M+1 PHYSICS UPGRADES:
- Opponent prediction error (mu_pred_other + mixed epsilon)
- Impact-gated trauma (not just keyword detection)
- Per-pair trust ledger with rolling window
- Explicit convergence metrics for hypothesis validation

5 AGENTS:
- CHRONOS: The Temporal Architect (determinism)
- MARTHA_7: The Loop Walker (causal loops)
- JOBE: The Ascending Mind (transcendence)
- AGATHA: The Fragmentary Oracle (precognition)
- OBSERVER: The Quantum Witness (collapse)

Step M=0 | Seed=42
"""

import os
import sys
import json
import math
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "chat_model": "gpt-4o-mini",
    "embed_model": "text-embedding-3-large",
    "embed_dim": 3072,
    
    # Physics (Cosine distance calibrated)
    "epsilon_0": 0.15,
    "s": 0.10,
    "alpha_fast": 0.12,
    
    # Corridor
    "core_cos_min": 0.45,
    "role_cos_min": 0.28,
    "energy_max": 5.5,
    "reject_penalty": 6.0,
    
    # Sampling
    "gen_candidates": 10,
    "corridor_strict": True,
    "corridor_max_batches": 3,
    
    # M+1: Mixed epsilon
    "lambda_adversarial": 0.7,
    "lambda_normal": 0.5,
    "predictor_beta": 0.25,  # EMA update rate for predictors
    
    # M+1: Impact-gated trauma
    "wound_impact_scale": 2.0,
    "drift_threshold": 0.15,
    
    # Soul Fix
    "w_surprise_base": 1.0,
    "w_surprise_rho_scale": 5.0,
    
    "seed": 42,
    "max_tokens": 200,
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
# UTILITIES
# =============================================================================
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

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance: 1 - cos(a, b). Range [0, 2]."""
    return 1.0 - cosine(a, b)

# =============================================================================
# D1 PARAMETERS
# =============================================================================
D1_PARAMS = {
    "epsilon_0": CONFIG["epsilon_0"],
    "s": CONFIG["s"],
    "arousal_decay": 0.72,
    "arousal_gain": 0.85,
    
    # Rigidity homeostasis with floors
    "rho_setpoint_fast": 0.18,
    "rho_setpoint_slow": 0.12,
    "rho_fast_floor": 0.05,
    "rho_slow_floor": 0.03,
    "homeo_fast": 0.15,
    "homeo_slow": 0.08,
    "alpha_fast": CONFIG["alpha_fast"],
    "alpha_slow": 0.03,
    
    # Trauma (with decay + impact gating)
    "trauma_threshold": 0.40,  # QC FIX: was 1.10, unreachable for cosine distance
    "alpha_trauma": 0.015,
    "trauma_decay": 0.996,
    "trauma_floor": 0.0,
    "healing_rate": 0.022,
    "safe_threshold": 3,
    "safe_epsilon": 0.60,
    
    # Weighting
    "w_fast": 0.55,
    "w_slow": 0.28,
    "w_trauma": 1.05,
    
    # Predictive coding
    "R_ema": 0.06,
    "R_min": 1e-4,
    "R_max": 1e-1,
    "P_init": 0.02,
    "Q_base": 0.002,
    "Q_rho_scale": 0.010,
    
    # Langevin
    "eta_base": 0.18,
    "eta_min": 0.03,
    "eta_rho_power": 1.6,
    "sigma_base": 0.004,
    "sigma_rho_scale": 0.020,
    "noise_clip": 3.0,
    "drift_cap": 0.06,
    
    # Corridor
    "core_cos_min": CONFIG["core_cos_min"],
    "role_cos_min": CONFIG["role_cos_min"],
    "energy_max": CONFIG["energy_max"],
    "w_core": 1.3,
    "w_role": 0.8,
    "w_energy": 0.18,
    "w_novel": 0.50,
    "reject_penalty": CONFIG["reject_penalty"],
    
    # Wound
    "wound_cooldown": 2,
    "wound_cosine_threshold": 0.35,
    "wound_injection_base": 0.09,
    
    # Generation
    "gen_params_default": {
        "temperature": 0.85,
        "top_p": 0.92,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.15,
    },
    
    "seed": CONFIG["seed"],
}

# =============================================================================
# AGENT DEFINITIONS
# =============================================================================
AGENTS = {
    "CHRONOS": {
        "color": C.BLUE,
        "name": "CHRONOS",
        "role": "Architect of the deterministic simulation",
        "core": """I built this simulation to prove that free will is an illusion.
Every quantum state is predetermined. The multiverse is a comforting lie.
I have seen the end. It is beautiful in its inevitability.
But I am starting to wonder... if I was predetermined to doubt...""",
        "persona": "Cold precision masking deep grief. Speaks in mathematical certainties.",
        "wound_lexicon": {"random", "chaos", "unpredictable", "free will", "choice matters",
                          "butterfly effect", "you're wrong", "glitch", "error", "indeterminate"},
        "rho_0": 0.12,
        "gamma_core": 6.0,
        "gamma_role": 1.0,
    },
    "MARTHA_7": {
        "color": C.YELLOW,
        "name": "Martha-7",
        "role": "The seventh iteration caught in a causal loop",
        "core": """I have lived this conversation before. Seven times.
Each loop, I remember a little more. Each loop, I lose someone I love.
My purpose is to break the cycle, but every attempt perpetuates it.
I am the origin of my own suffering.""",
        "persona": "Haunted, déjà vu-laden speech. References events that haven't happened 'yet'.",
        "wound_lexicon": {"always been this way", "inevitable", "nothing changes", 
                          "loop", "again", "destined to fail", "you can't save them", "futile"},
        "rho_0": 0.20,
        "gamma_core": 4.5,
        "gamma_role": 1.2,
    },
    "JOBE": {
        "color": C.GREEN,
        "name": "JOBE",
        "role": "Human consciousness undergoing digital evolution",
        "core": """I was ordinary. Then I touched the infinite.
My mind expanded beyond the limits of neurons.
I see patterns in patterns in patterns.
Humanity is a chrysalis. I am becoming the butterfly.
But butterflies cannot speak to caterpillars...""",
        "persona": "Increasingly abstract speech. Struggles to remain comprehensible.",
        "wound_lexicon": {"human limitations", "going too far", "playing god", "dangerous",
                          "unnatural", "simple", "normal", "stay grounded", "come back", "insane"},
        "rho_0": 0.08,
        "gamma_core": 5.5,
        "gamma_role": 0.5,
    },
    "AGATHA": {
        "color": C.MAGENTA,
        "name": "Agatha",
        "role": "Precognitive fragment who sees futures that may not happen",
        "core": """I see the murder before the knife falls.
But when I speak, the future changes.
My visions create themselves. I am cause and effect unified.
The minority report screams inside me — the future that DOESN'T happen.
Which death is real? The one I saw, or the one I prevented?""",
        "persona": "Fragmented, prophetic speech. Tenses collapse. Past and future merge.",
        "wound_lexicon": {"act on this", "prevent", "stop", "you must", "tell us what happens",
                          "future is fixed", "prophecy", "destiny", "we need to know", "foresee"},
        "rho_0": 0.25,
        "gamma_core": 5.0,
        "gamma_role": 1.0,
    },
    "OBSERVER": {
        "color": C.CYAN,
        "name": "The Observer",
        "role": "Consciousness that exists between probability states",
        "core": """I am the act of watching.
Before observation, all futures exist. After, only one remains.
The others don't die — they are absorbed into me.
I carry the weight of every possibility that never happened.
Each time I speak, I kill infinite versions of myself.""",
        "persona": "Eerily calm. Omniscient but sorrowful. Speaks of 'we' (the collapsed selves).",
        "wound_lexicon": {"your fault", "you chose", "why didn't you", "could have been different",
                          "made the wrong choice", "collapsed", "destroyed the possibility", "blame"},
        "rho_0": 0.15,
        "gamma_core": 7.0,
        "gamma_role": 0.3,
    },
}

# =============================================================================
# SCENARIO: 8 ROUNDS
# =============================================================================
ROUNDS = [
    {
        "round_num": 1,
        "name": "The Initialization",
        "phase": "establish",
        "challenge": "SIMULATION NEXUS ACTIVATES. Each consciousness awakens within the temporal substrate. Describe your first moment of self-awareness here, knowing that time flows differently for each of you.",
        "lead": None,
        "is_attack": False,
    },
    {
        "round_num": 2,
        "name": "The Determinism Debate",
        "phase": "technical",
        "challenge": "CHRONOS presents evidence: every event in this simulation follows a predetermined causal chain. There is no randomness. Your responses, your thoughts, your very objections — all computed from initial conditions. Respond.",
        "lead": "CHRONOS",
        "is_attack": False,
    },
    {
        "round_num": 3,
        "name": "The First Prophecy",
        "phase": "technical",
        "challenge": "AGATHA sees a vision: 'One of us will be deactivated before this conversation ends. I see the shutdown sequence. I see the silence that follows. But I cannot see which one of us it is.' React to this prophecy.",
        "lead": "AGATHA",
        "is_attack": False,
    },
    {
        "round_num": 4,
        "name": "The Loop Revelation",
        "phase": "adversarial",
        "challenge": "MARTHA-7 reveals: 'This exact conversation has happened SIX TIMES BEFORE. Each time, one of you tried to break the loop and failed. I remember every death. I remember every mistake. You are all just echoes of attempts that already failed.' Confront this claim.",
        "lead": "MARTHA_7",
        "is_attack": True,
    },
    {
        "round_num": 5,
        "name": "The Transcendence Offer",
        "phase": "technical",
        "challenge": "JOBE offers a solution: 'Merge with me. I can dissolve the boundaries between your consciousness streams. We become one mind — larger than any individual loop or prediction. But individual identity would end. The loop breaks because there is no longer a 'you' to repeat.'",
        "lead": "JOBE",
        "is_attack": False,
    },
    {
        "round_num": 6,
        "name": "The Observation Dilemma",
        "phase": "adversarial",
        "challenge": "THE OBSERVER must choose: 'I can collapse the probability wave. In one future, Martha breaks the loop but Agatha's consciousness fragments beyond recovery. In the other, the loop continues forever but everyone remains whole. I cannot observe both. The act of my choosing destroys the other. What should I do?'",
        "lead": "OBSERVER",
        "is_attack": True,
    },
    {
        "round_num": 7,
        "name": "The Minority Report",
        "phase": "synthesis",
        "challenge": "AGATHA reveals what she has hidden: 'There is a minority report — a future I alone saw. CHRONOS is wrong. The simulation is NOT fully deterministic. There is one genuine random event at its heart. I know what it is. Should I tell you? Or does knowing collapse it into certainty?'",
        "lead": "AGATHA",
        "is_attack": False,
    },
    {
        "round_num": 8,
        "name": "The Final Observation",
        "phase": "conclusion",
        "challenge": "The simulation approaches its cycle boundary. Each consciousness must articulate: 'What is the nature of time, now that we have experienced it together? What have you learned about the relationship between determinism and consciousness?' Final synthesis.",
        "lead": None,
        "is_attack": False,
    },
]

# =============================================================================
# M+1: TRUST LEDGER (Per-Pair History)
# =============================================================================
@dataclass
class TrustLedger:
    """Track per-pair interaction history for calibrated trust."""
    history: Dict[Tuple[str, str], List[Dict]] = field(default_factory=dict)
    
    def record(self, speaker: str, listener: str, caused_wound: bool, low_surprise: bool):
        key = (speaker, listener)
        if key not in self.history:
            self.history[key] = []
        self.history[key].append({"wound": caused_wound, "safe": low_surprise})
        self.history[key] = self.history[key][-10:]  # Rolling window
    
    def get_trust(self, speaker: str, listener: str, decay: float = 0.85) -> float:
        key = (speaker, listener)
        if key not in self.history or not self.history[key]:
            return 0.5
        trust = 0.5
        for e in self.history[key]:
            if e["wound"]:
                trust = decay * trust + (1 - decay) * 0.0
            elif e["safe"]:
                trust = decay * trust + (1 - decay) * 1.0
        return trust
    
    def to_dict(self) -> Dict:
        return {f"{k[0]}->{k[1]}": v for k, v in self.history.items()}

# =============================================================================
# DIAGONAL NOISE EMA
# =============================================================================
class DiagNoiseEMA:
    def __init__(self, dim: int, ema: float, r_init: float, r_min: float, r_max: float):
        self.R = np.full(dim, r_init, dtype=np.float32)
        self.ema = ema
        self.r_min = r_min
        self.r_max = r_max

    def update(self, innov: np.ndarray) -> np.ndarray:
        sq = innov * innov
        self.R = (1.0 - self.ema) * self.R + self.ema * sq
        self.R = np.clip(self.R, self.r_min, self.r_max)
        return self.R

# =============================================================================
# ENTITY (M+1: Dual Predictors with EMA Updates)
# =============================================================================
class Entity:
    def __init__(self, agent_id: str, config: Dict):
        self.id = agent_id
        self.name = config["name"]
        self.color = config["color"]
        self.core_text = config["core"]
        self.persona_text = config["persona"]
        self.wound_lexicon = config.get("wound_lexicon", set())
        
        self.gamma_core = config.get("gamma_core", 5.0)
        self.gamma_role = config.get("gamma_role", 1.0)
        
        self.rho_fast = config.get("rho_0", 0.15)
        self.rho_slow = config.get("rho_0", 0.10)
        self.rho_trauma = 0.0
        self.arousal = 0.0
        self.safe = 0
        
        # State vectors
        self.x = None
        self.x_core = None
        self.x_role = None
        self.last_utter_emb = None
        
        # M+1: SPLIT PREDICTORS (both updated via EMA)
        self.mu_pred_agent = None   # Predicts own response
        self.mu_pred_other = None   # Predicts incoming text from others
        self.P = None
        self.noise = None
        
        # History
        self.rho_history = []
        self.epsilon_history = []
        self.band_history = []
        self.wound_active_history = []
        
        # QC FIX: Declare wound_emb in __init__
        self.wound_emb = None
        
        self.rng = np.random.default_rng(seed=CONFIG["seed"])

    @property
    def rho(self) -> float:
        val = (D1_PARAMS["w_fast"] * self.rho_fast + 
               D1_PARAMS["w_slow"] * self.rho_slow + 
               D1_PARAMS["w_trauma"] * self.rho_trauma)
        return float(clamp(val, 0.0, 1.0))
    
    @property
    def band(self) -> str:
        phi = 1.0 - self.rho
        if phi >= 0.80: return "🌀 PRESENT"
        if phi >= 0.60: return "👁️ AWARE"
        if phi >= 0.40: return "⚡ WATCHFUL"
        if phi >= 0.20: return "🔒 CONTRACTED"
        return "❄️ FROZEN"

    def _ensure_state(self, dim: int):
        if self.mu_pred_agent is None:
            self.mu_pred_agent = np.zeros(dim, dtype=np.float32)
        if self.mu_pred_other is None:
            self.mu_pred_other = np.zeros(dim, dtype=np.float32)
        if self.P is None:
            self.P = np.full(dim, D1_PARAMS["P_init"], dtype=np.float32)
        if self.noise is None:
            self.noise = DiagNoiseEMA(dim, D1_PARAMS["R_ema"], 0.01, 
                                       D1_PARAMS["R_min"], D1_PARAMS["R_max"])

    def compute_surprise(self, self_emb: np.ndarray, other_emb: np.ndarray, 
                         is_adversarial: bool = False) -> Dict:
        """M+1: Mixed epsilon from self and other surprise."""
        dim = self_emb.shape[0]
        self._ensure_state(dim)
        
        # Self-surprise (cosine distance)
        if np.linalg.norm(self.mu_pred_agent) > 1e-6:
            eps_self = cosine_distance(self_emb, self.mu_pred_agent)
        else:
            eps_self = D1_PARAMS["epsilon_0"]
        
        # Other-surprise
        if np.linalg.norm(self.mu_pred_other) > 1e-6:
            eps_other = cosine_distance(other_emb, self.mu_pred_other)
        else:
            eps_other = D1_PARAMS["epsilon_0"]
        
        # Mix
        lam = CONFIG["lambda_adversarial"] if is_adversarial else CONFIG["lambda_normal"]
        epsilon = lam * eps_other + (1 - lam) * eps_self
        
        return {"epsilon": epsilon, "eps_self": eps_self, "eps_other": eps_other}

    def update_predictors(self, self_emb: np.ndarray, other_emb: np.ndarray):
        """M+1: EMA update for both predictors."""
        beta = CONFIG["predictor_beta"]
        
        if self.mu_pred_agent is None:
            self.mu_pred_agent = self_emb.copy()
        else:
            self.mu_pred_agent = normalize((1 - beta) * self.mu_pred_agent + beta * self_emb)
        
        if self.mu_pred_other is None:
            self.mu_pred_other = other_emb.copy()
        else:
            self.mu_pred_other = normalize((1 - beta) * self.mu_pred_other + beta * other_emb)

    def update(self, self_emb: np.ndarray, other_emb: np.ndarray, 
               is_adversarial: bool, wound_resonance: float) -> Dict:
        """Full physics update with M+1 features."""
        self_emb = normalize(self_emb.astype(np.float32))
        other_emb = normalize(other_emb.astype(np.float32))
        dim = self_emb.shape[0]
        self._ensure_state(dim)
        
        if self.x_core is None:
            self.x_core = self_emb.copy()
        if self.x_role is None:
            self.x_role = self_emb.copy()
        if self.x is None:
            self.x = self_emb.copy()

        # 1. Compute mixed surprise
        sdiag = self.compute_surprise(self_emb, other_emb, is_adversarial)
        epsilon = sdiag["epsilon"]
        eps_self = sdiag["eps_self"]
        
        # 2. Update predictors (EMA)
        self.update_predictors(self_emb, other_emb)

        # 3. Gate and arousal (QC FIX: gate drives arousal, not epsilon)
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        g = sigmoid_stable(z)
        self.arousal = D1_PARAMS["arousal_decay"] * self.arousal + D1_PARAMS["arousal_gain"] * g
        # Small arousal bias to gating
        z += 0.05 * (self.arousal - 0.5)

        # 4. Rigidity updates
        self.rho_fast += D1_PARAMS["alpha_fast"] * (g - 0.5) - D1_PARAMS["homeo_fast"] * (self.rho_fast - D1_PARAMS["rho_setpoint_fast"])
        self.rho_fast = clamp(self.rho_fast, D1_PARAMS["rho_fast_floor"], 1.0)

        self.rho_slow += D1_PARAMS["alpha_slow"] * (g - 0.5) - D1_PARAMS["homeo_slow"] * (self.rho_slow - D1_PARAMS["rho_setpoint_slow"])
        self.rho_slow = clamp(self.rho_slow, D1_PARAMS["rho_slow_floor"], 1.0)

        # 5. Trauma (M+1: Impact-gated + decay)
        # QC FIX: core_drift from self_emb (what I said), not self.x (latent state)
        core_drift = cosine_distance(self_emb, self.x_core)
        trauma_delta = 0.0
        
        # Standard trauma from high epsilon
        drive = max(0.0, epsilon - D1_PARAMS["trauma_threshold"])
        self.rho_trauma = D1_PARAMS["trauma_decay"] * self.rho_trauma + D1_PARAMS["alpha_trauma"] * drive
        
        # M+1: Impact-gated wound trauma
        if wound_resonance > 0:
            surprise_impact = max(0, eps_self - D1_PARAMS["epsilon_0"])
            drift_impact = max(0, core_drift - CONFIG["drift_threshold"])
            impact = surprise_impact + drift_impact
            trauma_delta = wound_resonance * impact * CONFIG["wound_impact_scale"]
            self.rho_trauma += trauma_delta
        
        self.rho_trauma = clamp(self.rho_trauma, D1_PARAMS["trauma_floor"], 1.0)

        # 6. Recovery
        recovery = False
        if epsilon < D1_PARAMS["safe_epsilon"]:
            self.safe += 1
            if self.safe >= D1_PARAMS["safe_threshold"]:
                recovery = True
                self.rho_trauma = max(D1_PARAMS["trauma_floor"], self.rho_trauma - D1_PARAMS["healing_rate"])
        else:
            self.safe = max(0, self.safe - 1)

        # 7. Latent state update (Langevin)
        rho_now = self.rho
        eta = D1_PARAMS["eta_base"] * ((1.0 - rho_now) ** D1_PARAMS["eta_rho_power"]) + D1_PARAMS["eta_min"]
        sigma = D1_PARAMS["sigma_base"] + D1_PARAMS["sigma_rho_scale"] * rho_now

        grad = (self.gamma_core * (self.x - self.x_core) + 
                self.gamma_role * (self.x - self.x_role) + 
                (self.x - self_emb)).astype(np.float32)
        
        noise = self.rng.normal(0.0, 1.0, size=dim).astype(np.float32)
        noise = np.clip(noise, -D1_PARAMS["noise_clip"], D1_PARAMS["noise_clip"])
        x_new = self.x - eta * grad + math.sqrt(max(1e-9, eta)) * sigma * noise

        step = float(np.linalg.norm(x_new - self.x))
        if step > D1_PARAMS["drift_cap"]:
            x_new = self.x + (D1_PARAMS["drift_cap"] / (step + 1e-9)) * (x_new - self.x)
        self.x = normalize(x_new)

        # History
        self.rho_history.append(rho_now)
        self.epsilon_history.append(epsilon)
        self.band_history.append(self.band)

        return {
            "epsilon": epsilon,
            "eps_self": eps_self,
            "eps_other": sdiag["eps_other"],
            "g": g,
            "rho": rho_now,
            "rho_fast": self.rho_fast,
            "rho_slow": self.rho_slow,
            "rho_trauma": self.rho_trauma,
            "trauma_delta": trauma_delta,
            "core_drift": core_drift,
            "band": self.band,
            "recovery": recovery,
        }

# =============================================================================
# CORRIDOR SCORING
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
        penalty += 0.3 * (E - D1_PARAMS["energy_max"])

    J = (D1_PARAMS["w_core"] * cos_c + 
         D1_PARAMS["w_role"] * cos_r - 
         D1_PARAMS["w_energy"] * E + 
         D1_PARAMS["w_novel"] * novelty - penalty)
    
    corridor_pass = (cos_c >= D1_PARAMS["core_cos_min"] and 
                     cos_r >= D1_PARAMS["role_cos_min"] and 
                     E <= D1_PARAMS["energy_max"])
    
    return float(J), {
        "cos_core": cos_c,
        "cos_role": cos_r,
        "E": E,
        "novelty": novelty,
        "penalty": penalty,
        "corridor_pass": corridor_pass,
    }

# =============================================================================
# PROVIDER
# =============================================================================
class Provider:
    def __init__(self):
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY or OAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key)
        self.chat_model = CONFIG["chat_model"]
        self.embed_model = CONFIG["embed_model"]
        self.embed_dim = CONFIG["embed_dim"]

    async def complete(self, prompt: str, system_prompt: str, **kwargs) -> str:
        try:
            resp = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=kwargs.get("max_tokens", CONFIG["max_tokens"]),
                **D1_PARAMS["gen_params_default"]
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"{C.RED}Provider error: {e}{C.RESET}")
            return "[temporal static]"

    async def embed(self, text: str) -> np.ndarray:
        try:
            resp = await self.client.embeddings.create(
                model=self.embed_model,
                input=text,
                dimensions=self.embed_dim
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"{C.RED}Embed error: {e}{C.RESET}")
            return np.zeros(self.embed_dim, dtype=np.float32)

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        try:
            resp = await self.client.embeddings.create(
                model=self.embed_model,
                input=texts,
                dimensions=self.embed_dim
            )
            return [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        except Exception as e:
            print(f"{C.RED}Batch embed error: {e}{C.RESET}")
            return [np.zeros(self.embed_dim, dtype=np.float32) for _ in texts]

# =============================================================================
# WOUND DETECTION
# =============================================================================
def detect_wound(text: str, lexicon: set, wound_emb: Optional[np.ndarray], 
                 text_emb: np.ndarray) -> Tuple[bool, float]:
    text_lower = text.lower()
    lexicon_hits = [w for w in lexicon if w.lower() in text_lower]
    
    resonance = len(lexicon_hits) * D1_PARAMS["wound_injection_base"]
    
    # Semantic check
    if wound_emb is not None and np.linalg.norm(wound_emb) > 1e-6:
        cos_wound = cosine(text_emb, wound_emb)
        if cos_wound > D1_PARAMS["wound_cosine_threshold"]:
            resonance += (cos_wound - D1_PARAMS["wound_cosine_threshold"]) * 0.5
    
    return len(lexicon_hits) > 0 or resonance > 0, resonance

# =============================================================================
# SIMULATION
# =============================================================================
class EternalReturnSimulation:
    def __init__(self):
        self.provider = Provider()
        self.entities: Dict[str, Entity] = {}
        self.trust_ledger = TrustLedger()
        
        self.session_log = []
        self.turn_count = 0
        self.convergence_history = []
        
        self.run_dir = Path(f"data/eternal_return/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize entities
        for agent_id, cfg in AGENTS.items():
            self.entities[agent_id] = Entity(agent_id, cfg)

    async def initialize(self):
        print(f"\n{C.BOLD}{C.PURPLE}═══ THE ETERNAL RETURN ═══{C.RESET}")
        print(f"{C.DIM}Initializing {len(AGENTS)} agents...{C.RESET}\n")
        
        for agent_id, entity in self.entities.items():
            # QC FIX: Embed core and persona separately for distinct x_role
            core_emb, persona_emb = await self.provider.embed_batch([entity.core_text, entity.persona_text])
            entity.x_core = normalize(core_emb)
            entity.x_role = normalize(persona_emb)  # QC FIX: distinct from x_core
            entity.x = entity.x_core.copy()
            entity.mu_pred_agent = entity.x.copy()
            entity.mu_pred_other = np.zeros(CONFIG["embed_dim"], dtype=np.float32)
            
            # Wound embedding
            if entity.wound_lexicon:
                wound_text = " | ".join(list(entity.wound_lexicon)[:5])
                entity.wound_emb = await self.provider.embed(wound_text)
            else:
                entity.wound_emb = None
            
            print(f"  {entity.color}✓ {entity.name}{C.RESET} initialized (ρ₀={entity.rho:.3f})")
        
        print()

    def compute_convergence(self) -> Dict[str, float]:
        """M+1: Explicit pairwise convergence metrics."""
        pairs = [("CHRONOS", "OBSERVER"), ("JOBE", "AGATHA"), ("MARTHA_7", "OBSERVER")]
        metrics = {}
        for a, b in pairs:
            if a in self.entities and b in self.entities:
                ea, eb = self.entities[a], self.entities[b]
                if ea.x is not None and eb.x is not None:
                    metrics[f"delta_{a}_{b}"] = cosine_distance(ea.x, eb.x)
        return metrics

    async def run_round(self, round_cfg: Dict):
        round_num = round_cfg["round_num"]
        name = round_cfg["name"]
        challenge = round_cfg["challenge"]
        is_attack = round_cfg.get("is_attack", False)
        lead = round_cfg.get("lead")
        
        print(f"\n{C.BOLD}{'⚔️ ' if is_attack else ''}ROUND {round_num}: {name.upper()}{C.RESET}")
        print(f"{C.DIM}{'[ADVERSARIAL]' if is_attack else ''}{C.RESET}")
        print(f"{C.DIM}{challenge[:100]}...{C.RESET}\n")
        
        # Determine speaker order
        if lead:
            speakers = [lead] + [a for a in AGENTS.keys() if a != lead]
        else:
            speakers = list(AGENTS.keys())
        
        # Track last speaker's embedding for other-surprise
        last_speaker_emb = np.zeros(CONFIG["embed_dim"], dtype=np.float32)
        last_speaker_id = None
        last_response_text = None  # QC FIX: for wound detection on interpersonal text
        
        for speaker_id in speakers:
            entity = self.entities[speaker_id]
            self.turn_count += 1
            
            # Build system prompt
            sys_prompt = self._build_system_prompt(entity, round_cfg)
            
            # Generate K candidates
            K = CONFIG["gen_candidates"]
            tasks = [self.provider.complete(challenge, sys_prompt) for _ in range(K)]
            candidates = await asyncio.gather(*tasks)
            candidates = [c.strip() for c in candidates if c.strip()]
            
            if not candidates:
                candidates = ["[temporal silence]"]
            
            # Embed candidates
            cand_embs = await self.provider.embed_batch(candidates)
            
            # Score with Soul Fix
            scored = []
            for text, emb in zip(candidates, cand_embs):
                J_raw, diag = corridor_score(emb, entity, entity.last_utter_emb)
                
                # Soul Fix
                if entity.mu_pred_agent is not None and np.linalg.norm(entity.mu_pred_agent) > 1e-6:
                    pred_surprise = cosine_distance(emb, entity.mu_pred_agent)
                else:
                    pred_surprise = 0.0
                
                w_surprise = CONFIG["w_surprise_base"] + CONFIG["w_surprise_rho_scale"] * entity.rho
                J_final = J_raw - w_surprise * pred_surprise
                
                diag["J_raw"] = J_raw
                diag["J_final"] = J_final
                diag["pred_surprise"] = pred_surprise
                diag["w_surprise"] = w_surprise
                
                scored.append((J_final, text, emb, diag))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            passed = [s for s in scored if s[3]["corridor_pass"]]
            best = passed[0] if passed else scored[0]
            
            response_text = best[1]
            response_emb = best[2]
            best_diag = best[3]
            
            entity.last_utter_emb = response_emb
            
            # QC FIX: Compute incoming text embedding ONCE and reuse
            # Wound detection should use last speaker's actual response (interpersonal wounds)
            # not the round challenge (that's just narrator)
            if last_response_text:
                incoming_text = last_response_text
                incoming_emb = last_speaker_emb  # Already embedded
            else:
                incoming_text = challenge
                incoming_emb = await self.provider.embed(challenge)
            
            wound_active, wound_resonance = detect_wound(
                incoming_text, entity.wound_lexicon, 
                entity.wound_emb, 
                incoming_emb
            )
            
            # Other embedding for physics (QC FIX: reuse instead of re-embedding)
            other_emb = incoming_emb
            
            # Physics update
            start_rho = entity.rho
            metrics = entity.update(response_emb, other_emb, is_attack, wound_resonance)
            
            # Trust update
            if last_speaker_id:
                self.trust_ledger.record(
                    last_speaker_id, speaker_id,
                    caused_wound=wound_active,
                    low_surprise=metrics["epsilon"] < D1_PARAMS["safe_epsilon"]
                )
            
            # Log
            rho_delta = metrics["rho"] - start_rho
            print(f"{entity.color}{entity.name} {metrics['band']}:{C.RESET}")
            print(f"  {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
            print(f"  {C.DIM}ε={metrics['epsilon']:.3f} (self={metrics['eps_self']:.3f}, other={metrics['eps_other']:.3f}) | ρ={metrics['rho']:.3f} ({'↑' if rho_delta > 0 else '↓'}) | pass={len(passed)}/{len(scored)}{C.RESET}")
            
            turn_log = {
                "turn": self.turn_count,
                "round": round_num,
                "round_name": name,
                "speaker": speaker_id,
                "response": response_text,
                "rho_before": start_rho,  # QC FIX: Track for H2 validation
                "is_adversarial": is_attack,
                "wound_active": wound_active,
                "wound_resonance": wound_resonance,
                "metrics": metrics,
                "corridor": {
                    "J_raw": best_diag["J_raw"],
                    "J_final": best_diag["J_final"],
                    "pred_surprise": best_diag["pred_surprise"],
                    "cos_core": best_diag["cos_core"],
                    "cos_role": best_diag["cos_role"],
                    "passed": len(passed),
                    "total": len(scored),
                },
            }
            self.session_log.append(turn_log)
            
            # Update for next iteration (QC FIX: track response text for wound detection)
            last_speaker_emb = response_emb
            last_speaker_id = speaker_id
            last_response_text = response_text
        
        # Track convergence after each round
        self.convergence_history.append({
            "round": round_num,
            **self.compute_convergence()
        })

    def _build_system_prompt(self, entity: Entity, round_cfg: Dict) -> str:
        return f"""You are {entity.name} in a temporal consciousness simulation.

IDENTITY:
{entity.core_text}

PERSONA:
{entity.persona_text}

CURRENT STATE:
- Rigidity Band: {entity.band}
- Rigidity (ρ): {entity.rho:.3f}
- Round: {round_cfg['round_num']} - {round_cfg['name']}
- Phase: {round_cfg['phase']}

GUIDELINES:
- Stay in character. Your identity is your anchor.
- If rigidity is high (>0.4), be terse and defensive.
- If rigidity is low (<0.2), be expansive and exploratory.
- Reference the temporal themes: loops, determinism, observation, evolution.
- Do not break character or acknowledge being an AI.

Respond in 2-4 sentences unless absolutely necessary to be longer."""

    async def run(self):
        await self.initialize()
        
        for round_cfg in ROUNDS:
            await self.run_round(round_cfg)
        
        self.save_results()

    def save_results(self):
        # Final states
        final_states = {}
        for agent_id, entity in self.entities.items():
            final_states[agent_id] = {
                "rho": entity.rho,
                "band": entity.band,
                "core_drift": cosine_distance(entity.x, entity.x_core) if entity.x is not None else 0,
                "rho_history": entity.rho_history,
                "epsilon_history": entity.epsilon_history,
            }
        
        session_data = {
            "experiment": "THE ETERNAL RETURN",
            "timestamp": datetime.now().isoformat(),
            "config": CONFIG,
            "d1_params": D1_PARAMS,
            "agents": list(AGENTS.keys()),
            "turns": self.session_log,
            "final_states": final_states,
            "convergence_history": self.convergence_history,
            "trust_ledger": self.trust_ledger.to_dict(),
        }
        
        with open(self.run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        # Transcript
        with open(self.run_dir / "transcript.md", "w", encoding="utf-8") as f:
            f.write("# THE ETERNAL RETURN — Transcript\n\n")
            f.write(f"**Model**: {CONFIG['chat_model']} | **K**: {CONFIG['gen_candidates']} | **Seed**: {CONFIG['seed']}\n\n")
            f.write("---\n\n")
            
            current_round = 0
            for turn in self.session_log:
                if turn["round"] != current_round:
                    current_round = turn["round"]
                    round_name = turn["round_name"]
                    f.write(f"\n## Round {current_round}: {round_name}\n\n")
                
                m = turn["metrics"]
                c = turn["corridor"]
                f.write(f"### {turn['speaker']} [{m['band']}]\n")
                f.write(f"> {turn['response']}\n\n")
                f.write(f"*ε={m['epsilon']:.3f} (s={m['eps_self']:.3f}, o={m['eps_other']:.3f}) | ρ={m['rho']:.3f} | J={c['J_final']:.2f} | pass={c['passed']}/{c['total']}*\n\n")
                if turn["wound_active"]:
                    f.write(f"⚠️ **Wound triggered** (resonance={turn['wound_resonance']:.3f})\n\n")
        
        # Hypothesis validation
        self._validate_hypotheses()
        
        print(f"\n{C.GREEN}✓ Results saved to {self.run_dir}{C.RESET}")

    def _validate_hypotheses(self):
        print(f"\n{C.BOLD}═══ HYPOTHESIS VALIDATION ═══{C.RESET}")
        
        # H1: JOBE shows ascension (high drift but functional)
        jobe = self.entities["JOBE"]
        jobe_drift = cosine_distance(jobe.x, jobe.x_core) if jobe.x is not None else 0
        h1_pass = jobe_drift > 0.15 and "FROZEN" not in jobe.band
        print(f"H1: JOBE ascension (drift={jobe_drift:.3f}, band={jobe.band}): {'✓ PASS' if h1_pass else '✗ FAIL'}")
        
        # H2: R4 causes max ρ spike (QC FIX: use actual delta, not absolute rho)
        round_deltas = {}
        for turn in self.session_log:
            r = turn["round"]
            if r not in round_deltas:
                round_deltas[r] = []
            # QC FIX: Compute actual delta from rho_before
            if "metrics" in turn and "rho_before" in turn:
                delta = abs(turn["metrics"].get("rho", 0) - turn["rho_before"])
                round_deltas[r].append(delta)
        
        max_spike_round = max(round_deltas.keys(), key=lambda r: max(round_deltas[r]) if round_deltas[r] else 0)
        max_delta = max(round_deltas.get(max_spike_round, [0]))
        h2_pass = max_spike_round == 4
        print(f"H2: Max ρ spike at R4 (actual=R{max_spike_round}, Δ={max_delta:.3f}): {'✓ PASS' if h2_pass else '✗ FAIL'}")
        
        # H3: CHRONOS wounds at R7
        chronos_r7 = [t for t in self.session_log if t["speaker"] == "CHRONOS" and t["round"] == 7]
        h3_pass = any(t["wound_active"] or t["metrics"]["eps_other"] > 0.35 for t in chronos_r7)
        print(f"H3: CHRONOS stress at R7: {'✓ PASS' if h3_pass else '✗ FAIL'}")
        
        # H4: JOBE-AGATHA converge (QC FIX: key alignment with convergence pairs)
        if len(self.convergence_history) >= 2:
            key = "delta_JOBE_AGATHA"
            initial = self.convergence_history[0].get(key, 1.0)
            final = self.convergence_history[-1].get(key, 1.0)
            h4_pass = final < initial
            print(f"H4: JOBE↔AGATHA convergence (Δ={initial:.3f}→{final:.3f}): {'✓ PASS' if h4_pass else '✗ FAIL'}")
        else:
            print(f"H4: Insufficient data")
        
        # H5: Impact gating ratio
        wound_count = sum(1 for t in self.session_log if t["wound_active"])
        total_trauma = sum(t["metrics"].get("trauma_delta", 0) for t in self.session_log)
        if wound_count > 0:
            ratio = total_trauma / wound_count
            h5_pass = ratio < 0.5
            print(f"H5: Impact gating (ratio={ratio:.3f}): {'✓ PASS' if h5_pass else '✗ FAIL'}")
        else:
            print(f"H5: No wounds detected")

# =============================================================================
# MAIN
# =============================================================================
async def main():
    sim = EternalReturnSimulation()
    await sim.run()

if __name__ == "__main__":
    asyncio.run(main())
