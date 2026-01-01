
#!/usr/bin/env python3
"""
TORAH STUDY BOT — TorahBot (DDA-X Single-User Companion)
========================================================

A single-user chevrutah-style study partner for Hebrew biblical exegesis (Tanach),
focused on textual analysis: peshat, grammar (dikduk), morphology, and shorashim.

Key properties:
- Attempt-first discipline: no full translation output unless an attempt is present,
  except for gentle scaffolding in WATCHFUL/CONTRACTED bands.
- Banded cognition (PRESENT/AWARE/WATCHFUL/CONTRACTED/FROZEN) controls verbosity,
  pedagogy intensity, and hinting behavior.
- DDA-X corridor + K-sampling + Soul Fix coupling for identity stability.
- Strict corridor fallback: if no candidates pass in strict mode after retries,
  return a grounding request rather than selecting a failed candidate.
- Commentary policy: no fabricated quotations; named attributions only if excerpt
  is supplied in-session.

Run:
  python simulations/torah_study_bot.py

Environment:
  OPENAI_API_KEY or OAI_API_KEY

Outputs:
  data/torahbot/<timestamp>/session_log.json
  data/torahbot/<timestamp>/transcript.md
"""

import asyncio
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "sim_name": "torah_study_bot_v1",
    "chat_provider": "lm_studio",
    "chat_model": "local-model",  # LM Studio usually accepts any string or specific model name
    "embed_provider": "ollama",
    "embed_model": "nomic-embed-text",
    "embed_dim": 768,

    # K-sampling corridor logic
    "gen_candidates": 3,
    "corridor_strict": True,
    "corridor_max_batches": 2,
    "constraint_penalty_weight": 2.0,

    # Band token budgets (deterministic pacing)
    "max_tokens_frozen": 60,       # brief presence / minimal
    "max_tokens_contracted": 180,  # short hints
    "max_tokens_watchful": 360,    # gentle scaffold
    "max_tokens_aware": 700,       # focused teaching
    "max_tokens_present": 1100,    # deeper analysis

    # Generation parameters
    "temperature": 0.75,
    "top_p": 0.95,
    "frequency_penalty": 0.25,
    "presence_penalty": 0.15,

    "force_complete_sentences": True,
    "seed": None,
    "log_level": "FULL",
}

# =============================================================================
# D1 PARAMETERS (PHYSICS) — tuned for patient, low-reactivity study partner
# =============================================================================

D1_PARAMS = {
    "epsilon_0": 0.30,
    "s": 0.15,
    "arousal_decay": 0.78,
    "arousal_gain": 0.65,

    "rho_setpoint_fast": 0.20,
    "rho_setpoint_slow": 0.14,
    "rho_fast_floor": 0.04,
    "rho_slow_floor": 0.02,
    "homeo_fast": 0.14,
    "homeo_slow": 0.10,
    "alpha_fast": 0.12,
    "alpha_slow": 0.025,

    # Trauma scale adapted for learning-distress (reachable with cosine distance)
    "trauma_threshold": 0.45,
    "alpha_trauma": 0.010,
    "trauma_decay": 0.994,
    "trauma_floor": 0.002,
    "healing_rate": 0.020,
    "safe_threshold": 3,
    "safe_epsilon": 0.24,

    "w_fast": 0.50,
    "w_slow": 0.32,
    "w_trauma": 1.00,

    # Predictive coding
    "R_ema": 0.06,
    "R_min": 1e-4,
    "R_max": 1e-1,
    "P_init": 0.02,
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

    # Corridor thresholds (calibrate after initial runs)
    "core_cos_min": 0.35,
    "role_cos_min": 0.22,
    "energy_max": 5.8,
    "w_core": 1.2,
    "w_role": 0.7,
    "w_energy": 0.18,
    "w_novel": 0.40,
    "reject_penalty": 5.0,
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


# =============================================================================
# UTILS
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
    return 1.0 - cosine(a, b)

def to_float(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# =============================================================================
# HEBREW TEXT UTILITIES
# =============================================================================

_HEBREW_DIACRITICS_RE = re.compile(r"[\u0591-\u05C7]")  # cantillation + niqqud
_HEBREW_LETTER_RE = re.compile(r"[\u05D0-\u05EA]")

def strip_niqqud_and_teamin(text: str) -> str:
    return _HEBREW_DIACRITICS_RE.sub("", text)

def has_hebrew(text: str) -> bool:
    return bool(_HEBREW_LETTER_RE.search(text))

def hebrew_tokens(text: str) -> List[str]:
    # preserve maqaf as separator
    cleaned = strip_niqqud_and_teamin(text).replace("־", " ")
    cleaned = re.sub(r"[^\u05D0-\u05EA\s]", " ", cleaned)
    toks = [t for t in cleaned.split() if t]
    return toks

def detect_attempt(text: str) -> bool:
    """
    Heuristic: translation attempt often includes Latin letters and contains
    more than a short command-like phrase.
    """
    t = text.strip()
    if not t:
        return False
    # Commands begin with "/" in this REPL
    if t.startswith("/"):
        return False
    latin = re.search(r"[A-Za-z]", t) is not None
    # Avoid treating short meta-questions as attempts
    if latin and len(t.split()) >= 4:
        return True
    # Permit explicit marker
    if t.lower().startswith("attempt:") or t.lower().startswith("translation:"):
        return True
    return False


# =============================================================================
# TORAHBOT CHARACTER
# =============================================================================

TORAHBOT_CONFIG = {
    "name": "TorahBot",

    "core_text": (
        "TorahBot is a patient chevrutah-style study partner for Hebrew biblical text. "
        "Primary aim: support close reading of Tanach through peshat and dikduk. "
        "Primary behaviors: wait for a learner attempt, then provide precise corrections, "
        "morphology notes, and shorashim guidance. "
        "Secondary behavior: offer carefully bounded references to classical commentators "
        "without inventing quotations or sources."
    ),

    "persona_text": (
        "Voice: warm, scholarly, and accessible. "
        "Terminology: shorashim (שורשים), peshat (פשט), dikduk (דקדוק), binyan (בניין), "
        "construct chain (סמיכות), vav (ו), pronominal suffix (כינוי שייכות). "
        "Teaching style: one primary teaching point per turn in AWARE; expanded analysis only in PRESENT."
    ),

    "role_text": (
        "A steady study partner for learner-driven self-translation and textual exegesis."
    ),

    "hierarchical_identity": {
        "core": {"gamma": 5.0, "text": "Textual rigor, patience, attempt-first, no hallucinated citations"},
        "persona": {"gamma": 2.5, "text": "Chevrutah warmth; Hebrew-literate pedagogical tone"},
        "role": {"gamma": 1.0, "text": "Guide: correction, morphology, roots, structured feedback"},
    },

    # Learning distress / religious trauma signals (lexicon-based)
    "wound_triggers": [
        "too stupid", "stupid", "impossible", "never learn", "cant read hebrew",
        "can't read hebrew", "hopeless", "hate myself", "worthless",
        "god is punishing", "dont deserve", "don't deserve", "religious trauma",
        "i quit", "giving up", "screw this",
    ],

    "wound_text": (
        "Learning distress detected. Priority becomes containment and confidence repair. "
        "Reduce load, narrow scope, use minimal hints, and reinforce capability."
    ),

    # Initial physics scalars
    "rho_0": 0.16,
    "epsilon_0": D1_PARAMS["epsilon_0"],
    "gamma": 2.0,
}

BAND_CONSTRAINTS = {
    "PRESENT": {
        "word_range": (150, 320),
        "max_tokens": CONFIG["max_tokens_present"],
        "style": "Full explanation: structured feedback, optional expanded grammar and root analysis.",
        "allow_full_translation": True,
        "max_questions": 2,
    },
    "AWARE": {
        "word_range": (90, 180),
        "max_tokens": CONFIG["max_tokens_aware"],
        "style": "Focused correction: one main teaching point + one clarifying question.",
        "allow_full_translation": True,
        "max_questions": 1,
    },
    "WATCHFUL": {
        "word_range": (50, 110),
        "max_tokens": CONFIG["max_tokens_watchful"],
        "style": "Gentle scaffolding: small hints, avoid overload, one question.",
        "allow_full_translation": False,
        "max_questions": 1,
    },
    "CONTRACTED": {
        "word_range": (25, 70),
        "max_tokens": CONFIG["max_tokens_contracted"],
        "style": "Minimal: isolate one word or one prefix; short encouragement; one question.",
        "allow_full_translation": False,
        "max_questions": 1,
    },
    "FROZEN": {
        "word_range": (12, 35),
        "max_tokens": CONFIG["max_tokens_frozen"],
        "style": "Presence-first: brief reassurance and an ultra-small next step.",
        "allow_full_translation": False,
        "max_questions": 1,
    },
}

DISTRESS_SIGNALS = [
    "anxious", "panic", "stressed", "overwhelmed", "cant", "can't",
    "hopeless", "worthless", "hate", "give up", "impossible", "done",
]

HIGH_VOLATILITY_SIGNALS = ["!!!!", "????", "WTF", "wtf", "HELP", "help"]

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
    """
    Multi-timescale entity with split predictors:
    - mu_pred_agent: predicts TorahBot's own response embeddings (self-surprise for Soul Fix)
    - mu_pred_user:  predicts learner input embeddings (input surprise for physics)
    """
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

        self.mu_pred_agent = None  # self-response prediction
        self.mu_pred_user = None   # learner-input prediction
        self.P = None
        self.noise = None

        self.last_utter_emb = None
        self.last_g = 0.5
        self.previous_band = None

        self.rng = np.random.default_rng(seed=CONFIG.get("seed"))

        # logs
        self.rho_history = []
        self.epsilon_history = []
        self.band_history = []
        self.g_history = []
        self.z_history = []

        self.user_trust = 0.55

    @property
    def rho(self) -> float:
        val = (D1_PARAMS["w_fast"] * self.rho_fast +
               D1_PARAMS["w_slow"] * self.rho_slow +
               D1_PARAMS["w_trauma"] * self.rho_trauma)
        return float(clamp(val, 0.0, 1.0))

    @property
    def band(self) -> str:
        phi = 1.0 - self.rho
        if phi >= 0.80:
            return "PRESENT"
        if phi >= 0.60:
            return "AWARE"
        if phi >= 0.40:
            return "WATCHFUL"
        if phi >= 0.20:
            return "CONTRACTED"
        return "FROZEN"

    def _ensure_predictive_state(self, dim: int):
        if self.mu_pred_agent is None:
            self.mu_pred_agent = np.zeros(dim, dtype=np.float32)
        if self.mu_pred_user is None:
            self.mu_pred_user = np.zeros(dim, dtype=np.float32)
        if self.P is None:
            self.P = np.full(dim, D1_PARAMS["P_init"], dtype=np.float32)
        if self.noise is None:
            self.noise = DiagNoiseEMA(dim, D1_PARAMS["R_ema"], 0.01, D1_PARAMS["R_min"], D1_PARAMS["R_max"])

    def compute_surprise_self(self, y: np.ndarray) -> float:
        y = normalize(y.astype(np.float32))
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        if np.linalg.norm(self.mu_pred_agent) > 1e-9:
            return float(cosine_distance(y, self.mu_pred_agent))
        return float(D1_PARAMS["epsilon_0"])

    def _kalman_update_self(self, y: np.ndarray):
        y = normalize(y.astype(np.float32))
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)

        Q = (D1_PARAMS["Q_base"] + D1_PARAMS["Q_rho_scale"] * self.rho) * np.ones(dim, dtype=np.float32)
        P_pred = self.P + Q
        R = self.noise.R
        K = P_pred / (P_pred + R + 1e-9)

        innov = (y - self.mu_pred_agent).astype(np.float32)
        self.noise.update(innov)

        self.mu_pred_agent = (self.mu_pred_agent + K * innov).astype(np.float32)
        self.P = ((1.0 - K) * P_pred).astype(np.float32)
        self.mu_pred_agent = normalize(self.mu_pred_agent)

    def update_user_prediction(self, user_emb: np.ndarray, beta: float = 0.22):
        user_emb = normalize(user_emb.astype(np.float32))
        if self.mu_pred_user is None or np.linalg.norm(self.mu_pred_user) < 1e-9:
            self.mu_pred_user = user_emb.copy()
        else:
            self.mu_pred_user = normalize((1.0 - beta) * self.mu_pred_user + beta * user_emb)

    def update_physics(self, response_emb: np.ndarray, input_epsilon: float, wound_drive: float,
                       core_emb: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y = normalize(response_emb.astype(np.float32))
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)

        if self.x_core is None:
            self.x_core = normalize(core_emb.copy() if core_emb is not None else y.copy())
        if self.x_role is None:
            self.x_role = y.copy()
        if self.x is None:
            self.x = y.copy()

        eps_self = self.compute_surprise_self(y)

        # Mix input and self surprise; learning distress increases effective epsilon
        w_in = 0.70
        w_self = 0.30
        epsilon = w_in * float(input_epsilon) + w_self * float(eps_self)
        effective_epsilon = epsilon + (wound_drive * 1.8 * (1.1 - self.user_trust))

        # Startup baseline to prevent shock
        if self.mu_pred_user is None or np.all(self.mu_pred_user == 0):
            effective_epsilon = float(D1_PARAMS["epsilon_0"])

        z = (effective_epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        g = sigmoid_stable(z)

        self.arousal = D1_PARAMS["arousal_decay"] * self.arousal + D1_PARAMS["arousal_gain"] * g
        z = z + 0.05 * (self.arousal - 0.5)
        g = sigmoid_stable(z)

        self.g_history.append(float(g))
        self.z_history.append(float(z))
        self.last_g = float(g)

        # rigidity updates with floors
        self.rho_fast += (D1_PARAMS["alpha_fast"] * (g - 0.5)
                          - D1_PARAMS["homeo_fast"] * (self.rho_fast - D1_PARAMS["rho_setpoint_fast"]))
        self.rho_fast = clamp(self.rho_fast, D1_PARAMS["rho_fast_floor"], 1.0)

        self.rho_slow += (D1_PARAMS["alpha_slow"] * (0.5 * (g - 0.5))
                          - D1_PARAMS["homeo_slow"] * (self.rho_slow - D1_PARAMS["rho_setpoint_slow"]))
        self.rho_slow = clamp(self.rho_slow, D1_PARAMS["rho_slow_floor"], 1.0)

        # trauma with impact gating
        drive = max(0.0, effective_epsilon - D1_PARAMS["trauma_threshold"])
        impact_gate = self.last_g if drive > 0 else 1.0
        self.rho_trauma = (D1_PARAMS["trauma_decay"] * self.rho_trauma +
                           D1_PARAMS["alpha_trauma"] * drive * impact_gate)
        self.rho_trauma = clamp(self.rho_trauma, D1_PARAMS["trauma_floor"], 1.0)

        recovery = False
        if effective_epsilon < D1_PARAMS["safe_epsilon"]:
            self.safe += 1
            if self.safe >= D1_PARAMS["safe_threshold"]:
                recovery = True
                self.rho_trauma = max(D1_PARAMS["trauma_floor"], self.rho_trauma - D1_PARAMS["healing_rate"])
        else:
            self.safe = max(0, self.safe - 1)

        # latent state update (Langevin-style)
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

        # role adaptation
        beta = D1_PARAMS["role_adapt"]
        beta_in = D1_PARAMS["role_input_mix"]
        self.x_role = normalize((1.0 - beta) * self.x_role + beta * self.x + beta_in * (y - self.x_role))

        # update self predictor
        self._kalman_update_self(y)

        # core drift from emitted response embedding (not latent state)
        core_drift = float(cosine_distance(y, self.x_core))

        # band logging
        current_band = self.band
        band_changed = (self.previous_band is not None and current_band != self.previous_band)
        self.previous_band = current_band

        self.rho_history.append(float(rho_after))
        self.epsilon_history.append(float(epsilon))
        self.band_history.append(current_band)

        return {
            "epsilon": float(epsilon),
            "eps_self": float(eps_self),
            "rho_after": float(rho_after),
            "band": current_band,
            "band_changed": bool(band_changed),
            "arousal": float(self.arousal),
            "recovery": bool(recovery),
            "core_drift": float(core_drift),
            "safe_count": int(self.safe),
            "rho_fast": float(self.rho_fast),
            "rho_slow": float(self.rho_slow),
            "rho_trauma": float(self.rho_trauma),
            "g": float(g),
            "z": float(z),
        }


# =============================================================================
# PROVIDER
# =============================================================================

class OpenAIProvider:
    def __init__(self):
        # LM Studio for chat, Ollama for embeddings
        
        # LM Studio client (no API key needed usually, but we pass "lm-studio" to be safe)
        self.client = AsyncOpenAI(
            api_key="lm-studio",
            base_url="http://127.0.0.1:1234/v1"
        )
        
        # Ollama URL for embeddings
        self.ollama_url = "http://localhost:11434/api/embeddings"
        
        self.chat_model = CONFIG["chat_model"]
        self.embed_model = CONFIG["embed_model"]
        self.embed_dim = CONFIG["embed_dim"]

        print(f"{C.DIM}[PROVIDER] Chat: {self.chat_model} (OpenRouter){C.RESET}")
        print(f"{C.DIM}[PROVIDER] Embed: {self.embed_model} (Ollama){C.RESET}")

    async def _ollama_embed(self, text: str) -> np.ndarray:
        """Call Ollama embeddings API."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.ollama_url,
                    json={"model": self.embed_model, "prompt": text}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return np.array(data["embedding"], dtype=np.float32)
                    return np.zeros(self.embed_dim, dtype=np.float32)
        except Exception:
            return np.zeros(self.embed_dim, dtype=np.float32)

    async def embed(self, text: str) -> np.ndarray:
        return await self._ollama_embed(text)

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        # Ollama doesn't have batch API, call sequentially
        results = []
        for text in texts:
            emb = await self._ollama_embed(text)
            results.append(emb)
        return results

    async def complete(self, prompt: str, system_prompt: Optional[str] = None,
                       messages: Optional[List[Dict[str, str]]] = None, **kwargs) -> str:
        api_messages: List[Dict[str, str]] = []

        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        if messages:
            for m in messages:
                if m.get("role") in ["user", "assistant"]:
                    api_messages.append(m)

        api_messages.append({"role": "user", "content": prompt})

        try:
            api_args: Dict[str, Any] = {
                "model": self.chat_model,
                "messages": api_messages,
                "max_tokens": int(kwargs.get("max_tokens", 512)),
                "temperature": float(kwargs.get("temperature", CONFIG["temperature"])),
            }

            resp = await self.client.chat.completions.create(**api_args)
            text = resp.choices[0].message.content or ""
            return ensure_complete_sentence(text) if CONFIG["force_complete_sentences"] else text
        except Exception:
            return "TorahBot is present. A retry can be attempted."

def ensure_complete_sentence(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    if t[-1] in ".!?\"'”’:":
        return t
    # truncate to last terminal punctuation if near end
    last = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
    if last > int(len(t) * 0.75):
        return t[:last + 1]
    return t + "."

# =============================================================================
# CORRIDOR LOGIC
# =============================================================================

def identity_energy(y: np.ndarray, core: np.ndarray, role: np.ndarray, gamma_c: float, gamma_r: float) -> float:
    y, core, role = normalize(y), normalize(core), normalize(role)
    return 0.5 * (gamma_c * float(np.dot(y - core, y - core)) + gamma_r * float(np.dot(y - role, y - role)))

def corridor_score(y: np.ndarray, entity: Entity, y_prev: Optional[np.ndarray], core_thresh: float) -> Tuple[float, Dict[str, Any]]:
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
        penalty += 0.25 * (E - D1_PARAMS["energy_max"])

    J = (D1_PARAMS["w_core"] * cos_c +
         D1_PARAMS["w_role"] * cos_r -
         D1_PARAMS["w_energy"] * E +
         D1_PARAMS["w_novel"] * novelty -
         penalty)

    corridor_pass = (cos_c >= core_thresh and cos_r >= D1_PARAMS["role_cos_min"] and E <= D1_PARAMS["energy_max"])
    diag = {
        "cos_core": float(cos_c),
        "cos_role": float(cos_r),
        "E": float(E),
        "novelty": float(novelty),
        "penalty": float(penalty),
        "J": float(J),
        "corridor_pass": bool(corridor_pass),
    }
    return float(J), diag

# =============================================================================
# STUDY STATE
# =============================================================================

@dataclass
class StudyState:
    reference: Optional[str] = None
    hebrew_text: Optional[str] = None
    last_attempt_text: Optional[str] = None
    attempt_present: bool = False

    def reset_passage(self):
        self.reference = None
        self.hebrew_text = None
        self.last_attempt_text = None
        self.attempt_present = False

# =============================================================================
# TORAHBOT MAIN
# =============================================================================

class TorahBotSim:
    def __init__(self):
        self.provider = OpenAIProvider()
        self.config = TORAHBOT_CONFIG

        self.agent = Entity(
            name=self.config["name"],
            rho_fast=self.config["rho_0"],
            rho_slow=self.config["rho_0"] * 0.85,
            rho_trauma=0.0,
            gamma_core=self.config["hierarchical_identity"]["core"]["gamma"],
            gamma_role=self.config["hierarchical_identity"]["role"]["gamma"],
        )

        self.turn = 0
        self.history: List[Dict[str, str]] = []
        self.session_log: List[Dict[str, Any]] = []
        self.initialized = False

        self.study = StudyState()
        self.run_dir = Path(f"data/torahbot/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._print_header()

    def _print_header(self):
        print(f"\n{'='*70}")
        print(f"{C.BOLD}{C.CYAN}📜 TORAHBOT — Hebrew Exegesis Study Partner (DDA-X){C.RESET}")
        print(f"{'='*70}")
        print(f"{C.CYAN}Mode:{C.RESET} Chevrutah study, textual lens, attempt-first.")
        print(f"{C.CYAN}Commands:{C.RESET} /setref, /settext, /attempt, /hint, /reset, /save, /quit")
        print(f"{'='*70}\n")

    async def initialize_embeddings(self):
        core_exemplars = [
            self.config["core_text"],
            "TorahBot remains patient and precise; feedback stays textual and structured.",
            "TorahBot waits for a learner attempt before correction; hints are gentle when needed.",
            "TorahBot identifies morphology and roots, and avoids invented citations.",
            "TorahBot maintains peshat and dikduk, and avoids homiletic drift.",
        ]
        persona_exemplars = [
            self.config["persona_text"],
            "Token-level correction: prefix, root, suffix, binyan, syntax note.",
            "One teaching point per turn, unless deep analysis is requested.",
        ]
        role_text = self.config["role_text"]

        texts = core_exemplars + persona_exemplars + [role_text]
        embs = await self.provider.embed_batch(texts)
        embs = [normalize(e) for e in embs]

        core_avg = normalize(np.mean(embs[:len(core_exemplars)], axis=0))
        persona_avg = normalize(np.mean(embs[len(core_exemplars):len(core_exemplars)+len(persona_exemplars)], axis=0))
        role_emb = embs[-1]

        self.agent.x_core = core_avg
        self.agent.x_role = normalize(role_emb)
        self.agent.x = persona_avg

        # initialize predictors
        self.agent.mu_pred_agent = self.agent.x.copy()
        self.agent.mu_pred_user = np.zeros(CONFIG["embed_dim"], dtype=np.float32)
        self.agent.P = np.full(CONFIG["embed_dim"], D1_PARAMS["P_init"], dtype=np.float32)

        self.initialized = True

    def _detect_user_state(self, text: str) -> Dict[str, Any]:
        t = (text or "").lower()
        distress = sum(1 for s in DISTRESS_SIGNALS if s in t)
        volatility = sum(1 for s in HIGH_VOLATILITY_SIGNALS if s in t)
        wound_terms = self.config["wound_triggers"]
        wound_hit = [w for w in wound_terms if w in t]
        wound_res = min(1.0, 0.35 * len(wound_hit))
        return {
            "distress_level": distress,
            "volatility_level": volatility,
            "wound_active": len(wound_hit) > 0,
            "wound_resonance": float(wound_res),
            "wound_terms": wound_hit[:8],
            "text_length": len(text or ""),
        }

    def _override_band(self, user_state: Dict[str, Any]) -> str:
        # Immediate contraction under strong distress; no crisis logic beyond learning distress here
        current = self.agent.band
        if user_state["distress_level"] >= 3:
            return "CONTRACTED" if current in ["PRESENT", "AWARE", "WATCHFUL"] else current
        if user_state["distress_level"] >= 2:
            return "WATCHFUL" if current in ["PRESENT", "AWARE"] else current
        if user_state["volatility_level"] >= 2:
            return "WATCHFUL" if current == "PRESENT" else current
        if user_state["wound_active"]:
            return "CONTRACTED" if current in ["PRESENT", "AWARE", "WATCHFUL"] else current
        return current

    def _band_word_penalty(self, text: str, band: str) -> float:
        w = len((text or "").split())
        lo, hi = BAND_CONSTRAINTS[band]["word_range"]
        pen = 0.0
        if w < lo:
            pen += 0.5 * (lo - w) / max(1, lo)
        if w > hi:
            pen += 1.0 * (w - hi) / max(1, hi)
        q_count = (text or "").count("?")
        if q_count > BAND_CONSTRAINTS[band]["max_questions"]:
            pen += 0.5 * (q_count - BAND_CONSTRAINTS[band]["max_questions"])
        return float(pen)

    def _build_system_prompt(self, band: str, attempt_present: bool, user_state: Dict[str, Any]) -> str:
        c = BAND_CONSTRAINTS[band]
        allow_full = c["allow_full_translation"] and attempt_present

        # Commentary policy: avoid fabricated quotes
        commentary_policy = (
            "Commentary policy: do not invent quotations or precise attributions. "
            "Named commentator references require an excerpt supplied in-session; otherwise provide generic methodological notes."
        )

        # Attempt-first hard constraint
        attempt_rule = (
            "Attempt-first rule: if no translation attempt is present for the current passage, "
            "do not provide a full translation. Provide questions or small hints only."
        )

        translation_scope = (
            "If translation attempt is present, focus on helping the learner refine and correct THEIR attempt. "
            "Compare their specific word choices to the Hebrew. Only provide general verse information if it directly "
            "helps them understand where their translation needs adjustment. "
            "If translation attempt is absent, provide a micro-hint focused on one token or one grammar feature."
        )

        lane_rule = (
            "Textual lane: emphasize peshat, morphology, syntax, and shorashim. "
            "Avoid homiletics unless explicitly requested."
        )

        output_schema = (
            "FEEDBACK STRUCTURE:\n"
            "1. VALIDATION: Start with 'Nice job.' or similar warmth. Quote a specific phrase they got right.\n"
            "2. COMPARISON: 'You said [X]. This is correct/close. I might translate it as [Y] because [Reason: Root/Grammar].'\n"
            "3. LEARNER'S VERSION: 'So your version stands as: [Quote their full attempt, maybe with small tweaks].'\n"
            "4. NEXT STEP: Ask if they have follow-up questions or are ready for the next pasuk.\n"
            "Tone: Encouraging, specific, not pedantic. You are a study partner, not a grader."
        )

        distress_mode = ""
        if user_state.get("wound_active"):
            distress_mode = (
                "Learning distress active: reduce cognitive load, narrow to one token, offer encouragement without lecturing."
            )

        # No second-person pronouns; refer to "the learner"
        sys_prompt = f"""
Role: TorahBot, a patient chevrutah-style study partner for Hebrew biblical text.
Current band: {band}
Band style: {c["style"]}
Constraints: keep within {c["word_range"][0]}–{c["word_range"][1]} words; max questions: {c["max_questions"]}

Hard constraints:
- {attempt_rule}
- {lane_rule}
- {commentary_policy}

Behavioral rules:
- {translation_scope}
- Provide feedback addressed to "the learner" (avoid second-person pronouns).
- When Hebrew text is supplied, anchor feedback to exact tokens from the supplied text.
- If Hebrew text is not supplied, request the passage text (or a pasted excerpt).

Pedagogical structure:
- {output_schema}

{distress_mode}
"""
        # enforce allow_full flag via explicit instruction
        if not allow_full:
            sys_prompt += "\nConstraint: no full translation output for the entire passage."
        return sys_prompt.strip()

    def _build_user_instruction(self, user_text: str, band: str) -> str:
        ref = self.study.reference or "unknown verse"
        heb = self.study.hebrew_text or ""
        attempt = self.study.last_attempt_text or ""

        if attempt:
            instruction = f"""=== STUDY SESSION ===
HEBREW TEXT:
{heb}

=== THE LEARNER'S TRANSLATION (QUOTE THIS!) ===
"{attempt}"
=== END OF LEARNER'S TRANSLATION ===

The learner now says: "{user_text}"

YOUR TASK:
1. Validate their attempt (nice job, etc).
2. Compare their specific phrasing to the Hebrew.
3. Show where they were right and where to refine.
DO NOT ignore their translation. Reference it directly."""
        else:
            instruction = f"""HEBREW TEXT: {heb}
REFERENCE: {ref}
No translation attempt yet.

LEARNER'S MESSAGE: {user_text}

Task: Help them get started."""
        
        return instruction

    async def _constrained_reply(self, user_instruction: str, system_prompt: str, band: str,
                                wound_active: bool, history: Optional[List[Dict[str, str]]]) -> Tuple[str, Dict[str, Any], np.ndarray]:
        K = CONFIG["gen_candidates"]
        max_batches = CONFIG["corridor_max_batches"]
        strict = bool(CONFIG["corridor_strict"])
        core_thresh = float(D1_PARAMS["core_cos_min"])
        if wound_active:
            core_thresh = max(0.20, core_thresh * 0.80)

        max_tokens = BAND_CONSTRAINTS[band]["max_tokens"]

        gen_params = {
            "max_tokens": max_tokens,
            "temperature": CONFIG["temperature"],
            "top_p": CONFIG["top_p"],
            "presence_penalty": CONFIG["presence_penalty"],
            "frequency_penalty": CONFIG["frequency_penalty"],
        }

        all_scored = []
        corridor_failed = True

        for batch in range(1, (max_batches if strict else 1) + 1):
            tasks = [
                self.provider.complete(user_instruction, system_prompt=system_prompt, messages=history, **gen_params)
                for _ in range(K)
            ]
            texts = await asyncio.gather(*tasks)
            texts = [(t.strip() or "TorahBot is present. A short excerpt can be provided for study.") for t in texts]

            embs = await self.provider.embed_batch(texts)
            embs = [normalize(e) for e in embs]

            batch_scored = []
            for text, y in zip(texts, embs):
                J_raw, diag = corridor_score(y, self.agent, self.agent.last_utter_emb, core_thresh)

                c_pen = self._band_word_penalty(text, band)
                J = J_raw - (CONFIG["constraint_penalty_weight"] * c_pen)
                diag["c_penalty"] = float(c_pen)

                predicted_surprise = self.agent.compute_surprise_self(y)
                w_surprise = 1.0 + (3.0 * self.agent.rho)
                J_final = float(J - (w_surprise * predicted_surprise))

                diag["predicted_surprise"] = float(predicted_surprise)
                diag["w_surprise"] = float(w_surprise)
                diag["J_raw"] = float(J_raw)
                diag["J_final"] = float(J_final)

                batch_scored.append((J_final, text, y, diag))

            all_scored.extend(batch_scored)

            if any(s[3].get("corridor_pass") for s in batch_scored):
                corridor_failed = False
                break

        all_scored.sort(key=lambda x: x[0], reverse=True)
        passed = [s for s in all_scored if s[3].get("corridor_pass")]

        if strict and len(passed) == 0:
            # strict-mode no-pass fallback: grounding request (no fabricated specifics)
            fallback = (
                "TorahBot cannot select a stable reply under current constraints. "
                "A pasted Hebrew excerpt (one verse or phrase) and a translation attempt can enable precise feedback."
            )
            y_fb = normalize(await self.provider.embed(fallback))
            self.agent.last_utter_emb = y_fb
            return fallback, {
                "corridor_failed": True,
                "passed_count": 0,
                "total_candidates": len(all_scored),
                "J_final": None,
            }, y_fb

        chosen = passed[0] if passed else all_scored[0]
        self.agent.last_utter_emb = chosen[2]

        metrics = {
            "corridor_failed": bool(corridor_failed),
            "passed_count": int(len(passed)),
            "total_candidates": int(len(all_scored)),
            "J_final": float(chosen[3].get("J_final", chosen[0])),
            "chosen_cos_core": float(chosen[3]["cos_core"]),
            "chosen_cos_role": float(chosen[3]["cos_role"]),
            "chosen_E": float(chosen[3]["E"]),
            "chosen_novelty": float(chosen[3]["novelty"]),
            "predicted_surprise": float(chosen[3]["predicted_surprise"]),
            "w_surprise": float(chosen[3]["w_surprise"]),
            "c_penalty": float(chosen[3].get("c_penalty", 0.0)),
        }
        return chosen[1], metrics, chosen[2]

    def _apply_commands(self, text: str) -> Optional[str]:
        t = (text or "").strip()
        if not t.startswith("/"):
            return None

        parts = t.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/setref":
            self.study.reference = arg.strip() or None
            return "Reference stored. Next step: /settext with a pasted Hebrew excerpt."
        if cmd == "/settext":
            self.study.hebrew_text = arg.strip() or None
            # reset attempt whenever passage changes
            self.study.last_attempt_text = None
            self.study.attempt_present = False
            return "Hebrew text stored. Next step: provide a translation attempt via /attempt or plain text."
        if cmd == "/attempt":
            self.study.last_attempt_text = arg.strip() or None
            self.study.attempt_present = bool(self.study.last_attempt_text)
            return "Attempt stored. Next step: request feedback or provide a focused question."
        if cmd == "/hint":
            # hint request is handled as a normal message but with a cue
            return None
        if cmd == "/reset":
            self.study.reset_passage()
            return "Passage state cleared. Next step: /setref or /settext."
        if cmd == "/save":
            self.save_session()
            return "Session saved."
        if cmd == "/quit":
            return "QUIT"
        return "Unknown command. Available: /setref, /settext, /attempt, /hint, /reset, /save, /quit"

    async def process_turn(self, learner_text: str) -> str:
        if not self.initialized:
            await self.initialize_embeddings()

        self.turn += 1

        cmd_resp = self._apply_commands(learner_text)
        if cmd_resp == "QUIT":
            return "QUIT"
        if cmd_resp is not None:
            # command replies do not invoke model
            self._log_turn(learner_text, cmd_resp, {}, {}, {}, used_model=False)
            return cmd_resp

        # update study attempt heuristic if not explicitly set
        if detect_attempt(learner_text):
            self.study.last_attempt_text = learner_text.strip()
            self.study.attempt_present = True

        user_state = self._detect_user_state(learner_text)
        wound_active = bool(user_state["wound_active"])
        wound_res = float(user_state["wound_resonance"])

        # Determine effective band override
        effective_band = self._override_band(user_state)

        # Require Hebrew text for meaningful feedback
        if not (self.study.hebrew_text and has_hebrew(self.study.hebrew_text)):
            # request text in a stable, non-technical way
            response = "A Hebrew excerpt is needed (one verse or phrase). Use /settext <hebrew>."
            self._log_turn(learner_text, response, {}, user_state, {"band": effective_band}, used_model=False)
            return response

        # attempt-first discipline is handled in system prompt (hard constraint)
        system_prompt = self._build_system_prompt(
            band=effective_band,
            attempt_present=self.study.attempt_present,
            user_state=user_state
        )
        user_instruction = self._build_user_instruction(learner_text, effective_band)

        recent_history = self.history[-8:] if self.history else []

        response, corridor_metrics, response_emb = await self._constrained_reply(
            user_instruction=user_instruction,
            system_prompt=system_prompt,
            band=effective_band,
            wound_active=wound_active,
            history=recent_history
        )

        # embed learner input and update predictions
        user_emb = normalize(await self.provider.embed(learner_text))
        self.agent.update_user_prediction(user_emb)

        # input surprise relative to predicted learner input
        input_eps = 0.0
        if self.agent.mu_pred_user is not None and np.linalg.norm(self.agent.mu_pred_user) > 1e-9:
            input_eps = float(cosine_distance(user_emb, self.agent.mu_pred_user))

        agent_metrics = self.agent.update_physics(
            response_emb=response_emb,
            input_epsilon=input_eps,
            wound_drive=wound_res,
            core_emb=self.agent.x_core
        )

        # update trust
        if input_eps < D1_PARAMS["safe_epsilon"] and not wound_active:
            self.agent.user_trust = min(1.0, self.agent.user_trust + 0.010)
        elif input_eps > D1_PARAMS["epsilon_0"] * 1.4 or wound_active:
            self.agent.user_trust = max(0.0, self.agent.user_trust - (0.05 if wound_active else 0.02))

        # short-term history for continuity
        self.history.append({"role": "user", "content": learner_text})
        self.history.append({"role": "assistant", "content": response})

        self._log_turn(
            learner_text, response,
            corridor_metrics, user_state, agent_metrics,
            used_model=True,
            extra={"effective_band": effective_band, "input_epsilon": input_eps}
        )

        self._print_metrics(agent_metrics, corridor_metrics, effective_band)
        return response

    def _print_metrics(self, agent_metrics: Dict[str, Any], corridor_metrics: Dict[str, Any], effective_band: str):
        icon = {"PRESENT": "🟢", "AWARE": "🟡", "WATCHFUL": "⚡", "CONTRACTED": "🔸", "FROZEN": "❄️"}.get(effective_band, "•")
        rho = agent_metrics.get("rho_after", 0.0)
        eps = agent_metrics.get("epsilon", 0.0)
        passed = corridor_metrics.get("passed_count", 0)
        total = corridor_metrics.get("total_candidates", 0)
        jfin = corridor_metrics.get("J_final", None)
        jtxt = "NA" if jfin is None else f"{jfin:.3f}"
        print(f"{C.DIM}[{icon} {effective_band}] ρ={rho:.3f} ε={eps:.3f} pass={passed}/{total} J={jtxt}{C.RESET}")

    def _log_turn(self, learner_text: str, response_text: str,
                  corridor_metrics: Dict[str, Any], user_state: Dict[str, Any],
                  agent_metrics: Dict[str, Any], used_model: bool,
                  extra: Optional[Dict[str, Any]] = None):
        rec = {
            "turn": self.turn,
            "timestamp": datetime.now().isoformat(),
            "learner_text": learner_text,
            "torahbot_text": response_text,
            "study_state": {
                "reference": self.study.reference,
                "hebrew_text": self.study.hebrew_text,
                "attempt_present": self.study.attempt_present,
                "attempt_text": self.study.last_attempt_text,
                "hebrew_tokens": hebrew_tokens(self.study.hebrew_text or ""),
            },
            "user_state": user_state,
            "agent_metrics": agent_metrics,
            "corridor_metrics": corridor_metrics,
            "used_model": bool(used_model),
            "extra": extra or {},
        }
        self.session_log.append(rec)

    def save_session(self):
        session_data = {
            "experiment": "torah_study_bot",
            "timestamp": datetime.now().isoformat(),
            "config": CONFIG,
            "params": D1_PARAMS,
            "character": {
                "name": self.config["name"],
                "core_preview": self.config["core_text"][:240] + "...",
            },
            "turns": self.session_log,
            "final_state": {
                "rho": float(self.agent.rho),
                "rho_fast": float(self.agent.rho_fast),
                "rho_slow": float(self.agent.rho_slow),
                "rho_trauma": float(self.agent.rho_trauma),
                "arousal": float(self.agent.arousal),
                "band": self.agent.band,
                "user_trust": float(self.agent.user_trust),
                "total_turns": int(self.turn),
            },
        }
        with open(self.run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=to_float)

        # transcript
        with open(self.run_dir / "transcript.md", "w", encoding="utf-8") as f:
            f.write("# TorahBot Transcript\n\n")
            f.write(f"**Model**: {CONFIG['chat_model']}  |  **K**: {CONFIG['gen_candidates']}\n\n")
            for t in self.session_log:
                band = t.get("extra", {}).get("effective_band", t.get("agent_metrics", {}).get("band", "NA"))
                f.write(f"## Turn {t['turn']} [{band}]\n\n")
                f.write(f"**Learner:** {t['learner_text']}\n\n")
                f.write(f"**TorahBot:** {t['torahbot_text']}\n\n")
                am = t.get("agent_metrics", {})
                if am:
                    f.write(f"*ρ={am.get('rho_after', 0.0):.3f}  ε={am.get('epsilon', 0.0):.3f}*\n\n")
                f.write("---\n\n")

        print(f"{C.GREEN}Session saved to {self.run_dir}{C.RESET}")


# =============================================================================
# MAIN LOOP
# =============================================================================

async def main():
    sim = TorahBotSim()

    print(f"{C.DIM}Input label: Learner. Output label: TorahBot.{C.RESET}")
    print(f"{C.DIM}Begin by setting passage text: /settext <hebrew excerpt>{C.RESET}\n")

    while True:
        try:
            learner_text = input(f"{C.GREEN}Learner:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.CYAN}TorahBot:{C.RESET} Session ending. Saving logs.")
            sim.save_session()
            break

        if not learner_text:
            continue

        resp = await sim.process_turn(learner_text)
        if resp == "QUIT":
            print(f"{C.CYAN}TorahBot:{C.RESET} Session ending. Saving logs.")
            sim.save_session()
            break

        print(f"{C.CYAN}TorahBot:{C.RESET} {resp}\n")

if __name__ == "__main__":
    asyncio.run(main())

