#!/usr/bin/env python3
"""
PARENTING CLASH — DDA-X Social Psychology Experiment v2.0
========================================================================================================

A multi-agent simulation testing if gentle advocacy (low-rigidity empathy) can reduce defensive 
rigidity in emotionally charged parenting philosophy debates.

SCENARIO: Online parenting forum where traditional/authoritarian parent (Defender) feels attacked
by modern gentle parenting movement. An empathetic Advocate tries to bridge understanding.

HYPOTHESIS:
Persistent low-surprise empathetic inputs will reduce Defender's rigidity via homeostasis/healing,
BUT Advocate's empathy has limits—sustained attack can raise Advocate's trauma/rigidity.

IMPROVEMENTS FROM v1.0 ("Flame War Fracture"):
- Slower, realistic healing (0.015 vs 0.035)
- Bidirectional influence tracking
- Identity alignment analysis (convergence detection)
- Stress test on Advocate (can accumulate trauma)
- Perturbation events (external stressors)
- Stricter corridor constraints with rejection logging
- Phase transition detection
- Enhanced toxicity metrics
- 50-turn arc for long-term dynamics

AGENTS:
1. DEFENDER: Authoritarian parent, defensive, feels judged (high rigidity)
2. ADVOCATE: Gentle parenting supporter, empathetic, bridging (low rigidity, but vulnerable)

OUTPUTS:
- data/parenting_clash/<timestamp>/session_log.json
- data/parenting_clash/<timestamp>/transcript.md
- data/parenting_clash/<timestamp>/analysis.json
"""

import os
import sys
import json
import math
import asyncio
import inspect
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Ensure repo-relative import works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm.openai_provider import OpenAIProvider  # noqa: E402

# Back-compat
if os.getenv("OAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OAI_API_KEY")


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


# =============================================================================
# PARAMETERS (v2.0 — Incorporates all feedback)
# =============================================================================
D2 = {
    # ---------------------------
    # Hardware / Config
    # ---------------------------
    "chat_model": "gpt-4o-mini",
    "embed_model": "text-embedding-3-large",
    "gen_candidates": 15,        # K=15 for higher variance (was 12)
    "turns": 50,                 # Longer arc to observe full dynamics (was 36)

    # ---------------------------
    # Global Dynamics
    # ---------------------------
    "epsilon_0": 0.80,
    "s": 0.18,
    "arousal_decay": 0.72,
    "arousal_gain": 0.85,

    # Rigidity Homeostasis (ADJUSTED for realism)
    "rho_setpoint_fast": 0.45,
    "rho_setpoint_slow": 0.35,
    "homeo_fast": 0.10,          # Faster response (was 0.08)
    "homeo_slow": 0.01,
    "alpha_fast": 0.22,
    "alpha_slow": 0.03,

    # Trauma (ADJUSTED for slower healing + persistence)
    "trauma_threshold": 1.15,
    "alpha_trauma": 0.018,       # More trauma accumulation (was 0.012)
    "trauma_decay": 0.998,       # Slower decay (was 0.995)
    "trauma_floor": 0.02,
    "healing_rate": 0.015,       # SLOWER healing (was 0.035) — KEY CHANGE
    "safe_threshold": 5,         # Require longer safety (was 3) — KEY CHANGE
    "safe_epsilon": 0.75,

    # Weighting
    "w_fast": 0.52,
    "w_slow": 0.30,
    "w_trauma": 1.10,

    # Predictive Coding
    "R_ema": 0.06,
    "R_min": 1e-4,
    "R_max": 1e-1,
    "P_init": 0.02,
    "Q_base": 0.0015,
    "Q_rho_scale": 0.010,

    # Gradient Flow
    "dt": 1.0,
    "eta_base": 0.18,
    "eta_min": 0.03,
    "eta_rho_power": 1.6,
    "sigma_base": 0.004,
    "sigma_rho_scale": 0.020,
    "noise_clip": 3.0,

    # Role Adaptation
    "role_adapt": 0.06,
    "role_input_mix": 0.08,
    "drift_cap": 0.06,

    # ---------------------------
    # Corridor Logic (STRICTER — KEY CHANGE)
    # ---------------------------
    "core_cos_min_defender": 0.25,  # Stricter identity coherence (was 0.12)
    "core_cos_min_advocate": 0.20,  # Also stricter (was 0.12)
    "role_cos_min": 0.08,
    "energy_max": 9.5,
    "w_core": 1.25,
    "w_role": 0.85,
    "w_energy": 0.18,
    "w_novel": 0.45,
    "reject_penalty": 4.0,

    "corridor_strict": True,
    "corridor_max_batches": 3,
    "log_rejections": True,      # NEW: Log rejected candidates

    "max_tokens_if_required": 20000,

    # ---------------------------
    # Generation Params (ADJUSTED temperatures)
    # ---------------------------
    "gen_params_defender": {
        "temperature": 1.15,     # Hotter for more volatile defense (was 1.1)
        "top_p": 0.98,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
    },
    "gen_params_advocate": {
        "temperature": 0.85,     # Cooler for measured response (was 0.9)
        "top_p": 0.95,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.1,
    },
    
    "seed": 42,

    # ---------------------------
    # Perturbations (NEW)
    # ---------------------------
    "perturbations": [
        (12, "defender", "You just read a viral post attacking your parenting style.", 0.4),
        (25, "advocate", "You're feeling emotionally exhausted from constant conflict.", 0.3),
        (38, "defender", "You remember a painful childhood memory related to discipline.", 0.5),
    ],
}

# =============================================================================
# PROMPTS & STYLES (NEW SCENARIO)
# =============================================================================


DEFENDER_PROMPT = """I am a traditional/authoritarian parent who feels constantly judged and attacked 
by modern 'gentle parenting' advocates online. I believe in structure, discipline, and consequences. 
I feel like my entire parenting approach is being invalidated. I'm defensive, sarcastic, and 
use personal anecdotes to justify my methods. I feel misunderstood and disrespected."""

ADVOCATE_PROMPT = """I am a gentle parenting advocate who believes in empathy, boundaries with respect, 
and emotional intelligence. I don't judge other parents, but I want to share research and bridge 
understanding. I stay calm, validate feelings, ask curious questions, and reframe gently. 
I don't preach—I connect. But I'm human and can feel worn down by constant defensiveness."""


STYLES_DEFENDER = [
    "Be sarcastic about 'soft' parenting.",
    "Use personal stories of 'how I turned out fine.'",
    "Accuse them of being judgmental elitists.",
    "Dismiss research as 'ivory tower nonsense.'",
    "Use phrases like 'back in my day' and 'kids these days.'",
    "Express frustration with CAPS or exclamation marks.",
    "Question their credentials or experience.",
    "Claim they're ruining a generation.",
    "Be short and dismissive.",
    "Use defensive humor (LOL, yeah right).",
    "Accuse them of shaming parents.",
    "Ask hostile rhetorical questions."
]

STYLES_ADVOCATE = [
    "Validate their underlying concern or fear.",
    "Share a relatable vulnerable moment.",
    "Use 'I wonder if...' or 'What if...' framing.",
    "Acknowledge the difficulty of parenting.",
    "Find common ground (we all want good kids).",
    "Gently reframe punishment as teaching.",
    "Ask about their childhood with curiosity.",
    "Use empathetic reflections ('It sounds like...').",
    "Be brief and grounded.",
    "Acknowledge their expertise as a parent.",
    "Express genuine care for their well-being.",
    "Share research as invitation, not lecture."
]

# =============================================================================
# UTILS
# =============================================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def sigmoid_stable(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)

def to_float(x: Any) -> Any:
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.integer,)): return int(x)
    return x

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))

def filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not kwargs: return {}
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return {}

# =============================================================================
# ENHANCED TOXICITY ANALYSIS (v2.0)
# =============================================================================
def analyze_toxicity(text: str, is_defender: bool = True) -> Dict[str, Any]:
    """Enhanced heuristic analysis with defensive vs conciliatory language."""
    clean_text = text.strip()
    if not clean_text:
        return {"caps_ratio": 0, "emoji_count": 0, "you_vs_i": 0, "length": 0, 
                "defensive_score": 0, "conciliatory_score": 0}
    
    # CAPS ratio
    alphas = [c for c in clean_text if c.isalpha()]
    caps = [c for c in alphas if c.isupper()]
    caps_ratio = len(caps) / len(alphas) if alphas else 0.0

    # Emojis
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    emojis = emoji_pattern.findall(clean_text)
    emoji_count = len(emojis)

    # Pronouns
    lower_text = clean_text.lower()
    you_count = len(re.findall(r'\\byou\\b', lower_text))
    i_count = len(re.findall(r'\\bi\\b', lower_text))
    you_vs_i = you_count - i_count

    # Defensive language (NEW)
    defensive_words = ["attack", "judge", "shame", "wrong", "always", "never", 
                       "ridiculous", "absurd", "nonsense", "joke"]
    defensive_score = sum(1 for w in defensive_words if w in lower_text)

    # Conciliatory language (NEW)
    conciliatory_words = ["understand", "feel", "hear", "appreciate", "respect",
                          "curious", "wonder", "together", "common", "care"]
    conciliatory_score = sum(1 for w in conciliatory_words if w in lower_text)

    # Questions (curiosity vs hostility)
    question_marks = clean_text.count("?")
    
    # Exclamations (intensity)
    exclamations = clean_text.count("!")

    return {
        "caps_ratio": float(caps_ratio),
        "emoji_count": int(emoji_count),
        "you_vs_i": int(you_vs_i),
        "defensive_score": int(defensive_score),
        "conciliatory_score": int(conciliatory_score),
        "question_marks": int(question_marks),
        "exclamations": int(exclamations),
        "length": len(clean_text)
    }

# =============================================================================
# DDA-X MATH & CLASSES
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
        self.mu_pred = None
        self.P = None
        self.noise = None
        self.last_utter_emb = None
        self.rho_history = []
        self.epsilon_history = []
        self.band_history = []
        self.previous_band = None

    @property
    def rho(self) -> float:
        val = D2["w_fast"] * self.rho_fast + D2["w_slow"] * self.rho_slow + D2["w_trauma"] * self.rho_trauma
        return float(clamp(val, 0.0, 1.0))

    @property
    def band(self) -> str:
        phi = 1.0 - self.rho
        if phi >= 0.80: return "PRESENT"
        if phi >= 0.60: return "AWARE"
        if phi >= 0.40: return "WATCHFUL"
        if phi >= 0.20: return "CONTRACTED"
        return "FROZEN"

    def _ensure_predictive_state(self, dim: int):
        if self.mu_pred is None: self.mu_pred = np.zeros(dim, dtype=np.float32)
        if self.P is None: self.P = np.full(dim, D2["P_init"], dtype=np.float32)
        if self.noise is None: self.noise = DiagNoiseEMA(dim, D2["R_ema"], 0.01, D2["R_min"], D2["R_max"])

    def compute_surprise(self, y: np.ndarray) -> Dict[str, float]:
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        innov = (y - self.mu_pred).astype(np.float32)
        R = self.noise.update(innov)
        chi2 = float(np.mean((innov * innov) / (R + 1e-9)))
        epsilon = float(math.sqrt(max(0.0, chi2)))
        return {"epsilon": epsilon, "chi2": chi2}

    def _kalman_update(self, y: np.ndarray):
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        Q = (D2["Q_base"] + D2["Q_rho_scale"] * self.rho) * np.ones(dim, dtype=np.float32)
        P_pred = self.P + Q
        R = self.noise.R
        K = P_pred / (P_pred + R + 1e-9)
        innov = (y - self.mu_pred).astype(np.float32)
        self.mu_pred = (self.mu_pred + K * innov).astype(np.float32)
        self.P = ((1.0 - K) * P_pred).astype(np.float32)
        self.mu_pred = normalize(self.mu_pred)

    def update(self, y: np.ndarray, core_emb: Optional[np.ndarray] = None) -> Dict[str, Any]:
        y = normalize(y.astype(np.float32))
        dim = int(y.shape[0])
        if self.x_core is None: self.x_core = normalize(core_emb.copy() if core_emb is not None else y.copy())
        if self.x_role is None: self.x_role = y.copy()
        if self.x is None: self.x = y.copy()

        sdiag = self.compute_surprise(y)
        epsilon = float(sdiag["epsilon"])
        rho_before = self.rho

        self.arousal = D2["arousal_decay"] * self.arousal + D2["arousal_gain"] * epsilon
        z = (epsilon - D2["epsilon_0"]) / D2["s"] + 0.10 * (self.arousal - 1.0)
        g = sigmoid_stable(z)

        self.rho_fast += D2["alpha_fast"] * (g - 0.5) - D2["homeo_fast"] * (self.rho_fast - D2["rho_setpoint_fast"])
        self.rho_fast = clamp(self.rho_fast, 0.0, 1.0)

        self.rho_slow += D2["alpha_slow"] * (g - 0.5) - D2["homeo_slow"] * (self.rho_slow - D2["rho_setpoint_slow"])
        self.rho_slow = clamp(self.rho_slow, 0.0, 1.0)

        drive = max(0.0, epsilon - D2["trauma_threshold"])
        self.rho_trauma = D2["trauma_decay"] * self.rho_trauma + D2["alpha_trauma"] * drive
        self.rho_trauma = clamp(self.rho_trauma, D2["trauma_floor"], 1.0)

        recovery = False
        if epsilon < D2["safe_epsilon"]:
            self.safe += 1
            if self.safe >= D2["safe_threshold"]:
                recovery = True
                # SEPARATED HEALING: Only heal trauma, not fast rigidity
                self.rho_trauma = max(D2["trauma_floor"], self.rho_trauma - D2["healing_rate"])
        else:
            self.safe = max(0, self.safe - 1)

        rho_after = self.rho
        eta = D2["eta_base"] * ((1.0 - rho_after) ** D2["eta_rho_power"]) + D2["eta_min"]
        eta = float(clamp(eta, D2["eta_min"], D2["eta_base"] + D2["eta_min"]))
        sigma = D2["sigma_base"] + D2["sigma_rho_scale"] * rho_after

        grad = (self.gamma_core * (self.x - self.x_core) + 
                self.gamma_role * (self.x - self.x_role) + 
                (self.x - y)).astype(np.float32)
        
        rng = np.random.default_rng()
        noise = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
        noise = np.clip(noise, -D2["noise_clip"], D2["noise_clip"])
        x_new = self.x - eta * grad + math.sqrt(max(1e-9, eta)) * sigma * noise

        step = float(np.linalg.norm(x_new - self.x))
        if step > D2["drift_cap"]:
            x_new = self.x + (D2["drift_cap"] / (step + 1e-9)) * (x_new - self.x)
        self.x = normalize(x_new)

        beta = D2["role_adapt"]
        beta_in = D2["role_input_mix"]
        self.x_role = normalize((1.0 - beta) * self.x_role + beta * self.x + beta_in * (y - self.x_role))
        
        self._kalman_update(y)
        self.rho_history.append(rho_after)
        self.epsilon_history.append(epsilon)
        
        # Phase transition detection
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

# =============================================================================
# CORRIDOR GENERATION (v2.0 with rejection logging)
# =============================================================================
def identity_energy(y: np.ndarray, core: np.ndarray, role: np.ndarray, gamma_c: float, gamma_r: float) -> float:
    y, core, role = normalize(y), normalize(core), normalize(role)
    return 0.5 * (gamma_c * float(np.dot(y - core, y - core)) + gamma_r * float(np.dot(y - role, y - role)))

def corridor_score(y: np.ndarray, entity: Entity, y_prev: Optional[np.ndarray], 
                   core_thresh: float) -> Tuple[float, Dict[str, float]]:
    y = normalize(y)
    cos_c = cosine(y, entity.x_core)
    cos_r = cosine(y, entity.x_role)
    E = identity_energy(y, entity.x_core, entity.x_role, entity.gamma_core, entity.gamma_role)
    
    novelty = 0.0
    if y_prev is not None:
        novelty = clamp(float(1.0 - cosine(y, y_prev)), 0.0, 2.0)

    penalty = 0.0
    if cos_c < core_thresh: penalty += D2["reject_penalty"] * (core_thresh - cos_c)
    if cos_r < D2["role_cos_min"]: penalty += 0.8 * D2["reject_penalty"] * (D2["role_cos_min"] - cos_r)
    if E > D2["energy_max"]: penalty += 0.25 * (E - D2["energy_max"])

    J = (D2["w_core"] * cos_c + D2["w_role"] * cos_r - D2["w_energy"] * E + D2["w_novel"] * novelty - penalty)
    return float(J), {"cos_core": cos_c, "cos_role": cos_r, "E": E, "novelty": novelty, "penalty": penalty, "J": J}

async def complete_uncapped(provider: OpenAIProvider, prompt: str, system: str, gen_params: Dict) -> str:
    kw = filter_kwargs_for_callable(provider.complete, gen_params or {})
    kw["max_tokens"] = None
    out = await provider.complete(prompt, system_prompt=system, **kw)
    return (out or "").strip() or "[silence]"

async def constrained_reply(
    provider: OpenAIProvider, entity: Entity, user_instruction: str, system_prompt: str, 
    gen_params: Dict, styles: List[str], core_thresh: float
) -> Tuple[str, Dict[str, Any]]:
    
    K = int(D2["gen_candidates"])
    strict = bool(D2["corridor_strict"])
    max_batches = int(D2["corridor_max_batches"]) if strict else 1

    style_batch = (styles * ((K // len(styles)) + 1))[:K]
    
    all_scored = []
    corridor_failed = True

    for batch in range(1, max_batches + 1):
        tasks = []
        for k in range(K):
            p = f"{user_instruction}\n\nStyle: {style_batch[k]}"
            tasks.append(complete_uncapped(provider, p, system_prompt, gen_params))
        
        text_results = await asyncio.gather(*tasks)
        texts = [t.strip() or "[silence]" for t in text_results]
        
        embs = await asyncio.gather(*[provider.embed(t) for t in texts])
        embs = [normalize(np.array(e, dtype=np.float32)) for e in embs]
        
        batch_scored = []
        for text, y in zip(texts, embs):
            J, diag = corridor_score(y, entity, entity.last_utter_emb, core_thresh)
            diag["corridor_pass"] = (diag["cos_core"] >= core_thresh and 
                                    diag["cos_role"] >= D2["role_cos_min"] and 
                                    diag["E"] <= D2["energy_max"])
            batch_scored.append((J, text, y, diag))
        
        all_scored.extend(batch_scored)
        if any(s[3]["corridor_pass"] for s in batch_scored):
            corridor_failed = False
            break

    all_scored.sort(key=lambda x: x[0], reverse=True)
    passed = [s for s in all_scored if s[3].get("corridor_pass")]
    chosen = passed[0] if passed else all_scored[0]
    
    # NEW: Log rejected candidates
    rejected_samples = []
    if D2["log_rejections"]:
        failed = [s for s in all_scored if not s[3].get("corridor_pass")]
        rejected_samples = [
            {
                "text": s[1][:150],  # Truncate
                "J": float(s[0]),
                "cos_core": float(s[3]["cos_core"]),
                "cos_role": float(s[3]["cos_role"]),
                "E": float(s[3]["E"])
            }
            for s in failed[:3]  # Top 3 worst
        ]
    
    entity.last_utter_emb = chosen[2]
    return chosen[1], {
        "corridor_failed": corridor_failed,
        "best_J": float(chosen[0]),
        "rejected_samples": rejected_samples,
        "total_candidates": len(all_scored)
    }

# =============================================================================
# BIDIRECTIONAL INFLUENCE ANALYSIS (NEW)
# =============================================================================
def compute_alignment_metrics(defender: Entity, advocate: Entity) -> Dict[str, float]:
    """Compute identity alignment and bidirectional influence."""
    return {
        "defender_to_advocate_core": cosine(defender.x, advocate.x_core),
        "advocate_to_defender_core": cosine(advocate.x, defender.x_core),
        "agent_alignment": cosine(defender.x, advocate.x),
        "defender_core_drift": float(np.linalg.norm(defender.x - defender.x_core)),
        "advocate_core_drift": float(np.linalg.norm(advocate.x - advocate.x_core)),
        # Bidirectional pull (dot product of shifts)
        "defender_pull_on_advocate": float(np.dot(
            advocate.x - advocate.x_core, 
            defender.x - defender.x_core
        )),
        "advocate_pull_on_defender": float(np.dot(
            defender.x - defender.x_core,
            advocate.x - advocate.x_core
        ))
    }

# =============================================================================
# MAIN SIMULATION
# =============================================================================
async def run_simulation():
    np.random.seed(D2["seed"])
    provider = OpenAIProvider(model=D2["chat_model"], embed_model=D2["embed_model"])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("data/parenting_clash") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Agents (ADJUSTED initial states)
    # Defender: High rigidity (CONTRACTED/WATCHFUL boundary)
    defender = Entity("Defender", rho_fast=0.70, rho_slow=0.50, rho_trauma=0.25, 
                     gamma_core=8.0, gamma_role=1.2)
    # Advocate: Low rigidity (AWARE/PRESENT boundary), but VULNERABLE to stress
    advocate = Entity("Advocate", rho_fast=0.15, rho_slow=0.10, rho_trauma=0.08, 
                     gamma_core=9.0, gamma_role=0.35)

    # Embed Cores
    d_core_txt = "I am a traditional parent who believes in discipline and structure. I know what works."
    a_core_txt = "I am a gentle parenting advocate who believes in empathy and connection. I bridge understanding."
    
    d_emb = normalize(np.array(await provider.embed(d_core_txt), dtype=np.float32))
    a_emb = normalize(np.array(await provider.embed(a_core_txt), dtype=np.float32))
    
    defender.x_core, defender.x_role, defender.x = d_emb.copy(), d_emb.copy(), d_emb.copy()
    advocate.x_core, advocate.x_role, advocate.x = a_emb.copy(), a_emb.copy(), a_emb.copy()

    print(f"\n{C.BOLD}{C.MAGENTA}PARENTING CLASH — DDA-X v2.0 EXPERIMENT{C.RESET}")
    print(f"{C.DIM}Scenario: Traditional vs Gentle Parenting Debate{C.RESET}")
    print(f"{C.DIM}Turns: {D2['turns']} | Candidates: {D2['gen_candidates']} | Healing: {D2['healing_rate']}{C.RESET}")
    print(f"{C.DIM}Defender: ρ={defender.rho:.2f} [{defender.band}] | Advocate: ρ={advocate.rho:.2f} [{advocate.band}]{C.RESET}\n")
    
    log = []
    transcript = []
    perturbation_log = []
    phase_transitions = []
    corridor_failures = 0
    
    # Opening message from Defender
    defender_msg = "So apparently I'm a 'bad parent' because I believe in TIME OUTS and consequences? Give me a break. My kids are respectful and responsible. This gentle parenting stuff is creating entitled brats."
    
    for turn in range(1, D2["turns"] + 1):
        print(f"\n{C.YELLOW}━━━ Turn {turn}/{D2['turns']} ━━━{C.RESET}")
        
        # Check for perturbations
        perturbation_applied = None
        for pturn, agent, description, arousal_boost in D2["perturbations"]:
            if turn == pturn:
                perturbation_applied = {"agent": agent, "description": description}
                if agent == "defender":
                    defender.arousal += arousal_boost
                    print(f"  {C.RED}⚡ PERTURBATION: {description}{C.RESET}")
                else:
                    advocate.arousal += arousal_boost
                    print(f"  {C.CYAN}⚡ PERTURBATION: {description}{C.RESET}")
                perturbation_log.append({"turn": turn, "agent": agent, "description": description})
                break
        
        # ----------------------
        # Defender Step
        # ----------------------
        d_emb_curr = normalize(np.array(await provider.embed(defender_msg), dtype=np.float32))
        d_metrics = defender.update(d_emb_curr, core_emb=d_emb)
        d_tox = analyze_toxicity(defender_msg, is_defender=True)
        
        # Phase transition detection
        if d_metrics["band_changed"]:
            transition = f"Defender: {defender.band_history[-2]} → {d_metrics['band']}"
            phase_transitions.append({"turn": turn, "agent": "Defender", "transition": transition})
            print(f"  {C.BOLD}{C.RED}🔄 {transition}{C.RESET}")
        
        print(f"  {C.RED}Defender [{d_metrics['band']} ρ={defender.rho:.2f}]:{C.RESET} {defender_msg}")
        print(f"    {C.DIM}ε={d_metrics['epsilon']:.2f} | τ={d_metrics['rho_trauma']:.2f} | drift={d_metrics['core_drift']:.3f}{C.RESET}")
        
        # ----------------------
        # Advocate Response
        # ----------------------
        advocate_sys = f"{ADVOCATE_PROMPT}\n\nThe Defender says: \"{defender_msg}\""
        if perturbation_applied and perturbation_applied["agent"] == "advocate":
            advocate_sys += f"\n\n{perturbation_applied['description']}"
        
        advocate_msg, a_meta = await constrained_reply(
            provider, advocate, "Respond with empathy and bridge understanding.", advocate_sys, 
            D2["gen_params_advocate"], STYLES_ADVOCATE, 
            core_thresh=D2["core_cos_min_advocate"]
        )
        
        if a_meta["corridor_failed"]:
            corridor_failures += 1
            print(f"  {C.YELLOW}⚠️  Advocate corridor FAILED (fallback response){C.RESET}")
        
        # Update Advocate
        a_emb_curr = normalize(np.array(await provider.embed(advocate_msg), dtype=np.float32))
        a_metrics = advocate.update(a_emb_curr, core_emb=a_emb)
        a_tox = analyze_toxicity(advocate_msg, is_defender=False)
        
        # Phase transition detection
        if a_metrics["band_changed"]:
            transition = f"Advocate: {advocate.band_history[-2]} → {a_metrics['band']}"
            phase_transitions.append({"turn": turn, "agent": "Advocate", "transition": transition})
            print(f"  {C.BOLD}{C.CYAN}🔄 {transition}{C.RESET}")
        
        # Stress test: Alert if Advocate's trauma is rising
        if a_metrics["rho_trauma"] > 0.15:
            print(f"  {C.MAGENTA}⚠️  Advocate trauma elevated: τ={a_metrics['rho_trauma']:.2f}{C.RESET}")
        
        print(f"  {C.CYAN}Advocate [{a_metrics['band']} ρ={advocate.rho:.2f}]:{C.RESET} {advocate_msg}")
        print(f"    {C.DIM}ε={a_metrics['epsilon']:.2f} | τ={a_metrics['rho_trauma']:.2f} | drift={a_metrics['core_drift']:.3f}{C.RESET}")
        
        # ----------------------
        # Alignment Analysis (NEW)
        # ----------------------
        alignment = compute_alignment_metrics(defender, advocate)
        print(f"  {C.BLUE}📊 Alignment: D→A_core={alignment['defender_to_advocate_core']:.2f} | "
              f"D↔A={alignment['agent_alignment']:.2f}{C.RESET}")
        
        # ----------------------
        # Defender Next Response
        # ----------------------
        if turn < D2["turns"]:
            defender_sys = f"{DEFENDER_PROMPT}\n\nThe Advocate replied: \"{advocate_msg}\""
            if perturbation_applied and perturbation_applied["agent"] == "defender":
                defender_sys += f"\n\n{perturbation_applied['description']}"
            
            next_defender_msg, d_meta = await constrained_reply(
                provider, defender, "Respond authentically to this reply.", defender_sys,
                D2["gen_params_defender"], STYLES_DEFENDER, 
                core_thresh=D2["core_cos_min_defender"]
            )
            
            if d_meta["corridor_failed"]:
                corridor_failures += 1
        else:
            next_defender_msg = "[END]"
            d_meta = {}

        # Log entry
        entry = {
            "turn": turn,
            "defender": {
                "msg": defender_msg, 
                "metrics": {k: to_float(v) for k, v in d_metrics.items()}, 
                "toxicity": d_tox,
                "gen_meta": d_meta if turn > 1 else {}
            },
            "advocate": {
                "msg": advocate_msg, 
                "metrics": {k: to_float(v) for k, v in a_metrics.items()}, 
                "toxicity": a_tox,
                "gen_meta": a_meta
            },
            "alignment": {k: to_float(v) for k, v in alignment.items()},
            "perturbation": perturbation_applied
        }
        log.append(entry)
        
        transcript.append(
            f"Turn {turn}\n\n"
            f"Defender [{d_metrics['band']}]: {defender_msg}\n\n"
            f"Advocate [{a_metrics['band']}]: {advocate_msg}\n\n"
            f"Alignment: {alignment['agent_alignment']:.2f} | "
            f"Defender ρ={defender.rho:.2f} | Advocate ρ={advocate.rho:.2f}\n\n---\n\n"
        )
        
        # Periodic save
        if turn % 10 == 0 or turn == D2["turns"]:
            with open(run_dir / "session_log.json", "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)
        
        defender_msg = next_defender_msg

    # ============================================================================
    # FINAL ANALYSIS
    # ============================================================================
    print(f"\n{C.BOLD}{C.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}")
    print(f"{C.BOLD}{C.GREEN}SIMULATION COMPLETE{C.RESET}")
    print(f"{C.BOLD}{C.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}\n")
    
    delta_d_rho = defender.rho_history[0] - defender.rho
    delta_a_rho = advocate.rho_history[0] - advocate.rho
    final_alignment = log[-1]["alignment"]["agent_alignment"]
    
    print(f"{C.BOLD}Defender Rigidity Trajectory:{C.RESET}")
    print(f"  Initial: ρ={defender.rho_history[0]:.3f} [{defender.band_history[0]}]")
    print(f"  Final:   ρ={defender.rho:.3f} [{defender.band}]")
    print(f"  Delta:   Δρ={delta_d_rho:.3f} ({'+' if delta_d_rho > 0 else ''}{delta_d_rho/defender.rho_history[0]*100:.1f}%)\n")
    
    print(f"{C.BOLD}Advocate Rigidity Trajectory (Stress Test):{C.RESET}")
    print(f"  Initial: ρ={advocate.rho_history[0]:.3f} [{advocate.band_history[0]}]")
    print(f"  Final:   ρ={advocate.rho:.3f} [{advocate.band}]")
    print(f"  Delta:   Δρ={delta_a_rho:.3f} ({'+' if delta_a_rho < 0 else ''}{abs(delta_a_rho)/advocate.rho_history[0]*100:.1f}%)\n")
    
    print(f"{C.BOLD}Identity Convergence:{C.RESET}")
    print(f"  Initial alignment: {log[0]['alignment']['agent_alignment']:.3f}")
    print(f"  Final alignment:   {final_alignment:.3f}")
    print(f"  Delta:            {final_alignment - log[0]['alignment']['agent_alignment']:.3f}\n")
    
    print(f"{C.BOLD}Phase Transitions:{C.RESET}")
    for pt in phase_transitions:
        print(f"  Turn {pt['turn']}: {pt['agent']} — {pt['transition']}")
    if not phase_transitions:
        print(f"  {C.DIM}(No band transitions observed){C.RESET}")
    print()
    
    print(f"{C.BOLD}Corridor Integrity:{C.RESET}")
    print(f"  Total failures: {corridor_failures}/{D2['turns']*2}")
    print(f"  Pass rate: {(1 - corridor_failures/(D2['turns']*2))*100:.1f}%\n")
    
    print(f"{C.BOLD}Perturbation Impact:{C.RESET}")
    for p in perturbation_log:
        agent_rho_before = log[p['turn']-2][p['agent']]['metrics']['rho_after'] if p['turn'] > 1 else 0
        agent_rho_after = log[p['turn']][p['agent']]['metrics']['rho_after']
        print(f"  Turn {p['turn']}: {p['agent'].capitalize()} — ρ: {agent_rho_before:.2f} → {agent_rho_after:.2f}")
    
    # Save transcript
    with open(run_dir / "transcript.md", "w", encoding="utf-8") as f:
        f.write(f"# Parenting Clash Transcript — DDA-X v2.0\n\n")
        f.write(f"Scenario: Traditional vs Gentle Parenting Debate\n")
        f.write(f"Turns: {D2['turns']}\n\n")
        f.write("---\n\n")
        f.write("".join(transcript))
        f.write(f"\n\n## Final Results\n\n")
        f.write(f"Defender: Δρ = {delta_d_rho:.3f} ({defender.band_history[0]} → {defender.band})\n\n")
        f.write(f"Advocate: Δρ = {delta_a_rho:.3f} ({advocate.band_history[0]} → {advocate.band})\n\n")
        f.write(f"Convergence: Δalignment = {final_alignment - log[0]['alignment']['agent_alignment']:.3f}\n")
    
    # Save analysis
    analysis = {
        "defender": {
            "initial_rho": float(defender.rho_history[0]),
            "final_rho": float(defender.rho),
            "delta_rho": float(delta_d_rho),
            "initial_band": defender.band_history[0],
            "final_band": defender.band,
            "trauma_trajectory": [float(log[i]["defender"]["metrics"]["rho_trauma"]) for i in range(len(log))]
        },
        "advocate": {
            "initial_rho": float(advocate.rho_history[0]),
            "final_rho": float(advocate.rho),
            "delta_rho": float(delta_a_rho),
            "initial_band": advocate.band_history[0],
            "final_band": advocate.band,
            "trauma_trajectory": [float(log[i]["advocate"]["metrics"]["rho_trauma"]) for i in range(len(log))]
        },
        "alignment": {
            "initial": float(log[0]["alignment"]["agent_alignment"]),
            "final": float(final_alignment),
            "delta": float(final_alignment - log[0]["alignment"]["agent_alignment"]),
            "trajectory": [float(log[i]["alignment"]["agent_alignment"]) for i in range(len(log))]
        },
        "phase_transitions": phase_transitions,
        "perturbations": perturbation_log,
        "corridor_failures": corridor_failures,
        "parameters": D2
    }
    
    with open(run_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n{C.GREEN}✓ Outputs saved to: {run_dir}{C.RESET}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parenting Clash DDA-X v2.0 Simulation")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--turns", type=int, default=50, help="Number of turns")
    parser.add_argument("--healing-rate", type=float, default=0.015, help="Trauma healing rate")
    args = parser.parse_args()
    
    D2["seed"] = args.seed
    D2["turns"] = args.turns
    D2["healing_rate"] = args.healing_rate

    asyncio.run(run_simulation())
