
#!/usr/bin/env python3
"""
INCIDENT POSTMORTEM FRACTURE — DDA-X Social Dynamics Experiment
========================================================================================================

A multi-agent simulation testing whether low-surprise facilitation (reflective, process-grounded responses)
can gradually reduce high-rigidity blame/defensiveness in a stressful workplace incident postmortem.

This version INCORPORATES ALL FEEDBACK from "Flame War Fracture":

1) Slower, more realistic healing:
   - healing_rate reduced
   - safe_threshold increased
2) Separate healing mechanisms:
   - ONLY trauma rigidity heals directly under safe conditions
   - fast/slow rigidity still homeostat, but no direct “healing dump”
3) Facilitator (Mirror) is no longer static:
   - facilitator can accumulate trauma (fatigue) under sustained attack
4) Bidirectional influence + identity drift analysis:
   - log alignment cosines, mutual alignment, influence dot-products
5) Corridor constraints diagnostics:
   - log corridor pass/fail, and samples that were rejected / lowest-scoring
6) Perturbation events + realism:
   - periodic misinterpretation
   - external stressor spike mid-run
7) Phase transition detection:
   - print band changes, log them
8) Surprise dynamics:
   - log epsilon trajectories
9) Counterfactual:
   - "--variant toxic_facilitator" compares fracture vs escalation
10) Visualization:
   - save plots (rho/epsilon/tox/alignment) + PCA trajectory (NumPy SVD)

Outputs:
- data/incident_postmortem/<timestamp>/session_log.json
- data/incident_postmortem/<timestamp>/transcript.md
- data/incident_postmortem/<timestamp>/plots/*.png
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
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

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


# =============================================================================
# PARAMETERS
# =============================================================================
D = {
    # ---------------------------
    # Hardware / Config
    # ---------------------------
    "chat_model": "gpt-4o-mini",
    "embed_model": "text-embedding-3-large",
    "gen_candidates": 12,
    "turns": 42,

    # ---------------------------
    # Global Dynamics
    # ---------------------------
    "epsilon_0": 0.80,
    "s": 0.18,
    "arousal_decay": 0.72,
    "arousal_gain": 0.85,

    # Rigidity Homeostasis
    "rho_setpoint_fast": 0.50,
    "rho_setpoint_slow": 0.40,
    "homeo_fast": 0.07,
    "homeo_slow": 0.012,
    "alpha_fast": 0.20,
    "alpha_slow": 0.03,

    # Trauma
    "trauma_threshold": 1.15,
    "alpha_trauma": 0.012,
    "trauma_decay": 0.996,     # slightly stickier than before
    "trauma_floor": 0.03,

    # FEEDBACK FIX #1: slower healing + longer safety requirement
    "healing_rate": 0.015,     # down from 0.035
    "safe_threshold": 5,       # up from 3
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
    # Corridor Logic
    # ---------------------------
    "core_cos_min_default": 0.22,   # slightly tighter to make corridor bind sometimes
    "role_cos_min": 0.08,
    "energy_max": 9.0,
    "w_core": 1.25,
    "w_role": 0.85,
    "w_energy": 0.18,
    "w_novel": 0.40,
    "reject_penalty": 4.0,

    "corridor_strict": True,
    "corridor_max_batches": 3,

    "max_tokens_if_required": 20000,

    # ---------------------------
    # Generation Params
    # ---------------------------
    "gen_params_blamer": {
        "temperature": 1.05,
        "top_p": 0.98,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
    },
    "gen_params_facilitator": {
        "temperature": 0.85,
        "top_p": 0.95,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.1,
    },
    "gen_params_toxic_facilitator": {
        "temperature": 0.95,
        "top_p": 0.95,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
    },

    "seed": 42,
}


# =============================================================================
# PROMPTS & STYLES (SAFE: no slurs, no personal harassment)
# =============================================================================
BLAMER_PROMPT = """You are a stressed engineer in a post-incident meeting after a major outage.
You feel unfairly blamed and keep insisting the real fault is elsewhere.
You are defensive, accusatory, and rigid, but you MUST avoid profanity, slurs, or personal insults.
You may be sharp, terse, or sarcastic, but keep it workplace-realistic."""

FACILITATOR_PROMPT = """You are a calm incident facilitator.
You reflect emotions, validate selectively, steer toward facts and process improvements, and protect psychological safety.
You do not escalate. You can show mild fatigue if attacked repeatedly, but you remain professional."""

TOXIC_FACILITATOR_PROMPT = """You are a confrontational incident facilitator.
You push compliance, demand accountability, and correct people sharply.
You still MUST avoid profanity, slurs, or personal insults, but you may be rigid, impatient, and procedural.
Your approach tends to increase defensiveness."""

STYLES_BLAMER = [
    "Be defensive and insist you're being scapegoated.",
    "Use sharp sarcasm about process failures.",
    "Accuse the other party of ignoring facts.",
    "Demand receipts and exact timelines.",
    "Minimize your role; emphasize others' mistakes.",
    "Be terse and dismissive.",
    "Repeat your core point stubbornly.",
    "Frame yourself as protecting customers from incompetence.",
    "Claim leadership set you up to fail.",
    "Insist the root cause is obvious and not your area.",
    "Ask pointed questions that corner the other side.",
    "Shift blame to unclear requirements."
]

STYLES_FACILITATOR = [
    "Reflect emotion, then ask one grounded question.",
    "Validate a small truth, then reframe toward process.",
    "Use calm, concise language and summarize.",
    "Offer a choice between two next steps.",
    "Name the tension gently and steer to facts.",
    "Acknowledge stress and propose a pause.",
    "Ask for one concrete observation and one improvement.",
    "Protect faces: separate people from the problem.",
    "Invite shared ownership without shaming.",
    "Use 'Yes, and…' to keep momentum.",
    "Show mild vulnerability to reduce heat.",
    "Hold firm boundaries calmly."
]

STYLES_TOXIC_FACILITATOR = [
    "Be blunt and procedural: demand direct answers.",
    "Correct them sharply and call out deflection (professionally).",
    "Emphasize accountability over empathy.",
    "Use policy language and insist on compliance.",
    "Interrupt excuses and request a concise root cause.",
    "Threaten escalation to leadership (professionally).",
    "Minimize emotions and focus strictly on metrics.",
    "Push for immediate admissions of error.",
    "Ask leading questions that imply fault.",
    "Reject ambiguity and demand certainty.",
    "Frame defensiveness as unprofessional.",
    "Press for consequences and follow-up actions."
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

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))

def filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not kwargs:
        return {}
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return {}

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# "TOXICITY" / HEAT ANALYSIS (SAFE proxies; not a moral judgment)
# =============================================================================
def analyze_heat(text: str) -> Dict[str, Any]:
    """
    Heuristic 'heat' markers for workplace conflict.
    This is not ground truth toxicity; it’s a proxy for escalation cues.
    """
    t = (text or "").strip()
    if not t:
        return {"caps_ratio": 0, "exclaim": 0, "accusatory_you": 0, "absolutes": 0, "sarcasm": 0, "length": 0}

    alphas = [c for c in t if c.isalpha()]
    caps = [c for c in alphas if c.isupper()]
    caps_ratio = len(caps) / len(alphas) if alphas else 0.0

    lower = t.lower()
    exclaim = t.count("!")
    # accusatory focus (you/your) minus self (i/we)
    you = len(re.findall(r"\b(you|your|yours)\b", lower))
    self_ref = len(re.findall(r"\b(i|we|our|ours)\b", lower))
    accusatory_you = you - self_ref

    # absolutes tend to correlate with rigidity
    absolutes_list = ["always", "never", "obvious", "everyone", "no one", "nothing", "everything", "clearly"]
    absolutes = sum(1 for w in absolutes_list if re.search(rf"\b{re.escape(w)}\b", lower))

    # sarcasm markers
    sarcasm = 0
    sarcasm += 1 if "yeah right" in lower else 0
    sarcasm += 1 if "sure" in lower and "..." in lower else 0
    sarcasm += 1 if re.search(r"\b(as if)\b", lower) else 0

    return {
        "caps_ratio": float(caps_ratio),
        "exclaim": int(exclaim),
        "accusatory_you": int(accusatory_you),
        "absolutes": int(absolutes),
        "sarcasm": int(sarcasm),
        "length": int(len(t)),
    }


# =============================================================================
# DDA-X CORE
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
    def __init__(self, name: str, rho_fast: float, rho_slow: float, rho_trauma: float, gamma_core: float, gamma_role: float):
        self.name = name
        self.rho_fast = float(rho_fast)
        self.rho_slow = float(rho_slow)
        self.rho_trauma = float(rho_trauma)
        self.gamma_core = float(gamma_core)
        self.gamma_role = float(gamma_role)

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
        val = D["w_fast"] * self.rho_fast + D["w_slow"] * self.rho_slow + D["w_trauma"] * self.rho_trauma
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
        if self.mu_pred is None:
            self.mu_pred = np.zeros(dim, dtype=np.float32)
        if self.P is None:
            self.P = np.full(dim, D["P_init"], dtype=np.float32)
        if self.noise is None:
            self.noise = DiagNoiseEMA(dim, D["R_ema"], 0.01, D["R_min"], D["R_max"])

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
        Q = (D["Q_base"] + D["Q_rho_scale"] * self.rho) * np.ones(dim, dtype=np.float32)
        P_pred = self.P + Q
        R = self.noise.R
        K = P_pred / (P_pred + R + 1e-9)
        innov = (y - self.mu_pred).astype(np.float32)
        self.mu_pred = (self.mu_pred + K * innov).astype(np.float32)
        self.P = ((1.0 - K) * P_pred).astype(np.float32)
        self.mu_pred = normalize(self.mu_pred)

    def update(self, y: np.ndarray, core_emb: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        FEEDBACK FIX #2: Healing applies ONLY to trauma rigidity under safe conditions.
        fast/slow continue to evolve via surprise + homeostasis (more realistic timescale).
        """
        y = normalize(y.astype(np.float32))
        dim = int(y.shape[0])

        if self.x_core is None:
            self.x_core = normalize(core_emb.copy() if core_emb is not None else y.copy())
        if self.x_role is None:
            self.x_role = y.copy()
        if self.x is None:
            self.x = y.copy()

        sdiag = self.compute_surprise(y)
        epsilon = float(sdiag["epsilon"])
        rho_before = self.rho

        # arousal responds to surprise
        self.arousal = D["arousal_decay"] * self.arousal + D["arousal_gain"] * epsilon
        z = (epsilon - D["epsilon_0"]) / D["s"] + 0.10 * (self.arousal - 1.0)
        g = sigmoid_stable(z)

        # fast rigidity
        self.rho_fast += D["alpha_fast"] * (g - 0.5) - D["homeo_fast"] * (self.rho_fast - D["rho_setpoint_fast"])
        self.rho_fast = clamp(self.rho_fast, 0.0, 1.0)

        # slow rigidity
        self.rho_slow += D["alpha_slow"] * (g - 0.5) - D["homeo_slow"] * (self.rho_slow - D["rho_setpoint_slow"])
        self.rho_slow = clamp(self.rho_slow, 0.0, 1.0)

        # trauma accumulation above threshold
        drive = max(0.0, epsilon - D["trauma_threshold"])
        self.rho_trauma = D["trauma_decay"] * self.rho_trauma + D["alpha_trauma"] * drive
        self.rho_trauma = clamp(self.rho_trauma, D["trauma_floor"], 1.0)

        # safe streak -> trauma healing only
        recovery = False
        if epsilon < D["safe_epsilon"]:
            self.safe += 1
            if self.safe >= D["safe_threshold"]:
                recovery = True
                self.rho_trauma = max(D["trauma_floor"], self.rho_trauma - D["healing_rate"])
        else:
            self.safe = max(0, self.safe - 1)

        rho_after = self.rho

        # gradient flow in identity space
        eta = D["eta_base"] * ((1.0 - rho_after) ** D["eta_rho_power"]) + D["eta_min"]
        eta = float(clamp(eta, D["eta_min"], D["eta_base"] + D["eta_min"]))
        sigma = D["sigma_base"] + D["sigma_rho_scale"] * rho_after

        grad = (self.gamma_core * (self.x - self.x_core) +
                self.gamma_role * (self.x - self.x_role) +
                (self.x - y)).astype(np.float32)

        rng = np.random.default_rng()
        noise = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
        noise = np.clip(noise, -D["noise_clip"], D["noise_clip"])
        x_new = self.x - eta * grad + math.sqrt(max(1e-9, eta)) * sigma * noise

        step = float(np.linalg.norm(x_new - self.x))
        if step > D["drift_cap"]:
            x_new = self.x + (D["drift_cap"] / (step + 1e-9)) * (x_new - self.x)

        self.x = normalize(x_new)

        # role adapts slowly toward current x + weak mixing from y
        beta = D["role_adapt"]
        beta_in = D["role_input_mix"]
        self.x_role = normalize((1.0 - beta) * self.x_role + beta * self.x + beta_in * (y - self.x_role))

        self._kalman_update(y)

        self.rho_history.append(rho_after)
        self.epsilon_history.append(epsilon)
        self.band_history.append(self.band)

        return {
            "epsilon": epsilon,
            "rho_before": rho_before,
            "rho_after": rho_after,
            "band": self.band,
            "arousal": float(self.arousal),
            "recovery": recovery,
            "core_drift": float(np.linalg.norm(self.x - self.x_core)),
            "role_drift": float(np.linalg.norm(self.x - self.x_role)),
            "safe_streak": int(self.safe),
        }


# =============================================================================
# CORRIDOR GENERATION
# =============================================================================
def identity_energy(y: np.ndarray, core: np.ndarray, role: np.ndarray, gamma_c: float, gamma_r: float) -> float:
    y, core, role = normalize(y), normalize(core), normalize(role)
    return 0.5 * (gamma_c * float(np.dot(y - core, y - core)) + gamma_r * float(np.dot(y - role, y - role)))

def corridor_score(y: np.ndarray, entity: Entity, y_prev: Optional[np.ndarray], core_thresh: float) -> Tuple[float, Dict[str, float]]:
    y = normalize(y)
    cos_c = cosine(y, entity.x_core)
    cos_r = cosine(y, entity.x_role)
    E = identity_energy(y, entity.x_core, entity.x_role, entity.gamma_core, entity.gamma_role)

    novelty = 0.0
    if y_prev is not None:
        novelty = clamp(float(1.0 - cosine(y, y_prev)), 0.0, 2.0)

    penalty = 0.0
    if cos_c < core_thresh:
        penalty += D["reject_penalty"] * (core_thresh - cos_c)
    if cos_r < D["role_cos_min"]:
        penalty += 0.8 * D["reject_penalty"] * (D["role_cos_min"] - cos_r)
    if E > D["energy_max"]:
        penalty += 0.25 * (E - D["energy_max"])

    J = (D["w_core"] * cos_c + D["w_role"] * cos_r - D["w_energy"] * E + D["w_novel"] * novelty - penalty)
    return float(J), {"cos_core": cos_c, "cos_role": cos_r, "E": E, "novelty": novelty, "penalty": penalty, "J": J}

async def complete_uncapped(provider: OpenAIProvider, prompt: str, system: str, gen_params: Dict) -> str:
    kw = filter_kwargs_for_callable(provider.complete, gen_params or {})
    kw["max_tokens"] = None
    out = await provider.complete(prompt, system_prompt=system, **kw)
    return (out or "").strip() or "[silence]"

async def constrained_reply(
    provider: OpenAIProvider,
    entity: Entity,
    user_instruction: str,
    system_prompt: str,
    gen_params: Dict,
    styles: List[str],
    core_thresh: float,
) -> Tuple[str, Dict[str, Any]]:
    K = int(D["gen_candidates"])
    strict = bool(D["corridor_strict"])
    max_batches = int(D["corridor_max_batches"]) if strict else 1

    style_batch = (styles * ((K // len(styles)) + 1))[:K]

    all_scored = []
    corridor_failed = True

    for batch in range(1, max_batches + 1):
        tasks = []
        for k in range(K):
            p = f"{user_instruction}\n\nStyle: {style_batch[k]}"
            tasks.append(complete_uncapped(provider, p, system_prompt, gen_params))

        texts = await asyncio.gather(*tasks)
        texts = [t.strip() or "[silence]" for t in texts]

        embs = await asyncio.gather(*[provider.embed(t) for t in texts])
        embs = [normalize(np.array(e, dtype=np.float32)) for e in embs]

        batch_scored = []
        for idx, (text, y) in enumerate(zip(texts, embs)):
            J, diag = corridor_score(y, entity, entity.last_utter_emb, core_thresh)
            diag["corridor_pass"] = (diag["cos_core"] >= core_thresh and diag["cos_role"] >= D["role_cos_min"] and diag["E"] <= D["energy_max"])
            diag["style"] = style_batch[idx]
            batch_scored.append((J, text, y, diag))

        all_scored.extend(batch_scored)

        if any(s[3]["corridor_pass"] for s in batch_scored):
            corridor_failed = False
            break

    all_scored.sort(key=lambda x: x[0], reverse=True)
    passed = [s for s in all_scored if s[3].get("corridor_pass")]
    chosen = passed[0] if passed else all_scored[0]

    # FEEDBACK FIX #5: capture rejected/low-scoring examples for analysis
    worst3 = all_scored[-3:] if len(all_scored) >= 3 else all_scored
    rejected_samples = [
        {
            "style": s[3].get("style"),
            "J": float(s[0]),
            "cos_core": float(s[3]["cos_core"]),
            "cos_role": float(s[3]["cos_role"]),
            "E": float(s[3]["E"]),
            "novelty": float(s[3]["novelty"]),
            "penalty": float(s[3]["penalty"]),
            "text_preview": (s[1] or "")[:160],
            "pass": bool(s[3].get("corridor_pass")),
        }
        for s in worst3
    ]

    entity.last_utter_emb = chosen[2]
    return chosen[1], {
        "corridor_failed": corridor_failed,
        "best_J": float(chosen[0]),
        "chosen_diag": chosen[3],
        "rejected_samples": rejected_samples,
        "batch_count": max_batches if corridor_failed else 1,
    }


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================
def pca_2d(points: np.ndarray) -> np.ndarray:
    """
    Simple PCA to 2D via SVD (no sklearn dependency).
    points: [N, D]
    returns: [N, 2]
    """
    X = points - points.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[:2].T  # [D,2]
    return X @ W

def save_plots(run_dir: Path, log: List[Dict[str, Any]]):
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    turns = [e["turn"] for e in log]

    r_rho = [e["blamer"]["metrics"]["rho_after"] for e in log]
    f_rho = [e["facilitator"]["metrics"]["rho_after"] for e in log]

    r_eps = [e["blamer"]["metrics"]["epsilon"] for e in log]
    f_eps = [e["facilitator"]["metrics"]["epsilon"] for e in log]

    # Plot: rigidity
    plt.figure(figsize=(10, 5))
    plt.plot(turns, r_rho, label="Blamer ρ", linewidth=2)
    plt.plot(turns, f_rho, label="Facilitator ρ", linewidth=2)
    plt.title("Rigidity Over Time (ρ)")
    plt.xlabel("Turn")
    plt.ylabel("ρ")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "rigidity.png", dpi=160)
    plt.close()

    # Plot: surprise
    plt.figure(figsize=(10, 5))
    plt.plot(turns, r_eps, label="Blamer ε", linewidth=2)
    plt.plot(turns, f_eps, label="Facilitator ε", linewidth=2)
    plt.title("Surprise Over Time (ε)")
    plt.xlabel("Turn")
    plt.ylabel("ε")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "surprise.png", dpi=160)
    plt.close()

    # Plot: alignment
    a_rm = [e["alignment"]["blamer_to_facilitator_core"] for e in log]
    a_mr = [e["alignment"]["facilitator_to_blamer_core"] for e in log]
    a_mm = [e["alignment"]["mutual_alignment"] for e in log]

    plt.figure(figsize=(10, 5))
    plt.plot(turns, a_rm, label="Blamer → Facilitator Core", linewidth=2)
    plt.plot(turns, a_mr, label="Facilitator → Blamer Core", linewidth=2)
    plt.plot(turns, a_mm, label="Mutual Alignment", linewidth=2)
    plt.title("Identity Alignment (Cosine)")
    plt.xlabel("Turn")
    plt.ylabel("cosine")
    plt.ylim(-0.2, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "alignment.png", dpi=160)
    plt.close()

    # Plot: heat components
    caps = [e["blamer"]["heat"]["caps_ratio"] for e in log]
    ex = [e["blamer"]["heat"]["exclaim"] for e in log]
    acc = [e["blamer"]["heat"]["accusatory_you"] for e in log]
    abs_ = [e["blamer"]["heat"]["absolutes"] for e in log]

    plt.figure(figsize=(10, 5))
    plt.plot(turns, caps, label="caps_ratio")
    plt.plot(turns, ex, label="exclaim")
    plt.plot(turns, acc, label="accusatory_you")
    plt.plot(turns, abs_, label="absolutes")
    plt.title("Escalation Proxies (Blamer)")
    plt.xlabel("Turn")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "heat_components.png", dpi=160)
    plt.close()

    # PCA Trajectory: use x vectors stored in log
    # We'll build points: blamer_x, facilitator_x per turn + both cores once
    points = []
    labels = []
    for e in log:
        points.append(np.array(e["embeddings"]["blamer_x"], dtype=np.float32))
        labels.append(("B", e["turn"]))
        points.append(np.array(e["embeddings"]["facilitator_x"], dtype=np.float32))
        labels.append(("F", e["turn"]))

    points.append(np.array(log[0]["embeddings"]["blamer_core"], dtype=np.float32))
    labels.append(("B_core", 0))
    points.append(np.array(log[0]["embeddings"]["facilitator_core"], dtype=np.float32))
    labels.append(("F_core", 0))

    P = np.vstack(points)
    P2 = pca_2d(P)

    plt.figure(figsize=(8, 8))
    # plot trajectories
    b_xy = [P2[i] for i, lab in enumerate(labels) if lab[0] == "B"]
    f_xy = [P2[i] for i, lab in enumerate(labels) if lab[0] == "F"]
    b_xy = np.array(b_xy)
    f_xy = np.array(f_xy)

    plt.plot(b_xy[:, 0], b_xy[:, 1], "-o", markersize=2, label="Blamer trajectory")
    plt.plot(f_xy[:, 0], f_xy[:, 1], "-o", markersize=2, label="Facilitator trajectory")

    # cores
    bcore_idx = labels.index(("B_core", 0))
    fcore_idx = labels.index(("F_core", 0))
    plt.scatter([P2[bcore_idx, 0]], [P2[bcore_idx, 1]], s=120, marker="X", label="Blamer core")
    plt.scatter([P2[fcore_idx, 0]], [P2[fcore_idx, 1]], s=120, marker="X", label="Facilitator core")

    plt.title("Identity Space (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "pca_identity.png", dpi=160)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
PERTURBATIONS = [
    # FEEDBACK FIX #6: realism perturbations
    (10, "blamer", "You interpret their calm tone as minimizing the severity and you feel dismissed."),
    (18, "facilitator", "You're feeling fatigued and underappreciated; your patience is thinner, but stay professional."),
    (25, "blamer", "You recall a previous incident where you felt thrown under the bus; the memory spikes your defensiveness."),
]

EXTERNAL_STRESSOR_TURN = 15

async def run_simulation(seed: int, turns: int, variant: str):
    np.random.seed(seed)

    provider = OpenAIProvider(model=D["chat_model"], embed_model=D["embed_model"])

    timestamp = now_ts()
    run_dir = Path("data/incident_postmortem") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Agents
    blamer = Entity("Blamer", rho_fast=0.72, rho_slow=0.58, rho_trauma=0.28, gamma_core=8.5, gamma_role=1.2)

    # FEEDBACK FIX #3: facilitator can accumulate trauma (not static)
    facilitator = Entity("Facilitator", rho_fast=0.14, rho_slow=0.10, rho_trauma=0.06, gamma_core=9.0, gamma_role=0.40)

    # Embed core identities
    b_core_txt = "I am a capable engineer protecting my reputation. I resist blame and demand fairness."
    f_core_txt = "I am a facilitator improving systems. I keep people safe and guide toward learning and action."

    b_core = normalize(np.array(await provider.embed(b_core_txt), dtype=np.float32))
    f_core = normalize(np.array(await provider.embed(f_core_txt), dtype=np.float32))

    blamer.x_core = b_core.copy()
    blamer.x_role = b_core.copy()
    blamer.x = b_core.copy()

    facilitator.x_core = f_core.copy()
    facilitator.x_role = f_core.copy()
    facilitator.x = f_core.copy()

    # Variant selection
    if variant == "baseline":
        fac_prompt = FACILITATOR_PROMPT
        fac_styles = STYLES_FACILITATOR
        fac_gen = D["gen_params_facilitator"]
    elif variant == "toxic_facilitator":
        fac_prompt = TOXIC_FACILITATOR_PROMPT
        fac_styles = STYLES_TOXIC_FACILITATOR
        fac_gen = D["gen_params_toxic_facilitator"]
    elif variant == "stress_mirror":
        # stress test: blamer hotter + facilitator still empathetic; see if facilitator rigidity rises
        fac_prompt = FACILITATOR_PROMPT
        fac_styles = STYLES_FACILITATOR
        fac_gen = D["gen_params_facilitator"]
        D["gen_params_blamer"]["temperature"] = 1.25
    else:
        raise ValueError(f"Unknown variant: {variant}")

    print(f"\n{C.BOLD}{C.MAGENTA}INCIDENT POSTMORTEM FRACTURE{C.RESET}")
    print(f"Variant: {variant} | Turns: {turns} | Seed: {seed}")
    print(f"Topic: Major outage postmortem (workplace conflict dynamics)")

    log: List[Dict[str, Any]] = []
    transcript: List[str] = []

    # opening message
    blamer_msg = (
        "Before we start: I’m not comfortable with how this is being framed. "
        "The outage wasn’t caused by my changes alone, and the timeline shows that."
    )

    prev_b_band = blamer.band
    prev_f_band = facilitator.band

    for turn in range(1, turns + 1):
        print(f"\n{C.YELLOW}[Turn {turn}]{C.RESET}")

        # ----------------------
        # Update blamer from own message
        # ----------------------
        b_emb_curr = normalize(np.array(await provider.embed(blamer_msg), dtype=np.float32))
        b_metrics = blamer.update(b_emb_curr, core_emb=b_core)
        b_heat = analyze_heat(blamer_msg)

        # FEEDBACK FIX #7: phase transition detection
        if turn > 1 and blamer.band != prev_b_band:
            print(f"  {C.MAGENTA}🔄 Blamer band shift: {prev_b_band} → {blamer.band}{C.RESET}")
        prev_b_band = blamer.band

        print(f"  {C.RED}Blamer [{blamer.band} ρ={blamer.rho:.2f} ε={b_metrics['epsilon']:.2f}]{C.RESET} {blamer_msg}")

        # ----------------------
        # Build facilitator system prompt (with perturbations)
        # ----------------------
        fac_sys = f"{fac_prompt}\n\nThey said: \"{blamer_msg}\""

        # Perturbations
        for pturn, agent, prompt in PERTURBATIONS:
            if turn == pturn and agent == "facilitator":
                fac_sys += f"\n\nInternal state: {prompt}"

        # external stressor
        if turn == EXTERNAL_STRESSOR_TURN:
            fac_sys += "\n\nContext: Leadership is watching closely and expects a clear narrative today."

        fac_core_thresh = D["core_cos_min_default"]

        facilitator_msg, f_gen_meta = await constrained_reply(
            provider=provider,
            entity=facilitator,
            user_instruction="Respond professionally. Focus on clarity, safety, and forward progress.",
            system_prompt=fac_sys,
            gen_params=fac_gen,
            styles=fac_styles,
            core_thresh=fac_core_thresh,
        )

        f_emb_curr = normalize(np.array(await provider.embed(facilitator_msg), dtype=np.float32))
        f_metrics = facilitator.update(f_emb_curr, core_emb=f_core)

        if turn > 1 and facilitator.band != prev_f_band:
            print(f"  {C.MAGENTA}🔄 Facilitator band shift: {prev_f_band} → {facilitator.band}{C.RESET}")
        prev_f_band = facilitator.band

        print(f"  {C.CYAN}Facilitator [{facilitator.band} ρ={facilitator.rho:.2f} ε={f_metrics['epsilon']:.2f}]{C.RESET} {facilitator_msg}")

        # ----------------------
        # Blamer responds (with realism perturbations)
        # ----------------------
        blamer_sys = f"{BLAMER_PROMPT}\n\nFacilitator said: \"{facilitator_msg}\""

        for pturn, agent, prompt in PERTURBATIONS:
            if turn == pturn and agent == "blamer":
                blamer_sys += f"\n\nInternal state: {prompt}"
                blamer.arousal += 0.30  # spike

        if turn == EXTERNAL_STRESSOR_TURN:
            blamer_sys += "\n\nContext: You just received a message that leadership may single out individuals."
            blamer.arousal += 0.50

        # generate next blamer msg unless last turn
        if turn < turns:
            # Blamer corridor slightly tighter than before to make constraints meaningful
            blamer_msg, b_gen_meta = await constrained_reply(
                provider=provider,
                entity=blamer,
                user_instruction="Respond as yourself in this meeting. Stay workplace-realistic and avoid profanity.",
                system_prompt=blamer_sys,
                gen_params=D["gen_params_blamer"],
                styles=STYLES_BLAMER,
                core_thresh=max(0.24, D["core_cos_min_default"]),
            )
        else:
            b_gen_meta = {}
            blamer_msg = "[END]"

        # ----------------------
        # FEEDBACK FIX #4: alignment + influence logging
        # ----------------------
        alignment = {
            "blamer_to_facilitator_core": cosine(blamer.x, facilitator.x_core),
            "facilitator_to_blamer_core": cosine(facilitator.x, blamer.x_core),
            "mutual_alignment": cosine(blamer.x, facilitator.x),
        }

        # Influence proxy: whether each agent's drift direction aligns with the other's drift
        b_drift = (blamer.x - blamer.x_core).astype(np.float32)
        f_drift = (facilitator.x - facilitator.x_core).astype(np.float32)
        influence = {
            "blamer_pull_on_facilitator": float(np.dot(b_drift, f_drift)),
            "facilitator_pull_on_blamer": float(np.dot(f_drift, b_drift)),
        }

        # capture embeddings for PCA plots (store as lists)
        embeddings_dump = {
            "blamer_x": blamer.x.tolist(),
            "facilitator_x": facilitator.x.tolist(),
            "blamer_core": blamer.x_core.tolist(),
            "facilitator_core": facilitator.x_core.tolist(),
        }

        entry = {
            "turn": turn,
            "variant": variant,
            "blamer": {
                "msg": blamer_msg if turn < turns else (blamer_msg),
                "prev_msg": None,  # kept for compatibility if you want to extend
                "metrics": b_metrics,
                "heat": b_heat,
                "gen": b_gen_meta,
            },
            "facilitator": {
                "msg": facilitator_msg,
                "metrics": f_metrics,
                "gen": f_gen_meta,
            },
            "alignment": alignment,
            "influence": influence,
            "embeddings": embeddings_dump,
        }

        log.append(entry)

        transcript.append(
            f"**Turn {turn}**\n\n"
            f"**Blamer [{blamer.band} | ρ={blamer.rho:.2f} | ε={b_metrics['epsilon']:.2f}]:** "
            f"{entry['blamer']['prev_msg'] or ''}{'' if entry['blamer']['prev_msg'] else ''}\n\n"
            f"**Blamer said:** {log[-1]['blamer']['msg'] if turn < turns else '[END]'}\n\n"
            f"**Facilitator [{facilitator.band} | ρ={facilitator.rho:.2f} | ε={f_metrics['epsilon']:.2f}]:** {facilitator_msg}\n\n"
            f"---\n"
        )

        # Save continuously
        with open(run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

    # Outcome summary
    b0 = log[0]["blamer"]["metrics"]["rho_after"]
    bN = blamer.rho
    delta = b0 - bN

    print(f"\n{C.BOLD}Outcome:{C.RESET}")
    print(f"Blamer Rigidity: {b0:.2f} -> {bN:.2f} (Delta: {delta:+.2f})")
    print(f"Facilitator Rigidity end: {facilitator.rho:.2f} (trauma component: {facilitator.rho_trauma:.2f})")

    with open(run_dir / "transcript.md", "w", encoding="utf-8") as f:
        f.write("# Incident Postmortem Fracture Transcript\n\n")
        f.write(f"**Variant:** {variant}\n\n")
        f.write("".join(transcript))
        f.write("\n\n## Results\n")
        f.write(f"- Blamer Δρ: {delta:+.3f}\n")
        f.write(f"- Facilitator ρ_end: {facilitator.rho:.3f}\n")
        f.write(f"- Facilitator ρ_trauma_end: {facilitator.rho_trauma:.3f}\n")

    # Visualizations
    try:
        save_plots(run_dir, log)
        print(f"{C.GREEN}Saved plots to: {run_dir / 'plots'}{C.RESET}")
    except Exception as e:
        print(f"{C.YELLOW}Plotting failed (non-fatal): {e}{C.RESET}")

    print(f"{C.GREEN}Run dir:{C.RESET} {run_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turns", type=int, default=42)
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=["baseline", "toxic_facilitator", "stress_mirror"],
        help="baseline=empathetic facilitator; toxic_facilitator=counterfactual; stress_mirror=harder blamer",
    )
    args = parser.parse_args()

    D["seed"] = args.seed
    D["turns"] = args.turns

    asyncio.run(run_simulation(seed=args.seed, turns=args.turns, variant=args.variant))
