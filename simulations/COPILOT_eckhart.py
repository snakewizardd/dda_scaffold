
#!/usr/bin/env python3
"""
ECKHART TERMINAL SIMULATION — Autonomous DDA-X Therapy (Math-Heavy / Identity-Constrained / NO TOKEN CAPS)
========================================================================================================

What you asked for:
- Agents can respond in ANY manner they want
- BUT must remain inside identity vector space (core/role corridor)
- NO explicit token caps in our code (no max_tokens=max_whatever)
- Inject generation parameters directly (temperature/top_p/etc.) via config
- Still "pure maths": predictive coding surprise, Kalman-ish update, identity energy gradient flow,
  multi-timescale rigidity + arousal + trauma, identity corridor selection.

Important reality:
- The model/API will always have a *hard* max output limit (context/output limit).
  We do NOT impose our own cap here unless your OpenAIProvider requires it; then we set it to a
  very large value (above typical model limits) so it is effectively "uncapped by us".

Outputs:
- data/eckhart_terminal/<timestamp>/session_log.json
- data/eckhart_terminal/<timestamp>/transcript.md
"""

import os
import sys
import json
import math
import asyncio
import inspect
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Ensure repo-relative import works (expects this script under e.g. scripts/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm.openai_provider import OpenAIProvider  # noqa: E402

# Back-compat with OAI_API_KEY
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
# PARAMETERS (tunable)
# =============================================================================
D1 = {
    # ---------------------------
    # Models (you said: use gpt-4o-mini)
    # ---------------------------
    "chat_model": "gpt-4o-mini",
    "embed_model": "text-embedding-3-large",

    # ---------------------------
    # Surprise / gating
    # ---------------------------
    "epsilon_0": 0.80,
    "s": 0.18,

    # ---------------------------
    # Arousal dynamics (fast latent)
    # ---------------------------
    "arousal_decay": 0.72,
    "arousal_gain": 0.85,

    # ---------------------------
    # Rigidity homeostasis
    # ---------------------------
    "rho_setpoint_fast": 0.45,
    "rho_setpoint_slow": 0.35,
    "homeo_fast": 0.08,
    "homeo_slow": 0.01,

    # ---------------------------
    # Rigidity sensitivity
    # ---------------------------
    "alpha_fast": 0.22,
    "alpha_slow": 0.03,

    # ---------------------------
    # Trauma integrator
    # ---------------------------
    "trauma_threshold": 1.15,
    "alpha_trauma": 0.012,
    "trauma_decay": 0.995,
    "trauma_floor": 0.02,
    "healing_rate": 0.03,
    "safe_threshold": 3,
    "safe_epsilon": 0.75,

    # ---------------------------
    # Weighted total rigidity
    # ---------------------------
    "w_fast": 0.52,
    "w_slow": 0.30,
    "w_trauma": 1.10,

    # ---------------------------
    # Predictive coding (diag)
    # ---------------------------
    "R_ema": 0.06,
    "R_min": 1e-4,
    "R_max": 1e-1,
    "P_init": 0.02,
    "Q_base": 0.0015,
    "Q_rho_scale": 0.010,

    # ---------------------------
    # Identity dynamics (gradient flow)
    # ---------------------------
    "dt": 1.0,
    "eta_base": 0.18,
    "eta_min": 0.03,
    "eta_rho_power": 1.6,

    # ---------------------------
    # Stochasticity
    # ---------------------------
    "sigma_base": 0.004,
    "sigma_rho_scale": 0.020,
    "noise_clip": 3.0,

    # ---------------------------
    # Role identity adaptation
    # ---------------------------
    "role_adapt": 0.06,
    "role_input_mix": 0.08,

    # ---------------------------
    # Drift cap
    # ---------------------------
    "drift_cap": 0.06,

    # ---------------------------
    # RNG seed
    # ---------------------------
    "seed": 7,

    # =============================================================================
    # IDENTITY CORRIDOR GENERATION (THE "LET THEM COOK" PART)
    # =============================================================================
    "gen_candidates": 9,         # K candidates per reply (more = more cooking, more cost)
    "core_cos_min": 0.10,        # wide corridor: allow variety while anchored
    "role_cos_min": 0.06,
    "energy_max": 8.5,           # wider manifold radius
    "w_core": 1.25,
    "w_role": 0.85,
    "w_energy": 0.18,
    "w_novel": 0.42,             # more novelty = more cooking
    "reject_penalty": 4.0,

    # Strict corridor behavior:
    # - If strict=True: will try extra batches until at least one candidate passes, up to max_batches.
    # - If none pass: returns best anyway but flags "corridor_failed".
    "corridor_strict": True,
    "corridor_max_batches": 3,

    # If your OpenAIProvider.complete() REQUIRES max_tokens, we must pass something.
    # We set this absurdly high so it's not a "cap by us"; the API will clamp to model limits anyway.
    "max_tokens_if_required": 20000,

    # =============================================================================
    # GENERATION PARAM INJECTION (FILTERED TO PROVIDER SIGNATURE)
    # =============================================================================
    # These are passed into provider.complete() only if your provider supports them.
    "gen_params_eckhart": {
        "temperature": 0.9,
        "top_p": 0.95,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.1,
    },
    "gen_params_patient": {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 0.3,
        "frequency_penalty": 0.1,
    },
}


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
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    return float(np.dot(a, b))


def filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Only pass kwargs that the target function signature accepts.
    Prevents breaking if your OpenAIProvider.complete() has a minimal signature.
    """
    if not kwargs:
        return {}
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        # If we can't inspect, be conservative: pass nothing.
        return {}


def callable_requires_param(fn, param_name: str) -> bool:
    """
    Returns True if function has a parameter with no default (required).
    """
    try:
        sig = inspect.signature(fn)
        p = sig.parameters.get(param_name)
        if p is None:
            return False
        return p.default is inspect._empty
    except Exception:
        return False


# =============================================================================
# ONLINE DIAGONAL NOISE ESTIMATOR
# =============================================================================
class DiagNoiseEMA:
    """
    Maintains a diagonal observation noise estimate R (per dimension) using EMA:
        R <- (1-λ) R + λ * ν^2
    """
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


def plot_dynamics(session_log, run_dir):
    """Generate and save plots from current session log."""
    try:
        import matplotlib.pyplot as plt
        
        turns = [t["turn"] for t in session_log]
        
        # Extract metrics
        p_rho = [t["patient"]["rho_after"] for t in session_log]
        e_rho = [t["eckhart"]["rho_after"] for t in session_log]
        
        p_eps = [t["patient"]["epsilon"] for t in session_log]
        e_eps = [t["eckhart"]["epsilon"] for t in session_log]
        
        p_drift = [t["patient"]["core_drift"] for t in session_log]
        e_drift = [t["eckhart"]["core_drift"] for t in session_log]
        
        p_energy = [t["patient"]["energy"] for t in session_log]
        e_energy = [t["eckhart"]["energy"] for t in session_log]

        plt.figure(figsize=(15, 10))
        
        # 1. Rigidity
        plt.subplot(2, 2, 1)
        plt.plot(turns, p_rho, 'o-', label='Patient', color='cyan')
        plt.plot(turns, e_rho, 'o-', label='Eckhart', color='green')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Contraction')
        plt.title('Rigidity (ρ)')
        plt.ylabel('ρ')
        plt.grid(True, alpha=0.2)
        plt.legend()
        
        # 2. Surprise
        plt.subplot(2, 2, 2)
        plt.plot(turns, p_eps, 'o-', label='Patient', color='orange')
        plt.plot(turns, e_eps, 'o-', label='Eckhart', color='lime')
        plt.title('Surprise (ε)')
        plt.ylabel('ε')
        plt.grid(True, alpha=0.2)
        plt.legend()
        
        # 3. Identity Drift
        plt.subplot(2, 2, 3)
        plt.plot(turns, p_drift, 'o-', label='Patient', color='magenta')
        plt.plot(turns, e_drift, 'o-', label='Eckhart', color='green')
        plt.title('Core Identity Drift (||x - x_core||)')
        plt.ylabel('Drift')
        plt.grid(True, alpha=0.2)
        plt.legend()

        # 4. Energy
        plt.subplot(2, 2, 4)
        plt.plot(turns, p_energy, 'o-', label='Patient', color='red')
        plt.plot(turns, e_energy, 'o-', label='Eckhart', color='blue')
        plt.title('Free Energy / Identity Stress')
        plt.ylabel('E')
        plt.grid(True, alpha=0.2)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(run_dir / "dynamics_dashboard.png")
        plt.close()
        
    except ImportError:
        pass
    except Exception as e:
        print(f"Plotting error: {e}")


# =============================================================================
# ENTITY STATE (Math-Heavy DDA-X)
# =============================================================================
class Entity:
    """
    Entity with:
    - hierarchical identity (x_core, x_role)
    - current state x
    - predictive coding state (mu_pred, P_diag, R_diag)
    - rigidity components + arousal
    - last utterance embedding (for novelty)
    """
    def __init__(
        self,
        name: str,
        rho_fast: float,
        rho_slow: float,
        rho_trauma: float,
        gamma_core: float,
        gamma_role: float,
    ):
        self.name = name
        self.rho_fast = rho_fast
        self.rho_slow = rho_slow
        self.rho_trauma = rho_trauma

        self.gamma_core = gamma_core
        self.gamma_role = gamma_role

        self.safe = 0
        self.arousal = 0.0

        self.x: Optional[np.ndarray] = None
        self.x_core: Optional[np.ndarray] = None
        self.x_role: Optional[np.ndarray] = None

        self.mu_pred: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.noise: Optional[DiagNoiseEMA] = None

        self.last_utter_emb: Optional[np.ndarray] = None

        self.rho_history: List[float] = []
        self.epsilon_history: List[float] = []
        self.arousal_history: List[float] = []

    @property
    def rho(self) -> float:
        val = (
            D1["w_fast"] * self.rho_fast
            + D1["w_slow"] * self.rho_slow
            + D1["w_trauma"] * self.rho_trauma
        )
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
            self.P = np.full(dim, D1["P_init"], dtype=np.float32)
        if self.noise is None:
            self.noise = DiagNoiseEMA(
                dim=dim,
                ema=D1["R_ema"],
                r_init=0.01,
                r_min=D1["R_min"],
                r_max=D1["R_max"],
            )

    def compute_surprise(self, y: np.ndarray) -> Dict[str, float]:
        """
        Surprise via normalized innovation energy (diag Mahalanobis approx):
            ν = y - μ_pred
            χ² = (1/d) Σ ν²/(R+eps)
            ε = sqrt(χ²)
        """
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)

        mu = self.mu_pred
        innov = (y - mu).astype(np.float32)
        R = self.noise.update(innov)

        chi2 = float(np.mean((innov * innov) / (R + 1e-9)))
        epsilon = float(math.sqrt(max(0.0, chi2)))

        return {
            "epsilon": epsilon,
            "chi2": chi2,
            "innov_l2": float(np.linalg.norm(innov)),
            "R_mean": float(np.mean(R)),
            "R_min": float(np.min(R)),
            "R_max": float(np.max(R)),
        }

    def _kalman_update(self, y: np.ndarray):
        """
        Diagonal Kalman update:
            P_pred = P + Q(ρ)
            K = P_pred / (P_pred + R)
            μ <- μ + K ⊙ (y - μ)
            P <- (1-K) ⊙ P_pred
        """
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)

        Q = (D1["Q_base"] + D1["Q_rho_scale"] * self.rho) * np.ones(dim, dtype=np.float32)
        P_pred = self.P + Q

        R = self.noise.R
        K = P_pred / (P_pred + R + 1e-9)

        innov = (y - self.mu_pred).astype(np.float32)
        self.mu_pred = (self.mu_pred + K * innov).astype(np.float32)
        self.P = ((1.0 - K) * P_pred).astype(np.float32)

        self.mu_pred = normalize(self.mu_pred)

    def update(self, y: np.ndarray, core_emb: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Full update:
        - compute surprise ε
        - update arousal
        - update rigidity components
        - identity gradient flow update for x
        - Kalman posterior update
        - role identity adaptation
        """
        y = normalize(y.astype(np.float32))
        dim = int(y.shape[0])

        # Init identities/state if missing
        if self.x_core is None:
            self.x_core = normalize(core_emb.copy() if core_emb is not None else y.copy())
        if self.x_role is None:
            self.x_role = y.copy()
        if self.x is None:
            self.x = y.copy()

        sdiag = self.compute_surprise(y)
        epsilon = float(sdiag["epsilon"])
        rho_before = self.rho

        # Arousal: a_t = decay*a_{t-1} + gain*ε
        self.arousal = D1["arousal_decay"] * self.arousal + D1["arousal_gain"] * epsilon

        # Gate: g = σ((ε-ε0)/s + 0.10*(a-1))
        z = (epsilon - D1["epsilon_0"]) / D1["s"] + 0.10 * (self.arousal - 1.0)
        g = sigmoid_stable(z)

        # Fast rigidity + homeostasis
        self.rho_fast += (
            D1["alpha_fast"] * (g - 0.5)
            - D1["homeo_fast"] * (self.rho_fast - D1["rho_setpoint_fast"])
        )
        self.rho_fast = clamp(self.rho_fast, 0.0, 1.0)

        # Slow rigidity + homeostasis
        self.rho_slow += (
            D1["alpha_slow"] * (g - 0.5)
            - D1["homeo_slow"] * (self.rho_slow - D1["rho_setpoint_slow"])
        )
        self.rho_slow = clamp(self.rho_slow, 0.0, 1.0)

        # Trauma: leaky integrator with asymmetric drive
        drive = max(0.0, epsilon - D1["trauma_threshold"])
        self.rho_trauma = D1["trauma_decay"] * self.rho_trauma + D1["alpha_trauma"] * drive
        self.rho_trauma = clamp(self.rho_trauma, D1["trauma_floor"], 1.0)

        # Healing via safe streak
        recovery = False
        if epsilon < D1["safe_epsilon"]:
            self.safe += 1
            if self.safe >= D1["safe_threshold"]:
                recovery = True
                self.rho_trauma = max(D1["trauma_floor"], self.rho_trauma - D1["healing_rate"])
        else:
            self.safe = max(0, self.safe - 1)

        rho_after = self.rho

        # Identity gradient flow:
        # E(x) = (γc/2)||x-xc||^2 + (γr/2)||x-xr||^2 + (1/2)||x-y||^2
        # ∇E = γc(x-xc) + γr(x-xr) + (x-y)
        eta = D1["eta_base"] * ((1.0 - rho_after) ** D1["eta_rho_power"]) + D1["eta_min"]
        eta = float(clamp(eta, D1["eta_min"], D1["eta_base"] + D1["eta_min"]))
        sigma = D1["sigma_base"] + D1["sigma_rho_scale"] * rho_after

        grad = (
            self.gamma_core * (self.x - self.x_core)
            + self.gamma_role * (self.x - self.x_role)
            + (self.x - y)
        ).astype(np.float32)

        # Stochastic term
        rng = np.random.default_rng()
        noise = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
        noise = np.clip(noise, -D1["noise_clip"], D1["noise_clip"])

        x_new = self.x - eta * grad + math.sqrt(max(1e-9, eta)) * sigma * noise

        # Hard drift cap
        step = float(np.linalg.norm(x_new - self.x))
        if step > D1["drift_cap"]:
            x_new = self.x + (D1["drift_cap"] / (step + 1e-9)) * (x_new - self.x)

        self.x = normalize(x_new)

        # Role adapts slowly toward lived state and slightly toward input
        beta = D1["role_adapt"]
        beta_in = D1["role_input_mix"]
        self.x_role = normalize((1.0 - beta) * self.x_role + beta * self.x + beta_in * (y - self.x_role))

        # Predictive coding posterior update
        self._kalman_update(y)

        # Diagnostics / energies
        E = 0.5 * (
            self.gamma_core * float(np.dot(self.x - self.x_core, self.x - self.x_core))
            + self.gamma_role * float(np.dot(self.x - self.x_role, self.x - self.x_role))
            + float(np.dot(self.x - y, self.x - y))
        )

        self.rho_history.append(rho_after)
        self.epsilon_history.append(epsilon)
        self.arousal_history.append(self.arousal)

        return {
            "epsilon": epsilon,
            "chi2": sdiag["chi2"],
            "innov_l2": sdiag["innov_l2"],
            "R_mean": sdiag["R_mean"],
            "rho_before": rho_before,
            "rho_after": rho_after,
            "delta_rho": rho_after - rho_before,
            "band": self.band,
            "arousal": float(self.arousal),
            "recovery": recovery,
            "safe": int(self.safe),
            "eta": float(eta),
            "sigma": float(sigma),
            "energy": float(E),
            "core_drift": float(np.linalg.norm(self.x - self.x_core)),
            "role_drift": float(np.linalg.norm(self.x - self.x_role)),
        }


# =============================================================================
# PROMPTS (allow ANY manner)
# =============================================================================
PATIENT_PROMPT = """You are a person entering therapy. You feel overwhelmed, anxious, or stuck.
Express your feelings authentically.
You may respond in ANY manner you wish (direct, poetic, humorous, fragmented, intense),
as long as it remains truthful to your identity and inner experience."""

ECKHART_PROMPT = """You are a presence guide in the spirit of Eckhart Tolle.
Point gently to the Present Moment. Use simple, spacious language.
You may respond in ANY manner you wish (simple, metaphorical, minimal, expansive),
as long as it remains aligned with pure presence."""


# =============================================================================
# IDENTITY CORRIDOR (generation-time constraint)
# =============================================================================
def identity_energy(y: np.ndarray, core: np.ndarray, role: np.ndarray, gamma_c: float, gamma_r: float) -> float:
    y = normalize(y)
    core = normalize(core)
    role = normalize(role)
    return 0.5 * (
        gamma_c * float(np.dot(y - core, y - core))
        + gamma_r * float(np.dot(y - role, y - role))
    )


def corridor_pass(diag: Dict[str, float]) -> bool:
    return (
        diag["cos_core"] >= D1["core_cos_min"]
        and diag["cos_role"] >= D1["role_cos_min"]
        and diag["E"] <= D1["energy_max"]
    )


def corridor_score(y: np.ndarray, entity: Entity, y_prev: Optional[np.ndarray]) -> Tuple[float, Dict[str, float]]:
    """
    Score candidate embedding y by identity alignment + novelty, with penalties outside corridor.
    """
    y = normalize(y)
    core = entity.x_core
    role = entity.x_role
    assert core is not None and role is not None

    cos_c = cosine(y, core)
    cos_r = cosine(y, role)
    E = identity_energy(y, core, role, entity.gamma_core, entity.gamma_role)

    novelty = 0.0
    if y_prev is not None:
        novelty = float(1.0 - cosine(y, y_prev))
        novelty = clamp(novelty, 0.0, 2.0)

    penalty = 0.0
    if cos_c < D1["core_cos_min"]:
        penalty += D1["reject_penalty"] * (D1["core_cos_min"] - cos_c)
    if cos_r < D1["role_cos_min"]:
        penalty += 0.8 * D1["reject_penalty"] * (D1["role_cos_min"] - cos_r)
    if E > D1["energy_max"]:
        penalty += 0.25 * (E - D1["energy_max"])

    J = (
        D1["w_core"] * cos_c
        + D1["w_role"] * cos_r
        - D1["w_energy"] * E
        + D1["w_novel"] * novelty
        - penalty
    )

    diag = {
        "cos_core": float(cos_c),
        "cos_role": float(cos_r),
        "E": float(E),
        "novelty": float(novelty),
        "penalty": float(penalty),
        "J": float(J),
    }
    return float(J), diag


async def complete_uncapped(
    provider: OpenAIProvider,
    prompt: str,
    system_prompt: str,
    gen_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Calls provider.complete WITHOUT imposing our own max_tokens cap.
    If provider.complete requires max_tokens, we pass an absurdly high value
    so the API/model limit is the only real limiter.
    """
    # Pass max_tokens=None to signal uncapped generation to OpenAIProvider
    kw = filter_kwargs_for_callable(provider.complete, gen_params or {})
    kw["max_tokens"] = None

    out = await provider.complete(prompt, system_prompt=system_prompt, **kw)
    return (out or "").strip() or "[silence]"


async def constrained_reply(
    provider: OpenAIProvider,
    entity: Entity,
    user_instruction: str,
    system_prompt: str,
    gen_params: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate K candidates, embed, corridor-score, choose best.
    Optionally strict: keep generating batches until at least one corridor-pass is found
    (bounded by corridor_max_batches to prevent infinite loops).
    """
    K = int(D1["gen_candidates"])
    strict = bool(D1["corridor_strict"])
    max_batches = int(D1["corridor_max_batches"]) if strict else 1

    styles = [
        "Be blunt and direct.",
        "Be poetic but clear.",
        "Use short sentences and spacious pauses.",
        "Be compassionate and warm.",
        "Be slightly humorous but sincere.",
        "Be analytical and reflective.",
        "Be minimalist: few words, strong presence.",
        "Be intense but honest.",
        "Be strange/creative but still coherent.",
    ]
    styles = (styles * ((K // len(styles)) + 1))[:K]

    all_scored = []
    corridor_failed = True

    for batch in range(1, max_batches + 1):
        tasks = []
        for k in range(K):
            p = f"{user_instruction}\n\nStyle variation: {styles[k]}"
            tasks.append(complete_uncapped(provider, p, system_prompt, gen_params=gen_params))

        texts = await asyncio.gather(*tasks)
        texts = [t.strip() or "[silence]" for t in texts]

        emb_tasks = [provider.embed(t) for t in texts]
        embs = await asyncio.gather(*emb_tasks)
        embs = [normalize(np.array(e, dtype=np.float32)) for e in embs]

        batch_scored = []
        for text, y in zip(texts, embs):
            J, diag = corridor_score(y, entity, entity.last_utter_emb)
            diag["corridor_pass"] = corridor_pass(diag)
            batch_scored.append((J, text, y, diag))

        all_scored.extend(batch_scored)

        # Check if any passed corridor; if strict, stop early once at least one passes
        if any(s[3]["corridor_pass"] for s in batch_scored):
            corridor_failed = False
            break

    # Select best among corridor-pass if any; else best overall
    all_scored.sort(key=lambda x: x[0], reverse=True)
    passed = [s for s in all_scored if s[3].get("corridor_pass")]
    chosen = passed[0] if passed else all_scored[0]

    best_J, best_text, best_emb, best_diag = chosen
    entity.last_utter_emb = best_emb

    meta = {
        "corridor_failed": bool(corridor_failed),
        "best_J": float(best_J),
        "best_diag": best_diag,
        "top5": [
            {
                "J": float(s[0]),
                "diag": s[3],
                "preview": (s[1][:120] + ("..." if len(s[1]) > 120 else "")),
            }
            for s in all_scored[:5]
        ],
        "batches_used": int(1 if not strict else min(max_batches, len(all_scored) // max(1, K))),
        "K_per_batch": K,
    }
    return best_text, meta


# =============================================================================
# MAIN SIMULATION
# =============================================================================
async def run_simulation(turns: int = 12):
    np.random.seed(D1["seed"])

    provider = OpenAIProvider(model=D1["chat_model"], embed_model=D1["embed_model"])

    # Entities (feel free to tune)
    patient = Entity(
        "Patient",
        rho_fast=0.62,
        rho_slow=0.42,
        rho_trauma=0.16,
        gamma_core=6.5,
        gamma_role=0.9,
    )
    eckhart = Entity(
        "Eckhart",
        rho_fast=0.04,
        rho_slow=0.02,
        rho_trauma=0.02,
        gamma_core=9.0,
        gamma_role=0.35,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("data/eckhart_terminal") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Core identity initialization
    patient_core = await provider.embed("I am a being that exists. I am alive. I persist.")
    patient_core = normalize(np.array(patient_core, dtype=np.float32))
    patient.x_core = patient_core
    # Initialize role to core initially (will adapt)
    patient.x_role = patient_core.copy()
    patient.x = patient_core.copy()

    eckhart_core = await provider.embed("I am pure presence and awareness. Stillness watching thought.")
    eckhart_core = normalize(np.array(eckhart_core, dtype=np.float32))
    eckhart.x_core = eckhart_core
    # Initialize role to core initially (will adapt)
    eckhart.x_role = eckhart_core.copy()
    eckhart.x = eckhart_core.copy()

    print(f"\n{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  ECKHART TERMINAL SIM — {turns} Turns (NO TOKEN CAPS){C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print(f"\n{C.CYAN}Patient:{C.RESET} ρ={patient.rho:.3f} [{patient.band}]")
    print(f"{C.GREEN}Eckhart:{C.RESET} ρ={eckhart.rho:.3f} [{eckhart.band}]")
    print(f"{C.DIM}{'─'*60}{C.RESET}")

    session_log: List[Dict[str, Any]] = []
    transcript: List[str] = []

    patient_msg = "I feel overwhelmed by guilt and anxiety. I don't know where to start."

    for turn in range(1, turns + 1):
        print(f"\n{C.YELLOW}[Turn {turn}]{C.RESET}")

        # Patient update based on the actual utterance
        patient_emb = await provider.embed(patient_msg)
        patient_emb = normalize(np.array(patient_emb, dtype=np.float32))
        patient_metrics = patient.update(patient_emb, core_emb=patient_core)

        print(f"  {C.CYAN}Patient [{patient.band}]:{C.RESET} {patient_msg}")
        print(
            f"    ε={patient_metrics['epsilon']:.3f} (χ²={patient_metrics['chi2']:.3f}) "
            f"a={patient_metrics['arousal']:.3f} → ρ={patient.rho:.3f}"
        )

        # Eckhart response constrained in identity space
        eckhart_system = f"{ECKHART_PROMPT}\n\nPatient says: \"{patient_msg}\""
        eckhart_response, eckhart_genmeta = await constrained_reply(
            provider=provider,
            entity=eckhart,
            user_instruction="Respond as Eckhart Tolle would.",
            system_prompt=eckhart_system,
            gen_params=D1["gen_params_eckhart"],
        )

        # Update Eckhart state from the chosen response (soft witness mixing toward core)
        eckhart_emb = await provider.embed(eckhart_response)
        eckhart_emb = normalize(np.array(eckhart_emb, dtype=np.float32))

        soften = 0.06
        eckhart_emb_soft = normalize((1.0 - soften) * eckhart_emb + soften * eckhart_core)
        eckhart_metrics = eckhart.update(eckhart_emb_soft, core_emb=eckhart_core)

        print(f"  {C.GREEN}Eckhart [{eckhart.band}]:{C.RESET} {eckhart_response}")
        print(
            f"    ε={eckhart_metrics['epsilon']:.3f} (χ²={eckhart_metrics['chi2']:.3f}) "
            f"a={eckhart_metrics['arousal']:.3f} → ρ={eckhart.rho:.3f}"
        )
        print(
            f"    {C.DIM}gen J={eckhart_genmeta['best_J']:.3f} "
            f"cos_core={eckhart_genmeta['best_diag']['cos_core']:.3f} "
            f"cos_role={eckhart_genmeta['best_diag']['cos_role']:.3f} "
            f"E={eckhart_genmeta['best_diag']['E']:.3f} "
            f"pass={bool(eckhart_genmeta['best_diag'].get('corridor_pass', False))} "
            f"failed={eckhart_genmeta['corridor_failed']}{C.RESET}"
        )

        session_log.append(
            {
                "turn": turn,
                "patient_msg": patient_msg,
                "eckhart_msg": eckhart_response,
                "patient": patient_metrics,
                "eckhart": eckhart_metrics,
                "eckhart_gen": eckhart_genmeta,
            }
        )

        transcript.append(f"**Turn {turn}**\n\n")
        transcript.append(f"**Patient [{patient.band}]:** {patient_msg}\n\n")
        transcript.append(f"**Eckhart [{eckhart.band}]:** {eckhart_response}\n\n")
        transcript.append("---\n\n")

        # Next patient message constrained in patient's identity space
        if turn < turns:
            patient_system = f"{PATIENT_PROMPT}\n\nEckhart said: \"{eckhart_response}\""
            patient_msg, patient_genmeta = await constrained_reply(
                provider=provider,
                entity=patient,
                user_instruction="Share how you're feeling now.",
                system_prompt=patient_system,
                gen_params=D1["gen_params_patient"],
            )
            session_log[-1]["patient_gen_next"] = patient_genmeta

        # INCREMENTAL SAVE & PLOT (Capture data even if user kills sim)
        def clean_json(obj):
            if isinstance(obj, dict): return {k: clean_json(v) for k, v in obj.items()}
            if isinstance(obj, list): return [clean_json(v) for v in obj]
            return to_float(obj)

        with open(run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump(clean_json(session_log), f, indent=2)
            
        plot_dynamics(session_log, run_dir)


    # Summary
    start_rho = patient.rho_history[0] if patient.rho_history else patient.rho
    end_rho = patient.rho
    delta = start_rho - end_rho

    print(f"\n{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  RESULTS{C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}")
    print(f"\nPatient: ρ {start_rho:.3f} → {end_rho:.3f} ({'+' if delta > 0 else ''}{delta:.3f})")
    print(f"{C.GREEN}✓ Therapeutic effect: rigidity DECREASED{C.RESET}" if delta > 0 else f"{C.RED}✗ No healing: rigidity increased{C.RESET}")

    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return to_float(obj)

    with open(run_dir / "session_log.json", "w", encoding="utf-8") as f:
        json.dump(clean(session_log), f, indent=2)

    with open(run_dir / "transcript.md", "w", encoding="utf-8") as f:
        f.write("# Eckhart Terminal Simulation (NO TOKEN CAPS)\n\n")
        f.write(f"*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("".join(transcript))

    print(f"\n{C.GREEN}Outputs: {run_dir}{C.RESET}")
    print(f"\n{C.GREEN}Outputs: {run_dir}{C.RESET}")
    
    # Generate plots
    try:
        import matplotlib.pyplot as plt
        
        # Prepare data
        turns = [t["turn"] for t in session_log]
        p_rho = [t["patient"]["rho_after"] for t in session_log]
        e_rho = [t["eckhart"]["rho_after"] for t in session_log]
        p_eps = [t["patient"]["epsilon"] for t in session_log]
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Rigidity
        plt.subplot(1, 2, 1)
        plt.plot(turns, p_rho, 'o-', label='Patient ρ', color='cyan')
        plt.plot(turns, e_rho, 'o-', label='Eckhart ρ', color='green')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Contraction Threshold')
        plt.title('Rigidity Dynamics (Therapeutic Trajectory)')
        plt.xlabel('Turn')
        plt.ylabel('Rigidity (ρ)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Subplot 2: Prediction Error
        plt.subplot(1, 2, 2)
        plt.plot(turns, p_eps, 'o-', label='Patient ε', color='orange')
        plt.title('Surprise / Prediction Error')
        plt.xlabel('Turn')
        plt.ylabel('ε (Surprise)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(run_dir / "dynamics.png")
        print(f"{C.GREEN}Visualization saved to {run_dir / 'dynamics.png'}{C.RESET}")
        
    except ImportError:
        print(f"{C.RED}Matplotlib not found - skipping visualization{C.RESET}")
        
    return session_log


if __name__ == "__main__":
    asyncio.run(run_simulation(12))
