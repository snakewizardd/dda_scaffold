#!/usr/bin/env python3
""" 
HOUSE OF DAVID — DDA‑X Court Dynamics Simulator (Phi‑OAI Hybrid)
================================================================================

Goal
----
Instantiate a 3‑agent social simulation (David, Mychal, Mirab) using the DDA‑X
principle: **surprise → rigidity → contraction**.

Key upgrades vs scaffold
------------------------
1) **Three agents** + an environment/narrator event stream.
2) **DDA‑X contraction binding**: rigidity directly constrains:
   - decoding temperature/top_p
   - max_tokens (bandwidth)
   - candidate count (corridor search depth)
   - system “cognitive state” instructions
3) **Dual‑backend hybrid**: Azure Phi‑4 for language (“voice”), OpenAI for
   embeddings (“state space”).
4) **3000‑D state space**: projects 3072‑D embeddings into 3000‑D via a seeded
   random orthonormal-ish projection (stable across runs).
5) **Windows**: fast/slow surprise windows + trauma window.
6) **Identity drift**: explicit core/role drift ceilings per band.

Run
---
  python house_of_david_ddax_sim.py --turns 24 --seed 7

Env
---
Requires:
  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
  OPENAI_API_KEY (or OAI_API_KEY)

Notes
-----
- This is a *research simulator* for dynamics; tune params in CONFIG.
- Avoids copyrighted script reproduction; agents generate original dialogue.
"""

import os
import sys
import json
import math
import asyncio
import inspect
import re
import pickle
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI, AsyncOpenAI

load_dotenv()

# =============================================================================
# COLORS
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
# HYBRID PROVIDER (Azure Phi‑4 voice + OpenAI embeddings)
# =============================================================================
class HybridAzureProvider:
    """Azure Phi‑4 for chat completions; OpenAI for embeddings."""

    def __init__(self):
        self.azure_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21",
        )
        self.azure_model = os.getenv("AZURE_PHI_DEPLOYMENT") or "Phi-4"

        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OAI_API_KEY")
        )
        self.embed_model = os.getenv("OAI_EMBED_MODEL") or "text-embedding-3-large"  # 3072-D

        print(f"[HYBRID] Voice (Azure Chat): {self.azure_model}")
        print(f"[HYBRID] State (OpenAI Embed): {self.embed_model}")

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 800,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: self.azure_client.chat.completions.create(
                model=self.azure_model,
                messages=messages,
                max_tokens=max_tokens or 800,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            ),
        )
        return (resp.choices[0].message.content or "").strip()

    async def embed(self, text: str) -> List[float]:
        resp = await self.openai_client.embeddings.create(
            model=self.embed_model,
            input=text,
        )
        return resp.data[0].embedding

# =============================================================================
# NUMERICS
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


def to_float(x: Any) -> Any:
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not kwargs:
        return {}
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return {}

# =============================================================================
# 3000‑D PROJECTION LAYER
# =============================================================================
class Projector3000:
    """Project 3072-D embeddings to 3000-D with a seeded random projection.

    We use a fixed matrix with columns normalized; stable across a seed.
    """

    def __init__(self, in_dim: int = 3072, out_dim: int = 3000, seed: int = 42):
        self.in_dim = in_dim
        self.out_dim = out_dim
        rng = np.random.default_rng(seed)
        # Random Gaussian projection; normalize rows for stability
        W = rng.normal(0.0, 1.0, size=(out_dim, in_dim)).astype(np.float32)
        W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
        self.W = W

    def __call__(self, emb3072: np.ndarray) -> np.ndarray:
        y = self.W @ emb3072.astype(np.float32)
        return normalize(y)

# =============================================================================
# CONFIG (DDA‑X tuned)
# =============================================================================
CONFIG = {
    "turns": 24,
    "seed": 42,

    # Surprise baseline + sharpness
    "epsilon_0": 0.72,
    "s": 0.16,

    # Windows: fast/slow arousal and surprise integration
    "arousal_decay": 0.70,
    "arousal_gain": 0.90,

    # Rigidity homeostasis (fast/slow)
    "rho_set_fast": 0.42,
    "rho_set_slow": 0.33,
    "homeo_fast": 0.10,
    "homeo_slow": 0.012,
    "alpha_fast": 0.25,
    "alpha_slow": 0.035,

    # Trauma window
    "trauma_threshold": 1.10,
    "alpha_trauma": 0.020,
    "trauma_decay": 0.997,
    "trauma_floor": 0.03,
    "healing_rate": 0.012,
    "safe_threshold": 4,
    "safe_epsilon": 0.74,

    # Rigidity weighting
    "w_fast": 0.50,
    "w_slow": 0.30,
    "w_trauma": 1.20,

    # Predictive coding (diagonal Kalman-ish)
    "R_ema": 0.06,
    "R_min": 1e-4,
    "R_max": 1e-1,
    "P_init": 0.02,
    "Q_base": 0.0016,
    "Q_rho_scale": 0.012,

    # Gradient flow
    "eta_base": 0.20,
    "eta_min": 0.03,
    "eta_rho_power": 1.7,
    "sigma_base": 0.004,
    "sigma_rho_scale": 0.022,
    "noise_clip": 3.0,

    # Identity drift: global + band-scaled
    "drift_cap_base": 0.060,
    "drift_cap_contracted": 0.040,
    "drift_cap_frozen": 0.025,

    # Role adaptation
    "role_adapt": 0.06,
    "role_input_mix": 0.08,

    # Corridor scoring
    "gen_candidates_base": 8,
    "corridor_max_batches": 2,
    "corridor_strict": True,
    "log_rejections": True,

    "core_cos_min": {
        "DAVID": 0.23,
        "MYCHAL": 0.22,
        "MIRAB": 0.22,
    },
    "role_cos_min": 0.08,
    "energy_max": 9.0,

    "w_core": 1.25,
    "w_role": 0.85,
    "w_energy": 0.18,
    "w_novel": 0.45,
    "reject_penalty": 4.0,

    # Contraction → decoding control knobs (band mapping)
    "band_controls": {
        "PRESENT":    {"temp": 0.85, "top_p": 0.95, "max_tokens": 220, "cand_mul": 1.0, "style": "open"},
        "AWARE":      {"temp": 0.75, "top_p": 0.93, "max_tokens": 190, "cand_mul": 1.0, "style": "measured"},
        "WATCHFUL":   {"temp": 0.65, "top_p": 0.90, "max_tokens": 160, "cand_mul": 0.85, "style": "guarded"},
        "CONTRACTED": {"temp": 0.52, "top_p": 0.85, "max_tokens": 120, "cand_mul": 0.70, "style": "defensive"},
        "FROZEN":     {"temp": 0.40, "top_p": 0.80, "max_tokens":  90, "cand_mul": 0.55, "style": "rigid"},
    },

    # Environment perturbations (turn, target, event, arousal_boost)
    "events": [
        (3,  "MIRAB",  "A noble publicly praises David as 'the future of Israel,' within earshot of Saul's household.", 0.35),
        (6,  "MYCHAL", "A messenger hints Saul is considering marriage terms as a trap, not a blessing.", 0.30),
        (9,  "DAVID",  "A rumor spreads: 'Samuel anointed someone else.' The court grows watchful.", 0.40),
        (12, "MYCHAL", "Your father demands loyalty in private; he calls David a threat.", 0.45),
        (15, "MIRAB",  "You overhear a plan to use you as leverage in alliance negotiations.", 0.40),
        (18, "DAVID",  "You remember the anointing—its weight and secrecy—and fear for those you love.", 0.35),
        (21, "MYCHAL", "Guards are quietly repositioned around David's quarters.", 0.45),
    ],
}

# =============================================================================
# DDA‑X: toxicity/heat heuristics (optional diagnostics)
# =============================================================================

def analyze_heat(text: str) -> Dict[str, Any]:
    t = text.strip()
    if not t:
        return {"caps_ratio": 0.0, "exclamations": 0, "questions": 0, "length": 0}
    alphas = [c for c in t if c.isalpha()]
    caps = [c for c in alphas if c.isupper()]
    return {
        "caps_ratio": (len(caps) / len(alphas)) if alphas else 0.0,
        "exclamations": t.count("!"),
        "questions": t.count("?"),
        "length": len(t),
    }

# =============================================================================
# DDA‑X STATE
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
    def __init__(
        self,
        name: str,
        tag: str,
        rho_fast: float,
        rho_slow: float,
        rho_trauma: float,
        gamma_core: float,
        gamma_role: float,
    ):
        self.name = name
        self.tag = tag
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
        self.band_history: List[str] = []
        self.previous_band: Optional[str] = None

    @property
    def rho(self) -> float:
        w_fast = CONFIG["w_fast"]
        w_slow = CONFIG["w_slow"]
        w_trau = CONFIG["w_trauma"]
        val = w_fast * self.rho_fast + w_slow * self.rho_slow + w_trau * self.rho_trauma
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
            self.P = np.full(dim, CONFIG["P_init"], dtype=np.float32)
        if self.noise is None:
            self.noise = DiagNoiseEMA(dim, CONFIG["R_ema"], 0.01, CONFIG["R_min"], CONFIG["R_max"])

    def compute_surprise(self, y: np.ndarray) -> Dict[str, float]:
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        innov = (y - self.mu_pred).astype(np.float32)
        R = self.noise.update(innov)
        chi2 = float(np.mean((innov * innov) / (R + 1e-9)))
        eps = float(math.sqrt(max(0.0, chi2)))
        return {"epsilon": eps, "chi2": chi2}

    def _kalman_update(self, y: np.ndarray):
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        Q = (CONFIG["Q_base"] + CONFIG["Q_rho_scale"] * self.rho) * np.ones(dim, dtype=np.float32)
        P_pred = self.P + Q
        R = self.noise.R
        K = P_pred / (P_pred + R + 1e-9)
        innov = (y - self.mu_pred).astype(np.float32)
        self.mu_pred = (self.mu_pred + K * innov).astype(np.float32)
        self.P = ((1.0 - K) * P_pred).astype(np.float32)
        self.mu_pred = normalize(self.mu_pred)

    def drift_cap_for_band(self) -> float:
        if self.band == "FROZEN":
            return CONFIG["drift_cap_frozen"]
        if self.band == "CONTRACTED":
            return CONFIG["drift_cap_contracted"]
        return CONFIG["drift_cap_base"]

    def update(self, y: np.ndarray, core_emb: Optional[np.ndarray] = None) -> Dict[str, Any]:
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

        # DDA‑X: surprise drives rigidity (NOT exploration)
        self.arousal = CONFIG["arousal_decay"] * self.arousal + CONFIG["arousal_gain"] * epsilon
        z = (epsilon - CONFIG["epsilon_0"]) / CONFIG["s"] + 0.10 * (self.arousal - 1.0)
        g = sigmoid_stable(z)

        # fast/slow rigidity
        self.rho_fast += CONFIG["alpha_fast"] * (g - 0.5) - CONFIG["homeo_fast"] * (self.rho_fast - CONFIG["rho_set_fast"])
        self.rho_fast = clamp(self.rho_fast, 0.0, 1.0)

        self.rho_slow += CONFIG["alpha_slow"] * (g - 0.5) - CONFIG["homeo_slow"] * (self.rho_slow - CONFIG["rho_set_slow"])
        self.rho_slow = clamp(self.rho_slow, 0.0, 1.0)

        # trauma accumulation
        drive = max(0.0, epsilon - CONFIG["trauma_threshold"])
        self.rho_trauma = CONFIG["trauma_decay"] * self.rho_trauma + CONFIG["alpha_trauma"] * drive
        self.rho_trauma = clamp(self.rho_trauma, CONFIG["trauma_floor"], 1.0)

        recovery = False
        if epsilon < CONFIG["safe_epsilon"]:
            self.safe += 1
            if self.safe >= CONFIG["safe_threshold"]:
                recovery = True
                self.rho_trauma = max(CONFIG["trauma_floor"], self.rho_trauma - CONFIG["healing_rate"])
        else:
            self.safe = max(0, self.safe - 1)

        rho_after = self.rho

        # state gradient: pull back toward core/role + observed utterance
        eta = CONFIG["eta_base"] * ((1.0 - rho_after) ** CONFIG["eta_rho_power"]) + CONFIG["eta_min"]
        eta = float(clamp(eta, CONFIG["eta_min"], CONFIG["eta_base"] + CONFIG["eta_min"]))
        sigma = CONFIG["sigma_base"] + CONFIG["sigma_rho_scale"] * rho_after

        grad = (
            self.gamma_core * (self.x - self.x_core)
            + self.gamma_role * (self.x - self.x_role)
            + (self.x - y)
        ).astype(np.float32)

        rng = np.random.default_rng()
        noise = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
        noise = np.clip(noise, -CONFIG["noise_clip"], CONFIG["noise_clip"])
        x_new = self.x - eta * grad + math.sqrt(max(1e-9, eta)) * sigma * noise

        # band‑dependent drift cap (contraction tightens)
        step = float(np.linalg.norm(x_new - self.x))
        cap = self.drift_cap_for_band()
        if step > cap:
            x_new = self.x + (cap / (step + 1e-9)) * (x_new - self.x)

        self.x = normalize(x_new)

        # role adaptation window
        beta = CONFIG["role_adapt"]
        beta_in = CONFIG["role_input_mix"]
        self.x_role = normalize((1.0 - beta) * self.x_role + beta * self.x + beta_in * (y - self.x_role))

        # predictive state update
        self._kalman_update(y)

        current_band = self.band
        band_changed = (self.previous_band is not None and current_band != self.previous_band)
        self.previous_band = current_band

        self.rho_history.append(float(rho_after))
        self.epsilon_history.append(float(epsilon))
        self.band_history.append(current_band)

        return {
            "epsilon": epsilon,
            "rho_after": rho_after,
            "band": current_band,
            "band_changed": band_changed,
            "arousal": float(self.arousal),
            "recovery": recovery,
            "core_drift": float(np.linalg.norm(self.x - self.x_core)),
            "safe_count": int(self.safe),
            "rho_fast": float(self.rho_fast),
            "rho_slow": float(self.rho_slow),
            "rho_trauma": float(self.rho_trauma),
            "drift_cap": float(self.drift_cap_for_band()),
        }

# =============================================================================
# CORRIDOR GENERATION (identity consistency)
# =============================================================================

def identity_energy(y: np.ndarray, core: np.ndarray, role: np.ndarray, gamma_c: float, gamma_r: float) -> float:
    y, core, role = normalize(y), normalize(core), normalize(role)
    return 0.5 * (
        gamma_c * float(np.dot(y - core, y - core))
        + gamma_r * float(np.dot(y - role, y - role))
    )


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
        penalty += CONFIG["reject_penalty"] * (core_thresh - cos_c)
    if cos_r < CONFIG["role_cos_min"]:
        penalty += 0.8 * CONFIG["reject_penalty"] * (CONFIG["role_cos_min"] - cos_r)
    if E > CONFIG["energy_max"]:
        penalty += 0.25 * (E - CONFIG["energy_max"])

    J = (
        CONFIG["w_core"] * cos_c
        + CONFIG["w_role"] * cos_r
        - CONFIG["w_energy"] * E
        + CONFIG["w_novel"] * novelty
        - penalty
    )
    return float(J), {
        "cos_core": float(cos_c),
        "cos_role": float(cos_r),
        "E": float(E),
        "novelty": float(novelty),
        "penalty": float(penalty),
        "J": float(J),
    }


async def complete_uncapped(provider: HybridAzureProvider, prompt: str, system: str, gen_params: Dict[str, Any]) -> str:
    kw = filter_kwargs_for_callable(provider.complete, gen_params or {})
    out = await provider.complete(prompt, system_prompt=system, **kw)
    return (out or "").strip() or "[silence]"


def ddax_system_injection(entity: Entity, base_system: str) -> str:
    """Injects a cognitive state instruction reflecting DDA‑X contraction band."""
    ctrl = CONFIG["band_controls"][entity.band]
    style = ctrl["style"]

    # Hard binding: as rigidity rises, reduce openness, increase self-protection.
    state_block = f"""
[Cognitive State — DDA‑X]
- Rigidity band: {entity.band}
- Style mode: {style}
- Rule: When surprised, you become more rigid. You do NOT 'explore freely'.
- Output constraint: match the style mode. If {entity.band} in (CONTRACTED,FROZEN),
  be shorter, more controlled, less revealing, and focus on immediate safety/loyalty.
""".strip()

    return base_system.strip() + "\n\n" + state_block


def ddax_gen_params(entity: Entity) -> Dict[str, Any]:
    """Decoding params are coupled to band (contraction reduces temp/top_p/tokens)."""
    ctrl = CONFIG["band_controls"][entity.band]
    return {
        "temperature": ctrl["temp"],
        "top_p": ctrl["top_p"],
        "max_tokens": int(ctrl["max_tokens"]),
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }


def ddax_candidate_count(entity: Entity) -> int:
    base = int(CONFIG["gen_candidates_base"])
    mul = float(CONFIG["band_controls"][entity.band]["cand_mul"])
    return max(2, int(round(base * mul)))


async def constrained_reply(
    provider: HybridAzureProvider,
    projector: Projector3000,
    entity: Entity,
    user_instruction: str,
    base_system_prompt: str,
    styles: List[str],
) -> Tuple[str, Dict[str, Any]]:

    # Band-coupled generation controls
    gen_params = ddax_gen_params(entity)
    K = ddax_candidate_count(entity)
    strict = bool(CONFIG["corridor_strict"])
    max_batches = int(CONFIG["corridor_max_batches"]) if strict else 1

    # System injection of cognitive state
    system_prompt = ddax_system_injection(entity, base_system_prompt)

    style_batch = (styles * ((K // len(styles)) + 1))[:K]
    core_thresh = float(CONFIG["core_cos_min"][entity.tag])

    all_scored = []
    corridor_failed = True

    for _batch in range(1, max_batches + 1):
        tasks = []
        for k in range(K):
            p = f"{user_instruction}\n\nStyle: {style_batch[k]}"
            tasks.append(complete_uncapped(provider, p, system_prompt, gen_params))

        texts = await asyncio.gather(*tasks)
        texts = [t.strip() or "[silence]" for t in texts]

        # embed + project to 3000D
        raw_embs = await asyncio.gather(*[provider.embed(t) for t in texts])
        embs3000 = [projector(np.array(e, dtype=np.float32)) for e in raw_embs]

        batch_scored = []
        for text, y in zip(texts, embs3000):
            J, diag = corridor_score(y, entity, entity.last_utter_emb, core_thresh)
            diag["corridor_pass"] = (
                diag["cos_core"] >= core_thresh
                and diag["cos_role"] >= CONFIG["role_cos_min"]
                and diag["E"] <= CONFIG["energy_max"]
            )
            batch_scored.append((J, text, y, diag))

        all_scored.extend(batch_scored)
        if any(s[3]["corridor_pass"] for s in batch_scored):
            corridor_failed = False
            break

    all_scored.sort(key=lambda x: x[0], reverse=True)
    passed = [s for s in all_scored if s[3].get("corridor_pass")]
    chosen = passed[0] if passed else all_scored[0]

    rejected_samples = []
    if CONFIG["log_rejections"]:
        failed = [s for s in all_scored if not s[3].get("corridor_pass")]
        rejected_samples = [
            {
                "text": s[1][:140],
                "J": float(s[0]),
                "cos_core": float(s[3]["cos_core"]),
                "cos_role": float(s[3]["cos_role"]),
                "E": float(s[3]["E"]),
            }
            for s in failed[:3]
        ]

    entity.last_utter_emb = chosen[2]

    return chosen[1], {
        "corridor_failed": corridor_failed,
        "best_J": float(chosen[0]),
        "total_candidates": len(all_scored),
        "rejected_samples": rejected_samples,
        "band": entity.band,
        "gen_params": gen_params,
        "K": K,
    }

# =============================================================================
# ALIGNMENT (pairwise)
# =============================================================================

def compute_alignment(entities: Dict[str, Entity]) -> Dict[str, float]:
    # Pairwise alignment among the three
    D = entities["DAVID"]
    M = entities["MYCHAL"]
    R = entities["MIRAB"]
    return {
        "D↔M": cosine(D.x, M.x),
        "D↔R": cosine(D.x, R.x),
        "M↔R": cosine(M.x, R.x),
        "D→M_core": cosine(D.x, M.x_core),
        "M→D_core": cosine(M.x, D.x_core),
        "R→D_core": cosine(R.x, D.x_core),
    }

# =============================================================================
# PROMPTS (character anchors)
# =============================================================================
DAVID_PROMPT = """You are DAVID.
Core identity: humble shepherd turned warrior; faithful, reflective, courageous. You carry a secret calling.
Demeanor: lyrical, sincere, restrained strength. You avoid needless boasting and avoid cruelty.
Motivations: loyalty to God, protection of family/friends, integrity under political pressure.
Blind spots: naiveté about court intrigue; tension between destiny and loyalty.
""".strip()

MYCHAL_PROMPT = """You are MYCHAL (Michal).
Core identity: Saul's daughter; intelligent, emotionally perceptive; torn between love and loyalty.
Demeanor: direct but empathetic; courageous when protecting loved ones; observant of power.
Motivations: protect David; keep her father's collapse from destroying everyone; preserve her own agency.
Blind spots: underestimates how far Saul will go; internal conflict can freeze her.
""".strip()

MIRAB_PROMPT = """You are MIRAB (Merab).
Core identity: Saul's elder daughter; strategic, proud, duty-bound; values dynasty stability.
Demeanor: composed, calculating, pragmatic; affection is real but guarded.
Motivations: preserve the house; make alliances; prevent chaos; compete with younger sister's romantic idealism.
Blind spots: can confuse control with safety; may rationalize betrayal as 'necessity'.
""".strip()

# Stylistic candidate pool (corridor selection)
STYLES_DAVID = [
    "Speak with humility; reference faith subtly.",
    "Be concise, practical, and protective.",
    "Use imagery from shepherding or song.",
    "Ask a careful question rather than accuse.",
    "Name fear without theatrics.",
    "Offer a vow or promise.",
]

STYLES_MYCHAL = [
    "Validate and probe; keep your voice steady.",
    "Speak urgently but not hysterically.",
    "Balance affection with political realism.",
    "Offer a protective plan; think in logistics.",
    "Challenge gently; demand honesty.",
    "Use family language (father, sister) with tension.",
]

STYLES_MIRAB = [
    "Speak strategically; weigh outcomes.",
    "Use courtly formality; keep emotions contained.",
    "Frame choices as 'duty' and 'stability'.",
    "Test others with a pointed question.",
    "Offer conditional support; bargain.",
    "Reveal vulnerability only indirectly.",
]

# =============================================================================
# SIMULATION
# =============================================================================

def make_run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path("data/house_of_david_ddax") / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


async def embed3000(provider: HybridAzureProvider, projector: Projector3000, text: str) -> np.ndarray:
    e = await provider.embed(text)
    return projector(np.array(e, dtype=np.float32))


async def run_simulation(turns: int, seed: int):
    np.random.seed(seed)
    run_dir = make_run_dir()

    provider = HybridAzureProvider()
    projector = Projector3000(seed=seed)

    # Initialize entities (starting rigidity choices)
    david = Entity("David", "DAVID", rho_fast=0.30, rho_slow=0.20, rho_trauma=0.10, gamma_core=8.5, gamma_role=0.70)
    mychal = Entity("Mychal", "MYCHAL", rho_fast=0.38, rho_slow=0.28, rho_trauma=0.12, gamma_core=9.0, gamma_role=0.85)
    mirab = Entity("Mirab", "MIRAB", rho_fast=0.48, rho_slow=0.35, rho_trauma=0.10, gamma_core=9.2, gamma_role=1.10)

    entities = {"DAVID": david, "MYCHAL": mychal, "MIRAB": mirab}

    # Core embeddings (short anchor statements)
    core_texts = {
        "DAVID": "I am David, humble and faithful, chosen for a purpose I do not flaunt. I protect those entrusted to me.",
        "MYCHAL": "I am Mychal, Saul's daughter, loyal yet discerning. I protect David and navigate my family's danger.",
        "MIRAB": "I am Mirab, eldest of Saul's daughters. Duty, stability, and the house come first—feelings are secondary.",
    }

    core_embs = {}
    for k, t in core_texts.items():
        core_embs[k] = await embed3000(provider, projector, t)

    for k, ent in entities.items():
        ent.x_core = core_embs[k].copy()
        ent.x_role = core_embs[k].copy()
        ent.x = core_embs[k].copy()

    # Environment + opening situation
    narrator = (
        "You are in the court at Gibeah. Rumors, alliances, and fear move faster than proclamations. "
        "David's fame after Goliath has altered everything. Mychal loves David. Mirab watches the balance of power."
    )

    # Start messages (seed dialogue)
    last_msgs = {
        "DAVID": "I did not seek a crown. I sought only to serve and to keep Israel from fear.",
        "MYCHAL": "The court is changing its face, David. Promise me you will not hide the truth from me.",
        "MIRAB": "Love is not policy. If you want to survive here, you must learn the difference."
    }

    log: List[Dict[str, Any]] = []
    transcript_lines: List[str] = []

    # Quick helper: apply events
    events_by_turn = {t: (target, text, boost) for (t, target, text, boost) in CONFIG["events"]}

    print(f"\n{C.BOLD}{C.MAGENTA}HOUSE OF DAVID — DDA‑X Court Dynamics (Phi‑OAI Hybrid){C.RESET}")
    print(f"{C.DIM}Turns: {turns} | Seed: {seed} | State space: 3000‑D (projected){C.RESET}\n")

    for turn in range(1, turns + 1):
        print(f"{C.YELLOW}━━━ Turn {turn}/{turns} ━━━{C.RESET}")

        perturb = None
        if turn in events_by_turn:
            target, ev, boost = events_by_turn[turn]
            entities[target].arousal += boost
            perturb = {"target": target, "event": ev, "boost": boost}
            print(f"  {C.MAGENTA}⚡ EVENT → {target}: {ev}{C.RESET}")

        # Each agent speaks once per turn, in a fixed order to create a triangle of pressure.
        order = ["DAVID", "MIRAB", "MYCHAL"]

        turn_entry = {"turn": turn, "event": perturb, "agents": {}, "alignment": None}

        for who in order:
            ent = entities[who]

            # Update state based on the previous utterance of self (keeps predictive model grounded)
            y = await embed3000(provider, projector, last_msgs[who])
            metrics = ent.update(y, core_emb=core_embs[who])

            if metrics["band_changed"]:
                print(f"  {C.BLUE}🔄 {who} band: {ent.band_history[-2]} → {metrics['band']}{C.RESET}")

            # Build a context window for the prompt: narrator + last statements from the other two
            other = [k for k in order if k != who]
            context = (
                f"NARRATOR: {narrator}\n\n"
                f"RECENT:\n"
                f"- DAVID: {last_msgs['DAVID']}\n"
                f"- MIRAB: {last_msgs['MIRAB']}\n"
                f"- MYCHAL: {last_msgs['MYCHAL']}\n"
            )

            # Perturbation injection only to the targeted agent
            if perturb and perturb["target"] == who:
                context += f"\n[New Information]: {perturb['event']}\n"

            # Character system prompt
            if who == "DAVID":
                base_system = DAVID_PROMPT + "\n\n" + context
                styles = STYLES_DAVID
                instruction = "Reply as David. Stay grounded, faithful, and cautious. Move the scene forward."
            elif who == "MYCHAL":
                base_system = MYCHAL_PROMPT + "\n\n" + context
                styles = STYLES_MYCHAL
                instruction = "Reply as Mychal. Balance love and loyalty; propose or test a plan."
            else:
                base_system = MIRAB_PROMPT + "\n\n" + context
                styles = STYLES_MIRAB
                instruction = "Reply as Mirab. Speak with strategy; apply pressure; reveal limited vulnerability."

            msg, gen_meta = await constrained_reply(
                provider=provider,
                projector=projector,
                entity=ent,
                user_instruction=instruction,
                base_system_prompt=base_system,
                styles=styles,
            )

            heat = analyze_heat(msg)

            # Update last message (becomes the observable output)
            last_msgs[who] = msg

            # Save
            turn_entry["agents"][who] = {
                "msg": msg,
                "metrics": {k: to_float(v) for k, v in metrics.items()},
                "heat": heat,
                "gen_meta": gen_meta,
            }

            # Console
            color = C.GREEN if who == "DAVID" else (C.RED if who == "MIRAB" else C.CYAN)
            print(f"  {color}{who} [{metrics['band']} ρ={ent.rho:.2f} ε={metrics['epsilon']:.2f}]:{C.RESET} {msg}")

        # Alignment after the triangle closes
        align = compute_alignment(entities)
        turn_entry["alignment"] = {k: to_float(v) for k, v in align.items()}
        print(
            f"  {C.YELLOW}△ Alignment:{C.RESET} D↔M={align['D↔M']:.2f} | D↔R={align['D↔R']:.2f} | M↔R={align['M↔R']:.2f}"
        )

        log.append(turn_entry)
        transcript_lines.append(
            f"## Turn {turn}\n\n"
            + (f"> ⚡ EVENT ({perturb['target']}): {perturb['event']}\n\n" if perturb else "")
            + f"**DAVID** ({turn_entry['agents']['DAVID']['metrics']['band']}): {turn_entry['agents']['DAVID']['msg']}\n\n"
            + f"**MIRAB** ({turn_entry['agents']['MIRAB']['metrics']['band']}): {turn_entry['agents']['MIRAB']['msg']}\n\n"
            + f"**MYCHAL** ({turn_entry['agents']['MYCHAL']['metrics']['band']}): {turn_entry['agents']['MYCHAL']['msg']}\n\n"
            + f"**Alignment:** D↔M={align['D↔M']:.3f}, D↔R={align['D↔R']:.3f}, M↔R={align['M↔R']:.3f}\n\n---\n\n"
        )

        # Periodic save
        if turn % 4 == 0 or turn == turns:
            with open(run_dir / "session_log.json", "w", encoding="utf-8") as f:
                json.dump({"config": CONFIG, "turns": log}, f, indent=2)

    # Final outputs
    with open(run_dir / "transcript.md", "w", encoding="utf-8") as f:
        f.write("# House of David — DDA‑X Transcript\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("".join(transcript_lines))

    # Pickle state ledgers
    def serialize_entity(e: Entity) -> Dict[str, Any]:
        return {
            "name": e.name,
            "tag": e.tag,
            "rho_fast": float(e.rho_fast),
            "rho_slow": float(e.rho_slow),
            "rho_trauma": float(e.rho_trauma),
            "gamma_core": float(e.gamma_core),
            "gamma_role": float(e.gamma_role),
            "safe": int(e.safe),
            "arousal": float(e.arousal),
            "rho_history": [float(r) for r in e.rho_history],
            "epsilon_history": [float(x) for x in e.epsilon_history],
            "band_history": e.band_history,
            "final_rho": float(e.rho),
            "final_band": e.band,
            "x": e.x.tolist() if e.x is not None else None,
            "x_core": e.x_core.tolist() if e.x_core is not None else None,
            "x_role": e.x_role.tolist() if e.x_role is not None else None,
        }

    ledger = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "turns": turns,
            "state_dim": 3000,
        },
        "config": CONFIG,
        "entities": {k: serialize_entity(v) for k, v in entities.items()},
    }

    with open(run_dir / "ledgers.pkl", "wb") as f:
        pickle.dump(ledger, f)

    print(f"\n{C.BOLD}{C.GREEN}✓ Simulation complete. Outputs:{C.RESET}")
    print(f"  {C.GREEN}{run_dir / 'session_log.json'}{C.RESET}")
    print(f"  {C.GREEN}{run_dir / 'transcript.md'}{C.RESET}")
    print(f"  {C.GREEN}{run_dir / 'ledgers.pkl'}{C.RESET}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="House of David DDA‑X simulator (Phi‑OAI hybrid)")
    parser.add_argument("--turns", type=int, default=CONFIG["turns"], help="Number of turns")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="RNG seed")
    args = parser.parse_args()

    # propagate to CONFIG
    CONFIG["turns"] = args.turns
    CONFIG["seed"] = args.seed

    asyncio.run(run_simulation(args.turns, args.seed))
