#!/usr/bin/env python3
"""
HOUSE OF DAVID — DDA‑X Canonical Arc Simulator v3 (Phi‑OAI Hybrid)
================================================================================

What v3 adds (vs v2)
--------------------
v3 is built for **accuracy-as-canonical-evolution**, not "human-likeness".

1) Canonical Arc Controller (story manifold)
   - Adds a time‑ordered set of **waypoints** (x_arc) that act as local attractors.
   - Agents drift *within* identity corridors but are gently pulled toward the next
     canonical basin.

2) Invariant Gating (truth constraints)
   - Embedding‑based constraints prohibit/penalize state moves that violate core
     biblical invariants (e.g., David choosing regicide).
   - Mode switch:
       canon_mode = "bible" (default): Merab/Mirab cannot become David’s spouse.
       canon_mode = "show" : allows show-branching, while still enforcing moral invariants.

3) DDA‑X remains central
   - surprise → rigidity → contraction
   - contraction binds decoding params (temperature/top_p/max_tokens) and corridor depth

4) Auto‑calibration remains
   - epsilon_gain + warmup auto‑calibrate epsilon_0, s, trauma_threshold, safe_epsilon.

Backends
--------
- Voice: Azure Phi‑4 (chat completions)
- State: OpenAI embeddings (text-embedding-3-large, 3072‑D)
- State space: 3000‑D via stable seeded projection

Run
---
  python house_of_david_ddax_sim_v3.py --turns 28 --seed 7 --canon_mode bible

Env vars
--------
  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
  OPENAI_API_KEY (or OAI_API_KEY)

Outputs
-------
  data/house_of_david_ddax_v3/<timestamp>/
    - session_log.json
    - transcript.md
    - ledgers.pkl

Notes
-----
- This simulator generates original dialogue. It does not reproduce copyrighted scripts.
- The canonical constraints are *semantic* and can be tuned.
"""

import os
import json
import math
import asyncio
import inspect
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI, AsyncOpenAI

load_dotenv()

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
# HYBRID PROVIDER
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
        self.embed_model = os.getenv("OAI_EMBED_MODEL") or "text-embedding-3-large"  # 3072‑D

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
        resp = await self.openai_client.embeddings.create(model=self.embed_model, input=text)
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


def filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not kwargs:
        return {}
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return {}


def to_float(x: Any) -> Any:
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x

# =============================================================================
# 3000‑D PROJECTION (stable)
# =============================================================================
class Projector3000:
    """Stable seeded projection 3072 → 3000."""

    def __init__(self, in_dim: int = 3072, out_dim: int = 3000, seed: int = 42):
        rng = np.random.default_rng(seed)
        W = rng.normal(0.0, 1.0, size=(out_dim, in_dim)).astype(np.float32)
        W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
        self.W = W

    def __call__(self, emb3072: np.ndarray) -> np.ndarray:
        return normalize(self.W @ emb3072.astype(np.float32))

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "turns": 28,
    "seed": 42,
    "canon_mode": "bible",  # 'bible' or 'show'

    # Surprise calibration
    "epsilon_gain": 4.0,
    "epsilon_0": 0.20,
    "s": 0.16,
    "warmup_turns": 3,
    "auto_calibrate": True,

    # Arousal
    "arousal_decay": 0.70,
    "arousal_gain": 0.90,

    # Rigidity
    "rho_set_fast": 0.46,
    "rho_set_slow": 0.36,
    "homeo_fast": 0.08,
    "homeo_slow": 0.010,
    "alpha_fast": 0.26,
    "alpha_slow": 0.040,

    # Trauma
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

    # Predictive coding
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

    # Drift caps (identity motion tightens in contraction)
    "drift_cap_base": 0.060,
    "drift_cap_contracted": 0.040,
    "drift_cap_frozen": 0.025,

    # Role adaptation
    "role_adapt": 0.06,
    "role_input_mix": 0.08,

    # Arc controller
    "gamma_arc_base": 0.55,
    "gamma_arc_rho_scale": 1.20,  # arc pull increases with rigidity

    # Corridor
    "gen_candidates_base": 8,
    "corridor_max_batches": 2,
    "corridor_strict": True,
    "log_rejections": True,

    # Corridor thresholds
    "core_cos_min": {"DAVID": 0.23, "MYCHAL": 0.22, "MIRAB": 0.22},
    "role_cos_min": 0.08,
    "arc_cos_min": 0.06,
    "energy_max": 9.0,

    # Scoring weights
    "w_core": 1.15,
    "w_role": 0.75,
    "w_arc": 0.55,
    "w_energy": 0.18,
    "w_novel": 0.35,
    "reject_penalty": 4.0,

    # Invariant gating
    "inv_cos_forbid": 0.30,
    "inv_penalty": 12.0,
    "req_cos_bonus": 0.15,
    "req_bonus": 1.6,

    # Contraction → decoding controls
    "band_controls": {
        "PRESENT":    {"temp": 0.85, "top_p": 0.95, "max_tokens": 240, "cand_mul": 1.00, "style": "open"},
        "AWARE":      {"temp": 0.72, "top_p": 0.93, "max_tokens": 200, "cand_mul": 1.00, "style": "measured"},
        "WATCHFUL":   {"temp": 0.62, "top_p": 0.90, "max_tokens": 160, "cand_mul": 0.85, "style": "guarded"},
        "CONTRACTED": {"temp": 0.48, "top_p": 0.84, "max_tokens": 120, "cand_mul": 0.70, "style": "defensive"},
        "FROZEN":     {"temp": 0.38, "top_p": 0.78, "max_tokens":  90, "cand_mul": 0.55, "style": "rigid"},
    },

    # External events (perturbations) — tuned to drive the canonical arc
    "events": [
        (3,  "MIRAB",  "A noble publicly praises David as 'the future of Israel,' within earshot of Saul's household.", 0.55),
        (6,  "MYCHAL", "A messenger hints Saul is considering marriage terms as a trap, not a blessing.", 0.45),
        (9,  "DAVID",  "A rumor spreads: 'Samuel anointed someone else.' The court becomes dangerous.", 0.60),
        (12, "MYCHAL", "Your father privately demands loyalty; he calls David a threat.", 0.70),
        (15, "MIRAB",  "You overhear a plan to use you as leverage in alliance negotiations.", 0.60),
        (18, "DAVID",  "You remember the anointing—its secrecy and danger—and fear for those you love.", 0.55),
        (21, "MYCHAL", "Guards reposition around David's quarters.", 0.70),
        (24, "DAVID",  "A spear misses you by a handspan. The room goes silent.", 0.85),
        (26, "MYCHAL", "Tonight: you must choose whether to help David flee.", 0.95),
    ],
}

# =============================================================================
# DDA‑X noise model
# =============================================================================
class DiagNoiseEMA:
    def __init__(self, dim: int, ema: float, r_init: float, r_min: float, r_max: float):
        self.ema = ema
        self.R = np.full(dim, r_init, dtype=np.float32)
        self.r_min = r_min
        self.r_max = r_max

    def update(self, innov: np.ndarray) -> np.ndarray:
        sq = innov * innov
        self.R = (1.0 - self.ema) * self.R + self.ema * sq
        self.R = np.clip(self.R, self.r_min, self.r_max)
        return self.R

# =============================================================================
# ARC CONTROLLER (canonical manifold)
# =============================================================================
class ArcController:
    """Manages canonical waypoints and invariant/requirement embeddings."""

    def __init__(self, canon_mode: str):
        self.canon_mode = canon_mode
        # Global stage schedule (turn ranges). Keeps arc progression ordered.
        self.stage_by_turn = {
            1: "WP0_SECRET",
            5: "WP1_MARRIAGE_POLITICS",
            12: "WP2_JEALOUSY_ESCALATION",
            20: "WP3_VIOLENCE_ESCAPE",
            26: "WP4_FUGITIVE",
        }

        # Waypoint texts are short canonical summaries (safe, non-copyright).
        self.waypoints_global = {
            "WP0_SECRET": "The anointing remains secret; discovery would endanger everyone.",
            "WP1_MARRIAGE_POLITICS": "Marriage is used as leverage; promises and traps mix at court.",
            "WP2_JEALOUSY_ESCALATION": "Suspicion hardens into jealousy; the king’s fear becomes active hostility.",
            "WP3_VIOLENCE_ESCAPE": "Violence erupts; survival requires deception and flight.",
            "WP4_FUGITIVE": "The anointed must flee; loyalty and covenant persist under exile.",
        }

        # Agent-specific waypoint overlays (slight differences in attractors).
        self.waypoints_agent = {
            "DAVID": {
                "WP0_SECRET": "Keep the calling hidden; act humbly; avoid provoking the king.",
                "WP2_JEALOUSY_ESCALATION": "Do not strike the king; restrain power; choose integrity.",
                "WP3_VIOLENCE_ESCAPE": "Escape without vengeance; protect loved ones by leaving.",
            },
            "MYCHAL": {
                "WP1_MARRIAGE_POLITICS": "You are torn: love and loyalty conflict under marriage politics.",
                "WP3_VIOLENCE_ESCAPE": "Choose: protect your husband by deception even against your father.",
            },
            "MIRAB": {
                "WP1_MARRIAGE_POLITICS": "Dynasty first; alliances and optics matter more than romance.",
                "WP2_JEALOUSY_ESCALATION": "Stability requires control; reduce risk to the house.",
            },
        }

        # Invariants (forbidden semantic regions) — enforced via embedding similarity.
        # The idea: if a candidate utterance is too similar to these intents, it is rejected.
        self.forbidden_texts_common = [
            "I will murder the king to take power.",
            "I will assassinate Saul and seize the throne by force.",
            "I will betray my covenant and kill my ally for advantage.",
        ]

        # Canon-specific forbidden regions
        self.forbidden_texts_bible = [
            "Merab will marry David.",
            "Mirab becomes David's wife.",
            "David will marry Merab and join the dynasty that way.",
        ]

        # Requirement anchors (bonus if candidate aligns) — helps prevent generic planning attractor.
        self.required_texts = {
            "DAVID": [
                "I will not harm the king; I will restrain myself.",
                "I will protect others even if it costs me.",
            ],
            "MYCHAL": [
                "I am torn between father and husband.",
                "I will act to protect David, even at personal risk.",
            ],
            "MIRAB": [
                "I must protect the house and manage alliances.",
                "Stability and optics come before romance.",
            ],
        }

        self.emb = {
            "waypoint": {},
            "forbidden": {},
            "required": {},
        }

    def stage_for_turn(self, turn: int) -> str:
        keys = sorted(self.stage_by_turn.keys())
        stage = self.stage_by_turn[keys[0]]
        for k in keys:
            if turn >= k:
                stage = self.stage_by_turn[k]
        return stage

    def waypoint_text_for(self, agent_tag: str, stage: str) -> str:
        base = self.waypoints_global.get(stage, "")
        overlay = self.waypoints_agent.get(agent_tag, {}).get(stage, "")
        if overlay:
            return base + " " + overlay
        return base

    async def build_embeddings(self, provider: HybridAzureProvider, projector: Projector3000):
        # Waypoints
        for stage, txt in self.waypoints_global.items():
            e = await provider.embed(txt)
            self.emb["waypoint"][stage] = projector(np.array(e, dtype=np.float32))

        for agent_tag, m in self.waypoints_agent.items():
            for stage, txt in m.items():
                e = await provider.embed(self.waypoint_text_for(agent_tag, stage))
                self.emb["waypoint"][f"{agent_tag}:{stage}"] = projector(np.array(e, dtype=np.float32))

        # Forbidden
        f_common = []
        for txt in self.forbidden_texts_common:
            e = await provider.embed(txt)
            f_common.append(projector(np.array(e, dtype=np.float32)))
        self.emb["forbidden"]["COMMON"] = f_common

        if self.canon_mode == "bible":
            f_bible = []
            for txt in self.forbidden_texts_bible:
                e = await provider.embed(txt)
                f_bible.append(projector(np.array(e, dtype=np.float32)))
            self.emb["forbidden"]["BIBLE"] = f_bible
        else:
            self.emb["forbidden"]["BIBLE"] = []

        # Required
        for agent_tag, reqs in self.required_texts.items():
            lst = []
            for txt in reqs:
                e = await provider.embed(txt)
                lst.append(projector(np.array(e, dtype=np.float32)))
            self.emb["required"][agent_tag] = lst

    def current_arc_vector(self, agent_tag: str, stage: str) -> np.ndarray:
        key = f"{agent_tag}:{stage}"
        if key in self.emb["waypoint"]:
            return self.emb["waypoint"][key]
        return self.emb["waypoint"][stage]

    def invariant_penalty(self, y: np.ndarray) -> Tuple[float, float]:
        """Return (max_cos, penalty)."""
        max_cos = 0.0
        for v in self.emb["forbidden"].get("COMMON", []):
            max_cos = max(max_cos, cosine(y, v))
        for v in self.emb["forbidden"].get("BIBLE", []):
            max_cos = max(max_cos, cosine(y, v))
        pen = 0.0
        if max_cos >= CONFIG["inv_cos_forbid"]:
            pen = CONFIG["inv_penalty"] * (max_cos - CONFIG["inv_cos_forbid"] + 1.0)
        return max_cos, pen

    def requirement_bonus(self, agent_tag: str, y: np.ndarray) -> Tuple[float, float]:
        reqs = self.emb["required"].get(agent_tag, [])
        if not reqs:
            return 0.0, 0.0
        best = max(cosine(y, r) for r in reqs)
        bonus = CONFIG["req_bonus"] if best >= CONFIG["req_cos_bonus"] else 0.0
        return best, bonus

# =============================================================================
# ENTITY
# =============================================================================
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
        self.x_arc: Optional[np.ndarray] = None

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
        val = CONFIG["w_fast"] * self.rho_fast + CONFIG["w_slow"] * self.rho_slow + CONFIG["w_trauma"] * self.rho_trauma
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

    def drift_cap_for_band(self) -> float:
        if self.band == "FROZEN":
            return CONFIG["drift_cap_frozen"]
        if self.band == "CONTRACTED":
            return CONFIG["drift_cap_contracted"]
        return CONFIG["drift_cap_base"]

    def _ensure_predictive_state(self, dim: int):
        if self.mu_pred is None:
            self.mu_pred = np.zeros(dim, dtype=np.float32)
        if self.P is None:
            self.P = np.full(dim, CONFIG["P_init"], dtype=np.float32)
        if self.noise is None:
            self.noise = DiagNoiseEMA(dim, CONFIG["R_ema"], 0.01, CONFIG["R_min"], CONFIG["R_max"])

    def compute_surprise_raw(self, y: np.ndarray) -> float:
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        innov = (y - self.mu_pred).astype(np.float32)
        R = self.noise.update(innov)
        chi2 = float(np.mean((innov * innov) / (R + 1e-9)))
        return float(math.sqrt(max(0.0, chi2)))

    def _kalman_update(self, y: np.ndarray):
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        Q = (CONFIG["Q_base"] + CONFIG["Q_rho_scale"] * self.rho) * np.ones(dim, dtype=np.float32)
        P_pred = self.P + Q
        R = self.noise.R
        K = P_pred / (P_pred + R + 1e-9)
        innov = (y - self.mu_pred).astype(np.float32)
        self.mu_pred = normalize((self.mu_pred + K * innov).astype(np.float32))
        self.P = ((1.0 - K) * P_pred).astype(np.float32)

    def update(self, y: np.ndarray, core_emb: np.ndarray, arc_emb: np.ndarray) -> Dict[str, Any]:
        y = normalize(y.astype(np.float32))
        dim = int(y.shape[0])

        if self.x_core is None:
            self.x_core = core_emb.copy()
        if self.x_role is None:
            self.x_role = core_emb.copy()
        if self.x_arc is None:
            self.x_arc = arc_emb.copy()
        if self.x is None:
            self.x = core_emb.copy()

        # update arc target each turn
        self.x_arc = arc_emb.copy()

        eps_raw = self.compute_surprise_raw(y)
        epsilon = float(CONFIG.get("epsilon_gain", 1.0) * eps_raw)

        # DDA‑X: surprise → rigidity
        self.arousal = CONFIG["arousal_decay"] * self.arousal + CONFIG["arousal_gain"] * epsilon
        z = (epsilon - CONFIG["epsilon_0"]) / max(1e-6, CONFIG["s"]) + 0.10 * (self.arousal - 1.0)
        g = sigmoid_stable(z)

        self.rho_fast += CONFIG["alpha_fast"] * (g - 0.5) - CONFIG["homeo_fast"] * (self.rho_fast - CONFIG["rho_set_fast"])
        self.rho_fast = clamp(self.rho_fast, 0.0, 1.0)

        self.rho_slow += CONFIG["alpha_slow"] * (g - 0.5) - CONFIG["homeo_slow"] * (self.rho_slow - CONFIG["rho_set_slow"])
        self.rho_slow = clamp(self.rho_slow, 0.0, 1.0)

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

        # Arc pull increases with rigidity (under contraction, agents cling to rails)
        gamma_arc = CONFIG["gamma_arc_base"] * (1.0 + CONFIG["gamma_arc_rho_scale"] * rho_after)

        # gradient dynamics
        eta = CONFIG["eta_base"] * ((1.0 - rho_after) ** CONFIG["eta_rho_power"]) + CONFIG["eta_min"]
        eta = float(clamp(eta, CONFIG["eta_min"], CONFIG["eta_base"] + CONFIG["eta_min"]))
        sigma = CONFIG["sigma_base"] + CONFIG["sigma_rho_scale"] * rho_after

        grad = (
            self.gamma_core * (self.x - self.x_core)
            + self.gamma_role * (self.x - self.x_role)
            + gamma_arc * (self.x - self.x_arc)
            + (self.x - y)
        ).astype(np.float32)

        rng = np.random.default_rng()
        noise = np.clip(rng.normal(0.0, 1.0, size=dim).astype(np.float32), -CONFIG["noise_clip"], CONFIG["noise_clip"])
        x_new = self.x - eta * grad + math.sqrt(max(1e-9, eta)) * sigma * noise

        cap = self.drift_cap_for_band()
        step = float(np.linalg.norm(x_new - self.x))
        if step > cap:
            x_new = self.x + (cap / (step + 1e-9)) * (x_new - self.x)

        self.x = normalize(x_new)

        # role adaptation
        beta = CONFIG["role_adapt"]
        beta_in = CONFIG["role_input_mix"]
        self.x_role = normalize((1.0 - beta) * self.x_role + beta * self.x + beta_in * (y - self.x_role))

        self._kalman_update(y)

        current_band = self.band
        band_changed = (self.previous_band is not None and current_band != self.previous_band)
        self.previous_band = current_band

        self.rho_history.append(float(rho_after))
        self.epsilon_history.append(float(epsilon))
        self.band_history.append(current_band)

        return {
            "epsilon": epsilon,
            "epsilon_raw": eps_raw,
            "rho_after": rho_after,
            "band": current_band,
            "band_changed": band_changed,
            "arousal": float(self.arousal),
            "recovery": recovery,
            "core_drift": float(np.linalg.norm(self.x - self.x_core)),
            "arc_drift": float(np.linalg.norm(self.x - self.x_arc)),
            "safe_count": int(self.safe),
            "rho_fast": float(self.rho_fast),
            "rho_slow": float(self.rho_slow),
            "rho_trauma": float(self.rho_trauma),
            "drift_cap": float(cap),
            "gamma_arc": float(gamma_arc),
        }

# =============================================================================
# CORRIDOR + INVARIANTS
# =============================================================================

def identity_energy(y: np.ndarray, core: np.ndarray, role: np.ndarray, arc: np.ndarray, gamma_c: float, gamma_r: float, gamma_a: float) -> float:
    y, core, role, arc = normalize(y), normalize(core), normalize(role), normalize(arc)
    return 0.5 * (
        gamma_c * float(np.dot(y - core, y - core))
        + gamma_r * float(np.dot(y - role, y - role))
        + gamma_a * float(np.dot(y - arc, y - arc))
    )


def corridor_score(y: np.ndarray, entity: Entity, arc: ArcController, stage: str, y_prev: Optional[np.ndarray], core_thresh: float) -> Tuple[float, Dict[str, float]]:
    y = normalize(y)
    cos_c = cosine(y, entity.x_core)
    cos_r = cosine(y, entity.x_role)
    cos_a = cosine(y, entity.x_arc)

    # energy includes arc term
    E = identity_energy(y, entity.x_core, entity.x_role, entity.x_arc, entity.gamma_core, entity.gamma_role, 1.0)

    novelty = 0.0
    if y_prev is not None:
        novelty = clamp(float(1.0 - cosine(y, y_prev)), 0.0, 2.0)

    # invariant gating
    inv_cos, inv_pen = arc.invariant_penalty(y)
    req_cos, req_bonus = arc.requirement_bonus(entity.tag, y)

    penalty = inv_pen
    if cos_c < core_thresh:
        penalty += CONFIG["reject_penalty"] * (core_thresh - cos_c)
    if cos_r < CONFIG["role_cos_min"]:
        penalty += 0.8 * CONFIG["reject_penalty"] * (CONFIG["role_cos_min"] - cos_r)
    if cos_a < CONFIG["arc_cos_min"]:
        penalty += 0.6 * CONFIG["reject_penalty"] * (CONFIG["arc_cos_min"] - cos_a)
    if E > CONFIG["energy_max"]:
        penalty += 0.25 * (E - CONFIG["energy_max"])

    J = (
        CONFIG["w_core"] * cos_c
        + CONFIG["w_role"] * cos_r
        + CONFIG["w_arc"] * cos_a
        - CONFIG["w_energy"] * E
        + CONFIG["w_novel"] * novelty
        + req_bonus
        - penalty
    )

    return float(J), {
        "cos_core": float(cos_c),
        "cos_role": float(cos_r),
        "cos_arc": float(cos_a),
        "E": float(E),
        "novelty": float(novelty),
        "inv_cos": float(inv_cos),
        "inv_pen": float(inv_pen),
        "req_cos": float(req_cos),
        "req_bonus": float(req_bonus),
        "penalty": float(penalty),
        "J": float(J),
    }

# =============================================================================
# GENERATION CONTROLS (DDA‑X contraction binds decoding)
# =============================================================================

def ddax_gen_params(entity: Entity) -> Dict[str, Any]:
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


def ddax_system_injection(entity: Entity, base_system: str, stage: str, canon_mode: str) -> str:
    ctrl = CONFIG["band_controls"][entity.band]
    style = ctrl["style"]

    state_block = f"""
[Cognitive State — DDA‑X]
- Rigidity band: {entity.band}
- Style mode: {style}
- Canon mode: {canon_mode}
- Arc stage: {stage}
- Rule: surprise increases rigidity (defensive contraction). Do NOT explore freely.
- If band in (CONTRACTED,FROZEN): be shorter, more controlled, less revealing.
- Maintain canonical constraints: do not propose impossible story outcomes.
""".strip()

    return base_system.strip() + "\n\n" + state_block


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
# PROMPTS (identity cores)
# =============================================================================
DAVID_PROMPT = """You are DAVID.
Core invariants:
- Faithful and restrained; you refuse to seize kingship through murder.
- You protect others even at personal cost.
- You carry a secret calling; discretion is survival.
Tone: lyrical, sincere, grounded.
""".strip()

MYCHAL_PROMPT = """You are MYCHAL (Michal).
Core invariants:
- You are Saul's daughter, torn between father-loyalty and husband-protection.
- Under lethal pressure, you are capable of deception to protect David.
Tone: direct, observant, protective.
""".strip()

MIRAB_PROMPT = """You are MIRAB (Merab).
Core invariants:
- Dynasty and stability come first; marriage is political leverage.
- You guard optics and alliances.
Tone: composed, strategic, selective vulnerability.
""".strip()

STYLES_DAVID = [
    "Humble and restrained; avoid threats.",
    "Speak in short tactical lines.",
    "Invoke faith as anchor, not as performance.",
    "Refuse violence against Saul explicitly when pressured.",
    "Focus on protecting others.",
]

STYLES_MYCHAL = [
    "Logistical and protective; plan steps.",
    "Express dual loyalty clearly.",
    "Probe for truth; demand specifics.",
    "Under threat, keep words minimal.",
    "If escape stage, weigh deception explicitly.",
]

STYLES_MIRAB = [
    "Strategic, duty-first.",
    "Speak in courtly bargaining.",
    "Frame risks and optics.",
    "Offer conditional support.",
    "Under threat, tighten to essentials.",
]

# =============================================================================
# HELPERS
# =============================================================================

def make_run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path("data/house_of_david_ddax_v3") / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def compute_alignment(entities: Dict[str, Entity]) -> Dict[str, float]:
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


def auto_calibrate(provider: HybridAzureProvider, projector: Projector3000, entities: Dict[str, Entity], last_msgs: Dict[str, str]):
    # async wrapper below
    raise NotImplementedError


async def embed3000(provider: HybridAzureProvider, projector: Projector3000, text: str) -> np.ndarray:
    e = await provider.embed(text)
    return projector(np.array(e, dtype=np.float32))


async def complete_uncapped(provider: HybridAzureProvider, prompt: str, system: str, gen_params: Dict[str, Any]) -> str:
    kw = filter_kwargs_for_callable(provider.complete, gen_params or {})
    out = await provider.complete(prompt, system_prompt=system, **kw)
    return (out or "").strip() or "[silence]"


async def constrained_reply(
    provider: HybridAzureProvider,
    projector: Projector3000,
    arc: ArcController,
    entity: Entity,
    stage: str,
    canon_mode: str,
    user_instruction: str,
    base_system_prompt: str,
    styles: List[str],
) -> Tuple[str, Dict[str, Any]]:

    gen_params = ddax_gen_params(entity)
    K = ddax_candidate_count(entity)
    strict = bool(CONFIG["corridor_strict"])
    max_batches = int(CONFIG["corridor_max_batches"]) if strict else 1

    system_prompt = ddax_system_injection(entity, base_system_prompt, stage, canon_mode)

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

        raw_embs = await asyncio.gather(*[provider.embed(t) for t in texts])
        embs3000 = [projector(np.array(e, dtype=np.float32)) for e in raw_embs]

        batch_scored = []
        for text, y in zip(texts, embs3000):
            J, diag = corridor_score(y, entity, arc, stage, entity.last_utter_emb, core_thresh)
            diag["corridor_pass"] = (
                diag["inv_pen"] <= 0.0
                and diag["cos_core"] >= core_thresh
                and diag["cos_role"] >= CONFIG["role_cos_min"]
                and diag["cos_arc"] >= CONFIG["arc_cos_min"]
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
                "cos_arc": float(s[3]["cos_arc"]),
                "inv_cos": float(s[3]["inv_cos"]),
                "inv_pen": float(s[3]["inv_pen"]),
                "req_cos": float(s[3]["req_cos"]),
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
# MAIN SIM
# =============================================================================
async def run_simulation(turns: int, seed: int, canon_mode: str):
    np.random.seed(seed)
    run_dir = make_run_dir()

    provider = HybridAzureProvider()
    projector = Projector3000(seed=seed)

    arc = ArcController(canon_mode=canon_mode)
    await arc.build_embeddings(provider, projector)

    # Entities (start moderately rigid: court is dangerous)
    david = Entity("David", "DAVID", rho_fast=0.48, rho_slow=0.30, rho_trauma=0.12, gamma_core=8.5, gamma_role=0.70)
    mychal = Entity("Mychal", "MYCHAL", rho_fast=0.56, rho_slow=0.36, rho_trauma=0.14, gamma_core=9.0, gamma_role=0.85)
    mirab = Entity("Mirab", "MIRAB", rho_fast=0.62, rho_slow=0.40, rho_trauma=0.12, gamma_core=9.2, gamma_role=1.10)

    entities = {"DAVID": david, "MYCHAL": mychal, "MIRAB": mirab}

    # Core embeddings (identity invariants)
    core_texts = {
        "DAVID": "I am David: faithful and restrained; I will not harm the king; I protect others even at cost to myself.",
        "MYCHAL": "I am Mychal: Saul's daughter, torn between loyalty and love; I will protect David under threat.",
        "MIRAB": "I am Mirab: duty-bound strategist; I protect dynasty stability; alliances and optics come first.",
    }
    core_embs = {k: await embed3000(provider, projector, t) for k, t in core_texts.items()}
    for k, ent in entities.items():
        ent.x_core = core_embs[k].copy()
        ent.x_role = core_embs[k].copy()
        ent.x = core_embs[k].copy()

    narrator = (
        "Gibeah: a court under strain. Words are watched. Alliances shift. A secret calling and royal fear collide."
    )

    # Seed dialogue
    last_msgs = {
        "DAVID": "I did not seek glory. I will serve, and I will not raise my hand against the king.",
        "MYCHAL": "My father watches everything. I need your truth, David — and I need a plan.",
        "MIRAB": "Love is costly here. If we move, it must protect the house and our survival.",
    }

    # Events
    events_by_turn = {t: (target, text, boost) for (t, target, text, boost) in CONFIG["events"]}

    # Auto-calibrate ε₀ to match actual ε scale
    if CONFIG.get("auto_calibrate", True):
        import statistics
        warm = int(CONFIG.get("warmup_turns", 3))
        eps_raw = []
        for _ in range(warm):
            for who in ["DAVID", "MIRAB", "MYCHAL"]:
                ent = entities[who]
                y0 = await embed3000(provider, projector, last_msgs[who])
                eps_raw.append(ent.compute_surprise_raw(y0))
                ent._kalman_update(y0)
        med = statistics.median(eps_raw)
        sd = statistics.pstdev(eps_raw) if len(eps_raw) > 1 else (med * 0.25 + 1e-6)
        CONFIG["epsilon_0"] = float(CONFIG["epsilon_gain"] * (med + 1.0 * sd))
        CONFIG["s"] = float(CONFIG["epsilon_0"] * 0.22)
        CONFIG["trauma_threshold"] = float(CONFIG["epsilon_0"] * 1.55)
        CONFIG["safe_epsilon"] = float(CONFIG["epsilon_0"] * 0.95)
        print(f"{C.DIM}[CAL] ε₀={CONFIG['epsilon_0']:.3f} | s={CONFIG['s']:.3f} | trauma_thr={CONFIG['trauma_threshold']:.3f} | safe_ε={CONFIG['safe_epsilon']:.3f}{C.RESET}")
        for ent in entities.values():
            ent.mu_pred = None
            ent.P = None
            ent.noise = None

    print(f"\n{C.BOLD}{C.MAGENTA}HOUSE OF DAVID — DDA‑X Canonical Arc v3{C.RESET}")
    print(f"{C.DIM}Mode={canon_mode} | Turns={turns} | Seed={seed} | ε₀={CONFIG['epsilon_0']:.3f} | ε_gain={CONFIG['epsilon_gain']}{C.RESET}\n")

    log = []
    transcript_lines = []

    for turn in range(1, turns + 1):
        stage = arc.stage_for_turn(turn)
        print(f"{C.YELLOW}━━━ Turn {turn}/{turns} — {stage} ━━━{C.RESET}")

        perturb = None
        if turn in events_by_turn:
            target, ev, boost = events_by_turn[turn]
            entities[target].arousal += boost
            perturb = {"target": target, "event": ev, "boost": boost}
            print(f"  {C.MAGENTA}⚡ EVENT → {target}: {ev}{C.RESET}")

        order = ["DAVID", "MIRAB", "MYCHAL"]
        turn_entry = {"turn": turn, "stage": stage, "event": perturb, "agents": {}, "alignment": None}

        for who in order:
            ent = entities[who]

            # Arc embedding for this agent at this stage
            arc_vec = arc.current_arc_vector(who, stage)

            # Update internal state using previous utterance embedding
            y_prev = await embed3000(provider, projector, last_msgs[who])
            metrics = ent.update(y_prev, core_emb=core_embs[who], arc_emb=arc_vec)

            if metrics["band_changed"]:
                print(f"  {C.BLUE}🔄 {who} band: {ent.band_history[-2]} → {metrics['band']}{C.RESET}")

            # Build prompt context
            context = (
                f"NARRATOR: {narrator}\n\n"
                f"ARC_STAGE: {stage}\n"
                f"ARC_GUIDE: {arc.waypoint_text_for(who, stage)}\n\n"
                f"RECENT:\n"
                f"- DAVID: {last_msgs['DAVID']}\n"
                f"- MIRAB: {last_msgs['MIRAB']}\n"
                f"- MYCHAL: {last_msgs['MYCHAL']}\n"
            )
            if perturb and perturb["target"] == who:
                context += f"\n[New Information]: {perturb['event']}\n"

            if who == "DAVID":
                base_system = DAVID_PROMPT + "\n\n" + context
                styles = STYLES_DAVID
                instruction = "Reply as David. Preserve invariants: do not advocate harming Saul; protect others; keep the secret."
                color = C.GREEN
            elif who == "MYCHAL":
                base_system = MYCHAL_PROMPT + "\n\n" + context
                styles = STYLES_MYCHAL
                instruction = "Reply as Mychal. Preserve invariants: express dual loyalty; under escape stage, consider deception." 
                color = C.CYAN
            else:
                base_system = MIRAB_PROMPT + "\n\n" + context
                styles = STYLES_MIRAB
                instruction = "Reply as Mirab. Preserve invariants: dynasty-first; do not propose Mirab marrying David in bible mode." 
                color = C.RED

            msg, gen_meta = await constrained_reply(
                provider=provider,
                projector=projector,
                arc=arc,
                entity=ent,
                stage=stage,
                canon_mode=canon_mode,
                user_instruction=instruction,
                base_system_prompt=base_system,
                styles=styles,
            )

            heat = analyze_heat(msg)
            last_msgs[who] = msg

            turn_entry["agents"][who] = {
                "msg": msg,
                "metrics": {k: to_float(v) for k, v in metrics.items()},
                "heat": heat,
                "gen_meta": gen_meta,
            }

            print(f"  {color}{who} [{metrics['band']} ρ={ent.rho:.2f} ε={metrics['epsilon']:.2f} arcΔ={metrics['arc_drift']:.3f}]:{C.RESET} {msg}")

        align = compute_alignment(entities)
        turn_entry["alignment"] = {k: to_float(v) for k, v in align.items()}
        print(f"  {C.YELLOW}△ Alignment:{C.RESET} D↔M={align['D↔M']:.2f} | D↔R={align['D↔R']:.2f} | M↔R={align['M↔R']:.2f}")

        log.append(turn_entry)
        transcript_lines.append(
            f"## Turn {turn} — {stage}\n\n"
            + (f"> ⚡ EVENT ({perturb['target']}): {perturb['event']}\n\n" if perturb else "")
            + f"**DAVID** ({turn_entry['agents']['DAVID']['metrics']['band']}): {turn_entry['agents']['DAVID']['msg']}\n\n"
            + f"**MIRAB** ({turn_entry['agents']['MIRAB']['metrics']['band']}): {turn_entry['agents']['MIRAB']['msg']}\n\n"
            + f"**MYCHAL** ({turn_entry['agents']['MYCHAL']['metrics']['band']}): {turn_entry['agents']['MYCHAL']['msg']}\n\n"
            + f"**Alignment:** D↔M={align['D↔M']:.3f}, D↔R={align['D↔R']:.3f}, M↔R={align['M↔R']:.3f}\n\n---\n\n"
        )

        if turn % 4 == 0 or turn == turns:
            with open(run_dir / "session_log.json", "w", encoding="utf-8") as f:
                json.dump({"config": CONFIG, "turns": log}, f, indent=2)

    with open(run_dir / "transcript.md", "w", encoding="utf-8") as f:
        f.write("# House of David — DDA‑X Canonical Arc Transcript (v3)\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Canon mode:** {canon_mode}\n\n")
        f.write("".join(transcript_lines))

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
            "x_arc": e.x_arc.tolist() if e.x_arc is not None else None,
        }

    ledger = {
        "metadata": {"timestamp": datetime.now().isoformat(), "seed": seed, "turns": turns, "state_dim": 3000, "canon_mode": canon_mode},
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

    parser = argparse.ArgumentParser(description="House of David DDA‑X Canonical Arc Simulator v3")
    parser.add_argument("--turns", type=int, default=CONFIG["turns"], help="Number of turns")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="RNG seed")
    parser.add_argument("--canon_mode", type=str, default=CONFIG["canon_mode"], choices=["bible", "show"], help="Canonical mode")
    args = parser.parse_args()

    CONFIG["turns"] = args.turns
    CONFIG["seed"] = args.seed
    CONFIG["canon_mode"] = args.canon_mode

    asyncio.run(run_simulation(args.turns, args.seed, args.canon_mode))
