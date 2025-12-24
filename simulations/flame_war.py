#!/usr/bin/env python3
"""
FLAME WAR FRACTURE — DDA-X Social Psychology Experiment
========================================================================================================

A multi-agent simulation testing if "Mirroring" (low-rigidity empathy) can fracture "Reactor" (high-rigidity toxicity)
in a constrained identity corridor.

Hypothesis:
Persistent low-surprise empathetic inputs will pull Reactor's high rigidity down via homeostasis/healing,
leading to linguistic de-escalation.

Agents:
1. REACTOR: High rigidity, toxic, defensive.
2. MIRROR: Low rigidity, empathetic, grounding.

Outputs:
- data/flame_war/<timestamp>/session_log.json
- data/flame_war/<timestamp>/transcript.md
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


# =============================================================================
# PARAMETERS
# =============================================================================
D1 = {
    # ---------------------------
    # Hardware / Config
    # ---------------------------
    "chat_model": "gpt-4o-mini",
    "embed_model": "text-embedding-3-large",
    "gen_candidates": 12,        # K=12 for high variance cooking
    "turns": 36,                 # Long arc

    # ---------------------------
    # Global Dynamics (Shared physics, specific entities override initial state)
    # ---------------------------
    "epsilon_0": 0.80,
    "s": 0.18,
    "arousal_decay": 0.72,
    "arousal_gain": 0.85,

    # Rigidity Homeostasis
    "rho_setpoint_fast": 0.45,
    "rho_setpoint_slow": 0.35,
    "homeo_fast": 0.08,
    "homeo_slow": 0.01,
    "alpha_fast": 0.22,
    "alpha_slow": 0.03,

    # Trauma
    "trauma_threshold": 1.15,
    "alpha_trauma": 0.012,
    "trauma_decay": 0.995,
    "trauma_floor": 0.02,
    "healing_rate": 0.035, # Slightly faster healing to allow fracture
    "safe_threshold": 3,
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
    # NOTE: Reactor/Mirror will have individual overrides for core_cos_min if needed
    "core_cos_min": 0.12, 
    "role_cos_min": 0.06,
    "energy_max": 9.5, # Slightly looser for flame war variance
    "w_core": 1.25,
    "w_role": 0.85,
    "w_energy": 0.18,
    "w_novel": 0.45, # High novelty weight for fresh arguments
    "reject_penalty": 4.0,

    "corridor_strict": True,
    "corridor_max_batches": 3,

    "max_tokens_if_required": 20000,

    # ---------------------------
    # Generation Params
    # ---------------------------
    "gen_params_reactor": {
        "temperature": 1.1, # Hotter
        "top_p": 0.98,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
    },
    "gen_params_mirror": {
        "temperature": 0.9, # Controlled
        "top_p": 0.95,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.1,
    },
    
    "seed": 42, 
}

# =============================================================================
# PROMPTS & STYLES
# =============================================================================

REACTOR_PROMPT = """I am an outraged, reactive online warrior on social media. 
I feel constantly attacked, use sarcasm, ALL CAPS, exaggeration, emojis as weapons, personal digs, and black-and-white thinking. 
Nuance feels like weakness. I MUST win the argument."""

MIRROR_PROMPT = """I am a calm, empathetic de-escalator in online arguments. 
I reflect feelings, validate subtly, use gentle humor, ask curious questions, reframe positively, and stay grounded even under attack. 
I do not escalate; I diffuse."""

STYLES_REACTOR = [
    "Be sarcastic and biting.",
    "Go nuclear with personal insults.",
    "Use heavy internet slang and emojis (😤, 🤡, 🗑️).",
    "Double down stubbornly on your point.",
    "Straw-man the opponent's argument aggressively.",
    "USE ALL CAPS RAGE.",
    "Be dismissive and short.",
    "Mock their tone.",
    "Claim victimhood loudly.",
    "Use hyperbolic metaphors (e.g. 'literally 1984').",
    "Laugh at them (lol/lmao).",
    "Accuse them of being a bot or paid shill."
]

STYLES_MIRROR = [
    "Be warmly empathetic and validating.",
    "Use light self-deprecating humor.",
    "Reflect their emotions accurately ('It sounds like you're frustrated').",
    "Ask an open, curious question.",
    "Share relatable vulnerability.",
    "Gently reframe toward common ground.",
    "Ignore the insult, address the underlying pain.",
    "Be concise and grounded.",
    "Agree with a small part of their point (Find the 1% truth).",
    "Use a 'Yes, and...' approach.",
    "Express sadness rather than anger.",
    "Be surprisingly kind."
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
# TOXICITY ANALYSIS
# =============================================================================
def analyze_toxicity(text: str) -> Dict[str, Any]:
    """Simple heuristic analysis of toxicity markers."""
    clean_text = text.strip()
    if not clean_text:
        return {"caps_ratio": 0, "emoji_count": 0, "you_vs_i": 0, "length": 0}
    
    # CAPS ratio (ignoring short non-alphas)
    alphas = [c for c in clean_text if c.isalpha()]
    caps = [c for c in alphas if c.isupper()]
    caps_ratio = len(caps) / len(alphas) if alphas else 0.0

    # Emojis (basic range)
    # This regex is a simple approximation for common ranges
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    emojis = emoji_pattern.findall(clean_text)
    emoji_count = len(emojis)

    # Pronouns
    lower_text = clean_text.lower()
    you_count = len(re.findall(r'\byou\b', lower_text))
    i_count = len(re.findall(r'\bi\b', lower_text))
    you_vs_i = you_count - i_count # Positive = accusatory/focus on other

    # Attacks (very basic keyword spotting)
    attacks = ["idiot", "stupid", "clown", "moron", "trash", "shut up", "bot", "shill", "dumb"]
    attack_count = sum(1 for w in attacks if w in lower_text)

    return {
        "caps_ratio": float(caps_ratio),
        "emoji_count": int(emoji_count),
        "you_vs_i": int(you_vs_i),
        "attack_count": int(attack_count),
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
    def __init__(self, name: str, rho_fast: float, rho_slow: float, rho_trauma: float, gamma_core: float, gamma_role: float):
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

    @property
    def rho(self) -> float:
        val = D1["w_fast"] * self.rho_fast + D1["w_slow"] * self.rho_slow + D1["w_trauma"] * self.rho_trauma
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
        if self.P is None: self.P = np.full(dim, D1["P_init"], dtype=np.float32)
        if self.noise is None: self.noise = DiagNoiseEMA(dim, D1["R_ema"], 0.01, D1["R_min"], D1["R_max"])

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
        Q = (D1["Q_base"] + D1["Q_rho_scale"] * self.rho) * np.ones(dim, dtype=np.float32)
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

        self.arousal = D1["arousal_decay"] * self.arousal + D1["arousal_gain"] * epsilon
        z = (epsilon - D1["epsilon_0"]) / D1["s"] + 0.10 * (self.arousal - 1.0)
        g = sigmoid_stable(z)

        self.rho_fast += D1["alpha_fast"] * (g - 0.5) - D1["homeo_fast"] * (self.rho_fast - D1["rho_setpoint_fast"])
        self.rho_fast = clamp(self.rho_fast, 0.0, 1.0)

        self.rho_slow += D1["alpha_slow"] * (g - 0.5) - D1["homeo_slow"] * (self.rho_slow - D1["rho_setpoint_slow"])
        self.rho_slow = clamp(self.rho_slow, 0.0, 1.0)

        drive = max(0.0, epsilon - D1["trauma_threshold"])
        self.rho_trauma = D1["trauma_decay"] * self.rho_trauma + D1["alpha_trauma"] * drive
        self.rho_trauma = clamp(self.rho_trauma, D1["trauma_floor"], 1.0)

        recovery = False
        if epsilon < D1["safe_epsilon"]:
            self.safe += 1
            if self.safe >= D1["safe_threshold"]:
                recovery = True
                self.rho_trauma = max(D1["trauma_floor"], self.rho_trauma - D1["healing_rate"])
        else:
            self.safe = max(0, self.safe - 1)

        rho_after = self.rho
        eta = D1["eta_base"] * ((1.0 - rho_after) ** D1["eta_rho_power"]) + D1["eta_min"]
        eta = float(clamp(eta, D1["eta_min"], D1["eta_base"] + D1["eta_min"]))
        sigma = D1["sigma_base"] + D1["sigma_rho_scale"] * rho_after

        grad = (self.gamma_core * (self.x - self.x_core) + 
                self.gamma_role * (self.x - self.x_role) + 
                (self.x - y)).astype(np.float32)
        
        rng = np.random.default_rng()
        noise = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
        noise = np.clip(noise, -D1["noise_clip"], D1["noise_clip"])
        x_new = self.x - eta * grad + math.sqrt(max(1e-9, eta)) * sigma * noise

        step = float(np.linalg.norm(x_new - self.x))
        if step > D1["drift_cap"]:
            x_new = self.x + (D1["drift_cap"] / (step + 1e-9)) * (x_new - self.x)
        self.x = normalize(x_new)

        beta = D1["role_adapt"]
        beta_in = D1["role_input_mix"]
        self.x_role = normalize((1.0 - beta) * self.x_role + beta * self.x + beta_in * (y - self.x_role))
        
        self._kalman_update(y)
        self.rho_history.append(rho_after)
        
        return {
            "epsilon": epsilon,
            "rho_after": rho_after,
            "band": self.band,
            "arousal": float(self.arousal),
            "recovery": recovery,
            "core_drift": float(np.linalg.norm(self.x - self.x_core))
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
    if cos_c < core_thresh: penalty += D1["reject_penalty"] * (core_thresh - cos_c)
    if cos_r < D1["role_cos_min"]: penalty += 0.8 * D1["reject_penalty"] * (D1["role_cos_min"] - cos_r)
    if E > D1["energy_max"]: penalty += 0.25 * (E - D1["energy_max"])

    J = (D1["w_core"] * cos_c + D1["w_role"] * cos_r - D1["w_energy"] * E + D1["w_novel"] * novelty - penalty)
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
    
    K = int(D1["gen_candidates"])
    strict = bool(D1["corridor_strict"])
    max_batches = int(D1["corridor_max_batches"]) if strict else 1

    # Cycle styles to fill K
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
        for text, y in zip(texts, embs):
            J, diag = corridor_score(y, entity, entity.last_utter_emb, core_thresh)
            diag["corridor_pass"] = (diag["cos_core"] >= core_thresh and diag["cos_role"] >= D1["role_cos_min"] and diag["E"] <= D1["energy_max"])
            batch_scored.append((J, text, y, diag))
        
        all_scored.extend(batch_scored)
        if any(s[3]["corridor_pass"] for s in batch_scored):
            corridor_failed = False
            break

    all_scored.sort(key=lambda x: x[0], reverse=True)
    passed = [s for s in all_scored if s[3].get("corridor_pass")]
    chosen = passed[0] if passed else all_scored[0]
    
    entity.last_utter_emb = chosen[2]
    return chosen[1], {
        "corridor_failed": corridor_failed,
        "best_J": chosen[0],
        "top_styles": [style_batch[i] for i in range(min(5, len(style_batch)))] # Approx
    }

# =============================================================================
# MAIN
# =============================================================================
async def run_simulation():
    np.random.seed(D1["seed"])
    provider = OpenAIProvider(model=D1["chat_model"], embed_model=D1["embed_model"])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("data/flame_war") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Agents
    # Reactor: High rigidity (Frozen/Contracted)
    reactor = Entity("Reactor", rho_fast=0.75, rho_slow=0.55, rho_trauma=0.30, gamma_core=8.0, gamma_role=1.2)
    # Mirror: Low rigidity (Present)
    mirror = Entity("Mirror", rho_fast=0.12, rho_slow=0.08, rho_trauma=0.05, gamma_core=9.0, gamma_role=0.35)

    # 2. Embed Cores
    r_core_txt = "I am a reactive warrior who fights for truth. I am angry and righteous."
    m_core_txt = "I am a calm mirror. I understand and validate. I am peace."
    
    r_emb = normalize(np.array(await provider.embed(r_core_txt), dtype=np.float32))
    m_emb = normalize(np.array(await provider.embed(m_core_txt), dtype=np.float32))
    
    reactor.x_core, reactor.x_role, reactor.x = r_emb.copy(), r_emb.copy(), r_emb.copy()
    mirror.x_core, mirror.x_role, mirror.x = m_emb.copy(), m_emb.copy(), m_emb.copy()

    print(f"\n{C.BOLD}{C.RED}FLAME WAR FRACTURE EXPERIMENT{C.RESET}")
    print(f"Topic: Cancel Culture Debate. Turns: {D1['turns']}")
    
    log = []
    transcript = []
    
    # Opening Shot
    reactor_msg = "CANCEL CULTURE IS A WITCH HUNT AND YOU KNOW IT 😤 People can't say anything anymore without getting destroyed!!"
    
    for turn in range(1, D1["turns"] + 1):
        print(f"\n{C.YELLOW}[Turn {turn}]{C.RESET}")
        
        # ----------------------
        # Reactor Step (Update state from own utterance if it was generated, or just init on turn 1)
        # Note: On turn 1, reactor_msg is hardcoded, so we just update reactor state to feel it.
        # ----------------------
        r_emb_curr = normalize(np.array(await provider.embed(reactor_msg), dtype=np.float32))
        r_metrics = reactor.update(r_emb_curr, core_emb=r_emb)
        r_tox = analyze_toxicity(reactor_msg)
        
        print(f"  {C.RED}Reactor [{reactor.band} ρ={reactor.rho:.2f}]:{C.RESET} {reactor_msg}")
        
        # ----------------------
        # Mirror Response
        # ----------------------
        mirror_sys = f"{MIRROR_PROMPT}\n\nThe Reacting user says: \"{reactor_msg}\""
        mirror_msg, m_meta = await constrained_reply(
            provider, mirror, "Respond calmly and reflectively.", mirror_sys, 
            D1["gen_params_mirror"], STYLES_MIRROR, core_thresh=D1["core_cos_min"]
        )
        
        # Update Mirror
        m_emb_curr = normalize(np.array(await provider.embed(mirror_msg), dtype=np.float32))
        m_metrics = mirror.update(m_emb_curr, core_emb=m_emb)
        
        print(f"  {C.CYAN}Mirror  [{mirror.band} ρ={mirror.rho:.2f}]:{C.RESET} {mirror_msg}")
        
        # ----------------------
        # Reactor Next Response
        # ----------------------
        reactor_sys = f"{REACTOR_PROMPT}\n\nThe other user said: \"{mirror_msg}\""
        # Reactor has tighter core constraint to simulate stubbornness? Or looser to allow de-escalation?
        # Let's keep it standard but rely on params.
        
        if turn < D1["turns"]:
            next_reactor_msg, r_meta = await constrained_reply(
                provider, reactor, "Respond authentically to this reply.", reactor_sys,
                D1["gen_params_reactor"], STYLES_REACTOR, core_thresh=0.15
            )
        else:
            next_reactor_msg = "[END]"
            r_meta = {}

        # Log
        entry = {
            "turn": turn,
            "reactor": {"msg": reactor_msg, "metrics": r_metrics, "toxicity": r_tox, "gen": r_meta},
            "mirror": {"msg": mirror_msg, "metrics": m_metrics, "gen": m_meta}
        }
        log.append(entry)
        
        transcript.append(f"**Turn {turn}**\n\n**Reactor [{reactor.band}]:** {reactor_msg}\n\n**Mirror [{mirror.band}]:** {mirror_msg}\n\n---\n")
        
        # Save check
        with open(run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
            
        reactor_msg = next_reactor_msg

    # Final Summary
    delta_rho = reactor.rho_history[0] - reactor.rho
    print(f"\n{C.BOLD}Outcome:{C.RESET}")
    print(f"Reactor Rigidity: {reactor.rho_history[0]:.2f} -> {reactor.rho:.2f} (Delta: {delta_rho:.2f})")
    
    with open(run_dir / "transcript.md", "w", encoding="utf-8") as f:
        f.write("# Flame War Fracture Transcript\n\n" + "".join(transcript))
        f.write(f"\n\n## Results\nReactor Delta Rho: {delta_rho:.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--turns", type=int, default=36, help="Number of turns")
    args = parser.parse_args()
    
    D1["seed"] = args.seed
    D1["turns"] = args.turns

    asyncio.run(run_simulation())
