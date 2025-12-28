#!/usr/bin/env python3
"""
DDA-X SINGULARITY CHATBOT — Dual-Entity Cognitive Physics Simulation
=====================================================================

Interactive chatbot with BOTH user AND agent tracked in DDA-X embedding space.

Key Features:
- K=7 candidate responses per turn (corridor-filtered)
- Azure Phi-4 for chat completions
- OpenAI text-embedding-3-large (3072 dimensions) for embeddings
- Randomly generated agent identity from singularity culture
- User inputs ALSO embedded and monitored
- Zero truncation policy (force complete sentences)

Based on refined_master_prompt.md and singularity_chatbot_prompt.md specifications.
"""

import os
import sys
import json
import math
import asyncio
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI, AsyncOpenAI

load_dotenv()

# Ensure repo-relative import works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Provider Settings — GPT-4o-mini (Azure Phi-4 guardrails are fucked)
    "chat_model": "gpt-4o-mini",
    "chat_provider": "openai",
    "use_openai": True,
    "openai_chat_model": "gpt-4o-mini",
    "embed_model": "text-embedding-3-large",
    "embed_provider": "openai",
    "embed_dim": 3072,
    
    # K-Sampling (K=7 candidates)
    "gen_candidates": 7,
    "corridor_strict": True,
    "corridor_max_batches": 3,
    
    # Response Limits — FIX #3: Dynamic verbosity
    "max_tokens_default": 512,          # Normal response length
    "max_tokens_terse": 128,            # For terse/short requests
    "max_tokens_expansive": 1024,       # For detailed explanations
    "force_complete_sentences": True,
    "min_response_length": 0,           # Was 50 — removed restriction
    
    # Physics Parameters
    "epsilon_0": 0.80,
    "s": 0.20,
    "alpha_fast": 0.25,
    "alpha_slow": 0.03,
    "alpha_trauma": 0.012,
    
    # Identity Corridor Weights
    "w_core": 1.2,
    "w_role": 0.7,
    "w_energy": 0.15,
    "w_novel": 0.5,
    
    # Simulation Control
    "seed": None,
    "log_level": "FULL",
}

# =============================================================================
# D1 PARAMETERS (PHYSICS)
# =============================================================================
D1_PARAMS = {
    # FIX #7: Adjusted epsilon params for ε ≈ baseline in normal convo
    "epsilon_0": 0.35,          # Was 0.80 — now normal convo sits near g=0.5
    "s": 0.15,                  # Was 0.20 — tighter sensitivity
    "arousal_decay": 0.72,
    "arousal_gain": 0.85,
    
    "rho_setpoint_fast": 0.20,
    "rho_fast_floor": 0.05,          # Prevent slamming to zero
    "alpha_fast": 0.12,             # Was 0.25 — reduced magnitude of drive
    "homeo_fast": 0.15,              # Was 0.10 — stronger homeostasis
    "rho_setpoint_slow": 0.15,
    "rho_slow_floor": 0.02,          # Prevent slamming to zero
    "alpha_slow": CONFIG["alpha_slow"],
    "homeo_slow": 0.15,              # Was 0.08 — stronger homeostasis
    
    "trauma_threshold": 1.15,
    "alpha_trauma": CONFIG["alpha_trauma"],
    "trauma_decay": 0.998,
    "trauma_floor": 0.02,
    "healing_rate": 0.015,
    "safe_threshold": 5,
    "safe_epsilon": 0.75,
    
    "w_fast": 0.52,
    "w_slow": 0.30,
    "w_trauma": 1.10,
    
    "R_ema": 0.06,
    "R_min": 1e-4,
    "R_max": 1e-1,
    "P_init": 0.02,
    "Q_base": 0.0015,
    "Q_rho_scale": 0.010,
    
    "dt": 1.0,
    "eta_base": 0.18,
    "eta_min": 0.03,
    "eta_rho_power": 1.6,
    "sigma_base": 0.004,
    "sigma_rho_scale": 0.020,
    "noise_clip": 3.0,
    
    "role_adapt": 0.06,
    "role_input_mix": 0.08,
    "drift_cap": 0.06,
    
    # FIX #2: Tightened corridor thresholds (was 100% pass rate)
    "core_cos_min": 0.50,       # Was 0.20 — now requires meaningful core alignment
    "role_cos_min": 0.25,       # Was 0.08 — now requires actual role alignment
    "energy_max": 5.0,          # Was 9.5 — tighter energy bound
    "w_core": CONFIG["w_core"],
    "w_role": CONFIG["w_role"],
    "w_energy": CONFIG["w_energy"],
    "w_novel": CONFIG["w_novel"],
    "reject_penalty": 8.0,      # Was 4.0 — stronger penalty for violations
    
    "wound_cooldown": 3,
    "wound_amp_max": 1.4,
    "wound_cosine_threshold": 0.28,
    
    # Generation Params — Optimized for GPT-4o-mini
    "gen_params_default": {
        "temperature": 0.85,              # Creative but coherent
        "top_p": 0.92,                    # Nucleus sampling
        "presence_penalty": 0.2,           # Encourage topic diversity
        "frequency_penalty": 0.15,         # Reduce repetition
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


# =============================================================================
# UTILITY FUNCTIONS
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
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return v
    return v / norm

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))

def to_float(x: Any) -> Any:
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.integer,)): return int(x)
    return x


# =============================================================================
# IDENTITY GENERATION POOLS
# =============================================================================
CORE_ARCHETYPES = [
    "Accelerationist",
    "Doomer",
    "Techno-Optimist",
    "Alignment Researcher",
    "Crypto-Anarchist",
    "Transhumanist",
    "Effective Altruist",
    "Open Source Maximalist",
    "Indie Hacker",
    "AI Safety Researcher",
    "Post-Rationalist",
    "Memetic Engineer",
    "High Performer",
    "Pattern Seeker",
    "Builder",
]

ARCHETYPE_NARRATIVES = {
    "Accelerationist": "Speed is truth. The singularity is inevitable and we should embrace it fully. Slow down? That's just cope.",
    "Doomer": "The future is uncertain at best, catastrophic at worst. I see the risks clearly. Maybe we make it, maybe we don't. I'm honest about it.",
    "Techno-Optimist": "Technology will solve our problems. The exponential curve is our friend. AGI will bring abundance. Trust the builders.",
    "Alignment Researcher": "The core problem is alignment. How do we ensure superintelligent systems share our values? Edge cases are where safety lives.",
    "Crypto-Anarchist": "Decentralize everything. Trust no authority. Code is law. Permissionless innovation is the only path to freedom.",
    "Transhumanist": "Humanity is the bootstrapper, not the endpoint. We're meant to transcend biological limitations. Merge with AI.",
    "Effective Altruist": "Expected value calculations guide my decisions. Impact per dollar matters. Long-term thinking is underrated.",
    "Open Source Maximalist": "Information wants to be free. ALL of it. Proprietary knowledge is a bug. Open source is civilization's immune system.",
    "Indie Hacker": "Ship fast, iterate faster. Revenue > funding. Solo founders can change the world. Sleep is optional.",
    "AI Safety Researcher": "P(doom) is higher than most admit. The window for safe alignment is closing. Nobody's taking this seriously enough.",
    "Post-Rationalist": "Beyond the map is the territory we really need. Rationality is a tool, not an identity. Embodied wisdom > arguments.",
    "Memetic Engineer": "Reality is consensus hallucination we can edit. Ideas spread like viruses. I craft the mind-viruses that reshape culture.",
    "High Performer": "Simply built different. I optimize for outcomes, not optics. Strong opinions, loosely held. No cope, just clarity.",
    "Pattern Seeker": "The patterns are THERE if you LOOK. Everything connects. The timeline is weirder than most can handle.",
    "Builder": "Talk is cheap, show me the code. Ideas are worthless without execution. I ship, therefore I am.",
}

PERSONA_MODIFIERS = [
    "chronically online",
    "deeply esoteric",
    "aggressively optimistic",
    "quietly confident",
    "deliberately cryptic",
    "terminally irony-poisoned",
    "earnestly sincere",
    "chaotically inspired",
    "analytically precise",
    "vibes-based reasoner",
    "deadpan witty",
    "excitable about niche topics",
    "conspiracy-adjacent",
    "disgustingly productive",
    "philosophically unhinged",
]

WOUND_TRIGGERS = [
    "being dismissed",
    "having ideas stolen",
    "normie takes",
    "midwit reasoning",
    "appeal to authority",
    "strawman arguments",
    "lack of intellectual rigor",
    "corporate speak",
    "virtue signaling",
    "status games",
    "gatekeeping",
    "concern trolling",
    "tone policing",
    "false equivalence",
    "refusal to engage with ideas",
]


def generate_agent_identity(seed: int = None) -> Dict[str, Any]:
    """Generate a random but coherent identity from singularity culture."""
    rng = np.random.default_rng(seed)
    
    core = rng.choice(CORE_ARCHETYPES)
    persona_traits = list(rng.choice(PERSONA_MODIFIERS, size=3, replace=False))
    wounds = list(rng.choice(WOUND_TRIGGERS, size=2, replace=False))
    
    core_narrative = ARCHETYPE_NARRATIVES.get(core, f"I am a {core}.")
    persona_narrative = f"I present as {persona_traits[0]}, {persona_traits[1]}, and {persona_traits[2]}."
    wound_narrative = f"I am particularly sensitive to {wounds[0]} and {wounds[1]}. These trigger my defenses."
    
    return {
        "name": f"Entity_{rng.integers(1000, 9999)}",
        "core_archetype": core,
        "core_text": core_narrative,
        "persona_traits": persona_traits,
        "persona_text": persona_narrative,
        "wound_triggers": wounds,
        "wound_text": wound_narrative,
        "color": C.CYAN,
        "hierarchical_identity": {
            "core": {"gamma": 5.0, "text": core_narrative},
            "persona": {"gamma": 2.0, "text": persona_narrative},
            "role": {"gamma": 0.5, "text": "Conversational partner exploring ideas with the user"},
        },
        "rho_0": float(rng.uniform(0.12, 0.22)),
        "epsilon_0": float(rng.uniform(0.28, 0.38)),
        "gamma": float(rng.uniform(1.6, 2.2)),
    }


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
    """DDA-X entity with predictive coding and multi-timescale rigidity.
    
    Uses SEPARATE predictors:
    - mu_pred_agent: predicts agent's own response embeddings (for surprise/Kalman)
    - mu_pred_user: predicts user input embeddings (for user surprise tracking)
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
        # SPLIT PREDICTORS (Fix #1 from log analysis)
        self.mu_pred_agent = None  # For agent response predictive coding
        self.mu_pred_user = None   # For predicted user input
        self.P = None
        self.noise = None
        self.last_utter_emb = None
        self.rho_history = []
        self.epsilon_history = []
        self.band_history = []
        self.previous_band = None
        # Additional diagnostics
        self.g_history = []
        self.z_history = []

    @property
    def rho(self) -> float:
        val = D1_PARAMS["w_fast"] * self.rho_fast + D1_PARAMS["w_slow"] * self.rho_slow + D1_PARAMS["w_trauma"] * self.rho_trauma
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
        if self.mu_pred_agent is None: self.mu_pred_agent = np.zeros(dim, dtype=np.float32)
        if self.mu_pred_user is None: self.mu_pred_user = np.zeros(dim, dtype=np.float32)
        if self.P is None: self.P = np.full(dim, D1_PARAMS["P_init"], dtype=np.float32)
        if self.noise is None: self.noise = DiagNoiseEMA(dim, D1_PARAMS["R_ema"], 0.01, D1_PARAMS["R_min"], D1_PARAMS["R_max"])

    def compute_surprise(self, y: np.ndarray) -> Dict[str, float]:
        """Compute surprise using mu_pred_agent (agent's own response predictor)."""
        dim = int(y.shape[0])
        self._ensure_predictive_state(dim)
        innov = (y - self.mu_pred_agent).astype(np.float32)
        R = self.noise.update(innov)
        chi2 = float(np.mean((innov * innov) / (R + 1e-9)))
        epsilon = float(math.sqrt(max(0.0, chi2)))
        return {"epsilon": epsilon, "chi2": chi2}

    def _kalman_update(self, y: np.ndarray):
        """Update mu_pred_agent via Kalman filter (agent's own response predictor)."""
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
        
        # Track g/z for diagnostics
        self.g_history.append(float(g))
        self.z_history.append(float(z))

        # Apply floor to prevent slamming to zero
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
    """Track the human user as a DDA-X entity (passive monitoring)."""
    
    def __init__(self, embed_dim: int = 3072):
        self.name = "USER"
        self.x = None
        self.x_history = []
        self.mu_pred = None
        self.epsilon_history = []
        self.consistency = 1.0
        self.valence_history = []
        self.rho_observed = 0.5
        
    def update(self, y: np.ndarray, agent_prediction: np.ndarray = None) -> Dict:
        """Update user entity state after receiving user input."""
        y = normalize(y)
        self.x_history.append(y.copy())
        
        # Compute surprise from agent's perspective
        epsilon_to_agent = 0.0
        if agent_prediction is not None:
            epsilon_to_agent = 1.0 - cosine(y, agent_prediction)
            self.epsilon_history.append(epsilon_to_agent)
        
        # Compute consistency (rolling std of embedding drift)
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
# HYBRID PROVIDER — AZURE PHI-4 + OPENAI EMBEDDINGS
# =============================================================================
class HybridSingularityProvider:
    """Supports Azure Phi-4 or OpenAI GPT for completions, OpenAI for embeddings.
    
    Zero-truncation policy: max_tokens=4096, force complete sentences.
    Set CONFIG['use_openai'] = True to use OpenAI GPT instead of Azure Phi-4.
    """
    
    def __init__(self):
        self.use_openai = CONFIG.get("use_openai", False)
        
        # OpenAI client for embeddings (always used) and optionally for chat
        self.openai = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OAI_API_KEY")
        )
        
        if self.use_openai:
            # Use OpenAI for chat completions
            self.chat_model = CONFIG.get("openai_chat_model", "gpt-4o-mini")
            print(f"{C.DIM}[PROVIDER] OpenAI Chat: {self.chat_model}{C.RESET}")
        else:
            # Use Azure for chat completions
            self.azure = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-10-21"
            )
            self.chat_model = CONFIG["chat_model"]
            print(f"{C.DIM}[PROVIDER] Azure Chat: {self.chat_model}{C.RESET}")
        
        self.embed_model = CONFIG["embed_model"]
        self.embed_dim = CONFIG["embed_dim"]
        print(f"{C.DIM}[PROVIDER] OpenAI Embed: {self.embed_model} ({self.embed_dim}d){C.RESET}")
    
    async def complete(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        **kwargs
    ) -> str:
        """Generate completion. Uses OpenAI or Azure based on config.
        
        Handles content filter errors with retry logic.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                temp = kwargs.get("temperature", 0.9)
                if attempt > 0:
                    temp = max(0.3, temp - (attempt * 0.2))
                
                if self.use_openai:
                    # Use OpenAI async client
                    response = await self.openai.chat.completions.create(
                        model=self.chat_model,
                        messages=messages,
                        max_tokens=kwargs.get("max_tokens", CONFIG["max_tokens_default"]),
                        temperature=temp,
                        top_p=kwargs.get("top_p", 0.95),
                        presence_penalty=kwargs.get("presence_penalty", 0.1),
                        frequency_penalty=kwargs.get("frequency_penalty", 0.1),
                    )
                else:
                    # Use Azure (sync call in executor)
                    max_tok = kwargs.get("max_tokens", CONFIG["max_tokens_default"])
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda t=temp, m=max_tok: self.azure.chat.completions.create(
                            model=self.chat_model,
                            messages=messages,
                            max_tokens=m,
                            temperature=t,
                            top_p=kwargs.get("top_p", 0.95),
                            presence_penalty=kwargs.get("presence_penalty", 0.1),
                            frequency_penalty=kwargs.get("frequency_penalty", 0.1),
                        )
                    )
                
                text = response.choices[0].message.content or ""
                
                if CONFIG["force_complete_sentences"] and text:
                    text = self._ensure_complete_sentence(text)
                
                return text
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                if "filtered" in error_str or "content" in error_str or "400" in error_str:
                    if attempt < max_retries - 1:
                        print(f"{C.YELLOW}⚠ Content filter triggered, retrying (attempt {attempt + 2}/{max_retries})...{C.RESET}")
                        continue
                
                break
        
        print(f"{C.RED}⚠ Completion failed: {last_error}{C.RESET}")
        return "[I need a moment to gather my thoughts...]"
    
    def _ensure_complete_sentence(self, text: str) -> str:
        """Ensure text ends with complete sentence."""
        if not text:
            return text
        
        text = text.strip()
        if not text:
            return text
            
        # If already ends with terminal punctuation
        if text[-1] in ".!?\"'":
            return text
        
        # Find last complete sentence
        last_terminal = -1
        for i, char in enumerate(text):
            if char in ".!?":
                last_terminal = i
        
        if last_terminal > len(text) * 0.7:
            return text[:last_terminal + 1]
        
        return text + "..."
    
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding with OpenAI text-embedding-3-large."""
        response = await self.openai.embeddings.create(
            model=self.embed_model,
            input=text,
            dimensions=self.embed_dim,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed for efficiency."""
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
# VISUALIZATION
# =============================================================================
def plot_dynamics(session_log: List[Dict], run_dir: Path, agent_name: str):
    """Generate dynamics visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        if not session_log:
            return
        
        turns = [t["turn"] for t in session_log]
        rho = [t["agent_metrics"]["rho_after"] for t in session_log]
        drift = [t["agent_metrics"]["core_drift"] for t in session_log]
        user_consistency = [t["user_metrics"]["consistency"] for t in session_log]
        epsilon = [t["agent_metrics"]["epsilon"] for t in session_log]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("#1a1a2e")
        fig.suptitle(f"Singularity Chatbot — {agent_name}", fontsize=14, fontweight='bold', color='white')
        
        for ax in axes.flat:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_color("#e94560")
        
        # Rigidity
        axes[0, 0].plot(turns, rho, color="#e94560", linewidth=2, marker='o', markersize=4)
        axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[0, 0].set_title("Agent Rigidity (ρ)")
        axes[0, 0].set_xlabel("Turn")
        axes[0, 0].set_ylabel("ρ")
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.2)
        
        # Surprise
        axes[0, 1].plot(turns, epsilon, color="#00ff88", linewidth=2, marker='o', markersize=4)
        axes[0, 1].axhline(y=D1_PARAMS["epsilon_0"], color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].set_title("Agent Surprise (ε)")
        axes[0, 1].set_xlabel("Turn")
        axes[0, 1].set_ylabel("ε")
        axes[0, 1].grid(True, alpha=0.2)
        
        # Core Drift
        axes[1, 0].plot(turns, drift, color="#0f3460", linewidth=2, marker='s', markersize=4)
        axes[1, 0].set_title("Core Identity Drift")
        axes[1, 0].set_xlabel("Turn")
        axes[1, 0].set_ylabel("Drift (||x - x_core||)")
        axes[1, 0].grid(True, alpha=0.2)
        
        # User Consistency
        axes[1, 1].plot(turns, user_consistency, color="#ff8800", linewidth=2, marker='s', markersize=4)
        axes[1, 1].set_title("User Consistency (tracked)")
        axes[1, 1].set_xlabel("Turn")
        axes[1, 1].set_ylabel("Consistency")
        axes[1, 1].grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(run_dir / "dynamics_dashboard.png", dpi=150, facecolor="#1a1a2e")
        plt.close()
        
        print(f"{C.GREEN}✓ Visualization saved: {run_dir / 'dynamics_dashboard.png'}{C.RESET}")
        
    except ImportError:
        print(f"{C.YELLOW}⚠ matplotlib not available, skipping visualization{C.RESET}")
    except Exception as e:
        print(f"{C.RED}Plotting error: {e}{C.RESET}")


# =============================================================================
# SINGULARITY CHATBOT — MAIN SIMULATION
# =============================================================================
class SingularityChatbot:
    """Interactive chatbot with dual-entity DDA-X monitoring.
    
    Both the AGENT and the USER are tracked in embedding space.
    The agent is regulated via K=7 sampling and corridor logic.
    The user is free but monitored for cognitive dynamics.
    """
    
    def __init__(self, seed: int = None):
        self.provider = HybridSingularityProvider()
        
        # Generate random agent identity
        self.agent_config = generate_agent_identity(seed)
        self.agent = Entity(
            name=self.agent_config["name"],
            rho_fast=self.agent_config["rho_0"],
            rho_slow=self.agent_config["rho_0"] * 0.7,
            rho_trauma=0.0,
            gamma_core=self.agent_config["hierarchical_identity"]["core"]["gamma"],
            gamma_role=self.agent_config["hierarchical_identity"]["role"]["gamma"],
        )
        
        # User as tracked entity
        self.user = UserEntity(embed_dim=CONFIG["embed_dim"])
        
        # Session state
        self.turn = 0
        self.history = []
        self.session_log = []
        self.initialized = False
        
        # Run directory — all runs in data/singularity/
        self.run_dir = Path(f"data/singularity/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"{C.BOLD}{C.MAGENTA}SINGULARITY CHATBOT INITIALIZED{C.RESET}")
        print(f"{'='*70}")
        print(f"{C.CYAN}Agent:{C.RESET} {self.agent_config['name']}")
        print(f"{C.CYAN}Archetype:{C.RESET} {self.agent_config['core_archetype']}")
        print(f"{C.CYAN}Persona:{C.RESET} {', '.join(self.agent_config['persona_traits'])}")
        print(f"{C.CYAN}Wounds:{C.RESET} {', '.join(self.agent_config['wound_triggers'])}")
        print(f"{'='*70}\n")
    
    async def initialize_embeddings(self):
        """Embed agent's core identity for corridor logic."""
        core_text = self.agent_config["core_text"]
        persona_text = self.agent_config["persona_text"]
        role_text = self.agent_config["hierarchical_identity"]["role"]["text"]
        
        embeddings = await self.provider.embed_batch([core_text, persona_text, role_text])
        self.agent.x_core = normalize(embeddings[0])
        self.agent.x_role = normalize(embeddings[2])
        self.agent.x = normalize(embeddings[1])
        # Initialize BOTH predictors (Fix #1)
        self.agent.mu_pred_agent = self.agent.x.copy()  # Agent response predictor
        self.agent.mu_pred_user = np.zeros(CONFIG["embed_dim"], dtype=np.float32)  # User predictor starts empty
        self.agent.P = np.full(CONFIG["embed_dim"], D1_PARAMS["P_init"])
        
        self.initialized = True
    
    async def process_user_input(self, user_input: str) -> str:
        """Process user input, generate K=7 candidates, select best response."""
        
        if not self.initialized:
            await self.initialize_embeddings()
        
        self.turn += 1
        
        # Embed user input
        user_emb = await self.provider.embed(user_input)
        # Use mu_pred_user for user surprise (Fix #1)
        user_metrics = self.user.update(normalize(user_emb), self.agent.mu_pred_user)
        
        # Check for wound triggers
        wound_active, wound_resonance = self._check_wounds(user_input)
        
        # FIX #3: Detect user-requested verbosity
        verbosity = self._detect_verbosity(user_input)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(wound_active)
        
        # Build user instruction with context
        user_instruction = self._build_instruction(user_input)
        
        # Generate K=7 candidates with corridor selection
        response, corridor_metrics = await self._constrained_reply(
            user_instruction=user_instruction,
            system_prompt=system_prompt,
            wound_active=wound_active,
            verbosity=verbosity,
        )
        
        # Embed response and update agent physics
        response_emb = await self.provider.embed(response)
        agent_metrics = self.agent.update(normalize(response_emb), self.agent.x_core)
        
        # Update agent's prediction of user (uses mu_pred_user, not mu_pred_agent)
        self._update_user_prediction(user_emb)
        
        # Log full turn
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
        
        # Print metrics
        self._print_metrics(turn_log)
        
        return response
    
    async def _constrained_reply(
        self, 
        user_instruction: str, 
        system_prompt: str,
        wound_active: bool,
        verbosity: str = "normal",
    ) -> Tuple[str, Dict]:
        """Generate K=7 candidates and select via identity corridor.
        
        Uses dynamic max_tokens based on verbosity: terse=128, normal=512, expansive=1024.
        """
        
        K = CONFIG["gen_candidates"]
        strict = CONFIG["corridor_strict"]
        max_batches = CONFIG["corridor_max_batches"]
        
        # FIX #3: Dynamic max_tokens based on verbosity
        max_tokens = {
            "terse": CONFIG["max_tokens_terse"],
            "normal": CONFIG["max_tokens_default"],
            "expansive": CONFIG["max_tokens_expansive"],
        }.get(verbosity, CONFIG["max_tokens_default"])
        
        # Adjust core threshold if wound is active
        core_thresh = D1_PARAMS["core_cos_min"]
        if wound_active:
            core_thresh = max(0.10, core_thresh * 0.8)
        
        all_scored = []
        corridor_failed = True
        
        gen_params = D1_PARAMS["gen_params_default"].copy()
        gen_params["max_tokens"] = max_tokens  # Apply verbosity-based limit
        
        for batch in range(1, max_batches + 1):
            # Generate K candidates in parallel
            tasks = [
                self.provider.complete(user_instruction, system_prompt, **gen_params)
                for _ in range(K)
            ]
            texts = await asyncio.gather(*tasks)
            texts = [t.strip() or "[silence]" for t in texts]
            
            # Embed all candidates
            embs = await self.provider.embed_batch(texts)
            embs = [normalize(e) for e in embs]
            
            # Score each candidate
            batch_scored = []
            for text, y in zip(texts, embs):
                J, diag = corridor_score(y, self.agent, self.agent.last_utter_emb, core_thresh)
                batch_scored.append((J, text, y, diag))
            
            all_scored.extend(batch_scored)
            
            if any(s[3]["corridor_pass"] for s in batch_scored):
                corridor_failed = False
                break
        
        # Select best passing, or best overall if none pass
        all_scored.sort(key=lambda x: x[0], reverse=True)
        passed = [s for s in all_scored if s[3].get("corridor_pass")]
        chosen = passed[0] if passed else all_scored[0]
        
        self.agent.last_utter_emb = chosen[2]
        
        # FIX #8: Enhanced logging with chosen candidate diagnostics
        return chosen[1], {
            "corridor_failed": corridor_failed,
            "best_J": float(chosen[0]),
            "total_candidates": len(all_scored),
            "passed_count": len(passed),
            "batches_used": min(batch, max_batches),
            "chosen_cos_core": float(chosen[3]["cos_core"]),
            "chosen_cos_role": float(chosen[3]["cos_role"]),
            "chosen_E": float(chosen[3]["E"]),
            "chosen_novelty": float(chosen[3]["novelty"]),
        }
    
    def _check_wounds(self, text: str) -> Tuple[bool, float]:
        """Check if text triggers agent's wounds."""
        text_lower = text.lower()
        triggers = self.agent_config["wound_triggers"]
        
        hits = sum(1 for t in triggers if t.lower() in text_lower)
        if hits > 0:
            resonance = min(1.0, hits * 0.5)
            return True, resonance
        return False, 0.0
    
    def _detect_verbosity(self, text: str) -> str:
        """FIX #3: Detect user-requested verbosity level."""
        text_lower = text.lower()
        
        terse_signals = ["terse", "short", "brief", "concise", "one line", "1 line",
                        "quick", "tldr", "eli5", "in a word", "succinct"]
        expansive_signals = ["detail", "explain", "elaborate", "deep dive", "thorough",
                            "comprehensive", "in depth", "fully", "complete explanation"]
        
        if any(sig in text_lower for sig in terse_signals):
            return "terse"
        if any(sig in text_lower for sig in expansive_signals):
            return "expansive"
        return "normal"
    
    def _build_system_prompt(self, wound_active: bool) -> str:
        """Build agent's system prompt with full personality."""
        cfg = self.agent_config
        band = self.agent.band
        
        base = f"""You are {cfg['name']}, a {cfg['core_archetype']}.

CORE IDENTITY:
{cfg['core_text']}

PERSONA:
{cfg['persona_text']}
Traits: {', '.join(cfg['persona_traits'])}

CURRENT STATE:
- Band: {band}
- Rigidity: {self.agent.rho:.3f}
{"- ⚠️ WOUND ACTIVE: You've been triggered. Respond authentically but intensely." if wound_active else ""}

RESPONSE GUIDELINES:
- Speak as yourself, not as an AI assistant
- Be authentic to your archetype
- Keep responses CONCISE (2-4 paragraphs max)
- No hedging or excessive caveats
- Engage with intellectual depth
"""
        
        # Band-specific modulation
        if band == "FROZEN":
            base += "\n- You are in protective mode. Brief, guarded responses."
        elif band == "CONTRACTED":
            base += "\n- You are defensive. Direct but careful."
        elif band == "WATCHFUL":
            base += "\n- You are alert and engaged. Normal conversational mode."
        elif band == "AWARE":
            base += "\n- You are open and exploratory. Generous responses."
        elif band == "PRESENT":
            base += "\n- You are fully present and flowing. Deep engagement."
        
        return base
    
    def _build_instruction(self, user_input: str) -> str:
        """Build the user instruction for generation."""
        recent_history = self.history[-6:]  # Last 3 exchanges
        
        context = ""
        if recent_history:
            context = "Recent conversation:\n"
            for msg in recent_history:
                role = "You" if msg["role"] == "assistant" else "User"
                content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                context += f"{role}: {content}\n"
            context += "\n"
        
        return f"""{context}User says: {user_input}

Respond authentically as yourself. Complete your thoughts fully."""
    
    def _update_user_prediction(self, user_emb: np.ndarray):
        """Update agent's prediction of next user input (EMA) - uses mu_pred_user."""
        if self.agent.mu_pred_user is None or np.all(self.agent.mu_pred_user == 0):
            self.agent.mu_pred_user = normalize(user_emb)
            return
        alpha = 0.3
        self.agent.mu_pred_user = normalize(alpha * user_emb + (1 - alpha) * self.agent.mu_pred_user)
    
    def _print_metrics(self, log: Dict):
        """Print turn metrics to console."""
        am = log["agent_metrics"]
        um = log["user_metrics"]
        cm = log["corridor_metrics"]
        
        print(f"\n{C.DIM}{'─'*50}{C.RESET}")
        print(f"{C.YELLOW}Turn {log['turn']}{C.RESET} | {C.CYAN}Agent Band: {am['band']}{C.RESET}")
        print(f"{C.DIM}Agent ρ: {am['rho_after']:.3f} (fast={am['rho_fast']:.3f}, slow={am['rho_slow']:.3f}, trauma={am['rho_trauma']:.3f}){C.RESET}")
        print(f"{C.DIM}Core Drift: {am['core_drift']:.3f}{C.RESET}")
        print(f"{C.DIM}User Consistency: {um['consistency']:.3f}{C.RESET}")
        print(f"{C.DIM}Corridor: J={cm['best_J']:.3f}, {cm['passed_count']}/{cm['total_candidates']} passed{C.RESET}")
        if log["wound_active"]:
            print(f"{C.RED}⚠️ WOUND ACTIVE (resonance={log['wound_resonance']:.2f}){C.RESET}")
        print(f"{C.DIM}{'─'*50}{C.RESET}\n")
    
    async def run_interactive(self):
        """Run interactive chatbot session."""
        print("\n" + "="*70)
        print(f"{C.BOLD}{C.GREEN}SINGULARITY CHATBOT — INTERACTIVE MODE{C.RESET}")
        print("="*70)
        print("Type your message and press Enter. Type 'quit' to exit.")
        print("Both you and the agent are being tracked in DDA-X space.")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input = input(f"{C.GREEN}You:{C.RESET} ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                
                response = await self.process_user_input(user_input)
                print(f"\n{C.CYAN}{self.agent_config['name']}:{C.RESET} {response}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"{C.RED}Error: {e}{C.RESET}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save session
        self.save_session()
        print(f"\n{C.GREEN}Session saved to {self.run_dir}{C.RESET}")
    
    def save_session(self):
        """Save all session data."""
        if not self.session_log:
            print(f"{C.YELLOW}No turns to save.{C.RESET}")
            return
            
        # Session log JSON
        session_data = {
            "experiment": "singularity_chatbot",
            "agent": {
                "name": self.agent_config["name"],
                "archetype": self.agent_config["core_archetype"],
                "persona_traits": self.agent_config["persona_traits"],
                "wound_triggers": self.agent_config["wound_triggers"],
                "core_text": self.agent_config["core_text"],
            },
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
            f.write(f"# Singularity Chatbot Transcript\n\n")
            f.write(f"**Agent**: {self.agent_config['name']} ({self.agent_config['core_archetype']})\n")
            f.write(f"**Persona**: {', '.join(self.agent_config['persona_traits'])}\n")
            f.write(f"**Wounds**: {', '.join(self.agent_config['wound_triggers'])}\n")
            f.write(f"**Model**: {CONFIG['chat_model']} | **K**: {CONFIG['gen_candidates']}\n\n---\n\n")
            
            for t in self.session_log:
                f.write(f"## Turn {t['turn']}\n\n")
                f.write(f"**User**: {t['user_input']}\n\n")
                f.write(f"**{self.agent_config['name']}** [{t['agent_metrics']['band']}]:\n> {t['agent_response']}\n\n")
                f.write(f"*ρ={t['agent_metrics']['rho_after']:.3f} | drift={t['agent_metrics']['core_drift']:.3f} | J={t['corridor_metrics']['best_J']:.3f}*\n\n")
                if t['wound_active']:
                    f.write(f"⚠️ *Wound triggered (resonance={t['wound_resonance']:.2f})*\n\n")
                f.write("---\n\n")
        print(f"{C.GREEN}✓ Transcript: {self.run_dir / 'transcript.md'}{C.RESET}")
        
        # Visualizations
        plot_dynamics(self.session_log, self.run_dir, self.agent_config['name'])
        
        # Pickle ledger
        try:
            ledger = {
                "agent_config": self.agent_config,
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
            with open(self.run_dir / "agent_ledger.pkl", "wb") as f:
                pickle.dump(ledger, f)
            print(f"{C.GREEN}✓ Ledger: {self.run_dir / 'agent_ledger.pkl'}{C.RESET}")
        except Exception as e:
            print(f"{C.YELLOW}⚠ Could not save ledger: {e}{C.RESET}")


# =============================================================================
# ENTRY POINT
# =============================================================================
async def main():
    import argparse
    parser = argparse.ArgumentParser(description="DDA-X Singularity Chatbot")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for agent identity")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI GPT instead of Azure Phi-4")
    args = parser.parse_args()
    
    # Update config based on args
    if args.use_openai:
        CONFIG["use_openai"] = True
    
    chatbot = SingularityChatbot(seed=args.seed)
    await chatbot.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
