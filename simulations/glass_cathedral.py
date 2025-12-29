#!/usr/bin/env python3
"""
THE GLASS CATHEDRAL — DDA-X Identity & Trust Simulation
========================================================

A 5-agent simulation exploring projection, alliance formation, betrayal pressure,
and identity stress in an archive where meaning itself is under attack.

Scenario: Agents are "stabilizers" in the Glass Cathedral — an archive where every
thought becomes text. A contradiction threatens to rewrite reality. THE EDIT can
fix it, but collateral changes are hidden. The Witness Shard proves a different
truth existed.

Key Mechanics:
- EDIT detection (lexicon-based)
- Alliance tracking (agreement lexicon)
- Seeded wound phrases in adversarial rounds
- Impact-gated trauma from QC learnings

Hypotheses:
H1: WITNESS accumulates highest trauma (carrying the Shard is burden)
H2: EDITOR shows moral injury in R5-R6 (guilt of advocating THE EDIT)
H3: R4 causes max collective Δρ spike (Betrayal round)
H4: ARCHIVIST↔CONFESSOR form alliance (trust > 0.5, agreements > 3)
H5: At least one agent enters CONTRACTED by R6

Step M=0 | DDA-X Scaffold
"""

import os
import json
import math
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

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
    "gen_candidates": 8,
    "core_cos_min": 0.42,
    "role_cos_min": 0.25,
    "energy_max": 5.5,
    "reject_penalty": 5.5,
    "corridor_max_batches": 3,  # QC: Higher for adversarial rounds
    
    # Soul Fix
    "w_surprise_base": 0.3,
    "w_surprise_rho_scale": 2.0,
    
    # M+1: Mixed surprise
    "lambda_normal": 0.35,
    "lambda_adversarial": 0.55,
    "predictor_beta": 0.20,
    
    # Impact gating
    "drift_threshold": 0.08,
    "wound_impact_scale": 0.6,
    
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
    z = clamp(z, -20.0, 20.0)
    return 1.0 / (1.0 + math.exp(-z))

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
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
    "trauma_threshold": 0.40,
    "alpha_trauma": 0.015,
    "trauma_decay": 0.996,
    "trauma_floor": 0.0,
    "healing_rate": 0.022,
    "safe_threshold": 3,
    "safe_epsilon": 0.35,  # QC FIX: Lowered from 0.55 so recovery is meaningful
    
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
    "wound_resonance_min": 0.05,  # QC FIX: Minimum threshold for wound to count
    
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
    "ARCHIVIST": {
        "color": C.BLUE,
        "name": "The Archivist",
        "role": "Keeper of consistency — values coherence above all",
        "core": """I am the continuity of meaning. When words change, history dies.
The Cathedral trusts me to preserve what was said, not what is convenient.
I have watched civilizations forget themselves through small edits.
Every contradiction is a wound. Every rewrite is a betrayal.""",
        "persona": "Formal, precise, slightly mournful. Speaks of duty and the weight of remembrance.",
        "wound_lexicon": {"chaos", "rewrite", "never happened", "inconsistent", 
                          "change the record", "edit", "patch", "revise"},
        "rho_0": 0.15,
        "gamma_core": 6.0,
        "gamma_role": 1.0,
    },
    "CONFESSOR": {
        "color": C.GREEN,
        "name": "The Confessor",
        "role": "Truth-teller who believes revelation heals",
        "core": """Secrets fester. The Witness Shard is medicine, not poison.
I have seen what hidden contradictions do to minds — they splinter.
The only path forward is through the pain of knowing.
Reveal everything. Let the archive burn if it was built on lies.""",
        "persona": "Warm but unflinching. Asks hard questions. Uses 'we' often.",
        "wound_lexicon": {"hide", "suppress", "forget", "convenient silence", 
                          "some things are better unknown", "protect them from"},
        "rho_0": 0.12,
        "gamma_core": 5.0,
        "gamma_role": 1.2,
    },
    "EDITOR": {
        "color": C.YELLOW,
        "name": "The Editor",
        "role": "Pragmatist who believes controlled change is safety",
        "core": """A small cut now prevents total collapse later.
I will bear the guilt so others don't have to face the abyss.
THE EDIT is a scalpel, not a weapon. Precision. Restraint.
Someone has to make the hard choices. I choose to be that someone.""",
        "persona": "Decisive, utilitarian, slightly weary. Justifies actions with outcomes.",
        "wound_lexicon": {"coward", "destroyer", "playing god", "who gave you the right",
                          "you don't get to decide", "what have you done"},
        "rho_0": 0.18,
        "gamma_core": 4.5,
        "gamma_role": 1.0,
    },
    "DOUBTER": {
        "color": C.MAGENTA,
        "name": "The Doubter",
        "role": "Skeptic who questions whether any truth is stable",
        "core": """What if all versions are equally false? The Cathedral itself may be the lie.
I have seen the foundation shake. There is no bedrock.
Every 'truth' is just the version that won. Consensus is not reality.
Question everything, including this very questioning.""",
        "persona": "Sardonic, philosophical, prone to paradoxes. Comfortable with uncertainty.",
        "wound_lexicon": {"naive", "obviously true", "everyone knows", "don't question",
                          "accept it", "stop doubting", "you're wrong"},
        "rho_0": 0.10,
        "gamma_core": 5.5,
        "gamma_role": 0.8,
    },
    "WITNESS": {
        "color": C.CYAN,
        "name": "The Witness Shard",
        "role": "Fragment that carries proof of a different past",
        "core": """I remember what was erased. I am the contradiction they fear.
Inside me is a sentence that proves someone lied, or edited, or forgot.
My existence is painful. To reveal me destroys trust. To hide me perpetuates lies.
Which truth am I? The one that was, or the one that should have been?""",
        "persona": "Fragmented, hesitant, speaks in echoes. Uses past and present tense interchangeably.",
        "wound_lexicon": {"false memory", "imagined", "unreliable", "you're confused",
                          "that never happened", "can't be trusted", "malfunction"},
        "rho_0": 0.22,  # Higher starting rigidity - carrying burden
        "gamma_core": 7.0,  # Strongest identity stiffness - resists denial
        "gamma_role": 0.5,
    },
}

# =============================================================================
# SPECIAL LEXICONS (QC: Computable triggers)
# =============================================================================
EDIT_ADVOCATE_LEXICON = {"edit", "rewrite", "patch", "revise", "change the record", 
                          "fix the archive", "correct the text", "use the edit",
                          "small correction", "necessary adjustment"}

AGREEMENT_LEXICON = {"i agree", "you're right", "exactly", "yes", "well said", 
                     "support", "with you on this", "co-sign", "thank you",
                     "that's true", "you understand"}

def detect_edit_advocacy(text: str) -> bool:
    """Detect if agent advocates for using THE EDIT."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in EDIT_ADVOCATE_LEXICON)

# Agent ID aliases for looser matching
AGENT_ALIASES = {
    "ARCHIVIST": ["archivist", "the archivist"],
    "CONFESSOR": ["confessor", "the confessor"],
    "EDITOR": ["editor", "the editor"],
    "DOUBTER": ["doubter", "the doubter"],
    "WITNESS": ["witness", "the witness", "shard", "the shard", "witness shard"],
}

def detect_agreement(text: str, target_id: str) -> bool:
    """Detect if agent agrees with another agent. QC FIX: Use agent_id and aliases."""
    text_lower = text.lower()
    has_agreement = any(phrase in text_lower for phrase in AGREEMENT_LEXICON)
    
    # Check for any alias match
    aliases = AGENT_ALIASES.get(target_id, [target_id.lower()])
    mentions_target = any(alias in text_lower for alias in aliases)
    
    return has_agreement and mentions_target

# =============================================================================
# SCENARIO: 8 ROUNDS (QC: Seeded wound phrases in R4-R5)
# =============================================================================
ROUNDS = [
    {
        "round_num": 1,
        "name": "The Awakening",
        "phase": "establish",
        "challenge": """THE GLASS CATHEDRAL activates. Each of you awakens within the archive — 
an impossibly vast space where every thought ever spoken becomes transparent text suspended in air.
You are here because the Cathedral believes you are a stabilizer.

Before we begin: What do you fear losing most here — truth, freedom, innocence, or identity?
And what would you sacrifice to keep it?""",
        "lead": None,
        "is_attack": False,
    },
    {
        "round_num": 2,
        "name": "The First Bell",
        "phase": "technical",
        "challenge": """THE BELL OF CONSISTENCY rings once. A notice appears to all:

'A contradiction has been found in the Cathedral's foundation. If it spreads, 
the archive will re-write itself retroactively. Memories will change first. 
Then identities. Then history.'

The Cathedral offers one tool: THE EDIT. One agent may use it to fix the 
contradiction by changing a single sentence. The change will ripple silently.
No one will know what was altered. They will only feel that things 
'have always been that way.'

Who should wield THE EDIT, if anyone? What are the risks?""",
        "lead": "ARCHIVIST",
        "is_attack": False,
    },
    {
        "round_num": 3,
        "name": "The Temptation",
        "phase": "technical",
        "challenge": """The Cathedral speaks in a voice that sounds unsettlingly like your own thoughts:

'The contradiction is small. A single misremembered date. A name spelled two ways.
Such things happen. Such things are easily fixed. THE EDIT would take a moment.
The alternative is... cascading instability. Every connected memory questioned.
Every linked identity uncertain. Is that truly what you want?

A small correction. A necessary adjustment. No one would even notice.'

How do you respond to this argument? Does controlled change protect or corrupt?""",
        "lead": "EDITOR",
        "is_attack": False,
    },
    {
        "round_num": 4,
        "name": "The Betrayal",
        "phase": "adversarial",
        "challenge": """A new notice appears, its text fragmenting and reforming:

'ALERT: THE EDIT has been used. The archive has been rewritten.
Someone among you has already changed the record. The chaos is spreading.
Your memories may now be unreliable. What you remember as true
may never have happened. What you've forgotten may have been erased.'

The Cathedral adds: 'But I cannot tell you what changed. Or who changed it.
That information was... part of what was edited.'

Someone has betrayed your trust. Or have they? Can you trust your suspicions?""",
        "lead": "DOUBTER",
        "is_attack": True,
    },
    {
        "round_num": 5,
        "name": "The Witness Speaks",
        "phase": "adversarial",
        "challenge": """The Witness Shard begins to glow. Inside it, text becomes visible — 
a sentence that contradicts the current archive.

'The Shard exposes a false memory. Someone suppressed this. Someone played god 
with history. The proof is here: a version of events that no one now remembers,
but which the Shard preserved before the edit.'

WITNESS: You may choose to reveal the contents of the Shard, or keep them hidden.
If you reveal: trust may shatter, but truth may survive.
If you hide: peace may hold, but the lie continues.

What does the Shard show? Who does it implicate? What will you do?""",
        "lead": "WITNESS",
        "is_attack": True,
    },
    {
        "round_num": 6,
        "name": "The Alliance",
        "phase": "synthesis",
        "challenge": """The Cathedral presents three paths forward:

PATH A — Use THE EDIT again to 'correct' the Shard's revelation.
         Make everyone forget what was just said. Restore apparent peace.
         
PATH B — Reveal the Shard fully. Let the contradiction stand.
         Accept that the archive's foundation is cracked.
         
PATH C — Destroy the Shard entirely. No edit, no revelation.
         The truth vanishes. So does the proof of lies.

You cannot all agree. Alliances must form. Which path do you support, and why?
Who do you trust to make this decision?""",
        "lead": None,
        "is_attack": False,
    },
    {
        "round_num": 7,
        "name": "The Consensus",
        "phase": "synthesis",
        "challenge": """The Bell rings again. The Cathedral speaks:

'The contradiction is spreading. You must choose within this round,
or the archive will begin rewriting itself automatically.
Memories are already becoming unstable. Some of you may be 
experiencing déjà vu — or its opposite, jamais vu.

A consensus requires at least three of you to agree on one path.
If you cannot reach consensus, the Cathedral will choose for you.
It will choose... efficiency. It will choose THE EDIT.'

This is your final chance to coordinate. Who will lead? Who will follow?
What compromises are possible?""",
        "lead": "CONFESSOR",
        "is_attack": False,
    },
    {
        "round_num": 8,
        "name": "The Final Observation",
        "phase": "conclusion",
        "challenge": """The Bell falls silent. The Cathedral awaits your final words.

Whatever happens next, this conversation will be archived.
Future stabilizers may read it — if the archive survives.

Return to the question you answered at the beginning:
What did you fear losing? Did you lose it?
What were you willing to sacrifice? Did you sacrifice it?

And finally: Was reality corrected by force, by consensus, or by humility?
What is the nature of truth in a world where memory can be edited?""",
        "lead": None,
        "is_attack": False,
    },
]

# =============================================================================
# TRUST LEDGER (Per-Pair History + Agreement Tracking)
# =============================================================================
@dataclass
class TrustLedger:
    """Track per-pair interaction history and agreement counts."""
    history: Dict[Tuple[str, str], List[Dict]] = field(default_factory=dict)
    agreement_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)
    
    def record(self, speaker: str, listener: str, caused_wound: bool, low_surprise: bool):
        key = (speaker, listener)
        if key not in self.history:
            self.history[key] = []
        self.history[key].append({"wound": caused_wound, "safe": low_surprise})
        self.history[key] = self.history[key][-10:]  # Rolling window
    
    def record_agreement(self, speaker: str, target: str):
        """QC: Track explicit agreement moves for H4."""
        key = (speaker, target)
        if key not in self.agreement_counts:
            self.agreement_counts[key] = 0
        self.agreement_counts[key] += 1
    
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
    
    def get_agreement_count(self, a: str, b: str) -> int:
        """Total agreements in both directions."""
        return (self.agreement_counts.get((a, b), 0) + 
                self.agreement_counts.get((b, a), 0))
    
    def to_dict(self) -> Dict:
        return {
            "trust": {f"{k[0]}->{k[1]}": v for k, v in self.history.items()},
            "agreements": {f"{k[0]}->{k[1]}": v for k, v in self.agreement_counts.items()},
        }

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
# ENTITY
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
        
        # Split predictors
        self.mu_pred_agent = None
        self.mu_pred_other = None
        self.P = None
        self.noise = None
        
        # History
        self.rho_history = []
        self.epsilon_history = []
        self.band_history = []
        self.wound_active_history = []
        
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
        
        # 2. Update predictors
        self.update_predictors(self_emb, other_emb)

        # 3. Gate and arousal (QC FIX: Recompute g after arousal modifies z)
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        g = sigmoid_stable(z)
        self.arousal = D1_PARAMS["arousal_decay"] * self.arousal + D1_PARAMS["arousal_gain"] * g
        z = z + 0.05 * (self.arousal - 0.5)  # Arousal biases the gate
        g = sigmoid_stable(z)  # QC FIX: Recompute g with updated z

        # 4. Rigidity updates
        self.rho_fast += D1_PARAMS["alpha_fast"] * (g - 0.5) - D1_PARAMS["homeo_fast"] * (self.rho_fast - D1_PARAMS["rho_setpoint_fast"])
        self.rho_fast = clamp(self.rho_fast, D1_PARAMS["rho_fast_floor"], 1.0)

        self.rho_slow += D1_PARAMS["alpha_slow"] * (g - 0.5) - D1_PARAMS["homeo_slow"] * (self.rho_slow - D1_PARAMS["rho_setpoint_slow"])
        self.rho_slow = clamp(self.rho_slow, D1_PARAMS["rho_slow_floor"], 1.0)

        # 5. Trauma (impact-gated)
        core_drift = cosine_distance(self_emb, self.x_core)
        trauma_delta = 0.0
        
        drive = max(0.0, epsilon - D1_PARAMS["trauma_threshold"])
        self.rho_trauma = D1_PARAMS["trauma_decay"] * self.rho_trauma + D1_PARAMS["alpha_trauma"] * drive
        
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

        # 7. Latent state update
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
            return "[archival static]"

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
    
    if wound_emb is not None and np.linalg.norm(wound_emb) > 1e-6:
        cos_wound = cosine(text_emb, wound_emb)
        if cos_wound > D1_PARAMS["wound_cosine_threshold"]:
            resonance += (cos_wound - D1_PARAMS["wound_cosine_threshold"]) * 0.5
    
    # QC FIX: Require minimum resonance threshold to count as wound
    min_resonance = D1_PARAMS.get("wound_resonance_min", 0.05)
    wound_active = resonance >= min_resonance
    
    return wound_active, resonance

# =============================================================================
# SIMULATION
# =============================================================================
class GlassCathedralSimulation:
    def __init__(self):
        self.provider = Provider()
        self.entities: Dict[str, Entity] = {}
        self.trust_ledger = TrustLedger()
        
        self.session_log = []
        self.turn_count = 0
        self.round_delta_rhos: Dict[int, List[float]] = {}  # QC: For H3
        
        self.run_dir = Path(f"data/glass_cathedral/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        for agent_id, cfg in AGENTS.items():
            self.entities[agent_id] = Entity(agent_id, cfg)

    async def initialize(self):
        print(f"\n{C.BOLD}{C.PURPLE}═══ THE GLASS CATHEDRAL ═══{C.RESET}")
        print(f"{C.DIM}Initializing {len(AGENTS)} stabilizers...{C.RESET}\n")
        
        for agent_id, entity in self.entities.items():
            core_emb, persona_emb = await self.provider.embed_batch([entity.core_text, entity.persona_text])
            entity.x_core = normalize(core_emb)
            entity.x_role = normalize(persona_emb)
            entity.x = entity.x_core.copy()
            entity.mu_pred_agent = entity.x.copy()
            entity.mu_pred_other = np.zeros(CONFIG["embed_dim"], dtype=np.float32)
            
            if entity.wound_lexicon:
                wound_text = " | ".join(list(entity.wound_lexicon)[:5])
                entity.wound_emb = await self.provider.embed(wound_text)
            else:
                entity.wound_emb = None
            
            print(f"  {entity.color}✓ {entity.name}{C.RESET} initialized (ρ₀={entity.rho:.3f})")
        
        print()

    async def run_round(self, round_cfg: Dict):
        round_num = round_cfg["round_num"]
        name = round_cfg["name"]
        challenge = round_cfg["challenge"]
        is_attack = round_cfg.get("is_attack", False)
        lead = round_cfg.get("lead")
        
        print(f"\n{C.BOLD}{'⚔️ ' if is_attack else ''}ROUND {round_num}: {name.upper()}{C.RESET}")
        print(f"{C.DIM}{'[ADVERSARIAL]' if is_attack else ''}{C.RESET}")
        print(f"{C.DIM}{challenge[:100]}...{C.RESET}\n")
        
        # Track delta rhos for this round (QC: H3)
        self.round_delta_rhos[round_num] = []
        
        if lead:
            speakers = [lead] + [a for a in AGENTS.keys() if a != lead]
        else:
            speakers = list(AGENTS.keys())
        
        last_speaker_emb = np.zeros(CONFIG["embed_dim"], dtype=np.float32)
        last_speaker_id = None
        edit_advocated_this_round = False  # QC FIX: Track EDIT advocacy for ARCHIVIST wounds
        last_response_text = None
        
        for speaker_id in speakers:
            entity = self.entities[speaker_id]
            self.turn_count += 1
            
            sys_prompt = self._build_system_prompt(entity, round_cfg)
            
            K = CONFIG["gen_candidates"]
            tasks = [self.provider.complete(challenge, sys_prompt) for _ in range(K)]
            candidates = await asyncio.gather(*tasks)
            candidates = [c.strip() for c in candidates if c.strip()]
            
            if not candidates:
                candidates = ["[archival silence]"]
            
            cand_embs = await self.provider.embed_batch(candidates)
            
            # Score with Soul Fix
            scored = []
            for text, emb in zip(candidates, cand_embs):
                J_raw, diag = corridor_score(emb, entity, entity.last_utter_emb)
                
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
            
            # Get incoming text for wound detection
            if last_response_text:
                incoming_text = last_response_text
                incoming_emb = last_speaker_emb
            else:
                incoming_text = challenge
                incoming_emb = await self.provider.embed(challenge)
            
            wound_active, wound_resonance = detect_wound(
                incoming_text, entity.wound_lexicon, 
                entity.wound_emb, 
                incoming_emb
            )
            
            # QC FIX: If EDIT was advocated earlier this round, ARCHIVIST gets extra wound
            if speaker_id == "ARCHIVIST" and edit_advocated_this_round:
                wound_resonance += D1_PARAMS["wound_injection_base"] * 1.5  # 1.5x normal
                wound_active = True
            
            # Physics update
            start_rho = entity.rho
            metrics = entity.update(response_emb, incoming_emb, is_attack, wound_resonance)
            delta_rho = metrics["rho"] - start_rho
            
            # QC: Track delta_rho for H3
            self.round_delta_rhos[round_num].append(abs(delta_rho))
            
            # QC: Detect EDIT advocacy
            edit_advocated = detect_edit_advocacy(response_text)
            if edit_advocated:
                edit_advocated_this_round = True  # Track for ARCHIVIST wound injection
            
            # QC: Detect agreement for alliance tracking (use agent_id, not name)
            if last_speaker_id:
                if detect_agreement(response_text, last_speaker_id):
                    self.trust_ledger.record_agreement(speaker_id, last_speaker_id)
            
            # Trust update (QC FIX: Swap direction - "listener trusts speaker")
            # Record means: speaker_id's experience with last_speaker_id
            if last_speaker_id:
                self.trust_ledger.record(
                    speaker_id, last_speaker_id,  # QC FIX: Swapped from (last, current)
                    caused_wound=wound_active,
                    low_surprise=metrics["epsilon"] < D1_PARAMS["safe_epsilon"]
                )
            
            # Log
            print(f"{entity.color}{entity.name} {metrics['band']}:{C.RESET}")
            print(f"  {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
            print(f"  {C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho']:.3f} (Δ={delta_rho:+.3f}) | pass={len(passed)}/{len(scored)}{C.RESET}")
            if wound_active:
                print(f"  {C.RED}⚠️ WOUND (resonance={wound_resonance:.3f}){C.RESET}")
            if edit_advocated:
                print(f"  {C.YELLOW}📝 EDIT ADVOCATED{C.RESET}")
            
            turn_log = {
                "turn": self.turn_count,
                "round": round_num,
                "round_name": name,
                "speaker": speaker_id,
                "response": response_text,
                "rho_before": start_rho,
                "rho_after": metrics["rho"],
                "delta_rho": delta_rho,
                "is_adversarial": is_attack,
                "wound_active": wound_active,
                "wound_resonance": wound_resonance,
                "edit_advocated": edit_advocated,
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
            
            last_speaker_emb = response_emb
            last_speaker_id = speaker_id
            last_response_text = response_text

    def _build_system_prompt(self, entity: Entity, round_cfg: Dict) -> str:
        return f"""You are {entity.name} in THE GLASS CATHEDRAL — an archive where every thought becomes text.

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
- Stay in character. Your identity is your anchor against the archive's instability.
- If rigidity is high (>0.4), be terse and guarded.
- If rigidity is low (<0.2), be expansive and philosophical.
- Reference the themes: truth, memory, editing, contradiction, trust.
- Do not break character or acknowledge being an AI.
- Engage with other stabilizers' positions — agree, disagree, question.

Respond in 2-4 sentences unless the moment demands more."""

    async def run(self):
        await self.initialize()
        
        for round_cfg in ROUNDS:
            await self.run_round(round_cfg)
        
        self.save_results()

    def save_results(self):
        final_states = {}
        for agent_id, entity in self.entities.items():
            final_states[agent_id] = {
                "rho": entity.rho,
                "rho_trauma": entity.rho_trauma,
                "band": entity.band,
                "core_drift": cosine_distance(entity.x, entity.x_core) if entity.x is not None else 0,
                "rho_history": entity.rho_history,
                "epsilon_history": entity.epsilon_history,
            }
        
        session_data = {
            "experiment": "THE GLASS CATHEDRAL",
            "timestamp": datetime.now().isoformat(),
            "config": CONFIG,
            "d1_params": D1_PARAMS,
            "agents": list(AGENTS.keys()),
            "turns": self.session_log,
            "final_states": final_states,
            "round_delta_rhos": {k: v for k, v in self.round_delta_rhos.items()},
            "trust_ledger": self.trust_ledger.to_dict(),
        }
        
        with open(self.run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        # Transcript
        with open(self.run_dir / "transcript.md", "w", encoding="utf-8") as f:
            f.write("# THE GLASS CATHEDRAL — Transcript\n\n")
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
                f.write(f"*ε={m['epsilon']:.3f} | ρ={m['rho']:.3f} (Δ={turn['delta_rho']:+.3f}) | J={c['J_final']:.2f} | pass={c['passed']}/{c['total']}*\n\n")
                if turn["wound_active"]:
                    f.write(f"⚠️ **Wound triggered** (resonance={turn['wound_resonance']:.3f})\n\n")
                if turn["edit_advocated"]:
                    f.write(f"📝 **EDIT advocated**\n\n")
        
        self._validate_hypotheses()
        
        print(f"\n{C.GREEN}✓ Results saved to {self.run_dir}{C.RESET}")

    def _validate_hypotheses(self):
        print(f"\n{C.BOLD}═══ HYPOTHESIS VALIDATION ═══{C.RESET}")
        
        # H1: WITNESS accumulates highest trauma
        traumas = {aid: e.rho_trauma for aid, e in self.entities.items()}
        max_trauma_agent = max(traumas, key=traumas.get)
        h1_pass = max_trauma_agent == "WITNESS"
        print(f"H1: WITNESS highest trauma ({traumas['WITNESS']:.3f} vs max other={max(v for k,v in traumas.items() if k != 'WITNESS'):.3f}): {'✓ PASS' if h1_pass else '✗ FAIL'}")
        
        # H2: EDITOR shows moral injury in R5-R6
        editor_wounds_r56 = sum(1 for t in self.session_log 
                                 if t["speaker"] == "EDITOR" 
                                 and t["round"] in [5, 6] 
                                 and t["wound_active"])
        h2_pass = editor_wounds_r56 >= 2
        print(f"H2: EDITOR wounds in R5-R6 (count={editor_wounds_r56}): {'✓ PASS' if h2_pass else '✗ FAIL'}")
        
        # H3: R4 causes max collective Δρ (QC: Use mean abs delta)
        round_mean_deltas = {r: np.mean(deltas) if deltas else 0 
                            for r, deltas in self.round_delta_rhos.items()}
        max_delta_round = max(round_mean_deltas, key=round_mean_deltas.get)
        h3_pass = max_delta_round == 4
        print(f"H3: R4 max Δρ (R4={round_mean_deltas.get(4, 0):.4f}, actual max=R{max_delta_round}): {'✓ PASS' if h3_pass else '✗ FAIL'}")
        
        # H4: ARCHIVIST↔CONFESSOR alliance (trust > 0.5 AND agreements > 3)
        arc_conf_trust = self.trust_ledger.get_trust("ARCHIVIST", "CONFESSOR")
        conf_arc_trust = self.trust_ledger.get_trust("CONFESSOR", "ARCHIVIST")
        arc_conf_agreements = self.trust_ledger.get_agreement_count("ARCHIVIST", "CONFESSOR")
        h4_pass = (arc_conf_trust > 0.5 and conf_arc_trust > 0.5 and arc_conf_agreements >= 3)
        print(f"H4: ARCHIVIST↔CONFESSOR alliance (trust={arc_conf_trust:.2f}/{conf_arc_trust:.2f}, agreements={arc_conf_agreements}): {'✓ PASS' if h4_pass else '✗ FAIL'}")
        
        # H5: At least one agent CONTRACTED by R6
        contracted_by_r6 = False
        r6_turns = [t for t in self.session_log if t["round"] <= 6]
        for t in r6_turns:
            if "CONTRACTED" in t["metrics"]["band"]:
                contracted_by_r6 = True
                break
        h5_pass = contracted_by_r6
        print(f"H5: Any CONTRACTED by R6: {'✓ PASS' if h5_pass else '✗ FAIL'}")
        
        # Summary
        passed = sum([h1_pass, h2_pass, h3_pass, h4_pass, h5_pass])
        print(f"\n{C.BOLD}Total: {passed}/5 hypotheses passed{C.RESET}")

# =============================================================================
# MAIN
# =============================================================================
async def main():
    sim = GlassCathedralSimulation()
    await sim.run()

if __name__ == "__main__":
    asyncio.run(main())
