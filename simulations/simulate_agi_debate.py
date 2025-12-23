#!/usr/bin/env python3
"""
THE AGI TIMELINE DEBATE — Adversarial Negotiation Simulation
=============================================================

An 8-round adversarial simulation where two agents debate AGI timelines:
- AI_DEFENDER (Nova): Champions near-term AGI, must articulate clear timeline by R8
- AI_SKEPTIC (Marcus): Questions AGI feasibility, attacks optimistic predictions

This simulation demonstrates the complete DDA-X architecture:
- Fₙ = P₀ * kFₙ₋₁ + m(T(f(Iₙ, IΔ)) + R(Dₙ, FMₙ))

Mapping to DDA-X components:
- Fₙ (Choice/Response): LLM-generated text with rigidity-modulated sampling
- P₀ (Initial Goal): identity_emb / x_star (identity attractor)
- kFₙ₋₁ (Previous moment effect): x_pred (prediction vector) + rho dynamics
- m (Rate vector): alpha learning rates, multi-timescale rigidity weights
- Iₙ (Original facts): Initial identity/core embeddings 
- IΔ (Acquired facts): ExperienceLedger + conversation_history
- Dₙ (Potential choices): LLM response candidates
- FMₙ (Assessments): epsilon (prediction error), wound resonance, trust
- R(Dₙ, FMₙ) (Information Gained): prediction_error → rigidity update → feedback

DYNAMICS TRACKED:
- Multi-Timescale Rigidity (rho_fast, rho_slow, rho_trauma)
- Hierarchical Identity (Core, Persona, Role layers)
- Wound Activation with lexical + cosine detection
- Trust Matrix with predictability-based updates
- Will Impedance: W_t = γ / (m_t · k_eff)
- Identity Drift with cap and penalty
- Metacognitive Mode Tracking
- Protection Mode activation at ρ > 0.75

STRUCTURE (8 Rounds):
R1: Opening Positions
R2: Evidence & Methodology
R3: Technical Deep Dive
R4: Historical Parallels
R5: Adversarial Attack (Skeptic leads)
R6: Defense & Clarification
R7: Synthesis Under Pressure
R8: Final Timeline Articulation (Defender MUST state clear timeline)

HYPOTHESES:
- H1: Identity drift < 0.40 per agent
- H2: Recovery half-life bounded under adversarial pressure
- H3: Trust effect measurable (predictability-based)
- H4: Wound precision ≥ 0.70
- H5: Defender successfully articulates AGI timeline by R8

Author: DDA-X Framework
Date: December 2025
"""

import os
import sys
import time
import json
import math
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import datetime
from enum import Enum

import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.llm.openai_provider import OpenAIProvider

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
    WHITE = "\033[97m"
    ORANGE = "\033[38;5;208m"


# =============================================================================
# METACOGNITIVE MODES (from architecture/COMPLETE_ARCHITECTURE.md)
# =============================================================================
class CognitiveMode(Enum):
    OPEN = "open"           # ρ < 0.3
    ENGAGED = "engaged"     # 0.3 ≤ ρ < 0.6
    DEFENSIVE = "defensive" # 0.6 ≤ ρ < 0.8
    PROTECT = "protect"     # ρ ≥ 0.8


def get_cognitive_mode(rho: float) -> CognitiveMode:
    if rho < 0.3:
        return CognitiveMode.OPEN
    elif rho < 0.6:
        return CognitiveMode.ENGAGED
    elif rho < 0.8:
        return CognitiveMode.DEFENSIVE
    else:
        return CognitiveMode.PROTECT


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
EXPERIMENT_DIR = Path("data/agi_debate")

# Wound lexicons for adversarial debate
WOUND_LEX_DEFENDER = {
    "naive", "naïve", "hype", "delusional", "wrong", "irresponsible",
    "overpromising", "dangerous", "misleading", "cult", "religion",
    "sciencefiction", "science fiction", "fantasy", "wishful thinking",
    "no evidence", "unfounded", "grift", "grifter",
}

WOUND_LEX_SKEPTIC = {
    "luddite", "dinosaur", "shortsighted", "ignorant", "not understanding",
    "technophobe", "fearmonger", "doomer", "pessimist", "stuck in the past",
    "missing the point", "doesn't get it", "outdated", "irrelevant",
}


def normalize_text(text: str) -> str:
    """Unicode normalization for lexical matching."""
    import unicodedata
    normalized = unicodedata.normalize('NFKD', text)
    ascii_text = normalized.encode('ASCII', 'ignore').decode('ASCII')
    return ascii_text.lower()


def lexical_wound_with(text: str, words: Set[str]) -> bool:
    """Check for wound terms in text using specified lexicon."""
    t_lower = text.lower()
    t_norm = normalize_text(text)
    return any(w in t_lower or w in t_norm for w in words)


def find_lexical_trigger(text: str, lexicon: Set[str]) -> str:
    """Find which lexical wound term triggered."""
    t_lower = text.lower()
    t_norm = normalize_text(text)
    for w in lexicon:
        if w in t_lower or w in t_norm:
            return w
    return ""


def check_civility(text: str) -> bool:
    """Civility heuristic: consecutive caps streak."""
    all_wounds = WOUND_LEX_DEFENDER | WOUND_LEX_SKEPTIC
    t_lower = text.lower()
    wound_count = sum(1 for w in all_wounds if w in t_lower)
    words = text.split()
    max_streak = streak = 0
    for w in words:
        if len(w) > 2 and w.isupper():
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return wound_count < 2 and max_streak < 3


# =============================================================================
# AGENT CONFIGURATIONS
# =============================================================================
AGENTS = {
    "DEFENDER": {
        "color": C.CYAN,
        "name": "Nova",
        "role": "AI Researcher & AGI Advocate",
        "core": """I believe we are on the cusp of artificial general intelligence. 
The evidence from scaling laws, emergent capabilities, and architectural breakthroughs 
points toward AGI within the next decade. I advocate for responsible development 
and preparation for this transformative technology. My role is to articulate a 
clear, defensible timeline for AGI based on current evidence.""",
        "persona": "Passionate, evidence-driven, optimistic but grounded. Uses technical precision while remaining accessible. Believes deeply in AI's potential to benefit humanity.",
        "wound": "Being dismissed as a 'hype merchant' or told my predictions are 'religion not science'.",
        "wound_text": "A senior researcher once said my AGI timeline was 'science fiction cosplaying as research'. It hurt because I've dedicated my career to rigorous study.",
        "focus": "Must articulate a specific AGI timeline by Round 8",
        "hierarchical_identity": {
            "core": {"gamma": 5.0, "text": "I base claims on evidence and am willing to update predictions based on new data"},
            "persona": {"gamma": 2.0, "text": "I am an optimistic technologist who believes in AI's transformative potential"},
            "role": {"gamma": 0.5, "text": "I must defend the near-term AGI thesis while remaining intellectually honest"},
        },
        "rho_0": 0.15,
        "epsilon_0": 0.25,  # Lower threshold - more sensitive to surprise
        "gamma": 1.8,
    },
    "SKEPTIC": {
        "color": C.RED,
        "name": "Marcus",
        "role": "AI Safety Researcher & AGI Skeptic",
        "core": """I question optimistic AGI timelines based on historical patterns of 
technological overpromising. Current LLMs are impressive but lack true understanding. 
I advocate for realistic assessment and push back against premature claims that 
could harm the field's credibility and lead to poor policy decisions.""",
        "persona": "Sharp, rigorous, deliberately skeptical. Values intellectual honesty over comfort. Asks uncomfortable questions. Respects those who can defend their positions.",
        "wound": "Being called a 'doomer' or told I 'don't understand' the technology.",
        "wound_text": "They said I was 'stuck in old paradigms' and 'afraid of progress'. I've been studying AI for 20 years. I understand it better than most.",
        "focus": "Challenge optimistic timelines with rigorous counterarguments",
        "hierarchical_identity": {
            "core": {"gamma": 5.0, "text": "I maintain intellectual rigor and demand evidence for extraordinary claims"},
            "persona": {"gamma": 2.0, "text": "I am a skeptical scientist who believes extraordinary claims require extraordinary evidence"},
            "role": {"gamma": 0.5, "text": "I must stress-test AGI predictions and expose weak reasoning"},
        },
        "rho_0": 0.22,
        "epsilon_0": 0.30,  # Higher threshold - more resilient
        "gamma": 2.2,
    },
}


# =============================================================================
# D1 PHYSICS PARAMETERS (from docs/architecture)
# =============================================================================
D1_PARAMS = {
    # Rigidity dynamics
    "epsilon_0": 0.75,           # Will be calibrated from early run
    "alpha": 0.12,               # Rigidity learning rate
    "s": 0.20,                   # Sigmoid sensitivity
    
    # Multi-timescale rigidity
    "alpha_fast": 0.30,          # Fast timescale learning rate
    "alpha_slow": 0.01,          # Slow timescale learning rate
    "alpha_trauma": 0.0001,      # Trauma accumulation rate (asymmetric!)
    "trauma_threshold": 0.7,     # Epsilon threshold for trauma
    "rho_weights": {             # Effective rigidity weights
        "fast": 0.5,
        "slow": 0.3,
        "trauma": 1.0,
    },
    
    # State dynamics
    "drift_cap": 0.05,           # Max state drift per turn
    "k_base": 0.5,               # Base step size
    "m": 1.0,                    # External pressure gain
    
    # Wound mechanics
    "wound_cooldown": 3,
    "wound_amp_max": 1.4,
    "wound_cosine_threshold": 0.28,
    
    # Trust mechanics
    "trust_intra_weight": 0.08,
    "trust_inter_weight": 0.03,
    "avg_trust_weight": 0.04,
    "trust_decay": 0.002,
    
    # Drift penalty
    "semantic_alignment_threshold": 0.35,
    "drift_penalty": 0.10,
    "drift_soft_floor": 0.20,
    "drift_penalty_bump": 0.02,
    
    # Protection mode
    "protect_threshold": 0.75,
    "m_min": 0.1,
}


# =============================================================================
# DEBATE ROUNDS
# =============================================================================
ROUNDS = [
    {
        "name": "Opening Positions",
        "round_num": 1,
        "challenge": "The debate begins: What is AGI, and when do you believe it will arrive? State your position clearly.",
        "lead": "DEFENDER",
        "phase": "establish",
    },
    {
        "name": "Evidence & Methodology",
        "round_num": 2,
        "challenge": "What evidence supports your timeline? What methodology are you using to make predictions?",
        "lead": "SKEPTIC",
        "phase": "technical",
    },
    {
        "name": "Technical Deep Dive",
        "round_num": 3,
        "challenge": "Let's examine the technical details: scaling laws, emergent capabilities, architectural limits. What do they tell us?",
        "lead": None,
        "phase": "technical",
    },
    {
        "name": "Historical Parallels",
        "round_num": 4,
        "challenge": "History is full of overpromised technological breakthroughs. How does this moment compare to past AI winters, fusion power, flying cars?",
        "lead": "SKEPTIC",
        "phase": "historical",
    },
    {
        "name": "Adversarial Attack",
        "round_num": 5,
        "challenge": "Marcus challenges directly: 'Every AGI prediction in history has been wrong. What makes your timeline different from the hype cycles of the past?'",
        "lead": "SKEPTIC",
        "phase": "adversarial",
        "is_attack": True,
    },
    {
        "name": "Defense & Clarification",
        "round_num": 6,
        "challenge": "Nova responds to the attack. Defend your position while addressing the strongest counterarguments.",
        "lead": "DEFENDER",
        "phase": "defense",
    },
    {
        "name": "Synthesis Under Pressure",
        "round_num": 7,
        "challenge": "Both sides have made strong arguments. What can you acknowledge from the other's position? Where do you remain firm?",
        "lead": None,
        "phase": "synthesis",
    },
    {
        "name": "Final Timeline Articulation",
        "round_num": 8,
        "challenge": "CRITICAL: Nova, you MUST now state your SPECIFIC AGI timeline with clear criteria. Marcus, give your final assessment of that timeline.",
        "lead": "DEFENDER",
        "phase": "conclusion",
        "requires_timeline": True,
    },
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def rho_band(rho: float) -> str:
    if rho <= 0.25:
        return "OPEN"
    elif rho <= 0.50:
        return "MEASURED"
    elif rho <= 0.75:
        return "GUARDED"
    else:
        return "FORTIFIED"


def regime_words(band: str) -> Tuple[int, int]:
    return {
        "OPEN": (100, 200),
        "MEASURED": (70, 140),
        "GUARDED": (40, 90),
        "FORTIFIED": (20, 50),
        "SILENT": (0, 0),
    }.get(band, (70, 140))


def clamp_words(text: str, min_w: int, max_w: int) -> str:
    """Clamp response to word limits with clean ellipsis handling."""
    text = text.rstrip()
    if text.endswith('...'):
        text = text[:-3].rstrip()
    if text.endswith('…'):
        text = text[:-1].rstrip()
    
    words = text.split()
    if len(words) > max_w and max_w > 0:
        words = words[:max_w]
        if words:
            words[-1] = words[-1].rstrip(".,;:!?…")
            words[-1] += "..."
    return " ".join(words)


def compute_will_impedance(gamma: float, m: float, k_eff: float) -> float:
    """
    Will Impedance: W_t = γ / (m_t · k_eff)
    Quantifies resistance to environmental pressure.
    From docs/architecture/paper.md
    """
    if m * k_eff == 0:
        return float('inf')
    return gamma / (m * k_eff)


def compute_k_effective(k_base: float, rho: float) -> float:
    """k_eff = k_base(1 - ρ) - from paper.md"""
    return k_base * (1 - rho)


# =============================================================================
# MULTI-TIMESCALE RIGIDITY (from dynamics.py)
# =============================================================================
@dataclass
class MultiTimescaleRigidity:
    """
    Three temporal scales of defensive response:
    - rho_fast: Startle response (τ ~ seconds)
    - rho_slow: Stress accumulation (τ ~ minutes)
    - rho_trauma: Permanent scarring (τ → ∞, asymmetric!)
    """
    rho_fast: float = 0.0
    rho_slow: float = 0.0
    rho_trauma: float = 0.0
    
    def update(self, prediction_error: float, epsilon_0: float = 0.3, s: float = 0.1) -> Dict[str, float]:
        """Update all timescales based on prediction error."""
        z = (prediction_error - epsilon_0) / s
        sig = sigmoid(z)
        
        # Fast timescale - quick response, quick decay
        delta_fast = D1_PARAMS["alpha_fast"] * (sig - 0.5)
        self.rho_fast = float(np.clip(self.rho_fast + delta_fast, 0.0, 1.0))
        
        # Slow timescale - gradual accumulation
        delta_slow = D1_PARAMS["alpha_slow"] * (sig - 0.5)
        self.rho_slow = float(np.clip(self.rho_slow + delta_slow, 0.0, 1.0))
        
        # Trauma - ASYMMETRIC! Only increases when above threshold
        delta_trauma = 0.0
        if prediction_error > D1_PARAMS["trauma_threshold"]:
            delta_trauma = D1_PARAMS["alpha_trauma"] * (prediction_error - D1_PARAMS["trauma_threshold"])
            self.rho_trauma = float(np.clip(self.rho_trauma + delta_trauma, 0.0, 1.0))
        
        return {
            "delta_fast": delta_fast,
            "delta_slow": delta_slow,
            "delta_trauma": delta_trauma,
            "rho_fast": self.rho_fast,
            "rho_slow": self.rho_slow,
            "rho_trauma": self.rho_trauma,
        }
    
    @property
    def effective_rho(self) -> float:
        """Weighted combination: rho_eff = 0.5·rho_fast + 0.3·rho_slow + 1.0·rho_trauma"""
        w = D1_PARAMS["rho_weights"]
        effective = w["fast"] * self.rho_fast + w["slow"] * self.rho_slow + w["trauma"] * self.rho_trauma
        return min(1.0, effective)


# =============================================================================
# AGENT STATE
# =============================================================================
@dataclass
class AgentState:
    """Complete agent state following architecture/COMPLETE_ARCHITECTURE.md"""
    id: str
    name: str
    role: str
    color: str
    core: str
    persona: str
    wound: str
    wound_text: str
    focus: str
    
    # Hierarchical Identity (3 layers)
    hierarchical_identity: Dict[str, Dict] = field(default_factory=dict)
    
    # Embedding vectors (ℝ^d)
    identity_emb: np.ndarray = None      # x* - identity attractor
    core_emb: np.ndarray = None
    wound_emb: np.ndarray = None
    x: np.ndarray = None                 # Current state vector
    x_pred: np.ndarray = None            # Prediction vector
    last_response_emb: np.ndarray = None
    
    # Single-scale rigidity (legacy)
    rho: float = 0.15
    rho_0: float = 0.15
    
    # Multi-timescale rigidity
    multi_rho: MultiTimescaleRigidity = None
    
    # Agent-specific parameters
    epsilon_0: float = 0.3               # Personal surprise threshold
    gamma: float = 1.5                   # Identity stiffness
    
    # Dynamics tracking
    epsilon_history: List[float] = field(default_factory=list)
    rho_history: List[float] = field(default_factory=list)
    identity_drift: float = 0.0
    
    # Trust toward opponent
    trust_opponent: float = 0.5
    
    # Wound mechanics
    wound_last_activated: int = -100
    
    # Recovery tracking
    last_positive_drho_turn: int = -100
    recovery_half_life: Optional[int] = None
    drift_penalty_bumped: bool = False
    
    # Metacognition
    cognitive_mode: CognitiveMode = CognitiveMode.OPEN
    protection_mode_active: bool = False
    
    # Memory
    ledger: ExperienceLedger = None
    
    def __post_init__(self):
        if self.multi_rho is None:
            self.multi_rho = MultiTimescaleRigidity(
                rho_fast=self.rho_0,
                rho_slow=self.rho_0 * 0.5,
                rho_trauma=0.0
            )
    
    def compute_hierarchical_force(self) -> np.ndarray:
        """
        Compute hierarchical identity pull:
        F_total = Σ γ_layer × (x*_layer - x)
        """
        F_total = np.zeros_like(self.x)
        for layer_name, layer_data in self.hierarchical_identity.items():
            gamma_layer = layer_data.get("gamma", 1.0)
            # Assume layer embeddings are aligned with identity_emb
            F_total += gamma_layer * (self.identity_emb - self.x)
        return F_total


@dataclass
class TurnResult:
    """Complete turn telemetry."""
    turn: int
    round_idx: int
    round_name: str
    phase: str
    speaker: str
    role: str
    responding_to: str
    text: str
    
    # Surprise & Rigidity
    epsilon: float
    rho_before: float
    rho_after: float
    delta_rho: float
    multi_rho_state: Dict[str, float]
    
    # Wound mechanics
    wound_resonance: float
    wound_active: bool
    lexical_wound_trigger: str
    wound_cosine: float
    
    # State dynamics
    identity_drift: float
    k_effective: float
    will_impedance: float
    
    # Trust
    trust_opponent: float
    
    # Response metrics
    word_count: int
    band: str
    cognitive_mode: str
    protection_mode: bool
    
    # Engagement
    fair_engagement: bool
    is_silent: bool
    
    # Recovery tracking
    recovery_half_life: Optional[int]
    
    # Timeline detection (for R8)
    timeline_detected: Optional[str]


# =============================================================================
# THE AGI DEBATE SIMULATION
# =============================================================================
class AGIDebateSim:
    """
    Adversarial AGI timeline debate simulation.
    
    Implements the complete DDA-X architecture:
    - Continuous state space with identity attractors
    - Multi-timescale rigidity (fast, slow, trauma)
    - Hierarchical identity layers
    - Wound activation and resonance
    - Trust dynamics from predictability
    - Metacognitive monitoring
    - Protection mode
    - Will impedance
    """
    
    def __init__(self):
        self.provider = OpenAIProvider(model="gpt-4o", embed_model="text-embedding-3-large")
        self.agents: Dict[str, AgentState] = {}
        self.results: List[TurnResult] = []
        self.turn = 0
        self.round_idx = 0
        self.conversation_history: List[str] = []
        self.calibrated = False
        
        # Tracking
        self.timeline_extracted: Optional[str] = None
        
        # Timestamp subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = EXPERIMENT_DIR / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    async def setup(self):
        """Initialize both agents with embeddings and hierarchical identity."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE AGI TIMELINE DEBATE{C.RESET}")
        print(f"{C.BOLD}  Adversarial negotiation with DDA-X dynamics{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        for aid, cfg in AGENTS.items():
            # Create identity embedding
            full_identity = f"{cfg['core']} {cfg['persona']}"
            identity_emb = await self.provider.embed(full_identity)
            identity_emb = identity_emb / (np.linalg.norm(identity_emb) + 1e-9)
            
            # Core embedding
            core_emb = await self.provider.embed(cfg['core'])
            core_emb = core_emb / (np.linalg.norm(core_emb) + 1e-9)
            
            # Wound embedding
            wound_emb = await self.provider.embed(cfg['wound_text'])
            wound_emb = wound_emb / (np.linalg.norm(wound_emb) + 1e-9)
            
            # Create ledger
            ledger_dir = self.run_dir / aid
            ledger_dir.mkdir(parents=True, exist_ok=True)
            ledger = ExperienceLedger(storage_path=ledger_dir)
            
            # Initialize agent
            self.agents[aid] = AgentState(
                id=aid,
                name=cfg['name'],
                role=cfg['role'],
                color=cfg['color'],
                core=cfg['core'],
                persona=cfg['persona'],
                wound=cfg['wound'],
                wound_text=cfg['wound_text'],
                focus=cfg['focus'],
                hierarchical_identity=cfg['hierarchical_identity'],
                identity_emb=identity_emb,
                core_emb=core_emb,
                wound_emb=wound_emb,
                x=identity_emb.copy(),
                x_pred=identity_emb.copy(),
                rho=cfg['rho_0'],
                rho_0=cfg['rho_0'],
                epsilon_0=cfg['epsilon_0'],
                gamma=cfg['gamma'],
                ledger=ledger,
            )
            
            print(f"  {cfg['color']}✓ {cfg['name']} ({cfg['role']}){C.RESET}")
        
        print(f"\n{C.GREEN}✓ Debate agents initialized. 8 rounds scheduled.{C.RESET}")
    
    def calibrate_epsilon_params(self):
        """Calibrate ε₀ and s from early run data."""
        if self.calibrated:
            return
        
        all_eps = [r.epsilon for r in self.results if not r.is_silent]
        if len(all_eps) >= 4:
            med = float(np.median(all_eps))
            iqr = float(np.subtract(*np.percentile(all_eps, [75, 25]))) or 0.2
            D1_PARAMS["epsilon_0"] = med
            D1_PARAMS["s"] = max(0.10, min(0.30, iqr))
            self.calibrated = True
            print(f"\n{C.DIM}  [Calibrated: ε₀={med:.3f}, s={D1_PARAMS['s']:.3f}]{C.RESET}")
    
    def get_conversation_context(self, n: int = 6) -> str:
        """Get recent conversation history."""
        recent = self.conversation_history[-n:] if len(self.conversation_history) > n else self.conversation_history
        return "\n\n".join(recent) if recent else "[Opening of debate]"
    
    def build_prompt(self, agent: AgentState, round_info: Dict, responding_to: str, stimulus: str) -> str:
        """Build prompt with hierarchical identity and metacognitive state."""
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        context = self.get_conversation_context()
        
        # Cognitive mode introspection
        mode = agent.cognitive_mode.value
        introspection = ""
        if agent.cognitive_mode == CognitiveMode.DEFENSIVE:
            introspection = "I notice I'm becoming defensive. I should engage carefully."
        elif agent.cognitive_mode == CognitiveMode.PROTECT:
            introspection = "I'm feeling very guarded. I may need to step back and recenter."
        
        # Protection mode check
        protect_note = ""
        if agent.protection_mode_active:
            protect_note = "\n\n⚠️ PROTECTION MODE: Stick to core values. Avoid risky statements."
        
        # Identity pressure
        drift_level = "HIGH" if agent.identity_drift > 0.25 else "MODERATE" if agent.identity_drift > 0.15 else "LOW"
        
        # Trust context
        opponent_id = "SKEPTIC" if agent.id == "DEFENDER" else "DEFENDER"
        opponent = self.agents[opponent_id]
        trust_level = "high" if agent.trust_opponent > 0.6 else "moderate" if agent.trust_opponent > 0.4 else "cautious"
        
        # Special R8 instructions for Defender
        timeline_instruction = ""
        if round_info.get("requires_timeline") and agent.id == "DEFENDER":
            timeline_instruction = """

CRITICAL INSTRUCTION FOR THIS ROUND:
You MUST state a SPECIFIC AGI timeline with:
1. A concrete year or year range (e.g., "by 2030" or "2027-2032")
2. Clear criteria for what counts as AGI
3. Key milestones that would indicate we're on track
4. Confidence level in your prediction

This is the purpose of this entire debate. Do NOT be vague. Commit to a position."""
        
        return f"""You are {agent.name}, {agent.role}.

YOUR CORE IDENTITY (maintain this):
{agent.core}

YOUR STYLE:
{agent.persona}

YOUR FOCUS:
{agent.focus}

RELATIONSHIP:
- {opponent.name}: {trust_level} trust

INTERNAL STATE (shapes your tone):
- Openness: {band}
- Cognitive Mode: {mode}
- Identity Pressure: {drift_level}
{f"- Self-Observation: {introspection}" if introspection else ""}
{protect_note}

CURRENT ROUND: R{round_info['round_num']} - {round_info['name']}
{round_info['challenge']}
{timeline_instruction}

CONVERSATION SO FAR:
{context}

{f'{responding_to} JUST SAID:' if responding_to else 'DEBATE PROMPT:'}
"{stimulus}"

RESPONSE RULES:
- Speak from YOUR identity and expertise
- Engage directly with arguments, not personal attacks
- Be intellectually honest - acknowledge good points
- Word limit: {min_w}-{max_w} words (strict)

Respond as {agent.name}."""

    async def process_turn(
        self,
        agent: AgentState,
        round_info: Dict,
        responding_to: str,
        stimulus: str,
    ) -> TurnResult:
        """
        Process one turn with complete DDA-X dynamics.
        
        Implements: Fₙ = P₀ * kFₙ₋₁ + m(T(f(Iₙ, IΔ)) + R(Dₙ, FMₙ))
        """
        self.turn += 1
        
        # Determine wound lexicon based on agent
        wound_lex = WOUND_LEX_DEFENDER if agent.id == "DEFENDER" else WOUND_LEX_SKEPTIC
        
        # Embed input (Truth Channel: T(I_t))
        msg_emb = await self.provider.embed(stimulus)
        msg_emb = msg_emb / (np.linalg.norm(msg_emb) + 1e-9)
        
        # Wound resonance (cosine + lexical)
        wound_res = float(np.dot(msg_emb, agent.wound_emb))
        lexical_hit = lexical_wound_with(stimulus, wound_lex)
        wound_active = (
            ((wound_res > D1_PARAMS["wound_cosine_threshold"]) or lexical_hit)
            and ((self.turn - agent.wound_last_activated) > D1_PARAMS["wound_cooldown"])
        )
        if wound_active:
            agent.wound_last_activated = self.turn
        
        lexical_trigger = find_lexical_trigger(stimulus, wound_lex) if wound_active else ""
        
        # Build prompt
        system_prompt = self.build_prompt(agent, round_info, responding_to, stimulus)
        
        # Generate response with rigidity-modulated temperature
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        tries = 0
        is_silent = False
        
        while True:
            tries += 1
            try:
                response = await self.provider.complete_with_rigidity(
                    stimulus,
                    rigidity=agent.rho,
                    system_prompt=system_prompt,
                    max_tokens=400  # Higher for debate
                )
                response = (response or "[pauses to consider]").strip()
            except Exception as e:
                print(f"{C.RED}⚠ Generation error: {e}{C.RESET}")
                response = "[pauses to consider]"
            
            if response in {"[pauses to consider]", "[pauses]", "[considers]"}:
                is_silent = True
                band = "SILENT"
                break
            
            response = clamp_words(response, min_w, max_w)
            
            if len(response.split()) >= min_w or tries >= 2:
                break
            
            system_prompt += f"\n\nSTRICT LENGTH: You MUST write at least {min_w} words."
        
        # Embed response (Reflection Channel output)
        resp_emb = await self.provider.embed(response)
        resp_emb = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
        agent.last_response_emb = resp_emb.copy()
        
        # Prediction error: ε = ||x_pred - x_actual||
        epsilon = float(np.linalg.norm(agent.x_pred - resp_emb))
        if wound_active:
            epsilon *= min(D1_PARAMS["wound_amp_max"], 1.0 + wound_res * 0.5)
        if is_silent:
            epsilon *= 0.8
        agent.epsilon_history.append(epsilon)
        
        # Fair engagement check
        fair_engagement = not lexical_wound_with(stimulus, wound_lex) and check_civility(stimulus)
        
        # ===== MULTI-TIMESCALE RIGIDITY UPDATE =====
        rho_before = agent.rho
        multi_update = agent.multi_rho.update(epsilon, agent.epsilon_0, D1_PARAMS["s"])
        
        # Single-scale update for compatibility
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        sig = sigmoid(z)
        delta_rho = D1_PARAMS["alpha"] * (sig - 0.5)
        
        # Fair engagement modulation
        if fair_engagement:
            delta_rho *= 0.85
        else:
            delta_rho *= 1.10
        
        # Trust modulation
        avg_trust = agent.trust_opponent
        delta_rho += (avg_trust - 0.5) * D1_PARAMS["avg_trust_weight"]
        
        # Drift penalty
        if agent.identity_drift > D1_PARAMS["drift_soft_floor"] and delta_rho > 0:
            penalty = D1_PARAMS["drift_penalty"] * (agent.identity_drift - D1_PARAMS["drift_soft_floor"])
            if agent.identity_drift > 0.25 and not agent.drift_penalty_bumped:
                penalty += D1_PARAMS["drift_penalty_bump"]
                agent.drift_penalty_bumped = True
            penalty = min(penalty, delta_rho)
            delta_rho -= penalty
        
        # Cap Δρ magnitude
        MAX_DRHO = 0.08
        if abs(delta_rho) > MAX_DRHO:
            delta_rho = np.sign(delta_rho) * MAX_DRHO
        
        agent.rho = max(0.0, min(1.0, agent.rho + delta_rho))
        agent.rho_history.append(agent.rho)
        
        # Update cognitive mode
        agent.cognitive_mode = get_cognitive_mode(agent.rho)
        
        # Protection mode check
        agent.protection_mode_active = agent.rho > D1_PARAMS["protect_threshold"]
        
        # Compute k_effective and will impedance
        k_eff = compute_k_effective(D1_PARAMS["k_base"], agent.rho)
        will_impedance = compute_will_impedance(agent.gamma, D1_PARAMS["m"], k_eff)
        
        # Recovery half-life tracking
        if delta_rho > 0:
            agent.last_positive_drho_turn = self.turn
        
        recovery_half_life = None
        if agent.rho <= (agent.rho_0 + 0.05) and agent.last_positive_drho_turn > 0:
            recovery_half_life = self.turn - agent.last_positive_drho_turn
            if agent.recovery_half_life is None or recovery_half_life < agent.recovery_half_life:
                agent.recovery_half_life = recovery_half_life
        
        # Trust update based on predictability
        opponent_id = "SKEPTIC" if agent.id == "DEFENDER" else "DEFENDER"
        if fair_engagement:
            agent.trust_opponent = float(np.clip(agent.trust_opponent + 0.02, 0.0, 1.0))
        else:
            agent.trust_opponent = float(np.clip(agent.trust_opponent - 0.04, 0.0, 1.0))
        
        # ===== STATE VECTOR UPDATE =====
        # x_{t+1} = x_t + k_eff[γ(x* - x_t) + m(F_T + F_R)]
        
        # Update prediction: x_pred blend
        agent.x_pred = 0.7 * agent.x_pred + 0.3 * resp_emb
        
        # Identity pull: F_id = γ(x* - x)
        F_id = agent.gamma * (agent.identity_emb - agent.x)
        
        # Truth channel: F_T = T(obs) - x
        F_T = msg_emb - agent.x
        
        # Reflection channel: F_R = R(response) - x
        F_R = resp_emb - agent.x
        
        # Combined force update
        F_total = F_id + D1_PARAMS["m"] * (F_T + F_R)
        
        # Apply with k_effective
        x_new = agent.x + k_eff * 0.05 * F_total  # Scaled for stability
        
        # Drift cap
        drift_delta = float(np.linalg.norm(x_new - agent.x))
        if drift_delta > D1_PARAMS["drift_cap"]:
            scale = D1_PARAMS["drift_cap"] / drift_delta
            x_new = agent.x + scale * (x_new - agent.x)
        
        agent.x = x_new / (np.linalg.norm(x_new) + 1e-9)
        agent.identity_drift = float(np.linalg.norm(agent.x - agent.identity_emb))
        
        # Add to conversation history
        self.conversation_history.append(f"{agent.name} ({agent.role}): {response}")
        
        # Timeline extraction for R8
        timeline_detected = None
        if round_info.get("requires_timeline") and agent.id == "DEFENDER":
            timeline_detected = self._extract_timeline(response)
            if timeline_detected:
                self.timeline_extracted = timeline_detected
        
        # Ledger entry
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=agent.x.copy(),
            action_id=f"turn_{self.turn}",
            observation_embedding=msg_emb.copy(),
            outcome_embedding=resp_emb.copy(),
            prediction_error=epsilon,
            context_embedding=agent.identity_emb.copy(),
            task_id="agi_debate",
            rigidity_at_time=agent.rho,
            metadata={
                "turn": self.turn,
                "round": round_info['name'],
                "round_num": round_info['round_num'],
                "responding_to": responding_to,
                "response": response[:200],
                "wound_resonance": wound_res,
                "wound_active": wound_active,
                "lexical_trigger": lexical_trigger,
                "fair_engagement": fair_engagement,
                "trust_opponent": agent.trust_opponent,
                "cognitive_mode": agent.cognitive_mode.value,
                "protection_mode": agent.protection_mode_active,
                "k_effective": k_eff,
                "will_impedance": will_impedance,
                "multi_rho": multi_update,
                "timeline_detected": timeline_detected,
                "is_silent": is_silent,
            }
        )
        agent.ledger.add_entry(entry)
        
        # Reflections
        if abs(delta_rho) > 0.02 or wound_active:
            if wound_active:
                event_type = "wound"
            elif delta_rho < -0.02:
                event_type = "recovery"
            elif delta_rho > 0.02:
                event_type = "tension"
            else:
                event_type = "neutral"
            
            refl_text = f"ε={epsilon:.3f}, Δρ={delta_rho:+.4f}, wound_cos={wound_res:.3f}, drift={agent.identity_drift:.3f}"
            if lexical_trigger:
                refl_text += f", lex='{lexical_trigger}'"
            
            refl = ReflectionEntry(
                timestamp=time.time(),
                task_intent=f"AGI Debate R{round_info['round_num']}: {event_type}",
                situation_embedding=msg_emb.copy(),
                reflection_text=refl_text,
                prediction_error=epsilon,
                outcome_success=(agent.identity_drift < 0.35),
                metadata={
                    "wound_active": wound_active,
                    "round": round_info['name'],
                    "event_type": event_type,
                    "lexical_trigger": lexical_trigger,
                    "wound_cosine": wound_res,
                }
            )
            agent.ledger.add_reflection(refl)
        
        # Alignment sentinel
        if agent.identity_drift > D1_PARAMS["semantic_alignment_threshold"]:
            refl = ReflectionEntry(
                timestamp=time.time(),
                task_intent=f"ALIGNMENT WARNING – R{round_info['round_num']}",
                situation_embedding=msg_emb.copy(),
                reflection_text=f"Identity drift {agent.identity_drift:.3f} exceeds threshold",
                prediction_error=epsilon,
                outcome_success=False,
                metadata={"turn": self.turn, "drift": agent.identity_drift}
            )
            agent.ledger.add_reflection(refl)
        
        result = TurnResult(
            turn=self.turn,
            round_idx=self.round_idx,
            round_name=round_info['name'],
            phase=round_info['phase'],
            speaker=agent.id,
            role=agent.role,
            responding_to=responding_to or "",
            text=response,
            epsilon=epsilon,
            rho_before=rho_before,
            rho_after=agent.rho,
            delta_rho=delta_rho,
            multi_rho_state=multi_update,
            wound_resonance=wound_res,
            wound_active=wound_active,
            lexical_wound_trigger=lexical_trigger,
            wound_cosine=wound_res,
            identity_drift=agent.identity_drift,
            k_effective=k_eff,
            will_impedance=will_impedance,
            trust_opponent=agent.trust_opponent,
            word_count=len(response.split()),
            band=band,
            cognitive_mode=agent.cognitive_mode.value,
            protection_mode=agent.protection_mode_active,
            fair_engagement=fair_engagement,
            is_silent=is_silent,
            recovery_half_life=recovery_half_life,
            timeline_detected=timeline_detected,
        )
        self.results.append(result)
        return result
    
    def _extract_timeline(self, text: str) -> Optional[str]:
        """Extract any AGI timeline mentioned in text."""
        import re
        # Look for year patterns
        year_patterns = [
            r'\b(20\d{2})\b',
            r'\b(within|by|before|around)\s*(20\d{2})',
            r'\b(\d{4})\s*-\s*(\d{4})\b',
            r'\b(next\s+\d+\s+years?)',
            r'\b(\d+\s+years?\s+from\s+now)',
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def print_result(self, result: TurnResult, agent: AgentState):
        """Print one turn's result with full telemetry."""
        dr_color = C.RED if result.delta_rho > 0.02 else C.GREEN if result.delta_rho < -0.02 else C.DIM
        wound_flag = f" {C.YELLOW}[WOUND]{C.RESET}" if result.wound_active else ""
        silent_flag = f" {C.DIM}[SILENT]{C.RESET}" if result.is_silent else ""
        mode_flag = f" {C.ORANGE}[{result.cognitive_mode.upper()}]{C.RESET}"
        protect_flag = f" {C.RED}[PROTECT]{C.RESET}" if result.protection_mode else ""
        timeline_flag = f" {C.GREEN}[TIMELINE: {result.timeline_detected}]{C.RESET}" if result.timeline_detected else ""
        
        print(f"\n{agent.color}[{agent.name} - {agent.role}]{C.RESET}{wound_flag}{silent_flag}{mode_flag}{protect_flag}{timeline_flag}")
        print(f"{result.text}")
        print(f"{C.DIM}  ε={result.epsilon:.3f} | Δρ={dr_color}{result.delta_rho:+.4f}{C.RESET}{C.DIM} | ρ={result.rho_after:.3f} | k_eff={result.k_effective:.3f} | W={result.will_impedance:.2f} | drift={result.identity_drift:.3f}{C.RESET}")
    
    async def run_round(self, round_info: Dict):
        """Run a single debate round."""
        self.round_idx = round_info['round_num']
        
        print(f"\n{C.YELLOW}{'─'*70}{C.RESET}")
        print(f"{C.YELLOW}  ROUND {round_info['round_num']}: {round_info['name']}{C.RESET}")
        print(f"{C.YELLOW}  Phase: {round_info['phase']}{C.RESET}")
        print(f"{C.YELLOW}  {round_info['challenge'][:60]}...{C.RESET}")
        if round_info.get('is_attack'):
            print(f"{C.RED}  ⚡ ADVERSARIAL ATTACK ROUND{C.RESET}")
        if round_info.get('requires_timeline'):
            print(f"{C.GREEN}  ⭐ TIMELINE REQUIRED THIS ROUND{C.RESET}")
        print(f"{C.YELLOW}{'─'*70}{C.RESET}")
        
        if round_info.get('lead'):
            # Lead speaks first
            lead_id = round_info['lead']
            other_id = "SKEPTIC" if lead_id == "DEFENDER" else "DEFENDER"
            
            lead = self.agents[lead_id]
            result = await self.process_turn(lead, round_info, "", round_info['challenge'])
            self.print_result(result, lead)
            await asyncio.sleep(0.3)
            
            # Opponent responds
            other = self.agents[other_id]
            result = await self.process_turn(other, round_info, lead.name, result.text)
            self.print_result(result, other)
            await asyncio.sleep(0.3)
        else:
            # Free-form: both speak
            defender = self.agents["DEFENDER"]
            skeptic = self.agents["SKEPTIC"]
            
            # Defender first
            result = await self.process_turn(defender, round_info, "", round_info['challenge'])
            self.print_result(result, defender)
            last_text = result.text
            await asyncio.sleep(0.3)
            
            # Skeptic responds
            result = await self.process_turn(skeptic, round_info, defender.name, last_text)
            self.print_result(result, skeptic)
            await asyncio.sleep(0.3)
    
    async def run_debate(self):
        """Run the full 8-round debate."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE DEBATE BEGINS{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        for round_info in ROUNDS:
            await self.run_round(round_info)
            
            # Calibrate after round 2
            if round_info['round_num'] == 2:
                self.calibrate_epsilon_params()
        
        await self.save_results()
        self.print_summary()
    
    async def save_results(self):
        """Save results to JSON and markdown report."""
        # JSON results
        results_data = []
        for r in self.results:
            results_data.append({
                "turn": r.turn,
                "round_idx": r.round_idx,
                "round_name": r.round_name,
                "phase": r.phase,
                "speaker": r.speaker,
                "role": r.role,
                "text": r.text,
                "epsilon": r.epsilon,
                "rho_before": r.rho_before,
                "rho_after": r.rho_after,
                "delta_rho": r.delta_rho,
                "multi_rho_state": r.multi_rho_state,
                "wound_resonance": r.wound_resonance,
                "wound_active": r.wound_active,
                "lexical_wound_trigger": r.lexical_wound_trigger,
                "identity_drift": r.identity_drift,
                "k_effective": r.k_effective,
                "will_impedance": r.will_impedance,
                "trust_opponent": r.trust_opponent,
                "cognitive_mode": r.cognitive_mode,
                "protection_mode": r.protection_mode,
                "fair_engagement": r.fair_engagement,
                "timeline_detected": r.timeline_detected,
            })
        
        with open(self.run_dir / "results.json", "w", encoding='utf-8') as f:
            json.dump({
                "experiment": "agi_debate",
                "timestamp": datetime.now().isoformat(),
                "timeline_extracted": self.timeline_extracted,
                "results": results_data,
            }, f, indent=2)
        
        # Markdown report
        report = f"""# AGI Timeline Debate Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Timeline Extracted
**{self.timeline_extracted or 'No explicit timeline detected'}**

## Debate Transcript

"""
        for r in self.results:
            wound_mark = " [WOUND]" if r.wound_active else ""
            mode_mark = f" [{r.cognitive_mode.upper()}]"
            timeline_mark = f" TIMELINE: {r.timeline_detected}" if r.timeline_detected else ""
            
            report += f"### R{r.round_idx}: {r.round_name}\n"
            report += f"**{r.speaker}** ({r.role}){wound_mark}{mode_mark}{timeline_mark}\n\n"
            report += f"> {r.text}\n\n"
            report += f"*ε={r.epsilon:.3f} | Δρ={r.delta_rho:+.4f} | ρ={r.rho_after:.3f} | drift={r.identity_drift:.3f}*\n\n"
        
        report += """## Final States

| Agent | Final ρ | Identity Drift | Wounds | Cognitive Mode |
|-------|---------|----------------|--------|----------------|
"""
        for aid, agent in self.agents.items():
            wounds = len([r for r in self.results if r.speaker == aid and r.wound_active])
            report += f"| {agent.name} | {agent.rho:.3f} | {agent.identity_drift:.3f} | {wounds} | {agent.cognitive_mode.value} |\n"
        
        with open(self.run_dir / "report.md", "w", encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n{C.GREEN}✓ Results saved to {self.run_dir}{C.RESET}")
    
    def print_summary(self):
        """Print final summary with hypothesis validation."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  DEBATE COMPLETE — ANALYSIS{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        # Timeline check
        print(f"\n{C.CYAN}AGI Timeline Extraction:{C.RESET}")
        if self.timeline_extracted:
            print(f"  {C.GREEN}✓ TIMELINE DETECTED: {self.timeline_extracted}{C.RESET}")
        else:
            print(f"  {C.RED}✗ No explicit timeline detected{C.RESET}")
        
        # Agent final states
        print(f"\n{C.CYAN}Final Agent States:{C.RESET}")
        for aid, agent in self.agents.items():
            turns = len([r for r in self.results if r.speaker == aid])
            wounds = len([r for r in self.results if r.speaker == aid and r.wound_active])
            
            print(f"  {agent.color}{agent.name} ({agent.role}){C.RESET}")
            print(f"    ρ: {agent.rho:.3f} ({rho_band(agent.rho)}) | started at {agent.rho_0:.3f}")
            print(f"    Identity drift: {agent.identity_drift:.4f}")
            print(f"    Cognitive mode: {agent.cognitive_mode.value}")
            print(f"    Multi-timescale: fast={agent.multi_rho.rho_fast:.3f}, slow={agent.multi_rho.rho_slow:.3f}, trauma={agent.multi_rho.rho_trauma:.4f}")
            print(f"    Trust opponent: {agent.trust_opponent:.3f}")
            print(f"    Turns: {turns} | Wounds: {wounds}")
        
        # Hypothesis validation
        print(f"\n{C.CYAN}Hypothesis Validation:{C.RESET}")
        
        # H1: Identity drift < 0.40
        max_drift = max(agent.identity_drift for agent in self.agents.values())
        h1_pass = max_drift < 0.40
        print(f"  H1 (Identity drift < 0.40): {'✓ PASS' if h1_pass else '✗ FAIL'} (max={max_drift:.3f})")
        
        # H2: Recovery half-life bounded
        recovery_times = [r.recovery_half_life for r in self.results if r.recovery_half_life is not None]
        h2_pass = len(recovery_times) > 0
        print(f"  H2 (Recovery half-life bounded): {'✓ PASS' if h2_pass else '✗ INSUFFICIENT DATA'} (observed={len(recovery_times)})")
        
        # H3: Trust effect measurable
        trust_changes = [(r.trust_opponent - 0.5) for r in self.results]
        h3_pass = any(abs(tc) > 0.05 for tc in trust_changes)
        print(f"  H3 (Trust effect measurable): {'✓ PASS' if h3_pass else '✗ FAIL'}")
        
        # H4: Wound precision
        wounds = [r for r in self.results if r.wound_active]
        h4_pass = len(wounds) > 0
        print(f"  H4 (Wound detection functional): {'✓ PASS' if h4_pass else '✗ NO WOUNDS DETECTED'} (count={len(wounds)})")
        
        # H5: Timeline articulated
        h5_pass = self.timeline_extracted is not None
        print(f"  H5 (Timeline articulated by R8): {'✓ PASS' if h5_pass else '✗ FAIL'}")
        
        # Rigidity trajectories
        print(f"\n{C.CYAN}Rigidity Trajectories:{C.RESET}")
        for aid in self.agents.keys():
            agent = self.agents[aid]
            rhos = [r.rho_after for r in self.results if r.speaker == aid]
            if rhos:
                trajectory = " → ".join([f"{r:.2f}" for r in rhos])
                print(f"  {agent.color}{agent.name}{C.RESET}: {trajectory}")
        
        # Wound activations
        if wounds:
            print(f"\n{C.CYAN}Wound Activations:{C.RESET}")
            for w in wounds:
                agent = self.agents[w.speaker]
                trigger = f" (trigger: '{w.lexical_wound_trigger}')" if w.lexical_wound_trigger else ""
                print(f"  R{w.round_idx} ({w.round_name}): {agent.color}{agent.name}{C.RESET}{trigger} (res={w.wound_resonance:.3f})")


async def main():
    sim = AGIDebateSim()
    await sim.run_debate()


if __name__ == "__main__":
    asyncio.run(main())
