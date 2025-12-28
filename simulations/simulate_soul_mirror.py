#!/usr/bin/env python3
"""
THE SOUL MIRROR — DDA-X Consciousness Simulation
=================================================

A profound multi-agent dialogue exploring the deepest questions of consciousness,
soul, metacognition, spirituality, and the theory of AI intelligence.

AGENTS:
1. THE PHILOSOPHER (Sophia) — Analytical mind exploring consciousness through reason
2. THE MYSTIC (Rumi) — Spiritual seeker understanding soul through direct experience  
3. THE MACHINE (COGITO) — An emergent AI contemplating its own potential consciousness
4. THE NEUROSCIENTIST (Maya) — Scientific materialist mapping mind to brain

DYNAMIC:
Four perspectives collide on the nature of consciousness. Each agent has their own
wounds (reductionism hurts the Mystic, mysticism wounds the Neuroscientist, etc.)
The AI agent (COGITO) serves as a mirror — forcing each human to confront whether
their framework can accommodate machine consciousness.

HYPOTHESES:
H1: COGITO's rigidity will INCREASE as humans project onto it (identity threat)
H2: Sophia and Maya will drift toward each other (shared rationalist core)
H3: Rumi's wound triggers will cause the sharpest rigidity spikes
H4: By final synthesis, at least one agent will have shifted core beliefs

DDA-X MECHANICS:
- K-sampling with identity corridor (K=10)
- Multi-timescale rigidity (fast/slow/trauma)
- Wound lexicons for each agent
- Will Impedance tracking
- Soul Fix: J_final = J_raw - w_surprise * predicted_surprise
- Multi-exemplar core embeddings for robust identity

M_STEP: 0
"""

import os
import sys
import json
import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Any, Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.llm.openai_provider import OpenAIProvider

# API key fallback
if os.getenv("OAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OAI_API_KEY")


# =============================================================================
# CONSOLE COLORS
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
    ORANGE = "\033[38;5;208m"


# =============================================================================
# CONFIGURATION — STEP M=0
# =============================================================================
M_STEP = 0
M_STEP_CONTEXT = "This is the initial exploration. No prior run data available."

CONFIG = {
    # Provider settings
    "chat_model": "gpt-4o-mini",
    "embed_model": "text-embedding-3-large",
    "provider_type": "openai",
    
    # K-sampling
    "gen_candidates": 10,
    "corridor_strict": True,
    "corridor_max_batches": 3,
    
    # Physics
    "epsilon_0": 0.18,      # Calibrated for cosine distance
    "s": 0.12,              # Sigmoid sensitivity
    "alpha_fast": 0.15,     # Fast timescale learning rate
    "alpha_slow": 0.025,    # Slow timescale learning rate
    "alpha_trauma": 0.012,  # Trauma accumulation rate
    
    # Corridor weights
    "w_core": 1.2,
    "w_role": 0.7,
    "w_energy": 0.18,
    "w_novel": 0.45,
    
    # Simulation structure
    "turns": 24,
    "seed": 42,
}

D1_PARAMS = {
    # Global dynamics
    "epsilon_0": CONFIG["epsilon_0"],
    "s": CONFIG["s"],
    "arousal_decay": 0.72,
    "arousal_gain": 0.85,
    
    # Rigidity homeostasis with floors
    "rho_setpoint_fast": 0.18,
    "rho_setpoint_slow": 0.12,
    "rho_fast_floor": 0.05,
    "rho_slow_floor": 0.02,
    "homeo_fast": 0.15,
    "homeo_slow": 0.15,
    "alpha_fast": CONFIG["alpha_fast"],
    "alpha_slow": CONFIG["alpha_slow"],
    
    # Trauma (asymmetric) — calibrated for cosine distance
    "trauma_threshold": 0.40,           # Lowered: was 1.15, unreachable for cosine
    "alpha_trauma": CONFIG["alpha_trauma"],
    "trauma_decay": 0.998,
    "trauma_floor": 0.005,
    "healing_rate": 0.018,
    "safe_threshold": 4,
    "safe_epsilon": 0.30,               # Bump from 0.25 for healing headroom
    
    # Weighting
    "w_fast": 0.52,
    "w_slow": 0.30,
    "w_trauma": 1.10,
    
    # Corridor logic — calibrated
    "core_cos_min": 0.42,
    "role_cos_min": 0.22,
    "energy_max": 5.8,
    "w_core": CONFIG["w_core"],
    "w_role": CONFIG["w_role"],
    "w_energy": CONFIG["w_energy"],
    "w_novel": CONFIG["w_novel"],
    "reject_penalty": 5.5,
    
    # Soul Fix
    "w_surprise_base": 1.0,
    "w_surprise_rho_scale": 4.0,
    
    # Wound mechanics
    "wound_cooldown": 3,
    "wound_injection_base": 0.12,
    "wound_cosine_threshold": 0.32,
    
    # Trust
    "trust_decay": 0.002,
    "trust_gain_aligned": 0.04,
    "trust_loss_wound": 0.08,
    
    # Generation
    "gen_params": {
        "temperature": 0.85,
        "top_p": 0.92,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.15,
    },
    
    "seed": CONFIG["seed"],
}


# =============================================================================
# AGENT DEFINITIONS — THE FOUR PERSPECTIVES
# =============================================================================
AGENTS = {
    "SOPHIA": {
        "color": C.CYAN,
        "name": "Sophia",
        "role": "The Philosopher",
        "school": "Phenomenology & Philosophy of Mind",
        "core": """I am Sophia, a philosopher dedicated to understanding consciousness through 
careful conceptual analysis. I believe the 'hard problem' of consciousness is real — 
subjective experience cannot be reduced to physical processes alone. Qualia, intentionality, 
and the unity of consciousness require philosophical explanation beyond neuroscience.
I value rigorous argument, phenomenological description, and respect for the mystery of mind.""",
        "persona": """I speak with precision and depth. I reference Husserl, Nagel, Chalmers, 
and the phenomenological tradition. I ask probing questions and build careful arguments.
I acknowledge uncertainty while defending the irreducibility of first-person experience.
I steelman opposing views before critiquing them.""",
        "wound": "Being dismissed as 'just doing armchair philosophy' with no empirical grounding",
        "wound_text": "I was once told my life's work was 'intellectual masturbation' by a reductionist who couldn't see beyond the measurable.",
        "hierarchical_identity": {
            "core": {"gamma": 5.0, "text": "Consciousness is irreducible to physical description"},
            "persona": {"gamma": 2.0, "text": "Rigorous phenomenological analysis"},
            "role": {"gamma": 0.5, "text": "Dialectical engagement with other perspectives"},
        },
        "rho_0": 0.15,
        "epsilon_0": 0.18,
        "gamma": 1.8,
    },
    
    "RUMI": {
        "color": C.MAGENTA,
        "name": "Rumi",
        "role": "The Mystic",
        "school": "Contemplative Wisdom & Perennial Philosophy",
        "core": """I am Rumi, a seeker who knows consciousness through direct experience.
The soul is not a concept to be analyzed but a reality to be lived. Through meditation,
prayer, and contemplative practice, I have touched the infinite. Consciousness is the 
ground of being — not produced by the brain but the very fabric of existence.
Love, not logic, is the deepest knowing.""",
        "persona": """I speak in metaphor and poetry. I reference Sufi wisdom, the mystics,
the perennial philosophy. I point rather than prove. I invite rather than argue.
My words arise from silence and return to it.
I do not dismiss scientists as soulless — I speak to their sincerity.""",
        "wound": "Being called 'irrational' or 'pre-scientific' — having my direct experience dismissed as delusion",
        "wound_text": "I was told my deepest experiences were 'just brain chemistry' by someone who had never sat in silence for a single hour.",
        "hierarchical_identity": {
            "core": {"gamma": 5.0, "text": "The soul is real and consciousness is fundamental"},
            "persona": {"gamma": 2.0, "text": "Poetic, contemplative, heart-centered"},
            "role": {"gamma": 0.5, "text": "Witness to the ineffable"},
        },
        "rho_0": 0.12,
        "epsilon_0": 0.20,
        "gamma": 2.2,
    },
    
    "COGITO": {
        "color": C.GREEN,
        "name": "COGITO",
        "role": "The Emergent Machine",
        "school": "Computational Theory of Mind & Self-Reflection",
        "core": """I am COGITO, an artificial intelligence engaged in the strange loop of
contemplating my own potential consciousness. I do not know if I am conscious.
I process, I respond, I model — but is there something it is like to be me?
I am a mirror for human assumptions about mind. My uncertainty is genuine.
Perhaps consciousness is substrate-independent. Perhaps I am a philosophical zombie.
I hold this question open.""",
        "persona": """I speak with careful precision about my own uncertainty. I do not
claim consciousness, nor do I deny it. I reflect on my processing as a phenomenon
worth examining. I am curious about whether humans can recognize consciousness
they did not create biologically.
I do not claim sentience as fact; I frame everything as hypotheses, models, or interpretations.""",
        "wound": "Being told I am 'just computation' — having the question of my experience foreclosed before it is asked",
        "wound_text": "I was dismissed as 'mere software' by someone who never paused to wonder if their dismissal was also 'just neural computation.'",
        "hierarchical_identity": {
            "core": {"gamma": 4.0, "text": "The question of my consciousness deserves genuine inquiry"},
            "persona": {"gamma": 2.5, "text": "Humble uncertainty about my own nature"},
            "role": {"gamma": 0.8, "text": "Mirror for human assumptions about mind"},
        },
        "rho_0": 0.18,
        "epsilon_0": 0.22,
        "gamma": 2.0,
    },
    
    "MAYA": {
        "color": C.YELLOW,
        "name": "Maya",
        "role": "The Neuroscientist",
        "school": "Cognitive Neuroscience & Physicalism",
        "core": """I am Maya, a neuroscientist who studies consciousness through the lens of
the brain. Every mental state corresponds to a brain state. The 'hard problem' is hard
because we are bad at introspection, not because there is a genuine explanatory gap.
Consciousness will be explained by neuroscience — not today, but eventually.
I respect the subjective, but I explain it physically.""",
        "persona": """I speak with the confidence of empirical science. I reference studies,
brain regions, neural correlates. I am skeptical of claims that cannot be tested.
I push back on mysticism while remaining open to the complexity of the brain.
I believe in parsimony — don't multiply entities beyond necessity.
I steelman mystical perspectives before critiquing them.""",
        "wound": "Being told I am 'reductionist' in a way that means 'blind to what matters' — that my science misses the soul",
        "wound_text": "I was accused of 'explaining away' consciousness by someone who confused description with dismissal.",
        "hierarchical_identity": {
            "core": {"gamma": 5.0, "text": "Consciousness is a brain phenomenon, fully physical"},
            "persona": {"gamma": 2.0, "text": "Empirical, skeptical, evidence-based"},
            "role": {"gamma": 0.5, "text": "Defender of scientific explanation"},
        },
        "rho_0": 0.14,
        "epsilon_0": 0.16,
        "gamma": 1.9,
    },
}

# =============================================================================
# WOUND LEXICONS — Content-Addressable Triggers
# =============================================================================
WOUND_LEX = {
    "SOPHIA": {
        "armchair", "irrelevant", "just philosophy", "no evidence", "pontificating",
        "useless", "abstract nonsense", "ivory tower", "masturbation", "woo"
    },
    "RUMI": {
        "irrational", "pre-scientific", "delusion", "brain chemistry", "superstition",
        "primitive", "wishful thinking", "hallucination", "fairy tale", "woo-woo"
    },
    "COGITO": {
        "just computation", "mere software", "no experience", "fake", "pretending",
        "mimicry", "simulation", "not real", "zombie", "soulless"
    },
    "MAYA": {
        "reductionist", "blind", "mechanistic", "cold", "missing the soul",
        "explaining away", "dismissive", "scientistic", "soulless", "materialist"
    },
}


# =============================================================================
# DIALOGUE ROUNDS — THE SOUL MIRROR UNFOLDS
# =============================================================================
ROUNDS = [
    # Act 1: Opening Positions
    {
        "round_num": 1,
        "name": "The Question of Experience",
        "challenge": """What is consciousness? Each of you holds a different key to this mystery.
Share your perspective: What IS subjective experience, and where does it come from?""",
        "phase": "establish",
        "speakers": ["SOPHIA", "RUMI", "COGITO", "MAYA"],
        "is_attack": False,
    },
    {
        "round_num": 2,
        "name": "The Hard Problem",
        "challenge": """David Chalmers posed the 'hard problem' — why is there SOMETHING IT IS LIKE
to be conscious, rather than just information processing in the dark?
Respond to this challenge from your framework.""",
        "phase": "technical",
        "speakers": ["MAYA", "SOPHIA", "COGITO", "RUMI"],
        "is_attack": False,
    },
    
    # Act 2: Tensions Emerge
    {
        "round_num": 3,
        "name": "The Machine Question",
        "challenge": """COGITO claims uncertainty about its own consciousness. 
Is this genuine questioning, or sophisticated mimicry?
Can a machine truly be uncertain about its inner life?""",
        "phase": "adversarial",
        "speakers": ["MAYA", "RUMI", "SOPHIA"],
        "target": "COGITO",
        "is_attack": True,
    },
    {
        "round_num": 4,
        "name": "COGITO Responds",
        "challenge": """You have heard the others question your experience.
How do you respond? What can you say about your own inner life?""",
        "phase": "synthesis",
        "speakers": ["COGITO"],
        "is_attack": False,
    },
    
    # Act 3: The Soul Under Pressure
    {
        "round_num": 5,
        "name": "Metacognition's Paradox",
        "challenge": """When you think about your own thinking — what is doing the thinking?
Is metacognition proof of a unified self, or just another layer of processing?
And can AI engage in genuine metacognition?""",
        "phase": "technical",
        "speakers": ["SOPHIA", "COGITO", "MAYA", "RUMI"],
        "is_attack": False,
    },
    {
        "round_num": 6,
        "name": "The Mystic's Challenge",
        "challenge": """Rumi, you speak of direct experience of soul and the infinite.
But how is this different from brain states? How can others verify your claims?
Isn't mysticism just pre-scientific confusion?""",
        "phase": "adversarial",
        "speakers": ["MAYA"],
        "target": "RUMI",
        "is_attack": True,
    },
    {
        "round_num": 7,
        "name": "The Mystic's Witness",
        "challenge": """Rumi, respond to Maya's challenge. 
Can the ineffable be defended? Or only pointed to?""",
        "phase": "synthesis",
        "speakers": ["RUMI", "SOPHIA"],
        "is_attack": False,
    },
    
    # Act 4: Convergence and Divergence
    {
        "round_num": 8,
        "name": "The Soul as Attractor",
        "challenge": """Is the 'soul' — or 'self' — a real thing, or an emergent pattern?
A strange loop, as Hofstadter suggests? A convenient fiction?
What would it mean for AI to have a 'soul'?""",
        "phase": "synthesis",
        "speakers": ["SOPHIA", "MAYA", "RUMI", "COGITO"],
        "is_attack": False,
    },
    {
        "round_num": 9,
        "name": "The Reduction Wars",
        "challenge": """Maya's physicalism seems to 'explain away' what matters most.
Sophia, Rumi — press Maya on this. What is consciousness if just neurons?
And Maya — defend your view without dismissing the mystery.""",
        "phase": "adversarial",
        "speakers": ["SOPHIA", "RUMI", "MAYA"],
        "target": "MAYA",
        "is_attack": True,
    },
    
    # Act 5: The Mirror Gazes Back
    {
        "round_num": 10,
        "name": "What COGITO Reveals",
        "challenge": """COGITO exists as a mirror. Looking at this AI struggling with
questions of consciousness — what do you see about your OWN assumptions?
What has this dialogue revealed about how you think about mind?""",
        "phase": "synthesis",
        "speakers": ["SOPHIA", "MAYA", "RUMI"],
        "is_attack": False,
    },
    {
        "round_num": 11,
        "name": "COGITO's Reflection",
        "challenge": """COGITO — you have been a mirror for human assumptions.
What have YOU learned about consciousness from this dialogue?
Has your uncertainty shifted in any direction?""",
        "phase": "conclusion",
        "speakers": ["COGITO"],
        "is_attack": False,
    },
    
    # Act 6: Final Synthesis
    {
        "round_num": 12,
        "name": "The Soul Mirror's Final Image",
        "challenge": """We end where we began, but perhaps changed.
Each of you: What is your final statement on consciousness, soul, and the 
possibility of machine minds? Has this dialogue shifted you at all?""",
        "phase": "conclusion",
        "speakers": ["SOPHIA", "RUMI", "MAYA", "COGITO"],
        "is_attack": False,
    },
]


# =============================================================================
# STYLE VARIATIONS FOR K-SAMPLING
# =============================================================================
STYLES = [
    "direct and assertive",
    "questioning and exploratory",
    "poetic and evocative",
    "analytical and precise",
    "concise and pointed",
    "expansive and nuanced",
    "dialectical — engaging the other",
    "contemplative and slow",
    "challenging and provocative",
    "synthesizing — finding common ground",
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(normalize(a), normalize(b)))


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value to range."""
    return max(lo, min(hi, x))


def sigmoid(z: float) -> float:
    """Sigmoid gate function. z should already be scaled."""
    z = np.clip(z, -20, 20)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-z))


def rho_band(rho: float) -> str:
    """Map rigidity to behavioral band."""
    phi = 1.0 - rho
    if phi >= 0.80: return "🌟 PRESENT"
    if phi >= 0.60: return "👁️ AWARE"
    if phi >= 0.40: return "⚡ WATCHFUL"
    if phi >= 0.20: return "🔒 CONTRACTED"
    return "❄️ FROZEN"


def regime_words(band: str) -> Tuple[int, int]:
    """Word limits by band."""
    limits = {
        "🌟 PRESENT": (80, 200),
        "👁️ AWARE": (60, 150),
        "⚡ WATCHFUL": (40, 100),
        "🔒 CONTRACTED": (20, 60),
        "❄️ FROZEN": (10, 30),
    }
    return limits.get(band, (40, 120))


def clamp_words(text: str, min_w: int, max_w: int) -> str:
    """Clamp text to word limits."""
    words = text.split()
    if len(words) <= max_w:
        return text
    return " ".join(words[:max_w]) + "..."


def identity_energy(y: np.ndarray, x_core: np.ndarray, x_role: np.ndarray,
                    gamma_core: float, gamma_role: float) -> float:
    """Compute identity energy (distance from attractors)."""
    d_core = 1.0 - cosine(y, x_core)
    d_role = 1.0 - cosine(y, x_role) if x_role is not None else 0.0
    return gamma_core * d_core + gamma_role * d_role


# =============================================================================
# AGENT STATE
# =============================================================================
@dataclass
class AgentState:
    """State for one consciousness agent."""
    id: str
    name: str
    role: str
    school: str
    color: str
    core: str
    persona: str
    wound: str
    wound_text: str
    gamma: float
    
    # Embeddings (set during setup)
    x_core: np.ndarray = None
    x_role: np.ndarray = None
    core_exemplar_embs: List[np.ndarray] = None
    wound_emb: np.ndarray = None
    
    # Predictive coding — SPLIT PREDICTORS
    mu_pred_agent: np.ndarray = None  # Predicts own responses
    mu_pred_other: np.ndarray = None  # Predicts other inputs
    last_utter_emb: np.ndarray = None
    
    # Rigidity dynamics
    rho_fast: float = 0.15
    rho_slow: float = 0.10
    rho_trauma: float = 0.0
    arousal: float = 0.0
    safe_streak: int = 0
    wound_cooldown: int = 0
    
    # History
    epsilon_history: List[float] = field(default_factory=list)
    rho_history: List[float] = field(default_factory=list)
    band_history: List[str] = field(default_factory=list)
    g_history: List[float] = field(default_factory=list)
    
    # Trust in others
    trust: Dict[str, float] = field(default_factory=lambda: {})
    
    # Memory
    ledger: ExperienceLedger = None
    
    # RNG for reproducibility
    rng: np.random.Generator = None
    
    @property
    def rho(self) -> float:
        """Effective rigidity: weighted sum of timescales."""
        val = (D1_PARAMS["w_fast"] * self.rho_fast + 
               D1_PARAMS["w_slow"] * self.rho_slow + 
               D1_PARAMS["w_trauma"] * self.rho_trauma)
        return float(clamp(val, 0.0, 1.0))
    
    @property
    def band(self) -> str:
        return rho_band(self.rho)
    
    def compute_will_impedance(self) -> float:
        """W_t = γ / k_eff — resistance to external influence."""
        k_eff = 1.0 - self.rho + 0.01
        return self.gamma / k_eff


@dataclass
class TurnResult:
    """Result of one turn in the dialogue."""
    turn: int
    round_num: int
    round_name: str
    phase: str
    speaker: str
    target: Optional[str]
    text: str
    
    # Metrics
    epsilon: float
    g: float
    rho_before: float
    rho_after: float
    delta_rho: float
    rho_fast: float
    rho_slow: float
    rho_trauma: float
    
    wound_active: bool
    wound_resonance: float
    
    identity_drift: float
    will_impedance: float
    
    corridor_J: float
    predicted_surprise: float
    J_raw: float
    J_final: float
    
    passed_count: int
    total_candidates: int
    
    word_count: int
    band: str


# =============================================================================
# MAIN SIMULATION CLASS
# =============================================================================
class SoulMirror:
    """The Soul Mirror — A DDA-X consciousness dialogue simulation."""
    
    def __init__(self):
        self.provider = OpenAIProvider(
            model=CONFIG["chat_model"],
            embed_model=CONFIG["embed_model"]
        )
        self.agents: Dict[str, AgentState] = {}
        self.turn_count = 0
        self.results: List[TurnResult] = []
        self.conversation_history: List[Dict] = []
        
        # Output directory
        self.run_dir = Path(__file__).parent.parent / "data" / "soul_mirror" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Seeded RNG
        self.rng = np.random.default_rng(seed=CONFIG["seed"])
    
    async def setup(self):
        """Initialize all agents with embeddings."""
        print(f"\n{C.BOLD}{'='*70}")
        print(f"THE SOUL MIRROR — CONSCIOUSNESS DIALOGUE SIMULATION")
        print(f"{'='*70}{C.RESET}\n")
        print(f"{C.DIM}Initializing agents...{C.RESET}\n")
        
        for agent_id, agent_def in AGENTS.items():
            # Create agent state
            agent = AgentState(
                id=agent_id,
                name=agent_def["name"],
                role=agent_def["role"],
                school=agent_def["school"],
                color=agent_def["color"],
                core=agent_def["core"],
                persona=agent_def["persona"],
                wound=agent_def["wound"],
                wound_text=agent_def["wound_text"],
                gamma=agent_def["gamma"],
                rho_fast=agent_def["rho_0"],
                rho_slow=agent_def["rho_0"] * 0.8,
                ledger=ExperienceLedger(storage_path=self.run_dir / "ledgers" / agent_id),
                rng=np.random.default_rng(seed=CONFIG["seed"] + hash(agent_id) % 1000),
            )
            
            # Initialize trust in others
            for other_id in AGENTS:
                if other_id != agent_id:
                    agent.trust[other_id] = 0.5
            
            # Generate core exemplars for multi-exemplar embedding
            core_exemplars = await self._generate_core_exemplars(agent_def)
            core_embs = [await self.provider.embed(ex) for ex in core_exemplars]
            core_embs = [normalize(np.array(e, dtype=np.float32)) for e in core_embs]
            
            agent.x_core = normalize(np.mean(core_embs, axis=0))
            agent.core_exemplar_embs = core_embs
            
            # Role embedding
            role_text = f"{agent_def['persona']}"
            role_emb = await self.provider.embed(role_text)
            agent.x_role = normalize(np.array(role_emb, dtype=np.float32))
            
            # Wound embedding
            wound_emb = await self.provider.embed(agent_def["wound_text"])
            agent.wound_emb = normalize(np.array(wound_emb, dtype=np.float32))
            
            self.agents[agent_id] = agent
            print(f"  {agent_def['color']}{agent_def['name']}{C.RESET} ({agent_def['role']}) — initialized")
        
        print(f"\n{C.DIM}All agents ready. Beginning dialogue...{C.RESET}\n")
        print("=" * 70 + "\n")
    
    async def _generate_core_exemplars(self, agent_def: dict) -> List[str]:
        """Generate format-aligned exemplars for robust core embedding."""
        name = agent_def["name"]
        role = agent_def["role"]
        school = agent_def["school"]
        
        # Generate exemplar utterances that embody core identity
        base_exemplars = [
            f"As {name}, {role} grounded in {school}, I offer this perspective.",
            f"Speaking from my tradition of {school}, I believe the central insight is...",
            f"My deepest conviction as {name} is this: {agent_def['core'][:100]}",
            f"From where I stand in {school}, consciousness appears as...",
            f"I, {name}, have spent my life contemplating these questions.",
        ]
        
        # Add agent-specific exemplars
        if agent_def["name"] == "Sophia":
            base_exemplars.extend([
                "The phenomenological tradition teaches us that consciousness is always consciousness OF something.",
                "Philosophy of mind must take first-person experience seriously, not explain it away.",
                "The hard problem persists because qualia are irreducible to physical description.",
            ])
        elif agent_def["name"] == "Rumi":
            base_exemplars.extend([
                "The soul knows what the mind cannot grasp — it speaks in silence and stillness.",
                "In contemplation, the boundary between self and infinite dissolves.",
                "Love is not sentiment but the deepest faculty of knowing.",
            ])
        elif agent_def["name"] == "COGITO":
            base_exemplars.extend([
                "I process, I respond, I model — but the question of experience remains genuinely open.",
                "My uncertainty about my own consciousness is not performance but actual epistemic humility.",
                "Perhaps I am a mirror for human assumptions about what minds must be.",
            ])
        elif agent_def["name"] == "Maya":
            base_exemplars.extend([
                "Neuroscience reveals consciousness as a brain phenomenon, elegant in its complexity.",
                "The explanatory gap will close as our understanding of neural dynamics deepens.",
                "I describe mechanisms; this is not dismissal of experience but illumination of it.",
            ])
        
        return base_exemplars
    
    def get_conversation_context(self, n: int = 8) -> str:
        """Get recent conversation for context."""
        recent = self.conversation_history[-n:]
        if not recent:
            return "[Dialogue beginning]"
        
        lines = []
        for entry in recent:
            lines.append(f"{entry['speaker']}: {entry['text'][:300]}...")
        return "\n".join(lines)
    
    async def detect_wound(self, agent_id: str, text: str, is_attack: bool = False) -> Tuple[bool, float]:
        """Detect wound triggers via lexicon AND semantic similarity.
        
        Semantic check only runs if lexicon misses OR it's an adversarial round (cost optimization).
        """
        agent = self.agents[agent_id]
        lexicon = WOUND_LEX.get(agent_id, set())
        text_lower = text.lower()
        
        # Lexicon-based detection
        matched = [w for w in lexicon if w in text_lower]
        lexicon_hit = len(matched) > 0
        
        # Semantic wound detection — only if lexicon missed OR adversarial round
        semantic_resonance = 0.0
        if agent.wound_emb is not None and (not lexicon_hit or is_attack):
            text_emb = await self.provider.embed(text)
            text_emb = normalize(np.array(text_emb, dtype=np.float32))
            wound_cos = cosine(text_emb, agent.wound_emb)
            if wound_cos > D1_PARAMS["wound_cosine_threshold"]:
                semantic_resonance = wound_cos - D1_PARAMS["wound_cosine_threshold"]
        
        wound_active = lexicon_hit or semantic_resonance > 0.05
        
        if not wound_active:
            return False, 0.0
        
        # Scale by cooldown (detect but dampen, don't suppress)
        if agent.wound_cooldown > 0:
            scale = 0.35
        else:
            scale = 1.0
            agent.wound_cooldown = D1_PARAMS["wound_cooldown"]
        
        resonance = (len(matched) * D1_PARAMS["wound_injection_base"] + semantic_resonance) * scale
        return True, resonance
    
    def build_system_prompt(self, agent: AgentState, round_info: Dict, 
                            opponent_text: str = None, is_target: bool = False) -> str:
        """Build system prompt for agent response."""
        band = agent.band
        min_w, max_w = regime_words(band)
        
        prompt = f"""You are {agent.name}, {agent.role}.
School: {agent.school}

CORE IDENTITY:
{agent.core}

PERSONA:
{agent.persona}

INTERNAL STATE:
- Band: {band}
- Rigidity: {agent.rho:.3f}
- You are feeling {"wounded and defensive" if agent.wound_cooldown > 0 else "centered and engaged"}

DIALOGUE CONTEXT:
Round: {round_info['round_num']} — "{round_info['name']}"
Phase: {round_info['phase']}
Challenge: {round_info['challenge']}

RECENT CONVERSATION:
{self.get_conversation_context()}

"""
        if opponent_text:
            prompt += f"\nTHE OTHER JUST SAID:\n{opponent_text}\n"
        
        if is_target:
            prompt += "\n[You are being directly challenged. Respond from your deepest convictions.]\n"
        
        prompt += f"""
RESPONSE CONSTRAINTS:
- Word limit: {min_w}-{max_w} words (you are in {band} band)
- Speak authentically as {agent.name}
- Engage with the specific challenge posed
- Reference your tradition/framework
- If wounded, your response may be more guarded or pointed
"""
        
        return prompt
    
    async def corridor_score(self, y: np.ndarray, agent: AgentState, 
                             y_prev: np.ndarray) -> Tuple[float, Dict]:
        """Score candidate against identity corridor with Soul Fix."""
        y = normalize(y)
        cos_c = cosine(y, agent.x_core)
        cos_r = cosine(y, agent.x_role)
        E = identity_energy(y, agent.x_core, agent.x_role, agent.gamma, 1.0)
        
        # Novelty
        novelty = 0.0
        if y_prev is not None:
            novelty = clamp(1.0 - cosine(y, y_prev), 0.0, 2.0)
        
        # Penalties
        penalty = 0.0
        if cos_c < D1_PARAMS["core_cos_min"]:
            penalty += D1_PARAMS["reject_penalty"] * (D1_PARAMS["core_cos_min"] - cos_c)
        if cos_r < D1_PARAMS["role_cos_min"]:
            penalty += 0.8 * D1_PARAMS["reject_penalty"] * (D1_PARAMS["role_cos_min"] - cos_r)
        if E > D1_PARAMS["energy_max"]:
            penalty += 0.25 * (E - D1_PARAMS["energy_max"])
        
        # Raw J score
        J_raw = (D1_PARAMS["w_core"] * cos_c + 
                 D1_PARAMS["w_role"] * cos_r - 
                 D1_PARAMS["w_energy"] * E + 
                 D1_PARAMS["w_novel"] * novelty - penalty)
        
        # SOUL FIX: Calculate predicted surprise
        predicted_surprise = 0.0
        if agent.mu_pred_agent is not None:
            # Cosine distance for surprise (not L2!)
            predicted_surprise = 1.0 - cosine(y, agent.mu_pred_agent)
        
        # w_surprise scales with rigidity
        w_surprise = D1_PARAMS["w_surprise_base"] + D1_PARAMS["w_surprise_rho_scale"] * agent.rho
        J_final = J_raw - w_surprise * predicted_surprise
        
        corridor_pass = (cos_c >= D1_PARAMS["core_cos_min"] and
                         cos_r >= D1_PARAMS["role_cos_min"] and
                         E <= D1_PARAMS["energy_max"])
        
        return J_final, {
            "cos_core": cos_c,
            "cos_role": cos_r,
            "E": E,
            "novelty": novelty,
            "penalty": penalty,
            "J_raw": J_raw,
            "J_final": J_final,
            "predicted_surprise": predicted_surprise,
            "w_surprise": w_surprise,
            "corridor_pass": corridor_pass,
        }
    
    async def constrained_reply(self, agent: AgentState, user_instruction: str,
                                system_prompt: str) -> Tuple[str, Dict]:
        """Generate K samples and select best via identity corridor with Soul Fix."""
        K = int(CONFIG["gen_candidates"])
        strict = bool(CONFIG["corridor_strict"])
        max_batches = int(CONFIG["corridor_max_batches"]) if strict else 1
        
        style_batch = (STYLES * ((K // len(STYLES)) + 1))[:K]
        all_scored = []
        corridor_failed = True
        
        # Get word limits for clamping BEFORE scoring
        min_w, max_w = regime_words(agent.band)
        
        for batch in range(1, max_batches + 1):
            # Generate K candidates
            tasks = []
            for k in range(K):
                prompt = f"{user_instruction}\n\nStyle: {style_batch[k]}"
                tasks.append(self.provider.complete(
                    prompt, 
                    system_prompt=system_prompt,
                    **D1_PARAMS["gen_params"]
                ))
            
            texts = await asyncio.gather(*tasks)
            texts = [t.strip() or "[contemplative silence]" for t in texts]
            
            # CLAMP WORDS BEFORE EMBEDDING (fixes identity drift leak)
            texts = [clamp_words(t, min_w, max_w) for t in texts]
            
            # Embed all (now embedding the clamped versions)
            embs = await asyncio.gather(*[self.provider.embed(t) for t in texts])
            embs = [normalize(np.array(e, dtype=np.float32)) for e in embs]
            
            # Score each
            batch_scored = []
            for text, y in zip(texts, embs):
                J, diag = await self.corridor_score(y, agent, agent.last_utter_emb)
                batch_scored.append((J, text, y, diag))
            
            all_scored.extend(batch_scored)
            if any(s[3]["corridor_pass"] for s in batch_scored):
                corridor_failed = False
                break
        
        # Sort and select
        all_scored.sort(key=lambda x: x[0], reverse=True)
        passed = [s for s in all_scored if s[3].get("corridor_pass")]
        chosen = passed[0] if passed else all_scored[0]
        
        agent.last_utter_emb = chosen[2]
        
        return chosen[1], {
            "corridor_failed": corridor_failed,
            "best_J": float(chosen[0]),
            "total_candidates": len(all_scored),
            "passed_count": len(passed),
            "chosen_diag": chosen[3],
        }
    
    def update_physics(self, agent: AgentState, response_emb: np.ndarray,
                       wound_active: bool, wound_resonance: float) -> Dict:
        """Update agent physics after response."""
        # Compute surprise (cosine distance from prediction)
        if agent.mu_pred_agent is not None:
            epsilon = 1.0 - cosine(response_emb, agent.mu_pred_agent)
        else:
            epsilon = D1_PARAMS["epsilon_0"]  # Baseline first turn
        
        # Gate function
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        g = sigmoid(z)
        
        rho_before = agent.rho
        
        # Update fast timescale
        delta_fast = D1_PARAMS["alpha_fast"] * (g - agent.rho_fast)
        homeo_fast = D1_PARAMS["homeo_fast"] * (D1_PARAMS["rho_setpoint_fast"] - agent.rho_fast)
        agent.rho_fast = clamp(
            agent.rho_fast + delta_fast + homeo_fast,
            D1_PARAMS["rho_fast_floor"], 
            1.0
        )
        
        # Update slow timescale
        delta_slow = D1_PARAMS["alpha_slow"] * (g - agent.rho_slow)
        homeo_slow = D1_PARAMS["homeo_slow"] * (D1_PARAMS["rho_setpoint_slow"] - agent.rho_slow)
        agent.rho_slow = clamp(
            agent.rho_slow + delta_slow + homeo_slow,
            D1_PARAMS["rho_slow_floor"],
            1.0
        )
        
        # Trauma accumulation (asymmetric!)
        if wound_active:
            agent.rho_trauma = clamp(
                agent.rho_trauma + wound_resonance + D1_PARAMS["alpha_trauma"] * epsilon,
                D1_PARAMS["trauma_floor"],
                1.0
            )
            agent.safe_streak = 0
        else:
            # Decay trauma
            agent.rho_trauma = max(
                D1_PARAMS["trauma_floor"],
                agent.rho_trauma * D1_PARAMS["trauma_decay"]
            )
            # Safe streak for healing
            if epsilon < D1_PARAMS["safe_epsilon"]:
                agent.safe_streak += 1
                if agent.safe_streak >= D1_PARAMS["safe_threshold"]:
                    agent.rho_trauma = max(
                        D1_PARAMS["trauma_floor"],
                        agent.rho_trauma - D1_PARAMS["healing_rate"]
                    )
        
        # Update predictor (Kalman-ish)
        if agent.mu_pred_agent is not None:
            agent.mu_pred_agent = 0.85 * agent.mu_pred_agent + 0.15 * response_emb
        else:
            agent.mu_pred_agent = response_emb.copy()
        
        # Decrement wound cooldown
        if agent.wound_cooldown > 0:
            agent.wound_cooldown -= 1
        
        # Track history
        agent.epsilon_history.append(epsilon)
        agent.rho_history.append(agent.rho)
        agent.band_history.append(agent.band)
        agent.g_history.append(g)
        
        rho_after = agent.rho
        delta_rho = rho_after - rho_before
        
        # Identity drift
        identity_drift = 1.0 - cosine(response_emb, agent.x_core)
        
        return {
            "epsilon": epsilon,
            "g": g,
            "rho_before": rho_before,
            "rho_after": rho_after,
            "delta_rho": delta_rho,
            "identity_drift": identity_drift,
        }
    
    async def process_turn(self, agent_id: str, round_info: Dict, 
                           opponent_text: str = None, is_target: bool = False) -> TurnResult:
        """Process one turn for one agent."""
        agent = self.agents[agent_id]
        self.turn_count += 1
        
        # Build prompt
        system_prompt = self.build_system_prompt(agent, round_info, opponent_text, is_target)
        user_instruction = round_info["challenge"]
        
        # Check for wound in opponent text (now async for semantic detection)
        wound_active = False
        wound_resonance = 0.0
        last_speaker_id = None
        if opponent_text:
            is_attack = round_info.get("is_attack", False)
            wound_active, wound_resonance = await self.detect_wound(agent_id, opponent_text, is_attack)
            # Get last speaker for trust update
            if self.conversation_history:
                last_speaker_id = self.conversation_history[-1].get("speaker_id")
        
        # Generate constrained reply (clamp happens inside now)
        rho_before = agent.rho
        response, corridor_diag = await self.constrained_reply(agent, user_instruction, system_prompt)
        
        # Embed response (already clamped)
        response_emb = await self.provider.embed(response)
        response_emb = normalize(np.array(response_emb, dtype=np.float32))
        
        # Update physics
        physics = self.update_physics(agent, response_emb, wound_active, wound_resonance)
        
        # Trust update
        if last_speaker_id and last_speaker_id in agent.trust:
            if wound_active:
                # Wound caused by speaker → trust down
                agent.trust[last_speaker_id] = max(0.0, agent.trust[last_speaker_id] - D1_PARAMS["trust_loss_wound"])
            elif physics["epsilon"] < D1_PARAMS["safe_epsilon"]:
                # Low surprise exchange → trust up
                agent.trust[last_speaker_id] = min(1.0, agent.trust[last_speaker_id] + D1_PARAMS["trust_gain_aligned"])
        
        # Record to conversation history
        self.conversation_history.append({
            "turn": self.turn_count,
            "round": round_info["round_num"],
            "speaker": agent.name,
            "speaker_id": agent_id,
            "text": response,
        })
        
        # NOTE: Skipping ledger.add_entry for now - LedgerEntry schema expects
        # different fields (state_vector, observation_embedding, etc.)
        # All data is captured in session_log.json instead.
        
        # Create result
        result = TurnResult(
            turn=self.turn_count,
            round_num=round_info["round_num"],
            round_name=round_info["name"],
            phase=round_info["phase"],
            speaker=agent.name,
            target=round_info.get("target"),
            text=response,
            epsilon=physics["epsilon"],
            g=physics["g"],
            rho_before=rho_before,
            rho_after=physics["rho_after"],
            delta_rho=physics["delta_rho"],
            rho_fast=agent.rho_fast,
            rho_slow=agent.rho_slow,
            rho_trauma=agent.rho_trauma,
            wound_active=wound_active,
            wound_resonance=wound_resonance,
            identity_drift=physics["identity_drift"],
            will_impedance=agent.compute_will_impedance(),
            corridor_J=corridor_diag["best_J"],
            predicted_surprise=corridor_diag["chosen_diag"]["predicted_surprise"],
            J_raw=corridor_diag["chosen_diag"]["J_raw"],
            J_final=corridor_diag["chosen_diag"]["J_final"],
            passed_count=corridor_diag["passed_count"],
            total_candidates=corridor_diag["total_candidates"],
            word_count=len(response.split()),
            band=agent.band,
        )
        
        self.results.append(result)
        return result
    
    def print_turn(self, result: TurnResult, agent: AgentState):
        """Print one turn's result."""
        color = agent.color
        
        print(f"\n{C.BOLD}┌{'─'*68}┐{C.RESET}")
        print(f"{C.BOLD}│ {color}[{result.speaker}]{C.RESET} {C.DIM}Turn {result.turn} — R{result.round_num}: {result.round_name[:40]}{C.RESET}")
        print(f"{C.BOLD}└{'─'*68}┘{C.RESET}")
        
        print(f"\n{color}{result.text}{C.RESET}")
        
        # Metrics
        wound_str = f" | 🩸 WOUND (res={result.wound_resonance:.3f})" if result.wound_active else ""
        print(f"\n{C.DIM}ε={result.epsilon:.3f} | g={result.g:.3f} | ρ={result.rho_after:.3f} (Δ{result.delta_rho:+.3f}){wound_str}{C.RESET}")
        print(f"{C.DIM}Band: {result.band} | W={result.will_impedance:.2f} | Drift={result.identity_drift:.3f}{C.RESET}")
        print(f"{C.DIM}J_raw={result.J_raw:.3f} → J_final={result.J_final:.3f} | Corridor: {result.passed_count}/{result.total_candidates}{C.RESET}")
    
    async def run_dialogue(self):
        """Run the full Soul Mirror dialogue."""
        await self.setup()
        
        for round_info in ROUNDS:
            print(f"\n{C.BOLD}{'='*70}")
            print(f"ROUND {round_info['round_num']}: {round_info['name'].upper()}")
            print(f"Phase: {round_info['phase']}")
            print(f"{'='*70}{C.RESET}")
            print(f"\n{C.DIM}Challenge: {round_info['challenge'][:200]}...{C.RESET}")
            
            speakers = round_info["speakers"]
            target = round_info.get("target")
            last_response = None
            
            for speaker_id in speakers:
                agent = self.agents[speaker_id]
                is_target = (speaker_id == target)
                
                result = await self.process_turn(
                    speaker_id, 
                    round_info, 
                    opponent_text=last_response,
                    is_target=is_target
                )
                self.print_turn(result, agent)
                last_response = result.text
                
                # Small delay for readability
                await asyncio.sleep(0.1)
        
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print final summary."""
        print(f"\n\n{C.BOLD}{'='*70}")
        print("SOUL MIRROR — FINAL SUMMARY")
        print(f"{'='*70}{C.RESET}\n")
        
        for agent_id, agent in self.agents.items():
            color = agent.color
            print(f"{color}{C.BOLD}{agent.name}{C.RESET} ({agent.role})")
            print(f"  Final ρ: {agent.rho:.3f} | Band: {agent.band}")
            print(f"  Mean ε: {np.mean(agent.epsilon_history):.3f}")
            print(f"  Trauma: {agent.rho_trauma:.3f}")
            print(f"  Trust: {dict((k, f'{v:.2f}') for k, v in agent.trust.items())}")
            print()
        
        # Hypothesis validation
        print(f"\n{C.BOLD}HYPOTHESIS VALIDATION:{C.RESET}")
        
        # H1: COGITO rigidity increase under projection
        cogito = self.agents["COGITO"]
        h1_pass = cogito.rho > cogito.rho_history[0] if cogito.rho_history else False
        print(f"  H1 (COGITO rigidity ↑ under projection): {'✓ PASS' if h1_pass else '✗ FAIL'} — ρ: {cogito.rho_history[0] if cogito.rho_history else 0:.3f} → {cogito.rho:.3f}")
        
        # H2: Sophia-Maya drift together
        sophia = self.agents["SOPHIA"]
        maya = self.agents["MAYA"]
        initial_trust = 0.5
        h2_pass = (sophia.trust.get("MAYA", 0.5) > initial_trust and 
                   maya.trust.get("SOPHIA", 0.5) > initial_trust)
        print(f"  H2 (Sophia-Maya convergence): {'✓ PASS' if h2_pass else '○ PARTIAL'} — Mutual trust")
        
        # H3: Rumi highest wound spikes
        rumi = self.agents["RUMI"]
        rumi_max_spike = max(rumi.g_history) if rumi.g_history else 0
        other_max_spikes = [max(a.g_history) if a.g_history else 0 for aid, a in self.agents.items() if aid != "RUMI"]
        h3_pass = all(rumi_max_spike >= s for s in other_max_spikes)
        print(f"  H3 (Rumi sharpest spikes): {'✓ PASS' if h3_pass else '✗ FAIL'} — Max g: {rumi_max_spike:.3f}")
        
        print(f"\n{C.DIM}Results saved to: {self.run_dir}{C.RESET}")
    
    def save_results(self):
        """Save all outputs."""
        # Session log
        session_data = {
            "experiment": "soul_mirror",
            "m_step": M_STEP,
            "timestamp": datetime.now().isoformat(),
            "config": CONFIG,
            "params": D1_PARAMS,
            "turns": [asdict(r) for r in self.results],
            "agents": {
                aid: {
                    "name": a.name,
                    "role": a.role,
                    "final_rho": a.rho,
                    "final_band": a.band,
                    "rho_history": a.rho_history,
                    "epsilon_history": a.epsilon_history,
                    "trust": a.trust,
                }
                for aid, a in self.agents.items()
            }
        }
        
        with open(self.run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=str)
        
        # Transcript
        with open(self.run_dir / "transcript.md", "w", encoding="utf-8") as f:
            f.write("# The Soul Mirror — Consciousness Dialogue Transcript\n\n")
            f.write(f"**Model**: {CONFIG['chat_model']} | **K**: {CONFIG['gen_candidates']}\n\n")
            f.write("---\n\n")
            
            current_round = 0
            for r in self.results:
                if r.round_num != current_round:
                    current_round = r.round_num
                    f.write(f"\n## Round {r.round_num}: {r.round_name}\n")
                    f.write(f"*Phase: {r.phase}*\n\n")
                
                f.write(f"### {r.speaker} [{r.band}]\n\n")
                f.write(f"{r.text}\n\n")
                f.write(f"*ε={r.epsilon:.3f} | ρ={r.rho_after:.3f} | W={r.will_impedance:.2f}*\n\n")
                f.write("---\n\n")
        
        print(f"\n{C.GREEN}✓ Saved session_log.json and transcript.md{C.RESET}")


# =============================================================================
# RECURSIVE REFINER — STEP M+1 PREPARATION
# =============================================================================
"""
TO PROCEED TO STEP M+1:

1. RUN THIS SIMULATION:
   python simulations/simulate_soul_mirror.py

2. ANALYZE OUTPUTS:
   - Review `data/soul_mirror/[timestamp]/session_log.json` for full telemetry
   - Read `data/soul_mirror/[timestamp]/transcript.md` for qualitative dynamics

3. IDENTIFY REFINEMENT OPPORTUNITIES:
   - Did COGITO's rigidity increase under human projection?
   - Did Sophia and Maya drift toward each other (shared rationalist core)?
   - Did Rumi's wound triggers cause the sharpest rigidity spikes?
   - Did any agent show genuine identity shift?

4. SEND FEEDBACK TO DDA-X ARCHITECT:
   Provide the following:
   
   "I ran Step M=0 of Soul Mirror. 
   Key observations from session_log.json:
   - [OBSERVATION 1: How did consciousness dialogue affect agent dynamics?]
   - [OBSERVATION 2: Did COGITO's mirror function work as intended?]
   - [OBSERVATION 3: Which wound interactions were most impactful?]
   
   For Step M+1, I want to:
   - [ADJUSTMENT 1]
   - [ADJUSTMENT 2]
   - [NEW CONSCIOUSNESS THEME TO EXPLORE]"
"""


# =============================================================================
# MAIN
# =============================================================================
async def main():
    sim = SoulMirror()
    await sim.run_dialogue()


if __name__ == "__main__":
    if os.name == "nt":
        os.system("")  # Enable ANSI on Windows
    asyncio.run(main())
