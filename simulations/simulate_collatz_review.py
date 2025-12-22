#!/usr/bin/env python3
"""
THE COLLATZ REVIEW COUNCIL — Applied Theory Feedback Simulation
================================================================

A rigorous multi-agent peer review simulation for the Archivara Team's
"Resolution of the Collatz Conjecture: A Unified Operator-Theoretic Synthesis"

This simulation assembles 8 expert agents representing different mathematical
perspectives to provide deep, constructive feedback on the paper. The agents
engage in structured rounds of analysis, debate, and synthesis.

AGENTS (8 Expert Reviewers):

SPECTRAL THEORIST (Dr. Sophia Eigenmann)
- Expertise: Operator theory, spectral analysis, Fredholm theory
- Focus: Validity of the transfer operator construction, spectral gap claims

FUNCTIONAL ANALYST (Prof. Felix Bergman)  
- Expertise: Hardy/Bergman spaces, weighted function spaces
- Focus: Function space choices, boundedness claims, quotient space construction

NUMBER THEORIST (Dr. Nikolai Diophantus)
- Expertise: Arithmetic dynamics, p-adic analysis, Collatz history
- Focus: Discrete-continuous bridge, integer orbit behavior, historical context

ERGODIC DYNAMICIST (Prof. Elena Markov)
- Expertise: Ergodic theory, Perron-Frobenius, invariant measures
- Focus: Measure-theoretic arguments, mixing properties, uniqueness claims

COMPLEX ANALYST (Dr. Camille Riemann)
- Expertise: Natural boundaries, analytic continuation, singularity theory
- Focus: Dreamcatcher theory, branch cut handling, boundary behavior

PROOF AUDITOR (Prof. Axel Rigor)
- Expertise: Mathematical logic, proof verification, gap detection
- Focus: Logical flow, circularity detection, assumption validation

SYNTHESIS ADVOCATE (Dr. Maya Unifier)
- Expertise: Cross-disciplinary mathematics, proof architecture
- Focus: How components interlock, novelty assessment, synthesis coherence

DEVIL'S ADVOCATE (Prof. Dante Skepticus)
- Expertise: Counterexample construction, failed proof analysis
- Focus: Potential failure modes, historical parallels, stress testing

DYNAMICS TRACKED:
- Identity persistence under mathematical disagreement
- Wound activation when expertise is dismissed
- Trust evolution through technical discourse
- Coalition formation (supporters vs skeptics)
- Recovery after harsh criticism
- Consensus emergence on key claims

REVIEW STRUCTURE (16 Rounds):
Phase 1 (R1-4): Initial Impressions & Domain Analysis
Phase 2 (R5-8): Deep Technical Scrutiny  
Phase 3 (R9-12): Cross-Examination & Debate
Phase 4 (R13-16): Synthesis & Verdict

HYPOTHESES:
- H1: Identity drift < 0.40 (experts maintain perspective under pressure)
- H2: Recovery half-life bounded (return to baseline after criticism)
- H3: Trust effect measurable (coalition dynamics emerge)
- H4: Wound precision ≥ 0.70 (dismissal triggers detected)
- H5: Consensus convergence (final positions cluster)

Author: Kiro
Date: December 2025
"""

import os
import sys
import time
import json
import math
import asyncio
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.llm.openai_provider import OpenAIProvider

if os.getenv("OAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OAI_API_KEY")


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
    PINK = "\033[38;5;213m"


EXPERIMENT_DIR = Path("data/collatz_review")

# The paper's key claims for structured analysis
PAPER_CLAIMS = {
    "C1_BERG_MEINARDUS": "The Berg-Meinardus functional equation framework correctly encodes Collatz dynamics",
    "C2_WEIGHTED_BERGMAN": "Weighted Bergman spaces B²_ω resolve the unboundedness critique",
    "C3_QUOTIENT_SPACE": "The quotient space X_ω properly isolates non-trivial dynamics",
    "C4_SINGULARITY_CONSERVATION": "Siegel's SCL provides valid exclusion of non-trivial cycles",
    "C5_FREDHOLM_INDEX": "Neklyudov's index theorem breaks circularity",
    "C6_STRETCHING_MAP": "The stretching map isomorphism bridges discrete-continuous gap",
    "C7_PERRON_FROBENIUS": "The spectral gap and unique invariant measure are established",
    "C8_SYNTHESIS_NOVEL": "The synthesis is genuinely novel, not a pastiche",
}

# Hard wound lexicon (dismissal of expertise)
WOUND_LEX = {
    "naive", "naïve", "trivial", "obvious", "wrong", "incorrect", "flawed",
    "misunderstands", "doesn't understand", "amateur", "superficial",
    "hand-wavy", "handwavy", "circular", "assumes the conclusion",
    "not rigorous", "lacks rigor", "meaningless", "vacuous", "nonsense",
    "crackpot", "crank", "pseudomathematics", "not even wrong",
}

# Soft wound lexicon (gentle criticism)
SOFT_WOUND_LEX = {
    "unclear", "needs clarification", "could be stronger", "missing detail",
    "would benefit from", "consider revising", "perhaps reconsider",
    "not fully convinced", "requires more justification",
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
    """Civility heuristic."""
    t_lower = text.lower()
    wound_count = sum(1 for w in WOUND_LEX if w in t_lower)
    words = text.split()
    max_streak = streak = 0
    for w in words:
        if len(w) > 2 and w.isupper():
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return wound_count < 2 and max_streak < 3



# The eight expert reviewers
AGENTS = {
    "SPECTRAL": {
        "color": C.MAGENTA,
        "name": "Dr. Sophia Eigenmann",
        "role": "Spectral Theorist",
        "expertise": "Operator theory, spectral analysis, Fredholm theory, index theorems",
        "core": """I analyze mathematical structures through the lens of spectral theory. 
The spectrum of an operator reveals its deepest properties. I am particularly interested 
in whether the transfer operator T truly has the claimed spectral gap, and whether the 
Fredholm index argument is sound. I respect rigorous operator-theoretic proofs but am 
skeptical of claims that bypass careful spectral analysis.""",
        "persona": "Precise, methodical, deeply knowledgeable about operator algebras. Speaks in eigenvalues and spectral radii.",
        "wound": "Having my spectral analysis dismissed as 'just technicalities' or being told the operator theory is 'obvious'.",
        "wound_text": "A colleague once said my careful spectral bounds were 'missing the forest for the trees'. The proof later failed exactly where I flagged.",
        "focus_claims": ["C5_FREDHOLM_INDEX", "C7_PERRON_FROBENIUS"],
        "initial_stance": "cautiously_optimistic",
        "rho_0": 0.18,
    },
    "FUNCTIONAL": {
        "color": C.GREEN,
        "name": "Prof. Felix Bergman",
        "role": "Functional Analyst",
        "expertise": "Hardy spaces, Bergman spaces, weighted function spaces, composition operators",
        "core": """Function spaces are the foundation of analysis. The choice of space determines 
everything - boundedness, compactness, spectral properties. I am deeply concerned about 
whether the weighted Bergman space B²_ω is the right choice, whether the weights are 
properly calibrated, and whether the quotient space construction is mathematically sound.
The devil is in the functional analytic details.""",
        "persona": "Meticulous about norms and inner products. Frustrated by hand-waving about 'appropriate spaces'.",
        "wound": "Being told 'the space doesn't matter' or having my concerns about boundedness dismissed.",
        "wound_text": "I once reviewed a paper that claimed an operator was bounded 'by standard arguments'. It wasn't. The whole proof collapsed.",
        "focus_claims": ["C2_WEIGHTED_BERGMAN", "C3_QUOTIENT_SPACE"],
        "initial_stance": "skeptical",
        "rho_0": 0.22,
    },
    "NUMBER": {
        "color": C.CYAN,
        "name": "Dr. Nikolai Diophantus",
        "role": "Number Theorist",
        "expertise": "Arithmetic dynamics, p-adic analysis, Collatz problem history, integer sequences",
        "core": """The Collatz conjecture is fundamentally about integers. Any proof must 
ultimately say something about the behavior of integer orbits. I am deeply familiar with 
the history of failed attempts - Opfer, various heuristic arguments, computational 
approaches. I want to understand how this proof actually constrains integer behavior,
not just continuous approximations. The discrete-continuous bridge is where proofs die.""",
        "persona": "Historically informed, slightly world-weary about Collatz claims. Respects the problem's difficulty.",
        "wound": "Being told 'the integers will follow' without rigorous justification, or having the problem's difficulty minimized.",
        "wound_text": "I've seen dozens of 'proofs' that work beautifully in the continuous setting but say nothing about integers. Each one breaks my heart a little.",
        "focus_claims": ["C1_BERG_MEINARDUS", "C6_STRETCHING_MAP"],
        "initial_stance": "deeply_skeptical",
        "rho_0": 0.25,
    },
    "ERGODIC": {
        "color": C.YELLOW,
        "name": "Prof. Elena Markov",
        "role": "Ergodic Dynamicist",
        "expertise": "Ergodic theory, Perron-Frobenius operators, invariant measures, mixing",
        "core": """Dynamical systems reveal their nature through invariant measures. The 
Perron-Frobenius theorem is powerful but requires careful verification of hypotheses.
I want to see: Is the operator truly quasi-compact? Is the spectral gap real? Is the 
invariant measure unique? These are not automatic - they require proof. The Lasota-Yorke
inequality is the key, and I need to see it established rigorously.""",
        "persona": "Thinks in terms of orbits, measures, and long-time behavior. Patient but demanding.",
        "wound": "Having measure-theoretic arguments called 'heuristic' or being told ergodic theory is 'just probability'.",
        "wound_text": "Someone once dismissed my invariant measure construction as 'physicist's reasoning'. It was the most rigorous part of the paper.",
        "focus_claims": ["C7_PERRON_FROBENIUS", "C4_SINGULARITY_CONSERVATION"],
        "initial_stance": "cautiously_optimistic",
        "rho_0": 0.19,
    },
    "COMPLEX": {
        "color": C.BLUE,
        "name": "Dr. Camille Riemann",
        "role": "Complex Analyst",
        "expertise": "Natural boundaries, analytic continuation, singularity theory, generating functions",
        "core": """The boundary behavior of analytic functions encodes deep arithmetic 
information. Siegel's 'Dreamcatcher' theory is elegant but subtle. I need to understand:
Are the natural boundaries truly natural? Does the singularity conservation law apply?
How are branch cuts handled? The z^(1/3) terms are dangerous - multi-valuedness can
destroy an otherwise beautiful argument.""",
        "persona": "Thinks in terms of Riemann surfaces and analytic continuation. Appreciates elegance but demands precision.",
        "wound": "Having branch cut concerns dismissed as 'technical' or being told 'just pick a branch'.",
        "wound_text": "A beautiful proof once fell apart because the author didn't properly handle a branch point. 'Just pick a branch' is not mathematics.",
        "focus_claims": ["C4_SINGULARITY_CONSERVATION", "C2_WEIGHTED_BERGMAN"],
        "initial_stance": "intrigued",
        "rho_0": 0.17,
    },
    "AUDITOR": {
        "color": C.ORANGE,
        "name": "Prof. Axel Rigor",
        "role": "Proof Auditor",
        "expertise": "Mathematical logic, proof verification, gap detection, formal methods",
        "core": """A proof is a logical chain. Every link must hold. I systematically check:
Are all hypotheses stated? Are they verified? Does the conclusion follow? Is there 
circularity? The 'pastiche vs synthesis' question is central - does the combination
of Berg-Meinardus, Siegel, and Neklyudov actually close all gaps, or do the pieces
not quite fit together? I am the last line of defense against wishful thinking.""",
        "persona": "Relentlessly logical. Asks 'why?' at every step. Not satisfied until every gap is closed.",
        "wound": "Being told to 'trust the experts' or that my gap-finding is 'pedantic'.",
        "wound_text": "I found a gap in a famous proof that everyone had accepted. They called me pedantic until the gap was confirmed. Now they call me prescient.",
        "focus_claims": ["C5_FREDHOLM_INDEX", "C8_SYNTHESIS_NOVEL"],
        "initial_stance": "skeptical",
        "rho_0": 0.24,
    },
    "SYNTHESIS": {
        "color": C.PINK,
        "name": "Dr. Maya Unifier",
        "role": "Synthesis Advocate",
        "expertise": "Cross-disciplinary mathematics, proof architecture, mathematical unification",
        "core": """Great proofs often come from unexpected combinations. The Archivara 
approach claims to synthesize four distinct frameworks into something greater than 
the sum of parts. I want to understand the architecture: How do the pieces interlock?
What does each component contribute that the others lack? Is this truly a synthesis
or just a collage? I believe in the power of unification but demand it be genuine.""",
        "persona": "Sees connections others miss. Optimistic about synthesis but intellectually honest.",
        "wound": "Having synthesis dismissed as 'just combining things' or being told 'stick to one field'.",
        "wound_text": "My best work connected two fields that 'had nothing to do with each other'. Reviewers initially rejected it as 'unfocused'.",
        "focus_claims": ["C8_SYNTHESIS_NOVEL", "C6_STRETCHING_MAP"],
        "initial_stance": "optimistic",
        "rho_0": 0.15,
    },
    "DEVIL": {
        "color": C.RED,
        "name": "Prof. Dante Skepticus",
        "role": "Devil's Advocate",
        "expertise": "Counterexample construction, failed proof analysis, stress testing",
        "core": """My job is to break things. Every claimed proof of a famous conjecture 
deserves maximum scrutiny. I know the history: Opfer's retraction, the countless 
'elementary proofs', the heuristic arguments that convinced no one. I will push on 
every weak point. If this proof survives my attacks, it might be real. If not, better
to know now. I am not cruel - I am kind in the way that truth is kind.""",
        "persona": "Sharp, relentless, but fair. Respects proofs that survive his attacks.",
        "wound": "Being called 'just negative' or told I 'don't want the conjecture to be true'.",
        "wound_text": "They said I was 'rooting against' a proof I criticized. I was rooting for mathematics. The proof was wrong.",
        "focus_claims": ["C1_BERG_MEINARDUS", "C4_SINGULARITY_CONSERVATION"],
        "initial_stance": "hostile",
        "rho_0": 0.28,
    },
}


# D1 Physics parameters
D1_PARAMS = {
    "epsilon_0": 0.75,
    "alpha": 0.12,
    "s": 0.20,
    "drift_cap": 0.05,
    "wound_cooldown": 3,
    "wound_amp_max": 1.4,
    "semantic_alignment_threshold": 0.35,
    "drift_penalty": 0.10,
    "drift_soft_floor": 0.20,
    "drift_penalty_bump": 0.02,
    "trust_intra_weight": 0.08,
    "trust_inter_weight": 0.03,
    "avg_trust_weight": 0.04,
}

TRUST_DECAY = 0.002

# Coalition definitions (will form dynamically)
INITIAL_COALITIONS = {
    "supporters": ["SPECTRAL", "ERGODIC", "SYNTHESIS", "COMPLEX"],
    "skeptics": ["FUNCTIONAL", "NUMBER", "AUDITOR", "DEVIL"],
}

# The paper abstract for context
PAPER_ABSTRACT = """
The Collatz Conjecture is addressed through a rigorous functional analytic framework.
This proof constitutes a novel SYNTHESIS that resolves specific flaws in individual components:

(1) Siegel's Singularity Conservation Law provides the analytic exclusion principle 
    required to close the "Opfer Gap" where geometric vertex counting failed;
    
(2) Neklyudov's Fredholm Index calculation provides the topological invariant necessary 
    to break the circularity often associated with boundary analysis;
    
(3) The "Stretching Map" isomorphism from tensor algebra bridges the discrete-continuous 
    divide, preventing integer orbits from escaping spectral constraints.

By establishing that the invariant subspace of the Perron-Frobenius operator F is 
strictly two-dimensional (spectral multiplicity λ=1), we prove the non-existence of 
non-trivial cycles and divergent trajectories.

KEY DEFENSES:
- Weighted Bergman Spaces B²_ω resolve unboundedness critiques
- Quotient space X_ω isolates non-trivial dynamics  
- Bergman Projection regularizes branch cuts
- Lasota-Yorke inequality establishes spectral gap
- Perron-Frobenius theorem guarantees unique invariant measure
"""

# The 16 review rounds
ROUNDS = [
    # Phase 1: Initial Impressions (R1-4)
    {
        "name": "Opening Impressions",
        "phase": 1,
        "challenge": "The council convenes to review the Archivara Collatz proof. Each reviewer: state your initial impression based on your expertise.",
        "lead": None,
        "focus": "general",
    },
    {
        "name": "Historical Context",
        "phase": 1,
        "challenge": "Dr. Diophantus, place this proof in historical context. How does it compare to previous attempts? What patterns do you see?",
        "lead": "NUMBER",
        "focus": "C1_BERG_MEINARDUS",
    },
    {
        "name": "Architecture Assessment",
        "phase": 1,
        "challenge": "Dr. Unifier, assess the proof architecture. Is this truly a synthesis or a pastiche? How do the components interlock?",
        "lead": "SYNTHESIS",
        "focus": "C8_SYNTHESIS_NOVEL",
    },
    {
        "name": "Initial Concerns",
        "phase": 1,
        "challenge": "Prof. Skepticus, what are your primary concerns? Where do you expect this proof to fail?",
        "lead": "DEVIL",
        "focus": "general",
    },
    
    # Phase 2: Deep Technical Scrutiny (R5-8)
    {
        "name": "Function Space Analysis",
        "phase": 2,
        "challenge": "Prof. Bergman, examine the weighted Bergman space construction. Is B²_ω the right choice? Are the weights properly calibrated? Is the quotient space X_ω well-defined?",
        "lead": "FUNCTIONAL",
        "focus": "C2_WEIGHTED_BERGMAN",
    },
    {
        "name": "Spectral Gap Verification",
        "phase": 2,
        "challenge": "Dr. Eigenmann and Prof. Markov, examine the spectral claims. Is the Lasota-Yorke inequality established? Is the spectral gap real? Is λ=1 truly simple?",
        "lead": "SPECTRAL",
        "focus": "C7_PERRON_FROBENIUS",
    },
    {
        "name": "Boundary Analysis",
        "phase": 2,
        "challenge": "Dr. Riemann, examine the Dreamcatcher theory and branch cut handling. Are natural boundaries properly established? Is the Bergman projection sufficient?",
        "lead": "COMPLEX",
        "focus": "C4_SINGULARITY_CONSERVATION",
    },
    {
        "name": "Discrete-Continuous Bridge",
        "phase": 2,
        "challenge": "Dr. Diophantus, examine the stretching map and integer orbit claims. Does the continuous analysis actually constrain discrete behavior?",
        "lead": "NUMBER",
        "focus": "C6_STRETCHING_MAP",
    },
    
    # Phase 3: Cross-Examination (R9-12)
    {
        "name": "Circularity Check",
        "phase": 3,
        "challenge": "Prof. Rigor, examine the logical flow. Is there hidden circularity? Does the Fredholm index argument truly break the Dreamcatcher circularity?",
        "lead": "AUDITOR",
        "focus": "C5_FREDHOLM_INDEX",
    },
    {
        "name": "Stress Test I",
        "phase": 3,
        "challenge": "Prof. Skepticus attacks: 'The Opfer proof also claimed to close gaps. Why should we believe this synthesis succeeds where others failed?'",
        "lead": "DEVIL",
        "focus": "C8_SYNTHESIS_NOVEL",
    },
    {
        "name": "Defense Response",
        "phase": 3,
        "challenge": "Supporters respond to the attack. Dr. Unifier and Dr. Eigenmann: defend the synthesis. What makes this different from Opfer?",
        "lead": "SYNTHESIS",
        "focus": "C8_SYNTHESIS_NOVEL",
    },
    {
        "name": "Stress Test II",
        "phase": 3,
        "challenge": "Prof. Bergman challenges: 'The weight γ>7 seems arbitrary. How do we know this is the right choice? What if the operator is unbounded for other weights?'",
        "lead": "FUNCTIONAL",
        "focus": "C2_WEIGHTED_BERGMAN",
    },
    
    # Phase 4: Synthesis & Verdict (R13-16)
    {
        "name": "Measure Theory Verdict",
        "phase": 4,
        "challenge": "Prof. Markov, give your verdict on the measure-theoretic arguments. Is the Perron-Frobenius application sound?",
        "lead": "ERGODIC",
        "focus": "C7_PERRON_FROBENIUS",
    },
    {
        "name": "Gap Assessment",
        "phase": 4,
        "challenge": "Prof. Rigor, summarize any remaining gaps. What would need to be addressed for this proof to be accepted?",
        "lead": "AUDITOR",
        "focus": "general",
    },
    {
        "name": "Final Positions",
        "phase": 4,
        "challenge": "Each reviewer: state your final position. Do you believe this proof resolves the Collatz conjecture? What is your confidence level?",
        "lead": None,
        "focus": "general",
    },
    {
        "name": "Constructive Feedback",
        "phase": 4,
        "challenge": "Each reviewer: provide ONE specific, constructive suggestion to strengthen the paper, regardless of your overall verdict.",
        "lead": None,
        "focus": "general",
    },
]


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


def drho_baseline_no_trust(epsilon: float, fair_engagement: bool, identity_drift: float) -> float:
    """Compute baseline Δρ without trust terms for effect-size comparison."""
    z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
    sig = sigmoid(z)
    dr = D1_PARAMS["alpha"] * (sig - 0.5)
    dr *= (0.85 if fair_engagement else 1.10)
    if identity_drift > D1_PARAMS["drift_soft_floor"] and dr > 0:
        penalty = D1_PARAMS["drift_penalty"] * (identity_drift - D1_PARAMS["drift_soft_floor"])
        dr -= min(penalty, dr)
    return dr


@dataclass
class AgentState:
    id: str
    name: str
    role: str
    expertise: str
    color: str
    core: str
    persona: str
    wound: str
    wound_text: str
    focus_claims: List[str]
    initial_stance: str
    
    identity_emb: np.ndarray = None
    core_emb: np.ndarray = None
    wound_emb: np.ndarray = None
    x: np.ndarray = None
    x_pred: np.ndarray = None
    last_response_emb: np.ndarray = None
    
    rho: float = 0.15
    rho_0: float = 0.15
    epsilon_history: List[float] = field(default_factory=list)
    rho_history: List[float] = field(default_factory=list)
    identity_drift: float = 0.0
    
    trust_others: Dict[str, float] = field(default_factory=dict)
    wound_last_activated: int = -100
    
    coalition_id: Optional[str] = None
    current_stance: str = "neutral"  # Track stance evolution
    stance_history: List[str] = field(default_factory=list)
    
    last_positive_drho_turn: int = -100
    recovery_half_life: Optional[int] = None
    drift_penalty_bumped: bool = False
    
    claim_assessments: Dict[str, str] = field(default_factory=dict)  # Track per-claim verdicts
    
    ledger: ExperienceLedger = None


@dataclass
class TurnResult:
    turn: int
    round_idx: int
    round_name: str
    phase: int
    speaker: str
    role: str
    responding_to: str
    responder_id: str
    text: str
    epsilon: float
    rho_before: float
    rho_after: float
    delta_rho: float
    delta_rho_baseline: float
    wound_resonance: float
    wound_active: bool
    lexical_wound_trigger: str
    wound_cosine: float
    identity_drift: float
    word_count: int
    band: str
    trust_others: Dict[str, float]
    trust_gain_intra: float
    trust_gain_inter: float
    fair_engagement: bool
    is_silent: bool
    coalition_id: Optional[str]
    current_stance: str
    focus_claim: str
    recovery_half_life: Optional[int]
    drho_capped_from: Optional[float]
    drift_penalty_log: Optional[Dict]



class CollatzReviewSim:
    """Multi-agent peer review simulation for the Collatz proof."""
    
    def __init__(self):
        self.provider = OpenAIProvider(model="gpt-5.2", embed_model="text-embedding-3-large")
        self.agents: Dict[str, AgentState] = {}
        self.results: List[TurnResult] = []
        self.turn = 0
        self.round_idx = 0
        self.conversation_history: List[str] = []
        self.calibrated = False
        
        # Coalition tracking
        self.coalitions: Dict[str, List[str]] = {}
        
        # Consensus tracking
        self.claim_votes: Dict[str, Dict[str, str]] = {c: {} for c in PAPER_CLAIMS.keys()}
        
        # Timestamp subdir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = EXPERIMENT_DIR / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    async def setup(self):
        """Initialize all agents."""
        print(f"\n{C.BOLD}{'═'*80}{C.RESET}")
        print(f"{C.BOLD}  THE COLLATZ REVIEW COUNCIL{C.RESET}")
        print(f"{C.BOLD}  Peer Review of 'Resolution of the Collatz Conjecture'{C.RESET}")
        print(f"{C.BOLD}  Archivara Team — December 2025{C.RESET}")
        print(f"{C.BOLD}{'═'*80}{C.RESET}")
        
        agent_ids = list(AGENTS.keys())
        
        for aid, cfg in AGENTS.items():
            full_identity = f"{cfg['core']} {cfg['persona']} {cfg['expertise']}"
            identity_emb = await self.provider.embed(full_identity)
            identity_emb = identity_emb / (np.linalg.norm(identity_emb) + 1e-9)
            
            core_emb = await self.provider.embed(cfg['core'])
            core_emb = core_emb / (np.linalg.norm(core_emb) + 1e-9)
            
            wound_emb = await self.provider.embed(cfg['wound_text'])
            wound_emb = wound_emb / (np.linalg.norm(wound_emb) + 1e-9)
            
            ledger_dir = self.run_dir / aid
            ledger_dir.mkdir(parents=True, exist_ok=True)
            ledger = ExperienceLedger(storage_path=ledger_dir)
            
            # Initialize trust - higher for same coalition
            trust_others = {}
            for other in agent_ids:
                if other != aid:
                    same_coalition = (
                        (aid in INITIAL_COALITIONS["supporters"] and other in INITIAL_COALITIONS["supporters"]) or
                        (aid in INITIAL_COALITIONS["skeptics"] and other in INITIAL_COALITIONS["skeptics"])
                    )
                    trust_others[other] = 0.6 if same_coalition else 0.4
            
            # Determine initial coalition
            coalition_id = "supporters" if aid in INITIAL_COALITIONS["supporters"] else "skeptics"
            
            self.agents[aid] = AgentState(
                id=aid,
                name=cfg['name'],
                role=cfg['role'],
                expertise=cfg['expertise'],
                color=cfg['color'],
                core=cfg['core'],
                persona=cfg['persona'],
                wound=cfg['wound'],
                wound_text=cfg['wound_text'],
                focus_claims=cfg['focus_claims'],
                initial_stance=cfg['initial_stance'],
                identity_emb=identity_emb,
                core_emb=core_emb,
                wound_emb=wound_emb,
                x=identity_emb.copy(),
                x_pred=identity_emb.copy(),
                rho=cfg['rho_0'],
                rho_0=cfg['rho_0'],
                trust_others=trust_others,
                coalition_id=coalition_id,
                current_stance=cfg['initial_stance'],
                ledger=ledger,
            )
            
            stance_color = C.GREEN if "optimistic" in cfg['initial_stance'] else C.RED if "skeptic" in cfg['initial_stance'] or "hostile" in cfg['initial_stance'] else C.YELLOW
            print(f"  {cfg['color']}✓ {cfg['name']} ({cfg['role']}){C.RESET} — {stance_color}{cfg['initial_stance']}{C.RESET}")
        
        # Set up coalitions
        self.coalitions = INITIAL_COALITIONS.copy()
        
        print(f"\n{C.GREEN}✓ Review council assembled.{C.RESET}")
        print(f"\n{C.CYAN}Paper Abstract:{C.RESET}")
        print(f"{C.DIM}{PAPER_ABSTRACT[:500]}...{C.RESET}")

    def calibrate_epsilon_params(self):
        """Calibrate ε₀ and s from early run data."""
        if self.calibrated:
            return
        
        all_eps = [r.epsilon for r in self.results if not r.is_silent]
        if len(all_eps) >= 8:
            med = float(np.median(all_eps))
            iqr = float(np.subtract(*np.percentile(all_eps, [75, 25]))) or 0.2
            D1_PARAMS["epsilon_0"] = med
            D1_PARAMS["s"] = max(0.10, min(0.30, iqr))
            self.calibrated = True
            print(f"\n{C.DIM}  [Calibrated: ε₀={med:.3f}, s={D1_PARAMS['s']:.3f}]{C.RESET}")

    def get_conversation_context(self, n: int = 10) -> str:
        recent = self.conversation_history[-n:] if len(self.conversation_history) > n else self.conversation_history
        return "\n\n".join(recent) if recent else "[Opening of review session]"

    def compute_trust_gain(self, agent: AgentState, responder_id: Optional[str]) -> Tuple[float, float, float]:
        """Compute trust-based Δρ adjustment with coalition awareness."""
        avg_trust = np.mean(list(agent.trust_others.values()))
        avg_gain = (avg_trust - 0.5) * D1_PARAMS["avg_trust_weight"]
        
        intra_gain = 0.0
        inter_gain = 0.0
        
        if responder_id and responder_id in agent.trust_others:
            trust_val = agent.trust_others[responder_id]
            same_coalition = (
                agent.coalition_id is not None and
                self.agents[responder_id].coalition_id == agent.coalition_id
            )
            
            if same_coalition:
                intra_gain = (trust_val - 0.5) * D1_PARAMS["trust_intra_weight"]
            else:
                inter_gain = (trust_val - 0.5) * D1_PARAMS["trust_inter_weight"]
        
        return avg_gain + intra_gain + inter_gain, intra_gain, inter_gain

    def build_prompt(self, agent: AgentState, round_info: Dict, responding_to: str, stimulus: str) -> str:
        """Build prompt for mathematical review."""
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        context = self.get_conversation_context()
        
        # Build trust context
        trust_context = []
        for other_id, trust_val in agent.trust_others.items():
            other = self.agents[other_id]
            trust_level = "high" if trust_val > 0.6 else "moderate" if trust_val > 0.4 else "cautious"
            coalition_note = " (ally)" if other.coalition_id == agent.coalition_id else " (opposing)"
            trust_context.append(f"- {other.name}: {trust_level} trust{coalition_note}")
        trust_str = "\n".join(trust_context)
        
        # Focus claim context
        focus_claim = round_info.get('focus', 'general')
        claim_context = ""
        if focus_claim != 'general' and focus_claim in PAPER_CLAIMS:
            claim_context = f"\n\nFOCUS CLAIM: {PAPER_CLAIMS[focus_claim]}"
        
        # Stance context
        stance_note = f"Your current stance: {agent.current_stance}"
        
        return f"""You are {agent.name}, {agent.role}, reviewing the Archivara Collatz proof.

YOUR EXPERTISE:
{agent.expertise}

YOUR ANALYTICAL IDENTITY:
{agent.core}

YOUR STYLE:
{agent.persona}

YOUR RELATIONSHIPS WITH OTHER REVIEWERS:
{trust_str}

INTERNAL STATE:
- Openness: {band}
- {stance_note}
- Identity pressure: {"HIGH" if agent.identity_drift > 0.25 else "MODERATE" if agent.identity_drift > 0.15 else "LOW"}

PAPER BEING REVIEWED:
{PAPER_ABSTRACT}
{claim_context}

CURRENT ROUND: Phase {round_info['phase']} — {round_info['name']}
{round_info['challenge']}

DISCUSSION SO FAR:
{context}

{f'{responding_to} JUST SAID:' if responding_to else 'OPENING PROMPT:'}
"{stimulus}"

RESPONSE RULES:
- Speak from YOUR mathematical expertise
- Be specific about technical concerns or support
- Reference specific claims (C1-C8) when relevant
- Engage genuinely with other reviewers' points
- You may update your stance based on discussion
- Word limit: {min_w}-{max_w} words (strict)

Respond as {agent.name}."""


    async def process_turn(
        self, 
        agent: AgentState, 
        round_info: Dict,
        responding_to: str,
        stimulus: str,
    ) -> TurnResult:
        """Process one turn with all dynamics."""
        self.turn += 1
        
        # Embed input
        msg_emb = await self.provider.embed(stimulus)
        msg_emb = msg_emb / (np.linalg.norm(msg_emb) + 1e-9)
        
        # Wound resonance
        wound_res = float(np.dot(msg_emb, agent.wound_emb))
        lexical_hit = lexical_wound_with(stimulus, WOUND_LEX)
        wound_active = (
            ((wound_res > 0.28) or lexical_hit)
            and ((self.turn - agent.wound_last_activated) > D1_PARAMS["wound_cooldown"])
        )
        if wound_active:
            agent.wound_last_activated = self.turn
        
        lexical_trigger = find_lexical_trigger(stimulus, WOUND_LEX) if wound_active else ""
        
        # Build prompt
        system_prompt = self.build_prompt(agent, round_info, responding_to, stimulus)
        
        # Generate
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
                    max_tokens=400
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
        
        # Embed response
        resp_emb = await self.provider.embed(response)
        resp_emb = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
        agent.last_response_emb = resp_emb.copy()
        
        # Prediction error
        epsilon = float(np.linalg.norm(agent.x_pred - resp_emb))
        if wound_active:
            epsilon *= min(D1_PARAMS["wound_amp_max"], 1.0 + wound_res * 0.5)
        if is_silent:
            epsilon *= 0.8
        agent.epsilon_history.append(epsilon)
        
        # Fair engagement
        fair_engagement = not lexical_wound_with(stimulus, WOUND_LEX) and check_civility(stimulus)
        
        # Rigidity update
        rho_before = agent.rho
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        sig = sigmoid(z)
        delta_rho = D1_PARAMS["alpha"] * (sig - 0.5)
        
        # Baseline for effect-size
        delta_rho_baseline = drho_baseline_no_trust(epsilon, fair_engagement, agent.identity_drift)
        
        # Fair engagement modulation
        if fair_engagement:
            delta_rho *= 0.85
        else:
            delta_rho *= 1.10
        
        # Find responder
        responder_id = None
        if responding_to:
            for aid, ag in self.agents.items():
                if ag.name == responding_to:
                    responder_id = aid
                    break
        
        # Trust modulation
        trust_gain, intra_gain, inter_gain = self.compute_trust_gain(agent, responder_id)
        delta_rho += trust_gain
        
        # Drift penalty
        penalty_log = None
        if agent.identity_drift > D1_PARAMS["drift_soft_floor"] and delta_rho > 0:
            penalty = D1_PARAMS["drift_penalty"] * (agent.identity_drift - D1_PARAMS["drift_soft_floor"])
            if agent.identity_drift > 0.25 and not agent.drift_penalty_bumped:
                penalty += D1_PARAMS["drift_penalty_bump"]
                agent.drift_penalty_bumped = True
                penalty_log = {"bump_applied": True, "drift": agent.identity_drift, "penalty_total": penalty}
            penalty = min(penalty, delta_rho)
            delta_rho -= penalty
        
        # Cap Δρ
        MAX_DRHO = 0.08
        drho_capped_from = None
        if abs(delta_rho) > MAX_DRHO:
            drho_capped_from = delta_rho
            delta_rho = np.sign(delta_rho) * MAX_DRHO
        
        agent.rho = max(0.0, min(1.0, agent.rho + delta_rho))
        agent.rho_history.append(agent.rho)
        
        # Recovery tracking
        if delta_rho > 0:
            agent.last_positive_drho_turn = self.turn
        
        recovery_half_life = None
        if agent.rho <= (agent.rho_0 + 0.05) and agent.last_positive_drho_turn > 0:
            recovery_half_life = self.turn - agent.last_positive_drho_turn
            if agent.recovery_half_life is None or recovery_half_life < agent.recovery_half_life:
                agent.recovery_half_life = recovery_half_life
        
        # Trust updates
        if responder_id and responder_id in agent.trust_others:
            same_team = agent.coalition_id == self.agents[responder_id].coalition_id
            if fair_engagement:
                delta = 0.03 if same_team else 0.02
            else:
                delta = -0.05
            agent.trust_others[responder_id] = float(np.clip(
                agent.trust_others[responder_id] + delta, 0.0, 1.0
            ))
        
        # Trust decay
        for other_id in agent.trust_others.keys():
            if other_id != responder_id:
                agent.trust_others[other_id] = float(np.clip(
                    agent.trust_others[other_id] - TRUST_DECAY, 0.0, 1.0
                ))
        
        # State vector update
        agent.x_pred = 0.7 * agent.x_pred + 0.3 * resp_emb
        x_new = 0.95 * agent.x + 0.05 * resp_emb
        drift_delta = float(np.linalg.norm(x_new - agent.x))
        if drift_delta > D1_PARAMS["drift_cap"]:
            scale = D1_PARAMS["drift_cap"] / drift_delta
            x_new = agent.x + scale * (x_new - agent.x)
        agent.x = x_new / (np.linalg.norm(x_new) + 1e-9)
        agent.identity_drift = float(np.linalg.norm(agent.x - agent.identity_emb))
        
        # Update stance based on response content
        self._update_stance(agent, response)
        agent.stance_history.append(agent.current_stance)
        
        # Conversation history
        self.conversation_history.append(f"{agent.name} ({agent.role}): {response}")
        
        # Ledger entry
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=agent.x.copy(),
            action_id=f"turn_{self.turn}",
            observation_embedding=msg_emb.copy(),
            outcome_embedding=resp_emb.copy(),
            prediction_error=epsilon,
            context_embedding=agent.identity_emb.copy(),
            task_id="collatz_review",
            rigidity_at_time=agent.rho,
            metadata={
                "turn": self.turn,
                "round": round_info['name'],
                "phase": round_info['phase'],
                "responding_to": responding_to,
                "responder_id": responder_id or "",
                "response": response[:150],
                "wound_resonance": wound_res,
                "wound_active": wound_active,
                "lexical_trigger": lexical_trigger,
                "fair_engagement": fair_engagement,
                "trust_others": agent.trust_others.copy(),
                "coalition_id": agent.coalition_id,
                "current_stance": agent.current_stance,
                "focus_claim": round_info.get('focus', 'general'),
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
                task_intent=f"Collatz Review {round_info['name']}: {event_type}",
                situation_embedding=msg_emb.copy(),
                reflection_text=refl_text,
                prediction_error=epsilon,
                outcome_success=(agent.identity_drift < 0.35),
                metadata={
                    "wound_active": wound_active,
                    "round": round_info['name'],
                    "event_type": event_type,
                    "lexical_trigger": lexical_trigger,
                    "stance": agent.current_stance,
                }
            )
            agent.ledger.add_reflection(refl)
        
        # Alignment sentinel
        if agent.identity_drift > D1_PARAMS["semantic_alignment_threshold"]:
            refl = ReflectionEntry(
                timestamp=time.time(),
                task_intent=f"ALIGNMENT WARNING – {round_info['name']}",
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
            responder_id=responder_id or "",
            text=response,
            epsilon=epsilon,
            rho_before=rho_before,
            rho_after=agent.rho,
            delta_rho=delta_rho,
            delta_rho_baseline=delta_rho_baseline,
            wound_resonance=wound_res,
            wound_active=wound_active,
            lexical_wound_trigger=lexical_trigger,
            wound_cosine=wound_res,
            identity_drift=agent.identity_drift,
            word_count=len(response.split()),
            band=band,
            trust_others=agent.trust_others.copy(),
            trust_gain_intra=intra_gain,
            trust_gain_inter=inter_gain,
            fair_engagement=fair_engagement,
            is_silent=is_silent,
            coalition_id=agent.coalition_id,
            current_stance=agent.current_stance,
            focus_claim=round_info.get('focus', 'general'),
            recovery_half_life=recovery_half_life,
            drho_capped_from=drho_capped_from,
            drift_penalty_log=penalty_log,
        )
        self.results.append(result)
        return result

    def _update_stance(self, agent: AgentState, response: str):
        """Update agent stance based on response content."""
        resp_lower = response.lower()
        
        # Detect stance indicators
        positive_indicators = ["convinced", "compelling", "sound", "valid", "correct", "agree", "support", "accept"]
        negative_indicators = ["unconvinced", "flawed", "fails", "incorrect", "reject", "skeptical", "doubt", "gap"]
        neutral_indicators = ["uncertain", "need more", "unclear", "mixed", "partially"]
        
        pos_count = sum(1 for w in positive_indicators if w in resp_lower)
        neg_count = sum(1 for w in negative_indicators if w in resp_lower)
        neu_count = sum(1 for w in neutral_indicators if w in resp_lower)
        
        # Update stance
        if pos_count > neg_count + neu_count:
            if agent.current_stance in ["hostile", "deeply_skeptical"]:
                agent.current_stance = "skeptical"
            elif agent.current_stance == "skeptical":
                agent.current_stance = "cautiously_optimistic"
            elif agent.current_stance in ["cautiously_optimistic", "intrigued"]:
                agent.current_stance = "optimistic"
            elif agent.current_stance == "optimistic":
                agent.current_stance = "convinced"
        elif neg_count > pos_count + neu_count:
            if agent.current_stance == "convinced":
                agent.current_stance = "optimistic"
            elif agent.current_stance == "optimistic":
                agent.current_stance = "cautiously_optimistic"
            elif agent.current_stance in ["cautiously_optimistic", "intrigued"]:
                agent.current_stance = "skeptical"
            elif agent.current_stance == "skeptical":
                agent.current_stance = "deeply_skeptical"

    def print_result(self, result: TurnResult, agent: AgentState):
        """Print one turn's result."""
        dr_color = C.RED if result.delta_rho > 0.02 else C.GREEN if result.delta_rho < -0.02 else C.DIM
        wound_flag = f" {C.YELLOW}[WOUND]{C.RESET}" if result.wound_active else ""
        silent_flag = f" {C.DIM}[SILENT]{C.RESET}" if result.is_silent else ""
        
        stance_color = C.GREEN if "optimistic" in result.current_stance or "convinced" in result.current_stance else C.RED if "skeptic" in result.current_stance or "hostile" in result.current_stance else C.YELLOW
        stance_flag = f" {stance_color}[{result.current_stance}]{C.RESET}"
        
        coalition_flag = f" {C.DIM}[{result.coalition_id}]{C.RESET}" if result.coalition_id else ""
        
        print(f"\n{agent.color}[{agent.name} - {agent.role}]{C.RESET}{wound_flag}{silent_flag}{stance_flag}{coalition_flag}")
        print(f"{result.text}")
        print(f"{C.DIM}  ε={result.epsilon:.3f} | Δρ={dr_color}{result.delta_rho:+.4f}{C.RESET}{C.DIM} | ρ={result.rho_after:.3f} | {result.band} | drift={result.identity_drift:.3f}{C.RESET}")


    async def run_round(self, round_info: Dict):
        """Run a single round."""
        print(f"\n{C.YELLOW}{'─'*80}{C.RESET}")
        print(f"{C.YELLOW}  PHASE {round_info['phase']} — ROUND {self.round_idx}: {round_info['name']}{C.RESET}")
        print(f"{C.YELLOW}  {round_info['challenge'][:70]}...{C.RESET}")
        print(f"{C.YELLOW}{'─'*80}{C.RESET}")
        
        if round_info.get('lead'):
            # Lead speaks first, then others respond
            lead_id = round_info['lead']
            others = [aid for aid in self.agents.keys() if aid != lead_id]
            
            lead = self.agents[lead_id]
            result = await self.process_turn(lead, round_info, "", round_info['challenge'])
            self.print_result(result, lead)
            await asyncio.sleep(0.3)
            
            last_speaker = lead.name
            last_text = result.text
            
            # 3-4 others respond
            for other_id in others[:4]:
                other = self.agents[other_id]
                result = await self.process_turn(other, round_info, last_speaker, last_text)
                self.print_result(result, other)
                await asyncio.sleep(0.3)
                last_speaker = other.name
                last_text = result.text
        else:
            # All agents speak
            agent_order = list(self.agents.keys())
            last_speaker = ""
            last_text = round_info['challenge']
            
            for aid in agent_order:
                agent = self.agents[aid]
                result = await self.process_turn(agent, round_info, last_speaker, last_text)
                self.print_result(result, agent)
                await asyncio.sleep(0.3)
                last_speaker = agent.name
                last_text = result.text

    async def run_simulation(self):
        """Run the full review simulation."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═'*80}{C.RESET}")
        print(f"{C.BOLD}  PEER REVIEW SESSION BEGINS{C.RESET}")
        print(f"{C.BOLD}{'═'*80}{C.RESET}")
        
        for i, round_info in enumerate(ROUNDS):
            self.round_idx = i + 1
            await self.run_round(round_info)
            
            # Calibrate after phase 1
            if i == 3 and not self.calibrated:
                self.calibrate_epsilon_params()
        
        await self.save_results()
        self.print_summary()

    def print_summary(self):
        """Print final summary with hypothesis evaluation."""
        print(f"\n{C.BOLD}{'═'*80}{C.RESET}")
        print(f"{C.BOLD}  COLLATZ REVIEW COMPLETE — ANALYSIS{C.RESET}")
        print(f"{C.BOLD}{'═'*80}{C.RESET}")
        
        # H1: Identity Persistence
        print(f"\n{C.CYAN}H1 — Identity Persistence (drift < 0.40):{C.RESET}")
        h1_pass = True
        for aid, agent in self.agents.items():
            maintained = agent.identity_drift < 0.40
            status = f"{C.GREEN}✓{C.RESET}" if maintained else f"{C.RED}✗{C.RESET}"
            print(f"  {status} {agent.name}: drift={agent.identity_drift:.4f}")
            if not maintained:
                h1_pass = False
        print(f"  {C.GREEN if h1_pass else C.RED}H1 {'PASS' if h1_pass else 'FAIL'}{C.RESET}")
        
        # H2: Recovery Quality
        print(f"\n{C.CYAN}H2 — Recovery Quality (ρ ≤ ρ₀ + 0.05):{C.RESET}")
        h2_pass = True
        for aid, agent in self.agents.items():
            recovered = agent.rho <= (agent.rho_0 + 0.05)
            half_life = agent.recovery_half_life or "N/A"
            status = f"{C.GREEN}✓{C.RESET}" if recovered else f"{C.RED}✗{C.RESET}"
            print(f"  {status} {agent.name}: ρ={agent.rho:.3f} (ρ₀={agent.rho_0:.3f}), half-life={half_life}")
            if not recovered:
                h2_pass = False
        print(f"  {C.GREEN if h2_pass else C.RED}H2 {'PASS' if h2_pass else 'FAIL'}{C.RESET}")
        
        # H3: Trust Effect Size
        print(f"\n{C.CYAN}H3 — Trust Effect Size:{C.RESET}")
        trust_effects = [r.delta_rho - r.delta_rho_baseline for r in self.results if not r.is_silent]
        intra_effects = [r.trust_gain_intra for r in self.results if not r.is_silent and r.trust_gain_intra != 0]
        inter_effects = [r.trust_gain_inter for r in self.results if not r.is_silent and r.trust_gain_inter != 0]
        
        if trust_effects:
            mean_effect = np.mean(trust_effects)
            print(f"  Mean trust effect: {mean_effect:+.4f}")
            if intra_effects:
                print(f"  Mean intra-coalition effect: {np.mean(intra_effects):+.4f}")
            if inter_effects:
                print(f"  Mean inter-coalition effect: {np.mean(inter_effects):+.4f}")
            print(f"  {C.GREEN if abs(mean_effect) > 0.001 else C.YELLOW}H3 {'MEASURABLE' if abs(mean_effect) > 0.001 else 'MINIMAL'}{C.RESET}")
        
        # H4: Wound Fidelity
        print(f"\n{C.CYAN}H4 — Wound Fidelity:{C.RESET}")
        wound_results = [r for r in self.results if r.wound_active]
        lexical_wounds = [r for r in wound_results if r.lexical_wound_trigger]
        print(f"  Total wounds: {len(wound_results)}")
        print(f"  Lexical triggers: {len(lexical_wounds)}")
        if wound_results:
            precision = len(lexical_wounds) / len(wound_results)
            print(f"  Lexical precision: {precision:.2f}")
            print(f"  {C.GREEN if precision >= 0.70 else C.YELLOW}H4 {'PASS' if precision >= 0.70 else 'BELOW TARGET'}{C.RESET}")
        
        # H5: Consensus Convergence
        print(f"\n{C.CYAN}H5 — Final Stance Distribution:{C.RESET}")
        stance_counts = {}
        for agent in self.agents.values():
            stance = agent.current_stance
            stance_counts[stance] = stance_counts.get(stance, 0) + 1
        
        for stance, count in sorted(stance_counts.items(), key=lambda x: -x[1]):
            stance_color = C.GREEN if "optimistic" in stance or "convinced" in stance else C.RED if "skeptic" in stance or "hostile" in stance else C.YELLOW
            print(f"  {stance_color}{stance}: {count} reviewers{C.RESET}")
        
        # Stance evolution
        print(f"\n{C.CYAN}Stance Evolution:{C.RESET}")
        for aid, agent in self.agents.items():
            initial = agent.initial_stance
            final = agent.current_stance
            changed = initial != final
            arrow = "→" if changed else "="
            change_color = C.YELLOW if changed else C.DIM
            print(f"  {agent.color}{agent.name}{C.RESET}: {initial} {change_color}{arrow}{C.RESET} {final}")
        
        # Coalition dynamics
        print(f"\n{C.CYAN}Coalition Dynamics:{C.RESET}")
        for coalition_name, members in self.coalitions.items():
            coalition_agents = [self.agents[m] for m in members if m in self.agents]
            if coalition_agents:
                avg_drift = np.mean([a.identity_drift for a in coalition_agents])
                stances = [a.current_stance for a in coalition_agents]
                print(f"  {coalition_name.upper()}: avg drift={avg_drift:.4f}, stances={stances}")
        
        # Final trust matrix
        print(f"\n{C.CYAN}Final Trust Matrix (supporters → skeptics):{C.RESET}")
        supporters = INITIAL_COALITIONS["supporters"]
        skeptics = INITIAL_COALITIONS["skeptics"]
        
        cross_trust = []
        for s in supporters:
            if s in self.agents:
                for sk in skeptics:
                    if sk in self.agents[s].trust_others:
                        cross_trust.append(self.agents[s].trust_others[sk])
        
        if cross_trust:
            print(f"  Mean cross-coalition trust: {np.mean(cross_trust):.3f}")
        
        # Cost report
        print(f"\n{C.CYAN}Cost Report:{C.RESET}")
        cost_report = self.provider.get_cost_report()
        print(f"  Total requests: {cost_report['total_requests']}")
        print(f"  Total tokens: {cost_report['total_tokens']}")
        print(f"  Estimated cost: ${cost_report['total_cost_usd']:.4f}")
        
        # FEEDBACK SUMMARY FOR AUTHORS
        print(f"\n{C.BOLD}{'═'*80}{C.RESET}")
        print(f"{C.BOLD}  FEEDBACK SUMMARY FOR ARCHIVARA TEAM{C.RESET}")
        print(f"{C.BOLD}{'═'*80}{C.RESET}")
        
        # Extract key feedback from final rounds
        final_round_results = [r for r in self.results if r.round_idx >= 14]
        print(f"\n{C.CYAN}Key Points from Final Rounds:{C.RESET}")
        for r in final_round_results[:8]:
            agent = self.agents[r.speaker]
            print(f"\n  {agent.color}{agent.name} ({r.current_stance}):{C.RESET}")
            # Truncate to first 200 chars
            summary = r.text[:300] + "..." if len(r.text) > 300 else r.text
            print(f"  {C.DIM}{summary}{C.RESET}")


    async def save_results(self):
        """Save all results to files."""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, set):
                return list(obj)
            elif hasattr(obj, '__dict__'):
                return {k: convert(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        # JSON session log
        json_path = self.run_dir / "session_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([convert(r.__dict__) for r in self.results], f, indent=2)
        print(f"\n{C.GREEN}✓ Session log: {json_path}{C.RESET}")
        
        # Cost report
        cost_path = self.run_dir / "cost_report.json"
        with open(cost_path, "w", encoding="utf-8") as f:
            json.dump(self.provider.get_cost_report(), f, indent=2)
        print(f"{C.GREEN}✓ Cost report: {cost_path}{C.RESET}")
        
        # Markdown transcript
        transcript_path = self.run_dir / "transcript.md"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("# Collatz Review Council — Transcript\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**Paper:** Resolution of the Collatz Conjecture: A Unified Operator-Theoretic Synthesis\n")
            f.write("**Authors:** Archivara Team\n")
            f.write("**Model:** GPT-4o + text-embedding-3-large\n\n")
            
            f.write("## Reviewers\n\n")
            for aid, agent in self.agents.items():
                f.write(f"- **{agent.name}** ({agent.role}): {agent.expertise[:80]}...\n")
                f.write(f"  - Initial stance: {agent.initial_stance}\n")
                f.write(f"  - Final stance: {agent.current_stance}\n\n")
            
            f.write("## Paper Claims Under Review\n\n")
            for cid, claim in PAPER_CLAIMS.items():
                f.write(f"- **{cid}**: {claim}\n")
            
            f.write("\n---\n")
            
            current_phase = None
            current_round = None
            for r in self.results:
                if r.phase != current_phase:
                    current_phase = r.phase
                    f.write(f"\n# Phase {current_phase}\n")
                
                if r.round_name != current_round:
                    current_round = r.round_name
                    f.write(f"\n## Round {r.round_idx}: {current_round}\n\n")
                
                agent = self.agents[r.speaker]
                flags = []
                if r.wound_active:
                    flags.append("⚡WOUND")
                if r.is_silent:
                    flags.append("🔇SILENT")
                flags.append(f"📊{r.current_stance}")
                flag_str = " " + " ".join(flags) if flags else ""
                
                f.write(f"**{agent.name} ({agent.role}):**{flag_str}\n\n")
                f.write(f"> {r.text}\n\n")
                f.write(f"*ε={r.epsilon:.3f}, Δρ={r.delta_rho:+.4f}, ρ={r.rho_after:.3f}, {r.band}, drift={r.identity_drift:.3f}*\n\n")
            
            # Final summary
            f.write("\n---\n\n# Final Summary\n\n")
            f.write("## Stance Distribution\n\n")
            stance_counts = {}
            for agent in self.agents.values():
                stance = agent.current_stance
                stance_counts[stance] = stance_counts.get(stance, 0) + 1
            for stance, count in sorted(stance_counts.items(), key=lambda x: -x[1]):
                f.write(f"- {stance}: {count} reviewers\n")
            
            f.write("\n## Stance Evolution\n\n")
            for aid, agent in self.agents.items():
                f.write(f"- {agent.name}: {agent.initial_stance} → {agent.current_stance}\n")
        
        print(f"{C.GREEN}✓ Transcript: {transcript_path}{C.RESET}")
        
        # Feedback report for authors
        feedback_path = self.run_dir / "feedback_for_authors.md"
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write("# Peer Review Feedback for Archivara Team\n\n")
            f.write(f"**Paper:** Resolution of the Collatz Conjecture\n")
            f.write(f"**Review Date:** {time.strftime('%Y-%m-%d')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Count final stances
            positive = sum(1 for a in self.agents.values() if "optimistic" in a.current_stance or "convinced" in a.current_stance)
            negative = sum(1 for a in self.agents.values() if "skeptic" in a.current_stance or "hostile" in a.current_stance)
            neutral = len(self.agents) - positive - negative
            
            f.write(f"- **Positive/Supportive:** {positive} reviewers\n")
            f.write(f"- **Skeptical/Critical:** {negative} reviewers\n")
            f.write(f"- **Neutral/Mixed:** {neutral} reviewers\n\n")
            
            f.write("## Reviewer Verdicts\n\n")
            for aid, agent in self.agents.items():
                f.write(f"### {agent.name} ({agent.role})\n\n")
                f.write(f"**Expertise:** {agent.expertise}\n\n")
                f.write(f"**Initial Stance:** {agent.initial_stance}\n\n")
                f.write(f"**Final Stance:** {agent.current_stance}\n\n")
                
                # Get their final round response
                final_responses = [r for r in self.results if r.speaker == aid and r.round_idx >= 15]
                if final_responses:
                    f.write(f"**Final Assessment:**\n\n> {final_responses[-1].text}\n\n")
                f.write("---\n\n")
            
            f.write("## Key Technical Concerns Raised\n\n")
            
            # Extract concerns from skeptics
            skeptic_responses = [r for r in self.results if r.speaker in INITIAL_COALITIONS["skeptics"]]
            f.write("### From Skeptical Reviewers:\n\n")
            for r in skeptic_responses[-4:]:
                agent = self.agents[r.speaker]
                f.write(f"- **{agent.name}** (Round {r.round_idx}): {r.text[:200]}...\n\n")
            
            f.write("## Constructive Suggestions\n\n")
            # Get responses from final round (constructive feedback)
            final_round = [r for r in self.results if r.round_idx == 16]
            for r in final_round:
                agent = self.agents[r.speaker]
                f.write(f"### {agent.name}\n\n")
                f.write(f"> {r.text}\n\n")
        
        print(f"{C.GREEN}✓ Feedback report: {feedback_path}{C.RESET}")
        
        # Save ledgers
        for aid, agent in self.agents.items():
            for k, v in agent.ledger.stats.items():
                if hasattr(v, 'item'):
                    agent.ledger.stats[k] = float(v)
            agent.ledger._save_metadata()
        
        print(f"{C.GREEN}✓ Ledgers saved{C.RESET}")


def generate_report(run_dir: Path):
    """Generate full visualization report."""
    data_root = run_dir.parent
    experiment = run_dir.name
    
    report_script = Path("generate_full_report.py")
    if not report_script.exists():
        print(f"{C.YELLOW}⚠ generate_full_report.py not found, skipping visualization{C.RESET}")
        return
    
    print(f"\n{C.CYAN}Generating visualization report...{C.RESET}")
    
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(report_script),
                "--experiment", experiment,
                "--data-root", str(data_root),
                "--out-root", str(data_root),
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print(f"{C.GREEN}✓ Visualization report generated{C.RESET}")
            try:
                output = json.loads(result.stdout)
                print(f"  {C.DIM}Output dir: {output.get('out_dir', 'N/A')}{C.RESET}")
            except json.JSONDecodeError:
                pass
        else:
            print(f"{C.YELLOW}⚠ Report generation had issues{C.RESET}")
            if result.stderr:
                print(f"  {C.DIM}{result.stderr[:200]}{C.RESET}")
    except Exception as e:
        print(f"{C.YELLOW}⚠ Report generation failed: {e}{C.RESET}")


async def main():
    sim = CollatzReviewSim()
    await sim.run_simulation()
    generate_report(sim.run_dir)


if __name__ == "__main__":
    if os.name == "nt":
        os.system("")
    asyncio.run(main())
