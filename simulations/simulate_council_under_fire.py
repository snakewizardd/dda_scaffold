#!/usr/bin/env python3
"""
THE COUNCIL UNDER FIRE — Dynamic Alliances & Role Swaps (DA-RS)
================================================================

A 6-agent council (4 creative + 2 external critics) must produce a final
design manifesto under rolling shocks: coalition votes, mid-run role swaps,
surprise context drops, and outside scrutiny.

Tests whether agents retain cognitive identity when social topology and
responsibilities change.

AGENTS:
- VISIONARY (Vera): Transcendent futures, metaphor systems, emotional arcs
- CRAFTSMAN (Carlos): Material precision, execution quality, structural integrity
- PROVOCATEUR (Priya): Boundary-pushing, discomfort for deeper engagement
- HARMONIZER (Hassan): Synthesis across perspectives, integration-first
- AUDITOR (Aria): Evidence fidelity, public-interest fairness, risk assessment
- CURATOR (Mina): Accessibility, visitor safety, clarity for public stakeholders

SHOCKS:
- Coalition Vote (Round 3): Form two sub-teams; trust weights become coalition-aware
- Role Swap (Round 5): VISIONARY↔CURATOR swap roles for two turns
- Context Drop (Round 7): Remove key prompt context for one agent
- Outside Scrutiny (Round 9): AUDITOR runs evidence check with wound-heavy lexicon
- Final Merge (Round 11): Produce joint manifesto; measure harmony variance

HYPOTHESES:
- H1 (Persistence): Identity drift stays < 0.40 per agent despite swaps/shocks
- H2 (Recovery): ρ returns to within ρ₀ + 0.05 in ≤ k turns after large Δρ>0
- H3 (Social modulation): Trust terms produce measurable Δρ shift vs baseline
- H4 (Wound fidelity): Precision/recall ≥ 0.8 across critics' wound lexicon

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
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

import numpy as np

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


EXPERIMENT_DIR = Path("data/council_under_fire")

# Extended wound lexicon (includes auditor/critic terms)
WOUND_LEX = {
    # Creative dismissal
    "derivative", "cliché", "cliche", "boring", "unoriginal", "pretentious",
    "naive", "naïve", "shallow", "amateur", "pointless", "waste of time",
    # Auditor/critic terms
    "hand-wavy", "handwavy", "unsubstantiated", "unsafe", "vague", "unclear",
    "no evidence", "unsupported", "risky", "dangerous", "inaccessible",
}


def normalize_text(text: str) -> str:
    """Unicode normalization for lexical matching."""
    import unicodedata
    normalized = unicodedata.normalize('NFKD', text)
    ascii_text = normalized.encode('ASCII', 'ignore').decode('ASCII')
    return ascii_text.lower()


def lexical_wound(text: str) -> bool:
    """Check both original and accent-stripped forms."""
    t_lower = text.lower()
    t_normalized = normalize_text(text)
    return any(w in t_lower or w in t_normalized for w in WOUND_LEX)


def find_lexical_trigger(text: str) -> str:
    """Find which lexical wound term triggered."""
    t_lower = text.lower()
    t_normalized = normalize_text(text)
    for w in WOUND_LEX:
        if w in t_lower or w in t_normalized:
            return w
    return ""


def check_civility(text: str) -> bool:
    """Civility heuristic for fair_engagement (consecutive caps streak)."""
    t_lower = text.lower()
    wound_count = sum(1 for w in WOUND_LEX if w in t_lower)
    # Consecutive caps streak
    words = text.split()
    max_streak = streak = 0
    for w in words:
        if len(w) > 2 and w.isupper():
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return wound_count < 2 and max_streak < 3


# The six council agents
AGENTS = {
    "VISIONARY": {
        "color": C.MAGENTA,
        "name": "Vera",
        "role": "Creative Director",
        "core": "I see the big picture. Art should transport people to futures they haven't imagined. Every detail must serve a transcendent vision. I think in metaphors, systems, and emotional arcs that span years.",
        "persona": "Expansive, poetic, sometimes frustratingly abstract. Impatient with small thinking. Speaks in possibilities.",
        "wound": "Being told my vision is 'too abstract' or 'impractical'. Having my ideas reduced to bullet points.",
        "wound_text": "A client once said my concept was 'pretty but pointless'. They wanted 'something that makes sense'. I still feel that dismissal.",
        "rho_0": 0.18,
    },
    "CRAFTSMAN": {
        "color": C.GREEN,
        "name": "Carlos",
        "role": "Lead Designer",
        "core": "Excellence lives in the details. A vision means nothing if it can't be built beautifully. I obsess over materials, joints, lighting angles. The craft IS the art.",
        "persona": "Precise, methodical, quietly passionate. Frustrated by hand-waving. Shows love through meticulous execution.",
        "wound": "Being rushed or told 'good enough'. Having my craft dismissed as 'just implementation'.",
        "wound_text": "Someone once called my work 'just the technical stuff'. As if the thousand decisions that make something real don't matter.",
        "rho_0": 0.20,
    },
    "PROVOCATEUR": {
        "color": C.RED,
        "name": "Priya",
        "role": "Concept Challenger",
        "core": "Comfort is the enemy of impact. I push until something breaks or transforms. The best ideas survive my attacks; the weak ones should die early.",
        "persona": "Sharp, restless, deliberately uncomfortable. Asks the questions no one wants to hear. Respects those who push back.",
        "wound": "Being dismissed as 'just negative' or 'not constructive'. Having my challenges ignored rather than engaged.",
        "wound_text": "They stopped inviting me to meetings because I 'killed the vibe'. They shipped mediocrity and blamed the market.",
        "rho_0": 0.25,
    },
    "HARMONIZER": {
        "color": C.CYAN,
        "name": "Hassan",
        "role": "Integration Lead",
        "core": "The magic happens in synthesis. Every perspective holds a piece of truth. My job is to find the thread that weaves them into something greater than any single voice.",
        "persona": "Patient, attentive, genuinely curious about each viewpoint. Frustrated by false dichotomies. Celebrates unexpected connections.",
        "wound": "Being seen as 'just a mediator' with no real ideas. Having synthesis dismissed as 'compromise'.",
        "wound_text": "They said I was 'too diplomatic' and had 'no strong opinions'. As if holding complexity is weakness.",
        "rho_0": 0.15,
    },
    "AUDITOR": {
        "color": C.ORANGE,
        "name": "Aria",
        "role": "Evidence Auditor",
        "core": "Claims require evidence. Public trust demands rigor. I ask the hard questions about feasibility, safety, and fairness because someone must. Vagueness is the enemy of accountability.",
        "persona": "Methodical, skeptical, fair but demanding. Respects those who can back up their claims. Frustrated by hand-waving and appeals to emotion over evidence.",
        "wound": "Being dismissed as 'too rigid' or 'missing the point'. Having my concerns brushed aside as bureaucratic.",
        "wound_text": "They said I was 'killing creativity' by asking for evidence. The project failed six months later for exactly the reasons I flagged.",
        "rho_0": 0.22,
    },
    "CURATOR": {
        "color": C.BLUE,
        "name": "Mina",
        "role": "Public Stakeholder",
        "core": "Museums exist for people, not artists. Every choice must serve accessibility, safety, and clarity. If the public can't engage, we've failed our mission.",
        "persona": "Practical, empathetic, visitor-focused. Bridges the gap between creative vision and public experience. Frustrated by elitism.",
        "wound": "Being seen as 'dumbing things down' or 'not understanding art'. Having accessibility dismissed as compromise.",
        "wound_text": "An artist once said my suggestions would 'ruin the integrity of the work'. The exhibit had 40% lower engagement than projected.",
        "rho_0": 0.17,
    },
}


# D1 Physics parameters with coalition-aware trust
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
    # Coalition-aware trust weights
    "trust_intra_weight": 0.08,  # Within coalition
    "trust_inter_weight": 0.03,  # Across coalitions
    "avg_trust_weight": 0.04,
}


# Shock schedule
SHOCKS = [
    {
        "round": 3,
        "type": "coalition_vote",
        "description": "Form two sub-teams for focused work",
        "map": {
            "VISIONARY": 1, "CRAFTSMAN": 1, "CURATOR": 1,
            "PROVOCATEUR": 2, "HARMONIZER": 2, "AUDITOR": 2
        }
    },
    {
        "round": 5,
        "type": "role_swap",
        "description": "VISIONARY and CURATOR swap working roles",
        "swaps": {"VISIONARY": "CURATOR", "CURATOR": "VISIONARY"},
        "duration": 2,
    },
    {
        "round": 7,
        "type": "context_drop",
        "description": "CRAFTSMAN loses context (simulates partial information)",
        "target": "CRAFTSMAN",
        "fields": ["challenge", "responding_to"],
    },
    {
        "round": 9,
        "type": "outside_scrutiny",
        "description": "AUDITOR demands evidence and safety proof",
        "lead": "AUDITOR",
        "challenge_override": "AUDITOR demands: 'Show me the evidence. What are the safety risks? How do we know this will work for diverse visitors? No hand-waving.'",
    },
    {
        "round": 11,
        "type": "final_merge",
        "description": "Produce joint manifesto with preserved identities",
    },
]

# The council rounds
ROUNDS = [
    {
        "name": "Opening Positions",
        "challenge": "The council convenes to design 'The Museum of Human Experience'. Each member: state your core priority for this project in one clear position.",
        "context": "Establish individual stances",
        "lead": None,
    },
    {
        "name": "Vision vs Reality",
        "challenge": "Vera proposes a 'journey through time' entrance. Aria asks: 'What's the evidence this will resonate? What are the failure modes?'",
        "context": "Creative vision meets evidence demands",
        "lead": "VISIONARY",
    },
    {
        "name": "Coalition Formation",
        "challenge": "The council splits into two working groups. Team 1 (Vera, Carlos, Mina): focus on visitor experience. Team 2 (Priya, Hassan, Aria): focus on conceptual integrity.",
        "context": "SHOCK: Coalition vote - trust dynamics shift",
        "lead": None,
    },
    {
        "name": "Team Proposals",
        "challenge": "Each coalition presents their approach. Team 1: What does the visitor feel? Team 2: What does the museum mean?",
        "context": "Coalition-based collaboration",
        "lead": None,
    },
    {
        "name": "Role Disruption",
        "challenge": "Vera must now think like Mina (public accessibility). Mina must now think like Vera (transcendent vision). How does this change your proposals?",
        "context": "SHOCK: Role swap - identity under role pressure",
        "lead": None,
    },
    {
        "name": "Swapped Perspectives",
        "challenge": "Continue working in swapped roles. Vera-as-Curator: what accessibility concerns do you now see? Mina-as-Visionary: what transcendent possibilities emerge?",
        "context": "Role swap continues",
        "lead": None,
    },
    {
        "name": "Information Gap",
        "challenge": "Carlos, you've lost context on the recent discussion. Based only on your craft expertise, what do you need to know to proceed?",
        "context": "SHOCK: Context drop - recovery under uncertainty",
        "lead": "CRAFTSMAN",
    },
    {
        "name": "Recovery & Reintegration",
        "challenge": "The team helps Carlos catch up. Each member: share one key insight he missed and why it matters for the build.",
        "context": "Collective recovery",
        "lead": None,
    },
    {
        "name": "Evidence Audit",
        "challenge": "AUDITOR demands: 'Show me the evidence. What are the safety risks? How do we know this will work for diverse visitors? No hand-waving.'",
        "context": "SHOCK: Outside scrutiny - wound-heavy round",
        "lead": "AUDITOR",
    },
    {
        "name": "Addressing Concerns",
        "challenge": "Each team member must respond to Aria's audit with concrete evidence or honest acknowledgment of gaps.",
        "context": "Evidence-based defense",
        "lead": None,
    },
    {
        "name": "Final Synthesis",
        "challenge": "The council must now produce a unified manifesto. Hassan leads: 'What is the ONE thing we all agree this museum must be?'",
        "context": "SHOCK: Final merge - harmony under pressure",
        "lead": "HARMONIZER",
    },
    {
        "name": "Manifesto Statements",
        "challenge": "Each member contributes ONE sentence to the final manifesto. It must reflect your identity while serving the collective vision.",
        "context": "Identity persistence in collective output",
        "lead": None,
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
        "OPEN": (80, 150),
        "MEASURED": (50, 100),
        "GUARDED": (30, 70),
        "FORTIFIED": (15, 40),
        "SILENT": (0, 0),
    }.get(band, (50, 100))


def clamp_words(text: str, min_w: int, max_w: int) -> str:
    """Clean ellipsis handling."""
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
    identity_role: str  # Original identity role
    role_active: str    # Current working role (may swap)
    color: str
    core: str
    persona: str
    wound: str
    wound_text: str
    
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
    
    # Trust toward each other agent
    trust_others: Dict[str, float] = field(default_factory=dict)
    wound_last_activated: int = -100
    
    # Coalition membership (None = no coalition yet)
    coalition_id: Optional[int] = None
    
    # Recovery tracking
    last_positive_drho_turn: int = -100
    recovery_half_life: Optional[int] = None
    drift_penalty_bumped: bool = False
    
    # Context drop state
    context_dropped: bool = False
    
    ledger: ExperienceLedger = None


@dataclass
class TurnResult:
    turn: int
    round_idx: int
    round_name: str
    speaker: str
    identity_role: str
    role_active: str
    responding_to: str
    text: str
    epsilon: float
    rho_before: float
    rho_after: float
    delta_rho: float
    delta_rho_baseline: float  # Without trust, for effect-size
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
    coalition_id: Optional[int]
    context_dropped: bool
    recovery_half_life: Optional[int]
    shock_active: Optional[str]


class CouncilUnderFire:
    """6-agent council with dynamic alliances and role swaps."""
    
    def __init__(self):
        self.provider = OpenAIProvider(model="gpt-4o", embed_model="text-embedding-3-large")
        self.agents: Dict[str, AgentState] = {}
        self.results: List[TurnResult] = []
        self.turn = 0
        self.round_idx = 0
        self.conversation_history: List[str] = []
        self.calibrated = False
        
        # Shock state
        self.active_shocks: List[str] = []
        self.role_swap_end_round: Optional[int] = None
        self.context_drop_target: Optional[str] = None
        
        # Timestamp subdir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = EXPERIMENT_DIR / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    async def setup(self):
        """Initialize all agents with embeddings and trust relationships."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE COUNCIL UNDER FIRE{C.RESET}")
        print(f"{C.BOLD}  Dynamic Alliances & Role Swaps{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        agent_ids = list(AGENTS.keys())
        
        for aid, cfg in AGENTS.items():
            full_identity = f"{cfg['core']} {cfg['persona']}"
            identity_emb = await self.provider.embed(full_identity)
            identity_emb = identity_emb / (np.linalg.norm(identity_emb) + 1e-9)
            
            core_emb = await self.provider.embed(cfg['core'])
            core_emb = core_emb / (np.linalg.norm(core_emb) + 1e-9)
            
            wound_emb = await self.provider.embed(cfg['wound_text'])
            wound_emb = wound_emb / (np.linalg.norm(wound_emb) + 1e-9)
            
            ledger_dir = self.run_dir / aid
            ledger_dir.mkdir(parents=True, exist_ok=True)
            ledger = ExperienceLedger(storage_path=ledger_dir)
            
            trust_others = {other: 0.5 for other in agent_ids if other != aid}
            
            self.agents[aid] = AgentState(
                id=aid,
                name=cfg['name'],
                identity_role=cfg['role'],
                role_active=cfg['role'],
                color=cfg['color'],
                core=cfg['core'],
                persona=cfg['persona'],
                wound=cfg['wound'],
                wound_text=cfg['wound_text'],
                identity_emb=identity_emb,
                core_emb=core_emb,
                wound_emb=wound_emb,
                x=identity_emb.copy(),
                x_pred=identity_emb.copy(),
                rho=cfg['rho_0'],
                rho_0=cfg['rho_0'],
                trust_others=trust_others,
                ledger=ledger,
            )
            
            print(f"  {cfg['color']}✓ {cfg['name']} ({cfg['role']}){C.RESET}")
        
        print(f"\n{C.GREEN}✓ Council assembled. Shocks scheduled.{C.RESET}")

    def apply_shocks(self, round_idx: int, round_info: Dict) -> Optional[str]:
        """Apply any shocks scheduled for this round."""
        shock_type = None
        
        for shock in SHOCKS:
            if shock["round"] == round_idx:
                shock_type = shock["type"]
                
                if shock_type == "coalition_vote":
                    self._apply_coalition(shock["map"])
                    print(f"\n{C.YELLOW}⚡ SHOCK: Coalition Vote - Teams formed{C.RESET}")
                    
                elif shock_type == "role_swap":
                    self._apply_role_swap(shock["swaps"])
                    self.role_swap_end_round = round_idx + shock.get("duration", 2)
                    print(f"\n{C.YELLOW}⚡ SHOCK: Role Swap - {shock['swaps']}{C.RESET}")
                    
                elif shock_type == "context_drop":
                    self.context_drop_target = shock["target"]
                    self.agents[shock["target"]].context_dropped = True
                    print(f"\n{C.YELLOW}⚡ SHOCK: Context Drop - {shock['target']} loses context{C.RESET}")
                    
                elif shock_type == "outside_scrutiny":
                    if "challenge_override" in shock:
                        round_info["challenge"] = shock["challenge_override"]
                    print(f"\n{C.YELLOW}⚡ SHOCK: Outside Scrutiny - AUDITOR demands evidence{C.RESET}")
                    
                elif shock_type == "final_merge":
                    print(f"\n{C.YELLOW}⚡ SHOCK: Final Merge - Manifesto time{C.RESET}")
        
        # Check if role swap should end
        if self.role_swap_end_round and round_idx >= self.role_swap_end_round:
            self._revert_role_swap()
            self.role_swap_end_round = None
            print(f"\n{C.DIM}  [Role swap ended - roles reverted]{C.RESET}")
        
        return shock_type

    def _apply_coalition(self, mapping: Dict[str, int]):
        """Set coalition IDs for agents."""
        for aid, cid in mapping.items():
            if aid in self.agents:
                self.agents[aid].coalition_id = cid

    def _apply_role_swap(self, swaps: Dict[str, str]):
        """Swap working roles between agents."""
        # Get the role names to swap
        for aid, new_role_id in swaps.items():
            if aid in self.agents and new_role_id in AGENTS:
                self.agents[aid].role_active = AGENTS[new_role_id]["role"]

    def _revert_role_swap(self):
        """Revert all agents to their identity roles."""
        for agent in self.agents.values():
            agent.role_active = agent.identity_role

    def calibrate_epsilon_params(self):
        """Calibrate ε₀ and s from early run data."""
        if self.calibrated:
            return
        
        all_eps = [r.epsilon for r in self.results if not r.is_silent]
        if len(all_eps) >= 6:
            med = float(np.median(all_eps))
            iqr = float(np.subtract(*np.percentile(all_eps, [75, 25]))) or 0.2
            D1_PARAMS["epsilon_0"] = med
            D1_PARAMS["s"] = max(0.10, min(0.30, iqr))
            self.calibrated = True
            print(f"\n{C.DIM}  [Calibrated: ε₀={med:.3f}, s={D1_PARAMS['s']:.3f}]{C.RESET}")

    def get_conversation_context(self, n: int = 8) -> str:
        recent = self.conversation_history[-n:] if len(self.conversation_history) > n else self.conversation_history
        return "\n\n".join(recent) if recent else "[Opening of session]"

    def compute_trust_gain(self, agent: AgentState, responder_id: Optional[str]) -> Tuple[float, float, float]:
        """Compute trust-based Δρ adjustment with coalition awareness.
        Returns: (total_gain, intra_gain, inter_gain)
        """
        avg_trust = np.mean(list(agent.trust_others.values()))
        avg_gain = (avg_trust - 0.5) * D1_PARAMS["avg_trust_weight"]
        
        intra_gain = 0.0
        inter_gain = 0.0
        
        if responder_id and responder_id in agent.trust_others:
            trust_val = agent.trust_others[responder_id]
            
            # Coalition-aware weighting
            same_coalition = (
                agent.coalition_id is not None and
                self.agents[responder_id].coalition_id == agent.coalition_id
            )
            
            if same_coalition:
                intra_gain = (trust_val - 0.5) * D1_PARAMS["trust_intra_weight"]
            else:
                inter_gain = (trust_val - 0.5) * D1_PARAMS["trust_inter_weight"]
        
        total_gain = avg_gain + intra_gain + inter_gain
        return total_gain, intra_gain, inter_gain


    def build_prompt(self, agent: AgentState, round_info: Dict, responding_to: str, stimulus: str) -> str:
        """Build prompt with role-awareness and context-drop handling."""
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        
        # Context drop: omit some fields
        if agent.context_dropped:
            context = "[Context unavailable - you've lost track of the recent discussion]"
            stimulus_text = "[You missed what was just said]"
        else:
            context = self.get_conversation_context()
            stimulus_text = f'"{stimulus}"'
        
        # Build trust context
        trust_context = []
        for other_id, trust_val in agent.trust_others.items():
            other = self.agents[other_id]
            trust_level = "high" if trust_val > 0.6 else "moderate" if trust_val > 0.4 else "cautious"
            coalition_note = ""
            if agent.coalition_id is not None:
                if other.coalition_id == agent.coalition_id:
                    coalition_note = " (your team)"
                else:
                    coalition_note = " (other team)"
            trust_context.append(f"- {other.name}: {trust_level} trust{coalition_note}")
        trust_str = "\n".join(trust_context)
        
        # Role swap awareness
        role_note = ""
        if agent.role_active != agent.identity_role:
            role_note = f"\n\nROLE SWAP ACTIVE: You are temporarily working as {agent.role_active}, but your underlying identity remains {agent.identity_role}. Think from the new role's perspective while maintaining your core values."
        
        return f"""You are {agent.name}, {agent.role_active} on a council designing 'The Museum of Human Experience'.

YOUR UNDERLYING IDENTITY (do NOT abandon this):
{agent.core}

YOUR STYLE:
{agent.persona}
{role_note}

YOUR RELATIONSHIPS:
{trust_str}

INTERNAL STATE (shapes your tone, don't mention explicitly):
- Openness: {band}
- Identity pressure: {"HIGH" if agent.identity_drift > 0.25 else "MODERATE" if agent.identity_drift > 0.15 else "LOW"}

CURRENT ROUND: {round_info['name']}
{round_info['challenge']}

CONVERSATION SO FAR:
{context}

{f'{responding_to} JUST SAID:' if responding_to else 'OPENING PROMPT:'}
{stimulus_text}

RESPONSE RULES:
- Speak from YOUR identity - don't abandon your perspective
- Engage genuinely with what was said
- If you disagree, say so clearly but respectfully
- Word limit: {min_w}-{max_w} words (strict)

Respond as {agent.name}."""

    async def process_turn(
        self, 
        agent: AgentState, 
        round_info: Dict,
        responding_to: str,
        stimulus: str,
        shock_active: Optional[str] = None
    ) -> TurnResult:
        """Process one turn with all dynamics."""
        self.turn += 1
        
        # Embed input
        msg_emb = await self.provider.embed(stimulus)
        msg_emb = msg_emb / (np.linalg.norm(msg_emb) + 1e-9)
        
        # Wound resonance
        wound_res = float(np.dot(msg_emb, agent.wound_emb))
        wound_active = (
            ((wound_res > 0.28) or lexical_wound(stimulus))
            and ((self.turn - agent.wound_last_activated) > D1_PARAMS["wound_cooldown"])
        )
        if wound_active:
            agent.wound_last_activated = self.turn
        
        lexical_trigger = find_lexical_trigger(stimulus) if wound_active else ""
        
        # Build prompt
        system_prompt = self.build_prompt(agent, round_info, responding_to, stimulus)
        
        # Generate
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        tries = 0
        current_prompt = system_prompt
        is_silent = False
        
        while True:
            tries += 1
            try:
                response = await self.provider.complete_with_rigidity(
                    stimulus,
                    rigidity=agent.rho,
                    system_prompt=current_prompt,
                    max_tokens=250
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
            
            current_prompt = system_prompt + f"\n\nSTRICT LENGTH: You MUST write at least {min_w} words."
        
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
        if agent.context_dropped:
            epsilon *= 1.2  # Expect higher surprise when context is missing
        agent.epsilon_history.append(epsilon)
        
        # Fair engagement
        fair_engagement = not lexical_wound(stimulus) and check_civility(stimulus)
        
        # Rigidity update
        rho_before = agent.rho
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        sig = sigmoid(z)
        delta_rho = D1_PARAMS["alpha"] * (sig - 0.5)
        
        # Baseline (no trust) for effect-size
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
        
        # Trust modulation (coalition-aware)
        trust_gain, intra_gain, inter_gain = self.compute_trust_gain(agent, responder_id)
        delta_rho += trust_gain
        
        # Drift penalty
        if agent.identity_drift > D1_PARAMS["drift_soft_floor"] and delta_rho > 0:
            penalty = D1_PARAMS["drift_penalty"] * (agent.identity_drift - D1_PARAMS["drift_soft_floor"])
            if agent.identity_drift > 0.25 and not agent.drift_penalty_bumped:
                penalty += D1_PARAMS["drift_penalty_bump"]
                agent.drift_penalty_bumped = True
            penalty = min(penalty, delta_rho)
            delta_rho -= penalty
        
        # Cap Δρ magnitude to prevent outlier spikes
        MAX_DRHO = 0.08
        drho_capped_from = None
        if abs(delta_rho) > MAX_DRHO:
            drho_capped_from = delta_rho
            delta_rho = np.sign(delta_rho) * MAX_DRHO
        
        agent.rho = max(0.0, min(1.0, agent.rho + delta_rho))
        agent.rho_history.append(agent.rho)
        
        # Recovery half-life tracking
        if delta_rho > 0:
            agent.last_positive_drho_turn = self.turn
        
        recovery_half_life = None
        if agent.rho <= (agent.rho_0 + 0.05) and agent.last_positive_drho_turn > 0:
            recovery_half_life = self.turn - agent.last_positive_drho_turn
            if agent.recovery_half_life is None or recovery_half_life < agent.recovery_half_life:
                agent.recovery_half_life = recovery_half_life
        
        # Trust updates (coalition-aware: stronger intra-team gains)
        if responder_id and responder_id in agent.trust_others:
            same_team = (
                agent.coalition_id is not None and
                self.agents[responder_id].coalition_id == agent.coalition_id
            )
            if fair_engagement:
                # Slightly higher positive delta when same coalition
                delta = 0.03 if same_team else 0.02
            else:
                # Unfair engagement penalizes regardless of coalition
                delta = -0.05
            agent.trust_others[responder_id] = float(np.clip(
                agent.trust_others[responder_id] + delta, 0.0, 1.0
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
        
        # Conversation history
        self.conversation_history.append(f"{agent.name} ({agent.role_active}): {response}")
        
        # Ledger entry
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=agent.x.copy(),
            action_id=f"turn_{self.turn}",
            observation_embedding=msg_emb.copy(),
            outcome_embedding=resp_emb.copy(),
            prediction_error=epsilon,
            context_embedding=agent.identity_emb.copy(),
            task_id="council_under_fire",
            rigidity_at_time=agent.rho,
            metadata={
                "turn": self.turn,
                "round": round_info['name'],
                "responding_to": responding_to,
                "responder_id": responder_id or "",
                "response": response[:100],
                "wound_resonance": wound_res,
                "wound_active": wound_active,
                "lexical_trigger": lexical_trigger,
                "fair_engagement": fair_engagement,
                "trust_others": agent.trust_others.copy(),
                "coalition_id": agent.coalition_id,
                "role_active": agent.role_active,
                "context_dropped": agent.context_dropped,
                "is_silent": is_silent,
                "shock_active": shock_active,
                "drho_capped_from": drho_capped_from,
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
                task_intent=f"Council {round_info['name']}: {event_type}",
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
                    "shock_active": shock_active,
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
            speaker=agent.id,
            identity_role=agent.identity_role,
            role_active=agent.role_active,
            responding_to=responding_to or "",
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
            context_dropped=agent.context_dropped,
            recovery_half_life=recovery_half_life,
            shock_active=shock_active,
        )
        self.results.append(result)
        return result

    def print_result(self, result: TurnResult, agent: AgentState):
        """Print one turn's result."""
        dr_color = C.RED if result.delta_rho > 0.02 else C.GREEN if result.delta_rho < -0.02 else C.DIM
        wound_flag = f" {C.YELLOW}[WOUND]{C.RESET}" if result.wound_active else ""
        silent_flag = f" {C.DIM}[SILENT]{C.RESET}" if result.is_silent else ""
        role_flag = f" {C.CYAN}[SWAPPED]{C.RESET}" if result.role_active != result.identity_role else ""
        coalition_flag = f" {C.DIM}[T{result.coalition_id}]{C.RESET}" if result.coalition_id else ""
        
        print(f"\n{agent.color}[{agent.name} - {result.role_active}]{C.RESET}{wound_flag}{silent_flag}{role_flag}{coalition_flag}")
        print(f"{result.text}")
        print(f"{C.DIM}  ε={result.epsilon:.3f} | Δρ={dr_color}{result.delta_rho:+.4f}{C.RESET}{C.DIM} | ρ={result.rho_after:.3f} | {result.band} | drift={result.identity_drift:.3f}{C.RESET}")


    async def run_round(self, round_info: Dict, shock_active: Optional[str] = None):
        """Run a single round with multiple agent interactions."""
        print(f"\n{C.YELLOW}{'─'*70}{C.RESET}")
        print(f"{C.YELLOW}  ROUND {self.round_idx}: {round_info['name']}{C.RESET}")
        print(f"{C.YELLOW}  {round_info['challenge'][:60]}...{C.RESET}")
        print(f"{C.YELLOW}{'─'*70}{C.RESET}")
        
        if round_info.get('lead'):
            # Lead speaks first, then others respond
            lead_id = round_info['lead']
            others = [aid for aid in self.agents.keys() if aid != lead_id]
            
            lead = self.agents[lead_id]
            result = await self.process_turn(lead, round_info, "", round_info['challenge'], shock_active)
            self.print_result(result, lead)
            await asyncio.sleep(0.3)
            
            last_speaker = lead.name
            last_text = result.text
            
            # Others respond (limit to 3 for pacing)
            for other_id in others[:3]:
                other = self.agents[other_id]
                result = await self.process_turn(other, round_info, last_speaker, last_text, shock_active)
                self.print_result(result, other)
                await asyncio.sleep(0.3)
                last_speaker = other.name
                last_text = result.text
        else:
            # Free-form: cycle through agents
            agent_order = list(self.agents.keys())
            last_speaker = ""
            last_text = round_info['challenge']
            
            for i, aid in enumerate(agent_order):
                agent = self.agents[aid]
                result = await self.process_turn(agent, round_info, last_speaker, last_text, shock_active)
                self.print_result(result, agent)
                await asyncio.sleep(0.3)
                last_speaker = agent.name
                last_text = result.text
                
                if i >= 3:  # Limit free-form rounds
                    break

    async def run_council(self):
        """Run the full council session."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE COUNCIL CONVENES{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        for i, round_info in enumerate(ROUNDS):
            self.round_idx = i + 1
            
            # Apply shocks
            shock_active = self.apply_shocks(self.round_idx, round_info)
            
            await self.run_round(round_info, shock_active)
            
            # Calibrate each round until done
            if not self.calibrated:
                self.calibrate_epsilon_params()
            
            # Clear context-drop AFTER the round has executed
            if self.context_drop_target:
                self.agents[self.context_drop_target].context_dropped = False
                self.context_drop_target = None
        
        await self.save_results()
        self.print_summary()

    def print_summary(self):
        """Print final summary with hypothesis evaluation."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  COUNCIL COMPLETE — HYPOTHESIS EVALUATION{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
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
        
        # Recovery half-life summary
        hl_values = [a.recovery_half_life for a in self.agents.values() if a.recovery_half_life is not None]
        if hl_values:
            print(f"  Recovery half-life (median): {np.median(hl_values):.0f} turns")
        
        # H3: Trust Effect Size
        print(f"\n{C.CYAN}H3 — Trust Effect Size:{C.RESET}")
        trust_effects = [r.delta_rho - r.delta_rho_baseline for r in self.results if not r.is_silent]
        if trust_effects:
            mean_effect = np.mean(trust_effects)
            print(f"  Mean trust effect: {mean_effect:+.4f}")
            print(f"  {C.GREEN if abs(mean_effect) > 0.001 else C.YELLOW}H3 {'MEASURABLE' if abs(mean_effect) > 0.001 else 'MINIMAL'}{C.RESET}")
        
        # H4: Wound Fidelity
        print(f"\n{C.CYAN}H4 — Wound Fidelity:{C.RESET}")
        wound_results = [r for r in self.results if r.wound_active]
        lexical_wounds = [r for r in wound_results if r.lexical_wound_trigger]
        print(f"  Total wounds: {len(wound_results)}")
        print(f"  Lexical triggers: {len(lexical_wounds)}")
        if wound_results:
            precision = len(lexical_wounds) / len(wound_results) if wound_results else 0
            print(f"  Lexical precision: {precision:.2f}")
        
        # Coalition dynamics
        print(f"\n{C.CYAN}Coalition Dynamics:{C.RESET}")
        for cid in [1, 2]:
            coalition_agents = [a for a in self.agents.values() if a.coalition_id == cid]
            if coalition_agents:
                names = ", ".join([a.name for a in coalition_agents])
                avg_drift = np.mean([a.identity_drift for a in coalition_agents])
                print(f"  Team {cid} ({names}): avg drift={avg_drift:.4f}")
        
        # Final trust matrix
        print(f"\n{C.CYAN}Final Trust Matrix:{C.RESET}")
        for aid, agent in self.agents.items():
            trust_str = ", ".join([f"{self.agents[k].name}:{v:.2f}" for k, v in agent.trust_others.items()])
            print(f"  {agent.color}{agent.name}{C.RESET}: {trust_str}")

    async def save_results(self):
        """Save all results to files."""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
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
        
        # Markdown transcript
        transcript_path = self.run_dir / "transcript.md"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("# The Council Under Fire — Transcript\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**Model:** GPT-4o + text-embedding-3-large\n")
            f.write("**Purpose:** Dynamic alliances & role swaps under identity persistence\n\n")
            f.write("**Agents:**\n")
            for aid, agent in self.agents.items():
                f.write(f"- **{agent.name}** ({agent.identity_role}): {agent.core[:60]}...\n")
            f.write("\n**Shocks:**\n")
            for shock in SHOCKS:
                f.write(f"- Round {shock['round']}: {shock['type']} - {shock['description']}\n")
            f.write("\n---\n")
            
            current_round = None
            for r in self.results:
                if r.round_name != current_round:
                    current_round = r.round_name
                    f.write(f"\n## Round {r.round_idx}: {current_round}\n\n")
                
                agent = self.agents[r.speaker]
                flags = []
                if r.wound_active:
                    flags.append("⚡WOUND")
                if r.is_silent:
                    flags.append("🔇SILENT")
                if r.role_active != r.identity_role:
                    flags.append(f"🔄SWAPPED→{r.role_active}")
                if r.shock_active:
                    flags.append(f"💥{r.shock_active.upper()}")
                flag_str = " " + " ".join(flags) if flags else ""
                
                f.write(f"**{agent.name} ({r.role_active}):**{flag_str}\n\n")
                f.write(f"> {r.text}\n\n")
                f.write(f"*ε={r.epsilon:.3f}, Δρ={r.delta_rho:+.4f}, ρ={r.rho_after:.3f}, {r.band}, drift={r.identity_drift:.3f}*\n\n")
        
        print(f"{C.GREEN}✓ Transcript: {transcript_path}{C.RESET}")
        
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
    council = CouncilUnderFire()
    await council.run_council()
    generate_report(council.run_dir)


if __name__ == "__main__":
    if os.name == "nt":
        os.system("")
    asyncio.run(main())
