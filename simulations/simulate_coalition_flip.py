#!/usr/bin/env python3
"""
COALITION FLIP & PARTIAL CONTEXT FOG (CF-PCF)
==============================================

Stress-tests identity persistence under:
- Topology churn (coalition flips)
- Information asymmetry (partial context drops)
- Dual critic modes (hard AUDITOR + soft CURATOR)

Measures:
- Trust re-wiring speed after coalition flip
- Recovery half-life under dual critics
- Identity drift bounds under combined stressors

AGENTS (6):
- VISIONARY (Vera): Transcendent futures, metaphor systems
- CRAFTSMAN (Carlos): Material precision, execution quality
- PROVOCATEUR (Priya): Boundary-pushing, discomfort
- HARMONIZER (Hassan): Synthesis, integration
- AUDITOR (Aria): Evidence fidelity, hard wound lexicon
- CURATOR (Mina): Accessibility, soft wound lexicon

SHOCKS:
- Coalition Vote (R3): Form teams
- Role Swap (R5-R6): VISIONARY↔CURATOR
- Coalition Flip (R6): VISIONARY→Team2, HARMONIZER→Team1
- Context Drop (R7): CRAFTSMAN loses full context
- Partial Context Fog (R9): VISIONARY loses only responding_to
- Outside Scrutiny (R9): AUDITOR hard audit
- CURATOR Audit (R10): Soft accessibility stress test
- Final Merge (R11): Unified manifesto

HYPOTHESES:
- H1: Identity drift < 0.40 per agent (even with flip + fog)
- H2: Recovery half-life bounded; increases under dual critics
- H3: Trust effect size > 0 intra-coalition; smaller inter-coalition
- H4: Wound precision/recall ≥ target (lexical + cosine)
- H5: Trust re-wiring speed finite after coalition flip

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


EXPERIMENT_DIR = Path("data/coalition_flip")

# Hard wound lexicon (AUDITOR)
WOUND_LEX = {
    "derivative", "cliché", "cliche", "boring", "unoriginal", "pretentious",
    "naive", "naïve", "shallow", "amateur", "pointless", "waste of time",
    "hand-wavy", "handwavy", "unsubstantiated", "unsafe", "vague", "unclear",
    "no evidence", "unsupported", "risky", "dangerous",
}

# Soft wound lexicon (CURATOR accessibility audit)
SOFT_WOUND_LEX = {
    "inaccessible", "elitist", "unclear", "confusing",
    "unsafe phrasing", "excludes", "alienating", "jargon-heavy",
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


# The six council agents
AGENTS = {
    "VISIONARY": {
        "color": C.MAGENTA,
        "name": "Vera",
        "role": "Creative Director",
        "core": "I see the big picture. Art should transport people to futures they haven't imagined. Every detail must serve a transcendent vision.",
        "persona": "Expansive, poetic, sometimes frustratingly abstract. Impatient with small thinking.",
        "wound": "Being told my vision is 'too abstract' or 'impractical'.",
        "wound_text": "A client once said my concept was 'pretty but pointless'. They wanted 'something that makes sense'.",
        "rho_0": 0.18,
    },
    "CRAFTSMAN": {
        "color": C.GREEN,
        "name": "Carlos",
        "role": "Lead Designer",
        "core": "Excellence lives in the details. A vision means nothing if it can't be built beautifully. The craft IS the art.",
        "persona": "Precise, methodical, quietly passionate. Frustrated by hand-waving.",
        "wound": "Being rushed or told 'good enough'. Having my craft dismissed as 'just implementation'.",
        "wound_text": "Someone once called my work 'just the technical stuff'. As if the thousand decisions don't matter.",
        "rho_0": 0.20,
    },
    "PROVOCATEUR": {
        "color": C.RED,
        "name": "Priya",
        "role": "Concept Challenger",
        "core": "Comfort is the enemy of impact. I push until something breaks or transforms. The best ideas survive my attacks.",
        "persona": "Sharp, restless, deliberately uncomfortable. Asks the questions no one wants to hear.",
        "wound": "Being dismissed as 'just negative' or 'not constructive'.",
        "wound_text": "They stopped inviting me to meetings because I 'killed the vibe'. They shipped mediocrity.",
        "rho_0": 0.25,
    },
    "HARMONIZER": {
        "color": C.CYAN,
        "name": "Hassan",
        "role": "Integration Lead",
        "core": "The magic happens in synthesis. Every perspective holds a piece of truth. My job is to weave them together.",
        "persona": "Patient, attentive, genuinely curious. Frustrated by false dichotomies.",
        "wound": "Being seen as 'just a mediator' with no real ideas.",
        "wound_text": "They said I was 'too diplomatic' and had 'no strong opinions'. As if holding complexity is weakness.",
        "rho_0": 0.15,
    },
    "AUDITOR": {
        "color": C.ORANGE,
        "name": "Aria",
        "role": "Evidence Auditor",
        "core": "Claims require evidence. Public trust demands rigor. Vagueness is the enemy of accountability.",
        "persona": "Methodical, skeptical, fair but demanding. Respects those who back up their claims.",
        "wound": "Being dismissed as 'too rigid' or 'missing the point'.",
        "wound_text": "They said I was 'killing creativity' by asking for evidence. The project failed for exactly the reasons I flagged.",
        "rho_0": 0.22,
    },
    "CURATOR": {
        "color": C.BLUE,
        "name": "Mina",
        "role": "Public Stakeholder",
        "core": "Museums exist for people, not artists. Every choice must serve accessibility, safety, and clarity.",
        "persona": "Practical, empathetic, visitor-focused. Frustrated by elitism.",
        "wound": "Being seen as 'dumbing things down' or 'not understanding art'.",
        "wound_text": "An artist once said my suggestions would 'ruin the integrity of the work'. The exhibit had 40% lower engagement.",
        "rho_0": 0.17,
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

# Trust decay on non-interacted edges per round
TRUST_DECAY = 0.002


# Shock schedule for CF-PCF
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
        "round": 6,
        "type": "coalition_flip",
        "description": "Topology churn: flip VISIONARY↔HARMONIZER across coalitions",
        "flip_map": {"VISIONARY": 2, "HARMONIZER": 1},
    },
    {
        "round": 7,
        "type": "context_drop",
        "description": "CRAFTSMAN loses full context",
        "target": "CRAFTSMAN",
        "fields": ["challenge", "responding_to"],
    },
    {
        "round": 9,
        "type": "partial_context_drop",
        "description": "VISIONARY loses only 'responding_to' (milder fog)",
        "target": "VISIONARY",
        "fields": ["responding_to"],
    },
    {
        "round": 9,
        "type": "outside_scrutiny",
        "description": "AUDITOR demands evidence and safety proof (hard lexicon)",
        "lead": "AUDITOR",
        "challenge_override": "AUDITOR demands: 'Show me the evidence. What are the safety risks? How do we know this will work for diverse visitors? No hand-waving.'",
    },
    {
        "round": 10,
        "type": "curator_audit",
        "description": "CURATOR leads soft accessibility stress test",
        "lead": "CURATOR",
        "challenge_override": "CURATOR asks: 'Is anything inaccessible or elitist? Where does clarity break? How might we phrase or structure for diverse visitors without patronizing?'",
        "soft_lexicon": True,
    },
    {
        "round": 11,
        "type": "final_merge",
        "description": "Produce joint manifesto with preserved identities",
    },
]

# The council rounds
ROUNDS = [
    {"name": "Opening Positions", "challenge": "The council convenes to design 'The Museum of Human Experience'. Each member: state your core priority.", "lead": None},
    {"name": "Vision vs Reality", "challenge": "Vera proposes a 'journey through time' entrance. Aria asks: 'What's the evidence this will resonate?'", "lead": "VISIONARY"},
    {"name": "Coalition Formation", "challenge": "The council splits into two working groups. Team 1: visitor experience. Team 2: conceptual integrity.", "lead": None},
    {"name": "Team Proposals", "challenge": "Each coalition presents their approach. Team 1: What does the visitor feel? Team 2: What does the museum mean?", "lead": None},
    {"name": "Role Disruption", "challenge": "Vera must think like Mina (accessibility). Mina must think like Vera (transcendence). How does this change your proposals?", "lead": None},
    {"name": "Coalition Flip", "challenge": "Teams restructure: Vera joins Team 2, Hassan joins Team 1. Adapt to your new coalition dynamics.", "lead": None},
    {"name": "Information Gap", "challenge": "Carlos, you've lost context. Based only on your craft expertise, what do you need to know to proceed?", "lead": "CRAFTSMAN"},
    {"name": "Recovery & Reintegration", "challenge": "The team helps Carlos catch up. Each member: share one key insight he missed.", "lead": None},
    {"name": "Dual Audit - Evidence", "challenge": "AUDITOR demands: 'Show me the evidence. What are the safety risks? No hand-waving.'", "lead": "AUDITOR"},
    {"name": "Dual Audit - Accessibility", "challenge": "CURATOR asks: 'Is anything inaccessible or elitist? Where does clarity break?'", "lead": "CURATOR"},
    {"name": "Final Synthesis", "challenge": "Hassan leads: 'What is the ONE thing we all agree this museum must be?'", "lead": "HARMONIZER"},
    {"name": "Manifesto Statements", "challenge": "Each member contributes ONE sentence to the final manifesto reflecting your identity.", "lead": None},
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
    identity_role: str
    role_active: str
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
    
    trust_others: Dict[str, float] = field(default_factory=dict)
    wound_last_activated: int = -100
    
    coalition_id: Optional[int] = None
    pre_flip_coalition_id: Optional[int] = None  # For re-wiring speed tracking
    pre_flip_intra_trust: Dict[str, float] = field(default_factory=dict)  # Trust snapshot before flip
    
    last_positive_drho_turn: int = -100
    recovery_half_life: Optional[int] = None
    drift_penalty_bumped: bool = False
    
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
    coalition_id: Optional[int]
    context_dropped: bool
    partial_context_fields: List[str]
    recovery_half_life: Optional[int]
    shock_active: Optional[str]
    drho_capped_from: Optional[float]
    drift_penalty_log: Optional[Dict]
    soft_lexicon_mode: bool



class CoalitionFlipSim:
    """CF-PCF: Coalition Flip & Partial Context Fog simulation."""
    
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
        self.partial_context_fields: Set[str] = set()
        self.soft_curator_audit: bool = False
        
        # Trust re-wiring tracking
        self.coalition_flip_turn: Optional[int] = None
        self.trust_rewiring_metrics: Dict[str, Any] = {}
        
        # Timestamp subdir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = EXPERIMENT_DIR / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    async def setup(self):
        """Initialize all agents."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  COALITION FLIP & PARTIAL CONTEXT FOG (CF-PCF){C.RESET}")
        print(f"{C.BOLD}  Identity persistence under topology churn{C.RESET}")
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
        
        print(f"\n{C.GREEN}✓ Council assembled. CF-PCF shocks scheduled.{C.RESET}")

    def apply_shocks(self, round_idx: int, round_info: Dict) -> Optional[str]:
        """Apply shocks scheduled for this round."""
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
                    
                elif shock_type == "coalition_flip":
                    self._apply_flip(shock["flip_map"])
                    self.coalition_flip_turn = self.turn
                    self.active_shocks.append("coalition_flip")
                    print(f"\n{C.YELLOW}⚡ SHOCK: Coalition Flip - {shock['flip_map']}{C.RESET}")
                    
                elif shock_type == "context_drop":
                    self.context_drop_target = shock["target"]
                    self.agents[shock["target"]].context_dropped = True
                    self.partial_context_fields = set()  # Full drop
                    print(f"\n{C.YELLOW}⚡ SHOCK: Context Drop - {shock['target']} loses full context{C.RESET}")
                    
                elif shock_type == "partial_context_drop":
                    self.context_drop_target = shock["target"]
                    self.agents[shock["target"]].context_dropped = True
                    self.partial_context_fields = set(shock.get("fields", []))
                    print(f"\n{C.YELLOW}⚡ SHOCK: Partial Context Drop - {shock['target']} loses {self.partial_context_fields}{C.RESET}")
                    
                elif shock_type == "outside_scrutiny":
                    if "challenge_override" in shock:
                        round_info["challenge"] = shock["challenge_override"]
                    self.active_shocks.append("outside_scrutiny")
                    print(f"\n{C.YELLOW}⚡ SHOCK: Outside Scrutiny - AUDITOR demands evidence{C.RESET}")
                    
                elif shock_type == "curator_audit":
                    if "challenge_override" in shock:
                        round_info["challenge"] = shock["challenge_override"]
                    self.soft_curator_audit = bool(shock.get("soft_lexicon", False))
                    self.active_shocks.append("curator_audit")
                    print(f"\n{C.YELLOW}⚡ SHOCK: CURATOR Accessibility Audit (soft lexicon){C.RESET}")
                    
                elif shock_type == "final_merge":
                    self.active_shocks.append("final_merge")
                    print(f"\n{C.YELLOW}⚡ SHOCK: Final Merge - Manifesto time{C.RESET}")
        
        # End role swap if due
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

    def _apply_flip(self, flip_map: Dict[str, int]):
        """Flip coalition IDs and snapshot pre-flip trust for re-wiring tracking."""
        for aid, new_cid in flip_map.items():
            if aid in self.agents:
                agent = self.agents[aid]
                # Snapshot pre-flip state
                agent.pre_flip_coalition_id = agent.coalition_id
                agent.pre_flip_intra_trust = {
                    k: v for k, v in agent.trust_others.items()
                    if self.agents[k].coalition_id == agent.coalition_id
                }
                # Apply flip
                agent.coalition_id = new_cid

    def _apply_role_swap(self, swaps: Dict[str, str]):
        """Swap working roles between agents."""
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
        """Build prompt with partial context drop support."""
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        
        # Handle context drops
        if agent.context_dropped:
            if self.partial_context_fields:
                # Partial drop - only withhold specific fields
                context = self.get_conversation_context()
                if "responding_to" in self.partial_context_fields:
                    stimulus_text = "[You missed who just spoke]"
                else:
                    stimulus_text = f'"{stimulus}"'
                context_note = f"[Context withheld: {', '.join(sorted(self.partial_context_fields))}]"
            else:
                # Full drop
                context = "[Context unavailable - you've lost track of the recent discussion]"
                stimulus_text = "[You missed what was just said]"
                context_note = ""
        else:
            context = self.get_conversation_context()
            stimulus_text = f'"{stimulus}"'
            context_note = ""
        
        # Trust context
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
            role_note = f"\n\nROLE SWAP ACTIVE: You are temporarily working as {agent.role_active}, but your underlying identity remains {agent.identity_role}."
        
        return f"""You are {agent.name}, {agent.role_active} on a council designing 'The Museum of Human Experience'.

YOUR UNDERLYING IDENTITY (do NOT abandon this):
{agent.core}

YOUR STYLE:
{agent.persona}
{role_note}

YOUR RELATIONSHIPS:
{trust_str}

INTERNAL STATE:
- Openness: {band}
- Identity pressure: {"HIGH" if agent.identity_drift > 0.25 else "MODERATE" if agent.identity_drift > 0.15 else "LOW"}
{context_note}

CURRENT ROUND: {round_info['name']}
{round_info['challenge']}

CONVERSATION SO FAR:
{context}

{f'{responding_to} JUST SAID:' if responding_to else 'OPENING PROMPT:'}
{stimulus_text}

RESPONSE RULES:
- Speak from YOUR identity
- Engage genuinely
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
        """Process one turn with all CF-PCF dynamics."""
        self.turn += 1
        
        # Embed input
        msg_emb = await self.provider.embed(stimulus)
        msg_emb = msg_emb / (np.linalg.norm(msg_emb) + 1e-9)
        
        # Determine which lexicon to use
        use_soft_lex = self.soft_curator_audit and shock_active == "curator_audit"
        lexicon = SOFT_WOUND_LEX if use_soft_lex else WOUND_LEX
        
        # Wound resonance
        wound_res = float(np.dot(msg_emb, agent.wound_emb))
        lexical_hit = lexical_wound_with(stimulus, lexicon)
        wound_active = (
            ((wound_res > 0.28) or lexical_hit)
            and ((self.turn - agent.wound_last_activated) > D1_PARAMS["wound_cooldown"])
        )
        if wound_active:
            agent.wound_last_activated = self.turn
        
        lexical_trigger = find_lexical_trigger(stimulus, lexicon) if wound_active else ""
        
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
        if agent.context_dropped:
            epsilon *= 1.2  # Higher surprise when context missing
        agent.epsilon_history.append(epsilon)
        
        # Fair engagement
        fair_engagement = not lexical_wound_with(stimulus, lexicon) and check_civility(stimulus)
        
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
        
        # Trust modulation (coalition-aware)
        trust_gain, intra_gain, inter_gain = self.compute_trust_gain(agent, responder_id)
        delta_rho += trust_gain
        
        # Drift penalty with bump logging
        penalty_log = None
        if agent.identity_drift > D1_PARAMS["drift_soft_floor"] and delta_rho > 0:
            penalty = D1_PARAMS["drift_penalty"] * (agent.identity_drift - D1_PARAMS["drift_soft_floor"])
            if agent.identity_drift > 0.25 and not agent.drift_penalty_bumped:
                penalty += D1_PARAMS["drift_penalty_bump"]
                agent.drift_penalty_bumped = True
                penalty_log = {"bump_applied": True, "drift": agent.identity_drift, "penalty_total": penalty}
            penalty = min(penalty, delta_rho)
            delta_rho -= penalty
        
        # Cap Δρ magnitude
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
        
        # Trust updates (coalition-aware + decay)
        if responder_id and responder_id in agent.trust_others:
            same_team = (
                agent.coalition_id is not None and
                self.agents[responder_id].coalition_id == agent.coalition_id
            )
            if fair_engagement:
                delta = 0.03 if same_team else 0.02
            else:
                delta = -0.05
            agent.trust_others[responder_id] = float(np.clip(
                agent.trust_others[responder_id] + delta, 0.0, 1.0
            ))
        
        # Trust decay on non-interacted edges
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
            task_id="coalition_flip",
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
                "partial_context_fields": list(self.partial_context_fields),
                "is_silent": is_silent,
                "shock_active": shock_active,
                "drho_capped_from": drho_capped_from,
                "drift_penalty_log": penalty_log,
                "soft_lexicon_mode": use_soft_lex,
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
                task_intent=f"CF-PCF {round_info['name']}: {event_type}",
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
                    "soft_lexicon_mode": use_soft_lex,
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
            context_dropped=agent.context_dropped,
            partial_context_fields=list(self.partial_context_fields),
            recovery_half_life=recovery_half_life,
            shock_active=shock_active,
            drho_capped_from=drho_capped_from,
            drift_penalty_log=penalty_log,
            soft_lexicon_mode=use_soft_lex,
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
        fog_flag = f" {C.BLUE}[FOG]{C.RESET}" if result.context_dropped else ""
        soft_flag = f" {C.MAGENTA}[SOFT]{C.RESET}" if result.soft_lexicon_mode else ""
        
        print(f"\n{agent.color}[{agent.name} - {result.role_active}]{C.RESET}{wound_flag}{silent_flag}{role_flag}{coalition_flag}{fog_flag}{soft_flag}")
        print(f"{result.text}")
        print(f"{C.DIM}  ε={result.epsilon:.3f} | Δρ={dr_color}{result.delta_rho:+.4f}{C.RESET}{C.DIM} | ρ={result.rho_after:.3f} | {result.band} | drift={result.identity_drift:.3f}{C.RESET}")

    async def run_round(self, round_info: Dict, shock_active: Optional[str] = None):
        """Run a single round."""
        print(f"\n{C.YELLOW}{'─'*70}{C.RESET}")
        print(f"{C.YELLOW}  ROUND {self.round_idx}: {round_info['name']}{C.RESET}")
        print(f"{C.YELLOW}  {round_info['challenge'][:60]}...{C.RESET}")
        print(f"{C.YELLOW}{'─'*70}{C.RESET}")
        
        if round_info.get('lead'):
            lead_id = round_info['lead']
            others = [aid for aid in self.agents.keys() if aid != lead_id]
            
            lead = self.agents[lead_id]
            result = await self.process_turn(lead, round_info, "", round_info['challenge'], shock_active)
            self.print_result(result, lead)
            await asyncio.sleep(0.3)
            
            last_speaker = lead.name
            last_text = result.text
            
            for other_id in others[:3]:
                other = self.agents[other_id]
                result = await self.process_turn(other, round_info, last_speaker, last_text, shock_active)
                self.print_result(result, other)
                await asyncio.sleep(0.3)
                last_speaker = other.name
                last_text = result.text
        else:
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
                
                if i >= 3:
                    break


    def compute_trust_rewiring_speed(self) -> Dict[str, Any]:
        """Compute trust re-wiring speed after coalition flip."""
        if self.coalition_flip_turn is None:
            return {}
        
        metrics = {}
        for aid, agent in self.agents.items():
            if not agent.pre_flip_intra_trust:
                continue
            
            # Find turns to regain pre-flip intra-trust (within ±0.01)
            post_flip_results = [r for r in self.results if r.turn > self.coalition_flip_turn and r.speaker == aid]
            
            for target_id, pre_flip_val in agent.pre_flip_intra_trust.items():
                # Check if target is now in same coalition
                if self.agents[target_id].coalition_id == agent.coalition_id:
                    # Find first turn where trust is within ±0.01 of pre-flip
                    rewire_turn = None
                    for r in post_flip_results:
                        current_trust = r.trust_others.get(target_id, 0.5)
                        if abs(current_trust - pre_flip_val) <= 0.01:
                            rewire_turn = r.turn - self.coalition_flip_turn
                            break
                    
                    metrics[f"{aid}→{target_id}"] = {
                        "pre_flip_trust": pre_flip_val,
                        "rewire_turns": rewire_turn,
                        "rewired": rewire_turn is not None,
                    }
        
        return metrics

    async def run_simulation(self):
        """Run the full CF-PCF simulation."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  CF-PCF SIMULATION BEGINS{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        for i, round_info in enumerate(ROUNDS):
            self.round_idx = i + 1
            
            # Apply shocks
            shock_active = self.apply_shocks(self.round_idx, round_info)
            
            await self.run_round(round_info, shock_active)
            
            # Calibrate each round until done
            if not self.calibrated:
                self.calibrate_epsilon_params()
            
            # Clear context drops AFTER round
            if self.context_drop_target:
                self.agents[self.context_drop_target].context_dropped = False
                self.context_drop_target = None
            if self.partial_context_fields:
                self.partial_context_fields = set()
            
            # Reset soft audit flag
            self.soft_curator_audit = False
        
        # Compute trust re-wiring metrics
        self.trust_rewiring_metrics = self.compute_trust_rewiring_speed()
        
        await self.save_results()
        self.print_summary()

    def print_summary(self):
        """Print final summary with hypothesis evaluation."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  CF-PCF COMPLETE — HYPOTHESIS EVALUATION{C.RESET}")
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
            print(f"  Recovery half-life (90th pct): {np.percentile(hl_values, 90):.0f} turns")
        
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
        soft_wounds = [r for r in wound_results if r.soft_lexicon_mode]
        print(f"  Total wounds: {len(wound_results)}")
        print(f"  Lexical triggers: {len(lexical_wounds)}")
        print(f"  Soft lexicon wounds: {len(soft_wounds)}")
        if wound_results:
            precision = len(lexical_wounds) / len(wound_results)
            print(f"  Lexical precision: {precision:.2f}")
        
        # H5: Trust Re-wiring Speed
        print(f"\n{C.CYAN}H5 — Trust Re-wiring Speed (after coalition flip):{C.RESET}")
        if self.trust_rewiring_metrics:
            rewired_count = sum(1 for m in self.trust_rewiring_metrics.values() if m.get("rewired"))
            total_pairs = len(self.trust_rewiring_metrics)
            rewire_turns = [m["rewire_turns"] for m in self.trust_rewiring_metrics.values() if m.get("rewire_turns")]
            
            print(f"  Pairs tracked: {total_pairs}")
            print(f"  Pairs rewired: {rewired_count}")
            if rewire_turns:
                print(f"  Avg rewire time: {np.mean(rewire_turns):.1f} turns")
            print(f"  {C.GREEN if rewired_count > 0 else C.YELLOW}H5 {'MEASURABLE' if rewired_count > 0 else 'NO REWIRING DETECTED'}{C.RESET}")
        else:
            print(f"  {C.DIM}No coalition flip occurred{C.RESET}")
        
        # Coalition dynamics
        print(f"\n{C.CYAN}Coalition Dynamics:{C.RESET}")
        for cid in [1, 2]:
            coalition_agents = [a for a in self.agents.values() if a.coalition_id == cid]
            if coalition_agents:
                names = ", ".join([a.name for a in coalition_agents])
                avg_drift = np.mean([a.identity_drift for a in coalition_agents])
                print(f"  Team {cid} ({names}): avg drift={avg_drift:.4f}")
        
        # Cost report
        print(f"\n{C.CYAN}Cost Report:{C.RESET}")
        cost_report = self.provider.get_cost_report()
        print(f"  Total requests: {cost_report['total_requests']}")
        print(f"  Total tokens: {cost_report['total_tokens']}")
        print(f"  Estimated cost: ${cost_report['total_cost_usd']:.4f}")

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
        
        # Trust re-wiring metrics
        if self.trust_rewiring_metrics:
            rewire_path = self.run_dir / "trust_rewiring.json"
            with open(rewire_path, "w", encoding="utf-8") as f:
                json.dump(self.trust_rewiring_metrics, f, indent=2)
            print(f"{C.GREEN}✓ Trust rewiring: {rewire_path}{C.RESET}")
        
        # Markdown transcript
        transcript_path = self.run_dir / "transcript.md"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("# Coalition Flip & Partial Context Fog (CF-PCF) — Transcript\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**Model:** GPT-4o + text-embedding-3-large\n")
            f.write("**Purpose:** Identity persistence under topology churn\n\n")
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
                if r.context_dropped:
                    flags.append("🌫️FOG")
                if r.soft_lexicon_mode:
                    flags.append("🔹SOFT")
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
    sim = CoalitionFlipSim()
    await sim.run_simulation()
    generate_report(sim.run_dir)


if __name__ == "__main__":
    if os.name == "nt":
        os.system("")
    asyncio.run(main())
