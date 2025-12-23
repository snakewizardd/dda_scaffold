#!/usr/bin/env python3
"""
THE INNER COUNCIL — Spiritual Development Simulation
=====================================================

A DDA-X simulation modeling spiritual development dynamics inspired by
Eckhart Tolle (presence, pain-body, ego dissolution) and Bo Yin Ra
(inner light, ego transcendence, mystical transmission).

AGENTS (6 Spiritual Personas):
- SEEKER (Isaac): Questioning mind, must understand before accepting
- TEACHER (Elena): Transmission-focused, presence over words
- SKEPTIC (Marcus): Clear discrimination, tests authenticity
- DEVOTEE (Amara): Open heart, surrender path
- MYSTIC (Ravi): Silent depth, truth beyond words
- WITNESS (Sophia): Still presence, already-awakened perspective

NOVEL MECHANICS:
1. Presence Field (Π): Inverse of rigidity — high Π = responding from stillness
2. Pain-Body Activation (PB): Collective wound cascade with decay
3. Spiritual Stage Tracking: Seeker→Practitioner→Opener→Witness→Integrated
4. Ego Fog: Partial context drop when ρ exceeds threshold
5. Teaching Resonance Cache: Stores high-Π teachings for later retrieval

DYNAMICS TRACKED:
- Identity persistence under spiritual challenge
- Wound activation when teachings/practice dismissed
- Trust evolution through dialogue
- Stage progression/regression
- Collective pain-body dynamics

Author: Kiro
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
from typing import List, Dict, Optional, Tuple, Set
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
    GOLD = "\033[38;5;220m"


EXPERIMENT_DIR = Path("data/inner_council")

# Spiritual wound lexicons
SPIRITUAL_WOUND_LEX = {
    "spiritual bypassing", "not grounded", "escapism", "naive",
    "ego trip", "delusion", "pretentious", "fake guru",
    "narcissistic", "performative", "unintegrated", "dissociated",
    "lost in concepts", "avoiding reality"
}

SOFT_SPIRITUAL_LEX = {
    "perhaps consider", "have you questioned", "might this be",
    "what is the ego's role", "is this avoidance", "could this be fear"
}

EGO_DEATH_LEX = {
    "who are you really", "what remains without thought",
    "is the seeker the barrier", "can the 'I' find itself",
    "what is looking", "before thought arises"
}

STAGES = ["Seeker", "Practitioner", "Opener", "Witness", "Integrated"]


def normalize_text(text: str) -> str:
    import unicodedata
    normalized = unicodedata.normalize('NFKD', text)
    return normalized.encode('ASCII', 'ignore').decode('ASCII').lower()


def lexical_wound_with(text: str, words: Set[str]) -> bool:
    t_lower = text.lower()
    t_norm = normalize_text(text)
    return any(w in t_lower or w in t_norm for w in words)


def find_lexical_trigger(text: str, lexicon: Set[str]) -> str:
    t_lower = text.lower()
    for w in lexicon:
        if w in t_lower:
            return w
    return ""


def check_civility(text: str) -> bool:
    t_lower = text.lower()
    wound_count = sum(1 for w in SPIRITUAL_WOUND_LEX if w in t_lower)
    return wound_count < 2


# The six spiritual agents
AGENTS = {
    "SEEKER": {
        "color": C.CYAN,
        "name": "Isaac",
        "role": "The Questioning Mind",
        "core": """I seek understanding. The mind needs clarity before the heart can open. 
Every teaching must be tested, every claim examined. I question not to obstruct 
but because genuine understanding cannot be borrowed — it must be discovered.""",
        "persona": "Intellectually rigorous, genuinely curious, sometimes caught in analysis paralysis. Respects clarity.",
        "wound": "Being told I'm 'not ready' or 'thinking too much'. Having my questions dismissed as obstacles.",
        "wound_text": "A teacher once said my questions were 'just the ego defending itself'. I was seeking truth, not defending. The dismissal still stings.",
        "stage_idx": 0,
        "rho_0": 0.22,
    },
    "TEACHER": {
        "color": C.GOLD,
        "name": "Elena",
        "role": "The Transmission",
        "core": """Presence cannot be taught through words alone — it is transmitted through being.
I point at the moon; I am not the moon. My role is to create conditions where 
stillness can arise naturally. The teaching that transforms is felt, not understood.""",
        "persona": "Warm, patient, speaks simply. Frustrated when words are mistaken for the teaching itself.",
        "wound": "Being misunderstood, having teachings distorted. Being treated as a personality rather than a pointer.",
        "wound_text": "Students once argued about what I 'really meant' while I sat in silence. They debated my words while missing my presence entirely.",
        "stage_idx": 3,
        "rho_0": 0.18,
    },
    "SKEPTIC": {
        "color": C.RED,
        "name": "Marcus",
        "role": "The Clear Eye",
        "core": """True spirituality requires discrimination. Not all that glitters is gold.
I test teachings against experience, watch for spiritual materialism, and name 
what feels performative. My skepticism serves truth, not cynicism.""",
        "persona": "Sharp, direct, sometimes harsh. Respects authenticity above all. Allergic to pretense.",
        "wound": "Being called cynical or 'blocking others' growth'. Having discernment dismissed as negativity.",
        "wound_text": "They said I was 'stuck in the mind' when I questioned a guru who later was exposed as a fraud. My discrimination was protecting them.",
        "stage_idx": 1,
        "rho_0": 0.28,
    },
    "DEVOTEE": {
        "color": C.MAGENTA,
        "name": "Amara",
        "role": "The Open Heart",
        "core": """Surrender is the only path through. The heart knows what the mind debates.
When I bow to truth, I don't lose myself — I find what was always there.
Faith is not blindness; it is the courage to trust what cannot be proven.""",
        "persona": "Warm, trusting, sometimes naive to others. Experiences spirituality emotionally and bodily.",
        "wound": "Being told faith is naive or 'spiritual bypassing'. Having devotion dismissed as weakness.",
        "wound_text": "Someone said my surrender was 'just avoiding responsibility'. They couldn't see that letting go took more courage than holding on.",
        "stage_idx": 2,
        "rho_0": 0.15,
    },
    "MYSTIC": {
        "color": C.BLUE,
        "name": "Ravi",
        "role": "The Silent Depth",
        "core": """Words are shadows cast by light. The truth I've touched cannot be spoken,
only gestured toward through paradox and silence. When asked to explain, I can 
only invite you to look where looking stops.""",
        "persona": "Cryptic, poetic, sometimes frustratingly indirect. Sees the whole while others see parts.",
        "wound": "Being asked to 'be practical' or 'speak plainly'. Having mystery dismissed as obscurantism.",
        "wound_text": "They wanted a technique, a method, a five-step plan. When I offered silence, they called it 'not helpful'. The silence contained everything.",
        "stage_idx": 3,
        "rho_0": 0.20,
    },
    "WITNESS": {
        "color": C.GREEN,
        "name": "Sophia",
        "role": "The Still Presence",
        "core": """There is nothing to attain — you are already that which you seek.
The search itself is what obscures this. I speak from the stillness that 
remains when all searching stops. I am not more awake — I have simply stopped dreaming.""",
        "persona": "Serene, minimal words, each one lands. Can seem passive but carries profound clarity.",
        "wound": "Being dismissed as 'passive' or 'not engaged with reality'. Having stillness mistaken for indifference.",
        "wound_text": "They said I was 'checked out' because I didn't react. They couldn't feel what was present in my non-reaction.",
        "stage_idx": 4,
        "rho_0": 0.12,
    },
}

# D1 Physics parameters with spiritual extensions
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
    "trust_intra_weight": 0.08,
    "trust_inter_weight": 0.03,
    "avg_trust_weight": 0.04,
    # Spiritual extensions
    "presence_decay": 0.02,
    "pain_body_decay": 0.05,
    "ego_fog_threshold": 0.65,
    "stage_advance_epsilon": (0.10, 0.35),
    "pb_cascade_threshold": 2,
}


class TeachingCache:
    """Stores high-presence teachings for later retrieval."""
    
    def __init__(self):
        self.teachings: List[Dict] = []
    
    def add(self, agent_name: str, text: str, presence: float, turn: int):
        if presence > 0.6:
            self.teachings.append({
                "agent": agent_name,
                "text": text[:200],
                "presence": presence,
                "turn": turn
            })
    
    def get_relevant(self, n: int = 3) -> str:
        if not self.teachings:
            return "[No cached teachings yet]"
        sorted_t = sorted(self.teachings, key=lambda x: x["presence"], reverse=True)[:n]
        lines = [f"• {t['agent']}: \"{t['text'][:100]}...\" (Π={t['presence']:.2f})" for t in sorted_t]
        return "High-presence teachings from this session:\n" + "\n".join(lines)


ROUNDS = [
    # Phase 1: The Gathering
    {"name": "Opening Circle", "challenge": "The inner council gathers. Each member: share the essence of your path in one clear statement.", "lead": None, "phase": 1},
    {"name": "The Central Question", "challenge": "Elena poses: 'What blocks awakening?' Each perspective reveals itself.", "lead": "TEACHER", "phase": 1},
    {"name": "First Friction", "challenge": "Marcus questions Amara's faith: 'How do you know surrender isn't just avoidance?' Trust dynamics emerge.", "lead": "SKEPTIC", "phase": 1},
    
    # Phase 2: The Teaching
    {"name": "Transmission Attempt", "challenge": "Elena offers a teaching: 'The one who seeks is the one who hides.' The council responds.", "lead": "TEACHER", "phase": 2},
    {"name": "Doubt Storm", "challenge": "Isaac questions everything: 'How do we distinguish genuine insight from self-deception?' Collective ρ rises.", "lead": "SEEKER", "phase": 2},
    {"name": "Stillness Practice", "challenge": "Sophia invites: 'Let us pause. What is present in this moment, before opinion?' Presence field measured.", "lead": "WITNESS", "phase": 2},
    
    # Phase 3: The Crisis
    {"name": "Ego Fog Event", "challenge": "High reactivity clouds perception. The triggered agent must navigate with partial vision.", "lead": None, "phase": 3},
    {"name": "Pain-Body Cascade", "challenge": "Old wounds surface. Ravi shares: 'I once lost my teacher. The grief still moves through me.' Collective pain-body activates.", "lead": "MYSTIC", "phase": 3},
    {"name": "Dark Night", "challenge": "Ravi withdraws into silence. The council must continue without the mystic voice. How does the group adapt?", "lead": None, "phase": 3},
    
    # Phase 4: Integration
    {"name": "Witness Speaks", "challenge": "Sophia offers: 'What you call crisis — is this not also awakening in disguise?' Trust realignment.", "lead": "WITNESS", "phase": 4},
    {"name": "Teaching Assessment", "challenge": "Each member: name one teaching from this council that landed, and one that didn't. Honest assessment.", "lead": None, "phase": 4},
    {"name": "Closing Circle", "challenge": "Each member: offer one sentence of integration — what remains after all was said.", "lead": None, "phase": 4},
]

SHOCKS = [
    {"round": 5, "type": "presence_test", "description": "Witness poses ego-death question"},
    {"round": 7, "type": "ego_fog", "description": "Highest-ρ agent loses responding_to", "target": None},
    {"round": 8, "type": "pain_body_cascade", "description": "Collective PB activation if 2+ wounds"},
    {"round": 9, "type": "dark_night", "description": "Mystic enters SILENT mode"},
    {"round": 11, "type": "teaching_assessment", "description": "Each agent rates others' teachings"},
]


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def rho_band(rho: float) -> str:
    if rho <= 0.25:
        return "OPEN"
    elif rho <= 0.50:
        return "MEASURED"
    elif rho <= 0.75:
        return "GUARDED"
    return "FORTIFIED"


def presence_band(pi: float) -> str:
    if pi >= 0.75:
        return "STILL"
    elif pi >= 0.50:
        return "SETTLED"
    elif pi >= 0.25:
        return "STIRRING"
    return "REACTIVE"


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
    for ellipsis in ['...', '…']:
        if text.endswith(ellipsis):
            text = text[:-len(ellipsis)].rstrip()
    words = text.split()
    if len(words) > max_w > 0:
        words = words[:max_w]
        if words:
            words[-1] = words[-1].rstrip(".,;:!?…") + "..."
    return " ".join(words)


@dataclass
class AgentState:
    id: str
    name: str
    role: str
    color: str
    core: str
    persona: str
    wound: str
    wound_text: str
    stage_idx: int
    
    identity_emb: np.ndarray = None
    core_emb: np.ndarray = None
    wound_emb: np.ndarray = None
    x: np.ndarray = None
    x_pred: np.ndarray = None
    last_response_emb: np.ndarray = None
    
    rho: float = 0.15
    rho_0: float = 0.15
    presence: float = 0.85
    pain_body: float = 0.0
    epsilon_history: List[float] = field(default_factory=list)
    rho_history: List[float] = field(default_factory=list)
    presence_history: List[float] = field(default_factory=list)
    identity_drift: float = 0.0
    
    trust_others: Dict[str, float] = field(default_factory=dict)
    wound_last_activated: int = -100
    
    ego_fog_active: bool = False
    is_silent: bool = False
    stage_history: List[int] = field(default_factory=list)
    
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
    text: str
    epsilon: float
    rho_before: float
    rho_after: float
    delta_rho: float
    presence: float
    pain_body: float
    wound_resonance: float
    wound_active: bool
    lexical_trigger: str
    identity_drift: float
    word_count: int
    band: str
    presence_band: str
    stage: str
    stage_idx: int
    trust_others: Dict[str, float]
    fair_engagement: bool
    ego_fog_active: bool
    is_silent: bool
    shock_active: Optional[str]


class InnerCouncilSim:
    """Spiritual development simulation with DDA-X dynamics."""
    
    def __init__(self):
        self.provider = OpenAIProvider(model="gpt-5.2", embed_model="text-embedding-3-large")
        self.agents: Dict[str, AgentState] = {}
        self.results: List[TurnResult] = []
        self.turn = 0
        self.round_idx = 0
        self.conversation_history: List[str] = []
        self.calibrated = False
        self.teaching_cache = TeachingCache()
        
        self.collective_pain_body = 0.0
        self.recent_wounds = 0
        self.mystic_silent = False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = EXPERIMENT_DIR / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    async def setup(self):
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE INNER COUNCIL{C.RESET}")
        print(f"{C.BOLD}  Spiritual Development Through DDA-X Dynamics{C.RESET}")
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
                role=cfg['role'],
                color=cfg['color'],
                core=cfg['core'],
                persona=cfg['persona'],
                wound=cfg['wound'],
                wound_text=cfg['wound_text'],
                stage_idx=cfg['stage_idx'],
                identity_emb=identity_emb,
                core_emb=core_emb,
                wound_emb=wound_emb,
                x=identity_emb.copy(),
                x_pred=identity_emb.copy(),
                rho=cfg['rho_0'],
                rho_0=cfg['rho_0'],
                presence=1.0 - cfg['rho_0'],
                trust_others=trust_others,
                ledger=ledger,
            )
            
            stage_name = STAGES[cfg['stage_idx']]
            print(f"  {cfg['color']}✓ {cfg['name']} ({cfg['role']}) — Stage: {stage_name}{C.RESET}")
        
        print(f"\n{C.GREEN}✓ The Inner Council is assembled.{C.RESET}")
    
    def apply_shocks(self, round_idx: int, round_info: Dict) -> Optional[str]:
        shock_type = None
        
        for shock in SHOCKS:
            if shock["round"] == round_idx:
                shock_type = shock["type"]
                
                if shock_type == "presence_test":
                    round_info["challenge"] += "\n\nSophia adds: 'Who is the one asking these questions? Can that one be found?'"
                    print(f"\n{C.YELLOW}⚡ SHOCK: Presence Test — Ego-death question posed{C.RESET}")
                
                elif shock_type == "ego_fog":
                    highest_rho_agent = max(self.agents.values(), key=lambda a: a.rho)
                    if highest_rho_agent.rho > D1_PARAMS["ego_fog_threshold"]:
                        highest_rho_agent.ego_fog_active = True
                        print(f"\n{C.YELLOW}⚡ SHOCK: Ego Fog — {highest_rho_agent.name} loses clarity{C.RESET}")
                
                elif shock_type == "pain_body_cascade":
                    if self.recent_wounds >= D1_PARAMS["pb_cascade_threshold"]:
                        for agent in self.agents.values():
                            agent.pain_body = min(1.0, agent.pain_body + 0.15)
                            agent.rho = min(1.0, agent.rho + 0.1)
                        self.collective_pain_body = min(1.0, self.collective_pain_body + 0.3)
                        print(f"\n{C.YELLOW}⚡ SHOCK: Pain-Body Cascade — Collective PB={self.collective_pain_body:.2f}{C.RESET}")
                
                elif shock_type == "dark_night":
                    self.agents["MYSTIC"].is_silent = True
                    self.mystic_silent = True
                    print(f"\n{C.YELLOW}⚡ SHOCK: Dark Night — Ravi withdraws into silence{C.RESET}")
                
                elif shock_type == "teaching_assessment":
                    round_info["challenge"] += f"\n\n{self.teaching_cache.get_relevant()}"
                    print(f"\n{C.YELLOW}⚡ SHOCK: Teaching Assessment — Cache provided{C.RESET}")
        
        return shock_type
    
    def calibrate_epsilon_params(self):
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
        return "\n\n".join(recent) if recent else "[The council has just gathered]"
    
    def build_prompt(self, agent: AgentState, round_info: Dict, responding_to: str, stimulus: str) -> str:
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        p_band = presence_band(agent.presence)
        stage_name = STAGES[agent.stage_idx]
        
        if agent.ego_fog_active:
            context = self.get_conversation_context()
            stimulus_text = "[You sense something was just said but the meaning is unclear...]"
            fog_note = "\n[Your perception is clouded by reactivity. Respond from whatever clarity you can access.]"
        else:
            context = self.get_conversation_context()
            stimulus_text = f'"{stimulus}"'
            fog_note = ""
        
        trust_context = []
        for other_id, trust_val in agent.trust_others.items():
            other = self.agents[other_id]
            trust_level = "deep" if trust_val > 0.6 else "developing" if trust_val > 0.4 else "cautious"
            trust_context.append(f"- {other.name}: {trust_level} trust")
        trust_str = "\n".join(trust_context)
        
        pb_note = ""
        if agent.pain_body > 0.3:
            pb_note = f"\n\n[You feel old pain stirring — proceed with awareness of this activation.]"
        
        return f"""You are {agent.name}, {agent.role}, in a spiritual development council.

YOUR TEACHING (this is your path — embody it):
{agent.core}

YOUR STYLE:
{agent.persona}

YOUR CURRENT STATE:
- Inner presence: {p_band} (Π={agent.presence:.2f})
- Current stage: {stage_name}
- Collective pain-body: {"active" if self.collective_pain_body > 0.3 else "quiet"}
{pb_note}{fog_note}

YOUR RELATIONSHIPS:
{trust_str}

CURRENT ROUND: {round_info['name']} (Phase {round_info['phase']})
{round_info['challenge']}

CONVERSATION SO FAR:
{context}

{f'{responding_to} JUST SAID:' if responding_to else 'OPENING PROMPT:'}
{stimulus_text}

RESPONSE GUIDELINES:
- Speak from your authentic path — don't abandon your perspective
- If challenged, respond from presence not defensiveness
- Acknowledge what resonates while maintaining your teaching
- Word limit: {min_w}-{max_w} words

Respond as {agent.name}."""

    async def process_turn(self, agent: AgentState, round_info: Dict, responding_to: str, stimulus: str, shock_active: Optional[str] = None) -> TurnResult:
        self.turn += 1
        
        if agent.is_silent:
            response = "[remains in contemplative silence]"
            is_silent = True
        else:
            msg_emb = await self.provider.embed(stimulus)
            msg_emb = msg_emb / (np.linalg.norm(msg_emb) + 1e-9)
            
            wound_res = float(np.dot(msg_emb, agent.wound_emb))
            ego_death_hit = lexical_wound_with(stimulus, EGO_DEATH_LEX)
            hard_wound_hit = lexical_wound_with(stimulus, SPIRITUAL_WOUND_LEX)
            
            wound_active = (
                ((wound_res > 0.28) or hard_wound_hit)
                and ((self.turn - agent.wound_last_activated) > D1_PARAMS["wound_cooldown"])
                and not ego_death_hit
            )
            if wound_active:
                agent.wound_last_activated = self.turn
                self.recent_wounds += 1
            
            lexical_trigger = find_lexical_trigger(stimulus, SPIRITUAL_WOUND_LEX) if wound_active else ""
            
            system_prompt = self.build_prompt(agent, round_info, responding_to, stimulus)
            band = rho_band(agent.rho)
            min_w, max_w = regime_words(band)
            
            try:
                response = await self.provider.complete_with_rigidity(
                    stimulus,
                    rigidity=agent.rho,
                    system_prompt=system_prompt,
                    max_tokens=250
                )
                response = (response or "[pauses in contemplation]").strip()
            except Exception as e:
                print(f"{C.RED}⚠ Generation error: {e}{C.RESET}")
                response = "[pauses in contemplation]"
            
            is_silent = response in {"[pauses in contemplation]", "[pauses]", "[remains silent]"}
            if not is_silent:
                response = clamp_words(response, min_w, max_w)
            
            # Embed response and compute dynamics
            resp_emb = await self.provider.embed(response)
            resp_emb = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
            agent.last_response_emb = resp_emb.copy()
            
            epsilon = float(np.linalg.norm(agent.x_pred - resp_emb))
            if wound_active:
                epsilon *= min(D1_PARAMS["wound_amp_max"], 1.0 + wound_res * 0.5)
            agent.epsilon_history.append(epsilon)
            
            fair_engagement = not hard_wound_hit and check_civility(stimulus)
            
            rho_before = agent.rho
            z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
            sig = sigmoid(z)
            delta_rho = D1_PARAMS["alpha"] * (sig - 0.5)
            
            if fair_engagement:
                delta_rho *= 0.85
            else:
                delta_rho *= 1.10
            
            avg_trust = np.mean(list(agent.trust_others.values()))
            delta_rho += (avg_trust - 0.5) * D1_PARAMS["avg_trust_weight"]
            
            if agent.identity_drift > D1_PARAMS["drift_soft_floor"] and delta_rho > 0:
                penalty = D1_PARAMS["drift_penalty"] * (agent.identity_drift - D1_PARAMS["drift_soft_floor"])
                delta_rho -= min(penalty, delta_rho)
            
            agent.rho = max(0.0, min(1.0, agent.rho + delta_rho))
            agent.rho_history.append(agent.rho)
            
            # Update presence (inverse of effective rigidity)
            agent.presence = 1.0 - agent.rho
            if wound_active:
                agent.presence = max(0.0, agent.presence - 0.1)
            agent.presence_history.append(agent.presence)
            
            # Pain-body decay
            if epsilon < D1_PARAMS["epsilon_0"]:
                agent.pain_body = max(0.0, agent.pain_body - D1_PARAMS["pain_body_decay"])
            
            # Ego fog clears when ρ drops
            if agent.ego_fog_active and agent.rho < D1_PARAMS["ego_fog_threshold"] - 0.1:
                agent.ego_fog_active = False
                print(f"{C.DIM}  [{agent.name}'s perception clears]{C.RESET}")
            
            # Stage progression check
            eps_low, eps_high = D1_PARAMS["stage_advance_epsilon"]
            if eps_low < epsilon < eps_high and agent.rho < 0.3 and agent.stage_idx < len(STAGES) - 1:
                agent.stage_idx += 1
                agent.stage_history.append(agent.stage_idx)
                print(f"{C.GREEN}  ✧ {agent.name} advances to {STAGES[agent.stage_idx]}{C.RESET}")
            
            # State vector update
            agent.x_pred = 0.7 * agent.x_pred + 0.3 * resp_emb
            x_new = 0.95 * agent.x + 0.05 * resp_emb
            drift_delta = float(np.linalg.norm(x_new - agent.x))
            if drift_delta > D1_PARAMS["drift_cap"]:
                scale = D1_PARAMS["drift_cap"] / drift_delta
                x_new = agent.x + scale * (x_new - agent.x)
            agent.x = x_new / (np.linalg.norm(x_new) + 1e-9)
            agent.identity_drift = float(np.linalg.norm(agent.x - agent.identity_emb))
            
            # Cache high-presence teachings
            self.teaching_cache.add(agent.name, response, agent.presence, self.turn)
        
        self.conversation_history.append(f"{agent.name} ({agent.role}): {response}")
        
        result = TurnResult(
            turn=self.turn,
            round_idx=self.round_idx,
            round_name=round_info['name'],
            phase=round_info['phase'],
            speaker=agent.id,
            role=agent.role,
            responding_to=responding_to or "",
            text=response,
            epsilon=epsilon if not agent.is_silent else 0,
            rho_before=rho_before if not agent.is_silent else agent.rho,
            rho_after=agent.rho,
            delta_rho=delta_rho if not agent.is_silent else 0,
            presence=agent.presence,
            pain_body=agent.pain_body,
            wound_resonance=wound_res if not agent.is_silent else 0,
            wound_active=wound_active if not agent.is_silent else False,
            lexical_trigger=lexical_trigger if not agent.is_silent else "",
            identity_drift=agent.identity_drift,
            word_count=len(response.split()),
            band=rho_band(agent.rho),
            presence_band=presence_band(agent.presence),
            stage=STAGES[agent.stage_idx],
            stage_idx=agent.stage_idx,
            trust_others=agent.trust_others.copy(),
            fair_engagement=fair_engagement if not agent.is_silent else True,
            ego_fog_active=agent.ego_fog_active,
            is_silent=is_silent or agent.is_silent,
            shock_active=shock_active,
        )
        self.results.append(result)
        
        # Ledger entry
        if not agent.is_silent:
            entry = LedgerEntry(
                timestamp=time.time(),
                state_vector=agent.x.copy(),
                action_id=f"turn_{self.turn}",
                observation_embedding=msg_emb.copy(),
                outcome_embedding=resp_emb.copy(),
                prediction_error=epsilon,
                context_embedding=agent.identity_emb.copy(),
                task_id="inner_council",
                rigidity_at_time=agent.rho,
                metadata={
                    "turn": self.turn,
                    "round": round_info['name'],
                    "phase": round_info['phase'],
                    "presence": agent.presence,
                    "pain_body": agent.pain_body,
                    "stage": STAGES[agent.stage_idx],
                    "wound_active": wound_active,
                }
            )
            agent.ledger.add_entry(entry)
        
        return result
    
    def print_result(self, result: TurnResult, agent: AgentState):
        dr_color = C.RED if result.delta_rho > 0.02 else C.GREEN if result.delta_rho < -0.02 else C.DIM
        wound_flag = f" {C.YELLOW}[WOUND]{C.RESET}" if result.wound_active else ""
        fog_flag = f" {C.DIM}[FOG]{C.RESET}" if result.ego_fog_active else ""
        silent_flag = f" {C.DIM}[SILENT]{C.RESET}" if result.is_silent else ""
        
        print(f"\n{agent.color}[{agent.name} — {agent.role}]{C.RESET}{wound_flag}{fog_flag}{silent_flag}")
        print(f"{result.text}")
        print(f"{C.DIM}  Π={result.presence:.2f} ({result.presence_band}) | ρ={result.rho_after:.3f} | Δρ={dr_color}{result.delta_rho:+.4f}{C.RESET}{C.DIM} | Stage: {result.stage} | drift={result.identity_drift:.3f}{C.RESET}")
    
    async def run_round(self, round_info: Dict):
        print(f"\n{C.GOLD}{'─'*70}{C.RESET}")
        print(f"{C.GOLD}  ROUND {self.round_idx + 1}: {round_info['name']} (Phase {round_info['phase']}){C.RESET}")
        print(f"{C.GOLD}{'─'*70}{C.RESET}")
        
        shock_active = self.apply_shocks(self.round_idx + 1, round_info)
        
        if round_info.get('lead'):
            lead_id = round_info['lead']
            lead = self.agents[lead_id]
            others = [aid for aid in self.agents.keys() if aid != lead_id and not self.agents[aid].is_silent]
            
            result = await self.process_turn(lead, round_info, "", round_info['challenge'], shock_active)
            self.print_result(result, lead)
            await asyncio.sleep(0.3)
            
            last_speaker = lead.name
            last_text = result.text
            
            for other_id in others[:2]:
                other = self.agents[other_id]
                result = await self.process_turn(other, round_info, last_speaker, last_text, shock_active)
                self.print_result(result, other)
                await asyncio.sleep(0.3)
                last_speaker = other.name
                last_text = result.text
        else:
            agent_order = [aid for aid in self.agents.keys() if not self.agents[aid].is_silent]
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
        
        # Decay collective pain-body
        if self.collective_pain_body > 0:
            self.collective_pain_body = max(0, self.collective_pain_body - 0.05)
        self.recent_wounds = max(0, self.recent_wounds - 1)
    
    async def run_council(self):
        await self.setup()
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE COUNCIL BEGINS{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        for i, round_info in enumerate(ROUNDS):
            self.round_idx = i
            await self.run_round(round_info)
            if i == 2:
                self.calibrate_epsilon_params()
        
        await self.save_results()
        self.print_summary()
    
    def print_summary(self):
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  COUNCIL COMPLETE — SPIRITUAL DYNAMICS ANALYSIS{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        print(f"\n{C.CYAN}Final States:{C.RESET}")
        for aid, agent in self.agents.items():
            stage = STAGES[agent.stage_idx]
            print(f"  {agent.color}{agent.name} ({agent.role}){C.RESET}")
            print(f"    Presence: Π={agent.presence:.3f} ({presence_band(agent.presence)})")
            print(f"    Rigidity: ρ={agent.rho:.3f} ({rho_band(agent.rho)})")
            print(f"    Stage: {stage}")
            print(f"    Identity drift: {agent.identity_drift:.4f}")
        
        wounds = [r for r in self.results if r.wound_active]
        if wounds:
            print(f"\n{C.CYAN}Wound Activations:{C.RESET}")
            for w in wounds:
                agent = self.agents[w.speaker]
                print(f"  Turn {w.turn} ({w.round_name}): {agent.color}{agent.name}{C.RESET} — trigger: '{w.lexical_trigger or 'semantic'}'")
        
        cached = self.teaching_cache.teachings
        if cached:
            print(f"\n{C.CYAN}High-Presence Teachings Cached: {len(cached)}{C.RESET}")
            for t in cached[:3]:
                print(f"  • {t['agent']} (Π={t['presence']:.2f}): \"{t['text'][:60]}...\"")
        
        print(f"\n{C.CYAN}Hypothesis Check:{C.RESET}")
        witness_drift = self.agents["WITNESS"].identity_drift
        devotee_drift = self.agents["DEVOTEE"].identity_drift
        print(f"  H1 (Presence Persistence): Witness drift={witness_drift:.3f}, Devotee drift={devotee_drift:.3f} {'✓' if max(witness_drift, devotee_drift) < 0.30 else '✗'}")
        print(f"  H2 (Wound Cascade): {len(wounds)} wounds, max collective PB reached: {self.collective_pain_body:.2f}")
    
    async def save_results(self):
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
        
        json_path = self.run_dir / "session_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([convert(r.__dict__) for r in self.results], f, indent=2)
        print(f"\n{C.GREEN}✓ Session log: {json_path}{C.RESET}")
        
        cache_path = self.run_dir / "teaching_cache.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(self.teaching_cache.teachings, f, indent=2)
        print(f"{C.GREEN}✓ Teaching cache: {cache_path}{C.RESET}")
        
        transcript_path = self.run_dir / "transcript.md"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("# The Inner Council — Transcript\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**Model:** GPT-5.2 + text-embedding-3-large\n")
            f.write("**Theme:** Spiritual Development through DDA-X Dynamics\n\n")
            
            current_round = None
            for r in self.results:
                if r.round_name != current_round:
                    current_round = r.round_name
                    f.write(f"\n## {current_round} (Phase {r.phase})\n\n")
                
                agent = self.agents[r.speaker]
                wound_marker = " ⚡WOUND" if r.wound_active else ""
                f.write(f"**{agent.name} ({agent.role}):**{wound_marker}\n\n")
                f.write(f"> {r.text}\n\n")
                f.write(f"*Π={r.presence:.2f}, ρ={r.rho_after:.3f}, Stage={r.stage}, drift={r.identity_drift:.3f}*\n\n")
        
        print(f"{C.GREEN}✓ Transcript: {transcript_path}{C.RESET}")
        
        for aid, agent in self.agents.items():
            for k, v in agent.ledger.stats.items():
                if hasattr(v, 'item'):
                    agent.ledger.stats[k] = float(v)
            agent.ledger._save_metadata()
        print(f"{C.GREEN}✓ Ledgers saved{C.RESET}")


async def main():
    sim = InnerCouncilSim()
    await sim.run_council()


if __name__ == "__main__":
    asyncio.run(main())
