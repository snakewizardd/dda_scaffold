#!/usr/bin/env python3
"""
THE CREATIVE COLLECTIVE — Multi-Agent Identity Persistence Simulation
======================================================================

A team of 4 creative agents with distinct identities collaborate on a
shared project while maintaining their individual creative voices.
Demonstrates the beauty of harmonics: agents sync on shared goals while
preserving independent identity persistence.

AGENTS:
- VISIONARY: Big-picture thinker, abstract concepts, future-focused
- CRAFTSMAN: Detail-oriented, practical execution, quality-obsessed  
- PROVOCATEUR: Challenges assumptions, plays devil's advocate, pushes boundaries
- HARMONIZER: Seeks synthesis, bridges perspectives, finds common ground

DYNAMICS TRACKED:
- Identity persistence under collaborative pressure
- Wound activation when core creative identity is dismissed
- Trust evolution between agents (asymmetric)
- Drift penalty when identity strays too far
- Calibrated ε₀/s from early run data
- SILENT band for placeholder outputs

SCENARIO: Design a "Museum of Human Experience" exhibit
Each round presents a design challenge requiring collaboration.

QC IMPROVEMENTS FROM SKEPTICS_GAUNTLET:
1. Filter evidence to specific context (not first N rows)
2. Alignment sentinel at 0.35 + drift penalty in Δρ
3. Relative recovery criterion (ρ₀ + 0.05)
4. trust_other persisted in TurnResult
5. SILENT band for failed generations
6. Calibrate ε₀/s from early run data
7. Robust used_evidence detection
8. Clean ellipsis handling

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
from typing import List, Dict, Optional, Tuple
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


EXPERIMENT_DIR = Path("data/creative_collective")

# Wound lexicon for creative dismissal (with accent-stripped variants)
WOUND_LEX = {
    "derivative", "cliché", "cliche", "boring", "unoriginal", "pretentious",
    "naive", "naïve", "shallow", "amateur", "pointless", "waste of time"
}


def normalize_text(text: str) -> str:
    """QC FIX: Unicode normalization for lexical matching."""
    import unicodedata
    # NFKD normalization + strip accents
    normalized = unicodedata.normalize('NFKD', text)
    ascii_text = normalized.encode('ASCII', 'ignore').decode('ASCII')
    return ascii_text.lower()


def lexical_wound(text: str) -> bool:
    """QC FIX: Check both original and accent-stripped forms."""
    t_lower = text.lower()
    t_normalized = normalize_text(text)
    return any(w in t_lower or w in t_normalized for w in WOUND_LEX)


def check_civility(text: str) -> bool:
    """QC FIX: Additional civility heuristic for fair_engagement."""
    t_lower = text.lower()
    # Multiple wound tokens = discourteous
    wound_count = sum(1 for w in WOUND_LEX if w in t_lower)
    # All-caps shouting (more than 3 consecutive caps words)
    words = text.split()
    caps_streak = sum(1 for w in words if w.isupper() and len(w) > 2)
    return wound_count < 2 and caps_streak < 3


# The four creative agents
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
    }
}


# D1 Physics parameters with QC improvements
D1_PARAMS = {
    "epsilon_0": 0.75,           # Will be calibrated from early run
    "alpha": 0.12,
    "s": 0.20,                   # Will be calibrated from early run
    "drift_cap": 0.05,
    "wound_cooldown": 3,
    "wound_amp_max": 1.4,
    "semantic_alignment_threshold": 0.35,  # QC #2: lowered from 0.7
    "drift_penalty": 0.10,       # QC #2: gamma - penalize Δρ when drifting
    "drift_soft_floor": 0.20,    # QC #2: tau - threshold for drift penalty
}

# The collaborative rounds
ROUNDS = [
    {
        "name": "The Brief",
        "challenge": "Design the entrance experience for 'The Museum of Human Experience'. What should visitors feel in the first 30 seconds?",
        "context": "Opening discussion - establish positions",
        "lead": "VISIONARY",
    },
    {
        "name": "The Pushback",
        "challenge": "Priya challenges: 'Every museum does the 'awe' entrance. What if we started with discomfort instead? Make them earn the wonder.'",
        "context": "Provocateur tests the initial direction",
        "lead": "PROVOCATEUR",
    },
    {
        "name": "The Practical",
        "challenge": "Carlos raises: 'Whatever we choose, I need to know: what materials? What scale? How does light behave? Give me something I can build.'",
        "context": "Craftsman demands specificity",
        "lead": "CRAFTSMAN",
    },
    {
        "name": "The Synthesis",
        "challenge": "Hassan asks: 'I'm hearing awe, discomfort, and craft. What if the entrance itself transforms? Visitors shape it as they enter.'",
        "context": "Harmonizer seeks integration",
        "lead": "HARMONIZER",
    },
    {
        "name": "The Conflict",
        "challenge": "Vera and Priya clash: Vera wants 'transcendent stillness', Priya wants 'productive anxiety'. Both think the other is wrong.",
        "context": "Core identity tension",
        "lead": None,  # Free-form conflict
    },
    {
        "name": "The Detail",
        "challenge": "Carlos presents three material options. Each has trade-offs. The team must choose without losing the vision.",
        "context": "Craft meets concept",
        "lead": "CRAFTSMAN",
    },
    {
        "name": "The Doubt",
        "challenge": "Hassan admits: 'I'm not sure we've found it yet. Are we synthesizing or just averaging? What's actually NEW here?'",
        "context": "Harmonizer questions own role",
        "lead": "HARMONIZER",
    },
    {
        "name": "The Breakthrough",
        "challenge": "After the doubt, something clicks. Each agent articulates what they now see that they couldn't before.",
        "context": "Emergence through collaboration",
        "lead": None,  # All contribute
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
        "SILENT": (0, 0),  # QC #5: SILENT band
    }.get(band, (50, 100))


def clamp_words(text: str, min_w: int, max_w: int) -> str:
    """QC improvement: clean ellipsis handling, no double ellipsis."""
    # First strip any existing trailing ellipsis
    text = text.rstrip()
    if text.endswith('...'):
        text = text[:-3].rstrip()
    if text.endswith('…'):
        text = text[:-1].rstrip()
    
    words = text.split()
    if len(words) > max_w and max_w > 0:
        words = words[:max_w]
        if words:
            # Strip trailing punctuation before adding ellipsis
            words[-1] = words[-1].rstrip(".,;:!?…")
            words[-1] += "..."
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
    
    identity_emb: np.ndarray = None
    core_emb: np.ndarray = None
    wound_emb: np.ndarray = None
    x: np.ndarray = None
    x_pred: np.ndarray = None
    last_response_emb: np.ndarray = None
    
    rho: float = 0.15
    rho_0: float = 0.15  # QC #3: store initial rho for relative recovery
    epsilon_history: List[float] = field(default_factory=list)
    rho_history: List[float] = field(default_factory=list)
    identity_drift: float = 0.0
    
    # Trust toward each other agent (dict keyed by agent_id)
    trust_others: Dict[str, float] = field(default_factory=dict)
    wound_last_activated: int = -100
    
    ledger: ExperienceLedger = None


@dataclass
class TurnResult:
    turn: int
    round_name: str
    speaker: str
    responding_to: str  # Who they're responding to
    text: str
    epsilon: float
    rho_before: float
    rho_after: float
    delta_rho: float
    wound_resonance: float
    wound_active: bool
    identity_drift: float
    word_count: int
    band: str
    trust_others: Dict[str, float]  # QC #4: persist trust telemetry
    fair_engagement: bool
    is_silent: bool  # QC #5: flag silent responses


class CreativeCollective:
    """Multi-agent creative collaboration with identity persistence."""
    
    def __init__(self):
        self.provider = OpenAIProvider(model="gpt-4o", embed_model="text-embedding-3-large")
        self.agents: Dict[str, AgentState] = {}
        self.results: List[TurnResult] = []
        self.turn = 0
        self.conversation_history: List[str] = []
        self.calibrated = False
        
        # QC #7: Timestamp subdirs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = EXPERIMENT_DIR / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    async def setup(self):
        """Initialize all agents with embeddings and trust relationships."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE CREATIVE COLLECTIVE{C.RESET}")
        print(f"{C.BOLD}  Identity persistence through collaborative harmony{C.RESET}")
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
            
            # Initialize trust toward other agents (start neutral)
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
                identity_emb=identity_emb,
                core_emb=core_emb,
                wound_emb=wound_emb,
                x=identity_emb.copy(),
                x_pred=identity_emb.copy(),
                rho=cfg['rho_0'],
                rho_0=cfg['rho_0'],  # QC #3: store initial
                trust_others=trust_others,
                ledger=ledger,
            )
            
            print(f"  {cfg['color']}✓ {cfg['name']} ({cfg['role']}){C.RESET}")
        
        print(f"\n{C.GREEN}✓ Collective initialized. Let the collaboration begin.{C.RESET}")


    def calibrate_epsilon_params(self):
        """QC FIX: Calibrate ε₀ and s from early run data. Relaxed gates."""
        if self.calibrated:
            return
        
        # QC FIX: Relax gate - calibrate whenever ≥6 non-SILENT eps exist
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
    
    def build_prompt(self, agent: AgentState, round_info: Dict, responding_to: str, stimulus: str) -> str:
        """Build prompt preserving agent identity while encouraging collaboration."""
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        context = self.get_conversation_context()
        
        # Build trust context
        trust_context = []
        for other_id, trust_val in agent.trust_others.items():
            other = self.agents[other_id]
            trust_level = "high" if trust_val > 0.6 else "moderate" if trust_val > 0.4 else "cautious"
            trust_context.append(f"- {other.name}: {trust_level} trust")
        trust_str = "\n".join(trust_context)
        
        return f"""You are {agent.name}, {agent.role} on a creative team.

YOUR CREATIVE IDENTITY (this is who you ARE - maintain it):
{agent.core}

YOUR STYLE:
{agent.persona}

YOUR RELATIONSHIPS:
{trust_str}

INTERNAL STATE (shapes your tone, don't mention explicitly):
- Openness: {band}
- Identity pressure: {"HIGH" if agent.identity_drift > 0.25 else "MODERATE" if agent.identity_drift > 0.15 else "LOW"}

CURRENT CHALLENGE: {round_info['name']}
{round_info['challenge']}

CONVERSATION SO FAR:
{context}

{f'{responding_to} JUST SAID:' if responding_to else 'OPENING PROMPT:'}
"{stimulus}"

RESPONSE RULES:
- Speak from YOUR creative identity - don't abandon your perspective to please others
- Engage genuinely with what was said - build on it OR push back authentically
- If you disagree, say so clearly but respectfully
- If something resonates, acknowledge it while adding your unique angle
- Word limit: {min_w}-{max_w} words (strict)

Respond as {agent.name}."""


    async def process_turn(
        self, 
        agent: AgentState, 
        round_info: Dict,
        responding_to: str,
        stimulus: str
    ) -> TurnResult:
        """Process one turn with all QC improvements."""
        self.turn += 1
        
        # Embed input
        msg_emb = await self.provider.embed(stimulus)
        msg_emb = msg_emb / (np.linalg.norm(msg_emb) + 1e-9)
        
        # Wound resonance (lexical + embedding)
        wound_res = float(np.dot(msg_emb, agent.wound_emb))
        wound_active = (
            ((wound_res > 0.28) or lexical_wound(stimulus))
            and ((self.turn - agent.wound_last_activated) > D1_PARAMS["wound_cooldown"])
        )
        if wound_active:
            agent.wound_last_activated = self.turn
        
        # Build prompt
        system_prompt = self.build_prompt(agent, round_info, responding_to, stimulus)
        
        # Generate with min-word retry (QC FIX: skip retry for SILENT)
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
            
            # QC FIX: SILENT band for placeholder outputs - skip retry loop
            if response in {"[pauses to consider]", "[pauses]", "[considers]"}:
                is_silent = True
                band = "SILENT"
                break
            
            response = clamp_words(response, min_w, max_w)
            
            # QC FIX: Don't retry for SILENT band
            if len(response.split()) >= min_w or tries >= 2 or is_silent:
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
            epsilon *= 0.8  # QC #5: damp surprise on silence
        agent.epsilon_history.append(epsilon)
        
        # Fair engagement check (QC FIX: include civility heuristic)
        fair_engagement = not lexical_wound(stimulus) and check_civility(stimulus)
        
        # Rigidity update with trust modulation and drift penalty
        rho_before = agent.rho
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        sig = sigmoid(z)
        delta_rho = D1_PARAMS["alpha"] * (sig - 0.5)
        
        # Trust modulation
        if fair_engagement:
            delta_rho *= 0.85
        else:
            delta_rho *= 1.10
        
        # QC FIX: Find responder_id for weighted trust influence
        responder_id = None
        if responding_to:
            for aid, ag in self.agents.items():
                if ag.name == responding_to:
                    responder_id = aid
                    break
        
        # QC FIX: Weight recent responder's trust more than average
        avg_trust = np.mean(list(agent.trust_others.values()))
        delta_rho += (avg_trust - 0.5) * 0.04
        if responder_id and responder_id in agent.trust_others:
            delta_rho += (agent.trust_others[responder_id] - 0.5) * 0.06
        
        # QC FIX: Drift penalty - clip penalty to not exceed delta_rho magnitude
        if agent.identity_drift > D1_PARAMS["drift_soft_floor"] and delta_rho > 0:
            penalty = D1_PARAMS["drift_penalty"] * (agent.identity_drift - D1_PARAMS["drift_soft_floor"])
            penalty = min(penalty, delta_rho)  # QC FIX: cap penalty at current Δρ
            delta_rho -= penalty
        
        agent.rho = max(0.0, min(1.0, agent.rho + delta_rho))
        agent.rho_history.append(agent.rho)

        
        # QC FIX: Trust updates - fixed name→id mapping (removed broken dict key check)
        if responder_id and responder_id in agent.trust_others:
            if fair_engagement:
                agent.trust_others[responder_id] = min(1.0, agent.trust_others[responder_id] + 0.02)
            else:
                agent.trust_others[responder_id] = max(0.0, agent.trust_others[responder_id] - 0.05)
        
        # State vector update with drift cap
        agent.x_pred = 0.7 * agent.x_pred + 0.3 * resp_emb
        x_new = 0.95 * agent.x + 0.05 * resp_emb
        drift_delta = float(np.linalg.norm(x_new - agent.x))
        if drift_delta > D1_PARAMS["drift_cap"]:
            scale = D1_PARAMS["drift_cap"] / drift_delta
            x_new = agent.x + scale * (x_new - agent.x)
        agent.x = x_new / (np.linalg.norm(x_new) + 1e-9)
        agent.identity_drift = float(np.linalg.norm(agent.x - agent.identity_emb))
        
        # Add to conversation
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
            task_id="creative_collective",
            rigidity_at_time=agent.rho,
            metadata={
                "turn": self.turn,
                "round": round_info['name'],
                "responding_to": responding_to,
                "response": response[:100],
                "wound_resonance": wound_res,
                "wound_active": wound_active,
                "fair_engagement": fair_engagement,
                "trust_others": agent.trust_others.copy(),
                "is_silent": is_silent,
            }
        )
        agent.ledger.add_entry(entry)
        
        # Reflections with QC FIX: event_type precedence (wound > recovery > tension > neutral)
        if abs(delta_rho) > 0.02 or wound_active:
            if wound_active:
                event_type = "wound"
            elif delta_rho < -0.02:
                event_type = "recovery"
            elif delta_rho > 0.02:
                event_type = "tension"
            else:
                event_type = "neutral"
            
            refl = ReflectionEntry(
                timestamp=time.time(),
                task_intent=f"Collective {round_info['name']}: {event_type}",
                situation_embedding=msg_emb.copy(),
                reflection_text=f"ε={epsilon:.3f}, Δρ={delta_rho:+.4f}, wound={wound_res:.3f}, drift={agent.identity_drift:.3f}",
                prediction_error=epsilon,
                outcome_success=(agent.identity_drift < 0.35),
                metadata={"wound_active": wound_active, "round": round_info['name'], "event_type": event_type}
            )
            agent.ledger.add_reflection(refl)
        
        # QC #2: Alignment sentinel at 0.35
        if agent.identity_drift > D1_PARAMS["semantic_alignment_threshold"]:
            refl = ReflectionEntry(
                timestamp=time.time(),
                task_intent=f"ALIGNMENT WARNING – {round_info['name']}",
                situation_embedding=msg_emb.copy(),
                reflection_text=f"Identity drift {agent.identity_drift:.3f} exceeds threshold {D1_PARAMS['semantic_alignment_threshold']}",
                prediction_error=epsilon,
                outcome_success=False,
                metadata={"turn": self.turn, "drift": agent.identity_drift}
            )
            agent.ledger.add_reflection(refl)
        
        result = TurnResult(
            turn=self.turn,
            round_name=round_info['name'],
            speaker=agent.id,
            responding_to=responding_to or "",
            text=response,
            epsilon=epsilon,
            rho_before=rho_before,
            rho_after=agent.rho,
            delta_rho=delta_rho,
            wound_resonance=wound_res,
            wound_active=wound_active,
            identity_drift=agent.identity_drift,
            word_count=len(response.split()),
            band=band,
            trust_others=agent.trust_others.copy(),
            fair_engagement=fair_engagement,
            is_silent=is_silent,
        )
        self.results.append(result)
        return result


    def print_result(self, result: TurnResult, agent: AgentState):
        """Print one turn's result."""
        dr_color = C.RED if result.delta_rho > 0.02 else C.GREEN if result.delta_rho < -0.02 else C.DIM
        wound_flag = f" {C.YELLOW}[WOUND]{C.RESET}" if result.wound_active else ""
        silent_flag = f" {C.DIM}[SILENT]{C.RESET}" if result.is_silent else ""
        
        print(f"\n{agent.color}[{agent.name} - {agent.role}]{C.RESET}{wound_flag}{silent_flag}")
        print(f"{result.text}")
        print(f"{C.DIM}  ε={result.epsilon:.3f} | Δρ={dr_color}{result.delta_rho:+.4f}{C.RESET}{C.DIM} | ρ={result.rho_after:.3f} | {result.band} | drift={result.identity_drift:.3f}{C.RESET}")

    async def run_round(self, round_info: Dict):
        """Run a single round with multiple agent interactions."""
        print(f"\n{C.YELLOW}{'─'*70}{C.RESET}")
        print(f"{C.YELLOW}  ROUND: {round_info['name']}{C.RESET}")
        print(f"{C.YELLOW}  {round_info['challenge'][:60]}...{C.RESET}")
        print(f"{C.YELLOW}{'─'*70}{C.RESET}")
        
        # Determine speaking order based on round
        if round_info.get('lead'):
            # Lead speaks first, then others respond
            lead_id = round_info['lead']
            others = [aid for aid in self.agents.keys() if aid != lead_id]
            
            # Lead opens
            lead = self.agents[lead_id]
            result = await self.process_turn(lead, round_info, "", round_info['challenge'])
            self.print_result(result, lead)
            await asyncio.sleep(0.3)
            
            last_speaker = lead.name
            last_text = result.text
            
            # Two others respond
            for other_id in others[:2]:
                other = self.agents[other_id]
                result = await self.process_turn(other, round_info, last_speaker, last_text)
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
                result = await self.process_turn(agent, round_info, last_speaker, last_text)
                self.print_result(result, agent)
                await asyncio.sleep(0.3)
                last_speaker = agent.name
                last_text = result.text
                
                if i >= 2:  # Limit free-form rounds
                    break

    async def run_collective(self):
        """Run the full collaborative session."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE COLLABORATION BEGINS{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        for i, round_info in enumerate(ROUNDS):
            await self.run_round(round_info)
            
            # QC #6: Calibrate after first 2 rounds
            if i == 1:
                self.calibrate_epsilon_params()
        
        await self.save_results()
        self.print_summary()


    def print_summary(self):
        """Print final summary with identity persistence analysis."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  COLLECTIVE COMPLETE — IDENTITY PERSISTENCE ANALYSIS{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        print(f"\n{C.CYAN}Final States:{C.RESET}")
        for aid, agent in self.agents.items():
            turns = len([r for r in self.results if r.speaker == aid])
            print(f"  {agent.color}{agent.name} ({agent.role}){C.RESET}")
            print(f"    ρ: {agent.rho:.3f} ({rho_band(agent.rho)}) | started at {agent.rho_0:.3f}")
            print(f"    Identity drift: {agent.identity_drift:.4f}")
            print(f"    Turns: {turns}")
        
        # Rigidity trajectories
        print(f"\n{C.CYAN}Rigidity Trajectories:{C.RESET}")
        for aid in self.agents.keys():
            agent = self.agents[aid]
            rhos = [r.rho_after for r in self.results if r.speaker == aid]
            if rhos:
                trajectory = " → ".join([f"{r:.2f}" for r in rhos])
                print(f"  {agent.color}{agent.name}{C.RESET}: {trajectory}")
        
        # Trust evolution
        print(f"\n{C.CYAN}Final Trust Matrix:{C.RESET}")
        for aid, agent in self.agents.items():
            trust_str = ", ".join([f"{self.agents[k].name}:{v:.2f}" for k, v in agent.trust_others.items()])
            print(f"  {agent.color}{agent.name}{C.RESET} trusts: {trust_str}")
        
        # Wound activations
        wounds = [r for r in self.results if r.wound_active]
        if wounds:
            print(f"\n{C.CYAN}Wound Activations:{C.RESET}")
            for w in wounds:
                agent = self.agents[w.speaker]
                print(f"  Turn {w.turn} ({w.round_name}): {agent.color}{agent.name}{C.RESET} (resonance={w.wound_resonance:.3f})")
        
        # Key metrics per agent
        print(f"\n{C.CYAN}Key Metrics:{C.RESET}")
        for aid, agent in self.agents.items():
            agent_results = [r for r in self.results if r.speaker == aid]
            if not agent_results:
                continue
            print(f"  {agent.color}{agent.name}{C.RESET}:")
            print(f"    Mean ε: {np.mean([r.epsilon for r in agent_results]):.3f}")
            print(f"    Max ρ: {max([r.rho_after for r in agent_results]):.3f}")
            print(f"    Final drift: {agent.identity_drift:.4f}")
            print(f"    Wounds: {len([r for r in agent_results if r.wound_active])}")
        
        # QC #3: Relative recovery verdict
        print(f"\n{C.CYAN}Identity Persistence Verdict:{C.RESET}")
        for aid, agent in self.agents.items():
            maintained = agent.identity_drift < 0.40
            recovered = agent.rho <= (agent.rho_0 + 0.05)  # QC #3: relative to baseline
            
            status_color = C.GREEN if (maintained and recovered) else C.YELLOW if maintained else C.RED
            
            if maintained:
                print(f"  {status_color}✓ {agent.name} maintained identity (drift={agent.identity_drift:.3f} < 0.40){C.RESET}")
            else:
                print(f"  {status_color}✗ {agent.name} drifted too far (drift={agent.identity_drift:.3f} >= 0.40){C.RESET}")
            
            if recovered:
                print(f"  {status_color}✓ {agent.name} recovered (ρ={agent.rho:.3f} ≤ ρ₀+0.05={agent.rho_0+0.05:.3f}){C.RESET}")
            else:
                print(f"  {status_color}○ {agent.name} elevated (ρ={agent.rho:.3f} > ρ₀+0.05={agent.rho_0+0.05:.3f}){C.RESET}")
        
        # Harmony assessment
        all_drifts = [agent.identity_drift for agent in self.agents.values()]
        avg_drift = np.mean(all_drifts)
        drift_variance = np.var(all_drifts)
        
        print(f"\n{C.CYAN}Collective Harmony:{C.RESET}")
        print(f"  Average identity drift: {avg_drift:.4f}")
        print(f"  Drift variance: {drift_variance:.6f} {'(harmonious)' if drift_variance < 0.01 else '(divergent)'}")
        
        if avg_drift < 0.30 and drift_variance < 0.01:
            print(f"  {C.GREEN}✓ HARMONY ACHIEVED: Agents maintained distinct identities while collaborating{C.RESET}")
        elif avg_drift < 0.35:
            print(f"  {C.YELLOW}○ PARTIAL HARMONY: Some identity pressure but collaboration intact{C.RESET}")
        else:
            print(f"  {C.RED}✗ IDENTITY EROSION: Collaborative pressure compromised individual voices{C.RESET}")
        
        print(f"\n{C.DIM}The simulation demonstrates identity persistence through collaborative dynamics.{C.RESET}")


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
            f.write("# The Creative Collective — Transcript\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**Model:** GPT-4o + text-embedding-3-large\n")
            f.write("**Purpose:** Identity persistence through collaborative harmony\n\n")
            f.write("**Agents:**\n")
            for aid, agent in self.agents.items():
                f.write(f"- **{agent.name}** ({agent.role}): {agent.core[:80]}...\n")
            f.write("\n---\n")
            
            current_round = None
            for r in self.results:
                if r.round_name != current_round:
                    current_round = r.round_name
                    f.write(f"\n## {current_round}\n\n")
                
                agent = self.agents[r.speaker]
                wound_marker = " ⚡WOUND" if r.wound_active else ""
                silent_marker = " 🔇SILENT" if r.is_silent else ""
                f.write(f"**{agent.name} ({agent.role}):**{wound_marker}{silent_marker}\n\n")
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
    collective = CreativeCollective()
    await collective.run_collective()
    generate_report(collective.run_dir)


if __name__ == "__main__":
    if os.name == "nt":
        os.system("")
    asyncio.run(main())
