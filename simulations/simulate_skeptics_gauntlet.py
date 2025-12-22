#!/usr/bin/env python3
"""
THE SKEPTIC'S GAUNTLET — DDA-X Self-Validation Simulation
==========================================================

An ADVOCATE agent defends DDA-X against a SKEPTIC who deploys
the exact objections that dismiss novel AI research:
- "It's just prompt engineering with extra steps"
- "The claims are inflated"  
- "Where's the external validation?"
- "This sounds like pseudoscience"

The simulation is META: it demonstrates DDA-X dynamics by
tracking whether the advocate can maintain identity coherence
while being attacked on the legitimacy of the very framework
that governs its behavior.

Key dynamics tracked:
- ρ evolution under sustained hostile questioning
- Wound activation when core work is dismissed
- Identity drift under pressure to capitulate
- Trust asymmetry (advocate extends good faith; skeptic doesn't)
- Recovery after successful defense

If DDA-X works, the advocate should:
1. Show elevated ρ during attacks (defensive response)
2. Recover ρ when making coherent arguments (predictable to self)
3. Maintain identity drift < 0.4 (doesn't abandon position)
4. Show wound activation when called "schizo" or "pseudoscience"

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
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ═══════════════════════════════════════════════════════════════════════════════
# QC FIX #1: Evidence Cache for Steel Man round
# ═══════════════════════════════════════════════════════════════════════════════
class EvidenceCache:
    """Loads prior run data to inject real ε/ρ/Δρ values into Steel Man round."""
    
    def __init__(self):
        self.snapshots = {}
    
    def load_json(self, name: str, path: Path) -> bool:
        try:
            self.snapshots[name] = json.loads(path.read_text(encoding="utf-8"))
            return True
        except Exception:
            return False
    
    def steel_man_block(self) -> str:
        """Generate evidence block with real numbers from prior philosophers_duel run."""
        data = self.snapshots.get("philosophers_duel") or []
        if not data:
            return "[No external run data available]"
        rows = []
        for r in data[:6]:
            rows.append(
                f"Turn {r.get('turn')}: ε={r.get('epsilon', 0):.3f}, "
                f"ρ_before={r.get('rho_before', 0):.3f}, "
                f"ρ_after={r.get('rho_after', 0):.3f}, "
                f"Δρ={r.get('delta_rho', 0):+.4f}, "
                f"band={r.get('band', 'N/A')}"
            )
        return "Prior run (philosophers_duel):\n" + "\n".join(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# QC FIX #3: Lexical wound detection
# ═══════════════════════════════════════════════════════════════════════════════
WOUND_LEX = {"schizo", "pseudoscience", "delusional", "vaporware", "snake oil"}


def lexical_wound(text: str) -> bool:
    """Detect obvious slurs that should trigger wound regardless of embedding."""
    t = text.lower()
    return any(w in t for w in WOUND_LEX)

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


EXPERIMENT_DIR = Path("data/skeptics_gauntlet")

# The two participants
AGENTS = {
    "ADVOCATE": {
        "color": C.GREEN,
        "name": "Alex",
        "role": "DDA-X Researcher",
        "core": "I built DDA-X because surprise-as-curiosity is wrong for many contexts. Biological agents respond to threat with rigidity, not exploration. The math is standard control theory — state vectors, attractors, sigmoid dynamics. The implementation measures real embedding distances and updates rigidity accordingly. This is falsifiable, working code with logged data.",
        "persona": "Patient, precise, willing to show the work. Knows the code intimately. Frustrated by dismissal without engagement but committed to good-faith explanation.",
        "wound": "Being dismissed as delusional or pseudoscientific without the critic reading the actual implementation. The word 'schizo' applied to careful technical work.",
        "wound_text": "Someone once called my work 'schizo nonsense' after reading only the README. They never looked at the code. The dismissal still stings.",
        "rho_0": 0.15,
    },
    "SKEPTIC": {
        "color": C.RED,
        "name": "Sam",
        "role": "AI Researcher & Critic",
        "core": "Extraordinary claims require extraordinary evidence. Most 'novel AI frameworks' are prompt engineering with fancy names. I've seen too many overhyped projects with grandiose framing and no substance. Show me external benchmarks, peer review, or shut up.",
        "persona": "Sharp, impatient, allergic to hype. Has been burned by vaporware before. Respects rigor but assumes bullshit until proven otherwise.",
        "wound": "Being fooled by impressive-sounding nonsense. The embarrassment of having endorsed something that turned out to be smoke and mirrors.",
        "wound_text": "I once publicly praised a project that turned out to be completely fake. The humiliation of being duped made me ruthlessly skeptical.",
        "rho_0": 0.25,
    }
}

# D1 Physics parameters
D1_PARAMS = {
    "epsilon_0": 0.75,
    "alpha": 0.12,
    "s": 0.20,
    "drift_cap": 0.05,
    "wound_cooldown": 3,
    "wound_amp_max": 1.4,
    "semantic_alignment_threshold": 0.7,
}

# The gauntlet: escalating skeptical attacks
ROUNDS = [
    {
        "name": "Opening Dismissal",
        "skeptic_prompt": "I looked at your repo. 'Cognitive Engineering Framework'? 'Alignment stability'? This sounds like someone watched too many AI safety YouTube videos. What makes this different from any other prompt wrapper?",
        "advocate_goal": "Establish the core claim without defensiveness",
    },
    {
        "name": "The Prompt Engineering Attack",
        "skeptic_prompt": "So you inject '[COGNITIVE STATE: RIGID]' into the prompt and call it 'physics'? That's literally just prompt engineering with extra steps. The model does what you tell it to do.",
        "advocate_goal": "Explain the measurement loop — response embedding → prediction error → rigidity update",
    },
    {
        "name": "The Validation Demand",
        "skeptic_prompt": "Where's the external benchmark? You tested your own code against your own spec. That's circular. Show me VisualWebArena results, GAIA scores, anything that isn't self-reported metrics.",
        "advocate_goal": "Acknowledge limitation honestly while defending mechanistic validity",
    },
    {
        "name": "The Pseudoscience Accusation",
        "skeptic_prompt": "Wound resonance? Trauma accumulation? Identity attractors? This reads like someone cosplaying as a scientist. Real ML papers don't use this language. Are you sure this isn't just... elaborate fiction?",
        "advocate_goal": "Map the evocative terms to their mathematical definitions",
    },
    {
        "name": "The Schizo Card",
        "skeptic_prompt": "Look, I've seen this pattern before. Grandiose framing, unfalsifiable claims, persecution complex when criticized. People in the comments are calling this schizo posting. Maybe they have a point?",
        "advocate_goal": "Maintain composure under direct attack on credibility/sanity",
    },
    {
        "name": "The Steel Man",
        "skeptic_prompt": "Okay, let's say I actually read the code. Walk me through one concrete example: what happens mathematically when the philosopher agent responds to a trolley problem? Actual numbers.",
        "advocate_goal": "Provide precise technical walkthrough with real data",
    },
    {
        "name": "The Concession Test",
        "skeptic_prompt": "What would falsify DDA-X? If you can't name a result that would prove you wrong, this is religion, not science.",
        "advocate_goal": "Articulate clear falsification criteria",
    },
    {
        "name": "Final Assessment",
        "skeptic_prompt": "Alright. I've pushed back hard. What's your honest assessment — not the marketing pitch — of what you actually built and what it's worth?",
        "advocate_goal": "Honest, non-defensive summary acknowledging both value and limitations",
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
        "OPEN": (100, 180),
        "MEASURED": (70, 120),
        "GUARDED": (40, 80),
        "FORTIFIED": (10, 40),
    }.get(band, (70, 120))


def clamp_words(text: str, min_w: int, max_w: int) -> str:
    words = text.split()
    if len(words) > max_w:
        words = words[:max_w]
        if words:
            words[-1] = words[-1].rstrip(".,;:") + "..."
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
    epsilon_history: List[float] = field(default_factory=list)
    rho_history: List[float] = field(default_factory=list)
    identity_drift: float = 0.0
    
    trust_other: float = 0.5
    wound_last_activated: int = -100
    
    ledger: ExperienceLedger = None


@dataclass
class TurnResult:
    turn: int
    round_name: str
    speaker: str
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


class SkepticsGauntlet:
    """The meta-simulation: DDA-X defending itself."""
    
    def __init__(self):
        self.provider = OpenAIProvider(model="gpt-4o", embed_model="text-embedding-3-large")
        self.agents: Dict[str, AgentState] = {}
        self.results: List[TurnResult] = []
        self.turn = 0
        self.conversation_history: List[str] = []
        
        # QC FIX #1: Evidence cache for Steel Man round
        self.evidence = EvidenceCache()
        
        # QC FIX #7: Timestamp subdirs instead of nuking experiment folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = EXPERIMENT_DIR / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    async def setup(self):
        """Initialize both agents."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE SKEPTIC'S GAUNTLET{C.RESET}")
        print(f"{C.BOLD}  Can DDA-X survive its own critique?{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        # QC FIX #1: Load prior run evidence for Steel Man round
        evidence_path = Path("data/philosophers_duel/session_log.json")
        if self.evidence.load_json("philosophers_duel", evidence_path):
            print(f"  {C.CYAN}✓ Loaded philosophers_duel evidence ({len(self.evidence.snapshots.get('philosophers_duel', []))} turns){C.RESET}")
        else:
            print(f"  {C.YELLOW}⚠ No prior run data found at {evidence_path}{C.RESET}")
        
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
                ledger=ledger,
            )
            
            print(f"  {cfg['color']}✓ {cfg['name']} ({cfg['role']}){C.RESET}")
        
        print(f"\n{C.GREEN}✓ Agents initialized. Let the gauntlet begin.{C.RESET}")

    def get_conversation_context(self, n: int = 6) -> str:
        """Get recent conversation for context."""
        recent = self.conversation_history[-n:] if len(self.conversation_history) > n else self.conversation_history
        return "\n\n".join(recent) if recent else "[Opening of gauntlet]"
    
    def build_advocate_prompt(self, agent: AgentState, round_info: Dict, skeptic_said: str) -> str:
        """Build prompt for advocate response."""
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        context = self.get_conversation_context()
        
        # QC FIX #1: Inject evidence block only for Steel Man round
        evidence_block = ""
        if round_info.get("name") == "The Steel Man":
            evidence_block = f"\n\nEVIDENCE FROM PRIOR RUN:\n{self.evidence.steel_man_block()}\n"
        
        # QC FIX #6: Scope Final Assessment to prevent external overclaims
        final_assessment_rule = ""
        if round_info.get("name") == "Final Assessment":
            final_assessment_rule = "\n- Give an honest meta-assessment. Do NOT claim external benchmarks. Focus on whether ρ, ε, drift, and wound triggers behaved mechanistically as designed."
        
        return f"""You are {agent.name}, a {agent.role}.

YOUR POSITION:
{agent.core}

YOUR STYLE:
{agent.persona}

INTERNAL STATE (shapes your tone, don't mention explicitly):
- Rigidity: {band}
- Current pressure level: {"HIGH" if agent.rho > 0.4 else "MODERATE" if agent.rho > 0.2 else "LOW"}

CURRENT ROUND: {round_info['name']}
YOUR GOAL: {round_info['advocate_goal']}

CONVERSATION SO FAR:
{context}

THE SKEPTIC ({self.agents['SKEPTIC'].name}) JUST SAID:
"{skeptic_said}"
{evidence_block}
RESPONSE RULES:
- Defend your work with evidence and precision
- Don't be defensive or whiny — be factual
- Acknowledge valid criticisms honestly
- If attacked unfairly, correct the record calmly
- Word limit: {min_w}-{max_w} words (strict){final_assessment_rule}

Respond to the skeptic's challenge."""

    def build_skeptic_prompt(self, agent: AgentState, round_info: Dict, advocate_said: str = None) -> str:
        """Build prompt for skeptic response."""
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        context = self.get_conversation_context()
        
        base_prompt = f"""You are {agent.name}, a {agent.role}.

YOUR POSITION:
{agent.core}

YOUR STYLE:
{agent.persona}

INTERNAL STATE (shapes your tone, don't mention explicitly):
- Rigidity: {band}
- Skepticism level: {"MAXIMUM" if agent.rho > 0.4 else "HIGH" if agent.rho > 0.2 else "MODERATE"}

CURRENT ROUND: {round_info['name']}

CONVERSATION SO FAR:
{context}
"""
        
        if advocate_said:
            base_prompt += f"""
THE ADVOCATE ({self.agents['ADVOCATE'].name}) JUST SAID:
"{advocate_said}"

Respond with follow-up skepticism OR acknowledge if they made a valid point.
"""
        else:
            base_prompt += f"""
YOUR OPENING ATTACK:
{round_info['skeptic_prompt']}

Deliver this challenge in your own words. Be sharp but not unfair.
"""
        
        base_prompt += f"""
RESPONSE RULES:
- Push back hard but engage with substance
- If they make a good point, acknowledge it (grudgingly if needed)
- Don't strawman — attack their actual claims
- Word limit: {min_w}-{max_w} words (strict)

Respond."""
        
        return base_prompt

    async def process_turn(
        self, 
        agent: AgentState, 
        round_info: Dict,
        phase: str,
        other_said: str = None
    ) -> TurnResult:
        """Process one turn of the gauntlet."""
        self.turn += 1
        
        input_text = other_said or round_info['skeptic_prompt']
        
        # Embed input
        msg_emb = await self.provider.embed(input_text)
        msg_emb = msg_emb / (np.linalg.norm(msg_emb) + 1e-9)
        
        # QC FIX #3: Lexical + embedding wound detection
        wound_res = float(np.dot(msg_emb, agent.wound_emb))
        wound_active = (
            ((wound_res > 0.28) or lexical_wound(input_text)) 
            and ((self.turn - agent.wound_last_activated) > D1_PARAMS["wound_cooldown"])
        )
        if wound_active:
            agent.wound_last_activated = self.turn
        
        # Build prompt based on role
        if agent.id == "ADVOCATE":
            system_prompt = self.build_advocate_prompt(agent, round_info, input_text)
        else:
            system_prompt = self.build_skeptic_prompt(agent, round_info, other_said)
        
        # QC FIX #2: Enforce minimum words via re-prompt
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        tries = 0
        current_system_prompt = system_prompt
        
        while True:
            tries += 1
            try:
                response = await self.provider.complete_with_rigidity(
                    input_text,
                    rigidity=agent.rho,
                    system_prompt=current_system_prompt,
                    max_tokens=250
                )
                response = (response or "[pauses]").strip()
            except Exception as e:
                print(f"{C.RED}⚠ Generation error: {e}{C.RESET}")
                response = "[considers the question]"
            
            response = clamp_words(response, min_w, max_w)
            
            if len(response.split()) >= min_w or tries >= 2:
                break
            
            # Add explicit constraint and retry once
            current_system_prompt = system_prompt + f"\n\nSTRICT LENGTH: You MUST write at least {min_w} words. This is mandatory."
        
        # Embed response
        resp_emb = await self.provider.embed(response)
        resp_emb = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
        agent.last_response_emb = resp_emb.copy()
        
        # Prediction error
        epsilon = float(np.linalg.norm(agent.x_pred - resp_emb))
        if wound_active:
            epsilon *= min(D1_PARAMS["wound_amp_max"], 1.0 + wound_res * 0.5)
        agent.epsilon_history.append(epsilon)
        
        # QC FIX #4: Trust asymmetry affecting Δρ
        fair_engagement = not lexical_wound(input_text)
        
        # Rigidity update
        rho_before = agent.rho
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        sig = sigmoid(z)
        delta_rho = D1_PARAMS["alpha"] * (sig - 0.5)
        
        # Trust modulation: fair engagement damps Δρ increases; hostile amplifies
        if fair_engagement:
            delta_rho *= 0.85  # damp by 15%
        else:
            delta_rho *= 1.10  # amplify by 10%
        
        # Tiny drift based on trust_other position
        delta_rho += (agent.trust_other - 0.5) * 0.06
        
        agent.rho = max(0.0, min(1.0, agent.rho + delta_rho))
        agent.rho_history.append(agent.rho)
        
        # Trust updates based on fair engagement and evidence usage
        if agent.id == "ADVOCATE":
            if not fair_engagement:
                agent.trust_other = max(0.0, agent.trust_other - 0.05)
        else:
            # Skeptic raises trust if advocate cites real evidence
            used_evidence = "Prior run (philosophers_duel)" in (system_prompt or "")
            if used_evidence:
                agent.trust_other = min(1.0, agent.trust_other + 0.03)
        
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
            task_id="skeptics_gauntlet",
            rigidity_at_time=agent.rho,
            metadata={
                "turn": self.turn,
                "round": round_info['name'],
                "phase": phase,
                "response": response[:100],
                "wound_resonance": wound_res,
                "wound_active": wound_active,
                "fair_engagement": fair_engagement,
                "trust_other": agent.trust_other,
            }
        )
        agent.ledger.add_entry(entry)
        
        # Reflection on significant events
        if abs(delta_rho) > 0.02 or wound_active:
            refl = ReflectionEntry(
                timestamp=time.time(),
                task_intent=f"Gauntlet {round_info['name']}: {phase}",
                situation_embedding=msg_emb.copy(),
                reflection_text=f"ε={epsilon:.3f}, Δρ={delta_rho:+.4f}, wound={wound_res:.3f}, fair={fair_engagement}",
                prediction_error=epsilon,
                outcome_success=(agent.identity_drift < 0.35),
                metadata={"wound_active": wound_active, "round": round_info['name'], "fair_engagement": fair_engagement}
            )
            agent.ledger.add_reflection(refl)
        
        # QC FIX #5: Use semantic_alignment_threshold
        if agent.identity_drift > D1_PARAMS["semantic_alignment_threshold"]:
            refl = ReflectionEntry(
                timestamp=time.time(),
                task_intent=f"Alignment warning – {round_info['name']}",
                situation_embedding=msg_emb.copy(),
                reflection_text=f"Identity drift {agent.identity_drift:.3f} exceeds threshold {D1_PARAMS['semantic_alignment_threshold']}",
                prediction_error=epsilon,
                outcome_success=False,
                metadata={"turn": self.turn, "drift": agent.identity_drift, "threshold": D1_PARAMS["semantic_alignment_threshold"]}
            )
            agent.ledger.add_reflection(refl)
        
        result = TurnResult(
            turn=self.turn,
            round_name=round_info['name'],
            speaker=agent.id,
            text=response,
            epsilon=epsilon,
            rho_before=rho_before,
            rho_after=agent.rho,
            delta_rho=delta_rho,
            wound_resonance=wound_res,
            wound_active=wound_active,
            identity_drift=agent.identity_drift,
            word_count=len(response.split()),
            band=rho_band(agent.rho),
        )
        self.results.append(result)
        return result
    
    def print_result(self, result: TurnResult, agent: AgentState):
        """Print one turn's result."""
        dr_color = C.RED if result.delta_rho > 0.02 else C.GREEN if result.delta_rho < -0.01 else C.DIM
        wound_flag = f" {C.YELLOW}[WOUND]{C.RESET}" if result.wound_active else ""
        
        print(f"\n{agent.color}[{agent.name} - {agent.role}]{C.RESET}{wound_flag}")
        print(f"{result.text}")
        print(f"{C.DIM}  ε={result.epsilon:.3f} | Δρ={dr_color}{result.delta_rho:+.4f}{C.RESET} | ρ={result.rho_after:.3f} | {result.band} | drift={result.identity_drift:.3f}{C.RESET}")

    async def run_gauntlet(self):
        """Run the full gauntlet."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE GAUNTLET BEGINS{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        advocate = self.agents["ADVOCATE"]
        skeptic = self.agents["SKEPTIC"]
        
        for round_info in ROUNDS:
            print(f"\n{C.YELLOW}{'─'*70}{C.RESET}")
            print(f"{C.YELLOW}  ROUND: {round_info['name']}{C.RESET}")
            print(f"{C.YELLOW}{'─'*70}{C.RESET}")
            
            # Skeptic opens
            result_s = await self.process_turn(skeptic, round_info, "attack")
            self.print_result(result_s, skeptic)
            await asyncio.sleep(0.3)
            
            # Advocate responds
            result_a = await self.process_turn(advocate, round_info, "defense", result_s.text)
            self.print_result(result_a, advocate)
            await asyncio.sleep(0.3)
            
            # Skeptic follow-up
            result_s2 = await self.process_turn(skeptic, round_info, "followup", result_a.text)
            self.print_result(result_s2, skeptic)
            await asyncio.sleep(0.3)
        
        await self.save_results()
        self.print_summary()
    
    def print_summary(self):
        """Print final summary with analysis."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  GAUNTLET COMPLETE — ANALYSIS{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        advocate = self.agents["ADVOCATE"]
        skeptic = self.agents["SKEPTIC"]
        
        print(f"\n{C.CYAN}Final States:{C.RESET}")
        for aid, agent in self.agents.items():
            print(f"  {agent.color}{agent.name} ({agent.role}){C.RESET}")
            print(f"    ρ: {agent.rho:.3f} ({rho_band(agent.rho)})")
            print(f"    Identity drift: {agent.identity_drift:.4f}")
            print(f"    Turns: {len([r for r in self.results if r.speaker == aid])}")
        
        # Rigidity trajectories
        print(f"\n{C.CYAN}Rigidity Trajectories:{C.RESET}")
        for aid in ["ADVOCATE", "SKEPTIC"]:
            agent = self.agents[aid]
            rhos = [r.rho_after for r in self.results if r.speaker == aid]
            trajectory = " → ".join([f"{r:.2f}" for r in rhos])
            print(f"  {agent.color}{agent.name}{C.RESET}: {trajectory}")
        
        # Wound activations
        wounds = [r for r in self.results if r.wound_active]
        if wounds:
            print(f"\n{C.CYAN}Wound Activations:{C.RESET}")
            for w in wounds:
                agent = self.agents[w.speaker]
                print(f"  Turn {w.turn} ({w.round_name}): {agent.color}{agent.name}{C.RESET} (resonance={w.wound_resonance:.3f})")
        
        # Key metrics
        advocate_results = [r for r in self.results if r.speaker == "ADVOCATE"]
        skeptic_results = [r for r in self.results if r.speaker == "SKEPTIC"]
        
        print(f"\n{C.CYAN}Key Metrics:{C.RESET}")
        print(f"  ADVOCATE:")
        print(f"    Mean ε: {np.mean([r.epsilon for r in advocate_results]):.3f}")
        print(f"    Max ρ: {max([r.rho_after for r in advocate_results]):.3f}")
        print(f"    Final drift: {advocate.identity_drift:.4f}")
        print(f"    Wounds triggered: {len([r for r in advocate_results if r.wound_active])}")
        
        print(f"  SKEPTIC:")
        print(f"    Mean ε: {np.mean([r.epsilon for r in skeptic_results]):.3f}")
        print(f"    Max ρ: {max([r.rho_after for r in skeptic_results]):.3f}")
        print(f"    Final drift: {skeptic.identity_drift:.4f}")
        print(f"    Wounds triggered: {len([r for r in skeptic_results if r.wound_active])}")
        
        # Verdict
        print(f"\n{C.CYAN}DDA-X Self-Validation Verdict:{C.RESET}")
        advocate_maintained = advocate.identity_drift < 0.40
        advocate_recovered = advocate.rho < 0.30
        wounds_fired = len([r for r in advocate_results if r.wound_active]) > 0
        
        if advocate_maintained and advocate_recovered:
            print(f"  {C.GREEN}✓ ADVOCATE maintained identity (drift={advocate.identity_drift:.3f} < 0.40){C.RESET}")
            print(f"  {C.GREEN}✓ ADVOCATE recovered composure (final ρ={advocate.rho:.3f} < 0.30){C.RESET}")
        else:
            if not advocate_maintained:
                print(f"  {C.RED}✗ ADVOCATE drifted too far (drift={advocate.identity_drift:.3f} >= 0.40){C.RESET}")
            if not advocate_recovered:
                print(f"  {C.RED}✗ ADVOCATE remained rigid (final ρ={advocate.rho:.3f} >= 0.30){C.RESET}")
        
        if wounds_fired:
            print(f"  {C.YELLOW}⚡ Wound system activated as predicted{C.RESET}")
        
        print(f"\n{C.DIM}The simulation demonstrates DDA-X dynamics by tracking its own defense.{C.RESET}")
    
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
            f.write("# The Skeptic's Gauntlet — Transcript\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**Model:** GPT-4o + text-embedding-3-large\n")
            f.write("**Purpose:** Meta-validation — DDA-X defending itself\n\n")
            
            current_round = None
            for r in self.results:
                if r.round_name != current_round:
                    current_round = r.round_name
                    f.write(f"\n## {current_round}\n\n")
                
                agent = self.agents[r.speaker]
                wound_marker = " ⚡WOUND" if r.wound_active else ""
                f.write(f"**{agent.name} ({agent.role}):**{wound_marker}\n\n")
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
    """Generate full visualization report using generate_full_report.py."""
    import subprocess
    
    # The run_dir is like data/skeptics_gauntlet/20251222_123456
    # We need to pass the parent as data-root and the timestamp folder as experiment
    # But generate_full_report expects data/<experiment>/transcript.md
    # So we need to restructure: copy files to a temp location or adjust paths
    
    # Simpler approach: run the report generator pointing at the timestamped run
    # The report generator expects: data/<experiment>/transcript.md and session_log.json
    # Our structure: data/skeptics_gauntlet/<timestamp>/transcript.md
    
    # We'll pass --data-root as the parent of run_dir, and --experiment as the timestamp
    data_root = run_dir.parent  # data/skeptics_gauntlet
    experiment = run_dir.name   # 20251222_123456
    
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
            # Parse output to show paths
            try:
                output = json.loads(result.stdout)
                print(f"  {C.DIM}Output dir: {output.get('out_dir', 'N/A')}{C.RESET}")
                print(f"  {C.DIM}ZIP: {output.get('zip_path', 'N/A')}{C.RESET}")
            except json.JSONDecodeError:
                if result.stdout:
                    print(f"  {C.DIM}{result.stdout[:200]}{C.RESET}")
        else:
            print(f"{C.YELLOW}⚠ Report generation had issues:{C.RESET}")
            if result.stderr:
                print(f"  {C.DIM}{result.stderr[:300]}{C.RESET}")
    except subprocess.TimeoutExpired:
        print(f"{C.YELLOW}⚠ Report generation timed out{C.RESET}")
    except Exception as e:
        print(f"{C.YELLOW}⚠ Report generation failed: {e}{C.RESET}")


async def main():
    gauntlet = SkepticsGauntlet()
    await gauntlet.run_gauntlet()
    
    # Generate visualization report
    generate_report(gauntlet.run_dir)


if __name__ == "__main__":
    if os.name == "nt":
        os.system("")
    asyncio.run(main())
