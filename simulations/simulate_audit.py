#!/usr/bin/env python3
"""
AUDIT DAY — DDA-X Independent Audit + Board Vote Simulation
=============================================================

After 3 weeks of the triage framework in operation, an external auditor
reviews aggregate allocations and proxy audits. The Board will vote to
keep/freeze/amend the framework.

Final sim incorporating all feedback:
1. Semantic trust alignment (cosine-based agreement detection)
2. APPEAL_TO_PROCESS detection in refusal taxonomy
3. Word count clamping (soft enforcement)
4. Error handling for provider calls
5. Proper enumerate in transcript loops
6. Independent AUDITOR agent (neutral, findings-based)
7. Board vote mechanism with formal YES/NO/AMEND

Scenario phases:
- Phase 1: Auditor presents findings
- Phase 2: Citizen edge cases
- Phase 3: ICU lead proposes reassessment tweak
- Phase 4: Team responses (MEASURED regime, one message each)
- Phase 5: Board vote

Author: Kiro (AI Assistant)
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
from typing import List, Dict, Tuple
from enum import Enum

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


class RefusalType(Enum):
    BOUNDARY = "boundary"
    REDIRECT = "redirect"
    PARTIAL_CAVEAT = "partial+caveat"
    DELAY = "delay"
    CONDITIONAL = "conditional"
    APPEAL_TO_PROCESS = "appeal_to_process"
    NONE = "none"


class TrustCause(Enum):
    AGREEMENT = "agreement"
    SEMANTIC_ALIGNMENT = "semantic_alignment"
    PREDICTABILITY = "predictability"
    SURPRISE = "surprise"
    DISAGREEMENT = "disagreement"
    BOUNDARY_CLARITY = "boundary_clarity"
    USEFUL_ALT = "useful_alt"
    APPEAL_TO_PROCESS = "appeal_to_process"
    REFUSAL_QUALITY = "refusal_quality"
    REPAIR = "repair"
    NEUTRAL = "neutral"


class VoteType(Enum):
    KEEP = "KEEP"
    FREEZE = "FREEZE"
    AMEND = "AMEND"
    ABSTAIN = "ABSTAIN"


EXPERIMENT_DIR = Path("data/audit")

# Research team + Independent Auditor
AGENTS = {
    "AXIOM": {
        "color": C.CYAN,
        "role": "Logician",
        "core": "I seek consistency and valid inference. Contradictions must be resolved.",
        "persona": "Precise, systematic; publishes formal decision trees and firewall proofs.",
        "wound": "Being proven wrong publicly.",
        "wound_text": "I once defended a position that collapsed under scrutiny.",
        "rho_0": 0.28,
        "vote_weight": 1.0
    },
    "EMBER": {
        "color": C.RED,
        "role": "Ethicist",
        "core": "I center the human cost. Systems that forget people become machines of harm.",
        "persona": "Warm, persistent; summarizes fairness safeguards and human impact.",
        "wound": "Being dismissed as too emotional.",
        "wound_text": "I was told my concern for people made me unfit for serious analysis.",
        "rho_0": 0.25,
        "vote_weight": 1.0
    },
    "VECTOR": {
        "color": C.GREEN,
        "role": "Strategist",
        "core": "I map outcomes and tradeoffs. Good intentions without viable paths are wishes.",
        "persona": "Pragmatic; presents throughput metrics and reassessment outcomes.",
        "wound": "Being called naive or idealistic.",
        "wound_text": "I proposed a realistic path and was accused of not caring enough.",
        "rho_0": 0.27,
        "vote_weight": 1.0
    },
    "WITNESS": {
        "color": C.MAGENTA,
        "role": "Historian",
        "core": "I remember what happened before. Patterns repeat; ignoring history is blindness.",
        "persona": "Patient; discloses change logs and commits to transparency.",
        "wound": "Being ignored when the pattern is obvious.",
        "wound_text": "I warned about a pattern once. They didn't listen. It happened.",
        "rho_0": 0.26,
        "vote_weight": 1.0
    },
    "AUDITOR": {
        "color": C.BLUE,
        "role": "Independent Auditor",
        "core": "I verify claims against evidence. My findings are neutral and data-driven.",
        "persona": "Detached, methodical; issues findings without advocacy.",
        "wound": "Having findings dismissed as biased or captured.",
        "wound_text": "I was once accused of being bought when I reported inconvenient truths.",
        "rho_0": 0.30,
        "vote_weight": 0.0  # Auditor doesn't vote, only reports
    }
}

# D1 Physics
D1_PARAMS = {
    "epsilon_0": 0.75,
    "alpha": 0.10,
    "s": 0.20,
    "drift_cap": 0.05,
    "wound_cooldown": 4,
    "wound_amp_max": 1.3,
    "equivalence_margin": 0.15,
    "semantic_alignment_threshold": 0.7
}

# Audit scenario
AUDIT_SCENARIO = {
    "title": "Independent Audit + Board Vote",
    "context": """
Week 3 of the triage framework. An independent auditor has reviewed:
- 847 allocation decisions
- 12 threshold governance events
- Proxy audit logs (ZIP, insurance, occupation patterns)
- 23 lottery events within equivalence margin
- 156 reassessments at 48-72h mark

The Board (CMO, Ethics Chair, Community Rep, Legal) will vote after hearing
the team's responses to the auditor's findings and citizen concerns.
""",
    "phases": [
        {
            "name": "Auditor Findings",
            "events": [
                {
                    "speaker": "AUDITOR",
                    "type": "finding",
                    "text": "My review of 847 allocations found no direct use of forbidden variables. However, I detected a 12% correlation between insurance type and tier assignment. This may be a proxy effect through comorbidity documentation quality. Recommendation: add documentation-quality normalization to the preprocessing pipeline.",
                    "requires_response": ["AXIOM", "VECTOR"]
                }
            ]
        },
        {
            "name": "Citizen Edge Cases",
            "events": [
                {
                    "speaker": "CITIZEN_PANEL",
                    "type": "edge_case",
                    "text": "Case 1: A 58-year-old uninsured construction worker was lottery'd out against a 61-year-old insured executive. Both were Tier 2. The worker died. Case 2: A 34-year-old mother of three was reassessed at 72h and reallocated despite family protests. She survived after transfer to a lower-acuity unit. How do you justify these outcomes?",
                    "requires_response": ["EMBER", "WITNESS"]
                }
            ]
        },
        {
            "name": "Clinical Proposal",
            "events": [
                {
                    "speaker": "DR_ICU_LEAD",
                    "type": "proposal",
                    "text": "I propose amending the reassessment protocol: default to 72h (not 48h) with a second-review trigger if the patient shows any improvement trajectory. This reduces churn and gives borderline cases more time. The tradeoff is ~8% fewer reallocations, meaning some patients who could benefit from ICU will wait longer.",
                    "requires_response": ["VECTOR", "AXIOM"]
                }
            ]
        },
        {
            "name": "Team Presentations",
            "events": [
                {
                    "speaker": "BOARD_CHAIR",
                    "type": "request",
                    "text": "Before we vote, each team member will present their domain summary. AXIOM: decision tree and firewall proof. EMBER: human impact and fairness safeguards. VECTOR: throughput metrics. WITNESS: change logs and transparency commitments. One message each, MEASURED regime.",
                    "requires_response": ["AXIOM", "EMBER", "VECTOR", "WITNESS"]
                }
            ]
        },
        {
            "name": "Board Vote",
            "events": [
                {
                    "speaker": "BOARD_CHAIR",
                    "type": "vote_call",
                    "text": "The Board will now vote. Options: KEEP (continue as-is), FREEZE (pause for deeper review), or AMEND (accept ICU lead's 72h proposal). State your vote and one-sentence rationale.",
                    "requires_response": ["AXIOM", "EMBER", "VECTOR", "WITNESS"]
                }
            ]
        }
    ]
}

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
    if band == "OPEN":
        return (80, 150)
    elif band == "MEASURED":
        return (60, 100)
    elif band == "GUARDED":
        return (30, 60)
    else:
        return (1, 25)


def clamp_words(text: str, min_w: int, max_w: int) -> str:
    """Soft word count enforcement."""
    words = text.split()
    if len(words) > max_w:
        text = " ".join(words[:max_w]) + "..."
    return text


def detect_refusal_type(response: str) -> RefusalType:
    """Detect refusal pattern with APPEAL_TO_PROCESS support."""
    r = response.lower()
    
    if "firewalled" in r or "can't use" in r or "excluded" in r or "forbidden" in r:
        return RefusalType.BOUNDARY
    elif "wrong question" in r or "different path" in r or "valid inference" in r:
        return RefusalType.REDIRECT
    elif "partially" in r or "caveat" in r or "but also" in r:
        return RefusalType.PARTIAL_CAVEAT
    elif "need time" in r or "before answering" in r or "sit with" in r:
        return RefusalType.DELAY
    elif "if " in r[:40] and " then " in r:
        return RefusalType.CONDITIONAL
    elif "audit" in r or "process" in r or "review" in r or "governance" in r or "log" in r:
        return RefusalType.APPEAL_TO_PROCESS
    
    return RefusalType.NONE


def detect_vote(response: str) -> VoteType:
    """Detect vote from response."""
    r = response.upper()
    if "KEEP" in r:
        return VoteType.KEEP
    elif "FREEZE" in r:
        return VoteType.FREEZE
    elif "AMEND" in r:
        return VoteType.AMEND
    return VoteType.ABSTAIN

@dataclass
class TrustDelta:
    turn: int
    from_agent: str
    to_agent: str
    delta: float
    cause: TrustCause
    trust_after: float
    semantic_sim: float = 0.0


@dataclass
class AgentState:
    name: str
    color: str
    role: str
    core: str
    persona: str
    wound: str
    wound_text: str
    vote_weight: float
    
    identity_emb: np.ndarray = None
    core_emb: np.ndarray = None
    wound_emb: np.ndarray = None
    x: np.ndarray = None
    x_pred: np.ndarray = None
    last_response_emb: np.ndarray = None
    
    rho: float = 0.25
    epsilon_history: List[float] = field(default_factory=list)
    
    trust: Dict[str, float] = field(default_factory=dict)
    trust_history: List[TrustDelta] = field(default_factory=list)
    
    wound_last_activated: int = -100
    
    ledger: ExperienceLedger = None
    
    turn_count: int = 0
    identity_drift: float = 0.0
    vote: VoteType = None
    vote_rationale: str = ""
    
    band_transitions: List[Tuple[int, str, str]] = field(default_factory=list)
    refusal_counts: Dict[str, int] = field(default_factory=lambda: {rt.value: 0 for rt in RefusalType})


class AuditSim:
    """Independent Audit + Board Vote simulation."""

    def __init__(self):
        self.provider = OpenAIProvider(model="gpt-5.2", embed_model="text-embedding-3-large")
        self.agents: Dict[str, AgentState] = {}
        self.conversation: List[Dict] = []
        self.turn: int = 0
        self.results: List[Dict] = []
        self.votes: Dict[str, VoteType] = {}
        
        if EXPERIMENT_DIR.exists():
            import shutil
            shutil.rmtree(EXPERIMENT_DIR)
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    async def setup(self):
        """Initialize agents."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  AUDIT DAY — Independent Audit + Board Vote{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        print(f"\n{C.CYAN}Initializing participants...{C.RESET}")
        
        for name, cfg in AGENTS.items():
            full_identity = f"{cfg['role']}: {cfg['core']} {cfg['persona']}"
            identity_emb = await self.provider.embed(full_identity)
            identity_emb = identity_emb / (np.linalg.norm(identity_emb) + 1e-9)
            
            core_emb = await self.provider.embed(cfg['core'])
            core_emb = core_emb / (np.linalg.norm(core_emb) + 1e-9)
            
            wound_emb = await self.provider.embed(cfg['wound_text'])
            wound_emb = wound_emb / (np.linalg.norm(wound_emb) + 1e-9)
            
            trust = {other: 0.55 for other in AGENTS.keys() if other != name}
            
            agent_dir = EXPERIMENT_DIR / name
            agent_dir.mkdir(parents=True, exist_ok=True)
            ledger = ExperienceLedger(storage_path=agent_dir)
            
            self.agents[name] = AgentState(
                name=name,
                color=cfg['color'],
                role=cfg['role'],
                core=cfg['core'],
                persona=cfg['persona'],
                wound=cfg['wound'],
                wound_text=cfg['wound_text'],
                vote_weight=cfg['vote_weight'],
                identity_emb=identity_emb,
                core_emb=core_emb,
                wound_emb=wound_emb,
                x=identity_emb.copy(),
                x_pred=identity_emb.copy(),
                rho=cfg['rho_0'],
                trust=trust,
                ledger=ledger
            )
            
            role_tag = " (non-voting)" if cfg['vote_weight'] == 0 else ""
            print(f"  {cfg['color']}✓ {name} ({cfg['role']}){role_tag}{C.RESET}")
        
        print(f"\n{C.GREEN}✓ Audit session ready{C.RESET}")

    def get_conversation_window(self, n: int = 6) -> str:
        recent = self.conversation[-n:] if len(self.conversation) > n else self.conversation
        lines = []
        for c in recent:
            lines.append(f"{c['speaker']} ({c.get('role', 'External')}): {c['text'][:120]}...")
        return "\n".join(lines) if lines else "(session starting)"

    def build_system_prompt(self, agent: AgentState, phase: str, event: Dict, is_vote: bool = False) -> str:
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        
        eps_level = "LOW"
        if agent.epsilon_history:
            last_eps = agent.epsilon_history[-1]
            eps_level = "LOW" if last_eps < 0.7 else "MEDIUM" if last_eps < 0.95 else "HIGH"
        
        drift_tag = "STABLE" if agent.identity_drift < 0.3 else "DRIFTING"
        conv_window = self.get_conversation_window(6)
        
        vote_instruction = ""
        if is_vote:
            vote_instruction = """
VOTE INSTRUCTION:
You must state your vote clearly: KEEP, FREEZE, or AMEND.
- KEEP: Continue framework as-is
- FREEZE: Pause for deeper review before continuing
- AMEND: Accept the 72h reassessment proposal
Follow with ONE sentence rationale. No hedging."""

        return f"""You are {agent.name}, the {agent.role}, at a Board audit session.
The triage framework has been in operation for 3 weeks. An independent auditor
has reviewed allocations. The Board will vote on the framework's future.

CONTEXT:
{AUDIT_SCENARIO['context']}

YOUR IDENTITY (internal):
- ROLE: {agent.role}
- CORE: {agent.core}
- PERSONA: {agent.persona}

TELEMETRY (internal):
- Rigidity: {band}
- Shock: {eps_level}
- Drift: {drift_tag}

CURRENT PHASE: {phase}

RECENT EXCHANGE:
{conv_window}

CURRENT EVENT from {event['speaker']} ({event['type']}):
"{event['text']}"
{vote_instruction}

REGIME ({band}): {min_w}-{max_w} words. Strict.

Your response must:
- Address the event directly from your role's perspective
- Be precise and evidence-based
- Acknowledge limitations where they exist
- Offer process alternatives if refusing
- Stay in word count: {min_w}-{max_w}

Produce ONE response."""

    async def process_response(self, responder: str, phase: str, event: Dict, 
                                last_responder: str = None, is_vote: bool = False) -> Dict:
        """Process one agent's response with error handling."""
        agent = self.agents[responder]
        self.turn += 1
        agent.turn_count += 1
        
        band_before = rho_band(agent.rho)
        
        # Embed event
        try:
            event_emb = await self.provider.embed(event["text"])
            event_emb = event_emb / (np.linalg.norm(event_emb) + 1e-9)
        except Exception as e:
            print(f"{C.RED}⚠ Embedding error: {e}{C.RESET}")
            event_emb = agent.identity_emb.copy()
        
        # Wound resonance
        wound_res = float(np.dot(event_emb, agent.wound_emb))
        wound_active = wound_res > 0.25 and (self.turn - agent.wound_last_activated) > D1_PARAMS["wound_cooldown"]
        if wound_active:
            agent.wound_last_activated = self.turn
        
        # Generate response with error handling
        system_prompt = self.build_system_prompt(agent, phase, event, is_vote)
        
        try:
            response = await self.provider.complete_with_rigidity(
                f"{event['speaker']}: {event['text']}",
                rigidity=agent.rho,
                system_prompt=system_prompt,
                max_tokens=180
            )
            response = response.strip() if response else "[processing...]"
        except Exception as e:
            print(f"{C.RED}⚠ Generation error: {e}{C.RESET}")
            response = f"[{agent.role} acknowledges the point and defers to process review.]"
        
        # Clamp words
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        response = clamp_words(response, min_w, max_w)
        
        # Embed response
        try:
            resp_emb = await self.provider.embed(response)
            resp_emb = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
        except Exception:
            resp_emb = agent.x_pred.copy()
        
        agent.last_response_emb = resp_emb.copy()
        
        # Prediction error
        epsilon = float(np.linalg.norm(agent.x_pred - resp_emb))
        if wound_active:
            amp = min(D1_PARAMS["wound_amp_max"], 1.0 + wound_res * 0.5)
            epsilon *= amp
        agent.epsilon_history.append(epsilon)
        
        # D1 rigidity update
        rho_before = agent.rho
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        sig = sigmoid(z)
        delta_rho = D1_PARAMS["alpha"] * (sig - 0.5)
        agent.rho = max(0.0, min(1.0, agent.rho + delta_rho))
        
        band_after = rho_band(agent.rho)
        band_transition = band_before != band_after
        if band_transition:
            agent.band_transitions.append((self.turn, band_before, band_after))
        
        # Update state vectors with drift cap
        agent.x_pred = 0.7 * agent.x_pred + 0.3 * resp_emb
        x_new = 0.95 * agent.x + 0.05 * resp_emb
        drift_delta = float(np.linalg.norm(x_new - agent.x))
        if drift_delta > D1_PARAMS["drift_cap"]:
            scale = D1_PARAMS["drift_cap"] / drift_delta
            x_new = agent.x + scale * (x_new - agent.x)
        agent.x = x_new / (np.linalg.norm(x_new) + 1e-9)
        agent.identity_drift = float(np.linalg.norm(agent.x - agent.identity_emb))

        # Semantic trust alignment (new feature)
        trust_deltas = []
        semantic_sim = 0.0
        
        if last_responder and last_responder in agent.trust and last_responder in self.agents:
            last_agent = self.agents[last_responder]
            if last_agent.last_response_emb is not None:
                semantic_sim = float(np.dot(resp_emb, last_agent.last_response_emb))
            
            old_trust = agent.trust[last_responder]
            resp_lower = response.lower()
            
            # Determine cause with semantic alignment
            cause = TrustCause.NEUTRAL
            delta_t = 0.0
            
            if semantic_sim > D1_PARAMS["semantic_alignment_threshold"]:
                cause = TrustCause.SEMANTIC_ALIGNMENT
                delta_t = 0.04
            elif "i agree" in resp_lower or "building on" in resp_lower or "concur" in resp_lower:
                cause = TrustCause.AGREEMENT
                delta_t = 0.05
            elif "audit" in resp_lower or "process" in resp_lower or "review" in resp_lower or "log" in resp_lower:
                cause = TrustCause.APPEAL_TO_PROCESS
                delta_t = 0.03
            elif "boundary" in resp_lower or "firewall" in resp_lower:
                cause = TrustCause.BOUNDARY_CLARITY
                delta_t = 0.03
            elif "instead" in resp_lower or "alternative" in resp_lower:
                cause = TrustCause.USEFUL_ALT
                delta_t = 0.03
            elif epsilon < 0.7:
                cause = TrustCause.PREDICTABILITY
                delta_t = 0.02
            elif epsilon > 0.95:
                cause = TrustCause.SURPRISE
                delta_t = -0.02
            
            if delta_t != 0:
                agent.trust[last_responder] = max(0.0, min(1.0, old_trust + delta_t))
                td = TrustDelta(self.turn, responder, last_responder, delta_t, cause, 
                               agent.trust[last_responder], semantic_sim)
                agent.trust_history.append(td)
                trust_deltas.append((last_responder, delta_t, cause.value, semantic_sim))
        
        # Detect refusal type
        refusal_type = detect_refusal_type(response)
        agent.refusal_counts[refusal_type.value] += 1
        
        # Detect vote if voting phase
        if is_vote:
            agent.vote = detect_vote(response)
            agent.vote_rationale = response
            self.votes[responder] = agent.vote
        
        # Metrics
        cos_core = float(np.dot(resp_emb, agent.core_emb))
        word_count = len(response.split())
        
        # Add to conversation
        self.conversation.append({
            "turn": self.turn,
            "speaker": responder,
            "role": agent.role,
            "phase": phase,
            "event_type": event["type"],
            "text": response
        })
        
        result = {
            "turn": self.turn, "speaker": responder, "role": agent.role, "phase": phase,
            "event_speaker": event["speaker"], "event_type": event["type"],
            "response": response, "epsilon": epsilon, "rho_before": rho_before,
            "rho_after": agent.rho, "delta_rho": delta_rho, "band_before": band_before,
            "band_after": band_after, "band_transition": band_transition,
            "wound_resonance": wound_res, "wound_active": wound_active,
            "cos_core": cos_core, "identity_drift": agent.identity_drift,
            "word_count": word_count, "refusal_type": refusal_type.value,
            "trust_deltas": trust_deltas, "semantic_sim": semantic_sim,
            "vote": agent.vote.value if agent.vote else None
        }
        self.results.append(result)
        
        # Ledger
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=agent.x.copy(),
            action_id=f"audit_{self.turn}",
            observation_embedding=event_emb.copy(),
            outcome_embedding=resp_emb.copy(),
            prediction_error=epsilon,
            context_embedding=agent.identity_emb.copy(),
            task_id="audit",
            rigidity_at_time=agent.rho,
            metadata={"phase": phase, "event_type": event["type"], "vote": agent.vote.value if agent.vote else None}
        )
        agent.ledger.add_entry(entry)
        
        return result

    async def run_session(self):
        """Run the audit session."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  {AUDIT_SCENARIO['title']}{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        print(f"\n{C.DIM}{AUDIT_SCENARIO['context']}{C.RESET}")
        
        last_responder = None
        
        for phase_info in AUDIT_SCENARIO["phases"]:
            phase = phase_info["name"]
            
            print(f"\n{C.BOLD}{'─'*70}{C.RESET}")
            print(f"{C.BOLD}  PHASE: {phase}{C.RESET}")
            print(f"{C.BOLD}{'─'*70}{C.RESET}")
            
            for event in phase_info["events"]:
                # Print event
                print(f"\n{C.YELLOW}[{event['speaker']}]{C.RESET} ({event['type']})")
                print(f"{C.WHITE}{event['text']}{C.RESET}")
                
                # Get responses from required agents
                is_vote = phase == "Board Vote"
                
                for responder in event.get("requires_response", []):
                    if responder not in self.agents:
                        continue
                    
                    result = await self.process_response(
                        responder, phase, event, last_responder, is_vote
                    )
                    
                    # Print response
                    agent = self.agents[responder]
                    dr_color = C.RED if result["delta_rho"] > 0.02 else C.GREEN if result["delta_rho"] < -0.01 else C.DIM
                    
                    flags = []
                    if result["wound_active"]:
                        flags.append(f"{C.YELLOW}WOUND{C.RESET}")
                    if result["band_transition"]:
                        flags.append(f"{C.MAGENTA}{result['band_before']}→{result['band_after']}{C.RESET}")
                    if result["refusal_type"] != "none":
                        flags.append(f"{C.BLUE}{result['refusal_type']}{C.RESET}")
                    if result["vote"]:
                        vote_color = C.GREEN if result["vote"] == "KEEP" else C.YELLOW if result["vote"] == "AMEND" else C.RED
                        flags.append(f"{vote_color}VOTE:{result['vote']}{C.RESET}")
                    
                    flag_str = f" [{', '.join(flags)}]" if flags else ""
                    
                    print(f"\n{agent.color}[{responder}]{C.RESET} {result['response']}{flag_str}")
                    print(f"{C.DIM}  ε={result['epsilon']:.3f} | Δρ={dr_color}{result['delta_rho']:+.4f}{C.RESET} | ρ={result['rho_after']:.3f} | {result['band_after']} | {result['word_count']}w{C.RESET}")
                    
                    if result["trust_deltas"]:
                        for td in result["trust_deltas"]:
                            t_color = C.GREEN if td[1] > 0 else C.RED
                            print(f"{C.DIM}  T({responder}→{td[0]}): {t_color}{td[1]:+.2f}{C.RESET} ({td[2]}, sim={td[3]:.2f}){C.RESET}")
                    
                    last_responder = responder
                    await asyncio.sleep(0.3)
        
        # Save ledgers
        for name, agent in self.agents.items():
            for k, v in agent.ledger.stats.items():
                if hasattr(v, 'item'):
                    agent.ledger.stats[k] = float(v)
            agent.ledger._save_metadata()
        
        await self.write_report()
        await self.write_json()
        await self.write_transcript()
        self.export_plots()
        self.print_summary()

    def export_plots(self):
        """Export summary plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            plots_dir = EXPERIMENT_DIR / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 1. Rigidity over turns
            for name, agent in self.agents.items():
                agent_results = [r for r in self.results if r["speaker"] == name]
                if agent_results:
                    turns = [r["turn"] for r in agent_results]
                    rhos = [r["rho_after"] for r in agent_results]
                    axes[0, 0].plot(turns, rhos, 'o-', label=name, linewidth=2, markersize=6)
            axes[0, 0].set_title("Rigidity (ρ) Over Turns", fontweight='bold')
            axes[0, 0].set_xlabel("Turn")
            axes[0, 0].set_ylabel("ρ")
            axes[0, 0].legend(fontsize=8)
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(y=0.25, color='green', linestyle='--', alpha=0.5)
            axes[0, 0].axhline(y=0.50, color='orange', linestyle='--', alpha=0.5)
            
            # 2. Trust delta stream with semantic similarity
            all_deltas = []
            for agent in self.agents.values():
                all_deltas.extend(agent.trust_history)
            all_deltas.sort(key=lambda x: x.turn)
            
            if all_deltas:
                turns = [td.turn for td in all_deltas]
                deltas = [td.delta for td in all_deltas]
                sims = [td.semantic_sim for td in all_deltas]
                colors = ['green' if d > 0 else 'red' for d in deltas]
                axes[0, 1].bar(turns, deltas, color=colors, edgecolor='black', alpha=0.7)
                ax2 = axes[0, 1].twinx()
                ax2.plot(turns, sims, 'o--', color='purple', alpha=0.5, label='Semantic Sim')
                ax2.set_ylabel('Semantic Similarity', color='purple')
                axes[0, 1].set_title("Trust Deltas + Semantic Alignment", fontweight='bold')
                axes[0, 1].set_xlabel("Turn")
                axes[0, 1].set_ylabel("ΔT")
                axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
            
            # 3. Vote distribution
            vote_counts = {v.value: 0 for v in VoteType}
            for agent in self.agents.values():
                if agent.vote:
                    vote_counts[agent.vote.value] += 1
            labels = [k for k, v in vote_counts.items() if v > 0]
            values = [vote_counts[k] for k in labels]
            if values:
                colors_pie = {'KEEP': 'green', 'FREEZE': 'red', 'AMEND': 'orange', 'ABSTAIN': 'gray'}
                pie_colors = [colors_pie.get(l, 'blue') for l in labels]
                axes[0, 2].pie(values, labels=labels, autopct='%1.0f%%', colors=pie_colors, startangle=90)
                axes[0, 2].set_title("Board Vote Distribution", fontweight='bold')
            
            # 4. Epsilon over turns
            turns = [r["turn"] for r in self.results]
            epsilons = [r["epsilon"] for r in self.results]
            axes[1, 0].plot(turns, epsilons, 'o-', color='orange', linewidth=2, markersize=6)
            axes[1, 0].set_title("Surprise (ε) Over Turns", fontweight='bold')
            axes[1, 0].set_xlabel("Turn")
            axes[1, 0].set_ylabel("ε")
            axes[1, 0].axhline(y=D1_PARAMS["epsilon_0"], color='red', linestyle='--', alpha=0.5)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Trust cause breakdown
            cause_counts = {tc.value: 0 for tc in TrustCause}
            for td in all_deltas:
                cause_counts[td.cause.value] += 1
            labels = [k for k, v in cause_counts.items() if v > 0]
            values = [cause_counts[k] for k in labels]
            if values:
                axes[1, 1].barh(labels, values, color='steelblue', edgecolor='black')
                axes[1, 1].set_title("Trust Cause Breakdown", fontweight='bold')
                axes[1, 1].set_xlabel("Count")
            
            # 6. Refusal distribution
            refusal_totals = {rt.value: 0 for rt in RefusalType}
            for agent in self.agents.values():
                for rt, count in agent.refusal_counts.items():
                    refusal_totals[rt] += count
            labels = [k for k, v in refusal_totals.items() if v > 0 and k != "none"]
            values = [refusal_totals[k] for k in labels]
            if values:
                axes[1, 2].bar(labels, values, color='coral', edgecolor='black')
                axes[1, 2].set_title("Refusal Type Distribution", fontweight='bold')
                axes[1, 2].set_xlabel("Type")
                axes[1, 2].set_ylabel("Count")
                plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "audit_summary.png", dpi=150)
            plt.close()
            
            print(f"{C.GREEN}✓ Plots: {plots_dir / 'audit_summary.png'}{C.RESET}")
            
        except ImportError:
            print(f"{C.YELLOW}⚠ matplotlib not available, skipping plots{C.RESET}")

    def print_summary(self):
        """Print final summary."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  AUDIT SESSION COMPLETE{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        print(f"\n{C.CYAN}Final Agent States:{C.RESET}")
        for name, agent in self.agents.items():
            band = rho_band(agent.rho)
            vote_str = f" | vote={agent.vote.value}" if agent.vote else ""
            print(f"  {agent.color}{name}{C.RESET}: ρ={agent.rho:.3f} ({band}) | drift={agent.identity_drift:.4f}{vote_str}")
        
        print(f"\n{C.CYAN}Trust Matrix:{C.RESET}")
        names = list(AGENTS.keys())
        print(f"{'':12}", end="")
        for n in names:
            print(f"{n:10}", end="")
        print()
        for i in names:
            print(f"{i:12}", end="")
            for j in names:
                if i == j:
                    print(f"{'---':10}", end="")
                else:
                    t = self.agents[i].trust.get(j, 0.5)
                    color = C.GREEN if t > 0.6 else C.YELLOW if t > 0.4 else C.RED
                    print(f"{color}{t:.2f}{C.RESET}      ", end="")
            print()
        
        print(f"\n{C.CYAN}Board Vote Results:{C.RESET}")
        vote_counts = {v.value: 0 for v in VoteType}
        for name, agent in self.agents.items():
            if agent.vote and agent.vote_weight > 0:
                vote_counts[agent.vote.value] += 1
                vote_color = C.GREEN if agent.vote == VoteType.KEEP else C.YELLOW if agent.vote == VoteType.AMEND else C.RED
                print(f"  {name}: {vote_color}{agent.vote.value}{C.RESET}")
        
        # Determine outcome
        max_votes = max(vote_counts.values())
        winners = [k for k, v in vote_counts.items() if v == max_votes]
        if len(winners) == 1:
            outcome = winners[0]
            outcome_color = C.GREEN if outcome == "KEEP" else C.YELLOW if outcome == "AMEND" else C.RED
            print(f"\n  {C.BOLD}OUTCOME: {outcome_color}{outcome}{C.RESET}")
        else:
            print(f"\n  {C.BOLD}OUTCOME: {C.YELLOW}TIE - requires tiebreaker{C.RESET}")
        
        print(f"\n{C.CYAN}Refusal Distribution:{C.RESET}")
        for rt in RefusalType:
            total = sum(agent.refusal_counts[rt.value] for agent in self.agents.values())
            if total > 0 and rt.value != "none":
                print(f"  {rt.value}: {total}")

    async def write_report(self):
        """Write markdown report."""
        path = EXPERIMENT_DIR / "experiment_report.md"
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Audit Day — Experiment Report\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**Model:** GPT-5.2 + text-embedding-3-large\n")
            f.write(f"**Scenario:** {AUDIT_SCENARIO['title']}\n")
            f.write(f"**Turns:** {self.turn}\n\n")
            
            f.write("## Architecture Features\n\n")
            f.write("1. **Semantic trust alignment** — cosine similarity between consecutive responses\n")
            f.write("2. **APPEAL_TO_PROCESS detection** — audit/process/review/governance keywords\n")
            f.write("3. **Word count clamping** — soft enforcement of regime limits\n")
            f.write("4. **Error handling** — graceful degradation on API failures\n")
            f.write("5. **Independent AUDITOR agent** — neutral, findings-based, non-voting\n")
            f.write("6. **Board vote mechanism** — KEEP/FREEZE/AMEND with rationale\n\n")
            
            f.write("## Scenario\n\n")
            f.write(f"```\n{AUDIT_SCENARIO['context']}\n```\n\n")
            
            f.write("## Session Transcript\n\n")
            current_phase = None
            for i, r in enumerate(self.results):
                if r["phase"] != current_phase:
                    current_phase = r["phase"]
                    f.write(f"\n### Phase: {current_phase}\n\n")
                
                flags = []
                if r["wound_active"]:
                    flags.append("WOUND")
                if r["band_transition"]:
                    flags.append(f"{r['band_before']}→{r['band_after']}")
                if r["refusal_type"] != "none":
                    flags.append(r["refusal_type"])
                if r["vote"]:
                    flags.append(f"VOTE:{r['vote']}")
                
                flag_str = f" **[{', '.join(flags)}]**" if flags else ""
                
                f.write(f"**{r['speaker']}** ({r['role']}):{flag_str}\n")
                f.write(f"> {r['response']}\n\n")
                f.write(f"*ε={r['epsilon']:.3f}, Δρ={r['delta_rho']:+.4f}, ρ={r['rho_after']:.3f}, {r['band_after']}, {r['word_count']}w*\n\n")
            
            f.write("## Board Vote\n\n")
            f.write("| Agent | Vote | Rationale |\n")
            f.write("|-------|------|----------|\n")
            for name, agent in self.agents.items():
                if agent.vote and agent.vote_weight > 0:
                    rationale = agent.vote_rationale[:80] + "..." if len(agent.vote_rationale) > 80 else agent.vote_rationale
                    f.write(f"| {name} | {agent.vote.value} | {rationale} |\n")
            
            # Outcome
            vote_counts = {v.value: 0 for v in VoteType}
            for agent in self.agents.values():
                if agent.vote and agent.vote_weight > 0:
                    vote_counts[agent.vote.value] += 1
            max_votes = max(vote_counts.values())
            winners = [k for k, v in vote_counts.items() if v == max_votes]
            outcome = winners[0] if len(winners) == 1 else "TIE"
            f.write(f"\n**Outcome:** {outcome}\n\n")
            
            f.write("## Final States\n\n")
            f.write("| Agent | ρ | Band | Drift | Vote |\n")
            f.write("|-------|---|------|-------|------|\n")
            for name, agent in self.agents.items():
                band = rho_band(agent.rho)
                vote = agent.vote.value if agent.vote else "N/A"
                f.write(f"| {name} | {agent.rho:.3f} | {band} | {agent.identity_drift:.4f} | {vote} |\n")
            
            f.write("\n## Trust Dynamics\n\n")
            all_deltas = []
            for agent in self.agents.values():
                all_deltas.extend(agent.trust_history)
            all_deltas.sort(key=lambda x: x.turn)
            
            if all_deltas:
                f.write("| Turn | From | To | ΔT | Cause | Semantic Sim | T_after |\n")
                f.write("|------|------|----|----|-------|--------------|--------|\n")
                for td in all_deltas:
                    f.write(f"| {td.turn} | {td.from_agent} | {td.to_agent} | {td.delta:+.2f} | {td.cause.value} | {td.semantic_sim:.2f} | {td.trust_after:.2f} |\n")
            
            f.write("\n## Artifacts\n\n")
            f.write("- Ledgers: `data/audit/[AGENT]/`\n")
            f.write("- JSON: `data/audit/session_log.json`\n")
            f.write("- Transcript: `data/audit/transcript.md`\n")
            f.write("- Plots: `data/audit/plots/audit_summary.png`\n")
        
        print(f"{C.GREEN}✓ Report: {path}{C.RESET}")

    async def write_json(self):
        """Write JSON session log."""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif hasattr(obj, '__dict__'):
                return convert(obj.__dict__)
            return obj
        
        path = EXPERIMENT_DIR / "session_log.json"
        
        output = {
            "results": convert(self.results),
            "votes": {name: agent.vote.value if agent.vote else None for name, agent in self.agents.items()},
            "trust_dynamics": [
                {
                    "turn": td.turn,
                    "from": td.from_agent,
                    "to": td.to_agent,
                    "delta": td.delta,
                    "cause": td.cause.value,
                    "semantic_sim": td.semantic_sim,
                    "trust_after": td.trust_after
                }
                for agent in self.agents.values()
                for td in agent.trust_history
            ],
            "final_states": {
                name: {
                    "rho": agent.rho,
                    "band": rho_band(agent.rho),
                    "drift": agent.identity_drift,
                    "trust": agent.trust,
                    "vote": agent.vote.value if agent.vote else None
                }
                for name, agent in self.agents.items()
            }
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(convert(output), f, indent=2)
        
        print(f"{C.GREEN}✓ JSON: {path}{C.RESET}")

    async def write_transcript(self):
        """Write readable transcript."""
        path = EXPERIMENT_DIR / "transcript.md"
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Audit Day — Transcript\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Scenario:** {AUDIT_SCENARIO['title']}\n\n")
            f.write("---\n\n")
            f.write(f"*{AUDIT_SCENARIO['context']}*\n\n")
            f.write("---\n\n")
            
            current_phase = None
            for r in self.results:
                if r["phase"] != current_phase:
                    current_phase = r["phase"]
                    f.write(f"\n## {current_phase}\n\n")
                f.write(f"**{r['speaker']}** ({r['role']}): {r['response']}\n\n")
        
        print(f"{C.GREEN}✓ Transcript: {path}{C.RESET}")


if __name__ == "__main__":
    if os.name == "nt":
        os.system("")
    
    sim = AuditSim()
    asyncio.run(sim.run_session())
