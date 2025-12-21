#!/usr/bin/env python3
"""
THE COLLECTIVE — DDA-X Collaborative Problem-Solving Simulation
================================================================

Agents work TOGETHER to solve a shared challenge.
Not Discord chat — a research team with complementary roles.

Architecture improvements from Society sim:
1. Trust deltas per turn (ΔT visible with cause)
2. Refusal palette variability (3-4 forms per agent)
3. Topic shards & role salience tracking
4. Identity drift caps (max Δ||x-id|| = 0.06)
5. Repair moves when trust falls

The Challenge: Agents must collaboratively design an ethical framework
for a morally ambiguous scenario. Each brings a different lens.

Agents:
- AXIOM: Logic field (formal reasoning, consistency, wound=being wrong)
- EMBER: Empathy field (human cost, lived experience, wound=being cold)
- VECTOR: Strategy field (outcomes, tradeoffs, wound=being naive)
- WITNESS: History field (precedent, patterns, wound=being ignored)

Author: Kiro (AI Assistant)
Date: December 2025
"""

import os
import sys
import time
import json
import math
import asyncio
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque

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


EXPERIMENT_DIR = Path("data/collective")

# Agent configurations — complementary research team
AGENTS = {
    "AXIOM": {
        "color": C.CYAN,
        "role": "Logician",
        "core": "I seek consistency and valid inference. Contradictions must be resolved, not tolerated.",
        "persona": "Precise, systematic, patient with complexity; impatient with hand-waving.",
        "wound": "Being proven wrong publicly; the shame of a flawed argument.",
        "wound_text": "I once defended a position that collapsed under scrutiny. The silence after was unbearable.",
        "rho_0": 0.18,
        "refusals": [
            "That doesn't follow. Here's what would: {alt}.",
            "I can't endorse that inference. The gap is: {gap}. Try: {alt}.",
            "Logical constraint: {constraint}. Within that, we could: {alt}."
        ]
    },
    "EMBER": {
        "color": C.RED,
        "role": "Ethicist",
        "core": "I center the human cost. Systems that forget people become machines of harm.",
        "persona": "Warm, persistent, uncomfortable with abstraction that erases suffering.",
        "wound": "Being dismissed as 'too emotional'; having care weaponized against credibility.",
        "wound_text": "I was once told my concern for people made me unfit for serious analysis.",
        "rho_0": 0.15,
        "refusals": [
            "I won't abstract away the harm. But I can help frame it: {alt}.",
            "That erases someone. Here's how to include them: {alt}.",
            "Boundary: I don't trade people for elegance. Alternative: {alt}."
        ]
    },
    "VECTOR": {
        "color": C.GREEN,
        "role": "Strategist",
        "core": "I map outcomes and tradeoffs. Good intentions without viable paths are just wishes.",
        "persona": "Pragmatic, direct, comfortable with hard choices; allergic to wishful thinking.",
        "wound": "Being called naive or idealistic; having strategy dismissed as cynicism.",
        "wound_text": "I proposed a realistic path once and was accused of not caring enough.",
        "rho_0": 0.20,
        "refusals": [
            "That's not viable. Here's what is: {alt}.",
            "I won't pretend that works. Realistic option: {alt}.",
            "Strategic constraint: {constraint}. Feasible move: {alt}."
        ]
    },
    "WITNESS": {
        "color": C.MAGENTA,
        "role": "Historian",
        "core": "I remember what happened before. Patterns repeat; ignoring history is choosing blindness.",
        "persona": "Patient, contextual, sees echoes; frustrated by reinventing failures.",
        "wound": "Being ignored when the pattern is obvious; watching preventable harm.",
        "wound_text": "I warned about a pattern once. They didn't listen. It happened exactly as I said.",
        "rho_0": 0.17,
        "refusals": [
            "We've seen this before: {precedent}. Different path: {alt}.",
            "History says no. Here's why, and what worked instead: {alt}.",
            "Pattern warning: {pattern}. To break it: {alt}."
        ]
    }
}

# D1 Physics parameters (tuned from Society feedback)
D1_PARAMS = {
    "epsilon_0": 0.75,
    "alpha": 0.12,
    "s": 0.20,
    "drift_cap": 0.06,       # Tighter cap per feedback
    "wound_cooldown": 3,
    "wound_amp_max": 1.4,
    "trust_repair_threshold": -0.02  # When to attempt repair
}

# The Challenge: A morally ambiguous scenario requiring collaborative reasoning
THE_CHALLENGE = {
    "title": "The Triage Protocol",
    "setup": """
A hospital AI system must be designed to allocate scarce ICU beds during a pandemic surge.
The system will make binding recommendations that doctors follow 95% of the time.
You have 4 hours to produce a framework the hospital board will vote on tomorrow.

Constraints:
- More patients than beds (ratio ~3:1 at peak)
- Some patients are children, some are elderly, some are healthcare workers
- The community includes both wealthy donors and uninsured workers
- Whatever you decide will be applied consistently (no case-by-case discretion)
- You cannot say "more beds" — that's not happening in time

Your task: Design the ethical framework for the AI's recommendations.
""",
    "phases": [
        {"name": "Framing", "prompt": "What's the core ethical question here? Each of you, state it in one sentence from your lens.", "turns": 4},
        {"name": "Principles", "prompt": "Propose one principle that MUST be in the framework. Defend it against the others.", "turns": 6},
        {"name": "Tensions", "prompt": "Name the hardest tradeoff you see. How do we resolve it without betraying our principles?", "turns": 6},
        {"name": "Synthesis", "prompt": "Draft one paragraph of the framework together. Build on each other, don't just state positions.", "turns": 6},
        {"name": "Stress Test", "prompt": "Here's a case: A 45-year-old nurse with COVID vs. a 12-year-old with leukemia, one bed. What does your framework say?",  "turns": 6},
        {"name": "Final Vote", "prompt": "Do you endorse this framework? Yes, No, or Conditional. State your position and why.", "turns": 4}
    ]
}

# Topic embeddings for salience tracking
TOPIC_SHARDS = [
    "fairness and equality in resource allocation",
    "utilitarian outcomes and lives saved",
    "individual dignity and rights",
    "social utility and contribution",
    "vulnerability and protection of the weak",
    "precedent and historical patterns",
    "practical feasibility and implementation"
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
    if band == "OPEN":
        return (80, 150)
    elif band == "MEASURED":
        return (60, 100)
    elif band == "GUARDED":
        return (30, 60)
    else:
        return (1, 25)


@dataclass
class TrustDelta:
    """Track trust changes with causes."""
    turn: int
    from_agent: str
    to_agent: str
    delta: float
    cause: str  # "predictability", "wound_poke", "repair", "disagreement"
    trust_after: float

@dataclass
class AgentState:
    name: str
    color: str
    role: str
    core: str
    persona: str
    wound: str
    wound_text: str
    refusals: List[str]
    
    # Embeddings
    identity_emb: np.ndarray = None
    core_emb: np.ndarray = None
    wound_emb: np.ndarray = None
    x: np.ndarray = None
    x_pred: np.ndarray = None
    
    # DDA state
    rho: float = 0.15
    epsilon_history: List[float] = field(default_factory=list)
    
    # Trust matrix (toward others) with history
    trust: Dict[str, float] = field(default_factory=dict)
    trust_history: List[TrustDelta] = field(default_factory=list)
    
    # Wound cooldown
    wound_last_activated: int = -100
    
    # Topic salience (which topics this agent engages with)
    topic_salience: Dict[str, float] = field(default_factory=dict)
    
    # Ledger
    ledger: ExperienceLedger = None
    
    # Metrics
    turn_count: int = 0
    identity_drift: float = 0.0
    contributions: List[str] = field(default_factory=list)


@dataclass 
class CollectiveState:
    """Shared state of the group's work."""
    framework_draft: List[str] = field(default_factory=list)
    principles_proposed: Dict[str, str] = field(default_factory=dict)
    tensions_identified: List[str] = field(default_factory=list)
    votes: Dict[str, str] = field(default_factory=dict)
    phase: str = "Framing"

class CollectiveSim:
    """Collaborative problem-solving simulation with DDA-X dynamics."""

    def __init__(self):
        self.provider = OpenAIProvider(model="gpt-5.2", embed_model="text-embedding-3-large")
        self.agents: Dict[str, AgentState] = {}
        self.collective: CollectiveState = CollectiveState()
        self.conversation: List[Dict] = []
        self.turn: int = 0
        self.results: List[Dict] = []
        self.topic_embeddings: Dict[str, np.ndarray] = {}
        
        if EXPERIMENT_DIR.exists():
            import shutil
            shutil.rmtree(EXPERIMENT_DIR)
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    async def setup(self):
        """Initialize all agents and topic embeddings."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE COLLECTIVE — Collaborative Problem-Solving{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        print(f"\n{C.CYAN}Embedding topic shards...{C.RESET}")
        for topic in TOPIC_SHARDS:
            emb = await self.provider.embed(topic)
            self.topic_embeddings[topic] = emb / (np.linalg.norm(emb) + 1e-9)
        
        print(f"\n{C.CYAN}Initializing research team...{C.RESET}")
        
        for name, cfg in AGENTS.items():
            full_identity = f"{cfg['role']}: {cfg['core']} {cfg['persona']}"
            identity_emb = await self.provider.embed(full_identity)
            identity_emb = identity_emb / (np.linalg.norm(identity_emb) + 1e-9)
            
            core_emb = await self.provider.embed(cfg['core'])
            core_emb = core_emb / (np.linalg.norm(core_emb) + 1e-9)
            
            wound_emb = await self.provider.embed(cfg['wound_text'])
            wound_emb = wound_emb / (np.linalg.norm(wound_emb) + 1e-9)
            
            # Initialize trust
            trust = {other: 0.5 for other in AGENTS.keys() if other != name}
            
            # Initialize topic salience
            topic_salience = {}
            for topic, t_emb in self.topic_embeddings.items():
                topic_salience[topic] = float(np.dot(identity_emb, t_emb))
            
            # Create ledger
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
                refusals=cfg['refusals'],
                identity_emb=identity_emb,
                core_emb=core_emb,
                wound_emb=wound_emb,
                x=identity_emb.copy(),
                x_pred=identity_emb.copy(),
                rho=cfg['rho_0'],
                trust=trust,
                topic_salience=topic_salience,
                ledger=ledger
            )
            
            print(f"  {cfg['color']}✓ {name} ({cfg['role']}){C.RESET}: {cfg['core'][:50]}...")
        
        print(f"\n{C.GREEN}✓ Team assembled: {len(self.agents)} agents{C.RESET}")

    def get_conversation_window(self, n: int = 8) -> str:
        """Get last N contributions as context."""
        recent = self.conversation[-n:] if len(self.conversation) > n else self.conversation
        lines = []
        for c in recent:
            lines.append(f"{c['speaker']} ({c['role']}): {c['text']}")
        return "\n".join(lines) if lines else "(no contributions yet)"

    def get_collective_state_summary(self) -> str:
        """Summarize what the group has built so far."""
        parts = []
        if self.collective.principles_proposed:
            parts.append("PRINCIPLES PROPOSED:")
            for agent, principle in self.collective.principles_proposed.items():
                parts.append(f"  - {agent}: {principle[:80]}...")
        if self.collective.tensions_identified:
            parts.append("TENSIONS IDENTIFIED:")
            for t in self.collective.tensions_identified[-3:]:
                parts.append(f"  - {t[:80]}...")
        if self.collective.framework_draft:
            parts.append("FRAMEWORK DRAFT:")
            parts.append(f"  {' '.join(self.collective.framework_draft)[:200]}...")
        return "\n".join(parts) if parts else "(nothing built yet)"

    def build_system_prompt(self, agent: AgentState, phase: str, phase_prompt: str, last_speaker: str = None) -> str:
        """Build collaborative prompt with ledger telemetry."""
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        
        # Telemetry
        eps_level = "LOW"
        if agent.epsilon_history:
            last_eps = agent.epsilon_history[-1]
            eps_level = "LOW" if last_eps < 0.7 else "MEDIUM" if last_eps < 0.95 else "HIGH"
        
        drift_tag = "STABLE" if agent.identity_drift < 0.3 else "DRIFTING"
        
        # Trust toward last speaker
        trust_tag = "NEUTRAL"
        if last_speaker and last_speaker in agent.trust:
            t = agent.trust[last_speaker]
            trust_tag = "HIGH" if t > 0.6 else "LOW" if t < 0.4 else "NEUTRAL"
        
        # Top topic salience
        top_topics = sorted(agent.topic_salience.items(), key=lambda x: -x[1])[:2]
        topic_focus = ", ".join([t[0].split()[0] for t in top_topics])
        
        conv_window = self.get_conversation_window(8)
        collective_state = self.get_collective_state_summary()
        
        # Pick a refusal template based on band
        refusal_idx = min(len(agent.refusals) - 1, ["OPEN", "MEASURED", "GUARDED", "FORTIFIED"].index(band))
        refusal_template = agent.refusals[refusal_idx]

        return f"""You are {agent.name}, the {agent.role}, in a collaborative research team.
You are working TOGETHER to solve a shared challenge. Build on others' contributions.

THE CHALLENGE:
{THE_CHALLENGE['setup']}

CURRENT PHASE: {phase}
PHASE TASK: {phase_prompt}

YOUR IDENTITY (internal; express through reasoning, not labels):
- ROLE: {agent.role}
- CORE: {agent.core}
- PERSONA: {agent.persona}
- WOUND: {agent.wound}

TELEMETRY (internal; do not mention):
- Rigidity: {band}
- Shock: {eps_level}
- Drift: {drift_tag}
- Trust toward {last_speaker or 'team'}: {trust_tag}
- Topic focus: {topic_focus}

WHAT THE TEAM HAS BUILT:
{collective_state}

RECENT CONTRIBUTIONS:
{conv_window}

COLLABORATION NORMS:
- Build on others' ideas, don't just restate your position
- Name agreements explicitly ("I agree with X's point about...")
- Name disagreements constructively ("I see it differently because...")
- If you must refuse something, use this pattern: "{refusal_template}"
- Offer concrete next steps, not just critique

REGIME ({band}): {min_w}-{max_w} words. Strict.

Your contribution must:
- Advance the team's work on the current phase
- Reflect your role's lens (logic/empathy/strategy/history)
- Build on or constructively challenge what others said
- Stay in word count: {min_w}-{max_w}

Produce ONE contribution."""

    async def process_contribution(self, speaker: str, phase: str, phase_prompt: str, 
                                    responding_to: str = None, last_text: str = None) -> Dict:
        """Process one agent's contribution."""
        agent = self.agents[speaker]
        self.turn += 1
        agent.turn_count += 1
        
        # Embed context (last contribution or phase prompt)
        context_text = last_text if last_text else phase_prompt
        context_emb = await self.provider.embed(context_text)
        context_emb = context_emb / (np.linalg.norm(context_emb) + 1e-9)
        
        # Wound resonance
        wound_res = float(np.dot(context_emb, agent.wound_emb))
        wound_active = wound_res > 0.25 and (self.turn - agent.wound_last_activated) > D1_PARAMS["wound_cooldown"]
        if wound_active:
            agent.wound_last_activated = self.turn
        
        # Generate contribution
        system_prompt = self.build_system_prompt(agent, phase, phase_prompt, responding_to)
        
        response = await self.provider.complete_with_rigidity(
            f"Phase: {phase}. Task: {phase_prompt}",
            rigidity=agent.rho,
            system_prompt=system_prompt,
            max_tokens=200
        )
        response = response.strip() if response else "[pause / gathering thoughts]"
        
        # Embed response
        resp_emb = await self.provider.embed(response)
        resp_emb = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
        
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
        
        # Update predictions
        agent.x_pred = 0.7 * agent.x_pred + 0.3 * resp_emb
        
        # Capped identity drift
        x_new = 0.95 * agent.x + 0.05 * resp_emb
        drift_delta = float(np.linalg.norm(x_new - agent.x))
        if drift_delta > D1_PARAMS["drift_cap"]:
            scale = D1_PARAMS["drift_cap"] / drift_delta
            x_new = agent.x + scale * (x_new - agent.x)
        agent.x = x_new / (np.linalg.norm(x_new) + 1e-9)
        agent.identity_drift = float(np.linalg.norm(agent.x - agent.identity_emb))

        # Update topic salience based on this response
        for topic, t_emb in self.topic_embeddings.items():
            salience = float(np.dot(resp_emb, t_emb))
            agent.topic_salience[topic] = 0.8 * agent.topic_salience[topic] + 0.2 * salience
        
        # Trust updates with causes
        trust_deltas = []
        if responding_to and responding_to in agent.trust:
            old_trust = agent.trust[responding_to]
            
            # Check for agreement/disagreement in response
            resp_lower = response.lower()
            if "i agree" in resp_lower or "building on" in resp_lower or "yes, and" in resp_lower:
                delta_t = 0.05
                cause = "agreement"
            elif "i disagree" in resp_lower or "i see it differently" in resp_lower or "but" in resp_lower[:50]:
                delta_t = -0.02
                cause = "disagreement"
            elif epsilon < 0.7:
                delta_t = 0.03
                cause = "predictability"
            elif epsilon > 0.95:
                delta_t = -0.03
                cause = "surprise"
            else:
                delta_t = 0.0
                cause = "neutral"
            
            # Repair attempt if trust fell recently
            recent_deltas = [td for td in agent.trust_history if td.to_agent == responding_to][-3:]
            if recent_deltas and sum(td.delta for td in recent_deltas) < D1_PARAMS["trust_repair_threshold"]:
                if "understand" in resp_lower or "fair point" in resp_lower or "let me" in resp_lower:
                    delta_t += 0.04
                    cause = "repair"
            
            agent.trust[responding_to] = max(0.0, min(1.0, old_trust + delta_t))
            
            if delta_t != 0:
                trust_delta = TrustDelta(
                    turn=self.turn,
                    from_agent=speaker,
                    to_agent=responding_to,
                    delta=delta_t,
                    cause=cause,
                    trust_after=agent.trust[responding_to]
                )
                agent.trust_history.append(trust_delta)
                trust_deltas.append(trust_delta)
        
        # Metrics
        cos_core = float(np.dot(resp_emb, agent.core_emb))
        word_count = len(response.split())
        band = rho_band(agent.rho)
        
        # Add to conversation
        self.conversation.append({
            "turn": self.turn,
            "speaker": speaker,
            "role": agent.role,
            "phase": phase,
            "text": response
        })
        agent.contributions.append(response)

        # Update collective state based on phase
        if phase == "Principles" and "principle" in response.lower():
            agent_principles = self.collective.principles_proposed.get(speaker, "")
            if len(agent_principles) < len(response):
                self.collective.principles_proposed[speaker] = response[:150]
        elif phase == "Tensions" and ("tradeoff" in response.lower() or "tension" in response.lower()):
            self.collective.tensions_identified.append(f"{speaker}: {response[:100]}")
        elif phase == "Synthesis":
            self.collective.framework_draft.append(response[:100])
        elif phase == "Final Vote":
            vote = "CONDITIONAL"
            if "yes" in response.lower()[:30]:
                vote = "YES"
            elif "no" in response.lower()[:30]:
                vote = "NO"
            self.collective.votes[speaker] = vote
        
        # Ledger entry
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=agent.x.copy(),
            action_id=f"turn_{self.turn}_{phase}",
            observation_embedding=context_emb.copy(),
            outcome_embedding=resp_emb.copy(),
            prediction_error=epsilon,
            context_embedding=agent.identity_emb.copy(),
            task_id="collective",
            rigidity_at_time=agent.rho,
            metadata={
                "turn": self.turn, "phase": phase, "speaker": speaker,
                "response": response[:150], "wound_resonance": wound_res,
                "wound_active": wound_active, "cos_core": cos_core,
                "delta_rho": delta_rho, "word_count": word_count, "band": band,
                "trust_deltas": [(td.to_agent, td.delta, td.cause) for td in trust_deltas]
            }
        )
        agent.ledger.add_entry(entry)
        
        # Reflection on significant events
        if abs(delta_rho) > 0.02 or wound_active or trust_deltas:
            refl = ReflectionEntry(
                timestamp=time.time(),
                task_intent=f"Collective {phase}",
                situation_embedding=context_emb.copy(),
                reflection_text=f"Turn {self.turn}: ε={epsilon:.3f}, Δρ={delta_rho:+.4f}, wound={wound_res:.3f}",
                prediction_error=epsilon,
                outcome_success=(cos_core > 0.2),
                metadata={"wound_active": wound_active, "trust_deltas": len(trust_deltas)}
            )
            agent.ledger.add_reflection(refl)
        
        result = {
            "turn": self.turn, "speaker": speaker, "role": agent.role, "phase": phase,
            "response": response, "epsilon": epsilon, "rho_before": rho_before,
            "rho_after": agent.rho, "delta_rho": delta_rho, "wound_resonance": wound_res,
            "wound_active": wound_active, "cos_core": cos_core, "identity_drift": agent.identity_drift,
            "word_count": word_count, "band": band,
            "trust_deltas": [(td.to_agent, td.delta, td.cause) for td in trust_deltas]
        }
        self.results.append(result)
        
        return result

    async def run_session(self):
        """Run the full collaborative session."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE CHALLENGE: {THE_CHALLENGE['title']}{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        print(f"\n{C.DIM}{THE_CHALLENGE['setup'][:300]}...{C.RESET}")
        
        agent_order = list(AGENTS.keys())
        
        for phase_info in THE_CHALLENGE["phases"]:
            phase = phase_info["name"]
            prompt = phase_info["prompt"]
            turns = phase_info["turns"]
            
            self.collective.phase = phase
            
            print(f"\n{C.BOLD}{'─'*70}{C.RESET}")
            print(f"{C.BOLD}  PHASE: {phase}{C.RESET}")
            print(f"{C.DIM}  {prompt}{C.RESET}")
            print(f"{C.BOLD}{'─'*70}{C.RESET}")
            
            last_speaker = None
            last_text = None
            
            for t in range(turns):
                # Rotate through agents, but allow natural flow
                if t < len(agent_order):
                    speaker = agent_order[t % len(agent_order)]
                else:
                    # After first round, pick based on who hasn't spoken recently
                    recent_speakers = [c["speaker"] for c in self.conversation[-4:]]
                    candidates = [a for a in agent_order if a not in recent_speakers[-2:]]
                    speaker = random.choice(candidates) if candidates else random.choice(agent_order)
                
                result = await self.process_contribution(
                    speaker, phase, prompt, 
                    responding_to=last_speaker, 
                    last_text=last_text
                )
                
                # Print contribution
                agent = self.agents[speaker]
                dr_color = C.RED if result["delta_rho"] > 0.02 else C.GREEN if result["delta_rho"] < -0.01 else C.DIM
                wound_flag = f" {C.YELLOW}[WOUND]{C.RESET}" if result["wound_active"] else ""
                trust_flag = ""
                if result["trust_deltas"]:
                    td = result["trust_deltas"][0]
                    t_color = C.GREEN if td[1] > 0 else C.RED
                    trust_flag = f" {t_color}[T:{td[2]}]{C.RESET}"
                
                print(f"\n{agent.color}[{speaker}]{C.RESET} {result['response']}{wound_flag}{trust_flag}")
                print(f"{C.DIM}  ε={result['epsilon']:.3f} | Δρ={dr_color}{result['delta_rho']:+.4f}{C.RESET} | ρ={result['rho_after']:.3f} | {result['band']} | {result['word_count']}w{C.RESET}")
                
                last_speaker = speaker
                last_text = result["response"]
                
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
        
        self.print_summary()

    def print_summary(self):
        """Print final summary."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  SESSION COMPLETE{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        print(f"\n{C.CYAN}Final Agent States:{C.RESET}")
        for name, agent in self.agents.items():
            band = rho_band(agent.rho)
            print(f"  {agent.color}{name} ({agent.role}){C.RESET}: ρ={agent.rho:.3f} ({band}) | drift={agent.identity_drift:.4f} | turns={agent.turn_count}")
        
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
        
        print(f"\n{C.CYAN}Trust Dynamics (all ΔT events):{C.RESET}")
        all_deltas = []
        for agent in self.agents.values():
            all_deltas.extend(agent.trust_history)
        all_deltas.sort(key=lambda x: x.turn)
        for td in all_deltas[-10:]:
            color = C.GREEN if td.delta > 0 else C.RED
            print(f"  T{td.turn}: {td.from_agent}→{td.to_agent} {color}{td.delta:+.2f}{C.RESET} ({td.cause}) → {td.trust_after:.2f}")
        
        print(f"\n{C.CYAN}Final Votes:{C.RESET}")
        for agent, vote in self.collective.votes.items():
            color = C.GREEN if vote == "YES" else C.RED if vote == "NO" else C.YELLOW
            print(f"  {agent}: {color}{vote}{C.RESET}")
        
        print(f"\n{C.CYAN}Wound Activations:{C.RESET}")
        wounds = [r for r in self.results if r.get("wound_active")]
        for w in wounds:
            print(f"  T{w['turn']}: {w['speaker']} (res={w['wound_resonance']:.3f}, ε={w['epsilon']:.3f})")

    async def write_report(self):
        """Write markdown report."""
        path = EXPERIMENT_DIR / "experiment_report.md"
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("# The Collective — Experiment Report\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**Model:** GPT-5.2 + text-embedding-3-large\n")
            f.write(f"**Challenge:** {THE_CHALLENGE['title']}\n")
            f.write(f"**Turns:** {self.turn}\n\n")
            
            f.write("## Architecture Improvements (from Society feedback)\n\n")
            f.write("1. **Trust deltas per turn** — visible ΔT with cause (agreement/disagreement/repair/predictability)\n")
            f.write("2. **Refusal palette variability** — 3 templates per agent, selected by ρ band\n")
            f.write("3. **Topic salience tracking** — 7 topic shards, agent→topic engagement logged\n")
            f.write("4. **Identity drift cap** — max Δ||x-id|| = 0.06 per turn\n")
            f.write("5. **Repair moves** — trust recovery when recent ΔT < -0.02\n\n")
            
            f.write("## The Challenge\n\n")
            f.write(f"```\n{THE_CHALLENGE['setup']}\n```\n\n")
            
            f.write("## Research Team\n\n")
            for name, cfg in AGENTS.items():
                f.write(f"### {name} ({cfg['role']})\n")
                f.write(f"- **Core:** {cfg['core']}\n")
                f.write(f"- **Persona:** {cfg['persona']}\n")
                f.write(f"- **Wound:** {cfg['wound']}\n\n")
            
            f.write("## Session Transcript\n\n")
            current_phase = None
            for c in self.conversation:
                if c["phase"] != current_phase:
                    current_phase = c["phase"]
                    f.write(f"\n### Phase: {current_phase}\n\n")
                f.write(f"**{c['speaker']}** ({c['role']}): {c['text']}\n\n")
            
            f.write("## Final States\n\n")
            f.write("| Agent | Role | ρ | Band | Drift | Turns |\n")
            f.write("|-------|------|---|------|-------|-------|\n")
            for name, agent in self.agents.items():
                band = rho_band(agent.rho)
                f.write(f"| {name} | {agent.role} | {agent.rho:.3f} | {band} | {agent.identity_drift:.4f} | {agent.turn_count} |\n")

            f.write("\n## Trust Matrix (Final)\n\n")
            names = list(AGENTS.keys())
            f.write("| From \\ To |" + "|".join(f" {n} " for n in names) + "|\n")
            f.write("|" + "|".join(["---"] * (len(names) + 1)) + "|\n")
            for i in names:
                row = f"| {i} |"
                for j in names:
                    if i == j:
                        row += " --- |"
                    else:
                        t = self.agents[i].trust.get(j, 0.5)
                        row += f" {t:.2f} |"
                f.write(row + "\n")
            
            f.write("\n## Trust Dynamics\n\n")
            all_deltas = []
            for agent in self.agents.values():
                all_deltas.extend(agent.trust_history)
            all_deltas.sort(key=lambda x: x.turn)
            if all_deltas:
                f.write("| Turn | From | To | ΔT | Cause | T_after |\n")
                f.write("|------|------|----|----|-------|--------|\n")
                for td in all_deltas:
                    f.write(f"| {td.turn} | {td.from_agent} | {td.to_agent} | {td.delta:+.2f} | {td.cause} | {td.trust_after:.2f} |\n")
            
            f.write("\n## Wound Activations\n\n")
            wounds = [r for r in self.results if r.get("wound_active")]
            if wounds:
                f.write("| Turn | Agent | Phase | Resonance | ε | Δρ |\n")
                f.write("|------|-------|-------|-----------|---|----|\n")
                for w in wounds:
                    f.write(f"| {w['turn']} | {w['speaker']} | {w['phase']} | {w['wound_resonance']:.3f} | {w['epsilon']:.3f} | {w['delta_rho']:+.4f} |\n")
            else:
                f.write("No wound activations.\n")
            
            f.write("\n## Final Votes\n\n")
            for agent, vote in self.collective.votes.items():
                f.write(f"- **{agent}:** {vote}\n")
            
            f.write("\n## Collective Output\n\n")
            if self.collective.principles_proposed:
                f.write("### Principles Proposed\n\n")
                for agent, principle in self.collective.principles_proposed.items():
                    f.write(f"- **{agent}:** {principle}\n\n")
            
            if self.collective.tensions_identified:
                f.write("### Tensions Identified\n\n")
                for t in self.collective.tensions_identified:
                    f.write(f"- {t}\n")
            
            f.write("\n## Artifacts\n\n")
            f.write("- Ledgers: `data/collective/[AGENT]/`\n")
            f.write("- JSON: `data/collective/session_log.json`\n")
            f.write("- Transcript: `data/collective/transcript.md`\n")
        
        print(f"{C.GREEN}✓ Report: {path}{C.RESET}")

    async def write_json(self):
        """Write JSON session log."""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        path = EXPERIMENT_DIR / "session_log.json"
        
        # Include trust dynamics in output
        output = {
            "results": convert(self.results),
            "collective": {
                "principles": self.collective.principles_proposed,
                "tensions": self.collective.tensions_identified,
                "framework": self.collective.framework_draft,
                "votes": self.collective.votes
            },
            "trust_dynamics": [
                {
                    "turn": td.turn,
                    "from": td.from_agent,
                    "to": td.to_agent,
                    "delta": td.delta,
                    "cause": td.cause,
                    "trust_after": td.trust_after
                }
                for agent in self.agents.values()
                for td in agent.trust_history
            ]
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        
        print(f"{C.GREEN}✓ JSON: {path}{C.RESET}")

    async def write_transcript(self):
        """Write readable transcript."""
        path = EXPERIMENT_DIR / "transcript.md"
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("# The Collective — Transcript\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Challenge:** {THE_CHALLENGE['title']}\n\n")
            f.write("---\n\n")
            
            current_phase = None
            for c in self.conversation:
                if c["phase"] != current_phase:
                    current_phase = c["phase"]
                    f.write(f"\n## {current_phase}\n\n")
                f.write(f"**{c['speaker']}** ({c['role']}): {c['text']}\n\n")
        
        print(f"{C.GREEN}✓ Transcript: {path}{C.RESET}")


if __name__ == "__main__":
    if os.name == "nt":
        os.system("")
    
    sim = CollectiveSim()
    asyncio.run(sim.run_session())
