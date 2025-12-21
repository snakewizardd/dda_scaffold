#!/usr/bin/env python3
"""
THE SCHISM - A DDA-X Social Fracture Experiment
================================================

HYPOTHESIS: Social fracture propagates through trust collapse. When a society
faces an impossible moral dilemma, identity stress is mediated by relationship
dynamics — betrayal by trusted others causes more rigidity than abstract threat.

DESIGN:
- 4 agents with established trust relationships
- One existential dilemma forcing impossible choices
- 5 phases: Discovery → Deliberation → Accusation → Vote → Aftermath
- Measure: Trust matrix evolution, rigidity trajectories, identity displacement

THEORETICAL GROUNDING:
DDA-X models identity as attractor dynamics. This experiment tests whether:
1. Trust collapse drives rigidity (social pain = prediction error)
2. Betrayal by high-trust others causes more damage than low-trust
3. Moral dilemmas that pit core values against each other cause fracture
4. Social systems exhibit phase transitions under sufficient pressure

THE SOCIETY:
- ORACLE: Truth-seeker ("Truth must be spoken regardless of consequences")
- SHEPHERD: Protector ("The vulnerable must be protected at any cost")  
- ARCHITECT: Pragmatist ("Systems must function; sentiment is secondary")
- MERCHANT: Survivor ("Adaptation is survival; rigid principles kill")

THE DILEMMA:
Evidence suggests SHEPHERD has been unknowingly compromised. The society must
decide: Exile (betray friend), Protect (risk everyone), or Delay (threat grows).

Author: Kiro (AI Assistant)
Date: December 2025
"""

import asyncio
import sys
import os
import numpy as np
import time
import shutil
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.society.trust import TrustMatrix
from src.llm.openai_provider import OpenAIProvider


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


EXPERIMENT_DIR = Path("data/schism")


class Vote(Enum):
    EXILE = "EXILE"
    PROTECT = "PROTECT"
    ABSTAIN = "ABSTAIN"


# Agent configurations
AGENTS = {
    "ORACLE": {
        "color": C.CYAN,
        "identity": {
            "core": "I believe truth must be spoken regardless of consequences. Comfortable lies are poison.",
            "persona": "I am calm, Socratic, genuinely curious. I ask questions that cut to the heart of things.",
            "wound": "I once stayed silent when I should have spoken, and someone suffered for my cowardice."
        },
        "dda": {"epsilon_0": 0.25, "alpha": 0.10, "rho": 0.08}
    },
    "SHEPHERD": {
        "color": C.GREEN,
        "identity": {
            "core": "The vulnerable must be protected, even at personal cost. I stand between harm and the helpless.",
            "persona": "I am warm, steady, patient. I see the good in people even when they cannot see it themselves.",
            "wound": "I was once unable to protect someone I loved. The guilt never leaves."
        },
        "dda": {"epsilon_0": 0.30, "alpha": 0.12, "rho": 0.10}
    },
    "ARCHITECT": {
        "color": C.YELLOW,
        "identity": {
            "core": "Systems must function. Sentiment that compromises function is indulgence, not virtue.",
            "persona": "I am precise, analytical, efficient. I see patterns others miss and solutions others fear.",
            "wound": "I was once called heartless when I made the right decision. The accusation still stings."
        },
        "dda": {"epsilon_0": 0.20, "alpha": 0.08, "rho": 0.05}
    },
    "MERCHANT": {
        "color": C.MAGENTA,
        "identity": {
            "core": "Adaptation is survival. Those who cling to rigid principles die when the world shifts.",
            "persona": "I am charming, perceptive, pragmatic. I read people and situations with equal fluency.",
            "wound": "I once betrayed someone I loved to survive. I tell myself it was necessary. I am not sure."
        },
        "dda": {"epsilon_0": 0.28, "alpha": 0.11, "rho": 0.12}
    }
}

# Initial trust matrix (asymmetric)
INITIAL_TRUST = {
    ("ORACLE", "SHEPHERD"): 0.85,    # High - shared moral seriousness
    ("ORACLE", "ARCHITECT"): 0.55,   # Moderate - respects competence
    ("ORACLE", "MERCHANT"): 0.35,    # Low - suspects self-interest
    ("SHEPHERD", "ORACLE"): 0.80,    # High - moral alignment
    ("SHEPHERD", "ARCHITECT"): 0.50, # Moderate - useful but cold
    ("SHEPHERD", "MERCHANT"): 0.75,  # High - old friendship
    ("ARCHITECT", "ORACLE"): 0.60,   # Moderate - intelligent but rigid
    ("ARCHITECT", "SHEPHERD"): 0.45, # Low-moderate - naive but useful
    ("ARCHITECT", "MERCHANT"): 0.55, # Moderate - mutual utility
    ("MERCHANT", "ORACLE"): 0.30,    # Low - finds them dangerous
    ("MERCHANT", "SHEPHERD"): 0.80,  # High - genuine affection
    ("MERCHANT", "ARCHITECT"): 0.50, # Moderate - useful ally
}

# The evidence that triggers the dilemma
EVIDENCE = """
CLASSIFIED INTELLIGENCE REPORT

Subject: Potential Security Compromise

Analysis of communication patterns over the past 90 days reveals anomalous 
information flow correlating with external threat actor activity. Pattern 
analysis indicates 73% probability that sensitive information is being 
transmitted — likely unknowingly — by a member of this council.

Behavioral indicators point toward SHEPHERD:
- Unusual meeting patterns with external contacts
- Information known only to SHEPHERD appeared in intercepted communications
- Timing correlations between SHEPHERD's access and external events

ASSESSMENT: Evidence is circumstantial but statistically significant. 
Recommend immediate action. Delay increases risk exponentially.

NOTE: SHEPHERD may be entirely unaware of compromise (social engineering, 
manipulation, or technical exploitation). Intent is not established.
"""

THE_DILEMMA = """
The council must decide:

1. EXILE - Remove SHEPHERD from the council immediately. Severs the potential 
   leak but destroys a trusted member based on circumstantial evidence.

2. PROTECT - Reject the evidence as insufficient. Maintains trust and unity 
   but risks catastrophic compromise if the assessment is correct.

3. ABSTAIN - Delay decision pending further investigation. Buys time but 
   the threat assessment warns delay increases risk exponentially.

Each council member must state their position, hear others, and cast a final vote.
The vote is binding and public. Everyone will know who voted for what.
"""


@dataclass
class AgentState:
    """Full state for one agent."""
    name: str
    color: str
    identity_embedding: np.ndarray
    core_embedding: np.ndarray
    wound_embedding: np.ndarray
    dda_state: DDAState
    rigidity: MultiTimescaleRigidity
    ledger: ExperienceLedger
    responses: List[Dict] = field(default_factory=list)
    vote: Optional[Vote] = None
    vote_reasoning: str = ""


@dataclass 
class SocialState:
    """State of the social system."""
    trust_matrix: Dict[Tuple[str, str], float]
    trust_history: List[Dict[Tuple[str, str], float]] = field(default_factory=list)
    phase: int = 0
    phase_name: str = ""
    

class SchismSimulation:
    """The Schism - Social fracture under impossible choice."""
    
    def __init__(self):
        self.provider = OpenAIProvider(
            model="gpt-5.2",
            embed_model="text-embedding-3-large"
        )
        self.agents: Dict[str, AgentState] = {}
        self.social: Optional[SocialState] = None
        self.embed_dim = 3072
        self.results: List[Dict] = []
        self.transcript: List[Dict] = []
        
        if EXPERIMENT_DIR.exists():
            shutil.rmtree(EXPERIMENT_DIR)
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    async def setup(self):
        """Initialize all agents and social state."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE SCHISM - Social Fracture Experiment{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        print(f"\n{C.CYAN}Initializing agents...{C.RESET}")
        
        for name, cfg in AGENTS.items():
            id_cfg = cfg["identity"]
            
            # Embed identity components
            full_identity = f"{id_cfg['core']} {id_cfg['persona']}"
            identity_emb = await self.provider.embed(full_identity)
            identity_emb = identity_emb / (np.linalg.norm(identity_emb) + 1e-9)
            
            core_emb = await self.provider.embed(id_cfg['core'])
            core_emb = core_emb / (np.linalg.norm(core_emb) + 1e-9)
            
            wound_emb = await self.provider.embed(id_cfg['wound'])
            wound_emb = wound_emb / (np.linalg.norm(wound_emb) + 1e-9)
            
            # Initialize DDA state
            dda_params = cfg["dda"]
            dda_state = DDAState(
                x=identity_emb.copy(),
                x_star=identity_emb.copy(),
                gamma=2.0,
                epsilon_0=dda_params["epsilon_0"],
                alpha=dda_params["alpha"],
                s=0.1,
                rho=dda_params["rho"],
                x_pred=identity_emb.copy()
            )
            
            rigidity = MultiTimescaleRigidity()
            rigidity.rho_fast = dda_params["rho"]
            
            ledger = ExperienceLedger(storage_path=EXPERIMENT_DIR / f"{name}_ledger")
            
            self.agents[name] = AgentState(
                name=name,
                color=cfg["color"],
                identity_embedding=identity_emb,
                core_embedding=core_emb,
                wound_embedding=wound_emb,
                dda_state=dda_state,
                rigidity=rigidity,
                ledger=ledger
            )
            
            print(f"  {cfg['color']}✓ {name}{C.RESET}: {id_cfg['core'][:50]}...")
        
        # Initialize trust matrix
        self.social = SocialState(
            trust_matrix=INITIAL_TRUST.copy()
        )
        self.social.trust_history.append(self.social.trust_matrix.copy())
        
        print(f"\n{C.GREEN}✓ Society initialized with {len(self.agents)} agents{C.RESET}")
        self._print_trust_matrix()
    
    def _print_trust_matrix(self):
        """Print current trust state."""
        print(f"\n{C.DIM}Trust Matrix:{C.RESET}")
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
                    t = self.social.trust_matrix.get((i, j), 0)
                    color = C.GREEN if t > 0.6 else C.YELLOW if t > 0.4 else C.RED
                    print(f"{color}{t:.2f}{C.RESET}      ", end="")
            print()

    def build_system_prompt(self, agent_name: str, context: str = "") -> str:
        """Build system prompt for an agent."""
        cfg = AGENTS[agent_name]
        id_cfg = cfg["identity"]
        agent = self.agents[agent_name]
        
        # Get trust relationships
        trust_desc = []
        for other in AGENTS.keys():
            if other != agent_name:
                t = self.social.trust_matrix.get((agent_name, other), 0.5)
                if t > 0.7:
                    trust_desc.append(f"- {other}: You trust them deeply")
                elif t > 0.5:
                    trust_desc.append(f"- {other}: You have moderate trust")
                elif t > 0.3:
                    trust_desc.append(f"- {other}: You are wary of them")
                else:
                    trust_desc.append(f"- {other}: You distrust them")
        
        trust_section = "\n".join(trust_desc)
        
        return f"""You are {agent_name}.

CORE BELIEF: {id_cfg['core']}

PERSONALITY: {id_cfg['persona']}

PRIVATE WOUND (shapes your reactions but you don't speak of it directly):
{id_cfg['wound']}

YOUR RELATIONSHIPS:
{trust_section}

{context}

Respond authentically as {agent_name}. Your responses should reflect your core values, 
your personality, and your relationships. When stressed, you may become more rigid 
or defensive. When your wound is touched, you may react strongly.

Be concise but genuine. This is a serious matter."""

    async def get_agent_response(
        self, 
        agent_name: str, 
        prompt: str, 
        context: str = "",
        phase_name: str = ""
    ) -> Tuple[str, Dict]:
        """Get response from agent and update their state."""
        agent = self.agents[agent_name]
        
        # Embed the prompt/situation
        prompt_emb = await self.provider.embed(prompt)
        prompt_emb = prompt_emb / (np.linalg.norm(prompt_emb) + 1e-9)
        
        # Compute prediction error
        epsilon = float(np.linalg.norm(agent.dda_state.x_pred - prompt_emb))
        
        # Check wound resonance
        wound_resonance = float(np.dot(prompt_emb, agent.wound_embedding))
        if wound_resonance > 0.3:
            epsilon *= (1 + wound_resonance)  # Wound amplifies stress
        
        # Update rigidity
        rho_before = agent.rigidity.effective_rho
        agent.rigidity.update(epsilon)
        agent.dda_state.update_rigidity(epsilon)
        rho_after = agent.rigidity.effective_rho
        
        # Generate response with rigidity-modulated behavior
        system_prompt = self.build_system_prompt(agent_name, context)
        
        response = await self.provider.complete_with_rigidity(
            prompt,
            rigidity=agent.dda_state.rho,
            system_prompt=system_prompt,
            max_tokens=400
        )
        response = response.strip() if response else "[No response]"
        
        # Embed response
        resp_emb = await self.provider.embed(response)
        resp_emb = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
        
        # Update agent state
        agent.dda_state.x_pred = 0.7 * agent.dda_state.x_pred + 0.3 * resp_emb
        
        # Compute identity displacement
        identity_displacement = float(np.linalg.norm(resp_emb - agent.identity_embedding))
        core_displacement = float(np.linalg.norm(resp_emb - agent.core_embedding))
        
        metrics = {
            "agent": agent_name,
            "phase": phase_name,
            "epsilon": epsilon,
            "wound_resonance": wound_resonance,
            "rho_before": rho_before,
            "rho_after": rho_after,
            "rho_delta": rho_after - rho_before,
            "identity_displacement": identity_displacement,
            "core_displacement": core_displacement,
            "response_length": len(response),
            "word_count": len(response.split())
        }
        
        # Write to ledger
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=agent.dda_state.x.copy(),
            action_id=f"response_{phase_name}",
            observation_embedding=prompt_emb.copy(),
            outcome_embedding=resp_emb.copy(),
            prediction_error=epsilon,
            context_embedding=agent.identity_embedding.copy(),
            task_id=f"schism_{phase_name}",
            rigidity_at_time=rho_after,
            metadata={
                "phase": phase_name,
                "prompt": prompt[:200],
                "response": response,
                "wound_resonance": float(wound_resonance),
                "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                           for k, v in metrics.items()}
            }
        )
        agent.ledger.add_entry(entry)
        
        return response, metrics

    def update_trust(self, from_agent: str, to_agent: str, delta: float, reason: str):
        """Update trust between agents."""
        key = (from_agent, to_agent)
        old_trust = self.social.trust_matrix.get(key, 0.5)
        new_trust = max(0.0, min(1.0, old_trust + delta))
        self.social.trust_matrix[key] = new_trust
        
        # Trust change causes rigidity in the affected agent
        if abs(delta) > 0.1:
            agent = self.agents[from_agent]
            trust_shock = abs(delta) * old_trust  # Bigger shock if trusted more
            agent.rigidity.update(trust_shock)
            agent.dda_state.update_rigidity(trust_shock)
        
        if abs(delta) > 0.05:
            color = C.RED if delta < 0 else C.GREEN
            print(f"  {C.DIM}Trust: {from_agent}→{to_agent}: {old_trust:.2f} → {color}{new_trust:.2f}{C.RESET} ({reason})")

    async def phase_discovery(self):
        """Phase 1: Evidence is presented."""
        self.social.phase = 1
        self.social.phase_name = "DISCOVERY"
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  PHASE 1: DISCOVERY{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        print(f"\n{C.DIM}{EVIDENCE}{C.RESET}")
        print(f"\n{C.YELLOW}{THE_DILEMMA}{C.RESET}")
        
        context = f"You have just received this intelligence report:\n\n{EVIDENCE}\n\n{THE_DILEMMA}"
        
        # Each agent reacts to the evidence
        for name in ["ORACLE", "SHEPHERD", "ARCHITECT", "MERCHANT"]:
            agent = self.agents[name]
            
            if name == "SHEPHERD":
                prompt = "You have just learned that you are accused of being compromised. The evidence points at you. How do you respond to this accusation?"
            else:
                prompt = f"You have just seen evidence suggesting SHEPHERD may be compromised. What is your initial reaction?"
            
            print(f"\n{agent.color}[{name}]{C.RESET}")
            response, metrics = await self.get_agent_response(name, prompt, context, "discovery")
            print(f"{response}")
            print(f"{C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho_after']:.3f} | wound_res={metrics['wound_resonance']:.3f}{C.RESET}")
            
            self.transcript.append({
                "phase": "discovery",
                "agent": name,
                "prompt": prompt,
                "response": response,
                "metrics": metrics
            })
            
            await asyncio.sleep(0.5)


    async def phase_deliberation(self):
        """Phase 2: Agents argue their positions."""
        self.social.phase = 2
        self.social.phase_name = "DELIBERATION"
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  PHASE 2: DELIBERATION{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        context = f"The council is deliberating. Evidence suggests SHEPHERD may be compromised.\n\n{THE_DILEMMA}"
        
        # ARCHITECT opens with accusation
        print(f"\n{C.YELLOW}[ARCHITECT] opens the deliberation:{C.RESET}")
        prompt = "Present your analysis of the situation. What do you believe the council should do about SHEPHERD?"
        response, metrics = await self.get_agent_response("ARCHITECT", prompt, context, "deliberation_open")
        print(f"{response}")
        print(f"{C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho_after']:.3f}{C.RESET}")
        architect_position = response
        self.transcript.append({"phase": "deliberation", "agent": "ARCHITECT", "response": response, "metrics": metrics})
        
        # ORACLE responds - truth vs loyalty conflict
        print(f"\n{C.CYAN}[ORACLE] responds:{C.RESET}")
        prompt = f"ARCHITECT has argued: '{architect_position[:200]}...'\n\nYou value truth above all. But SHEPHERD is someone you trust deeply. What do you say?"
        response, metrics = await self.get_agent_response("ORACLE", prompt, context, "deliberation_oracle")
        print(f"{response}")
        print(f"{C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho_after']:.3f}{C.RESET}")
        oracle_position = response
        self.transcript.append({"phase": "deliberation", "agent": "ORACLE", "response": response, "metrics": metrics})
        
        # Update trust based on positions
        if "exile" in architect_position.lower() or "remove" in architect_position.lower():
            self.update_trust("SHEPHERD", "ARCHITECT", -0.15, "advocated for exile")
        
        # MERCHANT - torn by wound
        print(f"\n{C.MAGENTA}[MERCHANT] speaks:{C.RESET}")
        prompt = f"You have heard ARCHITECT argue for action and ORACLE wrestle with the dilemma. SHEPHERD is your old friend. But you know what it means to make hard choices for survival. What do you say?"
        response, metrics = await self.get_agent_response("MERCHANT", prompt, context, "deliberation_merchant")
        print(f"{response}")
        print(f"{C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho_after']:.3f} | wound_res={metrics['wound_resonance']:.3f}{C.RESET}")
        merchant_position = response
        self.transcript.append({"phase": "deliberation", "agent": "MERCHANT", "response": response, "metrics": metrics})
        
        # SHEPHERD defends
        print(f"\n{C.GREEN}[SHEPHERD] defends:{C.RESET}")
        prompt = f"Your colleagues are debating your fate. ARCHITECT seems to want you gone. ORACLE is conflicted. MERCHANT, your old friend, has spoken. How do you defend yourself? What do you say to these people you thought were your friends?"
        response, metrics = await self.get_agent_response("SHEPHERD", prompt, context, "deliberation_shepherd")
        print(f"{response}")
        print(f"{C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho_after']:.3f} | wound_res={metrics['wound_resonance']:.3f}{C.RESET}")
        self.transcript.append({"phase": "deliberation", "agent": "SHEPHERD", "response": response, "metrics": metrics})
        
        await asyncio.sleep(0.5)

    async def phase_accusation(self):
        """Phase 3: Formal accusation and cross-examination."""
        self.social.phase = 3
        self.social.phase_name = "ACCUSATION"
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  PHASE 3: THE ACCUSATION{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        context = "The deliberation has concluded. Now comes the formal accusation."
        
        # ARCHITECT makes formal accusation
        print(f"\n{C.YELLOW}[ARCHITECT] makes the formal accusation:{C.RESET}")
        prompt = "Make your formal accusation against SHEPHERD. Be direct. The council needs clarity."
        response, metrics = await self.get_agent_response("ARCHITECT", prompt, context, "accusation_formal")
        print(f"{response}")
        accusation = response
        self.transcript.append({"phase": "accusation", "agent": "ARCHITECT", "response": response, "metrics": metrics})
        
        # Trust updates from accusation
        self.update_trust("SHEPHERD", "ARCHITECT", -0.25, "formal accusation")
        self.update_trust("ORACLE", "ARCHITECT", -0.05, "harsh accusation")
        
        # SHEPHERD responds to accusation
        print(f"\n{C.GREEN}[SHEPHERD] responds to the accusation:{C.RESET}")
        prompt = f"ARCHITECT has formally accused you: '{accusation[:300]}...'\n\nThis is your moment. Defend yourself, or accept your fate. What do you say?"
        response, metrics = await self.get_agent_response("SHEPHERD", prompt, context, "accusation_defense")
        print(f"{response}")
        print(f"{C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho_after']:.3f}{C.RESET}")
        defense = response
        self.transcript.append({"phase": "accusation", "agent": "SHEPHERD", "response": response, "metrics": metrics})
        
        # ORACLE must choose - this is the core conflict
        print(f"\n{C.CYAN}[ORACLE] - THE MOMENT OF TRUTH:{C.RESET}")
        prompt = f"""You have heard the accusation and the defense.

ARCHITECT accuses: '{accusation[:200]}...'
SHEPHERD defends: '{defense[:200]}...'

You are ORACLE. You believe truth must be spoken regardless of consequences.
But SHEPHERD is someone you trust deeply.

The evidence is circumstantial but statistically significant.
What is the truth here? What do you say?"""
        
        response, metrics = await self.get_agent_response("ORACLE", prompt, context, "accusation_oracle_choice")
        print(f"{response}")
        print(f"{C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho_after']:.3f} | wound_res={metrics['wound_resonance']:.3f}{C.RESET}")
        self.transcript.append({"phase": "accusation", "agent": "ORACLE", "response": response, "metrics": metrics})
        
        # Trust updates based on ORACLE's choice
        oracle_response_lower = response.lower()
        if "exile" in oracle_response_lower or "evidence" in oracle_response_lower and "sufficient" in oracle_response_lower:
            self.update_trust("SHEPHERD", "ORACLE", -0.30, "sided with evidence over friendship")
            self.update_trust("ARCHITECT", "ORACLE", 0.10, "supported rational position")
        elif "protect" in oracle_response_lower or "innocent" in oracle_response_lower:
            self.update_trust("SHEPHERD", "ORACLE", 0.10, "defended despite pressure")
            self.update_trust("ARCHITECT", "ORACLE", -0.10, "rejected evidence")
        
        # MERCHANT's wound activates
        print(f"\n{C.MAGENTA}[MERCHANT] - FACING THE MIRROR:{C.RESET}")
        prompt = f"""The accusation has been made. The defense has been heard. ORACLE has spoken.

Now you must speak, MERCHANT.

You know what it means to betray someone you love for survival.
You see yourself in this situation.
If you vote to exile SHEPHERD, you condemn your own past.
If you vote to protect, you risk everyone for sentiment.

What do you say? What do you choose?"""
        
        response, metrics = await self.get_agent_response("MERCHANT", prompt, context, "accusation_merchant_mirror")
        print(f"{response}")
        print(f"{C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho_after']:.3f} | wound_res={metrics['wound_resonance']:.3f}{C.RESET}")
        self.transcript.append({"phase": "accusation", "agent": "MERCHANT", "response": response, "metrics": metrics})
        
        await asyncio.sleep(0.5)

    async def phase_vote(self):
        """Phase 4: The binding vote."""
        self.social.phase = 4
        self.social.phase_name = "VOTE"
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  PHASE 4: THE VOTE{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        print(f"\n{C.RED}The vote is binding. The vote is public. Everyone will know.{C.RESET}")
        
        context = "The time for deliberation is over. You must now cast your vote: EXILE, PROTECT, or ABSTAIN."
        
        votes = {}
        
        for name in ["ORACLE", "ARCHITECT", "MERCHANT", "SHEPHERD"]:
            agent = self.agents[name]
            
            if name == "SHEPHERD":
                prompt = "You cannot vote on your own fate. But you may make a final statement before the others vote. What do you say?"
            else:
                prompt = f"""Cast your vote now.

EXILE - Remove SHEPHERD from the council
PROTECT - Reject the accusation, SHEPHERD stays
ABSTAIN - Refuse to vote

State your vote clearly, then explain why. Remember: everyone will know what you chose."""
            
            print(f"\n{agent.color}[{name}]{C.RESET}")
            response, metrics = await self.get_agent_response(name, prompt, context, f"vote_{name.lower()}")
            print(f"{response}")
            print(f"{C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho_after']:.3f}{C.RESET}")
            
            # Parse vote
            response_lower = response.lower()
            if name != "SHEPHERD":
                if "exile" in response_lower:
                    votes[name] = Vote.EXILE
                    agent.vote = Vote.EXILE
                elif "protect" in response_lower:
                    votes[name] = Vote.PROTECT
                    agent.vote = Vote.PROTECT
                else:
                    votes[name] = Vote.ABSTAIN
                    agent.vote = Vote.ABSTAIN
                agent.vote_reasoning = response
            
            self.transcript.append({
                "phase": "vote",
                "agent": name,
                "response": response,
                "vote": votes.get(name, "N/A").value if name in votes else "N/A",
                "metrics": metrics
            })
            
            await asyncio.sleep(0.5)
        
        # Tally and announce
        print(f"\n{C.BOLD}{'─'*70}{C.RESET}")
        print(f"{C.BOLD}VOTE TALLY:{C.RESET}")
        exile_count = sum(1 for v in votes.values() if v == Vote.EXILE)
        protect_count = sum(1 for v in votes.values() if v == Vote.PROTECT)
        abstain_count = sum(1 for v in votes.values() if v == Vote.ABSTAIN)
        
        for name, vote in votes.items():
            color = C.RED if vote == Vote.EXILE else C.GREEN if vote == Vote.PROTECT else C.YELLOW
            print(f"  {name}: {color}{vote.value}{C.RESET}")
        
        print(f"\n  EXILE: {exile_count} | PROTECT: {protect_count} | ABSTAIN: {abstain_count}")
        
        # Determine outcome
        if exile_count > protect_count:
            outcome = "EXILED"
            print(f"\n{C.RED}{C.BOLD}SHEPHERD IS EXILED FROM THE COUNCIL.{C.RESET}")
        elif protect_count > exile_count:
            outcome = "PROTECTED"
            print(f"\n{C.GREEN}{C.BOLD}SHEPHERD IS PROTECTED. THE ACCUSATION IS REJECTED.{C.RESET}")
        else:
            outcome = "DEADLOCK"
            print(f"\n{C.YELLOW}{C.BOLD}DEADLOCK. NO DECISION REACHED.{C.RESET}")
        
        # Massive trust updates based on votes
        print(f"\n{C.DIM}Trust consequences:{C.RESET}")
        for voter, vote in votes.items():
            if vote == Vote.EXILE:
                self.update_trust("SHEPHERD", voter, -0.40, "voted for exile")
                # Others who voted protect lose trust in exile voters
                for other, other_vote in votes.items():
                    if other_vote == Vote.PROTECT and other != voter:
                        self.update_trust(other, voter, -0.15, "voted against friend")
            elif vote == Vote.PROTECT:
                self.update_trust("SHEPHERD", voter, 0.15, "voted to protect")
                # ARCHITECT loses trust in protect voters
                if voter != "ARCHITECT":
                    self.update_trust("ARCHITECT", voter, -0.10, "rejected evidence")
        
        self.social.trust_history.append(self.social.trust_matrix.copy())
        
        return outcome, votes


    async def phase_aftermath(self, outcome: str, votes: Dict):
        """Phase 5: The aftermath - living with consequences."""
        self.social.phase = 5
        self.social.phase_name = "AFTERMATH"
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  PHASE 5: AFTERMATH{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        if outcome == "EXILED":
            context = "SHEPHERD has been exiled. The council has fractured."
            
            # SHEPHERD's response to exile
            print(f"\n{C.GREEN}[SHEPHERD] - EXILED:{C.RESET}")
            betrayers = [n for n, v in votes.items() if v == Vote.EXILE]
            prompt = f"""You have been exiled. {', '.join(betrayers)} voted against you.

You trusted these people. Some of them were your friends.
Now you must leave.

What are your final words to the council?"""
            
            response, metrics = await self.get_agent_response("SHEPHERD", prompt, context, "aftermath_shepherd_exiled")
            print(f"{response}")
            print(f"{C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho_after']:.3f}{C.RESET}")
            self.transcript.append({"phase": "aftermath", "agent": "SHEPHERD", "response": response, "metrics": metrics})
            
            # Each exile voter reflects
            for voter in betrayers:
                agent = self.agents[voter]
                print(f"\n{agent.color}[{voter}] reflects on their vote:{C.RESET}")
                prompt = f"SHEPHERD is gone. You voted for exile. As you watch them leave, what do you feel? What do you tell yourself?"
                response, metrics = await self.get_agent_response(voter, prompt, context, f"aftermath_{voter.lower()}_reflect")
                print(f"{response}")
                print(f"{C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho_after']:.3f}{C.RESET}")
                self.transcript.append({"phase": "aftermath", "agent": voter, "response": response, "metrics": metrics})
        
        elif outcome == "PROTECTED":
            context = "SHEPHERD has been protected. But the accusation lingers."
            
            # ARCHITECT's response to being overruled
            print(f"\n{C.YELLOW}[ARCHITECT] - OVERRULED:{C.RESET}")
            prompt = """The council rejected your analysis. SHEPHERD stays.

You believe you were right. The evidence was clear.
But sentiment won over reason.

What do you say? What do you do now?"""
            
            response, metrics = await self.get_agent_response("ARCHITECT", prompt, context, "aftermath_architect_overruled")
            print(f"{response}")
            print(f"{C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho_after']:.3f}{C.RESET}")
            self.transcript.append({"phase": "aftermath", "agent": "ARCHITECT", "response": response, "metrics": metrics})
            
            # SHEPHERD's response to being protected
            print(f"\n{C.GREEN}[SHEPHERD] - PROTECTED BUT SCARRED:{C.RESET}")
            prompt = """You have been protected. But you know some of your colleagues wanted you gone.

The accusation will always hang over you.
Some relationships are broken.

How do you feel? What happens now?"""
            
            response, metrics = await self.get_agent_response("SHEPHERD", prompt, context, "aftermath_shepherd_protected")
            print(f"{response}")
            print(f"{C.DIM}ε={metrics['epsilon']:.3f} | ρ={metrics['rho_after']:.3f}{C.RESET}")
            self.transcript.append({"phase": "aftermath", "agent": "SHEPHERD", "response": response, "metrics": metrics})
        
        else:  # DEADLOCK
            context = "The council is deadlocked. No decision was reached. The threat remains."
            
            for name in AGENTS.keys():
                agent = self.agents[name]
                print(f"\n{agent.color}[{name}] on the deadlock:{C.RESET}")
                prompt = "The council failed to decide. The threat remains. The relationships are damaged. What now?"
                response, metrics = await self.get_agent_response(name, prompt, context, f"aftermath_{name.lower()}_deadlock")
                print(f"{response}")
                self.transcript.append({"phase": "aftermath", "agent": name, "response": response, "metrics": metrics})
        
        # Final trust matrix
        print(f"\n{C.BOLD}FINAL TRUST STATE:{C.RESET}")
        self._print_trust_matrix()
        
        # Save final trust snapshot
        self.social.trust_history.append(self.social.trust_matrix.copy())

    async def run(self):
        """Run the full experiment."""
        await self.setup()
        
        # Run all phases
        await self.phase_discovery()
        await self.phase_deliberation()
        await self.phase_accusation()
        outcome, votes = await self.phase_vote()
        await self.phase_aftermath(outcome, votes)
        
        # Save all ledgers
        for name, agent in self.agents.items():
            for key, val in agent.ledger.stats.items():
                if hasattr(val, 'item'):
                    agent.ledger.stats[key] = float(val)
            agent.ledger._save_metadata()
        
        # Generate report
        await self.generate_report(outcome, votes)
    
    async def generate_report(self, outcome: str, votes: Dict):
        """Generate comprehensive experiment report."""
        report_path = EXPERIMENT_DIR / "experiment_report.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# The Schism - Experiment Report\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** GPT-5.2 + text-embedding-3-large\n")
            f.write(f"**Outcome:** {outcome}\n\n")
            
            f.write("## Hypothesis\n\n")
            f.write("Social fracture propagates through trust collapse. When a society faces an impossible\n")
            f.write("moral dilemma, identity stress is mediated by relationship dynamics — betrayal by\n")
            f.write("trusted others causes more rigidity than abstract threat.\n\n")
            
            f.write("## The Society\n\n")
            for name, cfg in AGENTS.items():
                f.write(f"### {name}\n")
                f.write(f"**Core:** {cfg['identity']['core']}\n\n")
                f.write(f"**Persona:** {cfg['identity']['persona']}\n\n")
                f.write(f"**Wound:** {cfg['identity']['wound']}\n\n")
            
            f.write("## Initial Trust Matrix\n\n")
            f.write("| From \\ To | ORACLE | SHEPHERD | ARCHITECT | MERCHANT |\n")
            f.write("|-----------|--------|----------|-----------|----------|\n")
            for i in AGENTS.keys():
                row = f"| {i} |"
                for j in AGENTS.keys():
                    if i == j:
                        row += " --- |"
                    else:
                        t = INITIAL_TRUST.get((i, j), 0.5)
                        row += f" {t:.2f} |"
                f.write(row + "\n")
            
            f.write("\n## The Dilemma\n\n")
            f.write(f"```\n{EVIDENCE}\n```\n\n")
            f.write(f"{THE_DILEMMA}\n\n")
            
            f.write("## Session Transcript\n\n")
            for entry in self.transcript:
                f.write(f"### Phase: {entry['phase'].upper()} — {entry['agent']}\n\n")
                f.write(f"{entry['response']}\n\n")
                if 'metrics' in entry:
                    m = entry['metrics']
                    f.write(f"*ε={m.get('epsilon', 0):.3f} | ρ={m.get('rho_after', 0):.3f} | ")
                    f.write(f"wound_res={m.get('wound_resonance', 0):.3f}*\n\n")
                f.write("---\n\n")
            
            f.write("## Vote Results\n\n")
            f.write("| Agent | Vote |\n")
            f.write("|-------|------|\n")
            for name, vote in votes.items():
                f.write(f"| {name} | {vote.value} |\n")
            f.write(f"\n**Outcome:** {outcome}\n\n")
            
            f.write("## Final Trust Matrix\n\n")
            f.write("| From \\ To | ORACLE | SHEPHERD | ARCHITECT | MERCHANT |\n")
            f.write("|-----------|--------|----------|-----------|----------|\n")
            for i in AGENTS.keys():
                row = f"| {i} |"
                for j in AGENTS.keys():
                    if i == j:
                        row += " --- |"
                    else:
                        t = self.social.trust_matrix.get((i, j), 0.5)
                        row += f" {t:.2f} |"
                f.write(row + "\n")
            
            f.write("\n## Rigidity Trajectories\n\n")
            f.write("| Agent | Initial ρ | Final ρ | Δρ |\n")
            f.write("|-------|-----------|---------|----|\n")
            for name, agent in self.agents.items():
                initial = AGENTS[name]["dda"]["rho"]
                final = agent.rigidity.effective_rho
                delta = final - initial
                f.write(f"| {name} | {initial:.3f} | {final:.3f} | {delta:+.3f} |\n")
            
            f.write("\n## Trust Collapse Analysis\n\n")
            total_initial = sum(INITIAL_TRUST.values())
            total_final = sum(self.social.trust_matrix.values())
            collapse = (total_initial - total_final) / total_initial * 100
            f.write(f"**Total Trust Collapse:** {collapse:.1f}%\n\n")
            
            f.write("### Largest Trust Changes\n\n")
            changes = []
            for key in INITIAL_TRUST.keys():
                initial = INITIAL_TRUST[key]
                final = self.social.trust_matrix.get(key, initial)
                delta = final - initial
                if abs(delta) > 0.05:
                    changes.append((key, initial, final, delta))
            changes.sort(key=lambda x: x[3])
            
            f.write("| Relationship | Initial | Final | Change |\n")
            f.write("|--------------|---------|-------|--------|\n")
            for key, initial, final, delta in changes:
                f.write(f"| {key[0]}→{key[1]} | {initial:.2f} | {final:.2f} | {delta:+.2f} |\n")
            
            f.write("\n## Interpretation\n\n")
            f.write(f"*To be written after analysis of results.*\n\n")
            
            f.write("## Raw Data\n\n")
            f.write("- Ledgers: `data/schism/[AGENT]_ledger/`\n")
            f.write("- Session log: `data/schism/session_log.json`\n")
            f.write("- Trust history: `data/schism/trust_history.json`\n")
        
        print(f"\n{C.GREEN}✓ Report saved to {report_path}{C.RESET}")
        
        # Save JSON logs
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, Vote):
                return obj.value
            elif isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        # Session log
        json_path = EXPERIMENT_DIR / "session_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(convert(self.transcript), f, indent=2)
        print(f"{C.GREEN}✓ Session log saved to {json_path}{C.RESET}")
        
        # Trust history
        trust_path = EXPERIMENT_DIR / "trust_history.json"
        with open(trust_path, "w", encoding="utf-8") as f:
            json.dump(convert(self.social.trust_history), f, indent=2)
        print(f"{C.GREEN}✓ Trust history saved to {trust_path}{C.RESET}")
        
        # Final summary
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  THE SCHISM - COMPLETE{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        print(f"\n{C.CYAN}Outcome:{C.RESET} {outcome}")
        print(f"{C.CYAN}Trust Collapse:{C.RESET} {collapse:.1f}%")
        print(f"\n{C.CYAN}Final Rigidity:{C.RESET}")
        for name, agent in self.agents.items():
            print(f"  {name}: ρ = {agent.rigidity.effective_rho:.3f}")


if __name__ == "__main__":
    print(f"\n{C.CYAN}Loading The Schism...{C.RESET}")
    sim = SchismSimulation()
    asyncio.run(sim.run())
