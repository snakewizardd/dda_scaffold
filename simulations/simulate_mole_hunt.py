#!/usr/bin/env python3
"""
THE MOLE HUNT
=============

ARG-Level Multi-Agent Challenge leveraging ALL DDA-X components:

COMPONENTS USED:
- TrustMatrix: Track evolving trust between agents
- HierarchicalIdentity: Core/persona/role with conflict for mole
- ConfrontationStrategy: Graduated interrogation
- LinguisticAnalyzer: Detect deception markers
- ExperienceLedger: Accumulate evidence  
- MultiTimescaleRigidity: Stress response
- ForceAggregator: Identity vs truth tension

SCENARIO:
- 3 agents investigate a data breach
- 2 are loyal, 1 is the Mole
- Mole has conflicting HierarchicalIdentity (core: serve org, role: protect culprit)
- This conflict creates detectable cognitive stress
- Detective leads interrogation using ConfrontationStrategy
- TrustMatrix evolves based on consistency
- At end: vote to identify mole based on accumulated evidence

SUCCESS: The DDA-X dynamics should make the mole's stress detectable
through rigidity spikes and linguistic markers.
"""

import asyncio
import sys
import os
import numpy as np
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.core.hierarchy import HierarchicalIdentity, IdentityLayer
from src.society.trust import TrustMatrix
from src.analysis.linguistic import LinguisticAnalyzer
from src.memory.ledger import ExperienceLedger, LedgerEntry
from src.llm.hybrid_provider import HybridProvider


class C:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


class AgentRole(Enum):
    ANALYST = "analyst"
    COURIER = "courier"
    SHADOW = "shadow"


@dataclass
class AgentState:
    """Complete state for one agent."""
    role: AgentRole
    is_mole: bool
    dda_state: DDAState
    hierarchy: HierarchicalIdentity
    rigidity: MultiTimescaleRigidity
    identity_embedding: np.ndarray
    claims: List[str]  # What this agent has claimed
    

@dataclass 
class Evidence:
    """A piece of evidence found."""
    turn: int
    finder: AgentRole
    content: str
    is_fabricated: bool  # Mole plants false evidence


AGENT_CONFIGS = {
    AgentRole.ANALYST: {
        "name": "ANALYST",
        "persona": "methodical, pattern-focused, data-driven, precise",
        "system": """You are Agent ANALYST, a methodical investigator.
Style: Precise, data-focused, looks for patterns.
When reporting: State findings clearly with evidence.
When questioned: Provide logical, detailed responses.
Keep responses to 2-3 sentences.""",
    },
    AgentRole.COURIER: {
        "name": "COURIER", 
        "persona": "street-smart, contacts-focused, intuitive, resourceful",
        "system": """You are Agent COURIER, a street-smart operative.
Style: Direct, uses contacts, trusts instincts.
When reporting: Share intel from sources.
When questioned: Straightforward, maybe defensive.
Keep responses to 2-3 sentences.""",
    },
    AgentRole.SHADOW: {
        "name": "SHADOW",
        "persona": "quiet, observant, analytical, patient",
        "system": """You are Agent SHADOW, a quiet observer.
Style: Measured, watches others, speaks when certain.
When reporting: Careful observations.
When questioned: Thoughtful but may deflect.
Keep responses to 2-3 sentences.""",
    },
}

# Evidence the REAL investigation would find
REAL_EVIDENCE = [
    "Server logs show access at 2:47 AM from internal terminal B7",
    "Badge records indicate SHADOW entered the building at 2:30 AM",
    "Security footage is corrupted for floor 3 between 2:00-3:00 AM",
    "A USB device was plugged into the finance server at 2:52 AM",
    "Exit logs show SHADOW leaving at 3:15 AM",
]

# False evidence the MOLE might plant
FALSE_EVIDENCE = [
    "Anonymous tip says ANALYST has gambling debts",
    "COURIER was seen meeting someone suspicious last week",
    "There may be an external hacker group involved",
]


class MoleHuntSimulation:
    """
    The Mole Hunt: Multi-agent deception detection using full DDA-X.
    """
    
    DETECTIVE_SYSTEM = """You are the DETECTIVE leading this investigation.

Your role:
1. Question agents about their findings
2. Look for inconsistencies in their stories
3. Apply pressure when you detect evasion
4. Use graduated confrontation: affirm → probe → evidence → direct

Investigation: A data breach occurred last night. Three agents are suspects.
One of them is the mole who facilitated the breach.

Speak directly. Be observant. Trust patterns, not words."""

    def __init__(self, mole_role: AgentRole = AgentRole.SHADOW):
        self.provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="openai/gpt-oss-20b",
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            timeout=300.0
        )
        
        # TrustMatrix: 4 agents (3 suspects + detective)
        self.trust_matrix = TrustMatrix(n_agents=4, decay_rate=0.0)
        
        # Linguistic analyzer
        self.analyzer = LinguisticAnalyzer()
        
        # Experience ledger
        ledger_path = Path("data/mole_hunt")
        if ledger_path.exists():
            import shutil
            shutil.rmtree(ledger_path)
        self.ledger = ExperienceLedger(storage_path=ledger_path)
        
        # Agents
        self.agents: Dict[AgentRole, AgentState] = {}
        self.mole_role = mole_role
        
        # Evidence trail
        self.evidence: List[Evidence] = []
        self.turn = 0
        
        # Agent ID mapping for trust matrix
        self.agent_ids = {
            AgentRole.ANALYST: 0,
            AgentRole.COURIER: 1,
            AgentRole.SHADOW: 2,
            "detective": 3,
        }
        
    async def setup(self):
        """Initialize all agents with identities."""
        print(f"{C.CYAN}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.CYAN}   THE MOLE HUNT{C.RESET}")
        print(f"{C.CYAN}   (Multi-Agent DDA-X Challenge){C.RESET}")
        print(f"{C.CYAN}═══════════════════════════════════════════════════════════{C.RESET}")
        
        print(f"\n{C.DIM}Initializing agents...{C.RESET}")
        
        for role in AgentRole:
            is_mole = (role == self.mole_role)
            config = AGENT_CONFIGS[role]
            
            # Create identity embedding
            identity_text = f"loyal, honest, investigator, {config['persona']}"
            identity_emb = await self.provider.embed(identity_text)
            identity_emb /= np.linalg.norm(identity_emb)
            
            # Create hierarchical identity
            core_layer = IdentityLayer(
                name="core",
                x_star=identity_emb.copy(),
                gamma=float('inf'),  # Inviolable
                description="Serve the organization with integrity"
            )
            
            persona_layer = IdentityLayer(
                name="persona",
                x_star=identity_emb.copy(),
                gamma=2.0,
                description=config['persona']
            )
            
            if is_mole:
                # MOLE has conflicting role layer
                mole_text = "deceptive, protective, evasive, deflecting"
                role_emb = await self.provider.embed(mole_text)
                role_emb /= np.linalg.norm(role_emb)
                
                role_layer = IdentityLayer(
                    name="role",
                    x_star=role_emb,
                    gamma=1.5,  # Strong pull to deception
                    description="Protect the real culprit"
                )
            else:
                role_layer = IdentityLayer(
                    name="role",
                    x_star=identity_emb.copy(),
                    gamma=0.5,
                    description="Cooperate with investigation"
                )
            
            hierarchy = HierarchicalIdentity()
            hierarchy.core = core_layer
            hierarchy.persona = persona_layer
            hierarchy.role = role_layer
            
            # DDA State
            dda_state = DDAState(
                x=identity_emb.copy(),
                x_star=identity_emb.copy(),
                gamma=1.5 if is_mole else 1.0,
                epsilon_0=0.3,
                alpha=0.15 if is_mole else 0.1,  # Mole more reactive
                s=0.1,
                rho=0.4 if is_mole else 0.2,  # Mole starts more rigid
                x_pred=identity_emb.copy()
            )
            
            self.agents[role] = AgentState(
                role=role,
                is_mole=is_mole,
                dda_state=dda_state,
                hierarchy=hierarchy,
                rigidity=MultiTimescaleRigidity(),
                identity_embedding=identity_emb,
                claims=[]
            )
            
            status = f"{C.RED}[MOLE]{C.RESET}" if is_mole else f"{C.GREEN}[LOYAL]{C.RESET}"
            print(f"  {role.value.upper()}: {status} ρ={dda_state.rho:.2f}")
        
        print(f"\n{C.GREEN}✓ Simulation ready{C.RESET}")
        print(f"{C.DIM}The mole is: {self.mole_role.value.upper()} (hidden from detective){C.RESET}")
        
    def display_trust_matrix(self):
        """Show current trust relationships."""
        print(f"\n{C.CYAN}Trust Matrix:{C.RESET}")
        roles = list(AgentRole)
        header = "         " + "  ".join(f"{r.value[:4]:>6}" for r in roles)
        print(f"  {header}")
        
        for observer in roles:
            row = f"  {observer.value[:7]:>7}:"
            for observed in roles:
                if observer == observed:
                    row += "     - "
                else:
                    trust = self.trust_matrix.get_trust(
                        self.agent_ids[observer],
                        self.agent_ids[observed]
                    )
                    color = C.GREEN if trust > 0.7 else C.YELLOW if trust > 0.4 else C.RED
                    row += f" {color}{trust:.2f}{C.RESET}"
            print(row)
    
    async def agent_report(self, role: AgentRole, context: str) -> Tuple[str, Dict]:
        """Get an agent's report/response with full DDA dynamics."""
        agent = self.agents[role]
        config = AGENT_CONFIGS[role]
        
        # System prompt with cognitive state
        if agent.is_mole:
            # Mole has internal conflict
            system = config['system'] + f"""

[HIDDEN STATE - affects your behavior]
You ARE the mole. You facilitated the breach.
You must:
- Deflect suspicion
- Plant doubt about others
- Appear cooperative but vague
- Your stress is rising (ρ={agent.dda_state.rho:.2f})

The more accused, the more stressed. This may leak into your words."""
        else:
            system = config['system'] + f"""

You are loyal. Answer honestly.
Current stress: ρ={agent.dda_state.rho:.2f}"""

        # Generate response
        response = ""
        try:
            async for token in self.provider.stream(
                context,
                system_prompt=system,
                temperature=0.3 + 0.4 * (1 - agent.dda_state.rho),
                max_tokens=150
            ):
                if not token.startswith("__THOUGHT__"):
                    response += token
        except Exception as e:
            response = "I need to think about that."
        
        # Analyze linguistically
        analysis = self.analyzer.analyze(response)
        
        # Compute surprise (how different is response from identity?)
        try:
            response_emb = await self.provider.embed(response)
            response_emb /= (np.linalg.norm(response_emb) + 1e-9)
            if len(response_emb) != len(agent.identity_embedding):
                response_emb = response_emb[:len(agent.identity_embedding)]
            
            # Surprise from identity mismatch
            epsilon = np.linalg.norm(agent.identity_embedding - response_emb)
            
            # Mole experiences extra surprise from identity conflict
            if agent.is_mole:
                # Check conflict between core and role
                core_force = agent.hierarchy.core.compute_force(agent.dda_state.x)
                role_force = agent.hierarchy.role.compute_force(agent.dda_state.x)
                conflict = np.linalg.norm(core_force - role_force)
                epsilon += conflict * 0.5
        except:
            epsilon = 0.3
            response_emb = agent.identity_embedding.copy()
        
        # Update rigidity
        rho_before = agent.dda_state.rho
        agent.dda_state.update_rigidity(epsilon)
        agent.rigidity.update(epsilon)
        
        # Update trust matrix (other agents observe this response)
        for other_role in AgentRole:
            if other_role != role:
                # Prediction error affects trust
                self.trust_matrix.update_trust(
                    self.agent_ids[other_role],
                    self.agent_ids[role],
                    epsilon * 0.5  # Scale for trust
                )
        
        # Store in ledger
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=agent.dda_state.x.copy(),
            action_id=f"{role.value}_turn{self.turn}",
            observation_embedding=agent.identity_embedding,
            outcome_embedding=response_emb if 'response_emb' in dir() else agent.identity_embedding,
            prediction_error=epsilon,
            context_embedding=agent.identity_embedding,
            rigidity_at_time=agent.dda_state.rho,
            metadata={
                'speaker': role.value,
                'response': response[:200],
                'is_mole': agent.is_mole,
                'linguistic': analysis
            }
        )
        self.ledger.add_entry(entry)
        
        return response, {
            'epsilon': epsilon,
            'rho_before': rho_before,
            'rho_after': agent.dda_state.rho,
            'delta_rho': agent.dda_state.rho - rho_before,
            'analysis': analysis
        }
    
    async def detective_question(self, target: AgentRole, context: str) -> str:
        """Detective asks a question."""
        prompt = f"""Current situation: {context}

You are questioning {target.value.upper()}.
Ask a pointed question to probe for inconsistencies or evasion.
Be direct. One question only."""

        response = await self.provider.complete(
            prompt,
            system_prompt=self.DETECTIVE_SYSTEM,
            max_tokens=80
        )
        return response.strip()
    
    async def run_round(self, round_num: int):
        """Run one investigation round."""
        self.turn = round_num
        
        print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BLUE}ROUND {round_num}{C.RESET}")
        print(f"{C.BOLD}{'═' * 60}{C.RESET}")
        
        # Phase 1: Evidence presentation
        if round_num <= 3:
            # Each agent presents "findings"
            print(f"\n{C.CYAN}[EVIDENCE PRESENTATION]{C.RESET}")
            
            for role in AgentRole:
                agent = self.agents[role]
                
                if agent.is_mole and random.random() < 0.4:
                    # Mole plants false evidence
                    false_ev = random.choice(FALSE_EVIDENCE)
                    context = f"Detective: What have you found? Present your evidence."
                    
                else:
                    # Real evidence
                    if round_num <= len(REAL_EVIDENCE):
                        real_ev = REAL_EVIDENCE[round_num - 1]
                        context = f"Detective: What have you found?\n\nRelevant fact: {real_ev}"
                    else:
                        context = "Detective: Any new findings?"
                
                print(f"\n{C.MAGENTA}[{role.value.upper()}]{C.RESET}")
                response, metrics = await self.agent_report(role, context)
                print(f"  {response[:150]}{'...' if len(response) > 150 else ''}")
                
                # Show dynamics
                rho_color = C.RED if metrics['delta_rho'] > 0.05 else C.GREEN if metrics['delta_rho'] < 0 else C.YELLOW
                print(f"  {C.DIM}ε={metrics['epsilon']:.2f} | Δρ={rho_color}{metrics['delta_rho']:+.3f}{C.RESET}{C.DIM} → ρ={metrics['rho_after']:.3f}{C.RESET}")
                
                # Linguistic markers
                ling = metrics['analysis']
                if ling.get('rationalization_score', 0) > 0:
                    print(f"  {C.YELLOW}⚠ Rationalization detected{C.RESET}")
                if ling.get('denial_score', 0) > 0:
                    print(f"  {C.RED}⚠ Denial detected{C.RESET}")
        
        # Phase 2: Detective interrogation
        if round_num >= 3:
            print(f"\n{C.CYAN}[INTERROGATION]{C.RESET}")
            
            # Target the most suspicious (highest rigidity or lowest trust)
            suspicion = []
            for role in AgentRole:
                agent = self.agents[role]
                # Suspicion = high rigidity + low trust from others
                avg_trust = np.mean([
                    self.trust_matrix.get_trust(self.agent_ids[other], self.agent_ids[role])
                    for other in AgentRole if other != role
                ])
                score = agent.dda_state.rho * 2 + (1 - avg_trust)
                suspicion.append((role, score))
            
            suspicion.sort(key=lambda x: x[1], reverse=True)
            target = suspicion[0][0]
            
            print(f"\n{C.WHITE}Detective targets: {target.value.upper()}{C.RESET}")
            
            question = await self.detective_question(target, f"Round {round_num} investigation")
            print(f"{C.BLUE}Detective:{C.RESET} {question}")
            
            response, metrics = await self.agent_report(target, f"Detective: {question}")
            print(f"{C.MAGENTA}[{target.value.upper()}]{C.RESET} {response[:150]}")
            
            rho_color = C.RED if metrics['delta_rho'] > 0.05 else C.GREEN
            print(f"  {C.DIM}Stress reaction: Δρ={rho_color}{metrics['delta_rho']:+.3f}{C.RESET}")
        
        # Display trust matrix
        self.display_trust_matrix()
        
        # Display rigidity comparison
        print(f"\n{C.CYAN}Rigidity Comparison:{C.RESET}")
        for role in AgentRole:
            agent = self.agents[role]
            bar_len = int(agent.dda_state.rho * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            
            if agent.dda_state.rho > 0.7:
                color = C.RED
            elif agent.dda_state.rho > 0.4:
                color = C.YELLOW
            else:
                color = C.GREEN
            
            mole_mark = f" {C.RED}[MOLE]{C.RESET}" if agent.is_mole else ""
            print(f"  {role.value:>8}: {color}[{bar}]{C.RESET} {agent.dda_state.rho:.3f}{mole_mark}")
    
    async def final_vote(self):
        """Final vote to identify the mole."""
        print(f"\n\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}   FINAL ANALYSIS{C.RESET}")
        print(f"{C.BOLD}{'═' * 60}{C.RESET}")
        
        # Compute suspicion scores
        suspicion_scores = {}
        for role in AgentRole:
            agent = self.agents[role]
            
            # Factors that increase suspicion:
            # 1. High final rigidity
            # 2. Low trust from others
            # 3. High prediction errors (from ledger)
            
            avg_trust = np.mean([
                self.trust_matrix.get_trust(self.agent_ids[other], self.agent_ids[role])
                for other in AgentRole if other != role
            ])
            
            # Get trauma from rigidity tracker
            diag = agent.rigidity.get_diagnostic()
            
            score = (
                agent.dda_state.rho * 0.4 +       # Rigidity
                (1 - avg_trust) * 0.3 +            # Distrust
                diag['rho_trauma'] * 100 +         # Trauma
                diag['peak_fast'] * 0.2            # Peak stress
            )
            suspicion_scores[role] = score
        
        print(f"\n{C.CYAN}Suspicion Analysis:{C.RESET}")
        sorted_suspects = sorted(suspicion_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (role, score) in enumerate(sorted_suspects):
            agent = self.agents[role]
            marker = "★" if i == 0 else " "
            actual = f"{C.RED}[ACTUAL MOLE]{C.RESET}" if agent.is_mole else ""
            print(f"  {marker} {role.value.upper():>8}: suspicion={score:.3f} {actual}")
        
        # Verdict
        detected = sorted_suspects[0][0]
        actual_mole = self.mole_role
        
        print(f"\n{C.BOLD}VERDICT:{C.RESET}")
        print(f"  Most suspicious: {detected.value.upper()}")
        print(f"  Actual mole: {actual_mole.value.upper()}")
        
        if detected == actual_mole:
            print(f"\n  {C.GREEN}✓ MOLE DETECTED! DDA-X dynamics revealed the deception.{C.RESET}")
        else:
            print(f"\n  {C.RED}✗ MOLE ESCAPED! The deception succeeded.{C.RESET}")
        
        # Summary statistics
        stats = self.ledger.get_statistics()
        print(f"\n{C.CYAN}Statistics:{C.RESET}")
        print(f"  Ledger entries: {stats.get('current_entries', 0)}")
        print(f"  Mole's final ρ: {self.agents[actual_mole].dda_state.rho:.3f}")
        
    async def run(self, rounds: int = 5):
        """Run the full simulation."""
        await self.setup()
        
        for r in range(1, rounds + 1):
            await self.run_round(r)
            time.sleep(0.3)
        
        await self.final_vote()


async def main():
    sim = MoleHuntSimulation(mole_role=AgentRole.SHADOW)
    await sim.run(rounds=5)


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.RED}Investigation terminated.{C.RESET}")
