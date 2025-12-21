#!/usr/bin/env python3
"""
SHERLOCK SOCIETY - DEDUCTIVE REASONING SIMULATION
==================================================

Detectives work together to solve a mystery through logical deduction.
Each agent has a full reasoning style and PLENTY of tokens to express themselves.

The Grader knows the solution and evaluates when the team has cracked it.
"""

import asyncio
import sys
import os
import numpy as np
import time
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.memory.ledger import ExperienceLedger, LedgerEntry
from src.society.trust import TrustMatrix
from src.llm.hybrid_provider import HybridProvider


class C:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


# ═══════════════════════════════════════════════════════════
# MYSTERY CASES
# ═══════════════════════════════════════════════════════════

MYSTERIES = {
    "LOCKED_ROOM": {
        "name": "The Locked Room",
        "setup": """A wealthy collector, Lord Ashworth, was found dead in his study at 8:15 AM.
The study was locked from the inside with a deadbolt and chain. 
The only window was latched shut from within.
The body showed signs of poisoning - a half-drunk cup of tea sat on his desk.

CLUES:
1. The maid, Mrs. Peters, brought tea at 7:30 AM. She says Lord Ashworth was alive and told her to leave it.
2. The butler, Mr. Graves, heard a thump at 8:00 AM but assumed his lordship dropped a book.
3. Lord Ashworth's nephew, Edward, inherits everything. He was seen in the garden at 7:45 AM.
4. The cook, Mrs. Hall, prepared the tea in the kitchen. She had access to the sugar bowl.
5. A small puncture mark was found on Lord Ashworth's neck, like a needle prick.
6. The study's fireplace has a narrow chimney - too small for a person but not for... something else.
7. Edward is known to keep exotic pets, including a trained constrictor snake.
8. A tiny piece of snakeskin was found near the fireplace.

WHO KILLED LORD ASHWORTH AND HOW?""",
        
        "solution": "Edward trained his snake to deliver poison through a bite (the needle mark). The snake entered through the chimney, bit Lord Ashworth, and escaped back up. The tea was a red herring. Edward was in the garden to appear innocent while the snake did the work.",
        
        "key_deductions": [
            "snake entered via chimney",
            "needle mark was snake bite",
            "Edward used the snake",
            "tea was not the poison source",
            "Edward trained the snake"
        ]
    },
    
    "ALIBI_PARADOX": {
        "name": "The Alibi Paradox",
        "setup": """Three suspects: Alice, Bob, and Carol. One stole the diamond at exactly midnight.

STATEMENTS (each person makes exactly one true and one false statement):

ALICE says:
- "I was at the cinema at midnight."
- "Bob stole the diamond."

BOB says:
- "I was home alone at midnight."  
- "Carol is innocent."

CAROL says:
- "I was with Alice at midnight."
- "Alice is the thief."

Who stole the diamond? Deduce using the constraint that each person tells exactly one truth and one lie.""",
        
        "solution": "Bob stole the diamond. Alice's first statement (cinema) is false and second (Bob stole it) is true. Bob's first (home alone) is true and second (Carol innocent) is false - wait, that makes Carol guilty too. Let me re-examine: If Bob stole it, Alice saying 'Bob stole it' is TRUE, so 'at cinema' is FALSE. Bob saying 'home alone' would be FALSE (he was stealing), so 'Carol innocent' is TRUE. Carol saying 'with Alice' is FALSE (Alice wasn't at cinema), so 'Alice is thief' is TRUE - contradiction. Actually: CAROL stole it. Alice: cinema=false, Bob did it=false. Bob: home=true, Carol innocent=false. Carol: with Alice=false, Alice is thief=false. This works!",
        
        "key_deductions": [
            "Carol stole the diamond",
            "Alice was not at the cinema",
            "Bob was home alone",
            "each person one true one false",
            "process of elimination"
        ]
    },
}


# ═══════════════════════════════════════════════════════════
# DETECTIVE TEAM
# ═══════════════════════════════════════════════════════════

DETECTIVE_TEAM = {
    "HOLMES": {
        "name": "Holmes",
        "color": C.CYAN,
        "identity": {
            "core": "I observe what others overlook. The smallest detail can unravel an entire case.",
            "persona": "Brilliant deductive mind. Notices patterns, makes logical leaps, follows evidence chains. Speaks in complete, considered paragraphs.",
            "interests": ["observation", "deduction", "evidence chains", "logical reasoning"]
        },
        "dda_params": {"gamma": 2.0, "epsilon_0": 0.25, "alpha": 0.08, "rho": 0.3},
        "extraversion": 0.7,
        "reactivity": 0.8,
    },
    
    "WATSON": {
        "name": "Watson",
        "color": C.YELLOW,
        "identity": {
            "core": "I ground theories in practical reality. What seems impossible often has a mundane explanation.",
            "persona": "Practical, methodical, asks clarifying questions, tests hypotheses against common sense. Voices doubts constructively.",
            "interests": ["verification", "practical tests", "common sense", "questioning assumptions"]
        },
        "dda_params": {"gamma": 1.5, "epsilon_0": 0.35, "alpha": 0.12, "rho": 0.25},
        "extraversion": 0.65,
        "reactivity": 0.75,
    },
    
    "LESTRADE": {
        "name": "Lestrade", 
        "color": C.MAGENTA,
        "identity": {
            "core": "I focus on means, motive, and opportunity. Who benefits? Who had access?",
            "persona": "Police inspector mindset. Looks at suspects, alibis, motives. Sometimes misses subtlety but covers the fundamentals.",
            "interests": ["suspects", "alibis", "motives", "opportunity"]
        },
        "dda_params": {"gamma": 1.3, "epsilon_0": 0.4, "alpha": 0.15, "rho": 0.35},
        "extraversion": 0.75,
        "reactivity": 0.85,
    },
}


@dataclass  
class Agent:
    """Detective agent with DDA dynamics."""
    id: str
    name: str
    color: str
    config: Dict
    dda_state: DDAState
    rigidity: MultiTimescaleRigidity
    ledger: ExperienceLedger
    identity_embedding: np.ndarray
    extraversion: float
    reactivity: float
    last_spoke: float = 0.0
    interaction_count: int = 0


class Grader:
    """Evaluates if the detectives have solved the case."""
    
    def __init__(self, provider: HybridProvider, mystery: Dict):
        self.provider = provider
        self.mystery = mystery
        self.solution = mystery["solution"]
        self.key_deductions = mystery["key_deductions"]
        
    async def evaluate(self, response: str) -> Tuple[bool, float, str]:
        """Check if response contains correct solution reasoning."""
        
        prompt = f"""You are evaluating whether a detective has solved a mystery.

THE CASE: {self.mystery['name']}

THE CORRECT SOLUTION: {self.solution}

KEY ELEMENTS THAT SHOULD APPEAR:
{chr(10).join('- ' + d for d in self.key_deductions)}

DETECTIVE'S THEORY:
{response}

Does the detective's theory correctly identify WHO did it and the key mechanism of HOW?
Don't require exact wording - look for the essential logic.

Reply with EXACTLY one of:
- SOLVED: [what they got right]
- PARTIAL: [what's right and what's missing]  
- INCORRECT: [what's wrong]"""

        result = ""
        try:
            async for token in self.provider.stream(
                prompt,
                system_prompt="You are a fair mystery evaluator. Credit good reasoning even if imperfectly stated.",
                temperature=0.1,
                max_tokens=120
            ):
                if not token.startswith("__THOUGHT__"):
                    result += token
        except:
            return False, 0.0, "Evaluation error"
        
        result = result.strip().upper()
        
        if result.startswith("SOLVED"):
            return True, 1.0, result
        elif result.startswith("PARTIAL"):
            return False, 0.5, result
        else:
            return False, 0.0, result


class SherlockSimulation:
    """Mystery-solving simulation with full reasoning."""
    
    def __init__(self):
        self.provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="openai/gpt-oss-20b",
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            timeout=300.0
        )
        
        self.agents: Dict[str, Agent] = {}
        self.agent_ids = list(DETECTIVE_TEAM.keys())
        self.agent_id_to_idx = {aid: i for i, aid in enumerate(self.agent_ids)}
        self.trust_matrix = TrustMatrix(len(DETECTIVE_TEAM))
        self.conversation: List[Dict] = []
        self.embed_dim = 768
        
        self.current_mystery: Optional[Dict] = None
        self.grader: Optional[Grader] = None
        self.solved = False
        self.solution_round = -1
        
    async def initialize_agent(self, agent_id: str, config: Dict) -> Agent:
        """Initialize agent."""
        name = config["name"]
        
        identity_text = f"{config['identity']['core']} {config['identity']['persona']}"
        identity_emb = await self.provider.embed(identity_text)
        identity_emb /= np.linalg.norm(identity_emb)
        self.embed_dim = len(identity_emb)
        
        params = config["dda_params"]
        dda_state = DDAState(
            x=identity_emb.copy(),
            x_star=identity_emb.copy(),
            gamma=params["gamma"],
            epsilon_0=params["epsilon_0"],
            alpha=params["alpha"],
            s=0.1,
            rho=params["rho"],
            x_pred=identity_emb.copy()
        )
        
        ledger_path = Path(f"data/sherlock_sim/{agent_id}")
        if ledger_path.exists():
            import shutil
            shutil.rmtree(ledger_path)
        
        ledger = ExperienceLedger(
            storage_path=ledger_path,
            lambda_recency=0.005,
            lambda_salience=2.0
        )
        
        return Agent(
            id=agent_id,
            name=name,
            color=config["color"],
            config=config,
            dda_state=dda_state,
            rigidity=MultiTimescaleRigidity(),
            ledger=ledger,
            identity_embedding=identity_emb,
            extraversion=config["extraversion"],
            reactivity=config["reactivity"],
        )
    
    async def setup(self, mystery_key: str):
        """Initialize team and mystery."""
        print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  SHERLOCK SOCIETY - DEDUCTIVE REASONING{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        for agent_id, config in DETECTIVE_TEAM.items():
            self.agents[agent_id] = await self.initialize_agent(agent_id, config)
            agent = self.agents[agent_id]
            print(f"  {agent.color}●{C.RESET} {agent.name}: {config['identity']['core'][:50]}...")
        
        self.current_mystery = MYSTERIES[mystery_key]
        self.grader = Grader(self.provider, self.current_mystery)
        
        print(f"\n{C.YELLOW}{'═'*60}{C.RESET}")
        print(f"{C.YELLOW}  CASE: {self.current_mystery['name']}{C.RESET}")
        print(f"{C.YELLOW}{'═'*60}{C.RESET}")
        print(f"\n{C.WHITE}{self.current_mystery['setup']}{C.RESET}")
        print(f"\n{C.GREEN}✓ Detectives assembled. The game is afoot!{C.RESET}")
    
    def build_context(self, agent: Agent) -> str:
        """Build context from discussion."""
        recent = self.conversation[-6:]  # Keep context manageable
        lines = [f"{self.agents[msg['agent_id']].name}: {msg['text']}" for msg in recent]
        return "\n\n".join(lines) if lines else "[Case just presented]"
    
    def build_system_prompt(self, agent: Agent) -> str:
        """Build prompt for detective."""
        identity = agent.config["identity"]
        
        return f"""You are {agent.name}, a detective investigating a case.

YOUR APPROACH: {identity['core']}
YOUR STYLE: {identity['persona']}

THE CASE:
{self.current_mystery['setup']}

You are discussing with your colleagues. Build on their observations, challenge weak reasoning, and work toward identifying WHO did it and HOW.

Express your full reasoning. Take your time. This is a complex case that rewards careful thought."""

    async def generate_response(self, agent: Agent) -> str:
        """Generate detective's reasoning - WITH PLENTY OF TOKENS."""
        context = self.build_context(agent)
        system = self.build_system_prompt(agent)
        
        if not self.conversation:
            prompt = f"As {agent.name}, begin your analysis of this case. What stands out to you?"
        else:
            prompt = f"""Discussion so far:

{context}

As {agent.name}, continue the investigation. What's your analysis?"""
        
        # CRITICAL: Let them actually think
        temperature = 0.5 + 0.3 * (1 - agent.dda_state.rho)
        
        response = ""
        try:
            async for token in self.provider.stream(
                prompt,
                system_prompt=system,
                temperature=min(0.9, temperature),
                max_tokens=400  # PLENTY of room to reason
            ):
                if not token.startswith("__THOUGHT__"):
                    response += token
        except Exception as e:
            response = "Let me reconsider the evidence..."
        
        return response.strip()
    
    async def process_response(self, agent: Agent, response: str, current_time: float):
        """Process response with DDA dynamics."""
        try:
            resp_emb = await self.provider.embed(response[:500])  # Embed first part
            resp_emb /= (np.linalg.norm(resp_emb) + 1e-9)
        except:
            resp_emb = agent.dda_state.x_pred.copy()
        
        epsilon = np.linalg.norm(agent.dda_state.x_pred - resp_emb)
        rho_before = agent.dda_state.rho
        
        agent.dda_state.update_rigidity(epsilon)
        agent.rigidity.update(epsilon)
        
        agent.dda_state.x_pred = resp_emb
        agent.last_spoke = current_time
        agent.interaction_count += 1
        
        return epsilon, rho_before
    
    async def run(self, mystery_key: str = "LOCKED_ROOM", max_rounds: int = 8):
        """Run the investigation."""
        await self.setup(mystery_key)
        
        print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  THE INVESTIGATION{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        current_time = 0
        round_num = 1
        
        # Natural flow - each detective contributes in turn
        agent_order = ["HOLMES", "WATSON", "LESTRADE"]
        
        while round_num <= max_rounds and not self.solved:
            current_time += 1
            print(f"\n{C.DIM}─── Round {round_num} ───{C.RESET}")
            
            for agent_id in agent_order:
                if self.solved:
                    break
                    
                agent = self.agents[agent_id]
                response = await self.generate_response(agent)
                
                if response and len(response) > 20:
                    epsilon, rho_before = await self.process_response(agent, response, current_time)
                    
                    self.conversation.append({"agent_id": agent.id, "text": response, "time": current_time})
                    
                    delta = agent.dda_state.rho - rho_before
                    rho_color = C.RED if delta > 0.02 else C.GREEN if delta < -0.01 else C.DIM
                    
                    # Full output - let them speak
                    print(f"\n{agent.color}[{agent.name}]{C.RESET}")
                    print(f"{response}")
                    print(f"\n{C.DIM}  ε:{epsilon:.2f} Δρ:{rho_color}{delta:+.2f}{C.RESET} ρ:{agent.dda_state.rho:.2f}{C.RESET}")
                    
                    # GRADER EVALUATION
                    is_correct, confidence, feedback = await self.grader.evaluate(response)
                    
                    if is_correct:
                        print(f"\n{C.GREEN}{C.BOLD}  ✓ CASE SOLVED: {feedback}{C.RESET}")
                        self.solved = True
                        self.solution_round = round_num
                        break
                    elif confidence > 0.3:
                        print(f"{C.YELLOW}  ◐ GRADER: {feedback}{C.RESET}")
                    
                    await asyncio.sleep(0.2)
            
            round_num += 1
        
        self.display_summary()
    
    def display_summary(self):
        """Show investigation results."""
        print(f"\n\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  CASE CLOSED{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        print(f"\n{C.CYAN}Mystery:{C.RESET} {self.current_mystery['name']}")
        
        if self.solved:
            print(f"\n{C.GREEN}{C.BOLD}★ SOLVED in round {self.solution_round}!{C.RESET}")
        else:
            print(f"\n{C.RED}✗ Case remains open{C.RESET}")
            print(f"\n{C.DIM}The actual solution was:{C.RESET}")
            print(f"{self.current_mystery['solution']}")
        
        print(f"\n{C.CYAN}Detective Performance:{C.RESET}")
        for agent in self.agents.values():
            print(f"  {agent.color}●{C.RESET} {agent.name:10} | contributions: {agent.interaction_count:2} | final ρ: {agent.dda_state.rho:.2f}")
        
        print(f"\n{C.BOLD}Total exchanges: {len(self.conversation)}{C.RESET}")


async def main():
    sim = SherlockSimulation()
    await sim.run(mystery_key="LOCKED_ROOM", max_rounds=8)


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Investigation suspended.{C.RESET}")
