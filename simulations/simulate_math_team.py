#!/usr/bin/env python3
"""
MATH TEAM SIMULATION - WITH GRADER AGENT
=========================================

A society of agents solve math problems collaboratively.
A GRADER agent evaluates each contribution and determines when
the correct solution has been reached.

Key improvement: The grader uses LLM evaluation against the known
answer, not keyword matching.
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
# MATH PROBLEMS WITH DEFINITE ANSWERS
# ═══════════════════════════════════════════════════════════

MATH_PROBLEMS = {
    "FIND_X": {
        "name": "Solve for X",
        "statement": "Solve: 3(2x + 4) - 5 = 2(x - 1) + 15. What is x?",
        "answer": "x = 1",
        "verification": "3(2*1 + 4) - 5 = 3*6 - 5 = 13. And 2(1-1) + 15 = 0 + 15 = 15. Wait that's wrong. Let me recheck: 3(2+4)-5 = 3*6-5 = 13. 2(1-1)+15 = 15. So x=1 doesn't work. Actually: 6x+12-5 = 2x-2+15, so 6x+7 = 2x+13, 4x=6, x=3/2=1.5",
        "correct_answer": "x = 1.5 or x = 3/2"
    },
    
    "PROBABILITY": {
        "name": "Dice Probability",
        "statement": "You roll two fair 6-sided dice. What is the probability that the sum is 7?",
        "answer": "6/36 = 1/6",
        "verification": "Combinations that sum to 7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) = 6 ways. Total outcomes = 36. P = 6/36 = 1/6.",
        "correct_answer": "1/6 or approximately 0.167 or 16.67%"
    },
    
    "SEQUENCE": {
        "name": "Number Sequence",
        "statement": "What is the next number in this sequence: 2, 6, 12, 20, 30, ?",
        "answer": "42",
        "verification": "Differences: 4, 6, 8, 10 (increasing by 2). Next diff = 12. So 30 + 12 = 42. Pattern: n(n+1) for n=1,2,3,4,5,6 gives 2,6,12,20,30,42.",
        "correct_answer": "42"
    },
    
    "GEOMETRY": {
        "name": "Triangle Area",
        "statement": "A right triangle has legs of length 5 and 12. What is its area?",
        "answer": "30",
        "verification": "Area = (1/2) * base * height = (1/2) * 5 * 12 = 30 square units.",
        "correct_answer": "30 square units"
    },
    
    "WORD_PROBLEM": {
        "name": "Age Problem",
        "statement": "Alice is twice as old as Bob. In 10 years, Alice will be 1.5 times as old as Bob. How old is Bob now?",
        "answer": "20",
        "verification": "Let Bob = x. Alice = 2x. In 10 years: 2x + 10 = 1.5(x + 10). 2x + 10 = 1.5x + 15. 0.5x = 5. x = 10. Wait let me check: Bob=10, Alice=20. In 10 years: Bob=20, Alice=30. 30/20 = 1.5. Yes!",
        "correct_answer": "Bob is 10 years old"
    },
}


# ═══════════════════════════════════════════════════════════
# MATH TEAM - SMALLER, FOCUSED GROUP
# ═══════════════════════════════════════════════════════════

MATH_TEAM = {
    "SOLVER": {
        "name": "Solver",
        "color": C.CYAN,
        "identity": {
            "core": "I work through problems step by step, showing my work clearly.",
            "persona": "Methodical problem solver, writes out equations and steps.",
            "interests": ["algebra", "step-by-step", "equations"]
        },
        "dda_params": {"gamma": 1.8, "epsilon_0": 0.3, "alpha": 0.1, "rho": 0.25},
        "extraversion": 0.7,
        "reactivity": 0.75,
    },
    
    "CHECKER": {
        "name": "Checker",
        "color": C.YELLOW,
        "identity": {
            "core": "I verify calculations and catch errors before they propagate.",
            "persona": "Careful verifier, substitutes back to check, catches mistakes.",
            "interests": ["verification", "double-checking", "errors"]
        },
        "dda_params": {"gamma": 1.5, "epsilon_0": 0.35, "alpha": 0.12, "rho": 0.3},
        "extraversion": 0.6,
        "reactivity": 0.8,
    },
    
    "INTUITIVE": {
        "name": "Intuitive",
        "color": C.MAGENTA,
        "identity": {
            "core": "I estimate and sanity-check before diving into calculation.",
            "persona": "Quick estimator, uses number sense, spots unreasonable answers.",
            "interests": ["estimation", "sanity checks", "quick math"]
        },
        "dda_params": {"gamma": 1.0, "epsilon_0": 0.4, "alpha": 0.15, "rho": 0.2},
        "extraversion": 0.75,
        "reactivity": 0.85,
    },
}


@dataclass
class Agent:
    """Math team member with DDA dynamics."""
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
    """
    The Grader evaluates responses against the known answer.
    Uses LLM to semantically check if the answer is correct.
    """
    
    def __init__(self, provider: HybridProvider, problem: Dict):
        self.provider = provider
        self.problem = problem
        self.correct_answer = problem["correct_answer"]
        self.confirmed_solved = False
        self.solution_confidence = 0.0
        
    async def evaluate(self, response: str) -> Tuple[bool, float, str]:
        """
        Evaluate if response contains the correct answer.
        Returns: (is_correct, confidence, feedback)
        """
        prompt = f"""You are a math grader. Evaluate if this response contains the correct answer.

PROBLEM: {self.problem['statement']}
CORRECT ANSWER: {self.correct_answer}

STUDENT RESPONSE: {response}

Does the student's response contain the correct answer (even if worded differently)?
Reply with EXACTLY one of:
- CORRECT: [brief reason]
- INCORRECT: [brief reason]
- PARTIAL: [brief reason]"""

        result = ""
        try:
            async for token in self.provider.stream(
                prompt,
                system_prompt="You are a strict but fair math grader. Be precise.",
                temperature=0.1,
                max_tokens=60
            ):
                if not token.startswith("__THOUGHT__"):
                    result += token
        except:
            return False, 0.0, "Grading error"
        
        result = result.strip().upper()
        
        if result.startswith("CORRECT"):
            return True, 1.0, result
        elif result.startswith("PARTIAL"):
            return False, 0.5, result
        else:
            return False, 0.0, result


class MathTeamSimulation:
    """
    Math team simulation with grader-based termination.
    """
    
    def __init__(self):
        self.provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="openai/gpt-oss-20b",
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            timeout=300.0
        )
        
        self.agents: Dict[str, Agent] = {}
        self.agent_ids = list(MATH_TEAM.keys())
        self.agent_id_to_idx = {aid: i for i, aid in enumerate(self.agent_ids)}
        self.trust_matrix = TrustMatrix(len(MATH_TEAM))
        self.conversation: List[Dict] = []
        self.embed_dim = 768
        
        self.current_problem: Optional[Dict] = None
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
        
        ledger_path = Path(f"data/math_team_sim/{agent_id}")
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
    
    async def setup(self, problem_key: str):
        """Initialize team and problem."""
        print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  MATH TEAM WITH GRADER{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        for agent_id, config in MATH_TEAM.items():
            self.agents[agent_id] = await self.initialize_agent(agent_id, config)
            agent = self.agents[agent_id]
            print(f"  {agent.color}●{C.RESET} {agent.name}: γ={config['dda_params']['gamma']}, ρ={agent.dda_state.rho:.2f}")
        
        self.current_problem = MATH_PROBLEMS[problem_key]
        self.grader = Grader(self.provider, self.current_problem)
        
        print(f"\n{C.YELLOW}═══ PROBLEM: {self.current_problem['name']} ═══{C.RESET}")
        print(f"{C.WHITE}{self.current_problem['statement']}{C.RESET}")
        print(f"{C.DIM}(Answer hidden from team, known only to Grader){C.RESET}")
        print(f"\n{C.GREEN}✓ Team ready. Grader watching.{C.RESET}")
    
    def build_context(self, agent: Agent) -> str:
        """Build context."""
        recent = self.conversation[-8:]
        lines = [f"{self.agents[msg['agent_id']].name}: {msg['text']}" for msg in recent]
        return "\n".join(lines) if lines else "[Starting fresh]"
    
    def build_system_prompt(self, agent: Agent) -> str:
        """Build prompt."""
        identity = agent.config["identity"]
        return f"""You are {agent.name} on a math team.

PROBLEM: {self.current_problem['statement']}

YOUR ROLE: {identity['core']}
YOUR STYLE: {identity['persona']}

Work with your teammates to solve this. Show your work clearly.
Keep responses focused and mathematical (2-3 sentences max)."""

    async def generate_response(self, agent: Agent) -> str:
        """Generate agent response."""
        context = self.build_context(agent)
        system = self.build_system_prompt(agent)
        
        prompt = f"""Team discussion so far:
{context}

As {agent.name}, contribute your next step toward solving the problem:"""
        
        temperature = 0.3 + 0.4 * (1 - agent.dda_state.rho)
        
        response = ""
        try:
            async for token in self.provider.stream(
                prompt,
                system_prompt=system,
                temperature=min(0.8, temperature),
                max_tokens=100
            ):
                if not token.startswith("__THOUGHT__"):
                    response += token
        except Exception as e:
            response = "Let me reconsider..."
        
        return response.strip()
    
    async def process_response(self, agent: Agent, response: str, current_time: float):
        """Process response with DDA dynamics."""
        try:
            resp_emb = await self.provider.embed(response)
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
    
    async def run(self, problem_key: str = "SEQUENCE", max_rounds: int = 12):
        """Run the math team session."""
        await self.setup(problem_key)
        
        print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  SOLVING SESSION{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        # Intuitive starts with estimation
        starter = self.agents["INTUITIVE"]
        start_msg = f"Looking at {self.current_problem['statement'].split(':')[0]}... Let me get a feel for this first."
        
        self.conversation.append({"agent_id": starter.id, "text": start_msg, "time": 0})
        print(f"\n{starter.color}[{starter.name}]{C.RESET} {start_msg}")
        
        current_time = 0
        round_num = 1
        
        # Round-robin with grader evaluation
        agent_order = ["SOLVER", "CHECKER", "INTUITIVE"]
        
        while round_num <= max_rounds and not self.solved:
            current_time += 1
            
            for agent_id in agent_order:
                if self.solved:
                    break
                    
                agent = self.agents[agent_id]
                response = await self.generate_response(agent)
                
                if response and len(response) > 5:
                    epsilon, rho_before = await self.process_response(agent, response, current_time)
                    
                    self.conversation.append({"agent_id": agent.id, "text": response, "time": current_time})
                    
                    delta = agent.dda_state.rho - rho_before
                    rho_color = C.RED if delta > 0.02 else C.GREEN if delta < -0.01 else C.DIM
                    
                    print(f"\n{agent.color}[{agent.name}]{C.RESET} {response}")
                    print(f"{C.DIM}  ε:{epsilon:.2f} Δρ:{rho_color}{delta:+.2f}{C.RESET} ρ:{agent.dda_state.rho:.2f}{C.RESET}")
                    
                    # GRADER EVALUATION
                    is_correct, confidence, feedback = await self.grader.evaluate(response)
                    
                    if is_correct:
                        print(f"\n{C.GREEN}{C.BOLD}  ✓ GRADER: {feedback}{C.RESET}")
                        self.solved = True
                        self.solution_round = round_num
                        break
                    elif confidence > 0.3:
                        print(f"{C.YELLOW}  ◐ GRADER: {feedback}{C.RESET}")
                    
                    await asyncio.sleep(0.1)
            
            round_num += 1
        
        self.display_summary()
    
    def display_summary(self):
        """Show results."""
        print(f"\n\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  SESSION RESULTS{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        print(f"\n{C.CYAN}Problem:{C.RESET} {self.current_problem['name']}")
        print(f"{C.CYAN}Correct Answer:{C.RESET} {self.current_problem['correct_answer']}")
        
        if self.solved:
            print(f"\n{C.GREEN}{C.BOLD}★ SOLVED in round {self.solution_round}!{C.RESET}")
        else:
            print(f"\n{C.RED}✗ Not solved within round limit{C.RESET}")
        
        print(f"\n{C.CYAN}Team Performance:{C.RESET}")
        for agent in self.agents.values():
            print(f"  {agent.color}●{C.RESET} {agent.name:10} | msgs: {agent.interaction_count:2} | final ρ: {agent.dda_state.rho:.2f}")
        
        print(f"\n{C.BOLD}Total exchanges: {len(self.conversation)}{C.RESET}")


async def main():
    sim = MathTeamSimulation()
    
    # Try the sequence problem - clear numeric answer
    await sim.run(problem_key="SEQUENCE", max_rounds=10)


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Session ended.{C.RESET}")
