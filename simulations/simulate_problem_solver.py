#!/usr/bin/env python3
"""
COLLABORATIVE PROBLEM SOLVING SOCIETY
======================================

A society of 6 agents with distinct cognitive styles work together
to solve a difficult logic/math problem through natural discourse.

Instead of casual chat, agents:
- Propose approaches and hypotheses
- Build on each other's reasoning
- Challenge flawed logic
- Converge toward solutions

DDA dynamics model:
- High surprise (ε) when encountering novel insights → rigidity drops → more exploration
- Low surprise when seeing expected reasoning → rigidity stable
- Trust builds with agents who provide valuable contributions
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
# PROBLEM DEFINITIONS - CLASSIC LOGIC PUZZLES
# ═══════════════════════════════════════════════════════════

PROBLEMS = {
    "RIVER_CROSSING": {
        "name": "River Crossing Puzzle",
        "statement": """A farmer needs to cross a river with a wolf, a goat, and a cabbage. 
The boat can only carry the farmer and one item at a time.
If left alone together: the wolf will eat the goat, and the goat will eat the cabbage.
How can the farmer get all three across safely?""",
        "solution_keywords": ["goat first", "return", "wolf or cabbage", "goat back", "other item", "goat last"],
        "difficulty": "medium"
    },
    
    "PRISONERS_HATS": {
        "name": "Prisoners and Hats",
        "statement": """Three prisoners stand in a line. Each wears a hat (either red or blue) that they cannot see.
There are 2 red hats and 3 blue hats available. Each prisoner can see the hats ahead of them (back sees middle and front, middle sees front).
Starting from the back, each prisoner must guess their hat color or say "I don't know."
If anyone is wrong, all are executed. If anyone is right, all go free.
The back prisoner says "I don't know." The middle prisoner says "I don't know."
The front prisoner then knows their hat color with certainty. How?""",
        "solution_keywords": ["both red", "back would know", "not both red", "middle reasoning", "deduction", "blue"],
        "difficulty": "hard"
    },
    
    "MONTY_HALL": {
        "name": "Monty Hall Problem", 
        "statement": """You're on a game show with 3 doors. Behind one is a car; behind the others, goats.
You pick door 1. The host (who knows what's behind all doors) opens door 3, revealing a goat.
The host offers: "Do you want to switch to door 2?"
Should you switch, stay, or does it not matter? Prove your answer with probability.""",
        "solution_keywords": ["switch", "2/3", "1/3", "conditional", "host reveals", "not random"],
        "difficulty": "hard"
    },
    
    "FIBONACCI_SUM": {
        "name": "Fibonacci Sum Pattern",
        "statement": """Prove that the sum of the first n Fibonacci numbers equals F(n+2) - 1.
Where F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, etc.
Verify for n=5: F(1)+F(2)+F(3)+F(4)+F(5) should equal F(7)-1.
Then provide a general proof or inductive argument.""",
        "solution_keywords": ["induction", "1+1+2+3+5=12", "F(7)=13", "13-1=12", "recurrence", "base case"],
        "difficulty": "medium"
    },
    
    "KNIGHTS_KNAVES": {
        "name": "Knights and Knaves",
        "statement": """On an island, knights always tell the truth and knaves always lie.
You meet three people: A, B, and C.
A says: "All of us are knaves."
B says: "Exactly one of us is a knight."
What are A, B, and C?""",
        "solution_keywords": ["A is knave", "if knight", "contradiction", "B knight", "C knave", "exactly one"],
        "difficulty": "hard"
    },
}


# ═══════════════════════════════════════════════════════════
# SOLVER SOCIETY - COGNITIVE ARCHETYPES
# ═══════════════════════════════════════════════════════════

SOLVER_SOCIETY = {
    "LOGICIAN": {
        "name": "Logician",
        "color": C.CYAN,
        "identity": {
            "core": "I break problems into formal logical steps. Every claim must follow from premises.",
            "persona": "Methodical, precise, uses symbolic notation, identifies logical fallacies, builds proof structures.",
            "interests": ["formal logic", "proof theory", "contradiction", "modus ponens"]
        },
        "dda_params": {
            "gamma": 2.0,      # Strong identity - formal methods
            "epsilon_0": 0.25, # High threshold - only surprised by genuine insights
            "alpha": 0.08,     # Slow to change approach
            "rho": 0.3,        # Starts somewhat open
        },
        "extraversion": 0.6,
        "reactivity": 0.7,
    },
    
    "INTUITOR": {
        "name": "Intuitor", 
        "color": C.MAGENTA,
        "identity": {
            "core": "I sense patterns before I can prove them. The answer often comes first, proof second.",
            "persona": "Leaps to conclusions, pattern-matcher, sometimes vague but often right, trusts gut feelings.",
            "interests": ["patterns", "analogies", "insight", "aha moments"]
        },
        "dda_params": {
            "gamma": 1.0,      # Flexible identity
            "epsilon_0": 0.35,
            "alpha": 0.15,     # Quick to adapt
            "rho": 0.2,        # Very open
        },
        "extraversion": 0.8,
        "reactivity": 0.85,
    },
    
    "SKEPTIC": {
        "name": "Skeptic",
        "color": C.RED,
        "identity": {
            "core": "I find holes in reasoning. If an argument can be attacked, it should be.",
            "persona": "Critical, plays devil's advocate, asks 'but what if...', demands rigor.",
            "interests": ["counterexamples", "edge cases", "assumptions", "falsification"]
        },
        "dda_params": {
            "gamma": 1.8,
            "epsilon_0": 0.3,
            "alpha": 0.1,
            "rho": 0.5,        # Somewhat guarded
        },
        "extraversion": 0.7,
        "reactivity": 0.9,     # Jumps on flawed reasoning
    },
    
    "CALCULATOR": {
        "name": "Calculator",
        "color": C.GREEN,
        "identity": {
            "core": "I work with numbers and concrete examples. Abstract? Show me the computation.",
            "persona": "Works through examples, verifies numerically, grounds abstract claims in arithmetic.",
            "interests": ["computation", "examples", "verification", "probability"]
        },
        "dda_params": {
            "gamma": 1.5,
            "epsilon_0": 0.35,
            "alpha": 0.12,
            "rho": 0.25,
        },
        "extraversion": 0.5,
        "reactivity": 0.6,
    },
    
    "SYNTHESIZER": {
        "name": "Synthesizer",
        "color": C.YELLOW,
        "identity": {
            "core": "I connect ideas from different contributors. The solution often lies between viewpoints.",
            "persona": "Builds consensus, notices complementary ideas, weaves threads together.",
            "interests": ["integration", "synthesis", "consensus", "combining approaches"]
        },
        "dda_params": {
            "gamma": 1.2,
            "epsilon_0": 0.4,
            "alpha": 0.18,
            "rho": 0.15,       # Very open to input
        },
        "extraversion": 0.7,
        "reactivity": 0.75,
    },
    
    "VISUALIZER": {
        "name": "Visualizer",
        "color": C.BLUE,
        "identity": {
            "core": "I think in diagrams and spatial relationships. Can we draw this?",
            "persona": "Describes visual representations, uses spatial metaphors, draws decision trees.",
            "interests": ["diagrams", "trees", "spatial reasoning", "visualization"]
        },
        "dda_params": {
            "gamma": 1.3,
            "epsilon_0": 0.38,
            "alpha": 0.14,
            "rho": 0.2,
        },
        "extraversion": 0.55,
        "reactivity": 0.65,
    },
}


@dataclass
class Agent:
    """Problem-solving agent with DDA dynamics."""
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
    contributions: List[str] = field(default_factory=list)  # Track valuable insights


class ProblemSolverSimulation:
    """
    Simulates collaborative problem solving.
    
    Key differences from chat simulation:
    - Agents respond to the PROBLEM not just each other
    - Progress tracking toward solution
    - Contributions rated for insight value
    - Termination when solution is found (or round limit)
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
        self.agent_ids = list(SOLVER_SOCIETY.keys())
        self.agent_id_to_idx = {aid: i for i, aid in enumerate(self.agent_ids)}
        self.trust_matrix = TrustMatrix(len(SOLVER_SOCIETY))
        self.conversation: List[Dict] = []
        self.embed_dim = 768
        
        # Problem state
        self.current_problem: Optional[Dict] = None
        self.problem_embedding: Optional[np.ndarray] = None
        self.solution_progress: float = 0.0  # 0-1 progress toward solution
        self.key_insights: List[str] = []
        
    async def initialize_agent(self, agent_id: str, config: Dict) -> Agent:
        """Initialize one agent with full DDA setup."""
        name = config["name"]
        
        # Create identity embedding
        identity_text = f"{config['identity']['core']} {config['identity']['persona']} {' '.join(config['identity']['interests'])}"
        identity_emb = await self.provider.embed(identity_text)
        identity_emb /= np.linalg.norm(identity_emb)
        self.embed_dim = len(identity_emb)
        
        # Initialize DDA state
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
        
        # Create ledger
        ledger_path = Path(f"data/problem_solver_sim/{agent_id}")
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
    
    async def setup(self, problem_key: str = "KNIGHTS_KNAVES"):
        """Initialize all agents and load the problem."""
        print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  COLLABORATIVE PROBLEM SOLVING SOCIETY{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        # Initialize agents
        for agent_id, config in SOLVER_SOCIETY.items():
            self.agents[agent_id] = await self.initialize_agent(agent_id, config)
            agent = self.agents[agent_id]
            print(f"  {agent.color}●{C.RESET} {agent.name:12}: γ={config['dda_params']['gamma']}, ρ={agent.dda_state.rho:.2f}")
        
        # Load problem
        self.current_problem = PROBLEMS[problem_key]
        self.problem_embedding = await self.provider.embed(self.current_problem["statement"])
        self.problem_embedding /= np.linalg.norm(self.problem_embedding)
        
        print(f"\n{C.YELLOW}═══ PROBLEM: {self.current_problem['name']} ═══{C.RESET}")
        print(f"{C.WHITE}{self.current_problem['statement']}{C.RESET}")
        print(f"{C.DIM}Difficulty: {self.current_problem['difficulty']}{C.RESET}")
        print(f"\n{C.GREEN}✓ {len(self.agents)} agents ready to collaborate{C.RESET}")
    
    def calculate_response_probability(
        self,
        agent: Agent,
        message_emb: np.ndarray,
        speaker_id: str,
        current_time: float
    ) -> float:
        """
        Calculate probability of responding.
        
        For problem solving:
        - Higher probability if message relates to agent's cognitive style
        - Higher probability if agent has been quiet but has something to add
        - Trust with speaker affects engagement
        """
        # Topic relevance
        relevance = np.dot(message_emb, agent.identity_embedding)
        relevance = max(0.1, relevance)  # Floor at 0.1 - everyone might contribute
        
        # Problem relevance - boost if message is about the problem
        problem_relevance = np.dot(message_emb, self.problem_embedding) if self.problem_embedding is not None else 0.5
        problem_boost = 1.0 + max(0, problem_relevance) * 0.5
        
        # Extraversion base
        base_prob = agent.extraversion
        
        # Rigidity penalty
        rho_factor = 1 - (agent.dda_state.rho * 0.4)
        
        # Cooldown
        time_since = current_time - agent.last_spoke
        cooldown_factor = min(1.0, time_since / 2.5)
        
        # Trust dynamics
        observer_idx = self.agent_id_to_idx[agent.id]
        speaker_idx = self.agent_id_to_idx[speaker_id]
        trust = self.trust_matrix.get_trust(observer_idx, speaker_idx)
        # High trust = build on their work; low trust = challenge them
        trust_factor = 0.8 + trust * 0.4
        
        prob = base_prob * relevance * agent.reactivity * rho_factor * cooldown_factor * trust_factor * problem_boost
        return min(0.95, max(0.05, prob))
    
    async def select_responders(self, message: Dict, current_time: float) -> List[str]:
        """Select who responds to a message."""
        speaker_id = message["agent_id"]
        
        try:
            msg_emb = await self.provider.embed(message["text"])
            msg_emb /= (np.linalg.norm(msg_emb) + 1e-9)
        except:
            msg_emb = np.random.randn(self.embed_dim)
            msg_emb /= np.linalg.norm(msg_emb)
        
        responders = []
        probs = {}
        
        for agent_id, agent in self.agents.items():
            if agent_id == speaker_id:
                continue
            
            prob = self.calculate_response_probability(agent, msg_emb, speaker_id, current_time)
            probs[agent_id] = prob
            
            if random.random() < prob:
                responders.append(agent_id)
        
        # Sort by probability
        responders.sort(key=lambda x: probs[x], reverse=True)
        
        # Limit to 2 responders for focused discourse
        return responders[:2]
    
    def build_context(self, agent: Agent) -> str:
        """Build context from recent reasoning."""
        recent = self.conversation[-12:]
        
        lines = []
        for msg in recent:
            speaker = self.agents[msg["agent_id"]].name
            lines.append(f"{speaker}: {msg['text']}")
        
        return "\n".join(lines) if lines else "[Discussion just started]"
    
    def build_system_prompt(self, agent: Agent) -> str:
        """Build system prompt for problem-solving agent."""
        identity = agent.config["identity"]
        problem = self.current_problem
        
        if agent.dda_state.rho < 0.25:
            mode = "exploratory, open to new approaches"
        elif agent.dda_state.rho < 0.5:
            mode = "focused, building on established reasoning"
        else:
            mode = "converging, defending current approach"
        
        progress_desc = ""
        if self.key_insights:
            progress_desc = f"\nKey insights established: {'; '.join(self.key_insights[-3:])}"
        
        return f"""You are {agent.name}, a problem-solving specialist.

THE PROBLEM: {problem['name']}
{problem['statement']}

YOUR COGNITIVE STYLE: {identity['core']}
YOUR APPROACH: {identity['persona']}
SPECIALTIES: {', '.join(identity['interests'])}

CURRENT STATE: {mode} (ρ={agent.dda_state.rho:.2f})
{progress_desc}

You are collaborating with other thinkers to solve this. Your task:
- Contribute reasoning that fits your cognitive style
- Build on others' insights when valid
- Challenge flawed reasoning constructively  
- Move toward the solution step by step

Respond with 1-3 sentences of substantive reasoning. No meta-commentary about the process."""

    async def generate_response(self, agent: Agent, trigger_msg: Dict) -> str:
        """Generate agent's contribution to the solution."""
        context = self.build_context(agent)
        system = self.build_system_prompt(agent)
        
        trigger_text = trigger_msg["text"]
        trigger_name = self.agents[trigger_msg["agent_id"]].name
        
        prompt = f"""Discussion so far:
{context}

{trigger_name} just said: "{trigger_text}"

As {agent.name}, contribute your next piece of reasoning:"""
        
        # Map cognitive state to generation params
        temperature = 0.4 + 0.4 * (1 - agent.dda_state.rho)
        
        response = ""
        try:
            async for token in self.provider.stream(
                prompt,
                system_prompt=system,
                temperature=min(0.9, temperature),
                max_tokens=120
            ):
                if not token.startswith("__THOUGHT__"):
                    response += token
        except Exception as e:
            print(f"{C.RED}[ERROR] {agent.name}: {e}{C.RESET}")
            response = "Let me think about this more carefully..."
        
        return response.strip()
    
    def check_solution_progress(self, response: str) -> float:
        """Check if response contains solution keywords."""
        if not self.current_problem:
            return 0.0
        
        keywords = self.current_problem["solution_keywords"]
        response_lower = response.lower()
        
        matches = sum(1 for kw in keywords if kw.lower() in response_lower)
        return matches / len(keywords)
    
    async def process_response(self, agent: Agent, response: str, trigger_msg: Dict, current_time: float):
        """Process agent's response - update dynamics and track progress."""
        # Embed response
        try:
            resp_emb = await self.provider.embed(response)
            resp_emb /= (np.linalg.norm(resp_emb) + 1e-9)
        except:
            resp_emb = agent.dda_state.x_pred.copy()
        
        # Compute surprise
        epsilon = np.linalg.norm(agent.dda_state.x_pred - resp_emb)
        
        # Check for solution progress
        progress = self.check_solution_progress(response)
        if progress > 0.15:  # Meaningful contribution
            agent.contributions.append(response[:80])
            self.solution_progress = max(self.solution_progress, progress)
            if progress > 0.3:  # Key insight
                self.key_insights.append(f"{agent.name}: {response[:60]}...")
        
        # Update rigidity
        rho_before = agent.dda_state.rho
        
        # Novel insights reduce rigidity (exploration mode)
        if progress > 0.2:
            epsilon *= 0.7  # Reduce surprise penalty for valuable contributions
        
        agent.dda_state.update_rigidity(epsilon)
        agent.rigidity.update(epsilon)
        
        # Update trust
        speaker_id = trigger_msg["agent_id"]
        observer_idx = self.agent_id_to_idx[agent.id]
        speaker_idx = self.agent_id_to_idx[speaker_id]
        self.trust_matrix.update_trust(observer_idx, speaker_idx, epsilon)
        
        # Ledger entry
        trigger_emb = await self.provider.embed(trigger_msg["text"])
        trigger_emb /= (np.linalg.norm(trigger_emb) + 1e-9)
        
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=agent.dda_state.x.copy(),
            action_id=f"reasoning_{agent.interaction_count}",
            observation_embedding=trigger_emb,
            outcome_embedding=resp_emb,
            prediction_error=epsilon,
            context_embedding=self.problem_embedding if self.problem_embedding is not None else trigger_emb,
            rigidity_at_time=agent.dda_state.rho,
            metadata={
                "type": "problem_solving",
                "progress": progress,
                "contribution": response[:80]
            }
        )
        agent.ledger.add_entry(entry)
        
        # Update state
        agent.dda_state.x_pred = resp_emb
        agent.dda_state.x = (1 - 0.1) * agent.dda_state.x + 0.1 * resp_emb
        agent.last_spoke = current_time
        agent.interaction_count += 1
        
        return epsilon, rho_before, progress
    
    async def run(self, problem_key: str = "KNIGHTS_KNAVES", max_rounds: int = 25):
        """Run the collaborative problem-solving session."""
        await self.setup(problem_key)
        
        print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  COLLABORATIVE REASONING SESSION{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        # Seed with a random agent presenting the problem
        seed_agent = self.agents["LOGICIAN"]  # Logician starts by framing
        seed_text = f"Let's work through this systematically. We have three people A, B, and C, and we need to determine who is a knight (truth-teller) and who is a knave (liar)."
        
        if problem_key == "MONTY_HALL":
            seed_text = "Interesting probability puzzle. Let me frame the initial setup: you choose door 1, host reveals a goat behind door 3. The question is whether switching changes your odds."
        elif problem_key == "RIVER_CROSSING":
            seed_text = "Classic constraint satisfaction problem. We need to track state across trips. The farmer can only take one item per trip, and certain pairs cannot be left alone."
        elif problem_key == "FIBONACCI_SUM":
            seed_text = "Let's verify this identity. We need to show that summing the first n Fibonacci numbers gives F(n+2) - 1. Let me start with a concrete example."
        elif problem_key == "PRISONERS_HATS":
            seed_text = "This is about nested reasoning and information theory. Each prisoner's statement gives us constraints. The back prisoner sees two hats..."
        
        seed_msg = {"agent_id": seed_agent.id, "text": seed_text, "time": 0}
        self.conversation.append(seed_msg)
        
        print(f"\n{seed_agent.color}[{seed_agent.name}]{C.RESET} {seed_text}")
        
        msg_count = 1
        current_time = 0
        
        while msg_count < max_rounds:
            current_time += 1
            last_msg = self.conversation[-1]
            
            # Select responders
            responders = await self.select_responders(last_msg, current_time)
            
            if not responders:
                # Encourage Skeptic or Intuitor if no one speaks
                available = [a for a in self.agents.values() if a.id != last_msg["agent_id"]]
                if available:
                    responders = [random.choice(available).id]
            
            for responder_id in responders:
                if msg_count >= max_rounds:
                    break
                
                agent = self.agents[responder_id]
                response = await self.generate_response(agent, last_msg)
                
                if response and len(response) > 10:
                    epsilon, rho_before, progress = await self.process_response(agent, response, last_msg, current_time)
                    
                    new_msg = {"agent_id": agent.id, "text": response, "time": current_time}
                    self.conversation.append(new_msg)
                    
                    # Display
                    delta = agent.dda_state.rho - rho_before
                    rho_color = C.RED if delta > 0.02 else C.GREEN if delta < -0.01 else C.DIM
                    progress_indicator = f" {C.GREEN}★{C.RESET}" if progress > 0.2 else ""
                    
                    print(f"\n{agent.color}[{agent.name}]{C.RESET} {response}{progress_indicator}")
                    print(f"{C.DIM}  ε:{epsilon:.2f} Δρ:{rho_color}{delta:+.2f}{C.RESET} ρ:{agent.dda_state.rho:.2f} progress:{progress:.0%}{C.RESET}")
                    
                    msg_count += 1
                    last_msg = new_msg
                    
                    # Check for solution
                    if self.solution_progress > 0.7:
                        print(f"\n{C.GREEN}{C.BOLD}━━━ SOLUTION CONVERGENCE DETECTED ━━━{C.RESET}")
                        break
                    
                    await asyncio.sleep(0.1)
            
            if self.solution_progress > 0.7:
                break
        
        self.display_summary()
    
    def display_summary(self):
        """Show final state and contributions."""
        print(f"\n\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  SESSION SUMMARY{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        print(f"\n{C.CYAN}Problem:{C.RESET} {self.current_problem['name']}")
        print(f"{C.CYAN}Solution Progress:{C.RESET} {self.solution_progress:.0%}")
        
        print(f"\n{C.CYAN}Key Insights:{C.RESET}")
        for insight in self.key_insights:
            print(f"  ★ {insight}")
        
        print(f"\n{C.CYAN}Agent Contributions:{C.RESET}")
        agents_sorted = sorted(self.agents.values(), key=lambda a: len(a.contributions), reverse=True)
        for agent in agents_sorted:
            stats = agent.ledger.get_statistics()
            diag = agent.rigidity.get_diagnostic()
            
            print(f"  {agent.color}●{C.RESET} {agent.name:12} | msgs: {agent.interaction_count:2} | ρ: {agent.dda_state.rho:.2f} | contributions: {len(agent.contributions)}")
        
        print(f"\n{C.CYAN}Trust Network:{C.RESET}")
        trust_stats = self.trust_matrix.get_network_stats()
        print(f"  Mean trust: {trust_stats.get('mean_trust', 0):.3f}")
        
        print(f"\n{C.BOLD}Total exchanges: {len(self.conversation)}{C.RESET}")


async def main():
    sim = ProblemSolverSimulation()
    
    # Choose problem
    problem = "KNIGHTS_KNAVES"  # Can change to any key from PROBLEMS
    
    await sim.run(problem_key=problem, max_rounds=30)


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Session ended.{C.RESET}")
