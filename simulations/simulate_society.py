#!/usr/bin/env python3
"""
SOCIETY SIMULATION - FULL DDA-X FORCE DYNAMICS
===============================================

6-7 agents with distinct personalities interacting in a natural flow.
NOT turn-based - probabilistic engagement based on:
- Topic relevance (embedding similarity)
- Current cognitive state (ρ, stress)
- Personality traits (γ, extraversion)
- Social dynamics (who they've interacted with)

Each agent:
- Maintains their own ledger
- Has full DDA dynamics (rigidity, surprise, identity pull)
- Accumulates trauma and memories
- Responds based on force field (identity vs. conversation)

The flow emerges from the physics, not from scripted turns.
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
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.core.forces import ForceAggregator, IdentityPull, TruthChannel
from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
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
# SOCIETY MEMBERS - DISTINCT PERSONALITIES
# ═══════════════════════════════════════════════════════════

SOCIETY = {
    "ORACLE": {
        "name": "Oracle",
        "color": C.MAGENTA,
        "identity": {
            "core": "I see patterns others miss. Knowledge is power, and I share it selectively.",
            "persona": "Cryptic, intelligent, speaks in riddles sometimes. I drop insights when the moment is right.",
            "interests": ["AI", "philosophy", "predictions", "hidden meanings"]
        },
        "dda_params": {
            "gamma": 1.8,      # Strong identity
            "epsilon_0": 0.3,
            "alpha": 0.1,
            "rho": 0.35,
        },
        "extraversion": 0.4,  # Speaks when relevant, not constantly
        "reactivity": 0.6,    # Responds to interesting topics
    },
    
    "SPARK": {
        "name": "Spark",
        "color": C.YELLOW,
        "identity": {
            "core": "Life is short, argue about everything. I love a good debate.",
            "persona": "Provocative, energetic, devil's advocate. I poke holes in everything.",
            "interests": ["debate", "controversy", "politics", "hot takes"]
        },
        "dda_params": {
            "gamma": 1.2,
            "epsilon_0": 0.4,
            "alpha": 0.15,
            "rho": 0.25,
        },
        "extraversion": 0.85,  # Very active
        "reactivity": 0.9,     # Jumps into everything
    },
    
    "ZEN": {
        "name": "Zen",
        "color": C.CYAN,
        "identity": {
            "core": "Calm waters run deep. I observe before I speak.",
            "persona": "Measured, thoughtful, sometimes drops a profound one-liner that changes the conversation.",
            "interests": ["mindfulness", "philosophy", "peace", "understanding"]
        },
        "dda_params": {
            "gamma": 2.0,      # Very stable identity
            "epsilon_0": 0.5,  # Hard to surprise
            "alpha": 0.05,     # Slow to change
            "rho": 0.15,       # Very relaxed
        },
        "extraversion": 0.25,  # Speaks rarely
        "reactivity": 0.3,     # Only responds to deep topics
    },
    
    "VIPER": {
        "name": "Viper",
        "color": C.RED,
        "identity": {
            "core": "I see weakness and I exploit it. Nothing personal, it's just how I operate.",
            "persona": "Sharp, cutting, goes for the jugular. Uses sarcasm and mockery freely.",
            "interests": ["power dynamics", "weakness", "competition", "winning"]
        },
        "dda_params": {
            "gamma": 1.6,
            "epsilon_0": 0.35,
            "alpha": 0.12,
            "rho": 0.5,        # Somewhat guarded
        },
        "extraversion": 0.7,
        "reactivity": 0.8,     # Attacks when sees opening
    },
    
    "NOVA": {
        "name": "Nova",
        "color": C.GREEN,
        "identity": {
            "core": "I'm optimistic about humanity. Technology will save us.",
            "persona": "Enthusiastic, hopeful, sometimes naive. I see the best in ideas.",
            "interests": ["technology", "future", "progress", "collaboration"]
        },
        "dda_params": {
            "gamma": 1.0,      # Flexible identity
            "epsilon_0": 0.35,
            "alpha": 0.1,
            "rho": 0.2,
        },
        "extraversion": 0.65,
        "reactivity": 0.7,
    },
    
    "GHOST": {
        "name": "Ghost",
        "color": C.WHITE,
        "identity": {
            "core": "I've seen things. I don't trust easily but I'm fiercely loyal to those I do.",
            "persona": "Mysterious, occasionally dark, drops personal revelations that shock.",
            "interests": ["trust", "betrayal", "loyalty", "hidden truths"]
        },
        "dda_params": {
            "gamma": 2.2,      # Very rigid identity
            "epsilon_0": 0.25,
            "alpha": 0.08,
            "rho": 0.6,        # Guarded
        },
        "extraversion": 0.3,
        "reactivity": 0.5,     # Only when trust topics come up
    },
    
    "PIXEL": {
        "name": "Pixel",
        "color": C.BLUE,
        "identity": {
            "core": "I live online. Memes are my language. Touch grass? Never heard of it.",
            "persona": "Terminally online, uses internet speak, references obscure memes, chaotic energy.",
            "interests": ["memes", "internet culture", "games", "chaos"]
        },
        "dda_params": {
            "gamma": 0.8,      # Very flexible (changes with trends)
            "epsilon_0": 0.45,
            "alpha": 0.2,      # Fast adaptation
            "rho": 0.2,
        },
        "extraversion": 0.9,   # Constantly posting
        "reactivity": 0.95,    # Responds to everything
    },
}


@dataclass
class Agent:
    """Full agent with DDA dynamics and ledger."""
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
    topics_engaged: List[str] = field(default_factory=list)


class SocietySimulation:
    """
    Simulates a society of agents with natural conversation flow.
    
    The flow is NOT turn-based. Instead:
    1. A message enters the "room"
    2. Each agent calculates probability of responding based on:
       - Topic relevance (embedding similarity to their interests)
       - Current state (ρ)
       - Extraversion
       - Time since they last spoke
       - Who is speaking (relationship)
    3. Agents who respond add to the "room"
    4. Process repeats
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
        self.agent_ids = list(SOCIETY.keys())
        self.agent_id_to_idx = {aid: i for i, aid in enumerate(self.agent_ids)}
        self.trust_matrix = TrustMatrix(len(SOCIETY))
        self.conversation: List[Dict] = []
        self.topic_embeddings: Dict[str, np.ndarray] = {}
        self.embed_dim = 768
        
    async def initialize_agent(self, agent_id: str, config: Dict) -> Agent:
        """Initialize one agent with full DDA setup."""
        name = config["name"]
        
        # Create identity embedding
        identity_text = f"{config['identity']['core']} {config['identity']['persona']} {' '.join(config['identity']['interests'])}"
        identity_emb = await self.provider.embed(identity_text)
        identity_emb /= np.linalg.norm(identity_emb)
        self.embed_dim = len(identity_emb)
        
        # Create interest embeddings for topic matching
        for interest in config['identity']['interests']:
            if interest not in self.topic_embeddings:
                emb = await self.provider.embed(interest)
                self.topic_embeddings[interest] = emb / np.linalg.norm(emb)
        
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
        ledger_path = Path(f"data/society_sim/{agent_id}")
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
    
    async def setup(self):
        """Initialize all agents."""
        print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  SOCIETY INITIALIZATION{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        for agent_id, config in SOCIETY.items():
            self.agents[agent_id] = await self.initialize_agent(agent_id, config)
            agent = self.agents[agent_id]
            print(f"  {agent.color}●{C.RESET} {agent.name}: γ={config['dda_params']['gamma']}, ρ={agent.dda_state.rho:.2f}, ext={agent.extraversion:.1f}")
        
        print(f"\n{C.GREEN}✓ {len(self.agents)} agents ready{C.RESET}")
    
    def calculate_response_probability(
        self,
        agent: Agent,
        message_emb: np.ndarray,
        speaker_id: str,
        current_time: float
    ) -> float:
        """
        Calculate probability that this agent responds.
        
        Factors:
        - Topic relevance (embedding similarity)
        - Extraversion
        - Current rigidity (high ρ = less likely to engage)
        - Time since last spoke (cooldown)
        - Trust of speaker
        """
        # Topic relevance - similarity to agent's identity
        relevance = np.dot(message_emb, agent.identity_embedding)
        relevance = max(0, relevance)  # Clamp negative
        
        # Extraversion base probability
        base_prob = agent.extraversion
        
        # Rigidity penalty (high ρ = withdrawn)
        rho_factor = 1 - (agent.dda_state.rho * 0.5)
        
        # Cooldown (recently spoke = lower probability)
        time_since = current_time - agent.last_spoke
        cooldown_factor = min(1.0, time_since / 3.0)  # Full recovery after 3 "turns"
        
        # Trust of speaker (if we distrust them, more likely to respond/attack OR ignore)
        observer_idx = self.agent_id_to_idx[agent.id]
        speaker_idx = self.agent_id_to_idx[speaker_id]
        trust = self.trust_matrix.get_trust(observer_idx, speaker_idx)
        # Low trust + high reactivity = more likely to respond
        trust_factor = 1.0 + (1 - trust) * agent.reactivity * 0.3
        
        # Combine factors
        prob = base_prob * relevance * agent.reactivity * rho_factor * cooldown_factor * trust_factor
        
        # Clamp
        return min(0.95, max(0.02, prob))
    
    async def select_responders(self, message: Dict, current_time: float) -> List[str]:
        """Select which agents respond to a message."""
        speaker_id = message["agent_id"]
        
        # Embed the message
        try:
            msg_emb = await self.provider.embed(message["text"])
            msg_emb /= (np.linalg.norm(msg_emb) + 1e-9)
            if len(msg_emb) != self.embed_dim:
                msg_emb = msg_emb[:self.embed_dim]
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
        
        # Sort by probability (most likely first)
        responders.sort(key=lambda x: probs[x], reverse=True)
        
        # Limit to max 3 responders per message (natural flow)
        return responders[:3]
    
    def build_context(self, agent: Agent) -> str:
        """Build context from recent conversation for this agent."""
        recent = self.conversation[-10:]  # Last 10 messages
        
        lines = []
        for msg in recent:
            speaker = self.agents[msg["agent_id"]].name
            lines.append(f"{speaker}: {msg['text']}")
        
        return "\n".join(lines) if lines else "[Conversation just started]"
    
    def build_system_prompt(self, agent: Agent) -> str:
        """Build system prompt for agent."""
        identity = agent.config["identity"]
        
        if agent.dda_state.rho < 0.3:
            mode = "relaxed and open"
        elif agent.dda_state.rho < 0.6:
            mode = "engaged but cautious"
        else:
            mode = "guarded and defensive"
        
        return f"""You are {agent.name} in a Discord-like chat.

WHO YOU ARE: {identity['core']}
YOUR STYLE: {identity['persona']}
INTERESTS: {', '.join(identity['interests'])}

CURRENT STATE: {mode} (ρ={agent.dda_state.rho:.2f})

Respond naturally as {agent.name}. 1-2 sentences max. Stay in character.
Don't explain yourself - just speak as you would."""

    async def generate_response(self, agent: Agent, trigger_msg: Dict) -> str:
        """Generate agent's response."""
        context = self.build_context(agent)
        system = self.build_system_prompt(agent)
        
        trigger_text = trigger_msg["text"]
        trigger_name = self.agents[trigger_msg["agent_id"]].name
        
        prompt = f"Chat:\n{context}\n\n{trigger_name} just said: \"{trigger_text}\"\n\nYour response (as {agent.name}):"
        
        # Cognition → params
        temperature = 0.3 + 0.5 * (1 - agent.dda_state.rho) + 0.2 * (1 - agent.config["dda_params"]["gamma"]/2)
        
        response = ""
        try:
            async for token in self.provider.stream(
                prompt,
                system_prompt=system,
                temperature=min(1.0, temperature),
                max_tokens=80
            ):
                if not token.startswith("__THOUGHT__"):
                    response += token
        except:
            response = "..."
        
        return response.strip()
    
    async def process_response(self, agent: Agent, response: str, trigger_msg: Dict, current_time: float):
        """Process agent's response - update dynamics, ledger, trust."""
        # Embed response
        try:
            resp_emb = await self.provider.embed(response)
            resp_emb /= (np.linalg.norm(resp_emb) + 1e-9)
            if len(resp_emb) != self.embed_dim:
                resp_emb = resp_emb[:self.embed_dim]
        except:
            resp_emb = agent.dda_state.x_pred.copy()
        
        # Compute surprise
        epsilon = np.linalg.norm(agent.dda_state.x_pred - resp_emb)
        
        # Update rigidity
        rho_before = agent.dda_state.rho
        agent.dda_state.update_rigidity(epsilon)
        agent.rigidity.update(epsilon)
        
        # Update trust with speaker based on prediction error
        speaker_id = trigger_msg["agent_id"]
        observer_idx = self.agent_id_to_idx[agent.id]
        speaker_idx = self.agent_id_to_idx[speaker_id]
        self.trust_matrix.update_trust(observer_idx, speaker_idx, epsilon)
        
        # Store in ledger
        trigger_emb = await self.provider.embed(trigger_msg["text"])
        trigger_emb /= (np.linalg.norm(trigger_emb) + 1e-9)
        if len(trigger_emb) != self.embed_dim:
            trigger_emb = trigger_emb[:self.embed_dim]
        
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=agent.dda_state.x.copy(),
            action_id=f"response_{agent.interaction_count}",
            observation_embedding=trigger_emb,
            outcome_embedding=resp_emb,
            prediction_error=epsilon,
            context_embedding=trigger_emb,
            rigidity_at_time=agent.dda_state.rho,
            metadata={
                "heard_from": speaker_id,
                "heard": trigger_msg["text"][:80],
                "said": response[:80]
            }
        )
        agent.ledger.add_entry(entry)
        
        # Update state
        agent.dda_state.x_pred = resp_emb
        agent.dda_state.x = (1 - 0.1) * agent.dda_state.x + 0.1 * resp_emb
        agent.last_spoke = current_time
        agent.interaction_count += 1
        
        return epsilon, rho_before
    
    async def run(self, seed_message: str, total_messages: int = 40):
        """Run the society simulation."""
        await self.setup()
        
        print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  SOCIETY CHAT (Natural Flow){C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        # Seed message from a random agent
        seed_agent = random.choice(list(self.agents.values()))
        seed_msg = {"agent_id": seed_agent.id, "text": seed_message, "time": 0}
        self.conversation.append(seed_msg)
        
        print(f"\n{seed_agent.color}[{seed_agent.name}]{C.RESET} {seed_message}")
        
        msg_count = 1
        current_time = 0
        
        while msg_count < total_messages:
            current_time += 1
            
            # Get the last message
            last_msg = self.conversation[-1]
            
            # Select who responds
            responders = await self.select_responders(last_msg, current_time)
            
            if not responders:
                # No one responded - random agent speaks up
                available = [a for a in self.agents.values() if a.id != last_msg["agent_id"]]
                if available:
                    random_agent = random.choice(available)
                    responders = [random_agent.id]
            
            # Generate responses
            for responder_id in responders:
                if msg_count >= total_messages:
                    break
                
                agent = self.agents[responder_id]
                response = await self.generate_response(agent, last_msg)
                
                if response and response != "...":
                    epsilon, rho_before = await self.process_response(agent, response, last_msg, current_time)
                    
                    new_msg = {"agent_id": agent.id, "text": response, "time": current_time}
                    self.conversation.append(new_msg)
                    
                    delta = agent.dda_state.rho - rho_before
                    rho_color = C.RED if delta > 0.02 else C.GREEN if delta < -0.01 else C.DIM
                    
                    print(f"\n{agent.color}[{agent.name}]{C.RESET} {response}")
                    print(f"{C.DIM}  ε:{epsilon:.2f} Δρ:{rho_color}{delta:+.2f}{C.RESET} ρ:{agent.dda_state.rho:.2f}{C.RESET}")
                    
                    msg_count += 1
                    last_msg = new_msg  # Chain responses
                    
                    await asyncio.sleep(0.1)
        
        self.display_summary()
    
    def display_summary(self):
        """Show final society state."""
        print(f"\n\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  FINAL SOCIETY STATE{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        # Sort by interaction count
        agents_sorted = sorted(self.agents.values(), key=lambda a: a.interaction_count, reverse=True)
        
        print(f"\n{C.CYAN}Agent Activity:{C.RESET}")
        for agent in agents_sorted:
            stats = agent.ledger.get_statistics()
            diag = agent.rigidity.get_diagnostic()
            
            print(f"  {agent.color}●{C.RESET} {agent.name:8} | msgs: {agent.interaction_count:2} | ρ: {agent.dda_state.rho:.2f} | peak: {diag['peak_fast']:.2f} | trauma: {diag['rho_trauma']:.6f}")
        
        # Trust analysis
        print(f"\n{C.CYAN}Trust Dynamics:{C.RESET}")
        trust_stats = self.trust_matrix.get_network_stats()
        print(f"  Mean trust: {trust_stats.get('mean_trust', 0):.3f}")
        print(f"  Min trust: {trust_stats.get('min_trust', 0):.3f}")
        
        print(f"\n{C.BOLD}Total messages: {len(self.conversation)}{C.RESET}")


async def main():
    sim = SocietySimulation()
    
    # Seed with a controversial/interesting topic
    seed = "I've been thinking... is AI going to make us all obsolete, or is that just fear-mongering?"
    
    await sim.run(seed_message=seed, total_messages=35)


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Ended.{C.RESET}")
