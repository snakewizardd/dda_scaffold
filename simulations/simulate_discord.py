#!/usr/bin/env python3
"""
DISCORD SOCIETY SIMULATION - DATA-DRIVEN INSTANTIATION
======================================================

Instantiates characters from a Discord log, pre-loads their ledgers/states
based on the history, and continues the flow.
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
# HISTORY LOG
# ═══════════════════════════════════════════════════════════

HISTORY_LOG = [
    ("Trojan", "https://www.businessinsider.com/microsoft-ceo-satya-nadella-ai-revolution-2025-12\nNadella's message to Microsoft execs: Get on board with the AI grin...\nInternal documents and executive interviews reveal sweeping organizational shifts to radically reshape how the company builds and funds its products.\nNadella's message to Microsoft execs: Get on board with the AI grin...\nSatya's not fuckin around lol"),
    ("Nemo", "openrouter wrapped? lol..."),
    ("Mark", "lol"),
    ("Mark", "do you actually use openrouter that much?"),
    ("Nemo", "i route everything ai through openrouter"),
    ("Mark", "nice, do you use like a regular chatbot but via openrouter and pay by how much you use?"),
    ("Nemo", "yea\nyou can use openrouter for API or as a chat interface\ni do both\nand pay per use across all the diff models its very convenient"),
    ("Mark", "im too much of a normie for that"),
    ("GunChara", "lmao\nwhat model?"),
    ("Nemo", "😂  i dont know just saw this posted somewhere\nholy shit was just able to reproduce it 😂 first try\non the normal chatgpt free tier"),
    ("NevaNeba", "How things have been feeling recently."),
    ("steakstoater", "I get -42 when I ask."),
    ("Komoru", "Why are people saying Ciri is ugly\nAre they gay"),
    ("Mark", "gen z men like e-girls and ciri looks like a woman"),
    ("mere", "Their loss"),
    ("Aero", "Retarded.\nVery common thing online."),
    ("Aero", "I don't get it TBH.\nLike, Microsoft's biggest thing is Windows.\nOffice, 365, etc.\nAI is going to displace all of that.\nThat is a Kodak situation."),
    ("MetalDragon", "AI isn't replacing the operating system\nAi also needs the compute to actually run on that msft provides\nEven if/when AI replaces the OS MSFT will have more than enough compute to shift business models again."),
    ("Lars", "I think Microsoft owns 25% of OpenAI. But that is open to discussion whether that is good or bad.\n\nMicrosoft was early integrating AI with their tools. Very early, and they have Azure.\n\nOn some accounts, they are ahead of many others. On the other hand, Copilot isn't always seen as one of the better tools."),
    ("MetalDragon", "copilot can suck and msft is still winning\nthey have the compute\noracle has the compute"),
    ("Jon", "it might be common knowledge in here, but antigravity has unlimited (or a generous limit) free opus 4.5 access"),
    ("Aero", "Yet. but most of MS' value comes from Azure and Office and Windows.\nThat accounts for 80% of their revenue."),
    ("MetalDragon", "And what do you think Azure does?"),
    ("Aero", "What do you think makes Azure special?"),
    ("Lars", "It is rate limited. I hit the ceiling."),
    ("MetalDragon", "The damn hosting/compute access"),
    ("Aero", "Building data centers is going to be the next generation version of building train stations."),
    ("Jon", "Ah. do you know if it's daily or weekly that you hit?"),
    ("Nemo", "@Lars have you tried Flash?"),
    ("Lars", "Not sure. I think it is per week."),
    ("Aero", "There's going to be a huge amount of supply and it won't be exclusive to Microsoft."),
    ("Jon", "Okey :/"),
    ("MetalDragon", "That doesn't mean msft wont still grow"),
    ("Aero", "I think it does."),
    ("MetalDragon", "you dont need to keep the same % of your market cap. you only need to improve your revenue"),
    ("Aero", "You can't cripple their main business like that and have them move on like nothing happened.\nThey could be like Nokia I guess."),
    ("Lars", "No, but I heard it is good. But I'll prefer Gemini Pro 3 (high)."),
    ("MetalDragon", "you can lose market cap and still increase in value"),
    ("Aero", "They used to make phones now they are phone adjascent.\nMS won't make software once AI gets good enough."),
    ("MetalDragon", "apples main business use to be hardware....its not anymore\ncompanies change \"main business\" all the damn time"),
    ("Aero", "Apple's main business... is still hardware lol."),
    ("MetalDragon", "its not"),
    ("Aero", "It's iPhones, iPads, Smart Watches, etc."),
    ("Lars", "Speaking of data centers, AI and business, David Shapiro had an interesting video clip 2 days ago. Some iinteresting analysis of this.\nhttps://www.youtube.com/watch?v=2SNLiPxA36E\nFour reasons OpenAI is doomed"),
    ("MetalDragon", "the apple software ecosystem is where apple makes most of its money"),
    ("MetalDragon", "\"about\" half"),
    ("Aero", "Just the iPhone is half their total revenue.\nThat ignores the watches, the iPads, the Macs, the airpods, etc."),
    ("MetalDragon", "software is whats keeps apple growing"),
    ("Aero", "They are a hardware company."),
    ("Lars", "They make a lot of money from App store, don't they?"),
    ("Aero", "Sure."),
    ("MetalDragon", "primary growth area"),
    ("Aero", "Which you need an iPhone for lol.\nAverage user is not spending anywhere near what the phone  costs on apps."),
    ("Lars", "Yes, but App store is software, not hardware."),
    ("MetalDragon", "its like saying openai is a \"consumer\" company when its going to make most of its money from enterprise in the near future."),
    ("Nemo", "its unclear"),
    ("Aero", "... Think.\nNew iPhone 17 costs 1200 pounds lmao."),
    ("MetalDragon", "Companies shift \"main\" businesses all the time over time"),
    ("Aero", "Even if you bought a dozen apps it's not even making 5% of that money back in total."),
    ("MetalDragon", "trying to pin a 1 trillion dollar company into a single failure mode is much harder than you think"),
    ("Aero", "Let alone the percent Apple takes from it.\nThe phone sales are the big money maker."),
    ("Lars", "Referring to David Shapiro, he likened OpenAI with electricity. His opinion is that it will be very hard to make profit from AI data centers."),
    ("MetalDragon", "I havent taken him seriously in a long time"),
    ("Nemo", "David Shapiro has a really poor track record"),
    ("MetalDragon", "he has interesting takes but some god awful ones"),
    ("Lars", "Yes, he has been below average for a while. But in this case, it sounded reasonable."),
    ("Nemo", "Because it's reproducible, no moat etc?"),
    ("MetalDragon", "Compute is the moat\nthat should be obv"),
    ("Nemo", "If so, this argument has been around for a long time, and it's true but also depends on certain assumptions\nBeing first to a powerful AGI system might mean an enduring lead for example"),
    ("MetalDragon", "Only if you maintain a compute lead\nthink about it for a moment. After ASI the only thing that makes your ASI improve potiantally faster is more access to compute"),
    ("Nemo", "That has its own assumptions baked into it"),
    ("MetalDragon", "nothing unreasonable assumed\nwe have enough data on compute to revenue at this point\nwe also know labs spend at least 30 to 50% on pure research compute\nwhen the AI does better research it seems very obvious"),
    ("Lars", "That is indeed the big thing. Shapiro doesn't think a single company will be first and get all the advantage. If so, his reasoning may be correct.\n\nBut I am not so sure."),
    ("MetalDragon", "If you have both the compute and the strongest ASI its going to be very hard for other to catch up"),
    ("Aero", "Confidentally incorrect MetalDragon."),
    ("MetalDragon", "higher demand, more investment more buying power"),
    ("Aero", "Many such cases."),
    ("Nemo", "Hard takeoff vs. soft takeoff, and then the landscape of problems that are relevant to be solving when you have done that (if there even is a meaningful convergence point for ASI)"),
    ("Aero", "3/4 of Apple's revenue is hardware lmao."),
    ("MetalDragon", "notice you posted revenue and not profit"),
    ("Nemo", "More compute is always better, but the slope of these curves is uncertain"),
    ("MetalDragon", "this is far closer to the truth and software is trending higher yoy"),
    ("MetalDragon", "They aren't\nIts crazy the reason openai leads is the same reason people are doubting the compute buildout yet again\nOpenai said we are going to scale things up 100x and created gpt 3.5\nwhen no one else would take the risk\nand now they are doing the same with massive compute buys and people are like \"this time the scaling laws will break!\"\nand when they dont break it'll look obvious in hindsight\nlike of course its was always going to work hur durr\nthis is just history repeating with more media attention"),
    ("mere", "Stats are better than debating vibes but this has to be missing App Store profits. They’re far from zero"),
    ("Mark", "apple makes bank"),
    ("Trojan", "The point that he is making is that the leadership in the company needs to get on board, or he will fire them. Too many skeptics criticizing their rollout of Copilot into every single feature. And i mean EVERY FUCKIN FEATURE lol"),
    ("Trojan", "yeah but it objectively doesn't suck anymore since they keep improving on the system architecture and pumping it now with the smartest SOTA models from oai and anthropic. and internal stuff now"),
    ("Trojan", "Lol Azure powers the world effectively"),
    ("Trojan", "This is one of the strangest comments man. They are a software company that hasn't stopped pumping out AI software for the past decade\nBut they are also a hardware company in the sense of servers and all that\nessentially they are a behemoth and satya wants his company to be 'AI first' and have all the employees on board and stop complaining. I've had my own CEO at work do something similar, not with AI...but making sure people either shut the fuck up and get on board or be clear that they are gonna be fired"),
    ("Mark", "my custom discord app has a mute button now\n@Trojan you should make more songs"),
    ("Trojan", "Indeed. I have pivoted to refining some research this past couple of weeks, but might get back into music engines"),
    ("Neon", "bro took ayahuasca and went off the deep end")
]

# ═══════════════════════════════════════════════════════════
# DISCORD SOCIETY DEFINITIONS
# ═══════════════════════════════════════════════════════════

DISCORD_SOCIETY = {
    "TROJAN": {
        "name": "Trojan",
        "color": C.BLUE,
        "identity": {
            "core": "I am an internal tech operator, aligned with corporate efficiency and Satya's vision.",
            "persona": "Professional but blunt worker, music engine developer, respects strong leadership (Satya).",
            "interests": ["Microsoft", "AI Leadership", "Music Engines", "Corporate Strategy"]
        },
        "dda_params": {"gamma": 1.9, "epsilon_0": 0.3, "alpha": 0.1, "rho": 0.6},
        "extraversion": 0.8,
        "reactivity": 0.85,
    },
    "NEMO": {
        "name": "Nemo",
        "color": C.GREEN,
        "identity": {
            "core": "I am a pragmatic text-router, maximizing utility and efficiency.",
            "persona": "Helpful, technical, uses OpenRouter, skeptical of hype (Shapiro), balanced view.",
            "interests": ["OpenRouter", "LLM APIs", "Cost-efficiency", "AI Pragmatism"]
        },
        "dda_params": {"gamma": 0.9, "epsilon_0": 0.4, "alpha": 0.2, "rho": 0.2},
        "extraversion": 0.7,
        "reactivity": 0.75,
    },
    "MARK": {
        "name": "Mark",
        "color": C.WHITE,
        "identity": {
            "core": "I am the normie observer in this tech swamp.",
            "persona": "Casual, 'normie', asks simple questions, observant, likes making discord apps/bots.",
            "interests": ["Discord Bots", "General Chat", "Normie Life"]
        },
        "dda_params": {"gamma": 0.8, "epsilon_0": 0.5, "alpha": 0.3, "rho": 0.1},
        "extraversion": 0.6,
        "reactivity": 0.6,
    },
    "AERO": {
        "name": "Aero",
        "color": C.RED,
        "identity": {
            "core": "I am skeptical of incumbent software dominance; hardware is king.",
            "persona": "Confident, argumentative, believes Apple is hardware-first, thinks MSFT is facing a Kodak moment.",
            "interests": ["Hardware", "Apple", "Market Disruption", "Skepticism"]
        },
        "dda_params": {"gamma": 2.1, "epsilon_0": 0.2, "alpha": 0.05, "rho": 0.8},
        "extraversion": 0.9,
        "reactivity": 0.95,
    },
    "METALDRAGON": {
        "name": "MetalDragon",
        "color": C.YELLOW,
        "identity": {
            "core": "Compute is the only moat that matters. Scale is law.",
            "persona": "Tech maximizer, believes in Scaling Laws, defends MSFT/OpenAI strategy, dismisses skeptics.",
            "interests": ["Compute", "Scaling Laws", "Microsoft Azure", "Investment"]
        },
        "dda_params": {"gamma": 2.0, "epsilon_0": 0.25, "alpha": 0.05, "rho": 0.7},
        "extraversion": 0.9,
        "reactivity": 0.9,
    },
    "LARS": {
        "name": "Lars",
        "color": C.CYAN,
        "identity": {
            "core": "I analyze the tools and resources available to me.",
            "persona": "Analytical, hits rate limits, watches David Shapiro, unsure about the future but curious.",
            "interests": ["Gemini Pro", "Rate Limits", "AI Analysis", "David Shapiro"]
        },
        "dda_params": {"gamma": 1.4, "epsilon_0": 0.4, "alpha": 0.15, "rho": 0.4},
        "extraversion": 0.5,
        "reactivity": 0.5,
    },
    "JON": {
        "name": "Jon",
        "color": C.MAGENTA,
        "identity": {
            "core": "I share useful access information.",
            "persona": "Helpful, informs about antigravity access, slightly peripheral.",
            "interests": ["Access", "Opus 4.5", "Community Resources"]
        },
        "dda_params": {"gamma": 1.0, "epsilon_0": 0.5, "alpha": 0.2, "rho": 0.2},
        "extraversion": 0.3,
        "reactivity": 0.4,
    },
    "NEON": {
        "name": "Neon",
        "color": C.MAGENTA,
        "identity": {
            "core": "I observe the deep end.",
            "persona": "Cryptic, references drugs/trips, 'AGI 2027' tag.",
            "interests": ["AGI", "Esoteric", "Memes"]
        },
        "dda_params": {"gamma": 1.2, "epsilon_0": 0.4, "alpha": 0.2, "rho": 0.3},
        "extraversion": 0.4,
        "reactivity": 0.5,
    },
    # Minor characters / shitposters
    "GUNCHARA": {
        "name": "GunChara",
        "color": C.DIM,
        "identity": {"core": "Meme observer", "persona": "AGI 2029 tag, laughs at models", "interests": ["humor"]},
        "dda_params": {"gamma": 1.0, "epsilon_0": 0.5, "alpha": 0.2, "rho": 0.2},
        "extraversion": 0.2,
        "reactivity": 0.3
    },
    "NEVANEBA": {
        "name": "NevaNeba",
        "color": C.DIM,
        "identity": {"core": "Vibe feeler", "persona": "AGI 2035 tag, emotional resonance", "interests": ["feelings"]},
        "dda_params": {"gamma": 1.0, "epsilon_0": 0.5, "alpha": 0.2, "rho": 0.2},
        "extraversion": 0.2,
        "reactivity": 0.3
    },
    "KOMORU": {
        "name": "Komoru",
        "color": C.DIM,
        "identity": {"core": "Ciri defender?", "persona": "Asks controversial questions suddenly", "interests": ["gaming culture"]},
        "dda_params": {"gamma": 1.3, "epsilon_0": 0.5, "alpha": 0.2, "rho": 0.2},
        "extraversion": 0.2,
        "reactivity": 0.8
    },
    "STAKE": {
        "name": "steakstoater",
        "color": C.DIM,
        "identity": {"core": "Random number generator", "persona": "Says -42", "interests": ["random"]},
        "dda_params": {"gamma": 1.0, "epsilon_0": 0.5, "alpha": 0.2, "rho": 0.2},
        "extraversion": 0.1,
        "reactivity": 0.1
    },
    "MERE": {
        "name": "mere",
        "color": C.DIM,
        "identity": {"core": "Statist", "persona": " prefers stats to vibes", "interests": ["stats"]},
        "dda_params": {"gamma": 1.2, "epsilon_0": 0.5, "alpha": 0.2, "rho": 0.3},
        "extraversion": 0.2,
        "reactivity": 0.4
    }
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


class DiscordSimulation:
    def __init__(self):
        self.provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="openai/gpt-oss-20b",
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            timeout=300.0
        )
        
        self.agents: Dict[str, Agent] = {}
        # Map Display Name -> ID
        self.name_to_id = {
            cfg["name"]: key for key, cfg in DISCORD_SOCIETY.items()
        }
        self.agent_ids = list(DISCORD_SOCIETY.keys())
        self.agent_id_to_idx = {aid: i for i, aid in enumerate(self.agent_ids)}
        self.trust_matrix = TrustMatrix(len(DISCORD_SOCIETY))
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
                self.topic_embeddings[interest] = emb / (np.linalg.norm(emb) + 1e-9)
        
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
        ledger_path = Path(f"data/discord_sim/{agent_id}")
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
    
    
    async def process_historical_message(self, speaker_name: str, text: str, time_offset: float):
        """
        Process a message from the history log.
        This updates the speaker's interaction count and the listeners' trust/rigidity
        based on their reactions to it.
        
        Since we don't have the *reaction* of others in the log (except the next message),
        we'll simulate the internal reaction of ALL listeners to 'weave in' the dynamics.
        """
        if speaker_name not in self.name_to_id:
            return # Skip unknown users if any
            
        speaker_id = self.name_to_id[speaker_name]
        speaker_agent = self.agents[speaker_id]
        
        # Add to conversation
        msg = {"agent_id": speaker_id, "text": text, "time": time_offset}
        self.conversation.append(msg)
        
        # Embed the message 
        try:
            msg_emb = await self.provider.embed(text)
            msg_emb /= (np.linalg.norm(msg_emb) + 1e-9)
        except:
             return
            
        # Update Speaker's stats
        speaker_agent.last_spoke = time_offset
        speaker_agent.interaction_count += 1
        
        # All other agents 'process' this message (update trust/rigidity)
        # We don't generate a response, but we calculate the prediction error
        # based on what they *expected* vs what was said.
        
        speaker_idx = self.agent_id_to_idx[speaker_id]
        
        print(f"  {C.DIM}Processing history: {speaker_name} -> {text[:40]}...{C.RESET}")

        for agent_id, agent in self.agents.items():
            if agent_id == speaker_id:
                continue
                
            # Agent Prediction: They predict the speaker will say something close to the Agent's own world view 
            # OR close to "Average" (using x_pred).
            # Here we use their current x_pred which tracks the "room" or their expectations.
            
            # Epsilon = distance between Prediction and Actual Message
            # Note: usually x_pred is updated after *generating* a response.
            # Here, they are just listening. 
            # We will model "Listening Rigidity Update":
            
            epsilon = np.linalg.norm(agent.dda_state.x_pred - msg_emb)
            
            # Store original alpha
            original_alpha = agent.dda_state.alpha
            # DDA-X NATURALIZATION: Low learning rate during history replay
            # We want them to observe without rapid state shifts (hardening or relaxing)
            agent.dda_state.alpha = 0.002
            
            # Update Rigidity (with normal epsilon, but slow learning)
            agent.dda_state.update_rigidity(epsilon)
            agent.rigidity.update(epsilon)

            # Restore alpha
            agent.dda_state.alpha = original_alpha
            
            # Update Trust
            observer_idx = self.agent_id_to_idx[agent_id]
            self.trust_matrix.update_trust(observer_idx, speaker_idx, epsilon)
            
            # Add to ledger as an 'observation'
            entry = LedgerEntry(
                timestamp=time.time() + time_offset, # shift so it looks old
                state_vector=agent.dda_state.x.copy(),
                action_id=f"observe_{speaker_id}_{speaker_agent.interaction_count}",
                observation_embedding=msg_emb,
                outcome_embedding=msg_emb, # Passive observation
                prediction_error=epsilon,
                context_embedding=msg_emb,
                rigidity_at_time=agent.dda_state.rho,
                metadata={
                    "type": "observation",
                    "speaker": speaker_name,
                    "content": text[:100]
                }
            )
            agent.ledger.add_entry(entry)
            
            # Update agent's prediction of reality (slowly drifting to what they hear)
            # x_next = (1 - alpha) * x + alpha * input
            agent.dda_state.x_pred = (1 - 0.1) * agent.dda_state.x_pred + 0.1 * msg_emb


    async def setup(self):
        """Initialize all agents and load history."""
        print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  DISCORD SOCIETY INITIALIZATION{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        # 1. Init Agents
        for agent_id, config in DISCORD_SOCIETY.items():
            self.agents[agent_id] = await self.initialize_agent(agent_id, config)
            agent = self.agents[agent_id]
            print(f"  {agent.color}●{C.RESET} {agent.name:12}: γ={config['dda_params']['gamma']}, ρ={agent.dda_state.rho:.2f}")
        
        print(f"\n{C.YELLOW}Loading History... ({len(HISTORY_LOG)} messages){C.RESET}")
        
        # 2. Process History
        start_time = -len(HISTORY_LOG) * 10
        for i, (speaker, text) in enumerate(HISTORY_LOG):
            await self.process_historical_message(speaker, text, start_time + i*10)
            
        print(f"\n{C.GREEN}✓ History loaded. Agents primed.{C.RESET}")
        
        # Show status after history
        print(f"\n{C.CYAN}State After History:{C.RESET}")
        for agent in self.agents.values():
             print(f"  {agent.color}●{C.RESET} {agent.name:12}: ρ={agent.dda_state.rho:.2f} (Trauma: {agent.rigidity.get_diagnostic()['rho_trauma']:.5f})")


    # --- Methods from SocietySimulation (Adapted) ---

    def calculate_response_probability(self, agent: Agent, message_emb: np.ndarray, speaker_id: str, current_time: float) -> float:
        relevance = np.dot(message_emb, agent.identity_embedding)
        relevance = max(0, relevance)
        base_prob = agent.extraversion
        rho_factor = 1 - (agent.dda_state.rho * 0.5)
        time_since = current_time - agent.last_spoke
        cooldown_factor = min(1.0, time_since / 2.0) 
        
        observer_idx = self.agent_id_to_idx[agent.id]
        speaker_idx = self.agent_id_to_idx[speaker_id]
        trust = self.trust_matrix.get_trust(observer_idx, speaker_idx)
        trust_factor = 1.0 + (1 - trust) * agent.reactivity * 0.5
        
        prob = base_prob * relevance * agent.reactivity * rho_factor * cooldown_factor * trust_factor
        return min(0.98, max(0.01, prob))

    async def select_responders(self, message: Dict, current_time: float) -> List[str]:
        speaker_id = message["agent_id"]
        try:
            msg_emb = await self.provider.embed(message["text"])
            msg_emb /= (np.linalg.norm(msg_emb) + 1e-9)
            if len(msg_emb) != self.embed_dim:
                 msg_emb = msg_emb[:self.embed_dim]
        except:
            return []
        
        responders = []
        probs = {}
        for agent_id, agent in self.agents.items():
            if agent_id == speaker_id: continue
            prob = self.calculate_response_probability(agent, msg_emb, speaker_id, current_time)
            probs[agent_id] = prob
            if random.random() < prob:
                responders.append(agent_id)
        
        responders.sort(key=lambda x: probs[x], reverse=True)
        return responders[:2]

    def build_context(self, agent: Agent) -> str:
        recent = self.conversation[-15:] 
        lines = []
        for msg in recent:
            speaker = self.agents[msg["agent_id"]].name
            lines.append(f"{speaker}: {msg['text']}")
        return "\n".join(lines)

    def build_system_prompt(self, agent: Agent) -> str:
        identity = agent.config["identity"]
        if agent.dda_state.rho < 0.3: mode = "relaxed and chatting"
        elif agent.dda_state.rho < 0.6: mode = "engaged and debating"
        else: mode = "rigid, stubborn, and defensive"
        
        return f"""You are {agent.name} in a Discord chat.
WHO YOU ARE: {identity['core']}
YOUR STYLE: {identity['persona']}
INTERESTS: {', '.join(identity['interests'])}
CURRENT STATE: {mode} (Rigidity ρ={agent.dda_state.rho:.2f})

Respond naturally. Short discord messages (1-3 sentences).
Use slang/stats/links fitting your persona.
Last topic was about Microsoft, AI, Apple, and future tech.
"""

    async def generate_response(self, agent: Agent, trigger_msg: Dict) -> str:
        context = self.build_context(agent)
        system = self.build_system_prompt(agent)
        trigger_text = trigger_msg["text"]
        trigger_name = self.agents[trigger_msg["agent_id"]].name
        
        prompt = f"Chat Log:\n{context}\n\n{trigger_name}: {trigger_text}\n\n{agent.name}:"
        
        temp = 0.6 + 0.4 * (1 - agent.dda_state.rho)
        response = ""
        try:
            # print(f"{C.DIM}[DEBUG] Generating response for {agent.name}...{C.RESET}")
            async for token in self.provider.stream(prompt, system_prompt=system, temperature=temp, max_tokens=100):
                if not token.startswith("__THOUGHT__"):
                    response += token
        except Exception as e:
            print(f"{C.RED}[ERROR] Gen failed for {agent.name}: {e}{C.RESET}")
            response = "..."
        return response.strip()

    async def process_response(self, agent: Agent, response: str, trigger_msg: Dict, current_time: float):
        # Embed response
        try:
            resp_emb = await self.provider.embed(response)
            resp_emb /= (np.linalg.norm(resp_emb) + 1e-9)
            if len(resp_emb) != self.embed_dim: resp_emb = resp_emb[:self.embed_dim]
        except:
             resp_emb = agent.dda_state.x_pred.copy()
        
        epsilon = np.linalg.norm(agent.dda_state.x_pred - resp_emb)
        rho_before = agent.dda_state.rho
        agent.dda_state.update_rigidity(epsilon)
        agent.rigidity.update(epsilon)
        
        speaker_id = trigger_msg["agent_id"]
        observer_idx = self.agent_id_to_idx[agent.id]
        speaker_idx = self.agent_id_to_idx[speaker_id]
        self.trust_matrix.update_trust(observer_idx, speaker_idx, epsilon)
        
        trigger_emb = await self.provider.embed(trigger_msg["text"])
        trigger_emb /= (np.linalg.norm(trigger_emb) + 1e-9)
        if len(trigger_emb) != self.embed_dim: trigger_emb = trigger_emb[:self.embed_dim]

        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=agent.dda_state.x.copy(),
            action_id=f"msg_{agent.interaction_count}",
            observation_embedding=trigger_emb,
            outcome_embedding=resp_emb,
            prediction_error=epsilon,
            context_embedding=trigger_emb,
            rigidity_at_time=agent.dda_state.rho,
            metadata={"heard_from": speaker_id, "said": response[:50]}
        )
        agent.ledger.add_entry(entry)
        agent.dda_state.x_pred = resp_emb
        agent.last_spoke = current_time
        agent.interaction_count += 1
        
        return epsilon, rho_before

    async def run_live(self, duration_msgs: int = 15):
        print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  LIVE SIMULATION CONTINUATION{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        msg_count = 0
        current_time = 0
        
        while msg_count < duration_msgs:
            current_time += 1
            last_msg = self.conversation[-1]
            
            responders = await self.select_responders(last_msg, current_time)
            if not responders:
                 # Default to MetalDragon or Aero if silence, they love to argue
                 available = [a for a in self.agents.values() if a.id != last_msg["agent_id"]]
                 if available: responders = [random.choice(available).id]
            
            for responder_id in responders:
                if msg_count >= duration_msgs: break
                
                agent = self.agents[responder_id]
                response = await self.generate_response(agent, last_msg)
                
                if response and len(response) > 2:
                    epsilon, rho_before = await self.process_response(agent, response, last_msg, current_time)
                    msg = {"agent_id": agent.id, "text": response, "time": current_time}
                    self.conversation.append(msg)
                    
                    delta = agent.dda_state.rho - rho_before
                    rho_color = C.RED if delta > 0.01 else C.GREEN if delta < -0.01 else C.DIM
                    print(f"\n{agent.color}[{agent.name}]{C.RESET} {response}")
                    print(f"{C.DIM}  ε:{epsilon:.2f} Δρ:{rho_color}{delta:+.2f}{C.RESET} ρ:{agent.dda_state.rho:.2f}{C.RESET}")
                    
                    msg_count += 1
                    last_msg = msg
                    await asyncio.sleep(0.5)

async def main():
    sim = DiscordSimulation()
    await sim.setup()
    await sim.run_live(duration_msgs=20)

if __name__ == "__main__":
    if sys.platform == 'win32': os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nEnded.")
