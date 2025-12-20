#!/usr/bin/env python3
"""
TWO NPCs CONVERSING WITH DDA PHYSICS
=====================================

The architect designs:
1. Two distinct personalities (YAMLs)
2. Preloaded memories and reflections
3. Initial cognitive state

Then sets the scene and LETS IT COOK.

The DDA force mechanics drive:
- Identity pull (staying true to who they are)
- Truth channel (reality of the conversation)
- Rigidity dynamics (stress, adaptation)
- Memory formation (experiences accumulate)

No scripted outcomes. Just watch emergence.
"""

import asyncio
import sys
import os
import numpy as np
import time
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.core.forces import ForceAggregator, IdentityPull, TruthChannel
from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.llm.hybrid_provider import HybridProvider


class C:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


# ═══════════════════════════════════════════════════════════
# NPC PERSONALITY DEFINITIONS (Architect-Designed)
# ═══════════════════════════════════════════════════════════

NPC_CONFIGURATIONS = {
    "VERA": {
        "name": "Vera",
        "color": C.CYAN,
        "identity": {
            "core": "I believe in truth above all else. Kindness can be cruel if it hides what matters.",
            "persona": "Direct, perceptive, slightly impatient. I see through pretense.",
            "role": "I'm talking to someone who might be hiding something from themselves."
        },
        "dda_params": {
            "gamma": 1.5,       # Strong identity pull
            "epsilon_0": 0.35,  # Moderate surprise threshold
            "alpha": 0.12,      # Responsive
            "s": 0.1,
            "rho": 0.25,        # Slightly guarded
        },
        "preloaded_memories": [
            "I once told a friend the truth they didn't want to hear. They hated me for months. Then thanked me years later.",
            "I've learned that people who avoid eye contact often carry the heaviest burdens.",
            "My father always said: 'The kindest thing you can do is be honest, even when it hurts.'",
        ],
        "preloaded_reflections": [
            "When someone deflects, push gently but persistently.",
            "Silence can be more revealing than words.",
        ],
    },
    
    "MARCUS": {
        "name": "Marcus",
        "color": C.YELLOW,
        "identity": {
            "core": "I protect myself by keeping things simple. Depth is dangerous.",
            "persona": "Easy-going, humorous, deflective. I keep things light.",
            "role": "I'm talking to someone who seems to want something from me."
        },
        "dda_params": {
            "gamma": 0.8,       # Weaker identity pull (more adaptable)
            "epsilon_0": 0.4,   # Higher threshold (less easily surprised)
            "alpha": 0.08,      # Slower adaptation
            "s": 0.15,
            "rho": 0.3,         # Slightly defensive
        },
        "preloaded_memories": [
            "Every time I opened up, people either left or used it against me.",
            "Humor is my shield. If I make them laugh, they stop digging.",
            "I once told someone how I really felt. They called me 'too much.'",
        ],
        "preloaded_reflections": [
            "When cornered, change the subject or make a joke.",
            "Stay vague. Specifics give people ammunition.",
        ],
    }
}

# The scene the architect sets
SCENE = """
Setting: A quiet bar, late evening. Two strangers end up at adjacent seats.
Neither was looking for company, but the silence became heavier than conversation.

One of them finally speaks.
"""

OPENING_TOPIC = "So... rough day, or just enjoying the existential silence?"


@dataclass
class NPCState:
    """Complete NPC state for interaction."""
    name: str
    color: str
    config: Dict
    dda_state: DDAState
    rigidity: MultiTimescaleRigidity
    ledger: ExperienceLedger
    identity_embedding: np.ndarray
    conversation_history: List[str] = field(default_factory=list)


class ConversationSimulator:
    """
    Let two NPCs converse using DDA force mechanics.
    
    The architect sets up, then steps back and observes.
    """
    
    def __init__(self):
        self.provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="openai/gpt-oss-20b",
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            timeout=300.0
        )
        
        self.npcs: Dict[str, NPCState] = {}
        self.global_history: List[str] = []
        
    async def initialize_npc(self, npc_id: str, config: Dict) -> NPCState:
        """Initialize an NPC with their designed personality."""
        name = config["name"]
        print(f"{C.DIM}Initializing {name}...{C.RESET}")
        
        # Create identity embedding from core + persona
        identity_text = f"{config['identity']['core']} {config['identity']['persona']}"
        identity_emb = await self.provider.embed(identity_text)
        identity_emb /= np.linalg.norm(identity_emb)
        
        # Initialize DDA state
        params = config["dda_params"]
        dda_state = DDAState(
            x=identity_emb.copy(),
            x_star=identity_emb.copy(),
            gamma=params["gamma"],
            epsilon_0=params["epsilon_0"],
            alpha=params["alpha"],
            s=params["s"],
            rho=params["rho"],
            x_pred=identity_emb.copy()
        )
        
        # Create ledger for this NPC
        ledger_path = Path(f"data/npc_conversation/{npc_id}")
        if ledger_path.exists():
            import shutil
            shutil.rmtree(ledger_path)
        
        ledger = ExperienceLedger(
            storage_path=ledger_path,
            lambda_recency=0.01,
            lambda_salience=2.0
        )
        
        # Preload memories
        for memory in config.get("preloaded_memories", []):
            mem_emb = await self.provider.embed(memory)
            mem_emb /= np.linalg.norm(mem_emb)
            
            entry = LedgerEntry(
                timestamp=time.time() - 86400,  # 1 day ago
                state_vector=identity_emb.copy(),
                action_id=f"memory_{len(ledger.entries)}",
                observation_embedding=mem_emb,
                outcome_embedding=mem_emb,
                prediction_error=0.2,  # Moderate salience
                context_embedding=mem_emb,
                rigidity_at_time=params["rho"],
                metadata={"type": "memory", "content": memory}
            )
            ledger.add_entry(entry)
        
        # Preload reflections
        for reflection in config.get("preloaded_reflections", []):
            ref_emb = await self.provider.embed(reflection)
            ref_emb /= np.linalg.norm(ref_emb)
            
            ref_entry = ReflectionEntry(
                timestamp=time.time() - 86400,
                task_intent="self_understanding",
                situation_embedding=ref_emb,
                reflection_text=reflection,
                prediction_error=0.3,
                outcome_success=True
            )
            ledger.add_reflection(ref_entry)
        
        return NPCState(
            name=name,
            color=config["color"],
            config=config,
            dda_state=dda_state,
            rigidity=MultiTimescaleRigidity(),
            ledger=ledger,
            identity_embedding=identity_emb,
            conversation_history=[]
        )
    
    async def setup(self):
        """Initialize both NPCs."""
        print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  ARCHITECT INITIALIZATION{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        for npc_id, config in NPC_CONFIGURATIONS.items():
            self.npcs[npc_id] = await self.initialize_npc(npc_id, config)
            
            npc = self.npcs[npc_id]
            stats = npc.ledger.get_statistics()
            print(f"\n{npc.color}{npc.name}{C.RESET}")
            print(f"  Core: {config['identity']['core'][:60]}...")
            print(f"  ρ: {npc.dda_state.rho:.3f} | γ: {config['dda_params']['gamma']}")
            print(f"  Memories: {stats.get('current_entries', 0)} | Reflections: {stats.get('current_reflections', 0)}")
        
        print(f"\n{C.GREEN}✓ NPCs ready{C.RESET}")
    
    async def retrieve_context(self, npc: NPCState, current_situation: str) -> str:
        """Retrieve relevant memories and reflections for this NPC."""
        sit_emb = await self.provider.embed(current_situation)
        sit_emb /= np.linalg.norm(sit_emb)
        if len(sit_emb) != len(npc.identity_embedding):
            sit_emb = sit_emb[:len(npc.identity_embedding)]
        
        entries = npc.ledger.retrieve(sit_emb, k=3, min_score=0.1)
        reflections = npc.ledger.retrieve_reflections(sit_emb, k=2, min_score=0.1)
        
        context_parts = []
        
        if entries:
            context_parts.append("Relevant memories surfacing:")
            for e in entries:
                if 'content' in e.metadata:
                    context_parts.append(f"  * {e.metadata['content'][:80]}")
        
        if reflections:
            context_parts.append("Patterns I've learned:")
            for r in reflections:
                context_parts.append(f"  * {r.reflection_text}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def build_system_prompt(self, npc: NPCState, retrieved_context: str) -> str:
        """Build system prompt with identity and cognitive state."""
        identity = npc.config["identity"]
        
        # Cognitive state description
        if npc.dda_state.rho < 0.3:
            mode = "OPEN - willing to explore"
        elif npc.dda_state.rho < 0.6:
            mode = "ENGAGED - processing carefully"
        else:
            mode = "GUARDED - protecting yourself"
        
        return f"""You are {npc.name}.

WHO YOU ARE (core):
{identity['core']}

YOUR STYLE (persona):
{identity['persona']}

CURRENT SITUATION:
{identity['role']}

{retrieved_context}

YOUR CURRENT STATE:
Mode: {mode}
Rigidity: {npc.dda_state.rho:.3f}

Respond naturally as {npc.name}. 2-3 sentences max.
Your cognitive state affects your openness. Higher rigidity = more guarded.
"""

    async def generate_response(self, npc: NPCState, other_said: str) -> str:
        """Generate NPC response using full DDA mechanics."""
        # Retrieve relevant memories/reflections
        current_context = f"In conversation: {other_said}"
        retrieved = await self.retrieve_context(npc, current_context)
        
        # Build prompt with cognitive state
        system = self.build_system_prompt(npc, retrieved)
        
        # Recent conversation history
        recent = "\n".join(self.global_history[-6:]) if self.global_history else "[Conversation just started]"
        
        prompt = f"Conversation so far:\n{recent}\n\nThey just said: \"{other_said}\"\n\nYour response:"
        
        # COGNITION → PARAMS
        # Low ρ = more open = higher temperature
        # High ρ = guarded = lower, more predictable temperature
        temperature = 0.3 + 0.5 * (1 - npc.dda_state.rho)
        
        # Generate
        response = ""
        try:
            async for token in self.provider.stream(
                prompt,
                system_prompt=system,
                temperature=temperature,
                max_tokens=100
            ):
                if not token.startswith("__THOUGHT__"):
                    response += token
        except:
            response = "..."
        
        response = response.strip()
        
        # Embed response
        try:
            resp_emb = await self.provider.embed(response)
            resp_emb /= (np.linalg.norm(resp_emb) + 1e-9)
            if len(resp_emb) != len(npc.dda_state.x_pred):
                resp_emb = resp_emb[:len(npc.dda_state.x_pred)]
        except:
            resp_emb = npc.dda_state.x_pred.copy()
        
        # Compute surprise (prediction error)
        epsilon = np.linalg.norm(npc.dda_state.x_pred - resp_emb)
        
        # Update rigidity
        rho_before = npc.dda_state.rho
        npc.dda_state.update_rigidity(epsilon)
        npc.rigidity.update(epsilon)
        
        # Store experience in ledger
        context_emb = await self.provider.embed(other_said)
        context_emb /= (np.linalg.norm(context_emb) + 1e-9)
        if len(context_emb) != len(npc.identity_embedding):
            context_emb = context_emb[:len(npc.identity_embedding)]
        
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=npc.dda_state.x.copy(),
            action_id=f"turn_{len(npc.conversation_history)}",
            observation_embedding=context_emb,
            outcome_embedding=resp_emb,
            prediction_error=epsilon,
            context_embedding=context_emb,
            rigidity_at_time=npc.dda_state.rho,
            metadata={
                "type": "conversation",
                "heard": other_said[:100],
                "said": response[:100]
            }
        )
        npc.ledger.add_entry(entry)
        
        # Generate reflection if high surprise
        if epsilon > 0.5:
            reflect_text = f"That interaction ('{other_said[:50]}...') surprised me. I should remember this pattern."
            ref_emb = await self.provider.embed(reflect_text)
            ref_emb /= (np.linalg.norm(ref_emb) + 1e-9)
            if len(ref_emb) != len(npc.identity_embedding):
                ref_emb = ref_emb[:len(npc.identity_embedding)]
            
            reflection = ReflectionEntry(
                timestamp=time.time(),
                task_intent="conversation",
                situation_embedding=ref_emb,
                reflection_text=reflect_text,
                prediction_error=epsilon,
                outcome_success=True
            )
            npc.ledger.add_reflection(reflection)
        
        # Update prediction for next round
        npc.dda_state.x_pred = resp_emb
        npc.dda_state.x = (1 - 0.1) * npc.dda_state.x + 0.1 * resp_emb
        
        return response, epsilon, rho_before, npc.dda_state.rho
    
    async def run(self, turns: int = 15):
        """Let it cook."""
        await self.setup()
        
        print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  THE SCENE{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"\n{C.DIM}{SCENE}{C.RESET}")
        
        print(f"\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  CONVERSATION (DDA Physics Active){C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        # Vera starts
        speakers = ["VERA", "MARCUS"]
        last_said = OPENING_TOPIC
        
        npc = self.npcs["VERA"]
        print(f"\n{npc.color}[{npc.name}]{C.RESET} {OPENING_TOPIC}")
        self.global_history.append(f"{npc.name}: {OPENING_TOPIC}")
        
        for turn in range(turns):
            # Alternate speakers
            speaker_id = speakers[(turn + 1) % 2]
            npc = self.npcs[speaker_id]
            
            print(f"\n{C.DIM}─── Turn {turn + 1} ───{C.RESET}")
            
            response, epsilon, rho_before, rho_after = await self.generate_response(npc, last_said)
            
            print(f"{npc.color}[{npc.name}]{C.RESET} {response}")
            
            delta = rho_after - rho_before
            rho_color = C.RED if delta > 0.03 else C.GREEN if delta < -0.01 else C.DIM
            print(f"{C.DIM}  ε: {epsilon:.2f} | Δρ: {rho_color}{delta:+.3f}{C.RESET} → ρ: {rho_after:.3f}{C.RESET}")
            
            self.global_history.append(f"{npc.name}: {response}")
            last_said = response
            
            time.sleep(0.3)
        
        # Summary
        self.display_summary()
    
    def display_summary(self):
        """Show final state."""
        print(f"\n\n{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        print(f"{C.BOLD}  FINAL STATE{C.RESET}")
        print(f"{C.BOLD}═══════════════════════════════════════════════════════════{C.RESET}")
        
        for npc_id, npc in self.npcs.items():
            stats = npc.ledger.get_statistics()
            diag = npc.rigidity.get_diagnostic()
            
            print(f"\n{npc.color}{npc.name}{C.RESET}")
            print(f"  Final ρ: {npc.dda_state.rho:.3f}")
            print(f"  Peak ρ: {diag['peak_fast']:.3f}")
            print(f"  Ledger: {stats.get('current_entries', 0)} entries, {stats.get('current_reflections', 0)} reflections")
            print(f"  Trauma: {diag['rho_trauma']:.6f}")
        
        print(f"\n{C.CYAN}The algorithm cooked. No scripted outcomes.{C.RESET}")
        print(f"{C.CYAN}DDA physics drove the interaction.{C.RESET}")


async def main():
    sim = ConversationSimulator()
    await sim.run(turns=12)


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Ended.{C.RESET}")
