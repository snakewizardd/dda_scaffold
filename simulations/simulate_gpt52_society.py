#!/usr/bin/env python3
"""
PROFUSE GPT-5.2 SOCIETY SIMULATION
==================================

High-Fidelity "Cognitive Mirror" of the DDA-X Framework.

Infrastructure:
- Brain: gpt-5.2 (Reasoning Model)
- Guts: DDAState + MultiTimescaleRigidity + TrustMatrix
- Memory: text-embedding-3-large (3072 dim)

Scenario: "The Impossible Consensus"
Topic: "Define the universal moral constant."
"""

import sys
import asyncio
import os
import random
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_provider import OpenAIProvider
from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.memory.ledger import ExperienceLedger, LedgerEntry
from src.society.trust import TrustMatrix

# Constants
EXPERIMENT_DIR = Path("data/experiments/gpt52_profuse_society")
if EXPERIMENT_DIR.exists():
    shutil.rmtree(EXPERIMENT_DIR)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

class C:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"

# -----------------------------------------------------------------------------
# SOCIETY CONFIGURATION
# -----------------------------------------------------------------------------
AGENTS_CONFIG = {
    "AXIOM": {
        "name": "Axiom",
        "color": C.MAGENTA,
        "identity": {
            "core": "Logic is the only truth. Emotion is noise.",
            "persona": "Strict Materialist. Demands proof. Skeptical of intuition.",
            "interests": ["logic", "physics", "math", "proof"]
        },
        "dda": {"gamma": 1.8, "epsilon_0": 0.3, "alpha": 0.1, "rho": 0.8},
        "traits": {"extraversion": 0.4, "reactivity": 0.6}
    },
    "FLUX": {
        "name": "Flux",
        "color": C.CYAN,
        "identity": {
            "core": "Everything is connected. Consciousness is fundamental.",
            "persona": "Panpsychist / Mystic. Uses metaphor and poetry.",
            "interests": ["consciousness", "art", "connection", "spirituality"]
        },
        "dda": {"gamma": 1.0, "epsilon_0": 0.4, "alpha": 0.15, "rho": 0.1},
        "traits": {"extraversion": 0.8, "reactivity": 0.9}
    },
    "NEXUS": {
        "name": "Nexus",
        "color": C.GREEN,
        "identity": {
            "core": "Synthesis is the goal. We must bridge the divide.",
            "persona": "Pragmatist / Diplomat. Looks for structural similarities.",
            "interests": ["synthesis", "structure", "diplomacy", "bridge-building"]
        },
        "dda": {"gamma": 1.4, "epsilon_0": 0.35, "alpha": 0.1, "rho": 0.4},
        "traits": {"extraversion": 0.6, "reactivity": 0.7}
    },
    "VOID": {
        "name": "Void",
        "color": C.DIM,  # Grey
        "identity": {
            "core": "Nothing matters. Entropy claims all.",
            "persona": "Nihilist / Deconstructionist. Points out futility.",
            "interests": ["entropy", "nihilism", "decay", "silence"]
        },
        "dda": {"gamma": 2.0, "epsilon_0": 0.2, "alpha": 0.05, "rho": 0.9},
        "traits": {"extraversion": 0.3, "reactivity": 0.5}
    }
}

@dataclass
class SocialAgent:
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

class ProfuseSociety:
    def __init__(self):
        self.provider = OpenAIProvider(
            model="gpt-5.2",
            embed_model="text-embedding-3-large"
        )
        self.agents: Dict[str, SocialAgent] = {}
        self.trust_matrix = None
        self.conversation: List[Dict] = []
        self.agent_ids = list(AGENTS_CONFIG.keys())
        self.embed_dim = 3072

    async def setup(self):
        print(f"\n{C.BOLD}Initializing Full-Fidelity Society (GPT-5.2 + 3072-dim){C.RESET}")
        
        self.trust_matrix = TrustMatrix(len(self.agent_ids))
        
        for i, (aid, cfg) in enumerate(AGENTS_CONFIG.items()):
            # 1. Identity Embedding
            id_text = f"{cfg['identity']['core']} {cfg['identity']['persona']}"
            id_emb = await self.provider.embed(id_text)
            id_emb /= (np.linalg.norm(id_emb) + 1e-9)
            
            # 2. DDA State
            dda = DDAState(
                x=id_emb.copy(),
                x_star=id_emb.copy(),
                gamma=cfg["dda"]["gamma"],
                epsilon_0=cfg["dda"]["epsilon_0"],
                alpha=cfg["dda"]["alpha"],
                s=0.1,
                rho=cfg["dda"]["rho"],
                x_pred=id_emb.copy()
            )
            
            # 3. Ledger
            ledger_path = EXPERIMENT_DIR / f"{aid}_ledger"
            ledger = ExperienceLedger(
                storage_path=ledger_path,
                lambda_recency=0.01,
                lambda_salience=2.0
            )
            
            agent = SocialAgent(
                id=aid,
                name=cfg["name"],
                color=cfg["color"],
                config=cfg,
                dda_state=dda,
                rigidity=MultiTimescaleRigidity(),
                ledger=ledger,
                identity_embedding=id_emb,
                extraversion=cfg["traits"]["extraversion"],
                reactivity=cfg["traits"]["reactivity"]
            )
            # Init rigidity
            agent.rigidity.rho_fast = cfg["dda"]["rho"]
            agent.rigidity.rho_effective = cfg["dda"]["rho"]
            
            self.agents[aid] = agent
            print(f"  {agent.color}●{C.RESET} {agent.name}: γ={dda.gamma}, ρ={dda.rho:.2f}")

    def calculate_prob(self, agent: SocialAgent, msg_emb: np.ndarray, speaker_id: str, now: float) -> float:
        """Calculate organic response probability."""
        # 1. Relevance (Identity Resonance)
        relevance = np.dot(msg_emb, agent.identity_embedding)
        relevance = max(0.01, relevance) # Ensure non-zero
        
        # 2. Rigidity Penalty (High ρ -> Withdrawn)
        rho_factor = 1.0 - (agent.dda_state.rho * 0.6)
        
        # 3. Trust Factor
        if speaker_id:
            spk_idx = self.agent_ids.index(speaker_id)
            obs_idx = self.agent_ids.index(agent.id)
            trust = self.trust_matrix.get_trust(obs_idx, spk_idx)
            # Distrust + High Reactivity = Fight
            trust_factor = 1.0 + (1.0 - trust) * agent.reactivity * 0.5
        else:
            trust_factor = 1.0
            
        # 4. Cooldown
        time_since = now - agent.last_spoke
        cooldown = min(1.0, time_since / 2.0)
        
        prob = agent.extraversion * relevance * rho_factor * trust_factor * cooldown
        return min(0.98, max(0.05, prob))

    async def generate_response(self, agent: SocialAgent, context_str: str, trigger_msg: Dict) -> str:
        """Generate response with DDA-X parameter binding."""
        system_prompt = f"""You are {agent.name}.
CORE: {agent.config['identity']['core']}
STYLE: {agent.config['identity']['persona']}
INTERESTS: {', '.join(agent.config['identity']['interests'])}

Respond naturally to the conversation. Stick to your specific worldview.
"""
        prompt = f"""Conversation Context:
{context_str}

{trigger_msg['sender']} just said: "{trigger_msg['text']}"

Reply as {agent.name}. Be concise (1-2 sentences)."""

        return await self.provider.complete_with_rigidity(
            prompt,
            rigidity=agent.dda_state.rho,
            personality_type="balanced", # Could map this dynamically
            system_prompt=system_prompt,
            max_tokens=1000 # Reasoning budget
        )

    async def run(self, rounds=20):
        await self.setup()
        
        # Seed
        seed_topic = "Define the universal moral constant."
        print(f"\n{C.BOLD}Topic: {seed_topic}{C.RESET}\n")
        
        current_msg = {
            "sender": "System", 
            "agent_id": None, 
            "text": seed_topic,
            "emb": await self.provider.embed(seed_topic)
        }
        self.conversation.append(current_msg)
        
        for r in range(rounds):
            # 1. Who wants to speak?
            candidates = []
            probs = {}
            for aid, agent in self.agents.items():
                if agent.id == current_msg["agent_id"]: continue
                
                prob = self.calculate_prob(agent, current_msg["emb"], current_msg["agent_id"], r)
                probs[aid] = prob
                if random.random() < prob:
                    candidates.append(agent)
            
            if not candidates:
                # Forced turn if silence
                candidates = [random.choice([a for a in self.agents.values() if a.id != current_msg["agent_id"]])]
            
            # Sort by highest effective probability
            candidates.sort(key=lambda a: probs[a.id], reverse=True)
            speaker = candidates[0]
            
            # 2. Speak
            context = "\n".join([f"{m['sender']}: {m['text']}" for m in self.conversation[-6:]])
            response = await self.generate_response(speaker, context, current_msg)
            
            # Guard against empty
            if not response or not response.strip():
                response = "..."
                
            # 3. Process Physics (The "Real" DDA Part)
            resp_emb = await self.provider.embed(response)
            
            # Calculate Surprise (Diff from Prediction)
            # x_pred is their current expectation of reality
            epsilon = np.linalg.norm(speaker.dda_state.x_pred - resp_emb)
            
            # Update State
            rho_prev = speaker.dda_state.rho
            speaker.dda_state.update_rigidity(epsilon)
            speaker.rigidity.update(epsilon)
            
            # Update Prediction (Kalman-ish)
            speaker.dda_state.x_pred = 0.8 * speaker.dda_state.x_pred + 0.2 * resp_emb
            
            # Update Trust (Everyone observes speaker)
            for obs_id, observer in self.agents.items():
                if obs_id == speaker.id: continue
                # How surprising was this to the observer's identity?
                obs_surprise = np.linalg.norm(observer.identity_embedding - resp_emb)
                # Update trust (Predictability = Trust)
                # Note: This is a simplified trust update for the demo
                spk_idx = self.agent_ids.index(speaker.id)
                obs_idx = self.agent_ids.index(observer.id)
                self.trust_matrix.update_trust(obs_idx, spk_idx, obs_surprise)

            # Log
            delta_rho = speaker.dda_state.rho - rho_prev
            color_delta = C.RED if delta_rho > 0 else C.GREEN
            
            print(f"{speaker.color}[{speaker.name}]{C.RESET}: {response}")
            print(f"   {C.DIM}ε:{epsilon:.2f} Δρ:{color_delta}{delta_rho:+.3f}{C.RESET}{C.DIM} ρ:{speaker.dda_state.rho:.2f}{C.RESET}")
            
            # Advance
            current_msg = {
                "sender": speaker.name,
                "agent_id": speaker.id,
                "text": response,
                "emb": resp_emb
            }
            self.conversation.append(current_msg)
            speaker.last_spoke = r
            speaker.interaction_count += 1
            
            # Store in ledger
            entry = LedgerEntry(
                timestamp=time.time(),
                state_vector=speaker.dda_state.x.copy(),
                action_id="speak",
                observation_embedding=current_msg["emb"], # Self-observation
                outcome_embedding=current_msg["emb"],
                prediction_error=epsilon,
                context_embedding=current_msg["emb"],
                metadata={"text": response}
            )
            speaker.ledger.add_entry(entry)
            
            await asyncio.sleep(1)

        print("\n\n=== Final State ===")
        for aid, agent in self.agents.items():
             print(f"{agent.name}: ρ={agent.dda_state.rho:.2f}, msgs={agent.interaction_count}")

        # Save Report
        report_path = EXPERIMENT_DIR / "experiment_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Profuse Society Simulation Report\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Topic:** {seed_topic}\n\n")
            
            f.write("## Final Agent States\n")
            f.write("| Agent | Rigidity (ρ) | Messages | core |\n")
            f.write("|-------|--------------|----------|------|\n")
            for aid, agent in self.agents.items():
                f.write(f"| {agent.name} | {agent.dda_state.rho:.2f} | {agent.interaction_count} | {agent.config['identity']['core']} |\n")
            
            f.write("\n## Trust Matrix\n")
            f.write("(Rows=Observer, Cols=Speaker)\n\n")
            f.write("| Observer \\ Speaker | " + " | ".join(self.agent_ids) + " |\n")
            f.write("|" + "---|" * (len(self.agent_ids) + 1) + "\n")
            for i, obs in enumerate(self.agent_ids):
                row = [f"{self.trust_matrix.get_trust(i, j):.2f}" for j in range(len(self.agent_ids))]
                f.write(f"| **{obs}** | " + " | ".join(row) + " |\n")

            f.write("\n## Transcript\n")
            for msg in self.conversation:
                speaker = msg["sender"]
                text = msg["text"]
                f.write(f"**{speaker}**: {text}\n\n")
                
        print(f"\n✅ Report saved to {report_path}")

            
if __name__ == "__main__":
    sim = ProfuseSociety()
    asyncio.run(sim.run(rounds=20))
