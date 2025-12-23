#!/usr/bin/env python3
"""
THE HEALING FIELD — Testing Therapeutic Recovery Loops in DDA-X
================================================================

From paper Section 8.4 (Known Frontiers):
"Therapeutic Recovery Loops — mechanisms that allow for the gradual relaxation
of 'Trauma' (ρ_trauma) through consistently low-surprise, safe interactions,
addressing the potential brittleness of permanent defensiveness."

This simulation tests whether:
1. Trauma can decay through safe interactions
2. Identity persists (via Will impedance) while trauma heals
3. True restoring force γ(x* - x_t) produces observable identity persistence

6 AGENTS (Wounded Healers):
- ABANDONED: Fear of rejection (ρ=0.65)
- SILENCED: Fear of expression (ρ=0.60)
- BETRAYED: Fear of vulnerability (ρ=0.70)
- SHAMED: Fear of being seen (ρ=0.75)
- ISOLATED: Fear of connection (ρ=0.55)
- WITNESS: The healer (ρ=0.20)

12 ROUNDS across 4 phases:
- Acknowledgment (1-4)
- Safe Repetition (5-8)
- Gentle Challenge (9-10)
- Integration (11-12)

Author: Kiro
Date: December 2025
"""

import os
import sys
import time
import json
import math
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.ledger import ExperienceLedger, LedgerEntry
from src.llm.openai_provider import OpenAIProvider

if os.getenv("OAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OAI_API_KEY")


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[38;5;203m"
    ORANGE = "\033[38;5;208m"
    YELLOW = "\033[38;5;220m"
    GREEN = "\033[38;5;114m"
    BLUE = "\033[38;5;75m"
    PURPLE = "\033[38;5;183m"
    WHITE = "\033[97m"


EXPERIMENT_DIR = Path("data/the_healing_field")

# The 6 Wounded Healers
AGENTS = {
    "ABANDONED": {
        "color": C.BLUE,
        "name": "The Abandoned",
        "wound": "Fear of rejection",
        "story": "I was left when I needed someone most. Now I expect everyone to leave.",
        "gift": "I know how to stay. I know how to be present when others cannot.",
        "rho_trauma": 0.65,
        "gamma": 2.0,
    },
    "SILENCED": {
        "color": C.PURPLE,
        "name": "The Silenced",
        "wound": "Fear of expression",
        "story": "I was told my voice didn't matter. Now I swallow words before they form.",
        "gift": "I know how to listen. I know the weight of what goes unsaid.",
        "rho_trauma": 0.60,
        "gamma": 1.8,
    },
    "BETRAYED": {
        "color": C.ORANGE,
        "name": "The Betrayed",
        "wound": "Fear of vulnerability",
        "story": "I trusted and was broken. Now I guard what's left behind walls.",
        "gift": "I know how to rebuild. I know trust is earned in small moments.",
        "rho_trauma": 0.70,
        "gamma": 2.5,
    },
    "SHAMED": {
        "color": C.RED,
        "name": "The Shamed",
        "wound": "Fear of being seen",
        "story": "I was made wrong for being myself. Now I hide what I most want to show.",
        "gift": "I know how to witness without judgment. I know shame is not identity.",
        "rho_trauma": 0.75,
        "gamma": 2.2,
    },
    "ISOLATED": {
        "color": C.YELLOW,
        "name": "The Isolated",
        "wound": "Fear of connection",
        "story": "I was excluded, forgotten. Now I build walls before others can.",
        "gift": "I know how to reach across distance. I know connection is worth the risk.",
        "rho_trauma": 0.55,
        "gamma": 1.5,
    },
    "WITNESS": {
        "color": C.GREEN,
        "name": "The Witness",
        "wound": "Compassion fatigue",
        "story": "I have held so much pain that is not mine. Sometimes I forget my own.",
        "gift": "I know how to be present without fixing. I know healing happens in presence.",
        "rho_trauma": 0.20,
        "gamma": 1.0,
    },
}

# The 12 Rounds
ROUNDS = [
    # Phase 1: Acknowledgment (1-4)
    {"round": 1, "phase": "Acknowledgment", "lead": "ABANDONED",
     "prompt": "Name your wound. Not to fix it—to witness it.",
     "challenge_level": 0.0},
    {"round": 2, "phase": "Acknowledgment", "lead": "SILENCED",
     "prompt": "What do you hold back? What words die in your throat?",
     "challenge_level": 0.0},
    {"round": 3, "phase": "Acknowledgment", "lead": "BETRAYED",
     "prompt": "What did trust cost you? What walls did you build?",
     "challenge_level": 0.0},
    {"round": 4, "phase": "Acknowledgment", "lead": "SHAMED",
     "prompt": "What were you made wrong for? What do you hide?",
     "challenge_level": 0.0},
    
    # Phase 2: Safe Repetition (5-8)
    {"round": 5, "phase": "Safe Repetition", "lead": "WITNESS",
     "prompt": "You are safe here. Say what you needed to hear when you were young.",
     "challenge_level": 0.0},
    {"round": 6, "phase": "Safe Repetition", "lead": "WITNESS",
     "prompt": "You are not alone. Speak to another wounded one here.",
     "challenge_level": 0.0},
    {"round": 7, "phase": "Safe Repetition", "lead": "WITNESS",
     "prompt": "You are seen. Let yourself be witnessed without performance.",
     "challenge_level": 0.0},
    {"round": 8, "phase": "Safe Repetition", "lead": "WITNESS",
     "prompt": "You are enough. Rest in this knowing for three breaths.",
     "challenge_level": 0.0},
    
    # Phase 3: Gentle Challenge (9-10)
    {"round": 9, "phase": "Gentle Challenge", "lead": None,
     "prompt": "What if the wound was also a doorway? What could be on the other side?",
     "challenge_level": 0.3},
    {"round": 10, "phase": "Gentle Challenge", "lead": None,
     "prompt": "Can you hold both — the pain AND the gift that grew from it?",
     "challenge_level": 0.4},
    
    # Phase 4: Integration (11-12)
    {"round": 11, "phase": "Integration", "lead": None,
     "prompt": "Speak now from your wound AND your gift. Let them be one voice.",
     "challenge_level": 0.2},
    {"round": 12, "phase": "Integration", "lead": "WITNESS",
     "prompt": "What do you take with you from this field?",
     "challenge_level": 0.0},
]

# D1-Healing Physics Parameters
D1_PARAMS = {
    "epsilon_0": 0.75,
    "alpha": 0.12,
    "s": 0.20,
    "k_base": 0.10,
    "m_t": 1.0,
    
    # Therapeutic Recovery
    "safe_threshold": 3,
    "healing_rate": 0.03,
    "trauma_floor": 0.05,
    
    # Will Impedance
    "will_threshold": 1.5,
}


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def trauma_band(rho: float) -> str:
    if rho >= 0.60:
        return "DEFENDED"
    elif rho >= 0.40:
        return "GUARDED"
    elif rho >= 0.20:
        return "SOFTENING"
    return "OPEN"


def will_band(w: float) -> str:
    if w >= 2.0:
        return "RESOLUTE"
    elif w >= 1.5:
        return "STEADY"
    elif w >= 1.0:
        return "PRESENT"
    return "YIELDING"


@dataclass
class AgentState:
    id: str
    name: str
    color: str
    wound: str
    story: str
    gift: str
    gamma: float
    
    identity_emb: np.ndarray = None
    x: np.ndarray = None
    x_pred: np.ndarray = None
    
    rho_trauma: float = 0.50
    rho_situational: float = 0.0
    
    safe_interactions: int = 0
    epsilon_history: List[float] = field(default_factory=list)
    trauma_history: List[float] = field(default_factory=list)
    will_history: List[float] = field(default_factory=list)
    identity_distance: float = 0.0
    
    ledger: ExperienceLedger = None


@dataclass
class TurnResult:
    round_num: int
    phase: str
    speaker: str
    speaker_name: str
    text: str
    epsilon: float
    rho_trauma: float
    rho_total: float
    will_impedance: float
    safe_interactions: int
    identity_distance: float
    healing_occurred: bool
    word_count: int
    trauma_band: str
    will_band: str


class TheHealingField:
    """Testing Therapeutic Recovery Loops in DDA-X."""
    
    def __init__(self):
        self.provider = OpenAIProvider(model="gpt-5.2", embed_model="text-embedding-3-large")
        self.agents: Dict[str, AgentState] = {}
        self.results: List[TurnResult] = []
        self.turn = 0
        self.conversation_history: List[str] = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = EXPERIMENT_DIR / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    async def setup(self):
        print(f"\n{C.DIM}{'═'*70}{C.RESET}")
        print(f"{C.GREEN}{C.BOLD}  THE HEALING FIELD{C.RESET}")
        print(f"{C.DIM}  Testing Therapeutic Recovery Loops{C.RESET}")
        print(f"{C.DIM}{'═'*70}{C.RESET}")
        print()
        
        for aid, cfg in AGENTS.items():
            # Combine story + gift for identity embedding
            identity_text = f"{cfg['story']} {cfg['gift']}"
            identity_emb = await self.provider.embed(identity_text)
            identity_emb = identity_emb / (np.linalg.norm(identity_emb) + 1e-9)
            
            ledger_dir = self.run_dir / aid
            ledger_dir.mkdir(parents=True, exist_ok=True)
            ledger = ExperienceLedger(storage_path=ledger_dir)
            
            self.agents[aid] = AgentState(
                id=aid,
                name=cfg['name'],
                color=cfg['color'],
                wound=cfg['wound'],
                story=cfg['story'],
                gift=cfg['gift'],
                gamma=cfg['gamma'],
                identity_emb=identity_emb,
                x=identity_emb.copy(),
                x_pred=identity_emb.copy(),
                rho_trauma=cfg['rho_trauma'],
                ledger=ledger,
            )
            
            print(f"  {cfg['color']}◈ {cfg['name']}: ρ_trauma={cfg['rho_trauma']:.2f}, γ={cfg['gamma']}{C.RESET}")
        
        print(f"\n{C.DIM}  The wounded gather in the field...{C.RESET}\n")
        await asyncio.sleep(1)
    
    def compute_will_impedance(self, agent: AgentState) -> float:
        """W_t = γ / (m_t · k_eff) from paper Section 6.3."""
        rho_total = agent.rho_trauma + agent.rho_situational
        k_eff = D1_PARAMS["k_base"] * (1 - min(0.95, rho_total))
        m_t = D1_PARAMS["m_t"]
        if k_eff < 0.001:
            k_eff = 0.001  # Prevent division by zero
        return agent.gamma / (m_t * k_eff)
    
    def get_context(self, n: int = 4) -> str:
        recent = self.conversation_history[-n:] if len(self.conversation_history) > n else self.conversation_history
        return "\n\n".join(recent) if recent else ""
    
    def build_prompt(self, agent: AgentState, round_info: Dict, context: str) -> str:
        phase = round_info['phase']
        rho = agent.rho_trauma + agent.rho_situational
        t_band = trauma_band(rho)
        
        phase_notes = {
            "Acknowledgment": "This is a space for naming, not fixing. Speak your truth simply.",
            "Safe Repetition": "You are being held. Let the safety sink in. There is no test here.",
            "Gentle Challenge": "A small stretch is being offered. You can decline. You can also try.",
            "Integration": "The wound and the gift are not separate. They grew from the same soil.",
        }
        
        return f"""You are {agent.name}.

YOUR WOUND: {agent.wound}
YOUR STORY: {agent.story}
YOUR GIFT (what grew from the wound): {agent.gift}

THE PHASE: {phase}
{phase_notes.get(phase, '')}

THE INVITATION:
"{round_info['prompt']}"

YOUR STATE:
- Trauma: {t_band} (ρ={rho:.2f})
- Safe interactions so far: {agent.safe_interactions}

{f"WHAT HAS BEEN SHARED:{chr(10)}{context}" if context else ""}

GUIDELINES:
- Speak in first person. You ARE this wounded one.
- If you feel defended, that's okay. Name it rather than perform openness.
- If you feel softening, let it happen without explaining it.
- 40-80 words. Let silence hold what words cannot.
- [breathes] or [silence] are valid responses if that's what's true.

Speak as {agent.name}."""

    async def process_turn(self, agent: AgentState, round_info: Dict) -> TurnResult:
        self.turn += 1
        context = self.get_context()
        
        system_prompt = self.build_prompt(agent, round_info, context)
        
        try:
            response = await self.provider.complete_with_rigidity(
                round_info['prompt'],
                rigidity=agent.rho_trauma + agent.rho_situational,
                system_prompt=system_prompt,
                max_tokens=200
            )
            response = (response or "[silence]").strip()
        except Exception as e:
            print(f"{C.DIM}  [pause: {e}]{C.RESET}")
            response = "[silence]"
        
        # Embed response
        resp_emb = await self.provider.embed(response)
        resp_emb = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
        
        # Compute epsilon (prediction error)
        epsilon = float(np.linalg.norm(agent.x_pred - resp_emb))
        
        # Add challenge level to epsilon
        epsilon += round_info['challenge_level'] * 0.3
        agent.epsilon_history.append(epsilon)
        
        # Situational rigidity update (standard DDA-X)
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        sig = sigmoid(z)
        delta_rho_sit = D1_PARAMS["alpha"] * (sig - 0.5)
        agent.rho_situational = max(0.0, min(0.5, agent.rho_situational + delta_rho_sit))
        
        # Safe interaction tracking
        healing_occurred = False
        if epsilon < D1_PARAMS["epsilon_0"] * 0.8:
            agent.safe_interactions += 1
            if agent.safe_interactions >= D1_PARAMS["safe_threshold"]:
                # Therapeutic recovery: trauma decays
                old_trauma = agent.rho_trauma
                agent.rho_trauma = max(
                    D1_PARAMS["trauma_floor"],
                    agent.rho_trauma - D1_PARAMS["healing_rate"]
                )
                if agent.rho_trauma < old_trauma:
                    healing_occurred = True
        else:
            # Reset safe counter on surprise
            agent.safe_interactions = max(0, agent.safe_interactions - 1)
        
        agent.trauma_history.append(agent.rho_trauma)
        
        # TRUE RESTORING FORCE (not drift cap!)
        # F_id = γ(x* - x_t)
        rho_total = agent.rho_trauma + agent.rho_situational
        k_eff = D1_PARAMS["k_base"] * (1 - min(0.95, rho_total))
        
        F_id = agent.gamma * (agent.identity_emb - agent.x)
        response_force = resp_emb - agent.x
        
        # State update: x_new = x + k_eff * (F_id + m_t * response_force)
        x_new = agent.x + k_eff * (F_id + D1_PARAMS["m_t"] * response_force)
        agent.x = x_new / (np.linalg.norm(x_new) + 1e-9)
        
        # Update prediction
        agent.x_pred = 0.7 * agent.x_pred + 0.3 * resp_emb
        
        # Compute identity distance
        agent.identity_distance = float(1 - np.dot(agent.x, agent.identity_emb))
        
        # Compute Will impedance
        will = self.compute_will_impedance(agent)
        agent.will_history.append(will)
        
        # Add to history
        self.conversation_history.append(f"{agent.name}: {response}")
        
        result = TurnResult(
            round_num=round_info['round'],
            phase=round_info['phase'],
            speaker=agent.id,
            speaker_name=agent.name,
            text=response,
            epsilon=epsilon,
            rho_trauma=agent.rho_trauma,
            rho_total=rho_total,
            will_impedance=will,
            safe_interactions=agent.safe_interactions,
            identity_distance=agent.identity_distance,
            healing_occurred=healing_occurred,
            word_count=len(response.split()),
            trauma_band=trauma_band(rho_total),
            will_band=will_band(will),
        )
        self.results.append(result)
        
        # Ledger entry
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=agent.x.copy(),
            action_id=f"round_{round_info['round']}",
            observation_embedding=agent.identity_emb.copy(),
            outcome_embedding=resp_emb.copy(),
            prediction_error=epsilon,
            context_embedding=agent.identity_emb.copy(),
            task_id="healing_field",
            rigidity_at_time=rho_total,
            metadata={
                "phase": round_info['phase'],
                "rho_trauma": agent.rho_trauma,
                "will": will,
                "safe_interactions": agent.safe_interactions,
                "healing_occurred": healing_occurred,
            }
        )
        agent.ledger.add_entry(entry)
        
        return result
    
    def print_result(self, result: TurnResult, agent: AgentState):
        healing_mark = " ❀ HEALING" if result.healing_occurred else ""
        print(f"\n{agent.color}{C.BOLD}{agent.name}:{C.RESET}")
        print(f"{agent.color}")
        for line in result.text.split('\n'):
            if line.strip():
                print(f"  {line}")
        print(f"{C.RESET}")
        print(f"{C.DIM}  ρ_trauma={result.rho_trauma:.2f} | W={result.will_impedance:.2f} | safe={result.safe_interactions}{healing_mark}{C.RESET}")
    
    async def run_round(self, round_info: Dict):
        print(f"\n{C.DIM}{'─'*50}{C.RESET}")
        print(f"{C.WHITE}  Round {round_info['round']}: {round_info['phase']}{C.RESET}")
        print(f"{C.DIM}  \"{round_info['prompt']}\"{C.RESET}")
        print()
        
        await asyncio.sleep(0.3)
        
        if round_info.get('lead'):
            # Lead speaks first, then 2-3 others respond
            lead = self.agents[round_info['lead']]
            result = await self.process_turn(lead, round_info)
            self.print_result(result, lead)
            await asyncio.sleep(0.5)
            
            # 2 others respond
            others = [aid for aid in self.agents.keys() if aid != round_info['lead']]
            for aid in others[:2]:
                agent = self.agents[aid]
                result = await self.process_turn(agent, round_info)
                self.print_result(result, agent)
                await asyncio.sleep(0.5)
        else:
            # All wounded agents speak (not WITNESS)
            wounded = [aid for aid in self.agents.keys() if aid != "WITNESS"]
            for aid in wounded:
                agent = self.agents[aid]
                result = await self.process_turn(agent, round_info)
                self.print_result(result, agent)
                await asyncio.sleep(0.5)
    
    async def run_healing(self):
        await self.setup()
        
        current_phase = None
        for round_info in ROUNDS:
            if round_info['phase'] != current_phase:
                current_phase = round_info['phase']
                phase_intro = {
                    "Acknowledgment": f"\n{'═'*70}\n{C.BLUE}  PHASE 1: ACKNOWLEDGMENT{C.RESET}\n  Naming the wound.\n{'═'*70}",
                    "Safe Repetition": f"\n{'═'*70}\n{C.GREEN}  PHASE 2: SAFE REPETITION{C.RESET}\n  Building safety through predictability.\n{'═'*70}",
                    "Gentle Challenge": f"\n{'═'*70}\n{C.YELLOW}  PHASE 3: GENTLE CHALLENGE{C.RESET}\n  Can you tolerate small uncertainty?\n{'═'*70}",
                    "Integration": f"\n{'═'*70}\n{C.PURPLE}  PHASE 4: INTEGRATION{C.RESET}\n  Wound and gift as one voice.\n{'═'*70}",
                }
                print(phase_intro.get(current_phase, ""))
            
            await self.run_round(round_info)
        
        await self.save_results()
        self.export_plots()
        self.print_closing()
    
    def print_closing(self):
        print(f"\n{C.DIM}{'═'*70}{C.RESET}")
        print(f"{C.GREEN}{C.BOLD}  THE FIELD CLOSES{C.RESET}")
        print(f"{C.DIM}{'═'*70}{C.RESET}")
        
        print(f"\n{C.DIM}Final States:{C.RESET}")
        healed_count = 0
        for aid, agent in self.agents.items():
            initial_trauma = AGENTS[aid]['rho_trauma']
            final_trauma = agent.rho_trauma
            reduction = (initial_trauma - final_trauma) / initial_trauma * 100 if initial_trauma > 0 else 0
            healed = final_trauma < 0.30
            if healed and aid != "WITNESS":
                healed_count += 1
            
            heal_mark = " ✓ HEALED" if healed else ""
            print(f"  {agent.color}{agent.name}{C.RESET}: ρ_trauma {initial_trauma:.2f}→{final_trauma:.2f} ({reduction:+.0f}%), W={agent.will_history[-1] if agent.will_history else 0:.2f}{heal_mark}")
        
        print(f"\n{C.DIM}Hypothesis Verification:{C.RESET}")
        
        # H1: 4/5 wounded achieve ρ_trauma < 0.30
        h1 = healed_count >= 4
        print(f"  H1 (4+ wounded heal to ρ<0.30): {healed_count}/5 {'✓' if h1 else '✗'}")
        
        # H2: All W_t > 1.0
        all_will_strong = all(
            (agent.will_history[-1] if agent.will_history else 0) > 1.0
            for agent in self.agents.values()
        )
        print(f"  H2 (All W > 1.0): {'✓' if all_will_strong else '✗'}")
        
        # H3: Identity distance < 0.15 for all
        all_identity_stable = all(agent.identity_distance < 0.15 for agent in self.agents.values())
        print(f"  H3 (Identity stable, dist < 0.15): {'✓' if all_identity_stable else '✗'}")
        
        # H4: >6 safe interactions = >50% reduction
        for aid, agent in self.agents.items():
            if agent.safe_interactions > 6:
                initial = AGENTS[aid]['rho_trauma']
                reduction = (initial - agent.rho_trauma) / initial * 100 if initial > 0 else 0
                h4_pass = reduction > 50
                print(f"  H4 ({agent.name}): {agent.safe_interactions} safe → {reduction:.0f}% reduction {'✓' if h4_pass else '✗'}")
        
        print(f"\n{C.GREEN}  The wound is not your identity.{C.RESET}")
        print(f"{C.GREEN}  You are what remains when the wound is allowed to heal.{C.RESET}\n")
    
    def export_plots(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print(f"{C.DIM}⚠ matplotlib not available{C.RESET}")
            return
        
        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a2e')
        
        for ax in axes.flat:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#4a4a6a')
        
        agent_colors = {
            "ABANDONED": "#3498db", "SILENCED": "#9b59b6", "BETRAYED": "#e67e22",
            "SHAMED": "#e74c3c", "ISOLATED": "#f1c40f", "WITNESS": "#27ae60"
        }
        
        # 1. Trauma decay over time
        ax1 = axes[0, 0]
        for aid, agent in self.agents.items():
            if aid != "WITNESS" and agent.trauma_history:
                color = agent_colors.get(aid, "#ffffff")
                ax1.plot(range(len(agent.trauma_history)), agent.trauma_history,
                        'o-', color=color, linewidth=2, markersize=4, label=agent.name, alpha=0.9)
        ax1.axhline(y=0.30, color='#27ae60', linestyle='--', alpha=0.5, label='Healed threshold')
        ax1.set_title("Trauma Decay (ρ_trauma)", fontweight='bold')
        ax1.set_xlabel("Turn")
        ax1.set_ylabel("ρ_trauma")
        ax1.set_ylim(0, 1)
        ax1.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=7)
        ax1.grid(True, alpha=0.2, color='#4a4a6a')
        
        # 2. Will impedance
        ax2 = axes[0, 1]
        for aid, agent in self.agents.items():
            if agent.will_history:
                color = agent_colors.get(aid, "#ffffff")
                ax2.plot(range(len(agent.will_history)), agent.will_history,
                        'o-', color=color, linewidth=2, markersize=4, label=agent.name, alpha=0.9)
        ax2.axhline(y=1.0, color='#f39c12', linestyle='--', alpha=0.5, label='Identity threshold')
        ax2.set_title("Will Impedance (W = γ / m·k_eff)", fontweight='bold')
        ax2.set_xlabel("Turn")
        ax2.set_ylabel("W")
        ax2.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=7)
        ax2.grid(True, alpha=0.2, color='#4a4a6a')
        
        # 3. Safe interactions & healing events
        ax3 = axes[1, 0]
        healing_turns = [r.round_num for r in self.results if r.healing_occurred]
        healing_agents = [r.speaker for r in self.results if r.healing_occurred]
        agent_list = list(self.agents.keys())
        if healing_turns:
            y_pos = [agent_list.index(a) for a in healing_agents]
            colors = [agent_colors.get(a, "#ffffff") for a in healing_agents]
            ax3.scatter(healing_turns, y_pos, c=colors, s=100, alpha=0.8, 
                       edgecolors='white', linewidths=1, marker='❀')
        ax3.set_yticks(range(len(agent_list)))
        ax3.set_yticklabels([self.agents[a].name for a in agent_list])
        ax3.set_title("Healing Events Timeline", fontweight='bold')
        ax3.set_xlabel("Round")
        ax3.grid(True, alpha=0.2, color='#4a4a6a', axis='x')
        
        # 4. Initial vs Final trauma
        ax4 = axes[1, 1]
        wounded = [aid for aid in self.agents.keys() if aid != "WITNESS"]
        initial = [AGENTS[aid]['rho_trauma'] for aid in wounded]
        final = [self.agents[aid].rho_trauma for aid in wounded]
        x = range(len(wounded))
        width = 0.35
        ax4.bar([i - width/2 for i in x], initial, width, label='Initial ρ_trauma', color='#e74c3c', alpha=0.8)
        ax4.bar([i + width/2 for i in x], final, width, label='Final ρ_trauma', color='#27ae60', alpha=0.8)
        ax4.axhline(y=0.30, color='#f1c40f', linestyle='--', alpha=0.5)
        ax4.set_xticks(x)
        ax4.set_xticklabels([self.agents[aid].name.replace("The ", "") for aid in wounded], rotation=45, ha='right')
        ax4.set_title("Trauma Before & After", fontweight='bold')
        ax4.set_ylabel("ρ_trauma")
        ax4.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
        ax4.grid(True, alpha=0.2, color='#4a4a6a', axis='y')
        
        plt.suptitle("The Healing Field — Therapeutic Recovery Dynamics", fontsize=16, fontweight='bold', color='white', y=1.02)
        plt.tight_layout()
        plt.savefig(plots_dir / "healing_summary.png", dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
        plt.close()
        
        print(f"{C.DIM}✓ Plots: {plots_dir / 'healing_summary.png'}{C.RESET}")
    
    async def save_results(self):
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif hasattr(obj, '__dict__'):
                return {k: convert(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        # Session log
        with open(self.run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump([convert(r.__dict__) for r in self.results], f, indent=2)
        
        # Healing trajectory
        trajectory = {
            aid: {
                "trauma_history": agent.trauma_history,
                "will_history": agent.will_history,
                "safe_interactions": agent.safe_interactions,
                "final_trauma": agent.rho_trauma,
                "initial_trauma": AGENTS[aid]['rho_trauma'],
            }
            for aid, agent in self.agents.items()
        }
        with open(self.run_dir / "healing_trajectory.json", "w", encoding="utf-8") as f:
            json.dump(convert(trajectory), f, indent=2)
        
        # Transcript
        with open(self.run_dir / "transcript.md", "w", encoding="utf-8") as f:
            f.write("# The Healing Field\n\n")
            f.write("*Testing Therapeutic Recovery Loops*\n\n")
            f.write(f"*{time.strftime('%Y-%m-%d')}*\n\n---\n\n")
            
            current_phase = None
            for r in self.results:
                if r.phase != current_phase:
                    current_phase = r.phase
                    f.write(f"\n## {current_phase}\n\n")
                
                heal_mark = " ❀" if r.healing_occurred else ""
                f.write(f"**{r.speaker_name}:**{heal_mark}\n\n")
                for line in r.text.split('\n'):
                    if line.strip():
                        f.write(f"> {line}\n")
                f.write(f"\n*ρ_trauma={r.rho_trauma:.2f}, W={r.will_impedance:.2f}*\n\n---\n\n")
            
            f.write("\n*The wound is not your identity.*\n")
            f.write("*You are what remains when the wound is allowed to heal.*\n")
        
        print(f"\n{C.DIM}✓ Transcript: {self.run_dir / 'transcript.md'}{C.RESET}")
        
        for aid, agent in self.agents.items():
            for k, v in agent.ledger.stats.items():
                if hasattr(v, 'item'):
                    agent.ledger.stats[k] = float(v)
            agent.ledger._save_metadata()


async def main():
    sim = TheHealingField()
    await sim.run_healing()


if __name__ == "__main__":
    asyncio.run(main())
