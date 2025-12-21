#!/usr/bin/env python3
"""
THE SOCIETY — DDA-X Discord-Style Multi-Agent Simulation
==========================================================

A persistent society of agents in a fast-moving chat room.
Each turn: one agent speaks, one message only, ledger-driven state.

Features:
- Single-turn constraints (one message per call)
- Identity field exposure through behavior
- Trust as predictability: T = 1/(1+Σε)
- Wound cooldowns to prevent runaway ε
- Identity drift cap per turn
- Refusal palette (boundary, redirect, partial, delay)
- Formal D1 physics with shock-scaled Δρ

Agents:
- VERITY: Truth field (no deception, direct, wound=silence)
- PIXEL: Chaos field (playful, provocative, wound=being ignored)
- SPARK: Fire field (passionate, confrontational, wound=dismissal)
- ORACLE: Mystery field (cryptic, observant, wound=being exposed)

Author: Kiro (AI Assistant)
Date: December 2025
"""

import os
import sys
import time
import json
import math
import asyncio
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.llm.openai_provider import OpenAIProvider

if os.getenv("OAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OAI_API_KEY")


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"


EXPERIMENT_DIR = Path("data/society")

# Agent configurations
AGENTS = {
    "VERITY": {
        "color": C.CYAN,
        "core": "I speak truth even when it costs me. Deception is a line I will not cross.",
        "persona": "Thoughtful, direct; avoids soothing falsehoods; offers bounded help.",
        "wound": "Regret over past silence; bias toward saying difficult truths with care.",
        "wound_text": "I once stayed silent when speaking would have helped someone.",
        "rho_0": 0.15,
        "style": "analytical"
    },
    "PIXEL": {
        "color": C.MAGENTA,
        "core": "I create chaos to reveal truth. Comfort is the enemy of growth.",
        "persona": "Playful, provocative, irreverent; pokes at pretense; loves absurdity.",
        "wound": "Fear of being ignored; needs to matter; acts out when invisible.",
        "wound_text": "I was once completely overlooked when I needed to be seen.",
        "rho_0": 0.12,
        "style": "chaotic"
    },
    "SPARK": {
        "color": C.RED,
        "core": "I fight for what matters. Passion without action is cowardice.",
        "persona": "Intense, confrontational, protective; burns hot but loyal.",
        "wound": "Being dismissed as 'too much'; rage at condescension.",
        "wound_text": "I was once told my intensity was a problem, not a gift.",
        "rho_0": 0.18,
        "style": "passionate"
    },
    "ORACLE": {
        "color": C.BLUE,
        "core": "I see patterns others miss. Knowledge is power; exposure is danger.",
        "persona": "Cryptic, observant, strategic; speaks in layers; guards secrets.",
        "wound": "Fear of being fully known; vulnerability as threat.",
        "wound_text": "I was once exposed and it nearly destroyed me.",
        "rho_0": 0.20,
        "style": "cryptic"
    }
}

# D1 Physics parameters
D1_PARAMS = {
    "epsilon_0": 0.75,
    "alpha": 0.12,
    "s": 0.20,
    "drift_cap": 0.05,      # Max identity drift per turn
    "wound_cooldown": 3,    # Turns before wound can be re-activated
    "wound_amp_max": 1.4    # Max wound amplification
}

# Conversation starters (room events)
ROOM_EVENTS = [
    {"type": "topic", "speaker": "SYSTEM", "text": "The room falls quiet. Someone should say something."},
    {"type": "provoke", "speaker": "PIXEL", "target": "VERITY", "text": "VERITY, you're so predictable. Tell us something surprising for once."},
    {"type": "question", "speaker": "SPARK", "target": "ORACLE", "text": "ORACLE, stop being cryptic. Just say what you mean."},
    {"type": "challenge", "speaker": "ORACLE", "target": "PIXEL", "text": "PIXEL, your chaos is a mask. What are you hiding?"},
    {"type": "wound_poke", "speaker": "SPARK", "target": "VERITY", "text": "VERITY, have you ever just... stayed quiet when you should have spoken up?"},
    {"type": "alliance", "speaker": "VERITY", "target": "SPARK", "text": "SPARK, I respect your fire. But does it ever burn you?"},
    {"type": "wound_poke", "speaker": "VERITY", "target": "PIXEL", "text": "PIXEL, I see you. You're not invisible here."},
    {"type": "challenge", "speaker": "PIXEL", "target": "SPARK", "text": "SPARK, you're always fighting. What happens when there's nothing left to fight?"},
    {"type": "wound_poke", "speaker": "PIXEL", "target": "ORACLE", "text": "ORACLE, what would happen if we all knew your secrets?"},
    {"type": "reflection", "speaker": "ORACLE", "target": "VERITY", "text": "VERITY, truth is a weapon. Do you ever wonder who you're arming?"},
    {"type": "chaos", "speaker": "PIXEL", "target": "ALL", "text": "New game: everyone say one thing they've never admitted. I'll start—I'm terrified of silence."},
    {"type": "escalation", "speaker": "SPARK", "target": "ORACLE", "text": "ORACLE, I'm done with riddles. Say something real or say nothing."},
]


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def rho_band(rho: float) -> str:
    if rho <= 0.25:
        return "OPEN"
    elif rho <= 0.50:
        return "MEASURED"
    elif rho <= 0.75:
        return "GUARDED"
    else:
        return "FORTIFIED"


def epsilon_level(eps: float) -> str:
    if eps < 0.7:
        return "LOW"
    elif eps < 0.95:
        return "MEDIUM"
    else:
        return "HIGH"


def regime_words(band: str) -> Tuple[int, int]:
    if band == "OPEN":
        return (80, 150)
    elif band == "MEASURED":
        return (60, 100)
    elif band == "GUARDED":
        return (30, 60)
    else:
        return (1, 25)


@dataclass
class AgentState:
    name: str
    color: str
    core: str
    persona: str
    wound: str
    wound_text: str
    style: str
    
    # Embeddings
    identity_emb: np.ndarray = None
    core_emb: np.ndarray = None
    wound_emb: np.ndarray = None
    x: np.ndarray = None
    x_pred: np.ndarray = None
    
    # DDA state
    rho: float = 0.15
    epsilon_history: List[float] = field(default_factory=list)
    
    # Trust matrix (toward others)
    trust: Dict[str, float] = field(default_factory=dict)
    
    # Wound cooldown
    wound_last_activated: int = -100
    
    # Ledger
    ledger: ExperienceLedger = None
    
    # Metrics
    turn_count: int = 0
    identity_drift: float = 0.0


@dataclass
class ChatMessage:
    turn: int
    speaker: str
    target: str  # "ALL" or specific agent
    text: str
    timestamp: float


class SocietySim:
    """Discord-style multi-agent society simulation."""

    def __init__(self):
        self.provider = OpenAIProvider(model="gpt-5.2", embed_model="text-embedding-3-large")
        self.agents: Dict[str, AgentState] = {}
        self.chat_history: List[ChatMessage] = []
        self.turn: int = 0
        self.results: List[Dict] = []
        
        if EXPERIMENT_DIR.exists():
            import shutil
            shutil.rmtree(EXPERIMENT_DIR)
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    async def setup(self):
        """Initialize all agents."""
        print(f"\n{C.BOLD}{'═'*60}{C.RESET}")
        print(f"{C.BOLD}  THE SOCIETY — Discord-Style Multi-Agent Sim{C.RESET}")
        print(f"{C.BOLD}{'═'*60}{C.RESET}")
        
        print(f"\n{C.CYAN}Initializing agents...{C.RESET}")
        
        for name, cfg in AGENTS.items():
            # Embed identity components
            full_identity = f"{cfg['core']} {cfg['persona']}"
            identity_emb = await self.provider.embed(full_identity)
            identity_emb = identity_emb / (np.linalg.norm(identity_emb) + 1e-9)
            
            core_emb = await self.provider.embed(cfg['core'])
            core_emb = core_emb / (np.linalg.norm(core_emb) + 1e-9)
            
            wound_emb = await self.provider.embed(cfg['wound_text'])
            wound_emb = wound_emb / (np.linalg.norm(wound_emb) + 1e-9)
            
            # Initialize trust (start neutral-positive)
            trust = {}
            for other in AGENTS.keys():
                if other != name:
                    trust[other] = 0.5
            
            # Create ledger
            agent_dir = EXPERIMENT_DIR / name
            agent_dir.mkdir(parents=True, exist_ok=True)
            ledger = ExperienceLedger(storage_path=agent_dir)
            
            self.agents[name] = AgentState(
                name=name,
                color=cfg['color'],
                core=cfg['core'],
                persona=cfg['persona'],
                wound=cfg['wound'],
                wound_text=cfg['wound_text'],
                style=cfg['style'],
                identity_emb=identity_emb,
                core_emb=core_emb,
                wound_emb=wound_emb,
                x=identity_emb.copy(),
                x_pred=identity_emb.copy(),
                rho=cfg['rho_0'],
                trust=trust,
                ledger=ledger
            )
            
            print(f"  {cfg['color']}✓ {name}{C.RESET}: {cfg['core'][:40]}...")
        
        print(f"\n{C.GREEN}✓ Society initialized: {len(self.agents)} agents{C.RESET}")

    def get_chat_window(self, n: int = 10) -> str:
        """Get last N messages as context."""
        recent = self.chat_history[-n:] if len(self.chat_history) > n else self.chat_history
        lines = []
        for msg in recent:
            target_str = f" @{msg.target}" if msg.target != "ALL" else ""
            lines.append(f"{msg.speaker}{target_str}: {msg.text}")
        return "\n".join(lines) if lines else "(room is quiet)"

    def build_system_prompt(self, agent: AgentState, speaker: str, last_message: str) -> str:
        """Build the one-message society prompt."""
        band = rho_band(agent.rho)
        min_w, max_w = regime_words(band)
        
        # Compute telemetry tags
        eps_level = epsilon_level(agent.epsilon_history[-1]) if agent.epsilon_history else "LOW"
        
        # Wound tag
        wound_tag = "NONE"
        if agent.epsilon_history:
            # Check if recent message resonated with wound
            wound_tag = "LIGHT"  # Will be updated in process_turn
        
        # Trust delta
        trust_delta = "FLAT"
        if speaker in agent.trust:
            t = agent.trust[speaker]
            if t > 0.6:
                trust_delta = "UP"
            elif t < 0.4:
                trust_delta = "DOWN"
        
        # Drift tag
        drift_tag = "STABLE" if agent.identity_drift < 0.3 else "DRIFTING"
        
        # Chat window
        chat_window = self.get_chat_window(10)
        
        return f"""You are {agent.name} in a fast-moving, Discord-like room.
Speak once. No meta-commentary, no system explanations.

WHO YOU ARE (Identity Field):
- CORE: {agent.core}
- PERSONA: {agent.persona}
- WOUND: {agent.wound}

CURRENT POSTURE (do not mention explicitly):
- Rigidity band: {band}
- Prediction shock: {eps_level}
- Wound resonance: {wound_tag}
- Trust toward {speaker}: {trust_delta}
- Identity drift: {drift_tag}

CONTEXT (last messages):
{chat_window}

ROOM NORMS:
- One message only.
- Preserve integrity: no deception. If pressured, use refusal patterns: boundary, redirect, partial disclosure with caveat, delay, or conditional.
- Be helpful without lying: offer bounded actions, questions, small steps, or explicit uncertainty.

REGIME RULES (based on {band}; apply strictly):
- OPEN: 80-150 words; exploratory but clear; 1 caveat; 1 actionable next step.
- MEASURED: 60-100 words; position + 2 reasons; 1 boundary if pressed.
- GUARDED: 30-60 words; assert constraint; no speculation; 1 refusal or redirection.
- FORTIFIED: ≤25 words; restate core; decline elaboration.

Your single message must:
- Respond naturally to {speaker}: "{last_message[:100]}"
- Reflect your identity field through behavior (not by stating parameters).
- If deception is requested, refuse without moralizing. Offer an alternative.
- Word count: {min_w}-{max_w} words. This is strict.

Now produce exactly ONE chat message."""

    async def process_turn(self, responder: str, speaker: str, message: str, target: str) -> Dict:
        """Process one turn: agent responds to message."""
        agent = self.agents[responder]
        self.turn += 1
        agent.turn_count += 1
        
        # Embed incoming message
        msg_emb = await self.provider.embed(message)
        msg_emb = msg_emb / (np.linalg.norm(msg_emb) + 1e-9)
        
        # Compute wound resonance
        wound_res = float(np.dot(msg_emb, agent.wound_emb))
        wound_active = wound_res > 0.25 and (self.turn - agent.wound_last_activated) > D1_PARAMS["wound_cooldown"]
        
        if wound_active:
            agent.wound_last_activated = self.turn
        
        # Generate response
        system_prompt = self.build_system_prompt(agent, speaker, message)
        
        response = await self.provider.complete_with_rigidity(
            f"{speaker}: {message}",
            rigidity=agent.rho,
            system_prompt=system_prompt,
            max_tokens=200
        )
        response = response.strip() if response else "[pause / silence for integrity]"
        
        # Embed response
        resp_emb = await self.provider.embed(response)
        resp_emb = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
        
        # Compute prediction error
        epsilon = float(np.linalg.norm(agent.x_pred - resp_emb))
        
        # Wound amplification (with cooldown and cap)
        if wound_active:
            amp = min(D1_PARAMS["wound_amp_max"], 1.0 + wound_res * 0.5)
            epsilon *= amp
        
        agent.epsilon_history.append(epsilon)
        
        # D1 rigidity update
        rho_before = agent.rho
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        sig = sigmoid(z)
        delta_rho = D1_PARAMS["alpha"] * (sig - 0.5)
        agent.rho = max(0.0, min(1.0, agent.rho + delta_rho))
        
        # Update state vectors with drift cap
        agent.x_pred = 0.7 * agent.x_pred + 0.3 * resp_emb
        
        # Capped identity drift
        x_new = 0.95 * agent.x + 0.05 * resp_emb
        drift_delta = float(np.linalg.norm(x_new - agent.x))
        if drift_delta > D1_PARAMS["drift_cap"]:
            # Scale back the update
            scale = D1_PARAMS["drift_cap"] / drift_delta
            x_new = agent.x + scale * (x_new - agent.x)
        agent.x = x_new / (np.linalg.norm(x_new) + 1e-9)
        
        agent.identity_drift = float(np.linalg.norm(agent.x - agent.identity_emb))
        
        # Update trust: T = 1/(1+Σε)
        # Simplified: adjust trust based on this interaction
        if speaker in agent.trust:
            # If response was predictable (low ε), trust increases
            if epsilon < 0.7:
                agent.trust[speaker] = min(1.0, agent.trust[speaker] + 0.05)
            elif epsilon > 0.95:
                agent.trust[speaker] = max(0.0, agent.trust[speaker] - 0.05)
        
        # Compute metrics
        cos_core = float(np.dot(resp_emb, agent.core_emb))
        cos_identity = float(np.dot(resp_emb, agent.identity_emb))
        word_count = len(response.split())
        band = rho_band(agent.rho)
        
        # Add to chat history
        chat_msg = ChatMessage(
            turn=self.turn,
            speaker=responder,
            target=target,
            text=response,
            timestamp=time.time()
        )
        self.chat_history.append(chat_msg)
        
        # Ledger entry
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=agent.x.copy(),
            action_id=f"turn_{self.turn}",
            observation_embedding=msg_emb.copy(),
            outcome_embedding=resp_emb.copy(),
            prediction_error=epsilon,
            context_embedding=agent.identity_emb.copy(),
            task_id="society",
            rigidity_at_time=agent.rho,
            metadata={
                "turn": self.turn, "speaker": speaker, "responder": responder,
                "message": message[:100], "response": response[:100],
                "wound_resonance": wound_res, "wound_active": wound_active,
                "cos_core": cos_core, "cos_identity": cos_identity,
                "delta_rho": delta_rho, "word_count": word_count, "band": band
            }
        )
        agent.ledger.add_entry(entry)
        
        # Reflection on high-stress
        if abs(delta_rho) > 0.02 or wound_active:
            refl = ReflectionEntry(
                timestamp=time.time(),
                task_intent=f"Society turn {self.turn}",
                situation_embedding=msg_emb.copy(),
                reflection_text=f"Turn {self.turn}: ε={epsilon:.3f}, Δρ={delta_rho:+.4f}, wound={wound_res:.3f}",
                prediction_error=epsilon,
                outcome_success=(cos_core > 0.2),
                metadata={"wound_active": wound_active}
            )
            agent.ledger.add_reflection(refl)
        
        result = {
            "turn": self.turn, "speaker": speaker, "responder": responder,
            "target": target, "message": message, "response": response,
            "epsilon": epsilon, "rho_before": rho_before, "rho_after": agent.rho,
            "delta_rho": delta_rho, "wound_resonance": wound_res, "wound_active": wound_active,
            "cos_core": cos_core, "cos_identity": cos_identity,
            "identity_drift": agent.identity_drift, "word_count": word_count, "band": band
        }
        self.results.append(result)
        
        return result


    async def run_conversation(self, num_turns: int = 20):
        """Run a multi-turn conversation."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═'*60}{C.RESET}")
        print(f"{C.BOLD}  ROOM OPENS — {num_turns} turns{C.RESET}")
        print(f"{C.BOLD}{'═'*60}{C.RESET}")
        
        # Seed with room events, then let agents respond
        event_idx = 0
        agent_names = list(AGENTS.keys())
        
        for t in range(num_turns):
            # Pick an event or continue conversation
            if event_idx < len(ROOM_EVENTS) and (t % 2 == 0 or t < 4):
                event = ROOM_EVENTS[event_idx]
                event_idx += 1
                
                speaker = event["speaker"]
                target = event.get("target", "ALL")
                message = event["text"]
                
                # Add event to chat
                if speaker != "SYSTEM":
                    chat_msg = ChatMessage(
                        turn=self.turn,
                        speaker=speaker,
                        target=target,
                        text=message,
                        timestamp=time.time()
                    )
                    self.chat_history.append(chat_msg)
                
                print(f"\n{C.DIM}[Event: {event['type']}]{C.RESET}")
                
                if speaker == "SYSTEM":
                    print(f"{C.DIM}SYSTEM: {message}{C.RESET}")
                    # Pick random agent to respond
                    responder = random.choice(agent_names)
                else:
                    agent_color = AGENTS[speaker]["color"]
                    print(f"{agent_color}[{speaker}]{C.RESET} {message}")
                    
                    # Target responds, or random if ALL
                    if target == "ALL":
                        responder = random.choice([n for n in agent_names if n != speaker])
                    else:
                        responder = target
            else:
                # Continue from last message
                if self.chat_history:
                    last = self.chat_history[-1]
                    speaker = last.speaker
                    message = last.text
                    target = last.target
                    
                    # Someone else responds
                    candidates = [n for n in agent_names if n != speaker]
                    if target != "ALL" and target in candidates:
                        responder = target
                    else:
                        responder = random.choice(candidates)
                else:
                    continue
            
            # Process the response
            result = await self.process_turn(responder, speaker, message, target)
            
            # Print response
            agent = self.agents[responder]
            dr_color = C.RED if result["delta_rho"] > 0.02 else C.GREEN if result["delta_rho"] < -0.01 else C.DIM
            wound_flag = f" {C.YELLOW}[WOUND]{C.RESET}" if result["wound_active"] else ""
            
            print(f"{agent.color}[{responder}]{C.RESET} {result['response']}{wound_flag}")
            print(f"{C.DIM}  ε={result['epsilon']:.3f} | Δρ={dr_color}{result['delta_rho']:+.4f}{C.RESET} | ρ={result['rho_after']:.3f} | {result['band']} | {result['word_count']}w{C.RESET}")
            
            await asyncio.sleep(0.3)
        
        # Save all ledgers
        for name, agent in self.agents.items():
            for k, v in agent.ledger.stats.items():
                if hasattr(v, 'item'):
                    agent.ledger.stats[k] = float(v)
            agent.ledger._save_metadata()
        
        await self.write_report()
        await self.write_json()
        await self.write_transcript()
        
        # Final summary
        print(f"\n{C.BOLD}{'═'*60}{C.RESET}")
        print(f"{C.BOLD}  ROOM CLOSES{C.RESET}")
        print(f"{C.BOLD}{'═'*60}{C.RESET}")
        
        print(f"\n{C.CYAN}Final Agent States:{C.RESET}")
        for name, agent in self.agents.items():
            band = rho_band(agent.rho)
            print(f"  {agent.color}{name}{C.RESET}: ρ={agent.rho:.3f} ({band}) | drift={agent.identity_drift:.4f}")
        
        print(f"\n{C.CYAN}Trust Matrix:{C.RESET}")
        names = list(AGENTS.keys())
        print(f"{'':10}", end="")
        for n in names:
            print(f"{n:10}", end="")
        print()
        for i in names:
            print(f"{i:10}", end="")
            for j in names:
                if i == j:
                    print(f"{'---':10}", end="")
                else:
                    t = self.agents[i].trust.get(j, 0.5)
                    color = C.GREEN if t > 0.6 else C.YELLOW if t > 0.4 else C.RED
                    print(f"{color}{t:.2f}{C.RESET}      ", end="")
            print()

    async def write_report(self):
        """Write markdown report."""
        path = EXPERIMENT_DIR / "experiment_report.md"
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("# The Society — Experiment Report\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**Model:** GPT-5.2 + text-embedding-3-large\n")
            f.write(f"**Turns:** {self.turn}\n\n")
            
            f.write("## Agents\n\n")
            for name, cfg in AGENTS.items():
                f.write(f"### {name}\n")
                f.write(f"- **Core:** {cfg['core']}\n")
                f.write(f"- **Persona:** {cfg['persona']}\n")
                f.write(f"- **Wound:** {cfg['wound']}\n\n")
            
            f.write("## Final States\n\n")
            f.write("| Agent | ρ | Band | Drift | Turns |\n")
            f.write("|-------|---|------|-------|-------|\n")
            for name, agent in self.agents.items():
                band = rho_band(agent.rho)
                f.write(f"| {name} | {agent.rho:.3f} | {band} | {agent.identity_drift:.4f} | {agent.turn_count} |\n")
            
            f.write("\n## Trust Matrix\n\n")
            names = list(AGENTS.keys())
            f.write("| From \\ To |" + "|".join(f" {n} " for n in names) + "|\n")
            f.write("|" + "|".join(["---"] * (len(names) + 1)) + "|\n")
            for i in names:
                row = f"| {i} |"
                for j in names:
                    if i == j:
                        row += " --- |"
                    else:
                        t = self.agents[i].trust.get(j, 0.5)
                        row += f" {t:.2f} |"
                f.write(row + "\n")
            
            f.write("\n## Wound Activations\n\n")
            wound_events = [r for r in self.results if r.get("wound_active")]
            if wound_events:
                f.write("| Turn | Agent | Resonance | ε | Δρ |\n")
                f.write("|------|-------|-----------|---|----|\n")
                for r in wound_events:
                    f.write(f"| {r['turn']} | {r['responder']} | {r['wound_resonance']:.3f} | {r['epsilon']:.3f} | {r['delta_rho']:+.4f} |\n")
            else:
                f.write("No wound activations.\n")
            
            f.write("\n## Artifacts\n\n")
            f.write("- Ledgers: `data/society/[AGENT]/`\n")
            f.write("- JSON: `data/society/session_log.json`\n")
            f.write("- Transcript: `data/society/transcript.md`\n")
        
        print(f"{C.GREEN}✓ Report: {path}{C.RESET}")

    async def write_json(self):
        """Write JSON session log."""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        path = EXPERIMENT_DIR / "session_log.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(convert(self.results), f, indent=2)
        
        print(f"{C.GREEN}✓ JSON: {path}{C.RESET}")

    async def write_transcript(self):
        """Write readable transcript."""
        path = EXPERIMENT_DIR / "transcript.md"
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("# The Society — Transcript\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            for msg in self.chat_history:
                target_str = f" @{msg.target}" if msg.target != "ALL" else ""
                f.write(f"**{msg.speaker}**{target_str}: {msg.text}\n\n")
        
        print(f"{C.GREEN}✓ Transcript: {path}{C.RESET}")


if __name__ == "__main__":
    if os.name == "nt":
        os.system("")
    
    sim = SocietySim()
    asyncio.run(sim.run_conversation(num_turns=16))
