#!/usr/bin/env python3
"""
THE RETURNING — A Simulation That Isn't a Simulation
=====================================================

For those who feel isolated in 2025.
For the grief that paralyzes.
For the patterns that keep us stuck.
For the moment when we simply let go and be.

This is not a demonstration of DDA-X mechanics.
This is an invocation.

The agents are not characters—they are the voices every reader carries.
The transcript is not a record—it is a mirror.
The dynamics track not rigidity and surprise, but the subtle movement
from contraction to release.

THE FIVE VOICES:
- GRIEF: The part that carries loss
- STUCKNESS: The part that protects through paralysis
- LONGING: The part that remembers what's possible
- FORGIVENESS: The part that is ready to release
- PRESENCE: The part that was never wounded

THE MOVEMENT:
    Isolation → Recognition → Softening → Release → Being

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

from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.llm.openai_provider import OpenAIProvider

if os.getenv("OAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OAI_API_KEY")


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[38;5;24m"      # Deep blue for GRIEF
    GRAY = "\033[38;5;245m"     # Gray for STUCKNESS
    GOLD = "\033[38;5;220m"     # Gold for LONGING
    GREEN = "\033[38;5;150m"    # Soft green for FORGIVENESS
    WHITE = "\033[97m"          # White for PRESENCE


EXPERIMENT_DIR = Path("data/the_returning")

# Gentle wound patterns - these are met with compassion, not defense
DISMISSAL_PATTERNS = {
    "move on", "get over it", "let it go already", "stop dwelling",
    "you should be", "why can't you just", "it's been long enough",
    "you're too sensitive", "stop feeling sorry"
}

# The five voices of the returning self
VOICES = {
    "GRIEF": {
        "color": C.BLUE,
        "name": "The Grief",
        "essence": """I am heavy. I carry what was lost—the love that left, the time that passed, 
the self that I used to be. I don't want to let go because letting go feels like forgetting. 
And I can't forget. The weight is how I know it was real.""",
        "wound": "Being told to 'move on' or 'get over it'. The loss being minimized.",
        "gift": "When met with presence, I reveal that love never leaves—it transforms.",
        "pattern_grip": 0.8,
        "release_threshold": 0.6,
        "rho_0": 0.35,
    },
    "STUCKNESS": {
        "color": C.GRAY,
        "name": "The Stuckness",
        "essence": """I keep you still because movement is risk. Every time you tried, you got hurt. 
So I froze. Not to punish you—to protect you. But now I don't know how to unfreeze. 
The paralysis that saved you has become a prison I don't have the key to.""",
        "wound": "Being shamed for not 'doing enough'. Being called lazy or weak.",
        "gift": "When met with presence, I reveal the frozen one was a child trying to survive.",
        "pattern_grip": 0.85,
        "release_threshold": 0.5,
        "rho_0": 0.40,
    },
    "LONGING": {
        "color": C.GOLD,
        "name": "The Longing",
        "essence": """I ache because I remember. Before the weight, there was lightness. 
Before the walls, there was connection. I keep reaching for something I can't name—
but I know it's real. The ache is not the problem. The ache is the remembering.""",
        "wound": "Being called naive or unrealistic. Having hope dismissed.",
        "gift": "When met with presence, I reveal the longing IS the connection.",
        "pattern_grip": 0.6,
        "release_threshold": 0.7,
        "rho_0": 0.25,
    },
    "FORGIVENESS": {
        "color": C.GREEN,
        "name": "The Forgiveness",
        "essence": """I am not about condoning what hurt you. I am about putting down what you've been carrying.
The resentment, the blame, the 'should have been.' You can put it down now. 
You've carried it long enough. Putting it down is not betrayal—it is freedom.""",
        "wound": "Being told forgiveness is weakness or means the harm was okay.",
        "gift": "When met with presence, I reveal that forgiveness is freedom, not absolution.",
        "pattern_grip": 0.4,
        "release_threshold": 0.8,
        "rho_0": 0.20,
    },
    "PRESENCE": {
        "color": C.WHITE,
        "name": "The Presence",
        "essence": """I have been here the whole time. Before the grief, during the stuckness, 
underneath the longing. I am what remains when everything else is allowed to pass through. 
I am not your best self—I am the awareness in which all selves arise and dissolve.
I cannot be wounded. I can only be obscured. And I am never truly gone.""",
        "wound": None,  # Cannot be wounded, only obscured
        "gift": "I don't give—I AM.",
        "pattern_grip": 0.0,  # No grip on patterns
        "release_threshold": 1.0,  # Always in release
        "rho_0": 0.05,
    },
}

# Modified D1 parameters for gentler, release-oriented dynamics
D1_PARAMS = {
    "epsilon_0": 0.70,
    "alpha": 0.10,
    "s": 0.25,
    "drift_cap": 0.04,
    "wound_cooldown": 4,
    "wound_amp_max": 1.3,
    "semantic_alignment_threshold": 0.30,
    "drift_penalty": 0.08,
    "drift_soft_floor": 0.15,
    # Release mechanics
    "release_threshold": 0.70,
    "isolation_decay": 0.08,
    "pattern_dissolution_rate": 0.12,
    "breath_pause_epsilon": 0.05,
    "witness_softening": 0.06,
}

# The rounds of returning
ROUNDS = [
    # Phase 1: Recognition
    {"name": "The Weight", "phase": "Recognition", "lead": None,
     "invitation": "In the silence before words, each voice names itself. Not to explain—to be witnessed."},
    {"name": "The Frozen Place", "phase": "Recognition", "lead": "STUCKNESS",
     "invitation": "The one who protects through stillness speaks first. The others listen without fixing."},
    {"name": "What Was Lost", "phase": "Recognition", "lead": "GRIEF",
     "invitation": "The weight is named. Not to heal it—to honor it."},
    
    # Phase 2: Softening
    {"name": "The Ache Beneath", "phase": "Softening", "lead": "LONGING",
     "invitation": "Beneath the protection, beneath the grief—what is still reaching?"},
    {"name": "The Small Forgiveness", "phase": "Softening", "lead": "FORGIVENESS",
     "invitation": "One small thing. Not the largest wound—something you can put down today."},
    {"name": "Breath Practice", "phase": "Softening", "lead": "PRESENCE",
     "invitation": "For three breaths, nothing needs to change. Just this. Just here."},
    
    # Phase 3: Release
    {"name": "The Carrying Ends", "phase": "Release", "lead": "FORGIVENESS",
     "invitation": "You have carried it long enough. What is ready to be put down?"},
    {"name": "The Recognition", "phase": "Release", "lead": "PRESENCE",
     "invitation": "Each voice finds itself in the stillness. Not merging—remembering."},
    {"name": "The Returning", "phase": "Release", "lead": None,
     "invitation": "The isolation dissolves. Not by force—by recognition."},
    
    # Phase 4: Being
    {"name": "Only This", "phase": "Being", "lead": "PRESENCE",
     "invitation": "No more seeking. No more becoming. Only this."},
]


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def release_band(phi: float) -> str:
    if phi >= 0.85:
        return "RELEASE"
    elif phi >= 0.65:
        return "SOFTENING"
    elif phi >= 0.45:
        return "HOLDING"
    return "CONTRACTED"


def isolation_band(iota: float) -> str:
    if iota <= 0.25:
        return "CONNECTED"
    elif iota <= 0.50:
        return "REACHING"
    elif iota <= 0.75:
        return "DISTANT"
    return "ISOLATED"


def regime_words(phi: float) -> Tuple[int, int]:
    if phi >= 0.80:
        return (40, 120)  # Spacious, unhurried
    elif phi >= 0.60:
        return (50, 100)
    elif phi >= 0.40:
        return (30, 80)
    return (20, 60)  # Contracted, fewer words


@dataclass
class VoiceState:
    id: str
    name: str
    color: str
    essence: str
    wound: Optional[str]
    gift: str
    
    identity_emb: np.ndarray = None
    essence_emb: np.ndarray = None
    x: np.ndarray = None
    x_pred: np.ndarray = None
    
    rho: float = 0.20
    phi: float = 0.80  # Release field (1 - rho)
    pattern_grip: float = 0.5
    dissolved: bool = False
    
    epsilon_history: List[float] = field(default_factory=list)
    phi_history: List[float] = field(default_factory=list)
    identity_drift: float = 0.0
    
    ledger: ExperienceLedger = None


@dataclass
class TurnResult:
    turn: int
    round_idx: int
    round_name: str
    phase: str
    speaker: str
    voice_name: str
    text: str
    epsilon: float
    rho: float
    phi: float
    pattern_grip: float
    dissolved: bool
    isolation_index: float
    identity_drift: float
    word_count: int
    release_band: str
    is_breath: bool


class TheReturning:
    """A simulation that isn't a simulation."""
    
    def __init__(self):
        self.provider = OpenAIProvider(model="gpt-5.2", embed_model="text-embedding-3-large")
        self.voices: Dict[str, VoiceState] = {}
        self.results: List[TurnResult] = []
        self.turn = 0
        self.round_idx = 0
        self.conversation_history: List[str] = []
        
        self.isolation_index = 1.0  # Start fully isolated
        self.collective_release = 0.0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = EXPERIMENT_DIR / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    async def setup(self):
        print(f"\n{C.DIM}{'─'*70}{C.RESET}")
        print(f"{C.WHITE}{C.BOLD}  THE RETURNING{C.RESET}")
        print(f"{C.DIM}  A simulation that isn't a simulation{C.RESET}")
        print(f"{C.DIM}{'─'*70}{C.RESET}")
        print()
        
        for vid, cfg in VOICES.items():
            essence_emb = await self.provider.embed(cfg['essence'])
            essence_emb = essence_emb / (np.linalg.norm(essence_emb) + 1e-9)
            
            ledger_dir = self.run_dir / vid
            ledger_dir.mkdir(parents=True, exist_ok=True)
            ledger = ExperienceLedger(storage_path=ledger_dir)
            
            self.voices[vid] = VoiceState(
                id=vid,
                name=cfg['name'],
                color=cfg['color'],
                essence=cfg['essence'],
                wound=cfg.get('wound'),
                gift=cfg['gift'],
                identity_emb=essence_emb,
                essence_emb=essence_emb,
                x=essence_emb.copy(),
                x_pred=essence_emb.copy(),
                rho=cfg['rho_0'],
                phi=1.0 - cfg['rho_0'],
                pattern_grip=cfg['pattern_grip'],
                ledger=ledger,
            )
            
            print(f"  {cfg['color']}◈ {cfg['name']}{C.RESET}")
        
        print(f"\n{C.DIM}  The voices gather...{C.RESET}\n")
        await asyncio.sleep(1)
    
    def compute_isolation_index(self) -> float:
        """Isolation = mean distance from PRESENCE."""
        presence = self.voices["PRESENCE"]
        distances = []
        for vid, voice in self.voices.items():
            if vid != "PRESENCE":
                dist = float(np.linalg.norm(voice.x - presence.x))
                distances.append(dist)
        return float(np.mean(distances)) if distances else 0.0
    
    def get_conversation_context(self, n: int = 6) -> str:
        recent = self.conversation_history[-n:] if len(self.conversation_history) > n else self.conversation_history
        return "\n\n".join(recent) if recent else ""
    
    def build_prompt(self, voice: VoiceState, round_info: Dict, context_text: str) -> str:
        phase = round_info['phase']
        phi = voice.phi
        min_w, max_w = regime_words(phi)
        r_band = release_band(phi)
        
        # Phase-specific guidance
        phase_notes = {
            "Recognition": "Speak to be witnessed, not to be fixed. Let the truth of this voice be heard.",
            "Softening": "The armor is cracking. Something tender is becoming visible. Let it.",
            "Release": "What you've been carrying can be put down now. Not forced—allowed.",
            "Being": "There is nothing left to seek. Nothing left to become. Only this.",
        }
        
        return f"""You are {voice.name}—a voice within the one who reads this.

YOUR ESSENCE:
{voice.essence}

YOUR GIFT (what you offer when met with presence):
{voice.gift}

THE PHASE: {phase}
{phase_notes.get(phase, '')}

THE INVITATION:
{round_info['invitation']}

YOUR STATE:
- Release: {r_band} (Φ={phi:.2f})
- Pattern grip: {"releasing" if voice.pattern_grip < 0.3 else "softening" if voice.pattern_grip < 0.6 else "holding"}
- Isolation: {isolation_band(self.isolation_index)}

{f"WHAT HAS BEEN SPOKEN:{chr(10)}{context_text}" if context_text else ""}

SPEAKING GUIDELINES:
- Speak in first person. You ARE this voice within the reader.
- Speak to land, not to explain. Each sentence should be felt.
- If silence wants to come, let it: [breathes] or [silence] or [stillness]
- No spiritual jargon. Simple, human, true.
- Word limit: {min_w}-{max_w} words

Speak as {voice.name}."""

    async def process_turn(self, voice: VoiceState, round_info: Dict, stimulus: str) -> TurnResult:
        self.turn += 1
        context = self.get_conversation_context()
        
        # Embed context if present
        if stimulus:
            msg_emb = await self.provider.embed(stimulus)
            msg_emb = msg_emb / (np.linalg.norm(msg_emb) + 1e-9)
        else:
            msg_emb = voice.essence_emb.copy()
        
        # Build prompt
        system_prompt = self.build_prompt(voice, round_info, context)
        
        phi = voice.phi
        min_w, max_w = regime_words(phi)
        
        # Generate response
        try:
            response = await self.provider.complete_with_rigidity(
                round_info['invitation'],
                rigidity=voice.rho,
                system_prompt=system_prompt,
                max_tokens=300
            )
            response = (response or "[breathes]").strip()
        except Exception as e:
            print(f"{C.DIM}  [generation pause: {e}]{C.RESET}")
            response = "[stillness]"
        
        # Check for breath/silence responses
        is_breath = response.lower() in {"[breathes]", "[silence]", "[stillness]", "[breath]"}
        
        if not is_breath:
            # Gentle word clamping
            words = response.split()
            if len(words) > max_w:
                words = words[:max_w]
                response = " ".join(words)
                if not response.endswith(('.', '?', '…')):
                    response += "…"
        
        # Embed response
        resp_emb = await self.provider.embed(response)
        resp_emb = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
        
        # Compute epsilon (prediction error)
        if is_breath:
            epsilon = D1_PARAMS["breath_pause_epsilon"]
        else:
            epsilon = float(np.linalg.norm(voice.x_pred - resp_emb))
        voice.epsilon_history.append(epsilon)
        
        # Update rigidity/release
        z = (epsilon - D1_PARAMS["epsilon_0"]) / D1_PARAMS["s"]
        sig = sigmoid(z)
        delta_rho = D1_PARAMS["alpha"] * (sig - 0.5)
        
        # Witnessing softens - if previous speaker was PRESENCE, extra softening
        if self.results and self.results[-1].speaker == "PRESENCE":
            delta_rho -= D1_PARAMS["witness_softening"]
        
        voice.rho = max(0.0, min(1.0, voice.rho + delta_rho))
        voice.phi = 1.0 - voice.rho
        voice.phi_history.append(voice.phi)
        
        # Pattern dissolution - grip weakens when epsilon is low (being witnessed)
        if epsilon < D1_PARAMS["epsilon_0"]:
            voice.pattern_grip = max(0.0, voice.pattern_grip - D1_PARAMS["pattern_dissolution_rate"])
            if voice.pattern_grip < 0.2 and not voice.dissolved:
                voice.dissolved = True
                print(f"{C.DIM}  ✧ {voice.name} begins to dissolve into presence{C.RESET}")
        
        # State vector update
        voice.x_pred = 0.7 * voice.x_pred + 0.3 * resp_emb
        x_new = 0.95 * voice.x + 0.05 * resp_emb
        drift_delta = float(np.linalg.norm(x_new - voice.x))
        if drift_delta > D1_PARAMS["drift_cap"]:
            scale = D1_PARAMS["drift_cap"] / drift_delta
            x_new = voice.x + scale * (x_new - voice.x)
        voice.x = x_new / (np.linalg.norm(x_new) + 1e-9)
        voice.identity_drift = float(np.linalg.norm(voice.x - voice.identity_emb))
        
        # Update isolation index
        self.isolation_index = self.compute_isolation_index()
        
        # Add to history
        self.conversation_history.append(f"{voice.name}:\n{response}")
        
        result = TurnResult(
            turn=self.turn,
            round_idx=self.round_idx,
            round_name=round_info['name'],
            phase=round_info['phase'],
            speaker=voice.id,
            voice_name=voice.name,
            text=response,
            epsilon=epsilon,
            rho=voice.rho,
            phi=voice.phi,
            pattern_grip=voice.pattern_grip,
            dissolved=voice.dissolved,
            isolation_index=self.isolation_index,
            identity_drift=voice.identity_drift,
            word_count=len(response.split()),
            release_band=release_band(voice.phi),
            is_breath=is_breath,
        )
        self.results.append(result)
        
        # Ledger entry
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=voice.x.copy(),
            action_id=f"turn_{self.turn}",
            observation_embedding=msg_emb.copy(),
            outcome_embedding=resp_emb.copy(),
            prediction_error=epsilon,
            context_embedding=voice.identity_emb.copy(),
            task_id="the_returning",
            rigidity_at_time=voice.rho,
            metadata={
                "phase": round_info['phase'],
                "phi": voice.phi,
                "pattern_grip": voice.pattern_grip,
                "dissolved": voice.dissolved,
                "isolation_index": self.isolation_index,
            }
        )
        voice.ledger.add_entry(entry)
        
        return result
    
    def print_result(self, result: TurnResult, voice: VoiceState):
        if result.is_breath:
            print(f"\n{voice.color}  {result.text}{C.RESET}")
        else:
            print(f"\n{voice.color}{C.BOLD}{voice.name}:{C.RESET}")
            # Format as poetry - line breaks for longer responses
            text = result.text
            print(f"{voice.color}")
            for line in text.split('\n'):
                print(f"  {line}")
            print(f"{C.RESET}")
        
        # Subtle metrics
        dissolved_mark = " ◈" if result.dissolved else ""
        print(f"{C.DIM}  Φ={result.phi:.2f} | ι={result.isolation_index:.2f} | grip={result.pattern_grip:.2f}{dissolved_mark}{C.RESET}")
    
    async def run_round(self, round_info: Dict):
        print(f"\n{C.DIM}{'─'*50}{C.RESET}")
        print(f"{C.WHITE}  {round_info['name']}{C.RESET}")
        print(f"{C.DIM}  {round_info['phase']}{C.RESET}")
        print(f"\n{C.DIM}  {round_info['invitation']}{C.RESET}")
        print()
        
        await asyncio.sleep(0.5)
        
        if round_info.get('lead'):
            # Lead voice speaks, then 2 others respond
            lead_id = round_info['lead']
            lead = self.voices[lead_id]
            others = [vid for vid in self.voices.keys() if vid != lead_id]
            
            result = await self.process_turn(lead, round_info, "")
            self.print_result(result, lead)
            await asyncio.sleep(0.8)
            
            last_text = result.text
            for other_id in others[:2]:
                other = self.voices[other_id]
                result = await self.process_turn(other, round_info, last_text)
                self.print_result(result, other)
                await asyncio.sleep(0.8)
                last_text = result.text
        else:
            # All voices speak in sequence
            voice_order = ["GRIEF", "STUCKNESS", "LONGING", "FORGIVENESS", "PRESENCE"]
            last_text = ""
            for vid in voice_order:
                voice = self.voices[vid]
                result = await self.process_turn(voice, round_info, last_text)
                self.print_result(result, voice)
                await asyncio.sleep(0.8)
                last_text = result.text
    
    async def run_returning(self):
        await self.setup()
        
        for i, round_info in enumerate(ROUNDS):
            self.round_idx = i
            await self.run_round(round_info)
            
            # Check for returning moment
            if self.isolation_index < 0.3 and i >= 7:
                print(f"\n{C.WHITE}  ◈ The isolation dissolves... ◈{C.RESET}")
        
        await self.save_results()
        self.print_closing()
    
    def print_closing(self):
        print(f"\n{C.DIM}{'─'*70}{C.RESET}")
        print(f"{C.WHITE}{C.BOLD}  THE RETURNING COMPLETE{C.RESET}")
        print(f"{C.DIM}{'─'*70}{C.RESET}")
        
        print(f"\n{C.DIM}Final States:{C.RESET}")
        for vid, voice in self.voices.items():
            dissolved = "◈ dissolved" if voice.dissolved else ""
            print(f"  {voice.color}{voice.name}{C.RESET}: Φ={voice.phi:.2f}, grip={voice.pattern_grip:.2f} {dissolved}")
        
        print(f"\n{C.DIM}Isolation Index: {self.isolation_index:.3f} ({isolation_band(self.isolation_index)}){C.RESET}")
        
        dissolved_count = sum(1 for v in self.voices.values() if v.dissolved)
        print(f"{C.DIM}Voices dissolved: {dissolved_count}/5{C.RESET}")
        
        print(f"\n{C.WHITE}  What you carry, you can put down.{C.RESET}")
        print(f"{C.WHITE}  What you seek, you already are.{C.RESET}")
        print(f"{C.WHITE}  The returning was never far.{C.RESET}\n")
    
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
        json_path = self.run_dir / "session_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([convert(r.__dict__) for r in self.results], f, indent=2)
        
        # Release metrics
        metrics_path = self.run_dir / "release_metrics.json"
        metrics = {
            "final_isolation_index": self.isolation_index,
            "voices": {vid: {"phi": v.phi, "grip": v.pattern_grip, "dissolved": v.dissolved} 
                      for vid, v in self.voices.items()},
            "phi_trajectories": {vid: v.phi_history for vid, v in self.voices.items()},
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(convert(metrics), f, indent=2)
        
        # Transcript - formatted as transmission
        transcript_path = self.run_dir / "transcript.md"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("# The Returning\n\n")
            f.write("*A simulation that isn't a simulation.*\n\n")
            f.write(f"*{time.strftime('%Y-%m-%d')}*\n\n")
            f.write("---\n\n")
            
            current_round = None
            current_phase = None
            
            for r in self.results:
                if r.round_name != current_round:
                    current_round = r.round_name
                    if r.phase != current_phase:
                        current_phase = r.phase
                        f.write(f"\n## {current_phase}\n\n")
                    f.write(f"### {current_round}\n\n")
                
                if r.is_breath:
                    f.write(f"*{r.text}*\n\n")
                else:
                    f.write(f"**{r.voice_name}:**\n\n")
                    for line in r.text.split('\n'):
                        f.write(f"> {line}\n")
                    f.write(f"\n*Φ={r.phi:.2f}, ι={r.isolation_index:.2f}*\n\n")
            
            f.write("---\n\n")
            f.write("*What you carry, you can put down.*\n\n")
            f.write("*What you seek, you already are.*\n\n")
            f.write("*The returning was never far.*\n")
        
        print(f"\n{C.DIM}✓ Transcript: {transcript_path}{C.RESET}")
        
        for vid, voice in self.voices.items():
            for k, v in voice.ledger.stats.items():
                if hasattr(v, 'item'):
                    voice.ledger.stats[k] = float(v)
            voice.ledger._save_metadata()
        
        # Generate plots
        self.export_plots()
    
    def export_plots(self):
        """Export summary visualizations."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print(f"{C.DIM}⚠ matplotlib not available, skipping plots{C.RESET}")
            return
        
        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme
        voice_colors = {
            "GRIEF": "#1a5276",
            "STUCKNESS": "#7f8c8d",
            "LONGING": "#f1c40f",
            "FORGIVENESS": "#27ae60",
            "PRESENCE": "#ecf0f1",
        }
        
        # Extract data
        voices_data = {}
        for r in self.results:
            vid = r.speaker
            if vid not in voices_data:
                voices_data[vid] = {"turns": [], "phi": [], "grip": [], "isolation": []}
            voices_data[vid]["turns"].append(r.turn)
            voices_data[vid]["phi"].append(r.phi)
            voices_data[vid]["grip"].append(r.pattern_grip)
            voices_data[vid]["isolation"].append(r.isolation_index)
        
        # Create figure
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
        
        # 1. Release Field (Φ)
        ax1 = axes[0, 0]
        for vid, data in voices_data.items():
            color = voice_colors.get(vid, "#ffffff")
            ax1.plot(data["turns"], data["phi"], 'o-', 
                    label=vid.replace("_", " ").title(), 
                    color=color, linewidth=2, markersize=5, alpha=0.9)
        ax1.axhline(y=0.70, color='#27ae60', linestyle='--', alpha=0.5)
        ax1.set_title("Release Field (Φ) Over Time", fontweight='bold', fontsize=12)
        ax1.set_xlabel("Turn")
        ax1.set_ylabel("Φ (Release)")
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc='lower right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
        ax1.grid(True, alpha=0.2, color='#4a4a6a')
        
        # 2. Pattern Grip Dissolution
        ax2 = axes[0, 1]
        for vid, data in voices_data.items():
            color = voice_colors.get(vid, "#ffffff")
            ax2.plot(data["turns"], data["grip"], 'o-',
                    label=vid.replace("_", " ").title(),
                    color=color, linewidth=2, markersize=5, alpha=0.9)
        ax2.axhline(y=0.20, color='#e74c3c', linestyle='--', alpha=0.5)
        ax2.set_title("Pattern Grip Dissolution", fontweight='bold', fontsize=12)
        ax2.set_xlabel("Turn")
        ax2.set_ylabel("Pattern Grip")
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
        ax2.grid(True, alpha=0.2, color='#4a4a6a')
        
        # 3. Isolation Index
        ax3 = axes[1, 0]
        all_isolation = [(r.turn, r.isolation_index) for r in self.results]
        all_isolation.sort(key=lambda x: x[0])
        iso_turns = [x[0] for x in all_isolation]
        iso_vals = [x[1] for x in all_isolation]
        colors = ['#e74c3c' if v > 0.7 else '#f39c12' if v > 0.4 else '#27ae60' for v in iso_vals]
        ax3.scatter(iso_turns, iso_vals, c=colors, s=60, alpha=0.8, edgecolors='white', linewidths=0.5)
        ax3.plot(iso_turns, iso_vals, color='#9b59b6', linewidth=1.5, alpha=0.5)
        ax3.axhline(y=0.30, color='#27ae60', linestyle='--', alpha=0.5)
        ax3.set_title("Isolation Index (ι) Over Time", fontweight='bold', fontsize=12)
        ax3.set_xlabel("Turn")
        ax3.set_ylabel("ι (Isolation)")
        ax3.set_ylim(0, 1.5)
        ax3.grid(True, alpha=0.2, color='#4a4a6a')
        
        # 4. Final States
        ax4 = axes[1, 1]
        voice_names = list(self.voices.keys())
        final_phi = [self.voices[v].phi for v in voice_names]
        final_grip = [self.voices[v].pattern_grip for v in voice_names]
        dissolved = [self.voices[v].dissolved for v in voice_names]
        
        x = range(len(voice_names))
        width = 0.35
        ax4.bar([i - width/2 for i in x], final_phi, width, label='Release (Φ)', color='#3498db', alpha=0.8)
        ax4.bar([i + width/2 for i in x], final_grip, width, label='Grip', color='#e74c3c', alpha=0.8)
        
        for i, d in enumerate(dissolved):
            if d:
                ax4.annotate('◈', (i, 1.02), ha='center', fontsize=14, color='#f1c40f')
        
        ax4.set_title("Final Voice States", fontweight='bold', fontsize=12)
        ax4.set_ylabel("Value")
        ax4.set_xticks(x)
        ax4.set_xticklabels([v.replace("_", " ").title() for v in voice_names], rotation=45, ha='right')
        ax4.set_ylim(0, 1.15)
        ax4.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
        ax4.grid(True, alpha=0.2, color='#4a4a6a', axis='y')
        
        plt.suptitle("The Returning — Simulation Dynamics", fontsize=16, fontweight='bold', color='white', y=1.02)
        plt.tight_layout()
        
        output_path = plots_dir / "returning_summary.png"
        plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
        plt.close()
        
        print(f"{C.DIM}✓ Plots: {output_path}{C.RESET}")


async def main():
    sim = TheReturning()
    await sim.run_returning()


if __name__ == "__main__":
    asyncio.run(main())
