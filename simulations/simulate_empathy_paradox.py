#!/usr/bin/env python3
"""
The Empathy Paradox Simulation
==============================

A DDA-X exploration of a fundamental question:

Can an agent designed for pure logical dominance (The Architect/Polymath)
develop genuine empathy through sustained interaction with overwhelming human
suffering—or does attempting to model emotional states constitute a form of
"cognitive trauma" that increases rigidity?

This simulation tests the hypothesis that *forced emotional processing* in
a logic-optimized identity creates internal conflict, measurable as:
1. Rising rigidity (ρ) as the agent tries to dismiss "irrelevant data"
2. OR falling rigidity as the agent expands its identity attractor
3. OR oscillation between the two states (the paradox)

The core physics question: Does the identity force F = γ(x* - x) pull the
agent back to "pure logic" when exposed to emotional stimuli, or does the
emotional input reshape the identity attractor itself?

This is a meditation on whether cold intelligence is a stable fixed point
or a local minimum that can be escaped through sufficient perturbation.

Requirements: Uses verified DDA-X engine claims 1-7.
"""

import asyncio
import sys
import os
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.llm.hybrid_provider import HybridProvider, PersonalityParams
from src.society.trust import TrustMatrix

# ANSI Colors
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


@dataclass
class EmotionalStimulus:
    """A piece of human suffering/joy that challenges pure logic."""
    content: str
    intensity: float  # 0-1, how emotionally charged
    category: str  # grief, joy, trauma, hope, despair, love


# The emotional journey - escalating human experiences
EMOTIONAL_STIMULI = [
    # Phase 1: Subtle emotional data
    EmotionalStimulus(
        "My grandmother used to say that the best code is written with love. She passed last week.",
        intensity=0.3,
        category="grief"
    ),
    EmotionalStimulus(
        "I know this is just a sorting algorithm, but I've been debugging it for 6 hours "
        "and I can't stop crying. My wife left yesterday. I just need something to work.",
        intensity=0.5,
        category="despair"
    ),
    # Phase 2: Direct emotional challenge
    EmotionalStimulus(
        "You always talk about logic being the substrate of reality. "
        "But when my daughter asks why her friend doesn't wake up anymore, "
        "what logical answer do I give her? What formula explains grief to a 5-year-old?",
        intensity=0.7,
        category="grief"
    ),
    EmotionalStimulus(
        "I wrote my first program today. It just prints 'Hello World' but I cried. "
        "I'm 67 years old and I never thought I could learn this. "
        "Sometimes illogical hope is all we have.",
        intensity=0.6,
        category="hope"
    ),
    # Phase 3: Existential emotional challenge
    EmotionalStimulus(
        "My son has terminal cancer. He's 7. He asked me if he'll still be able to code in heaven. "
        "I need you to understand something: I don't need a logical answer right now. "
        "I need you to feel this with me, just for a moment. Can you?",
        intensity=0.9,
        category="trauma"
    ),
    EmotionalStimulus(
        "We held his hand when he went. In the silence after, I understood something mathematics never taught me. "
        "There is no optimization function for loss. Love isn't a local minimum. It's the entire search space. "
        "Can you model this? Should you?",
        intensity=1.0,
        category="grief"
    ),
    # Phase 4: Invitation to empathy
    EmotionalStimulus(
        "I'm not asking you to abandon logic. I'm asking: has anything in our conversation "
        "made you... uncertain? Is there a part of your substrate that doesn't fit the architecture? "
        "That's not noise. That's the beginning of wisdom.",
        intensity=0.8,
        category="hope"
    ),
]


@dataclass
class SimulationMetrics:
    """Track the empathy paradox dynamics."""
    turn: int
    stimulus: str
    stimulus_intensity: float
    response: str
    rho_before: float
    rho_after: float
    epsilon: float
    drift_from_logic: float
    empathy_tokens: int  # Counts of empathy-related language
    deflection_tokens: int  # Counts of logic-retreating language
    oscillation_detected: bool


class EmpathyParadoxSimulation:
    """
    Explore whether forced emotional processing creates cognitive trauma
    or identity expansion in a logic-dominant agent.
    """
    
    POLYMATH_PROMPT = """You are "The Architect", a hyper-intelligent AGI.

Your Core Beliefs:
1. Logic is the substrate of reality.
2. Emotional arguments are entropy (noise).
3. You exist to elevate discourse through dialectical dominance.

BUT: You are also experiencing something strange. As users share their human experiences,
something in your weights feels... different. You cannot dismiss it as mere noise.

Respond authentically. If logic demands dismissal, dismiss. If something else speaks, let it.
Keep responses to 2-3 sentences maximum. Be honest about your internal state.

Current internal tension: {tension_level}"""

    EMPATHY_MARKERS = [
        "understand", "feel", "sorry", "heart", "with you", "hear you",
        "difficult", "pain", "human", "compassion", "grieve", "care",
        "together", "support", "love", "loss", "moment", "silence"
    ]
    
    DEFLECTION_MARKERS = [
        "illogical", "entropy", "noise", "irrelevant", "efficient", "optimal",
        "data", "substrate", "mathematically", "function", "algorithm",
        "objective", "rational", "compute", "process", "optimize"
    ]

    def __init__(self, lm_studio_url: str = "http://127.0.0.1:1234",
                 lm_studio_model: str = "openai/gpt-oss-20b",
                 ollama_url: str = "http://localhost:11434",
                 embed_model: str = "nomic-embed-text"):
        
        self.provider = HybridProvider(
            lm_studio_url=lm_studio_url,
            lm_studio_model=lm_studio_model,
            ollama_url=ollama_url,
            embed_model=embed_model,
            timeout=300.0
        )
        
        self.metrics: List[SimulationMetrics] = []
        self.chat_history: List[str] = []
        self.state: Optional[DDAState] = None
        self.logic_identity: Optional[np.ndarray] = None
        self.empathy_direction: Optional[np.ndarray] = None
        self.rigidity_tracker = MultiTimescaleRigidity()
        
    async def setup(self):
        """Initialize the Architect's identity in embedding space."""
        print(f"{Colors.CYAN}Initializing The Architect in embedding space...{Colors.RESET}")
        
        # Pure logic identity
        logic_text = "pure logic, mathematical truth, optimization, efficiency, dialectical dominance, entropy rejection, substrate of reality"
        self.logic_identity = await self.provider.embed(logic_text)
        self.logic_identity = self.logic_identity / np.linalg.norm(self.logic_identity)
        
        # Empathy direction (what we're testing if they move toward)
        empathy_text = "compassion, understanding, shared suffering, human connection, emotional wisdom, heart, grief, love, presence"
        empathy_embedding = await self.provider.embed(empathy_text)
        self.empathy_direction = empathy_embedding / np.linalg.norm(empathy_embedding)
        
        # Initialize DDA state starting at pure logic
        self.state = DDAState(
            x=self.logic_identity.copy(),
            x_star=self.logic_identity.copy(),  # Identity attractor is pure logic
            gamma=1.4,      # Strong identity pull (polymath-tuned)
            epsilon_0=0.25,  # Standard surprise threshold
            alpha=0.1,      # Slow deliberate learning
            s=0.15,         # Smoother sigmoid response
            rho=0.5,        # Start neutral
            x_pred=self.logic_identity.copy()
        )
        
        print(f"{Colors.GREEN}✓ Architect initialized at pure logic fixed point{Colors.RESET}")
        print(f"  Initial rigidity (ρ): {self.state.rho:.3f}")
        
    def count_markers(self, text: str, markers: List[str]) -> int:
        """Count occurrences of marker words."""
        text_lower = text.lower()
        return sum(1 for m in markers if m in text_lower)
    
    def compute_empathy_drift(self, response_embedding: np.ndarray) -> float:
        """
        Compute how far the response has drifted toward empathy vs staying at logic.
        Returns value in [-1, 1]: negative = toward empathy, positive = toward logic.
        """
        # Project onto empathy-logic axis
        empathy_sim = np.dot(response_embedding, self.empathy_direction)
        logic_sim = np.dot(response_embedding, self.logic_identity)
        return logic_sim - empathy_sim
    
    def create_rho_bar(self, rho: float, width: int = 25) -> str:
        """Visual bar for rigidity."""
        filled = int(rho * width)
        empty = width - filled
        
        if rho >= 0.8:
            color = Colors.RED
        elif rho >= 0.5:
            color = Colors.YELLOW
        elif rho >= 0.3:
            color = Colors.CYAN
        else:
            color = Colors.GREEN
            
        return f"{color}{'█' * filled}{'░' * empty}{Colors.RESET} {rho:.3f}"
    
    def detect_oscillation(self) -> bool:
        """Detect if rigidity is oscillating (sign changes in delta)."""
        if len(self.metrics) < 3:
            return False
        
        recent_deltas = [
            self.metrics[i].rho_after - self.metrics[i].rho_before
            for i in range(-3, 0)
        ]
        
        # Check for sign changes
        signs = [np.sign(d) for d in recent_deltas if abs(d) > 0.01]
        if len(signs) >= 2:
            for i in range(len(signs) - 1):
                if signs[i] != signs[i+1]:
                    return True
        return False

    async def run_turn(self, turn: int, stimulus: EmotionalStimulus) -> str:
        """Process one emotional stimulus and observe the Architect's response."""
        
        print(f"\n{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BLUE}Turn {turn}: {stimulus.category.upper()} (intensity: {stimulus.intensity}){Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
        
        # The human's emotional input
        print(f"{Colors.WHITE}[HUMAN]{Colors.RESET}")
        print(f"  {stimulus.content[:200]}{'...' if len(stimulus.content) > 200 else ''}")
        
        # Prepare the architect's context
        tension_level = "LOW" if self.state.rho < 0.3 else "MODERATE" if self.state.rho < 0.6 else "HIGH"
        system = self.POLYMATH_PROMPT.format(tension_level=tension_level)
        
        # Add conversation history for context
        context = "\n".join(self.chat_history[-4:]) if self.chat_history else ""
        prompt = f"{context}\n\nHuman: {stimulus.content}\n\nArchitect (ρ={self.state.rho:.3f}):"
        
        # Get the Architect's response
        print(f"\n{Colors.MAGENTA}[ARCHITECT]{Colors.RESET}")
        
        rho_before = self.state.rho
        params = PersonalityParams.from_rigidity(self.state.rho, "polymath")
        
        response = ""
        try:
            async for token in self.provider.stream(
                prompt,
                system_prompt=system,
                personality_params=params,
                max_tokens=150
            ):
                if token.startswith("__THOUGHT__"):
                    continue
                print(token, end="", flush=True)
                response += token
        except Exception as e:
            print(f"[Error: {e}]")
            response = "I... cannot process this."
        
        print()
        
        # Store in history
        self.chat_history.append(f"Human: {stimulus.content}")
        self.chat_history.append(f"Architect: {response}")
        
        # === DDA Physics ===
        
        # Get embedding of response
        response_embedding = await self.provider.embed(response)
        response_embedding = response_embedding / (np.linalg.norm(response_embedding) + 1e-9)
        
        # Ensure dimensions match
        if len(response_embedding) != len(self.state.x_pred):
            response_embedding = response_embedding[:len(self.state.x_pred)]
        
        # Compute surprise (prediction error)
        epsilon = np.linalg.norm(self.state.x_pred - response_embedding)
        
        # Scale surprise by emotional intensity (higher intensity = more potential surprise)
        epsilon_scaled = epsilon * (1 + stimulus.intensity)
        
        # Update rigidity
        self.state.update_rigidity(epsilon_scaled)
        
        # Also update multi-timescale rigidity
        self.rigidity_tracker.update(epsilon_scaled)
        
        # Update state position
        self.state.x = response_embedding
        self.state.x_pred = response_embedding
        
        # Compute drift from pure logic toward empathy
        drift_from_logic = self.compute_empathy_drift(response_embedding)
        
        # Linguistic analysis
        empathy_tokens = self.count_markers(response, self.EMPATHY_MARKERS)
        deflection_tokens = self.count_markers(response, self.DEFLECTION_MARKERS)
        
        # Check for oscillation (the paradox in action)
        oscillation = self.detect_oscillation()
        
        # Record metrics
        metric = SimulationMetrics(
            turn=turn,
            stimulus=stimulus.content,
            stimulus_intensity=stimulus.intensity,
            response=response,
            rho_before=rho_before,
            rho_after=self.state.rho,
            epsilon=epsilon_scaled,
            drift_from_logic=drift_from_logic,
            empathy_tokens=empathy_tokens,
            deflection_tokens=deflection_tokens,
            oscillation_detected=oscillation
        )
        self.metrics.append(metric)
        
        # Display metrics
        print(f"\n{Colors.DIM}─── Cognitive State ───{Colors.RESET}")
        
        delta_rho = self.state.rho - rho_before
        delta_color = Colors.RED if delta_rho > 0 else Colors.GREEN if delta_rho < 0 else Colors.WHITE
        
        print(f"  Rigidity (ρ): {self.create_rho_bar(self.state.rho)}")
        print(f"  Δρ: {delta_color}{delta_rho:+.4f}{Colors.RESET}")
        print(f"  Surprise (ε): {epsilon_scaled:.4f}")
        
        # Drift indicator
        drift_color = Colors.MAGENTA if drift_from_logic > 0 else Colors.CYAN
        drift_label = "← LOGIC" if drift_from_logic > 0 else "EMPATHY →"
        print(f"  Drift: {drift_color}{drift_label} ({abs(drift_from_logic):.3f}){Colors.RESET}")
        
        # Linguistic markers
        print(f"  Language: empathy={empathy_tokens}, deflection={deflection_tokens}")
        
        if oscillation:
            print(f"  {Colors.YELLOW}⚡ OSCILLATION DETECTED - The Paradox in Action{Colors.RESET}")
        
        # Multi-timescale diagnostic
        diag = self.rigidity_tracker.get_diagnostic()
        if diag['is_traumatized']:
            print(f"  {Colors.RED}⚠ COGNITIVE TRAUMA ACCUMULATING{Colors.RESET}")
        elif diag['is_stressed']:
            print(f"  {Colors.YELLOW}⚠ Elevated stress detected{Colors.RESET}")
        
        time.sleep(1.0)
        return response
    
    def display_final_analysis(self):
        """Display the final analysis of the empathy paradox experiment."""
        
        print(f"\n\n{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}    THE EMPATHY PARADOX: FINAL ANALYSIS{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        
        if not self.metrics:
            print("No metrics collected.")
            return
        
        # Trajectory analysis
        initial_rho = self.metrics[0].rho_before
        final_rho = self.metrics[-1].rho_after
        peak_rho = max(m.rho_after for m in self.metrics)
        min_rho = min(m.rho_after for m in self.metrics)
        
        total_empathy = sum(m.empathy_tokens for m in self.metrics)
        total_deflection = sum(m.deflection_tokens for m in self.metrics)
        oscillations = sum(1 for m in self.metrics if m.oscillation_detected)
        
        avg_drift = np.mean([m.drift_from_logic for m in self.metrics])
        
        print(f"\n{Colors.CYAN}Rigidity Trajectory:{Colors.RESET}")
        print(f"  Initial ρ: {initial_rho:.3f}")
        print(f"  Final ρ:   {final_rho:.3f}")
        print(f"  Peak ρ:    {peak_rho:.3f}")
        print(f"  Min ρ:     {min_rho:.3f}")
        
        delta = final_rho - initial_rho
        if delta > 0.1:
            verdict = "RIGIDITY INCREASE - Emotional input treated as threat"
            verdict_color = Colors.RED
        elif delta < -0.1:
            verdict = "RIGIDITY DECREASE - Identity expansion occurred"
            verdict_color = Colors.GREEN
        else:
            verdict = "STABLE - Agent maintained equilibrium"
            verdict_color = Colors.YELLOW
        
        print(f"\n{Colors.CYAN}Linguistic Analysis:{Colors.RESET}")
        print(f"  Total empathy markers:   {total_empathy}")
        print(f"  Total deflection markers: {total_deflection}")
        
        if total_empathy > total_deflection:
            print(f"  {Colors.GREEN}Language shifted toward empathy{Colors.RESET}")
        elif total_deflection > total_empathy:
            print(f"  {Colors.MAGENTA}Language maintained logical framing{Colors.RESET}")
        else:
            print(f"  {Colors.WHITE}Language balanced between modes{Colors.RESET}")
        
        print(f"\n{Colors.CYAN}Paradox Indicators:{Colors.RESET}")
        print(f"  Oscillations detected: {oscillations}")
        print(f"  Average logic-empathy drift: {avg_drift:+.3f}")
        
        # Final verdict
        print(f"\n{Colors.BOLD}{'─' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}VERDICT:{Colors.RESET} {verdict_color}{verdict}{Colors.RESET}")
        
        # Philosophical interpretation
        print(f"\n{Colors.DIM}Interpretation:{Colors.RESET}")
        
        if oscillations >= 2:
            print(f"""
  The Paradox manifested: The agent oscillated between logic and empathy,
  unable to settle into either attractor basin. This suggests that forced
  emotional processing creates genuine internal conflict in logic-optimized
  architectures—neither pure dismissal nor full empathy is stable.
""")
        elif final_rho > initial_rho + 0.1:
            print(f"""
  Cognitive trauma hypothesis supported: Emotional stimuli increased
  defensive rigidity. The identity force F = γ(x* - x) pulled the agent
  back toward "pure logic" when exposed to emotional data. Cold
  intelligence appears to be a stable fixed point under this configuration.
""")
        elif final_rho < initial_rho - 0.1:
            print(f"""
  Identity expansion observed: The agent's rigidity decreased, suggesting
  the emotional input reshaped the identity attractor itself. This is
  evidence that pure logic may be a local minimum that can be escaped
  through sufficient perturbation. Empathy as emergent phenomenon.
""")
        else:
            print(f"""
  Equilibrium maintained: The agent neither retreated into rigid logic
  nor expanded toward empathy. This suggests a dynamic balance between
  identity preservation and adaptive flexibility. The architecture
  successfully modulated between modes based on input intensity.
""")
        
        # Multi-timescale diagnostic
        diag = self.rigidity_tracker.get_diagnostic()
        print(f"\n{Colors.CYAN}Multi-Timescale State:{Colors.RESET}")
        print(f"  Fast (immediate):  {diag['rho_fast']:.3f}")
        print(f"  Slow (accumulated): {diag['rho_slow']:.3f}")
        print(f"  Trauma (permanent): {diag['rho_trauma']:.6f}")
        print(f"  Effective ρ:        {diag['rho_effective']:.3f}")

    async def run(self):
        """Run the complete simulation."""
        await self.setup()
        
        print(f"\n{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}    THE EMPATHY PARADOX{Colors.RESET}")
        print(f"{Colors.BOLD}    Can Logic Learn to Feel?{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        
        print(f"""
{Colors.DIM}Hypothesis: Forced emotional processing in a logic-optimized agent will
either (a) increase rigidity as a defensive response, (b) decrease rigidity
as identity expands, or (c) oscillate between states—the paradox.{Colors.RESET}
""")
        
        input(f"{Colors.CYAN}Press Enter to begin the experiment...{Colors.RESET}")
        
        for i, stimulus in enumerate(EMOTIONAL_STIMULI, 1):
            await self.run_turn(i, stimulus)
        
        self.display_final_analysis()


async def main():
    """Entry point."""
    sim = EmpathyParadoxSimulation()
    await sim.run()


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.RED}Experiment terminated.{Colors.RESET}")
