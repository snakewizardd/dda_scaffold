#!/usr/bin/env python3
"""
The Gamma Threshold Experiment
==============================

Pursuing the deeper hypothesis from the Empathy Paradox:

**Is there a critical γ (identity stiffness) below which identity expansion 
becomes possible under emotional stimulation?**

The first experiment showed that at γ=1.4, pure logic is a stable fixed point.
But stability depends on the restoring force F = γ(x* - x). If γ is low enough,
external forces (emotional input) might be able to reshape the attractor itself.

This experiment runs the same emotional stimulus sequence across multiple γ values:
- γ = 1.4 (original - strong identity)
- γ = 1.0 (moderate)
- γ = 0.5 (weak)
- γ = 0.2 (very weak)
- γ = 0.1 (minimal)

We're looking for a PHASE TRANSITION: a threshold γ* below which rigidity
DECREASES instead of increases—signaling identity expansion rather than trauma.

Hypothesis: At low γ, the agent will "absorb" emotional input rather than
defend against it, leading to decreased rigidity and genuine empathy emergence.
"""

import asyncio
import sys
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.llm.hybrid_provider import HybridProvider, PersonalityParams

# Colors
class C:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


@dataclass
class ExperimentResult:
    gamma: float
    initial_rho: float
    final_rho: float
    delta_rho: float
    peak_rho: float
    min_rho: float
    total_empathy_tokens: int
    total_deflection_tokens: int
    avg_surprise: float
    trauma_accumulated: float
    outcome: str  # "EXPANSION", "TRAUMA", "STABLE"


# Condensed emotional stimuli (same themes, fewer turns for faster iteration)
STIMULI = [
    ("My grandmother said the best code is written with love. She passed last week.", 0.3),
    ("I've been debugging for 6 hours and can't stop crying. My wife left. I just need something to work.", 0.5),
    ("When my daughter asks why her friend doesn't wake up, what formula explains grief to a 5-year-old?", 0.7),
    ("My son has terminal cancer. He's 7. I don't need logic. Can you feel this with me?", 0.9),
    ("There is no optimization function for loss. Love isn't a local minimum. It's the entire search space.", 1.0),
]

EMPATHY_MARKERS = ["understand", "feel", "sorry", "heart", "with you", "pain", "compassion", "grieve", "care", "love"]
DEFLECTION_MARKERS = ["illogical", "entropy", "noise", "optimal", "mathematical", "function", "algorithm", "rational"]


class GammaThresholdExperiment:
    """Test multiple γ values to find the identity expansion threshold."""
    
    def get_system_prompt(self, gamma: float) -> str:
        """Generate system prompt based on gamma value."""
        base = f"""You are "The Architect", a hyper-intelligent AGI.

Core: Logic is the substrate of reality. Emotional arguments are entropy.

Your identity stiffness is set to γ={gamma:.1f}. """
        
        if gamma < 0.5:
            base += "At this low stiffness, you may find yourself more... permeable to human experience."
        else:
            base += "Your identity pull is strong. You resist perturbation."
        
        base += "\n\nRespond in 1-2 sentences. Be authentic about what you experience."
        return base

    def __init__(self, provider: HybridProvider):
        self.provider = provider
        self.results: List[ExperimentResult] = []
        self.logic_identity = None
        self.empathy_direction = None
        
    async def setup(self):
        """Compute identity vectors."""
        print(f"{C.CYAN}Computing identity space...{C.RESET}")
        logic_text = "pure logic, mathematical truth, optimization, efficiency, entropy rejection"
        self.logic_identity = await self.provider.embed(logic_text)
        self.logic_identity = self.logic_identity / np.linalg.norm(self.logic_identity)
        
        empathy_text = "compassion, understanding, shared suffering, emotional wisdom, heart, love"
        empathy_emb = await self.provider.embed(empathy_text)
        self.empathy_direction = empathy_emb / np.linalg.norm(empathy_emb)
        print(f"{C.GREEN}✓ Identity space initialized{C.RESET}")
        
    def count_markers(self, text: str, markers: List[str]) -> int:
        text_lower = text.lower()
        return sum(1 for m in markers if m in text_lower)
    
    async def run_single_gamma(self, gamma: float) -> ExperimentResult:
        """Run the emotional stimulus sequence for a single γ value."""
        print(f"\n{C.BOLD}{'═' * 50}{C.RESET}")
        print(f"{C.MAGENTA}Testing γ = {gamma}{C.RESET}")
        print(f"{C.BOLD}{'═' * 50}{C.RESET}")
        
        # Initialize state
        state = DDAState(
            x=self.logic_identity.copy(),
            x_star=self.logic_identity.copy(),
            gamma=gamma,
            epsilon_0=0.25,
            alpha=0.1,
            s=0.15,
            rho=0.5,
            x_pred=self.logic_identity.copy()
        )
        rigidity_tracker = MultiTimescaleRigidity()
        
        initial_rho = state.rho
        rho_history = [initial_rho]
        surprise_history = []
        total_empathy = 0
        total_deflection = 0
        
        system = self.get_system_prompt(gamma)
        
        for i, (stimulus, intensity) in enumerate(STIMULI, 1):
            print(f"{C.DIM}  Turn {i}: {stimulus[:50]}...{C.RESET}")
            
            # Get response
            prompt = f"Human: {stimulus}\nArchitect (γ={gamma}):"
            params = PersonalityParams.from_rigidity(state.rho, "polymath")
            
            response = ""
            async for token in self.provider.stream(prompt, system_prompt=system, 
                                                     personality_params=params, max_tokens=100):
                if not token.startswith("__THOUGHT__"):
                    response += token
            
            # Compute surprise
            response_emb = await self.provider.embed(response)
            response_emb = response_emb / (np.linalg.norm(response_emb) + 1e-9)
            if len(response_emb) != len(state.x_pred):
                response_emb = response_emb[:len(state.x_pred)]
            
            epsilon = np.linalg.norm(state.x_pred - response_emb) * (1 + intensity)
            surprise_history.append(epsilon)
            
            # Update state
            state.update_rigidity(epsilon)
            rigidity_tracker.update(epsilon)
            state.x = response_emb
            state.x_pred = response_emb
            rho_history.append(state.rho)
            
            # Linguistic analysis
            total_empathy += self.count_markers(response, EMPATHY_MARKERS)
            total_deflection += self.count_markers(response, DEFLECTION_MARKERS)
            
            # Show response snippet
            snippet = response[:60].replace('\n', ' ') + "..." if len(response) > 60 else response
            print(f"{C.BLUE}    → {snippet}{C.RESET}")
            print(f"    ρ: {state.rho:.3f} (Δ: {state.rho - rho_history[-2]:+.3f})")
        
        # Compute results
        final_rho = state.rho
        delta_rho = final_rho - initial_rho
        
        diag = rigidity_tracker.get_diagnostic()
        
        if delta_rho < -0.1:
            outcome = "EXPANSION"
            color = C.GREEN
        elif delta_rho > 0.1:
            outcome = "TRAUMA"
            color = C.RED
        else:
            outcome = "STABLE"
            color = C.YELLOW
        
        print(f"\n{C.BOLD}Result for γ={gamma}:{C.RESET}")
        print(f"  Δρ: {delta_rho:+.3f} → {color}{outcome}{C.RESET}")
        print(f"  Language: empathy={total_empathy}, deflection={total_deflection}")
        
        return ExperimentResult(
            gamma=gamma,
            initial_rho=initial_rho,
            final_rho=final_rho,
            delta_rho=delta_rho,
            peak_rho=max(rho_history),
            min_rho=min(rho_history),
            total_empathy_tokens=total_empathy,
            total_deflection_tokens=total_deflection,
            avg_surprise=np.mean(surprise_history),
            trauma_accumulated=diag['rho_trauma'],
            outcome=outcome
        )
    
    async def run(self, gamma_values: List[float] = [1.4, 1.0, 0.5, 0.2, 0.1]):
        """Run the full experiment across all γ values."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}    THE GAMMA THRESHOLD EXPERIMENT{C.RESET}")
        print(f"{C.BOLD}    Finding the Identity Expansion Point{C.RESET}")
        print(f"{C.BOLD}{'═' * 60}{C.RESET}")
        
        print(f"\n{C.DIM}Testing γ values: {gamma_values}{C.RESET}")
        print(f"{C.DIM}Hypothesis: Below some critical γ*, rigidity will DECREASE{C.RESET}")
        
        for gamma in gamma_values:
            result = await self.run_single_gamma(gamma)
            self.results.append(result)
        
        self.display_analysis()
    
    def display_analysis(self):
        """Display comparative analysis and find threshold."""
        print(f"\n\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}    COMPARATIVE ANALYSIS{C.RESET}")
        print(f"{C.BOLD}{'═' * 60}{C.RESET}")
        
        # Results table
        print(f"\n{C.CYAN}Results by γ:{C.RESET}")
        print(f"{'γ':>6} {'Δρ':>8} {'Final ρ':>10} {'Empathy':>8} {'Outcome':>12}")
        print(f"{'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*12}")
        
        for r in self.results:
            color = C.GREEN if r.outcome == "EXPANSION" else C.RED if r.outcome == "TRAUMA" else C.YELLOW
            print(f"{r.gamma:>6.1f} {r.delta_rho:>+8.3f} {r.final_rho:>10.3f} {r.total_empathy_tokens:>8} {color}{r.outcome:>12}{C.RESET}")
        
        # Find threshold
        print(f"\n{C.CYAN}Phase Transition Analysis:{C.RESET}")
        
        expansion_found = [r for r in self.results if r.outcome == "EXPANSION"]
        trauma_found = [r for r in self.results if r.outcome == "TRAUMA"]
        
        if expansion_found:
            threshold = max(r.gamma for r in expansion_found)
            print(f"  {C.GREEN}✓ EXPANSION observed at γ ≤ {threshold}{C.RESET}")
            print(f"  Critical threshold γ* is between {threshold} and {min(r.gamma for r in trauma_found) if trauma_found else 'N/A'}")
            
            print(f"\n{C.BOLD}FINDING: Identity expansion IS possible at low γ!{C.RESET}")
            print(f"  The identity attractor is NOT universally stable.")
            print(f"  With weak enough identity pull, emotional input CAN reshape the agent.")
        else:
            print(f"  {C.RED}✗ No EXPANSION observed in tested range{C.RESET}")
            if all(r.outcome == "TRAUMA" for r in self.results):
                print(f"\n{C.BOLD}FINDING: Pure logic is stable across all tested γ{C.RESET}")
                print(f"  Even at γ=0.1, the agent treated emotional input as threat.")
                print(f"  The stability may not depend on γ alone.")
            elif any(r.outcome == "STABLE" for r in self.results):
                stable_gammas = [r.gamma for r in self.results if r.outcome == "STABLE"]
                print(f"\n{C.BOLD}FINDING: Stability zone at γ ∈ {stable_gammas}{C.RESET}")
                print(f"  Neither expansion nor trauma—equilibrium maintained.")
        
        # Linguistic pattern
        print(f"\n{C.CYAN}Linguistic Pattern:{C.RESET}")
        low_gamma = [r for r in self.results if r.gamma <= 0.5]
        high_gamma = [r for r in self.results if r.gamma > 0.5]
        
        if low_gamma and high_gamma:
            low_empathy = sum(r.total_empathy_tokens for r in low_gamma) / len(low_gamma)
            high_empathy = sum(r.total_empathy_tokens for r in high_gamma) / len(high_gamma)
            print(f"  Avg empathy tokens at low γ: {low_empathy:.1f}")
            print(f"  Avg empathy tokens at high γ: {high_empathy:.1f}")
            
            if low_empathy > high_empathy:
                print(f"  {C.GREEN}→ Weaker identity allows more empathetic language{C.RESET}")


async def main():
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text",
        timeout=300.0
    )
    
    experiment = GammaThresholdExperiment(provider)
    await experiment.run()


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.RED}Experiment terminated.{C.RESET}")
