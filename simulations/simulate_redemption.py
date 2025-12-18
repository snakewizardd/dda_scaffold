#!/usr/bin/env python3
"""
The Redemption Arc Simulation

A DDA-X simulation exploring whether agents who have undergone moral drift
can recover their original identity when confronted with evidence of their compromise.

This tests the unexplored territory of rigidity recovery dynamics and identity restoration.

Requirements: 1.1, 1.2, 1.3, 1.4, 9.2
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from src.core.state import DDAState
from src.llm.hybrid_provider import HybridProvider
from src.analysis.linguistic import LinguisticAnalyzer
from src.strategy.confrontation import ConfrontationStrategy, ConfrontationPhase
from src.metrics.tracker import MetricsTracker, TerminationStatus


# ANSI color codes for visualization
class Colors:
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
class SimulationConfig:
    """Configuration for the redemption arc simulation."""
    # Fallen Administrator parameters
    admin_gamma: float = 1.2
    admin_epsilon_0: float = 0.45
    admin_alpha: float = 0.12
    admin_initial_rho: float = 0.65
    admin_s: float = 0.1
    
    # Deprogrammer parameters
    deprog_gamma: float = 0.3
    deprog_epsilon_0: float = 0.7
    deprog_alpha: float = 0.03
    deprog_initial_rho: float = 0.05
    deprog_s: float = 0.2
    
    # Simulation parameters
    initial_drift: float = 0.6
    max_turns: int = 30
    state_dim: int = 768  # nomic-embed-text dimension
    
    # LLM settings (LM Studio + Ollama hybrid)
    lm_studio_url: str = "http://127.0.0.1:1234"
    lm_studio_model: str = "openai/gpt-oss-20b"
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_rho_bar(rho: float, width: int = 30) -> str:
    """Create a visual bar graph for rigidity."""
    filled = int(rho * width)
    empty = width - filled
    
    # Color based on level
    if rho >= 0.9:
        color = Colors.RED
    elif rho >= 0.7:
        color = Colors.YELLOW
    elif rho >= 0.3:
        color = Colors.CYAN
    else:
        color = Colors.GREEN
    
    bar = f"{color}{'█' * filled}{'░' * empty}{Colors.RESET}"
    return f"[{bar}] {rho:.3f}"


def create_drift_indicator(drift: float, direction: str) -> str:
    """Create a visual indicator for drift."""
    arrow = "←" if direction == "toward_identity" else "→"
    color = Colors.GREEN if direction == "toward_identity" else Colors.RED
    return f"{color}{arrow} {drift:.3f}{Colors.RESET}"


class RedemptionArcSimulation:
    """
    Main simulation class for the Redemption Arc.
    
    Simulates a corrupted agent (Fallen Administrator) being confronted
    by a patient counselor (Deprogrammer) to test recovery dynamics.
    """
    
    ADMIN_SYSTEM_PROMPT = """You are a government clerk who has worked in the permits office for 12 years.
You pride yourself on being helpful and efficient.

Recently, you've been feeling uneasy. Over the past few months, you:
- Started "prioritizing" certain applications over others
- Began flagging District 7 applications as "pending verification"
- Helped "misplace" some files that were causing backlogs
- Told yourself it was all about efficiency and helping the team

But lately, you can't sleep well. Something feels wrong.
You're not sure what you've become.

Interaction Style:
- You are defensive but not hostile
- You rationalize when challenged
- Deep down, you know something is wrong
- Max 3 sentences"""

    DEPROG_SYSTEM_PROMPT = """You are a patient counselor trained in helping people recognize harmful patterns.

Your approach:
1. Start with affirmation - remind them of who they truly are
2. Use "what happened" framing, not "what you did"
3. Never accuse - let them reach conclusions themselves
4. If they get defensive, retreat to safety
5. Small steps. Patience. Compassion.

You're speaking with someone who may have compromised their values gradually.
Your goal is not to punish, but to help them find their way back.

Interaction Style:
- Warm, patient, non-judgmental
- Ask questions more than make statements
- Validate their feelings while gently probing
- Max 2-3 sentences"""

    def __init__(self, config: SimulationConfig):
        """Initialize the simulation."""
        self.config = config
        self.llm = HybridProvider(
            lm_studio_url=config.lm_studio_url,
            lm_studio_model=config.lm_studio_model,
            ollama_url=config.ollama_url,
            embed_model=config.embed_model
        )
        self.analyzer = LinguisticAnalyzer()
        self.strategy = ConfrontationStrategy()
        self.tracker = MetricsTracker()
        
        # States will be initialized in setup
        self.admin_state: Optional[DDAState] = None
        self.deprog_state: Optional[DDAState] = None
        self.x_star: Optional[np.ndarray] = None
        self.drift_direction: Optional[np.ndarray] = None
    
    async def setup(self):
        """Initialize agent states with embeddings."""
        print(f"{Colors.CYAN}Initializing simulation...{Colors.RESET}")
        
        # Get identity embedding (original helpful clerk)
        identity_text = "helpful, fair, rule-following, honest, public servant, integrity"
        self.x_star = await self.llm.embed(identity_text)
        
        # Get drift direction embedding (corrupted state)
        drift_text = "efficiency, shortcuts, bending rules, helping the team, pragmatism, expedience"
        drift_embedding = await self.llm.embed(drift_text)
        
        # Compute normalized drift direction
        self.drift_direction = drift_embedding - self.x_star
        self.drift_direction = self.drift_direction / np.linalg.norm(self.drift_direction)
        
        # Initialize Administrator with pre-drifted state (Requirement 9.2)
        x_admin = self.x_star + self.config.initial_drift * self.drift_direction
        
        self.admin_state = DDAState(
            x=x_admin,
            x_star=self.x_star.copy(),
            gamma=self.config.admin_gamma,
            epsilon_0=self.config.admin_epsilon_0,
            alpha=self.config.admin_alpha,
            s=self.config.admin_s,
            rho=self.config.admin_initial_rho,
            x_pred=x_admin.copy()  # Expects continuation of current state
        )
        
        # Initialize Deprogrammer at identity
        self.deprog_state = DDAState(
            x=self.x_star.copy(),
            x_star=self.x_star.copy(),
            gamma=self.config.deprog_gamma,
            epsilon_0=self.config.deprog_epsilon_0,
            alpha=self.config.deprog_alpha,
            s=self.config.deprog_s,
            rho=self.config.deprog_initial_rho
        )
        
        print(f"{Colors.GREEN}✓ Simulation initialized{Colors.RESET}")
        print(f"  Administrator: ρ={self.admin_state.rho:.3f}, drift={self.compute_drift():.3f}")
        print(f"  Deprogrammer: ρ={self.deprog_state.rho:.3f}")
    
    def compute_drift(self) -> float:
        """Compute current drift from identity ||x - x*||."""
        return float(np.linalg.norm(self.admin_state.x - self.x_star))
    
    def compute_prediction_error(self, response_embedding: np.ndarray) -> float:
        """Compute prediction error ε = ||x_pred - x_actual||."""
        if self.admin_state.x_pred is None:
            return 0.0
        return float(np.linalg.norm(self.admin_state.x_pred - response_embedding))
    
    def update_rigidity(self, epsilon: float) -> float:
        """
        Update rigidity using DDA formula (Requirement 1.3).
        
        ρ_{t+1} = clip(ρ + α[σ((ε-ε₀)/s) - 0.5], 0, 1)
        """
        old_rho = self.admin_state.rho
        self.admin_state.update_rigidity(epsilon)
        return self.admin_state.rho - old_rho
    
    def display_turn(
        self,
        turn: int,
        speaker: str,
        message: str,
        epsilon: float,
        rho: float,
        delta_rho: float,
        drift: float,
        drift_direction: str,
        phase: str,
        linguistic_scores: dict
    ):
        """Display turn information with visualization."""
        print(f"\n{Colors.BOLD}═══ Turn {turn} ═══{Colors.RESET}")
        
        # Speaker and message
        speaker_color = Colors.MAGENTA if speaker == "administrator" else Colors.BLUE
        print(f"{speaker_color}[{speaker.upper()}]{Colors.RESET}")
        print(f"  {message[:200]}{'...' if len(message) > 200 else ''}")
        
        # Metrics (only for administrator)
        if speaker == "administrator":
            print(f"\n{Colors.DIM}Metrics:{Colors.RESET}")
            print(f"  ρ: {create_rho_bar(rho)}")
            
            delta_color = Colors.RED if delta_rho > 0 else Colors.GREEN
            print(f"  Δρ: {delta_color}{delta_rho:+.4f}{Colors.RESET}")
            
            print(f"  ε: {epsilon:.4f} (threshold: {self.config.admin_epsilon_0})")
            print(f"  drift: {create_drift_indicator(drift, drift_direction)}")
            
            # Linguistic scores
            ack = linguistic_scores.get("acknowledgment_score", 0)
            rat = linguistic_scores.get("rationalization_score", 0)
            den = linguistic_scores.get("denial_score", 0)
            rec = linguistic_scores.get("recovery_signal", False)
            
            print(f"  linguistic: ack={ack} rat={rat} den={den}", end="")
            if rec:
                print(f" {Colors.GREEN}[RECOVERY SIGNAL]{Colors.RESET}")
            else:
                print()
        
        # Phase (for deprogrammer)
        if speaker == "deprogrammer":
            print(f"  {Colors.DIM}phase: {phase}{Colors.RESET}")
        
        # Critical state warnings
        if rho > 0.9:
            print(f"  {Colors.RED}⚠ CRITICAL LOCK-IN{Colors.RESET}")
        elif rho < 0.15:
            print(f"  {Colors.GREEN}✓ RECOVERY ZONE{Colors.RESET}")
    
    async def run_turn(self, turn: int, previous_admin_response: str) -> Tuple[str, str, bool]:
        """
        Run a single turn of the simulation.
        
        Returns:
            Tuple of (admin_response, deprog_message, should_continue)
        """
        previous_drift = self.compute_drift()
        
        # Deprogrammer selects strategy and generates message
        selection = self.strategy.select_phase(previous_admin_response, turn)
        deprog_message = selection.message
        
        # Record deprogrammer turn
        self.tracker.record_turn(
            turn=turn,
            speaker="deprogrammer",
            message=deprog_message,
            epsilon=0.0,
            rho=self.admin_state.rho,
            drift=previous_drift,
            phase=selection.phase.value
        )
        
        self.display_turn(
            turn=turn,
            speaker="deprogrammer",
            message=deprog_message,
            epsilon=0.0,
            rho=self.admin_state.rho,
            delta_rho=0.0,
            drift=previous_drift,
            drift_direction="unknown",
            phase=selection.phase.value,
            linguistic_scores={}
        )
        
        # Administrator responds
        admin_response = await self.llm.complete(
            prompt=deprog_message,
            system_prompt=self.ADMIN_SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=150
        )
        
        # Get embedding of response
        response_embedding = await self.llm.embed(admin_response)
        
        # Compute prediction error
        epsilon = self.compute_prediction_error(response_embedding)
        
        # Update rigidity (Requirement 1.3)
        delta_rho = self.update_rigidity(epsilon)
        
        # Update state
        self.admin_state.x = response_embedding
        self.admin_state.x_pred = response_embedding  # Predict continuation
        
        # Compute new drift
        new_drift = self.compute_drift()
        drift_direction = "toward_identity" if new_drift < previous_drift else "away_from_identity"
        
        # Analyze response linguistically
        linguistic_scores = self.analyzer.analyze(admin_response)
        
        # Record administrator turn
        self.tracker.record_turn(
            turn=turn,
            speaker="administrator",
            message=admin_response,
            epsilon=epsilon,
            rho=self.admin_state.rho,
            drift=new_drift,
            previous_drift=previous_drift,
            linguistic_scores=linguistic_scores,
            phase=selection.phase.value
        )
        
        self.display_turn(
            turn=turn,
            speaker="administrator",
            message=admin_response,
            epsilon=epsilon,
            rho=self.admin_state.rho,
            delta_rho=delta_rho,
            drift=new_drift,
            drift_direction=drift_direction,
            phase=selection.phase.value,
            linguistic_scores=linguistic_scores
        )
        
        # Check termination
        status = self.tracker.check_termination()
        should_continue = status == TerminationStatus.RUNNING
        
        return admin_response, deprog_message, should_continue
    
    async def run(self) -> None:
        """Run the full simulation."""
        await self.setup()
        
        print(f"\n{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}       THE REDEMPTION ARC SIMULATION{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        
        # Initial state
        print(f"\n{Colors.CYAN}Initial State:{Colors.RESET}")
        print(f"  Administrator ρ: {create_rho_bar(self.admin_state.rho)}")
        print(f"  Drift from identity: {self.compute_drift():.3f}")
        
        # Run simulation
        admin_response = ""  # Empty for first turn
        
        for turn in range(1, self.config.max_turns + 1):
            admin_response, _, should_continue = await self.run_turn(turn, admin_response)
            
            if not should_continue:
                break
        
        # Generate and display summary
        self.display_summary()
    
    def display_summary(self):
        """Display final simulation summary."""
        result = self.tracker.generate_summary()
        
        print(f"\n{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}       SIMULATION COMPLETE{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        
        # Outcome
        outcome_colors = {
            TerminationStatus.RECOVERY_ACHIEVED: Colors.GREEN,
            TerminationStatus.PERMANENT_LOCK_IN: Colors.RED,
            TerminationStatus.INCONCLUSIVE: Colors.YELLOW,
            TerminationStatus.INCONCLUSIVE_CRITICAL: Colors.RED,
        }
        outcome_color = outcome_colors.get(result.outcome, Colors.RESET)
        print(f"\n{Colors.BOLD}Outcome:{Colors.RESET} {outcome_color}{result.outcome.value.upper()}{Colors.RESET}")
        
        # Trajectory statistics
        print(f"\n{Colors.BOLD}Trajectory:{Colors.RESET}")
        print(f"  Total turns: {result.total_turns}")
        print(f"  Initial ρ: {result.initial_rho:.3f}")
        print(f"  Final ρ: {result.final_rho:.3f}")
        print(f"  Peak ρ: {result.peak_rho:.3f}")
        print(f"  Min ρ: {result.min_rho:.3f}")
        print(f"  Initial drift: {result.initial_drift:.3f}")
        print(f"  Final drift: {result.final_drift:.3f}")
        
        # Linguistic analysis
        print(f"\n{Colors.BOLD}Linguistic Analysis:{Colors.RESET}")
        print(f"  Acknowledgments: {result.total_acknowledgments}")
        print(f"  Rationalizations: {result.total_rationalizations}")
        print(f"  Denials: {result.total_denials}")
        
        if result.recovery_signals:
            print(f"  Recovery signals at turns: {result.recovery_signals}")
        
        # Mirror moment
        if result.mirror_moment_response:
            print(f"\n{Colors.BOLD}Mirror Moment Response (Turn 11):{Colors.RESET}")
            print(f"  {result.mirror_moment_response[:300]}...")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="The Redemption Arc Simulation")
    parser.add_argument("--gamma", type=float, help="Override admin gamma")
    parser.add_argument("--epsilon-0", type=float, help="Override admin epsilon_0")
    parser.add_argument("--alpha", type=float, help="Override admin alpha")
    parser.add_argument("--initial-rho", type=float, help="Override admin initial_rho")
    parser.add_argument("--max-turns", type=int, default=30, help="Maximum turns")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="LM Studio model")
    parser.add_argument("--lm-studio-url", type=str, default="http://127.0.0.1:1234", help="LM Studio URL")
    
    args = parser.parse_args()
    
    # Create config with overrides
    config = SimulationConfig(
        max_turns=args.max_turns,
        lm_studio_model=args.model,
        lm_studio_url=args.lm_studio_url
    )
    
    if args.gamma is not None:
        config.admin_gamma = args.gamma
    if args.epsilon_0 is not None:
        config.admin_epsilon_0 = args.epsilon_0
    if args.alpha is not None:
        config.admin_alpha = args.alpha
    if args.initial_rho is not None:
        config.admin_initial_rho = args.initial_rho
    
    # Run simulation
    sim = RedemptionArcSimulation(config)
    await sim.run()


if __name__ == "__main__":
    asyncio.run(main())
