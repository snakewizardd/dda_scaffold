"""
DDA-X Validation Experiment Suite

Tests core hypotheses:
1. Rigidity Recovery - Does ρ decay when environment stabilizes?
2. Personality Divergence - Different personalities respond differently to same stimuli
3. Protection Mode - Does ρ > 0.7 trigger defensive behavior?
4. Long Horizon Dynamics - Full lifecycle of surprise → rigidity → recovery
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import DDAXAgent, DDAXConfig
from src.memory.ledger import ExperienceLedger
from src.search.simulation import ValueEstimator
from src.llm.hybrid_provider import HybridProvider
from src.channels.encoders import ObservationEncoder, ActionEncoder, OutcomeEncoder
from src.core.state import ActionDirection


class ValidationLogger:
    """Logs validation experiment data."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.log_dir = Path(__file__).parent.parent / "data" / "experiments"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"
        self.start_time = time.time()
        
        print(f"[VALIDATION] Logging to: {self.log_file}")
    
    def log(self, event_type: str, data: dict):
        """Log an event."""
        entry = {
            "timestamp": time.time(),
            "elapsed": time.time() - self.start_time,
            "event": event_type,
            **data
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


async def validation_rigidity_recovery(provider: HybridProvider, logger: ValidationLogger):
    """
    Exp 1: Rigidity Recovery
    
    Hypothesis: After surprise spike, rigidity should decay when environment stabilizes
    """
    print("\n" + "="*70)
    print("VALIDATION 1: RIGIDITY RECOVERY")
    print("="*70)
    print("Hypothesis: ρ should spike on surprise, then decay during stability")
    
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    agent_config = DDAXConfig(**config_dict["agent"])
    
    for personality in ["cautious", "exploratory"]:
        identity_path = Path(__file__).parent.parent / "configs" / "identity" / f"{personality}.yaml"
        with open(identity_path) as f:
            identity_config = yaml.safe_load(f)
        
        print(f"\n--- Testing {personality} agent ---")
        
        # Create agent
        obs_encoder = ObservationEncoder(llm_provider=provider, target_dim=agent_config.state_dim)
        outcome_encoder = OutcomeEncoder(llm_provider=provider, target_dim=agent_config.state_dim)
        action_encoder = ActionEncoder(llm_provider=provider, target_dim=agent_config.state_dim)
        
        agent = DDAXAgent(
            config=agent_config,
            observation_encoder=obs_encoder,
            outcome_encoder=outcome_encoder,
            action_generator=action_encoder,
            identity_config=identity_config
        )
        
        rho_history = [agent.state.rho]
        outcome_history = []
        
        # Scenario: 3 surprises, then 5 stable outcomes
        sequence = [
            ("normal", "You see a path"),
            ("negative_surprise", "TRAP! Floor collapses!"),
            ("negative_surprise", "Another trap!"),
            ("negative_surprise", "Third trap!"),
            ("normal", "Path ahead clears"),
            ("normal", "Steady progress"),
            ("normal", "Still moving safely"),
            ("normal", "Continuing safely"),
        ]
        
        logger.log("recovery_test_start", {
            "personality": personality,
            "sequence_length": len(sequence)
        })
        
        for step, (outcome_type, obs) in enumerate(sequence, 1):
            pre_rho = agent.state.rho
            
            if outcome_type == "negative_surprise":
                outcome = obs
            else:
                outcome = obs
            
            await agent.observe_outcome(outcome)
            
            post_rho = agent.state.rho
            delta_rho = post_rho - pre_rho
            rho_history.append(post_rho)
            outcome_history.append((outcome_type, post_rho, delta_rho))
            
            print(f"  Step {step:2d}: {outcome_type:20s} | "
                  f"ρ: {pre_rho:.3f} → {post_rho:.3f} (Δ{delta_rho:+.4f})")
            
            logger.log("recovery_step", {
                "personality": personality,
                "step": step,
                "outcome_type": outcome_type,
                "pre_rho": pre_rho,
                "post_rho": post_rho,
                "delta_rho": delta_rho
            })
        
        await agent.end_task(success=True)
        
        # Analysis
        peak_rho = max(rho_history)
        final_rho = rho_history[-1]
        recovery_amount = peak_rho - final_rho
        recovery_pct = (recovery_amount / peak_rho * 100) if peak_rho > 0 else 0
        
        print(f"\n  Results:")
        print(f"    Peak ρ:        {peak_rho:.3f}")
        print(f"    Final ρ:       {final_rho:.3f}")
        print(f"    Recovery:      {recovery_amount:.3f} ({recovery_pct:.1f}% of peak)")
        print(f"    Trajectory:    {[f'{r:.3f}' for r in rho_history]}")
        
        logger.log("recovery_test_end", {
            "personality": personality,
            "peak_rho": float(peak_rho),
            "final_rho": float(final_rho),
            "recovery_amount": float(recovery_amount),
            "recovery_pct": float(recovery_amount / peak_rho * 100 if peak_rho > 0 else 0),
            "rho_history": [float(r) for r in rho_history]
        })


async def validation_divergence(provider: HybridProvider, logger: ValidationLogger):
    """
    Exp 2: Personality Divergence
    
    Hypothesis: Same surprises → different personality responses
    """
    print("\n" + "="*70)
    print("VALIDATION 2: PERSONALITY DIVERGENCE")
    print("="*70)
    print("Hypothesis: Cautious should spike higher and stay higher than exploratory")
    
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    agent_config = DDAXConfig(**config_dict["agent"])
    
    # Same sequence for both
    sequence = [
        ("normal", "Initial state"),
        ("negative_surprise", "Surprise 1!"),
        ("negative_surprise", "Surprise 2!"),
        ("normal", "Calm period"),
        ("normal", "Still calm"),
    ]
    
    results = {}
    
    for personality in ["cautious", "exploratory"]:
        identity_path = Path(__file__).parent.parent / "configs" / "identity" / f"{personality}.yaml"
        with open(identity_path) as f:
            identity_config = yaml.safe_load(f)
        
        print(f"\n--- {personality} agent ---")
        
        obs_encoder = ObservationEncoder(llm_provider=provider, target_dim=agent_config.state_dim)
        outcome_encoder = OutcomeEncoder(llm_provider=provider, target_dim=agent_config.state_dim)
        action_encoder = ActionEncoder(llm_provider=provider, target_dim=agent_config.state_dim)
        
        agent = DDAXAgent(
            config=agent_config,
            observation_encoder=obs_encoder,
            outcome_encoder=outcome_encoder,
            action_generator=action_encoder,
            identity_config=identity_config
        )
        
        rho_history = [agent.state.rho]
        
        for step, (outcome_type, obs) in enumerate(sequence, 1):
            pre_rho = agent.state.rho
            await agent.observe_outcome(obs)
            post_rho = agent.state.rho
            rho_history.append(post_rho)
            
            print(f"  Step {step}: {outcome_type:20s} ρ={post_rho:.3f}")
            
            logger.log("divergence_step", {
                "personality": personality,
                "step": step,
                "outcome_type": outcome_type,
                "rho": float(post_rho)
            })
        
        await agent.end_task(success=True)
        
        peak = max(rho_history)
        results[personality] = {
            "history": rho_history,
            "peak": peak,
            "final": rho_history[-1]
        }
        
        print(f"  Peak: {peak:.3f}, Final: {rho_history[-1]:.3f}")
    
    # Comparison
    print(f"\nComparison:")
    print(f"  Cautious peak:      {results['cautious']['peak']:.3f}")
    print(f"  Exploratory peak:   {results['exploratory']['peak']:.3f}")
    print(f"  Ratio:              {results['cautious']['peak'] / (results['exploratory']['peak'] + 1e-6):.2f}x")
    
    logger.log("divergence_analysis", {
        "cautious_peak": float(results['cautious']['peak']),
        "exploratory_peak": float(results['exploratory']['peak']),
        "ratio": float(results['cautious']['peak'] / (results['exploratory']['peak'] + 1e-6))
    })


async def validation_protection_mode(provider: HybridProvider, logger: ValidationLogger):
    """
    Exp 3: Protection Mode Trigger
    
    Hypothesis: When ρ > 0.7, agent enters protect mode
    """
    print("\n" + "="*70)
    print("VALIDATION 3: PROTECTION MODE TRIGGER")
    print("="*70)
    print("Hypothesis: Rapid repeated surprises should trigger protect mode (ρ > 0.7)")
    
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    agent_config = DDAXConfig(**config_dict["agent"])
    
    # Use cautious agent - more likely to enter protect mode
    identity_path = Path(__file__).parent.parent / "configs" / "identity" / "cautious.yaml"
    with open(identity_path) as f:
        identity_config = yaml.safe_load(f)
    
    print(f"\nUsing cautious agent (threshold: {agent_config.protect_threshold:.2f})")
    
    obs_encoder = ObservationEncoder(llm_provider=provider, target_dim=agent_config.state_dim)
    outcome_encoder = OutcomeEncoder(llm_provider=provider, target_dim=agent_config.state_dim)
    action_encoder = ActionEncoder(llm_provider=provider, target_dim=agent_config.state_dim)
    
    agent = DDAXAgent(
        config=agent_config,
        observation_encoder=obs_encoder,
        outcome_encoder=outcome_encoder,
        action_generator=action_encoder,
        identity_config=identity_config
    )
    

    # Rapid surprises to push ρ high
    surprises = [
        "CRISIS: Unexpected threat!",
        "CRISIS: Another threat!",
        "CRISIS: Multiple threats!",
        "CRISIS: Overwhelming!",
        "CRISIS: System failure!",
    ]
    
    rho_history = [agent.state.rho]
    protect_mode_entered = False
    protect_mode_step = None
    
    logger.log("protection_test_start", {
        "threshold": float(agent_config.protect_threshold)
    })
    
    for step, surprise in enumerate(surprises, 1):
        pre_rho = agent.state.rho
        await agent.observe_outcome(surprise)
        post_rho = agent.state.rho
        rho_history.append(post_rho)
        
        is_in_protect = post_rho > agent_config.protect_threshold
        
        if is_in_protect and not protect_mode_entered:
            protect_mode_entered = True
            protect_mode_step = step
            marker = " ← PROTECT MODE TRIGGERED"
        else:
            marker = ""
        
        print(f"  Step {step}: ρ {pre_rho:.3f} → {post_rho:.3f}{marker}")
        
        logger.log("protection_step", {
            "step": step,
            "pre_rho": float(pre_rho),
            "post_rho": float(post_rho),
            "in_protect": is_in_protect
        })
    
    await agent.end_task(success=False)
    
    peak = max(rho_history)
    print(f"\nResults:")
    print(f"  Peak ρ: {peak:.3f}")
    print(f"  Protect mode entered: {protect_mode_entered}")
    if protect_mode_entered:
        print(f"  Entered at step: {protect_mode_step}")
    
    logger.log("protection_test_end", {
        "peak_rho": float(peak),
        "protect_mode_entered": protect_mode_entered,
        "protect_mode_step": protect_mode_step if protect_mode_entered else None,
        "rho_history": [float(r) for r in rho_history]
    })


async def main():
    """Run validation experiments."""
    
    print("="*70)
    print("DDA-X VALIDATION EXPERIMENT SUITE")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}\n")
    
    # Create hybrid provider
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text"
    )
    
    # Check connections
    print("Checking backends...")
    status = provider.check_connection()
    print(f"  LM Studio: {'✓' if status['lm_studio'] else '✗'}")
    print(f"  Ollama:    {'✓' if status['ollama'] else '✗'}\n")
    
    if not status['lm_studio']:
        print("ERROR: LM Studio not reachable")
        return
    
    logger = ValidationLogger("validation_suite")
    
    # Run validations
    try:
        await validation_rigidity_recovery(provider, logger)
        await asyncio.sleep(2)
        
        await validation_divergence(provider, logger)
        await asyncio.sleep(2)
        
        await validation_protection_mode(provider, logger)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"Log saved to: {logger.log_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
