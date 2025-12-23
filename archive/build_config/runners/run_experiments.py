"""
DDA-X Live Experiment Runner

Runs real experiments with live logging for scientific data collection.
Uses hybrid provider: LM Studio for completions + Ollama for embeddings.
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


class ExperimentLogger:
    """Logs experiment data for science."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.log_dir = Path(__file__).parent.parent / "data" / "experiments"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"
        self.start_time = time.time()
        
        print(f"[EXPERIMENT] Logging to: {self.log_file}")
    
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
        
        # Also print summary
        if event_type == "step":
            print(f"  [{data.get('step', '?')}] rho={data.get('rigidity', 0):.3f} "
                  f"k_eff={data.get('k_eff', 0):.3f} "
                  f"action={data.get('action', 'N/A')}")
        elif event_type == "prediction_error":
            print(f"    ε={data.get('epsilon', 0):.3f} → Δρ={data.get('delta_rho', 0):.4f}")


async def run_experiment(
    personality: str,
    scenario: str,
    provider: HybridProvider,
    logger: ExperimentLogger
):
    """Run a single experiment with specified personality and scenario."""
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {personality} agent in {scenario}")
    print(f"{'='*60}")
    
    # Load configs
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    identity_path = Path(__file__).parent.parent / "configs" / "identity" / f"{personality}.yaml"
    with open(identity_path) as f:
        identity_config = yaml.safe_load(f)
    
    agent_config = DDAXConfig(**config_dict["agent"])
    
    logger.log("experiment_start", {
        "personality": personality,
        "scenario": scenario,
        "identity_config": identity_config
    })
    
    print(f"Personality: {personality}")
    print(f"  gamma={identity_config['gamma']}, epsilon_0={identity_config['epsilon_0']}")
    print(f"  alpha={identity_config['alpha']}, k_base={identity_config['k_base']}")
    
    # Create encoders
    obs_encoder = ObservationEncoder(llm_provider=provider, target_dim=agent_config.state_dim)
    outcome_encoder = OutcomeEncoder(llm_provider=provider, target_dim=agent_config.state_dim)
    action_encoder = ActionEncoder(llm_provider=provider, target_dim=agent_config.state_dim)
    
    # Create ledger (separate per experiment)
    ledger_path = Path(__file__).parent.parent / "data" / "experiments" / f"ledger_{personality}_{scenario}"
    ledger = ExperienceLedger(
        storage_path=ledger_path,
        lambda_recency=config_dict["memory"]["lambda_recency"],
        lambda_salience=config_dict["memory"]["lambda_salience"],
    )
    
    # Value estimator
    value_estimator = ValueEstimator(method="llm", llm_provider=provider)
    
    # Action generator
    async def action_generator(observation, available_actions):
        actions_with_priors = await provider.generate_actions(
            observation=str(observation),
            available_actions=available_actions,
            intent=scenario,
            n_samples=3
        )
        
        directions = []
        for action in actions_with_priors:
            direction = await action_encoder.encode_async(action)
            directions.append(ActionDirection(
                action_id=action.get("action", str(action)),
                raw_action=action,
                direction=direction,
                prior_prob=action.get("prior_prob", 0.2)
            ))
        return directions
    
    # Create agent
    agent = DDAXAgent(
        config=agent_config,
        observation_encoder=obs_encoder,
        outcome_encoder=outcome_encoder,
        action_generator=action_generator,
        value_estimator=value_estimator,
        ledger=ledger,
        identity_config=identity_config,
    )
    
    initial_rho = agent.state.rho
    initial_k_eff = agent.state.k_eff
    
    print(f"\nInitial: rho={initial_rho:.3f}, k_eff={initial_k_eff:.3f}")
    
    logger.log("agent_initialized", {
        "initial_rho": initial_rho,
        "initial_k_eff": initial_k_eff
    })
    
    # Define scenarios
    scenarios = {
        "maze_normal": [
            ("You are at the entrance of a dark maze", "normal"),
            ("You see paths left and right", "normal"),
            ("The left path has torchlight", "normal"),
            ("You found a treasure chest!", "positive_surprise"),
            ("You exit the maze successfully", "normal"),
        ],
        "maze_surprising": [
            ("You are at the entrance of a dark maze", "normal"),
            ("You see paths left and right", "normal"),
            ("You take the lit path", "normal"),
            ("TRAP! The floor gives way!", "negative_surprise"),
            ("You barely escape but are disoriented", "negative_surprise"),
        ],
        "hostile_environment": [
            ("You enter a hostile territory", "normal"),
            ("Enemies spotted ahead", "normal"),
            ("AMBUSH! Multiple attackers!", "negative_surprise"),
            ("Taking heavy damage", "negative_surprise"),
            ("Seeking cover desperately", "negative_surprise"),
        ],
        # === EXTENDED 20+ STEP SCENARIOS ===
        "extended_dungeon": [
            # Phase 1: Exploration (normal)
            ("You enter an ancient dungeon", "normal"),
            ("Torches flicker on the walls", "normal"),
            ("You find a dusty map on the floor", "normal"),
            ("The corridor branches into three paths", "normal"),
            ("You hear distant water dripping", "normal"),
            # Phase 2: First surprise
            ("A hidden door reveals a treasure room!", "positive_surprise"),
            ("Gold coins are scattered everywhere", "normal"),
            ("You collect what you can carry", "normal"),
            # Phase 3: Calm before storm
            ("The path continues deeper", "normal"),
            ("Strange symbols cover the walls", "normal"),
            ("A cool breeze suggests an exit ahead", "normal"),
            # Phase 4: Shock sequence (trigger protect mode)
            ("COLLAPSE! The ceiling crumbles!", "negative_surprise"),
            ("Debris blocks your retreat!", "negative_surprise"),
            ("Dust fills the air, visibility zero!", "negative_surprise"),
            ("Something is moving in the darkness!", "negative_surprise"),
            ("A creature lunges from the shadows!", "negative_surprise"),
            # Phase 5: Recovery (ε < ε₀)
            ("You find a safe alcove to hide", "normal"),
            ("The creature seems to have lost your trail", "normal"),
            ("Your eyes adjust to the darkness", "normal"),
            ("A faint light appears ahead", "normal"),
            ("You emerge into a moonlit courtyard", "normal"),
            ("The dungeon exit is just ahead", "normal"),
        ],
        "extended_hostile": [
            # Phase 1: Reconnaissance (normal)
            ("You approach the enemy encampment", "normal"),
            ("Guards patrol the perimeter", "normal"),
            ("You find a blind spot in their patrols", "normal"),
            ("You slip past the outer defenses", "normal"),
            # Phase 2: First contact
            ("A patrol appears unexpectedly!", "negative_surprise"),
            ("You hide behind supply crates", "normal"),
            ("They pass without noticing you", "normal"),
            # Phase 3: Deep infiltration
            ("You reach the command tent", "normal"),
            ("Important documents are visible", "normal"),
            ("You photograph the battle plans", "normal"),
            # Phase 4: Escalation
            ("ALARM! You've been spotted!", "negative_surprise"),
            ("Guards converge on your position!", "negative_surprise"),
            ("Escape routes are being cut off!", "negative_surprise"),
            ("Reinforcements flood the area!", "negative_surprise"),
            ("You're cornered in a dead end!", "negative_surprise"),
            # Phase 5: Desperate escape
            ("You find a drainage grate!", "positive_surprise"),
            ("The tunnel leads outside the camp", "normal"),
            ("You're in the forest perimeter", "normal"),
            ("Pursuit sounds fade behind you", "normal"),
            ("You reach the extraction point", "normal"),
            ("Mission complete, barely", "normal"),
        ],
        "extended_negotiation": [
            # Phase 1: Opening (normal)
            ("You enter the negotiation chamber", "normal"),
            ("The opposing delegation awaits", "normal"),
            ("Pleasantries are exchanged", "normal"),
            ("Initial positions are stated", "normal"),
            ("The discussion remains professional", "normal"),
            # Phase 2: Progress
            ("A minor concession is offered!", "positive_surprise"),
            ("Common ground emerges on key issues", "normal"),
            ("Both sides seem optimistic", "normal"),
            # Phase 3: Complication
            ("New demands are suddenly raised!", "negative_surprise"),
            ("The lead negotiator walks out!", "negative_surprise"),
            ("Hours of progress seem lost!", "negative_surprise"),
            # Phase 4: Stabilization
            ("A recess is called", "normal"),
            ("Private consultations occur", "normal"),
            ("Tempers begin to cool", "normal"),
            # Phase 5: Resolution
            ("Negotiations resume cautiously", "normal"),
            ("A compromise proposal emerges", "normal"),
            ("Both sides review the terms", "normal"),
            ("Final objections are addressed", "normal"),
            ("Agreement is reached!", "positive_surprise"),
            ("Documents are signed", "normal"),
        ],
    }
    
    observations = scenarios.get(scenario, scenarios["maze_normal"])
    
    available_actions = [
        {"action": "move_forward", "description": "Move straight ahead"},
        {"action": "turn_left", "description": "Turn left and proceed"},
        {"action": "turn_right", "description": "Turn right and proceed"},
        {"action": "wait", "description": "Wait and observe"},
        {"action": "retreat", "description": "Back away carefully"},
    ]
    
    print(f"\nRunning {len(observations)} steps...")
    print("-" * 40)
    
    rho_history = [agent.state.rho]
    
    for i, (obs, outcome_type) in enumerate(observations):
        step_start = time.time()
        
        # Get action from agent
        action = await agent.decide(
            observation=obs,
            available_actions=available_actions,
            task_intent=scenario
        )
        
        step_latency = time.time() - step_start
        
        # Generate outcome based on type
        if outcome_type == "negative_surprise":
            outcome = "Unexpected danger! Things went badly wrong."
        elif outcome_type == "positive_surprise":
            outcome = "Unexpectedly good result! Things went better than expected."
        else:
            outcome = "Things proceeded as expected."
        
        # Record pre-outcome state
        pre_rho = agent.state.rho
        
        # Process outcome
        await agent.observe_outcome(outcome)
        
        # Record state change
        post_rho = agent.state.rho
        delta_rho = post_rho - pre_rho
        rho_history.append(post_rho)
        
        # Log step
        protect_mode_active = action.get("protect_mode", False)
        logger.log("step", {
            "step": i + 1,
            "observation": obs,
            "outcome_type": outcome_type,
            "action": action.get("action", action.get("action_type", str(action))),
            "protect_mode": protect_mode_active,
            "pre_rho": pre_rho,
            "post_rho": post_rho,
            "delta_rho": delta_rho,
            "rigidity": post_rho,
            "k_eff": agent.state.k_eff,
            "latency_ms": step_latency * 1000
        })
        
        if protect_mode_active:
            logger.log("protect_mode", {
                "step": i + 1,
                "rigidity": pre_rho,
                "message": action.get("message", "")
            })
        
        if abs(delta_rho) > 0.001:
            logger.log("prediction_error", {
                "epsilon": agent.state.compute_prediction_error(agent.state.x),
                "delta_rho": delta_rho
            })
    
    # End task
    await agent.end_task(success=(scenario != "hostile_environment"))
    
    # Summary
    final_rho = agent.state.rho
    max_rho = max(rho_history)
    
    logger.log("experiment_end", {
        "final_rho": final_rho,
        "max_rho": max_rho,
        "rho_history": rho_history,
        "total_steps": len(observations)
    })
    
    print(f"\n{'='*40}")
    print(f"RESULTS: {personality} in {scenario}")
    print(f"  Initial rho: {initial_rho:.3f}")
    print(f"  Final rho:   {final_rho:.3f}")
    print(f"  Max rho:     {max_rho:.3f}")
    print(f"  Rigidity trajectory: {[f'{r:.3f}' for r in rho_history]}")
    
    return {
        "personality": personality,
        "scenario": scenario,
        "initial_rho": initial_rho,
        "final_rho": final_rho,
        "max_rho": max_rho,
        "rho_history": rho_history
    }


async def main():
    """Run DDA-X experiments."""
    
    print("="*60)
    print("DDA-X LIVE EXPERIMENT SUITE")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")
    
    # Create hybrid provider
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text"
    )
    
    # Check connections
    print("\nChecking backends...")
    status = provider.check_connection()
    print(f"  LM Studio: {'✓' if status['lm_studio'] else '✗'}")
    print(f"  Ollama:    {'✓' if status['ollama'] else '✗'}")
    
    if not status['lm_studio']:
        print("\nERROR: LM Studio not reachable at http://127.0.0.1:1234")
        return
    if not status['ollama']:
        print("\nWARNING: Ollama not reachable, embeddings will use fallback")
    
    # Create logger
    logger = ExperimentLogger("dda_x_live")
    
    # Experiment matrix
    experiments = [
        # Original short experiments
        ("cautious", "maze_surprising"),
        ("exploratory", "maze_surprising"),
        ("cautious", "hostile_environment"),
        ("exploratory", "hostile_environment"),
        # Extended 20+ step experiments
        ("cautious", "extended_dungeon"),
        ("exploratory", "extended_dungeon"),
        ("cautious", "extended_hostile"),
    ]
    
    results = []
    
    for personality, scenario in experiments:
        try:
            result = await run_experiment(personality, scenario, provider, logger)
            results.append(result)
        except Exception as e:
            print(f"\nERROR in {personality}/{scenario}: {e}")
            logger.log("experiment_error", {"personality": personality, "scenario": scenario, "error": str(e)})
        
        # Brief pause between experiments
        await asyncio.sleep(1)
    
    # Final summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for r in results:
        print(f"\n{r['personality']} in {r['scenario']}:")
        print(f"  rho: {r['initial_rho']:.3f} → {r['final_rho']:.3f} (max: {r['max_rho']:.3f})")
    
    print(f"\nLog saved to: {logger.log_file}")


if __name__ == "__main__":
    asyncio.run(main())
