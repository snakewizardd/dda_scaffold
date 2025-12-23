"""
Example runner for DDA-X agent with Ollama LLM integration.

Demonstrates using real LLM for action generation and value estimation.
"""

import asyncio
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import DDAXAgent, DDAXConfig
from src.memory.ledger import ExperienceLedger
from src.search.simulation import ValueEstimator
from src.llm.providers import OllamaProvider
from src.channels.encoders import ObservationEncoder, ActionEncoder, OutcomeEncoder
from src.core.state import ActionDirection


async def check_ollama_connection(provider: OllamaProvider) -> bool:
    """Check if Ollama is running and models are available."""
    print("Checking Ollama connection...")
    
    if not provider.check_connection():
        print("ERROR: Cannot connect to Ollama. Make sure it's running!")
        print("  Start Ollama with: ollama serve")
        return False
    
    print("  Ollama is connected!")
    
    # Try to list models (may fail on some API versions)
    try:
        models = provider.list_models()
        if models:
            print(f"  Available models: {models}")
            
            # Check required models
            required = [provider.model, provider.embed_model]
            missing = [m for m in required if not any(m in model for model in models)]
            
            if missing:
                print(f"  WARNING: May be missing models: {missing}")
                print(f"  If errors occur, pull them with:")
                for m in missing:
                    print(f"    ollama pull {m}")
    except Exception as e:
        print(f"  Note: Could not list models ({e}), continuing anyway...")
    
    print("  Proceeding with run...")
    return True


async def run_with_ollama():
    """Run DDA-X with Ollama LLM backend."""
    
    print("=" * 60)
    print("DDA-X with Ollama LLM Integration")
    print("=" * 60)
    
    # Create Ollama provider
    provider = OllamaProvider(
        model="llama3.2",  # Or your preferred model
        embed_model="nomic-embed-text",
        host="http://localhost:11434"
    )
    
    # Check connection
    if not await check_ollama_connection(provider):
        return
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    agent_config = DDAXConfig(**config_dict["agent"])
    
    # Load identity
    identity_path = Path(__file__).parent.parent / "configs" / "identity" / "exploratory.yaml"
    with open(identity_path) as f:
        identity_config = yaml.safe_load(f)
    
    print(f"\nUsing identity: exploratory")
    print(f"  gamma={identity_config['gamma']}, epsilon_0={identity_config['epsilon_0']}")
    
    # Create encoders with LLM
    obs_encoder = ObservationEncoder(
        llm_provider=provider,
        target_dim=agent_config.state_dim
    )
    outcome_encoder = OutcomeEncoder(
        llm_provider=provider,
        target_dim=agent_config.state_dim
    )
    action_encoder = ActionEncoder(
        llm_provider=provider,
        target_dim=agent_config.state_dim
    )
    
    # Create ledger
    ledger_path = Path(__file__).parent.parent / "data" / "experiences"
    ledger = ExperienceLedger(
        storage_path=ledger_path,
        lambda_recency=config_dict["memory"]["lambda_recency"],
        lambda_salience=config_dict["memory"]["lambda_salience"],
    )
    
    # Create value estimator with LLM
    value_estimator = ValueEstimator(
        method="llm",  # Use LLM for value estimation
        llm_provider=provider
    )
    
    # Action generator that uses LLM
    async def action_generator(observation, available_actions):
        """Generate action directions using LLM."""
        # Get LLM action priors
        actions_with_priors = await provider.generate_actions(
            observation=str(observation),
            available_actions=available_actions,
            intent="navigate_maze",
            n_samples=3
        )
        
        # Convert to ActionDirection objects
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
        observation_encoder=obs_encoder,  # Pass object, not method
        outcome_encoder=outcome_encoder,  # Pass object, not method
        action_generator=action_generator,
        value_estimator=value_estimator,
        ledger=ledger,
        identity_config=identity_config,
    )
    
    print(f"\nInitial state:")
    print(f"  Rigidity: {agent.state.rho:.3f}")
    print(f"  k_eff: {agent.state.k_eff:.3f}")
    
    # Simulate a task with LLM-powered decisions
    task_intent = "navigate_maze"
    observations = [
        "You are at the entrance of a dark maze. You can see paths leading left and right.",
        "The left path has a torch on the wall, providing some light. The right path is pitch dark.",
        "You are now in a room with three doors: one red, one blue, one green.",
        "You found a dead end! The path is blocked by rubble.",
        "You backtrack and find a hidden passage behind a loose stone.",
    ]
    
    available_actions = [
        {"action": "move_forward", "description": "Move straight ahead"},
        {"action": "turn_left", "description": "Turn left and proceed"},
        {"action": "turn_right", "description": "Turn right and proceed"},
        {"action": "wait", "description": "Wait and observe"},
        {"action": "backtrack", "description": "Go back the way you came"},
    ]
    
    print(f"\nStarting task: {task_intent}")
    print("-" * 40)
    
    for i, obs in enumerate(observations):
        print(f"\nStep {i+1}: {obs[:60]}...")
        
        # Make decision with LLM
        action = await agent.decide(
            observation=obs,
            available_actions=available_actions,
            task_intent=task_intent
        )
        
        print(f"  Action: {action}")
        
        # Simulate outcome
        if i == 3:  # Dead end - surprising!
            outcome = "You ran into an unexpected wall. Dead end!"
            print("  Outcome: DEAD END (surprise!)")
        else:
            outcome = f"You successfully moved to a new location."
            print(f"  Outcome: {outcome}")
        
        # Process outcome
        await agent.observe_outcome(outcome)
        
        # Show state changes
        state_info = agent.get_state_info()
        print(f"  State: rho={state_info['rigidity']:.3f}, k_eff={state_info['k_eff']:.3f}")
        
        # Small delay to be nice to the GPU
        await asyncio.sleep(0.5)
    
    # End task
    await agent.end_task(success=True)
    
    print("\n" + "=" * 60)
    print("Task completed!")
    print(f"Final rigidity: {agent.state.rho:.3f}")
    
    # Show ledger stats
    stats = ledger.get_statistics()
    print(f"\nLedger statistics:")
    print(f"  Total entries: {stats['current_entries']}")
    print(f"  Total reflections: {stats['current_reflections']}")


async def main():
    """Main entry point."""
    await run_with_ollama()


if __name__ == "__main__":
    asyncio.run(main())
