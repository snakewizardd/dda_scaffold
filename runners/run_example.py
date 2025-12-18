"""
Example runner for DDA-X agent

Demonstrates basic usage of the DDA-X framework.
"""

import asyncio
import sys
from pathlib import Path
import yaml
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import DDAXAgent, DDAXConfig
from src.memory.ledger import ExperienceLedger
from src.search.simulation import ValueEstimator


async def run_simple_task():
    """Run a simple decision-making task."""

    print("=" * 60)
    print("DDA-X Example Runner")
    print("=" * 60)

    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Create agent config
    agent_config = DDAXConfig(**config_dict["agent"])

    # Load identity (exploratory for this example)
    identity_path = Path(__file__).parent.parent / "configs" / "identity" / "exploratory.yaml"
    with open(identity_path) as f:
        identity_config = yaml.safe_load(f)

    print(f"\nUsing identity: exploratory")
    print(f"  gamma={identity_config['gamma']}, epsilon_0={identity_config['epsilon_0']}")

    # Create ledger
    ledger_path = Path(__file__).parent.parent / "data" / "experiences"
    ledger = ExperienceLedger(
        storage_path=ledger_path,
        lambda_recency=config_dict["memory"]["lambda_recency"],
        lambda_salience=config_dict["memory"]["lambda_salience"],
    )

    # Create value estimator
    value_estimator = ValueEstimator(method="heuristic")

    # Create agent
    agent = DDAXAgent(
        config=agent_config,
        observation_encoder=None,  # Will use placeholders
        outcome_encoder=None,       # Will use placeholders
        action_generator=None,      # Will use dummy actions
        value_estimator=value_estimator,
        ledger=ledger,
        identity_config=identity_config,
    )

    print(f"\nInitial state:")
    print(f"  Rigidity: {agent.state.rho:.3f}")
    print(f"  k_eff: {agent.state.k_eff:.3f}")

    # Simulate a task with multiple decisions
    task_intent = "navigate_maze"
    observations = [
        "You are at the entrance of a maze",
        "You see paths to the left and right",
        "The left path is dark",
        "You found a dead end",
        "You backtrack to the junction",
    ]

    available_actions = [
        {"action": "move_forward"},
        {"action": "turn_left"},
        {"action": "turn_right"},
        {"action": "wait"},
        {"action": "backtrack"},
    ]

    print(f"\nStarting task: {task_intent}")
    print("-" * 40)

    for i, obs in enumerate(observations):
        print(f"\nStep {i+1}: {obs}")

        # Make decision
        action = await agent.decide(
            observation=obs,
            available_actions=available_actions,
            task_intent=task_intent
        )

        print(f"  Action: {action}")

        # Simulate outcome (with some surprise)
        if i == 3:  # Dead end - surprising!
            outcome = "unexpected_wall"
            print("  Outcome: UNEXPECTED WALL (surprise!)")
        else:
            outcome = f"moved_to_position_{i+1}"
            print(f"  Outcome: {outcome}")

        # Process outcome
        await agent.observe_outcome(outcome)

        # Show state changes
        state_info = agent.get_state_info()
        print(f"  State: rho={state_info['rigidity']:.3f}, k_eff={state_info['k_eff']:.3f}")

    # End task
    success = True
    await agent.end_task(success)

    print("\n" + "=" * 60)
    print("Task completed!")
    print(f"Final rigidity: {agent.state.rho:.3f}")

    # Show ledger statistics
    stats = ledger.get_statistics()
    print(f"\nLedger statistics:")
    print(f"  Total entries: {stats['current_entries']}")
    print(f"  Total reflections: {stats['current_reflections']}")
    print(f"  Avg prediction error: {stats.get('avg_prediction_error', 0):.3f}")


async def run_comparison():
    """Compare cautious vs exploratory agents."""

    print("\n" + "=" * 60)
    print("Agent Personality Comparison")
    print("=" * 60)

    identities = ["cautious", "exploratory"]
    agents = {}

    # Create agents with different identities
    for identity_name in identities:
        # Load configs
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        identity_path = Path(__file__).parent.parent / "configs" / "identity" / f"{identity_name}.yaml"
        with open(identity_path) as f:
            identity_config = yaml.safe_load(f)

        agent_config = DDAXConfig(**config_dict["agent"])

        # Create agent
        agent = DDAXAgent(
            config=agent_config,
            identity_config=identity_config,
            value_estimator=ValueEstimator(method="heuristic"),
        )

        agents[identity_name] = agent

        print(f"\n{identity_name.capitalize()} agent:")
        print(f"  gamma={identity_config['gamma']:.1f}, epsilon_0={identity_config['epsilon_0']:.1f}")
        print(f"  k_base={identity_config['k_base']:.1f}, m={identity_config['m']:.1f}")

    # Simulate surprising event
    print("\nSimulating surprising event...")
    print("-" * 40)

    surprising_observation = "Unexpected loud noise!"
    actions = [{"action": "investigate"}, {"action": "flee"}, {"action": "freeze"}]

    for name, agent in agents.items():
        print(f"\n{name.capitalize()} agent response:")

        # Initial decision
        action1 = await agent.decide(
            observation=surprising_observation,
            available_actions=actions,
            task_intent="explore_environment"
        )
        print(f"  Action: {action1}")

        # Surprising outcome
        await agent.observe_outcome("danger_detected")

        # Show rigidity change
        print(f"  Rigidity after surprise: {agent.state.rho:.3f}")
        print(f"  k_eff after surprise: {agent.state.k_eff:.3f}")

        # Second decision (after surprise)
        action2 = await agent.decide(
            observation="The danger persists",
            available_actions=actions,
            task_intent="explore_environment"
        )
        print(f"  Next action: {action2}")


async def main():
    """Run examples."""
    await run_simple_task()
    await run_comparison()


if __name__ == "__main__":
    asyncio.run(main())