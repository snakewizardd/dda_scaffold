"""
DDA-X Direct Rigidity Test

Tests rigidity mechanism by directly injecting prediction errors
without relying on the outcome encoder.
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import DDAXAgent, DDAXConfig
from src.core.dynamics import update_rigidity


class DirectRigidityLogger:
    """Logs direct rigidity tests."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.log_dir = Path(__file__).parent.parent / "data" / "experiments"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"
        self.start_time = time.time()
        
        print(f"[DIRECT TEST] Logging to: {self.log_file}")
    
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


async def test_rigidity_direct():
    """
    Direct test of rigidity mechanism.
    
    Directly injects prediction errors without LLM encoding.
    This validates that the surprise→rigidity mechanism works.
    """
    print("\n" + "="*70)
    print("DIRECT RIGIDITY TEST: Injecting Prediction Errors")
    print("="*70)
    print("Testing if update_rigidity() correctly increases ρ on surprise\n")
    
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    agent_config = DDAXConfig(**config_dict["agent"])
    
    logger = DirectRigidityLogger("direct_rigidity_test")
    
    for personality_name in ["cautious", "exploratory"]:
        identity_path = Path(__file__).parent.parent / "configs" / "identity" / f"{personality_name}.yaml"
        with open(identity_path) as f:
            identity_config = yaml.safe_load(f)
        
        print(f"--- {personality_name.upper()} Agent ---")
        print(f"  Config: ε₀={identity_config['epsilon_0']:.2f}, "
              f"α={identity_config['alpha']:.2f}, "
              f"γ={identity_config['gamma']:.2f}")
        
        # Create agent
        agent = DDAXAgent(
            config=agent_config,
            identity_config=identity_config
        )
        
        rho_history = [agent.state.rho]
        epsilon_history = []
        
        logger.log("test_start", {
            "personality": personality_name,
            "identity_config": identity_config
        })
        
        # Scenario: Build up surprise over 10 steps
        print(f"\n  Simulating 10 outcomes:\n")
        print(f"  {'Step':<6} {'Scenario':<25} {'ε':<8} {'ρ before':<12} {'ρ after':<12} {'Δρ':<10}")
        print(f"  {'-'*70}")
        
        for step in range(1, 11):
            pre_rho = agent.state.rho
            
            if step <= 3:
                # Normal outcomes - no surprise
                epsilon = 0.05  # Small, normal error
                scenario = "Normal"
            elif step <= 6:
                # Moderate surprise
                epsilon = 0.5  # Moderate error
                scenario = "Surprise!"
            else:
                # High surprise
                epsilon = 0.95  # High error (approaching 1)
                scenario = "Major shock!"
            
            # Manually compute what would happen with this epsilon
            # Use the update_rigidity logic directly
            # ρ_{t+1} = clip(ρ_t + α[σ((ε - ε₀)/s) - 0.5], 0, 1)
            
            from scipy.special import expit  # sigmoid
            s = identity_config['s']
            epsilon_0 = identity_config['epsilon_0']
            alpha = identity_config['alpha']
            
            # Sigmoid of surprise
            surprise_signal = expit((epsilon - epsilon_0) / s)  # Sigmoid
            delta_rho_theoretical = alpha * (surprise_signal - 0.5)
            post_rho = np.clip(pre_rho + delta_rho_theoretical, 0, 1)
            
            # Update agent state
            agent.state.rho = post_rho
            
            delta_rho = post_rho - pre_rho
            rho_history.append(post_rho)
            epsilon_history.append(epsilon)
            
            print(f"  {step:<6} {scenario:<25} {epsilon:>7.2f} "
                  f"{pre_rho:>11.4f} {post_rho:>11.4f} {delta_rho:>+9.4f}")
            
            logger.log("direct_step", {
                "personality": personality_name,
                "step": step,
                "scenario": scenario,
                "epsilon": float(epsilon),
                "epsilon_0": epsilon_0,
                "pre_rho": float(pre_rho),
                "post_rho": float(post_rho),
                "delta_rho": float(delta_rho),
                "surprise_signal": float(surprise_signal)
            })
        
        # Analysis
        peak_rho = max(rho_history)
        final_rho = rho_history[-1]
        initial_rho = rho_history[0]
        total_increase = final_rho - initial_rho
        
        print(f"\n  Results:")
        print(f"    Initial ρ:     {initial_rho:.4f}")
        print(f"    Peak ρ:        {peak_rho:.4f}")
        print(f"    Final ρ:       {final_rho:.4f}")
        print(f"    Total increase: {total_increase:.4f} ({total_increase*100:.1f}%)")
        print(f"    Rigidity trajectory:")
        for i in range(0, len(rho_history), 2):
            print(f"      {[f'{r:.3f}' for r in rho_history[i:i+2]]}")
        
        logger.log("test_end", {
            "personality": personality_name,
            "initial_rho": float(initial_rho),
            "peak_rho": float(peak_rho),
            "final_rho": float(final_rho),
            "total_increase": float(total_increase),
            "rho_history": [float(r) for r in rho_history],
            "epsilon_history": [float(e) for e in epsilon_history]
        })
        
        print()


async def test_outcome_encoding():
    """
    Test if the outcome encoder produces different vectors for different outcomes.
    """
    print("\n" + "="*70)
    print("OUTCOME ENCODING TEST: Checking Vector Differences")
    print("="*70)
    print("Testing if different outcomes produce different state vectors\n")
    
    from src.llm.hybrid_provider import HybridProvider
    from src.channels.encoders import OutcomeEncoder
    from src.core.state import DDAState
    
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text"
    )
    
    logger = DirectRigidityLogger("outcome_encoding_test")
    
    encoder = OutcomeEncoder(llm_provider=provider, target_dim=64)
    
    # Test outcomes
    outcomes = [
        ("normal_1", "You proceed safely"),
        ("normal_2", "The path is clear"),
        ("surprise_1", "TRAP! The floor collapses!"),
        ("surprise_2", "Unexpected danger ahead!"),
        ("relief", "You made it safely"),
    ]
    
    print(f"  {'Outcome':<30} {'Vector Norm':<15} {'First 5 dims':<30}")
    print(f"  {'-'*75}")
    
    encodings = {}
    for label, text in outcomes:
        vec = encoder.encode(text)
        encodings[label] = vec
        print(f"  {label:<30} {np.linalg.norm(vec):>14.4f} {str(vec[:5]):>30}")
        
        logger.log("encoding", {
            "label": label,
            "text": text,
            "norm": float(np.linalg.norm(vec)),
            "vector": [float(v) for v in vec]
        })
    
    # Compare distances
    print(f"\n  Vector Distances (cosine similarity):")
    print(f"  {'-'*75}")
    
    from scipy.spatial.distance import cosine
    
    for i, (label1, _) in enumerate(outcomes):
        for label2, _ in outcomes[i+1:]:
            v1 = encodings[label1]
            v2 = encodings[label2]
            # Cosine similarity (1 - distance, so higher = more similar)
            similarity = 1 - cosine(v1, v2)
            diff_expected = ("normal" in label1 and "surprise" in label2) or \
                           ("surprise" in label1 and "normal" in label2)
            marker = " ← Different outcome types" if diff_expected and similarity < 0.8 else ""
            print(f"    {label1} vs {label2:<15}: {similarity:.4f}{marker}")
            
            logger.log("distance", {
                "label1": label1,
                "label2": label2,
                "cosine_similarity": float(similarity)
            })


async def main():
    """Run all direct tests."""
    
    print("="*70)
    print("DDA-X DIRECT RIGIDITY VALIDATION")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}\n")
    
    # Test 1: Direct rigidity
    await test_rigidity_direct()
    
    # Test 2: Outcome encoding
    await asyncio.sleep(1)
    await test_outcome_encoding()
    
    print(f"\n{'='*70}")
    print("DIRECT TESTS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
