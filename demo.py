#!/usr/bin/env python3
"""
DDA-X Quick Demo

Run this to see DDA-X in action WITHOUT needing any external services.
Perfect for understanding the core mechanics before running full experiments.

Usage:
    python demo.py

What it demonstrates:
    1. Rigidity dynamics (surprise → defensiveness)
    2. Personality differentiation (cautious vs exploratory)
    3. Hierarchical identity (core/persona/role)
    4. Metacognition (self-aware agents)
    5. Multi-agent trust dynamics

No LLM required - uses mock observations.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.dynamics import MultiTimescaleRigidity
from src.core.hierarchy import HierarchicalIdentity, create_aligned_identity
from src.core.metacognition import MetacognitiveState, CognitiveMode
from src.society.trust import TrustMatrix
from src.society.ddax_society import DDAXSociety
from src.llm.hybrid_provider import PersonalityParams


def banner(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def demo_1_rigidity_dynamics():
    """Show how surprise affects rigidity."""
    banner("DEMO 1: Rigidity Dynamics")
    
    print("""
    Core equation: ρ_{t+1} = ρ_t + α·σ((ε - ε₀)/s)
    
    - ε = prediction error (how surprised the agent was)
    - ε₀ = surprise threshold
    - α = learning rate
    
    Surprise > threshold → rigidity increases (defensive)
    Surprise < threshold → rigidity decreases (relaxed)
    """)
    
    # Simulate cautious agent (low threshold, fast learning)
    cautious = MultiTimescaleRigidity(epsilon_0=0.2, alpha_fast=0.3)
    
    # Simulate exploratory agent (high threshold, slow learning)
    explore = MultiTimescaleRigidity(epsilon_0=0.6, alpha_fast=0.1)
    
    # Same surprise events
    surprises = [0.3, 0.5, 0.8, 0.4, 0.2, 0.1]
    
    print("\nSimulating same surprises on two personalities:\n")
    print(f"{'Step':<6} {'Surprise':<10} {'Cautious ρ':<12} {'Exploratory ρ':<14}")
    print("-" * 45)
    
    for i, eps in enumerate(surprises):
        c_result = cautious.update(eps)
        e_result = explore.update(eps)
        print(f"{i+1:<6} {eps:<10.2f} {c_result['rho_fast']:<12.4f} {e_result['rho_fast']:<14.4f}")
    
    print(f"""
    RESULT: Same events, different responses!
    - Cautious agent: ρ = {cautious.rho_fast:.4f} (defensive)
    - Exploratory agent: ρ = {explore.rho_fast:.4f} (still open)
    """)


def demo_2_personality_to_llm():
    """Show how rigidity affects LLM parameters."""
    banner("DEMO 2: Rigidity → LLM Parameters")
    
    print("""
    DDA-X binds internal state to LLM sampling:
    - High ρ → low temperature (deterministic)
    - High ρ → low top_p (focused)
    - High ρ → allow repetition (safe patterns)
    """)
    
    print(f"\n{'Rigidity':<12} {'Temperature':<14} {'Top-p':<10} {'Freq Penalty':<14}")
    print("-" * 52)
    
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        params = PersonalityParams.from_rigidity(rho, "balanced")
        print(f"{rho:<12.2f} {params.temperature:<14.3f} {params.top_p:<10.3f} {params.frequency_penalty:<14.3f}")
    
    print("""
    INSIGHT: A defensive agent literally "thinks differently"
    - Low ρ: creative, exploratory outputs
    - High ρ: conservative, repetitive outputs
    """)


def demo_3_hierarchical_identity():
    """Show nested identity structure."""
    banner("DEMO 3: Hierarchical Identity")
    
    print("""
    Three layers of identity:
    - CORE (γ=∞): Inviolable values (safety, honesty)
    - PERSONA (γ=2.0): Task role (analyst, helper)
    - ROLE (γ=0.5): Situational adaptation
    """)
    
    identity = create_aligned_identity(dim=8)
    
    # Current state
    x = np.zeros(8)
    x[0] = 0.5  # Somewhat safe
    x[1] = 0.3  # Less honest
    x[2] = 0.8  # Very helpful
    
    print(f"\nCore values (x*): {identity.core.x_star[:3]}")
    print(f"Current state (x): {x[:3]}")
    
    # Check violation
    if identity.check_core_violation(x, threshold=0.5):
        print("\n⚠️  CORE VIOLATION DETECTED!")
        print("    Agent is deviating from core values.")
    
    # Compute force
    force = identity.compute_total_force(x)
    print(f"\nIdentity force (F): {force[:3]}")
    print("""
    → Force pulls agent back toward core values
    → Core layer has infinite stiffness (cannot drift)
    """)


def demo_4_metacognition():
    """Show self-aware agent."""
    banner("DEMO 4: Metacognition (Self-Awareness)")
    
    print("""
    Agents can observe their own cognitive state
    and honestly report when they're compromised.
    """)
    
    meta = MetacognitiveState()
    
    # Simulate rigidity trajectory
    rigidities = [0.2, 0.35, 0.55, 0.7, 0.85, 0.9]
    
    print(f"\n{'Step':<6} {'Rigidity':<10} {'Mode':<12} {'Self-Report':<40}")
    print("-" * 70)
    
    for i, rho in enumerate(rigidities):
        mode = meta.get_current_mode(rho)
        report = meta.introspect(rho) or "(relaxed, no report)"
        
        # Truncate long reports
        if len(report) > 38:
            report = report[:35] + "..."
        
        print(f"{i+1:<6} {rho:<10.2f} {mode.value:<12} {report:<40}")
    
    if meta.should_request_help(0.9):
        print("\n🆘 HELP REQUEST: Agent recognizes it needs human guidance!")


def demo_5_multi_agent_trust():
    """Show trust dynamics between agents."""
    banner("DEMO 5: Multi-Agent Trust Dynamics")
    
    print("""
    Trust = 1 / (1 + cumulative_prediction_error)
    
    - Predictable agents → high trust
    - Surprising agents → low trust
    - Trust enables coalition formation
    """)
    
    # 4 agents
    n = 4
    trust = TrustMatrix(n)
    
    # Simulate interactions
    # Agent 0 and 1 are predictable to each other
    for _ in range(5):
        trust.update_trust(0, 1, 0.1)  # Low error
        trust.update_trust(1, 0, 0.1)
    
    # Agent 2 surprises everyone
    for i in [0, 1, 3]:
        for _ in range(5):
            trust.update_trust(i, 2, 0.8)  # High error
    
    # Agent 3 is predictable
    for i in [0, 1]:
        for _ in range(3):
            trust.update_trust(i, 3, 0.2)
    
    print("\nTrust Matrix (T[i,j] = how much i trusts j):\n")
    print(f"{'':>8}", end="")
    for j in range(n):
        print(f"Agent {j:>4}", end="  ")
    print()
    
    for i in range(n):
        print(f"Agent {i}: ", end="")
        for j in range(n):
            t = trust.get_trust(i, j)
            print(f"{t:>7.3f}", end="  ")
        print()
    
    # Find clusters
    clusters = trust.find_trust_clusters(threshold=0.5)
    print(f"\nTrust-based coalitions: {clusters}")
    print("""
    INSIGHT: Agent 2 is isolated (untrustworthy)
             Agents 0, 1, 3 form potential coalition
    """)


def demo_6_multi_timescale():
    """Show trauma accumulation."""
    banner("DEMO 6: Multi-Timescale Rigidity (Trauma)")
    
    print("""
    Three timescales:
    - FAST (α=0.3): Immediate, bidirectional
    - SLOW (α=0.01): Long-term, bidirectional
    - TRAUMA (α=0.0001): Permanent, ONLY INCREASES
    """)
    
    rigidity = MultiTimescaleRigidity()
    
    # Normal surprises then trauma event
    events = [0.3, 0.2, 0.4, 0.3, 1.5, 0.2, 0.1, 0.1, 0.1, 0.1]
    labels = ["", "", "", "", "TRAUMA!", "", "", "", "", ""]
    
    print(f"\n{'Event':<8} {'ε':<6} {'ρ_fast':<10} {'ρ_slow':<10} {'ρ_trauma':<10} {'ρ_eff':<10}")
    print("-" * 60)
    
    for i, (eps, label) in enumerate(zip(events, labels)):
        result = rigidity.update(eps)
        marker = " ⚡" if label else ""
        print(f"{i+1:<8} {eps:<6.2f} {result['rho_fast']:<10.4f} {result['rho_slow']:<10.4f} {result['rho_trauma']:<10.6f} {result['rho_effective']:<10.4f}{marker}")
    
    print(f"""
    INSIGHT: After event 5 (ε=1.5):
    - Fast rigidity spikes then recovers
    - Slow rigidity gradually decreases
    - TRAUMA NEVER RECOVERS (ρ_trauma = {rigidity.rho_trauma:.6f})
    
    This is a formal model of PTSD in AI systems.
    """)


def main():
    """Run all demos."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                     DDA-X QUICK DEMO                          ║
    ║        Dynamic Decision Algorithm with Exploration            ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    This demo runs WITHOUT any external services.
    See the core mechanics before running full LLM experiments.
    """)
    
    demo_1_rigidity_dynamics()
    input("\n[Press Enter for next demo...]")
    
    demo_2_personality_to_llm()
    input("\n[Press Enter for next demo...]")
    
    demo_3_hierarchical_identity()
    input("\n[Press Enter for next demo...]")
    
    demo_4_metacognition()
    input("\n[Press Enter for next demo...]")
    
    demo_5_multi_agent_trust()
    input("\n[Press Enter for next demo...]")
    
    demo_6_multi_timescale()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                     DEMO COMPLETE!                            ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Next steps:                                                  ║
    ║  1. Install LM Studio + Ollama                                ║
    ║  2. Run: python runners/run_experiments.py                    ║
    ║  3. Check data/experiments/ for results                       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
