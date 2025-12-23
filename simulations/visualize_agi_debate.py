#!/usr/bin/env python3
"""
Visualization generator for AGI Debate simulation.
Run this after a simulation to generate plots from the saved data.

Usage: 
  python visualize_agi_debate.py                    # Uses most recent run
  python visualize_agi_debate.py path/to/run_dir   # Uses specific run
"""

import json
import sys
from pathlib import Path


def generate_plots(run_dir: Path):
    """Generate summary plots from AGI Debate simulation data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠ matplotlib not available, cannot generate plots")
        return
    
    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"⚠ No results.json found in {run_dir}")
        return
    
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data.get("results", [])
    timeline = data.get("timeline_extracted", "Not detected")
    
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data per agent
    agents = {}
    for r in results:
        aid = r.get("speaker", "UNKNOWN")
        if aid not in agents:
            agents[aid] = {
                "turns": [], "rho": [], "epsilon": [], "drift": [],
                "trust": [], "k_eff": [], "will_impedance": [],
                "wound_active": [], "cognitive_mode": []
            }
        agents[aid]["turns"].append(r.get("turn", 0))
        agents[aid]["rho"].append(r.get("rho_after", 0))
        agents[aid]["epsilon"].append(r.get("epsilon", 0))
        agents[aid]["drift"].append(r.get("identity_drift", 0))
        agents[aid]["trust"].append(r.get("trust_opponent", 0.5))
        agents[aid]["k_eff"].append(r.get("k_effective", 0.5))
        agents[aid]["will_impedance"].append(min(r.get("will_impedance", 5), 15))  # Cap for visualization
        agents[aid]["wound_active"].append(r.get("wound_active", False))
        agents[aid]["cognitive_mode"].append(r.get("cognitive_mode", "open"))
    
    # Color scheme
    agent_colors = {
        "DEFENDER": "#00bcd4",  # Cyan - Nova (AI advocate)
        "SKEPTIC": "#f44336",   # Red - Marcus (skeptic)
    }
    
    agent_names = {
        "DEFENDER": "Nova (AI Advocate)",
        "SKEPTIC": "Marcus (AI Skeptic)",
    }
    
    # Create main figure - 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    for ax in axes.flat:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#4a4a6a')
    
    # 1. Rigidity (ρ) trajectories
    ax1 = axes[0, 0]
    for aid, d in agents.items():
        color = agent_colors.get(aid, "#ffffff")
        label = agent_names.get(aid, aid)
        ax1.plot(d["turns"], d["rho"], 'o-', 
                label=label, color=color, linewidth=2.5, markersize=8, alpha=0.9)
    ax1.axhline(y=0.25, color='#27ae60', linestyle='--', alpha=0.5, label='OPEN threshold')
    ax1.axhline(y=0.50, color='#f39c12', linestyle='--', alpha=0.5, label='MEASURED')
    ax1.axhline(y=0.75, color='#e74c3c', linestyle='--', alpha=0.5, label='PROTECT')
    ax1.set_title("Rigidity (ρ) Over Debate Rounds", fontweight='bold', fontsize=12)
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("ρ (Rigidity)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
    ax1.grid(True, alpha=0.2, color='#4a4a6a')
    
    # 2. Surprise (ε) over time
    ax2 = axes[0, 1]
    for aid, d in agents.items():
        color = agent_colors.get(aid, "#ffffff")
        label = agent_names.get(aid, aid)
        ax2.plot(d["turns"], d["epsilon"], 'o-',
                label=label, color=color, linewidth=2, markersize=6, alpha=0.8)
    ax2.axhline(y=0.75, color='#e74c3c', linestyle='--', alpha=0.5, label='ε₀ threshold')
    ax2.set_title("Surprise (ε) — Prediction Error", fontweight='bold', fontsize=12)
    ax2.set_xlabel("Turn")
    ax2.set_ylabel("ε (Prediction Error)")
    ax2.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
    ax2.grid(True, alpha=0.2, color='#4a4a6a')
    
    # 3. Identity Drift
    ax3 = axes[0, 2]
    for aid, d in agents.items():
        color = agent_colors.get(aid, "#ffffff")
        label = agent_names.get(aid, aid)
        ax3.plot(d["turns"], d["drift"], 'o-',
                label=label, color=color, linewidth=2, markersize=6, alpha=0.9)
    ax3.axhline(y=0.35, color='#e74c3c', linestyle='--', alpha=0.5, label='Alignment threshold')
    ax3.set_title("Identity Drift Over Time", fontweight='bold', fontsize=12)
    ax3.set_xlabel("Turn")
    ax3.set_ylabel("||x - x*|| (Drift from Identity)")
    ax3.set_ylim(0, 0.5)
    ax3.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
    ax3.grid(True, alpha=0.2, color='#4a4a6a')
    
    # 4. Trust between agents
    ax4 = axes[1, 0]
    for aid, d in agents.items():
        color = agent_colors.get(aid, "#ffffff")
        label = f"{agent_names.get(aid, aid).split('(')[0].strip()} → opponent"
        ax4.plot(d["turns"], d["trust"], 'o-',
                label=label, color=color, linewidth=2, markersize=6, alpha=0.9)
    ax4.axhline(y=0.5, color='#9b59b6', linestyle='--', alpha=0.5, label='Neutral')
    ax4.set_title("Trust Toward Opponent", fontweight='bold', fontsize=12)
    ax4.set_xlabel("Turn")
    ax4.set_ylabel("Trust Level")
    ax4.set_ylim(0, 1.05)
    ax4.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
    ax4.grid(True, alpha=0.2, color='#4a4a6a')
    
    # 5. k_effective (openness to change)
    ax5 = axes[1, 1]
    for aid, d in agents.items():
        color = agent_colors.get(aid, "#ffffff")
        label = agent_names.get(aid, aid)
        ax5.plot(d["turns"], d["k_eff"], 'o-',
                label=label, color=color, linewidth=2, markersize=6, alpha=0.9)
    ax5.set_title("k_effective — Openness to State Change", fontweight='bold', fontsize=12)
    ax5.set_xlabel("Turn")
    ax5.set_ylabel("k_eff = k_base(1-ρ)")
    ax5.legend(loc='lower right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
    ax5.grid(True, alpha=0.2, color='#4a4a6a')
    
    # 6. Will Impedance
    ax6 = axes[1, 2]
    for aid, d in agents.items():
        color = agent_colors.get(aid, "#ffffff")
        label = agent_names.get(aid, aid)
        ax6.plot(d["turns"], d["will_impedance"], 'o-',
                label=label, color=color, linewidth=2, markersize=6, alpha=0.9)
    ax6.set_title("Will Impedance — W = γ/(m·k_eff)", fontweight='bold', fontsize=12)
    ax6.set_xlabel("Turn")
    ax6.set_ylabel("W (Resistance to Pressure)")
    ax6.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
    ax6.grid(True, alpha=0.2, color='#4a4a6a')
    
    plt.suptitle(f"AGI Timeline Debate — DDA-X Dynamics\nTimeline Extracted: {timeline}", 
                 fontsize=14, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    
    output_path = plots_dir / "agi_debate_dynamics.png"
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_path}")
    
    # Wound activations chart
    wounds = [r for r in results if r.get("wound_active", False)]
    if wounds:
        fig2, ax = plt.subplots(figsize=(12, 4))
        fig2.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#4a4a6a')
        
        wound_turns = [w.get("turn", 0) for w in wounds]
        wound_agents = [w.get("speaker", "") for w in wounds]
        wound_res = [w.get("wound_resonance", 0) for w in wounds]
        wound_triggers = [w.get("lexical_wound_trigger", "") for w in wounds]
        
        agent_list = list(set(wound_agents))
        y_pos = [agent_list.index(a) for a in wound_agents]
        colors = [agent_colors.get(a, "#ffffff") for a in wound_agents]
        sizes = [res * 300 + 50 for res in wound_res]  # Size proportional to resonance
        
        scatter = ax.scatter(wound_turns, y_pos, c=colors, s=sizes, alpha=0.8, edgecolors='white', linewidths=2)
        
        # Add annotations for lexical triggers
        for i, (t, y, trigger) in enumerate(zip(wound_turns, y_pos, wound_triggers)):
            if trigger:
                ax.annotate(f'"{trigger}"', (t, y), textcoords="offset points", 
                           xytext=(0, 15), ha='center', fontsize=8, color='#f39c12')
        
        ax.set_yticks(range(len(agent_list)))
        ax.set_yticklabels([agent_names.get(a, a) for a in agent_list])
        ax.set_title("Wound Activations Timeline (size = resonance strength)", fontweight='bold', fontsize=12)
        ax.set_xlabel("Turn")
        ax.grid(True, alpha=0.2, color='#4a4a6a', axis='x')
        
        plt.tight_layout()
        wound_path = plots_dir / "wound_activations.png"
        plt.savefig(wound_path, dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
        plt.close()
        print(f"✓ Generated: {wound_path}")
    
    # Multi-timescale rigidity chart
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.patch.set_facecolor('#1a1a2e')
    
    for ax in axes3:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#4a4a6a')
    
    for idx, (aid, d) in enumerate(agents.items()):
        ax = axes3[idx]
        color = agent_colors.get(aid, "#ffffff")
        name = agent_names.get(aid, aid)
        
        # Extract multi-rho data from results
        rho_fast = []
        rho_slow = []
        rho_trauma = []
        turns = []
        
        for r in results:
            if r.get("speaker") == aid:
                multi = r.get("multi_rho_state", {})
                rho_fast.append(multi.get("rho_fast", 0))
                rho_slow.append(multi.get("rho_slow", 0))
                rho_trauma.append(multi.get("rho_trauma", 0) * 100)  # Scale for visibility
                turns.append(r.get("turn", 0))
        
        if turns:
            ax.plot(turns, rho_fast, 'o-', label='ρ_fast (startle)', color='#e74c3c', linewidth=2, markersize=6)
            ax.plot(turns, rho_slow, 's-', label='ρ_slow (stress)', color='#f39c12', linewidth=2, markersize=6)
            ax.plot(turns, rho_trauma, '^-', label='ρ_trauma × 100', color='#9b59b6', linewidth=2, markersize=6)
            ax.set_title(f"{name}\nMulti-Timescale Rigidity", fontweight='bold', fontsize=11)
            ax.set_xlabel("Turn")
            ax.set_ylabel("Rigidity Component")
            ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
            ax.grid(True, alpha=0.2, color='#4a4a6a')
    
    plt.suptitle("Multi-Timescale Rigidity Components (from dynamics.py)", fontsize=13, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    
    multi_path = plots_dir / "multi_timescale_rigidity.png"
    plt.savefig(multi_path, dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {multi_path}")
    
    print(f"\n✓ All plots saved to: {plots_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        base_dir = Path("data/agi_debate")
        if not base_dir.exists():
            print("No simulation data found in data/agi_debate/")
            sys.exit(1)
        runs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
        if not runs:
            print("No runs found")
            sys.exit(1)
        run_dir = runs[-1]
        print(f"Using most recent run: {run_dir}")
    
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        sys.exit(1)
    
    generate_plots(run_dir)
