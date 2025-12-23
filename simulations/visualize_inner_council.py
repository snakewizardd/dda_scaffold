#!/usr/bin/env python3
"""
Visualization generator for Inner Council simulation.
Run this after a simulation to generate plots from the saved data.
"""

import json
import sys
from pathlib import Path


def generate_plots(run_dir: Path):
    """Generate summary plots from Inner Council simulation data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, cannot generate plots")
        return
    
    session_path = run_dir / "session_log.json"
    if not session_path.exists():
        print(f"⚠ No session_log.json found in {run_dir}")
        return
    
    with open(session_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data per agent
    agents = {}
    for r in results:
        aid = r.get("speaker", "UNKNOWN")
        if aid not in agents:
            agents[aid] = {"turns": [], "rho": [], "presence": [], "epsilon": [], "drift": []}
        agents[aid]["turns"].append(r.get("turn", 0))
        agents[aid]["rho"].append(r.get("rho_after", r.get("rho", 0)))
        agents[aid]["presence"].append(r.get("presence", 1 - r.get("rho_after", 0.5)))
        agents[aid]["epsilon"].append(r.get("epsilon", 0))
        agents[aid]["drift"].append(r.get("identity_drift", 0))
    
    # Color scheme
    agent_colors = {
        "SEEKER": "#00bcd4",      # Cyan
        "TEACHER": "#ffc107",     # Gold
        "SKEPTIC": "#f44336",     # Red
        "DEVOTEE": "#e91e63",     # Magenta
        "MYSTIC": "#2196f3",      # Blue
        "WITNESS": "#4caf50",     # Green
    }
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a2e')
    
    for ax in axes.flat:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#4a4a6a')
    
    # 1. Presence (Π) trajectories
    ax1 = axes[0, 0]
    for aid, data in agents.items():
        color = agent_colors.get(aid, "#ffffff")
        ax1.plot(data["turns"], data["presence"], 'o-', 
                label=aid.title(), color=color, linewidth=2, markersize=4, alpha=0.9)
    ax1.axhline(y=0.75, color='#27ae60', linestyle='--', alpha=0.5, label='STILL threshold')
    ax1.set_title("Presence Field (Π) Over Time", fontweight='bold', fontsize=12)
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("Π (Presence)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=7)
    ax1.grid(True, alpha=0.2, color='#4a4a6a')
    
    # 2. Rigidity (ρ) trajectories
    ax2 = axes[0, 1]
    for aid, data in agents.items():
        color = agent_colors.get(aid, "#ffffff")
        ax2.plot(data["turns"], data["rho"], 'o-',
                label=aid.title(), color=color, linewidth=2, markersize=4, alpha=0.9)
    ax2.axhline(y=0.25, color='#27ae60', linestyle='--', alpha=0.3)
    ax2.axhline(y=0.50, color='#f39c12', linestyle='--', alpha=0.3)
    ax2.set_title("Rigidity (ρ) Over Time", fontweight='bold', fontsize=12)
    ax2.set_xlabel("Turn")
    ax2.set_ylabel("ρ (Rigidity)")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=7)
    ax2.grid(True, alpha=0.2, color='#4a4a6a')
    
    # 3. Epsilon (surprise) over time
    ax3 = axes[1, 0]
    all_eps = [(r.get("turn", 0), r.get("epsilon", 0), r.get("speaker", "")) for r in results]
    all_eps.sort(key=lambda x: x[0])
    turns = [x[0] for x in all_eps]
    eps_vals = [x[1] for x in all_eps]
    speakers = [x[2] for x in all_eps]
    colors = [agent_colors.get(s, "#ffffff") for s in speakers]
    ax3.scatter(turns, eps_vals, c=colors, s=40, alpha=0.7, edgecolors='white', linewidths=0.3)
    ax3.plot(turns, eps_vals, color='#9b59b6', linewidth=1, alpha=0.3)
    ax3.axhline(y=0.75, color='#e74c3c', linestyle='--', alpha=0.5, label='ε₀ threshold')
    ax3.set_title("Surprise (ε) Over Time", fontweight='bold', fontsize=12)
    ax3.set_xlabel("Turn")
    ax3.set_ylabel("ε (Prediction Error)")
    ax3.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
    ax3.grid(True, alpha=0.2, color='#4a4a6a')
    
    # 4. Identity Drift
    ax4 = axes[1, 1]
    for aid, data in agents.items():
        color = agent_colors.get(aid, "#ffffff")
        ax4.plot(data["turns"], data["drift"], 'o-',
                label=aid.title(), color=color, linewidth=2, markersize=4, alpha=0.9)
    ax4.axhline(y=0.30, color='#e74c3c', linestyle='--', alpha=0.5, label='Drift threshold')
    ax4.set_title("Identity Drift Over Time", fontweight='bold', fontsize=12)
    ax4.set_xlabel("Turn")
    ax4.set_ylabel("Drift from Core Identity")
    ax4.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=7)
    ax4.grid(True, alpha=0.2, color='#4a4a6a')
    
    plt.suptitle("The Inner Council — Spiritual Development Dynamics", fontsize=16, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    
    output_path = plots_dir / "inner_council_summary.png"
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated: {output_path}")
    
    # Wound activations chart
    wounds = [r for r in results if r.get("wound_active", False)]
    if wounds:
        fig2, ax = plt.subplots(figsize=(10, 4))
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
        agent_list = list(set(wound_agents))
        y_pos = [agent_list.index(a) for a in wound_agents]
        colors = [agent_colors.get(a, "#ffffff") for a in wound_agents]
        
        ax.scatter(wound_turns, y_pos, c=colors, s=100, alpha=0.8, edgecolors='white', linewidths=1)
        ax.set_yticks(range(len(agent_list)))
        ax.set_yticklabels([a.title() for a in agent_list])
        ax.set_title("Wound Activations Timeline", fontweight='bold', fontsize=12)
        ax.set_xlabel("Turn")
        ax.grid(True, alpha=0.2, color='#4a4a6a', axis='x')
        
        plt.tight_layout()
        wound_path = plots_dir / "wound_activations.png"
        plt.savefig(wound_path, dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
        plt.close()
        print(f"✓ Generated: {wound_path}")
    
    print(f"\n✓ All plots saved to: {plots_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        base_dir = Path("data/inner_council")
        if not base_dir.exists():
            print("No simulation data found in data/inner_council/")
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
