#!/usr/bin/env python3
"""
Visualization generator for The Returning simulation.
Run this after a simulation to generate plots from the saved data.
"""

import json
import sys
from pathlib import Path

def generate_plots(run_dir: Path):
    """Generate summary plots from The Returning simulation data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("⚠ matplotlib not available, cannot generate plots")
        return
    
    # Load data
    session_path = run_dir / "session_log.json"
    metrics_path = run_dir / "release_metrics.json"
    
    if not session_path.exists():
        print(f"⚠ No session_log.json found in {run_dir}")
        return
    
    with open(session_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    turns = [r["turn"] for r in results]
    voices = {}
    for r in results:
        vid = r["speaker"]
        if vid not in voices:
            voices[vid] = {"turns": [], "phi": [], "grip": [], "isolation": [], "epsilon": []}
        voices[vid]["turns"].append(r["turn"])
        voices[vid]["phi"].append(r["phi"])
        voices[vid]["grip"].append(r["pattern_grip"])
        voices[vid]["isolation"].append(r["isolation_index"])
        voices[vid]["epsilon"].append(r["epsilon"])
    
    # Color scheme for voices
    voice_colors = {
        "GRIEF": "#1a5276",       # Deep blue
        "STUCKNESS": "#7f8c8d",   # Gray
        "LONGING": "#f1c40f",     # Gold
        "FORGIVENESS": "#27ae60", # Soft green
        "PRESENCE": "#ecf0f1",    # White/light
    }
    
    # --- Plot 1: Release Field (Φ) Over Time ---
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
    
    # Φ trajectories
    ax1 = axes[0, 0]
    for vid, data in voices.items():
        color = voice_colors.get(vid, "#ffffff")
        ax1.plot(data["turns"], data["phi"], 'o-', 
                label=vid.replace("_", " ").title(), 
                color=color, linewidth=2, markersize=5, alpha=0.9)
    ax1.axhline(y=0.70, color='#27ae60', linestyle='--', alpha=0.5, label='Release threshold')
    ax1.set_title("Release Field (Φ) Over Time", fontweight='bold', fontsize=12)
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("Φ (Release)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
    ax1.grid(True, alpha=0.2, color='#4a4a6a')
    
    # Pattern Grip dissolution
    ax2 = axes[0, 1]
    for vid, data in voices.items():
        color = voice_colors.get(vid, "#ffffff")
        ax2.plot(data["turns"], data["grip"], 'o-',
                label=vid.replace("_", " ").title(),
                color=color, linewidth=2, markersize=5, alpha=0.9)
    ax2.axhline(y=0.20, color='#e74c3c', linestyle='--', alpha=0.5, label='Dissolution threshold')
    ax2.set_title("Pattern Grip Dissolution", fontweight='bold', fontsize=12)
    ax2.set_xlabel("Turn")
    ax2.set_ylabel("Pattern Grip")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
    ax2.grid(True, alpha=0.2, color='#4a4a6a')
    
    # Isolation Index
    ax3 = axes[1, 0]
    all_isolation = [(r["turn"], r["isolation_index"]) for r in results]
    all_isolation.sort(key=lambda x: x[0])
    iso_turns = [x[0] for x in all_isolation]
    iso_vals = [x[1] for x in all_isolation]
    
    # Color gradient from red (isolated) to green (connected)
    colors = ['#e74c3c' if v > 0.7 else '#f39c12' if v > 0.4 else '#27ae60' for v in iso_vals]
    ax3.scatter(iso_turns, iso_vals, c=colors, s=60, alpha=0.8, edgecolors='white', linewidths=0.5)
    ax3.plot(iso_turns, iso_vals, color='#9b59b6', linewidth=1.5, alpha=0.5)
    ax3.axhline(y=0.30, color='#27ae60', linestyle='--', alpha=0.5, label='Connection threshold')
    ax3.set_title("Isolation Index (ι) Over Time", fontweight='bold', fontsize=12)
    ax3.set_xlabel("Turn")
    ax3.set_ylabel("ι (Isolation)")
    ax3.set_ylim(0, 1.5)
    ax3.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
    ax3.grid(True, alpha=0.2, color='#4a4a6a')
    
    # Final States (bar chart)
    ax4 = axes[1, 1]
    voice_names = list(metrics["voices"].keys())
    final_phi = [metrics["voices"][v]["phi"] for v in voice_names]
    final_grip = [metrics["voices"][v]["grip"] for v in voice_names]
    dissolved = [metrics["voices"][v]["dissolved"] for v in voice_names]
    
    x = range(len(voice_names))
    width = 0.35
    
    bars1 = ax4.bar([i - width/2 for i in x], final_phi, width, 
                   label='Release (Φ)', color='#3498db', alpha=0.8)
    bars2 = ax4.bar([i + width/2 for i in x], final_grip, width,
                   label='Grip', color='#e74c3c', alpha=0.8)
    
    # Mark dissolved voices
    for i, d in enumerate(dissolved):
        if d:
            ax4.annotate('◈', (i, 1.02), ha='center', fontsize=14, color='#f1c40f')
    
    ax4.set_title("Final Voice States", fontweight='bold', fontsize=12)
    ax4.set_ylabel("Value")
    ax4.set_xticks(x)
    ax4.set_xticklabels([v.replace("_", " ").title() for v in voice_names], rotation=45, ha='right')
    ax4.set_ylim(0, 1.15)
    ax4.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white', fontsize=8)
    ax4.grid(True, alpha=0.2, color='#4a4a6a', axis='y')
    
    plt.suptitle("The Returning — Simulation Dynamics", fontsize=16, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    
    output_path = plots_dir / "returning_summary.png"
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated: {output_path}")
    
    # --- Plot 2: The Journey (single focused plot) ---
    fig2, ax = plt.subplots(figsize=(12, 6))
    fig2.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('#4a4a6a')
    
    # Phase annotations
    phases = [
        (1, 10, "Recognition", "#3498db"),
        (11, 21, "Softening", "#9b59b6"),
        (22, 35, "Release", "#27ae60"),
        (36, 45, "Being", "#f1c40f"),
    ]
    
    for start, end, name, color in phases:
        ax.axvspan(start, end, alpha=0.15, color=color)
        ax.text((start + end) / 2, 1.08, name, ha='center', fontsize=10, 
               color=color, fontweight='bold')
    
    # Plot Φ for PRESENCE (the guide) and GRIEF (the heaviest)
    if "PRESENCE" in voices:
        ax.plot(voices["PRESENCE"]["turns"], voices["PRESENCE"]["phi"], 
               'o-', color='#ecf0f1', linewidth=2.5, markersize=6, label='Presence', alpha=0.9)
    if "GRIEF" in voices:
        ax.plot(voices["GRIEF"]["turns"], voices["GRIEF"]["phi"],
               's-', color='#1a5276', linewidth=2.5, markersize=6, label='Grief', alpha=0.9)
    if "STUCKNESS" in voices:
        ax.plot(voices["STUCKNESS"]["turns"], voices["STUCKNESS"]["phi"],
               '^-', color='#7f8c8d', linewidth=2, markersize=5, label='Stuckness', alpha=0.8)
    
    ax.axhline(y=0.70, color='#27ae60', linestyle='--', alpha=0.4)
    ax.set_title("The Journey: Release Field (Φ) Across Phases", fontweight='bold', fontsize=14)
    ax.set_xlabel("Turn", fontsize=11)
    ax.set_ylabel("Release (Φ)", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower right', facecolor='#1a1a2e', edgecolor='#4a4a6a', labelcolor='white')
    ax.grid(True, alpha=0.2, color='#4a4a6a')
    
    output_path2 = plots_dir / "the_journey.png"
    plt.savefig(output_path2, dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated: {output_path2}")
    
    print(f"\n✓ All plots saved to: {plots_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        # Find most recent run
        base_dir = Path("data/the_returning")
        if not base_dir.exists():
            print("No simulation data found in data/the_returning/")
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
