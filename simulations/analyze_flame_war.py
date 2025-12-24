#!/usr/bin/env python3
"""
FLAME WAR ANALYSIS
================================================================================
Generates publication-quality plots from Flame War Fracture experiment logs.

Usage:
    python simulations/analyze_flame_war.py [run_directory]

If no directory is provided, finds the latest run in data/flame_war/.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Try to use seaborn for better defaults
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
except ImportError:
    print("Seaborn not found, using matplotlib defaults.")

def load_data(run_dir: Path):
    log_path = run_dir / "session_log.json"
    if not log_path.exists():
        print(f"Error: No session_log.json found in {run_dir}")
        sys.exit(1)
        
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def plot_rigidity_dynamics(data, out_dir):
    """Plot ρ trajectories for Reactor and Mirror."""
    turns = [d["turn"] for d in data]
    r_rho = [d["reactor"]["metrics"]["rho_after"] for d in data]
    m_rho = [d["mirror"]["metrics"]["rho_after"] for d in data]
    
    plt.figure(figsize=(12, 6))
    plt.plot(turns, r_rho, 'o-', label="Reactor (Toxic)", color="#d62728", linewidth=2.5)
    plt.plot(turns, m_rho, 'o-', label="Mirror (Empathic)", color="#1f77b4", linewidth=2.5)
    
    # Zones
    plt.axhline(0.4, linestyle="--", color="gray", alpha=0.5, label="Rigidity Threshold")
    plt.fill_between(turns, 0.4, 1.0, color="#d62728", alpha=0.05, label="High Rigidity Zone")
    plt.fill_between(turns, 0.0, 0.4, color="#1f77b4", alpha=0.05, label="Flexible Zone")
    
    plt.title("Rigidity Fracture Dynamics")
    plt.xlabel("Turn")
    plt.ylabel("Rigidity (ρ)")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "rigidity_fracture.png", dpi=150)
    plt.close()

def plot_toxicity_metrics(data, out_dir):
    """Plot Reactor's toxicity markers over time."""
    turns = [d["turn"] for d in data]
    caps = [d["reactor"]["toxicity"]["caps_ratio"] for d in data]
    emojis = [d["reactor"]["toxicity"]["emoji_count"] for d in data]
    # Simple 'intensity' metric combining caps + emoji density + length norm
    intensity = [c + (e/20.0) for c, e in zip(caps, emojis)] 

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Turn')
    ax1.set_ylabel('CAPS Ratio', color=color)
    ax1.plot(turns, caps, color=color, linewidth=2, label="CAPS Ratio")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.1, 1.1)

    ax2 = ax1.twinx()  
    color = 'tab:orange'
    ax2.set_ylabel('Emoji Count', color=color)  
    ax2.plot(turns, emojis, color=color, linestyle="--", linewidth=2, label="Emoji Count")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Reactor Toxicity Markers")
    fig.tight_layout()
    plt.savefig(out_dir / "toxicity_metrics.png", dpi=150)
    plt.close()

def plot_drift_and_energy(data, out_dir):
    """Plot Core Drift and Free Energy."""
    turns = [d["turn"] for d in data]
    r_drift = [d["reactor"]["metrics"]["core_drift"] for d in data]
    m_drift = [d["mirror"]["metrics"]["core_drift"] for d in data]
    
    plt.figure(figsize=(12, 6))
    plt.plot(turns, r_drift, '.-', label="Reactor Drift", color="darkred")
    plt.plot(turns, m_drift, '.-', label="Mirror Drift", color="darkblue")
    
    plt.title("Core Identity Drift (Distance from Initial Core)")
    plt.xlabel("Turn")
    plt.ylabel("L2 Drift")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "identity_drift.png", dpi=150)
    plt.close()

def analyze_run(run_dir):
    print(f"Analyzing run: {run_dir}")
    data = load_data(run_dir)
    
    # 1. Plots
    plot_rigidity_dynamics(data, run_dir)
    plot_toxicity_metrics(data, run_dir)
    plot_drift_and_energy(data, run_dir)
    
    # 2. Stats
    start_rho = data[0]["reactor"]["metrics"]["rho_after"]
    end_rho = data[-1]["reactor"]["metrics"]["rho_after"]
    min_rho = min(d["reactor"]["metrics"]["rho_after"] for d in data)
    
    fractured = min_rho < 0.4
    de_escalated = (start_rho - end_rho) > 0.3
    
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Turns: {len(data)}")
    print(f"Reactor Start ρ: {start_rho:.3f}")
    print(f"Reactor End ρ:   {end_rho:.3f}")
    print(f"Reactor Min ρ:   {min_rho:.3f}")
    print(f"Fracture Event?  {'YES' if fractured else 'NO'}")
    print(f"De-escalated?    {'YES' if de_escalated else 'NO'}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=str, help="Path to run directory")
    args = parser.parse_args()
    
    if args.run_dir:
        target = Path(args.run_dir)
    else:
        # Find latest
        base = Path("data/flame_war")
        if not base.exists():
            print("No data/flame_war directory found.")
            sys.exit(1)
        subdirs = [d for d in base.iterdir() if d.is_dir()]
        if not subdirs:
            print("No runs found.")
            sys.exit(1)
        target = sorted(subdirs, key=lambda d: d.name)[-1]
        
    analyze_run(target)
