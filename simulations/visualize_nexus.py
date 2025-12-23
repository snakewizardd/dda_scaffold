#!/usr/bin/env python3
"""
Enhanced visualization for THE NEXUS simulation.
Generates multiple charts showing entity distributions, collision networks, and energy flows.
"""

import json
import sys
from pathlib import Path


def generate_plots(run_dir: Path):
    """Generate enhanced visualizations from Nexus simulation data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available")
        return
    
    state_path = run_dir / "state.json"
    if not state_path.exists():
        print(f"No state.json found in {run_dir}")
        return
    
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    entities = data.get("entities", {})
    collisions = data.get("collisions", [])
    total_years = data.get("total_years", 0)
    
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Sector colors
    sector_colors = {
        "A": '#00bcd4',  # Fluids & Optics
        "B": '#e91e63',  # Biology & Anatomy
        "C": '#4caf50',  # Botany & Geology
        "D": '#ff5722',  # Mechanics & War
        "E": '#9c27b0',  # Abstraction & Society
    }
    
    sector_names = {
        "A": "Fluids & Optics",
        "B": "Biology & Anatomy",
        "C": "Botany & Geology",
        "D": "Mechanics & War",
        "E": "Abstraction & Society",
    }
    
    # =========================================================================
    # FIGURE 1: Entity Map with Energy
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(14, 14))
    fig1.patch.set_facecolor('#1a1a2e')
    ax1.set_facecolor('#0a0a1a')
    
    for name, e in entities.items():
        pos = e.get("position", [0.5, 0.5])
        color = sector_colors.get(e.get("sector", "A"), "#ffffff")
        energy = e.get("energy", 1.0)
        rho = e.get("rho", 0.0)
        
        # Size based on energy, alpha based on rigidity
        size = energy * 8 + 30
        alpha = max(0.4, 1 - rho)
        
        ax1.scatter(pos[0], pos[1], s=size, c=color, alpha=alpha,
                   edgecolors='white', linewidths=0.8)
        
        # Label with name and energy
        ax1.annotate(f"{name}\n({energy:.1f})", (pos[0], pos[1] + 0.02),
                    fontsize=5, ha='center', va='bottom', color='white', alpha=0.9)
    
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    ax1.set_title(f"THE NEXUS — Entity Map (Year {total_years})\nSize = Energy | Opacity = 1-Rigidity",
                 color='white', fontsize=14, fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.set_xlabel("X Position", color='white')
    ax1.set_ylabel("Y Position", color='white')
    
    # Legend
    for sector, color in sector_colors.items():
        ax1.scatter([], [], c=color, label=f"Sector {sector}: {sector_names[sector]}", s=100)
    ax1.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#4a4a6a',
              labelcolor='white', fontsize=8)
    
    for spine in ax1.spines.values():
        spine.set_color('#4a4a6a')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "entity_map.png", dpi=150, facecolor='#1a1a2e',
               edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated: {plots_dir / 'entity_map.png'}")
    
    # =========================================================================
    # FIGURE 2: Energy Distribution by Sector
    # =========================================================================
    fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig2.patch.set_facecolor('#1a1a2e')
    
    for ax in axes.flat:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#4a4a6a')
    
    # Sector breakdown
    for idx, (sector, sname) in enumerate(sector_names.items()):
        if idx >= 5:
            break
        ax = axes[idx // 3, idx % 3]
        
        sector_entities = {n: e for n, e in entities.items() if e.get("sector") == sector}
        names = list(sector_entities.keys())
        energies = [e.get("energy", 1.0) for e in sector_entities.values()]
        
        colors = [sector_colors[sector]] * len(names)
        bars = ax.barh(names, energies, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        ax.set_title(f"Sector {sector}: {sname}", color='white', fontsize=10, fontweight='bold')
        ax.set_xlabel("Energy", color='white')
        ax.tick_params(axis='y', labelsize=6)
    
    # Summary in last subplot
    ax_summary = axes[1, 2]
    sector_totals = {}
    for sector in sector_names.keys():
        sector_totals[sector] = sum(e.get("energy", 0) for e in entities.values() 
                                   if e.get("sector") == sector)
    
    wedges, texts, autotexts = ax_summary.pie(
        sector_totals.values(),
        labels=[f"Sector {s}" for s in sector_totals.keys()],
        colors=[sector_colors[s] for s in sector_totals.keys()],
        autopct='%1.1f%%',
        textprops={'color': 'white', 'fontsize': 8}
    )
    ax_summary.set_title("Total Energy by Sector", color='white', fontsize=10, fontweight='bold')
    
    plt.suptitle(f"Energy Distribution — Year {total_years}", fontsize=14, 
                fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "energy_distribution.png", dpi=150, facecolor='#1a1a2e',
               edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated: {plots_dir / 'energy_distribution.png'}")
    
    # =========================================================================
    # FIGURE 3: Collision Type Breakdown
    # =========================================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    fig3.patch.set_facecolor('#1a1a2e')
    
    for ax in axes3:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#4a4a6a')
    
    # Count collision types
    collision_types = {}
    for c in collisions:
        ctype = c.get("type", "unknown")
        collision_types[ctype] = collision_types.get(ctype, 0) + 1
    
    type_colors = {
        "synthesis": "#00ff00",
        "decay": "#ff0000",
        "design": "#0088ff",
        "resonance": "#ffff00",
        "chaos": "#ff00ff",
    }
    
    ax3a = axes3[0]
    types = list(collision_types.keys())
    counts = list(collision_types.values())
    colors = [type_colors.get(t, "#ffffff") for t in types]
    
    bars = ax3a.bar(types, counts, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax3a.set_title("Collision Types", color='white', fontsize=12, fontweight='bold')
    ax3a.set_xlabel("Type", color='white')
    ax3a.set_ylabel("Count", color='white')
    
    # Pie chart
    ax3b = axes3[1]
    wedges, texts, autotexts = ax3b.pie(
        counts, labels=types, colors=colors,
        autopct='%1.1f%%', textprops={'color': 'white', 'fontsize': 9}
    )
    ax3b.set_title("Collision Distribution", color='white', fontsize=12, fontweight='bold')
    
    plt.suptitle(f"Collision Analysis — {len(collisions):,} Total Collisions",
                fontsize=14, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "collision_analysis.png", dpi=150, facecolor='#1a1a2e',
               edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated: {plots_dir / 'collision_analysis.png'}")
    
    # =========================================================================
    # FIGURE 4: Top Synthesizers Network
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(12, 12))
    fig4.patch.set_facecolor('#1a1a2e')
    ax4.set_facecolor('#0a0a1a')
    
    # Count syntheses per entity
    synthesis_counts = {}
    for name, e in entities.items():
        synth_count = len(e.get("syntheses", []))
        synthesis_counts[name] = synth_count
    
    # Top 20 synthesizers
    top_synth = sorted(synthesis_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    names = [t[0] for t in top_synth]
    counts = [t[1] for t in top_synth]
    colors = [sector_colors.get(entities[n].get("sector", "A"), "#ffffff") for n in names]
    
    bars = ax4.barh(names, counts, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
    ax4.set_title("Top 20 Synthesizers", color='white', fontsize=14, fontweight='bold')
    ax4.set_xlabel("Number of Syntheses", color='white')
    ax4.invert_yaxis()
    
    for spine in ax4.spines.values():
        spine.set_color('#4a4a6a')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "top_synthesizers.png", dpi=150, facecolor='#1a1a2e',
               edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated: {plots_dir / 'top_synthesizers.png'}")
    
    # =========================================================================
    # FIGURE 5: Rigidity vs Energy (DDA-X Dynamics)
    # =========================================================================
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    fig5.patch.set_facecolor('#1a1a2e')
    ax5.set_facecolor('#16213e')
    
    for name, e in entities.items():
        energy = e.get("energy", 1.0)
        rho = e.get("rho", 0.0)
        sector = e.get("sector", "A")
        color = sector_colors.get(sector, "#ffffff")
        
        ax5.scatter(rho, energy, s=100, c=color, alpha=0.8,
                   edgecolors='white', linewidths=0.5)
        ax5.annotate(name, (rho, energy), fontsize=6, ha='left', va='bottom',
                    color='white', alpha=0.7)
    
    ax5.set_xlabel("Rigidity (ρ)", color='white', fontsize=12)
    ax5.set_ylabel("Energy", color='white', fontsize=12)
    ax5.set_title("DDA-X Dynamics: Rigidity vs Energy\nHigh rigidity → Decay victims | Low rigidity + High energy → Synthesizers",
                 color='white', fontsize=12, fontweight='bold')
    ax5.tick_params(colors='white')
    
    for spine in ax5.spines.values():
        spine.set_color('#4a4a6a')
    
    ax5.axhline(y=10, color='#27ae60', linestyle='--', alpha=0.5, label='Energy threshold')
    ax5.axvline(x=0.5, color='#e74c3c', linestyle='--', alpha=0.5, label='Rigidity threshold')
    ax5.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#4a4a6a',
              labelcolor='white', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "rigidity_vs_energy.png", dpi=150, facecolor='#1a1a2e',
               edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Generated: {plots_dir / 'rigidity_vs_energy.png'}")
    
    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        base_dir = Path("data/nexus")
        if not base_dir.exists():
            print("No simulation data found in data/nexus/")
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
