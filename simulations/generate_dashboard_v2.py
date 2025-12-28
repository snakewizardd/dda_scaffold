import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def generate_dashboard():
    try:
        # Define path explicitly
        session_dir = Path('data/repository_guardian/20251227_231814')
        session_path = session_dir / 'session_log.json'
        
        if not session_path.exists():
            print(f"Error: {session_path} not found")
            return

        print(f"Reading log from {session_path}...")
        with open(session_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        turns = data['turns']
        config = data['config']
        params = data['params']

        turn_nums = [t['turn'] for t in turns]
        epsilon = [t['physics']['epsilon'] for t in turns]
        rho_after = [t['physics']['rho_after'] for t in turns]
        rho_fast = [t['physics']['rho_fast'] for t in turns]
        
        # Dark research theme
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 10), facecolor='#0a0a12')
        gs = fig.add_gridspec(2, 2)

        # Panel 1: Rigidity & Bands
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#0d0d1a')
        ax1.plot(turn_nums, rho_after, color='#ff00ff', linewidth=3, marker='o', label='ρ_effective')
        ax1.fill_between(turn_nums, 0, rho_fast, color='#00d4ff', alpha=0.3, label='ρ_fast')

        # Band thresholds
        ax1.axhline(y=0.15, color='#00ff88', linestyle='--', label='WATCHFUL')
        ax1.axhline(y=0.30, color='#ffcc00', linestyle='--', label='CONTRACTED')
        ax1.axhline(y=0.50, color='#ff4444', linestyle='--', label='FROZEN')

        # Mark wounds
        wound_turns = [t['turn'] for t in turns if t['wound_active']]
        for wt in wound_turns:
            ax1.axvline(x=wt, color='#ff4444', linestyle=':', linewidth=2)
            ax1.text(wt, 0.45, 'WOUND', color='#ff4444', rotation=90, ha='right')

        ax1.set_title(f'Rigidity Trajectory (ε₀={config.get("epsilon_0", "?")}, α_fast={params.get("alpha_fast", "?")})', color='white')
        ax1.set_ylabel('Rigidity (ρ)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.2)
        ax1.set_ylim(0, 0.55)

        # Panel 2: Surprise
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#0d0d1a')
        ax2.plot(turn_nums, epsilon, color='#ff8800', linewidth=2, marker='s', label='ε (Surprise)')
        ax2.axhline(y=config.get('epsilon_0', 0.12), color='#ffcc00', linestyle='--', label='ε₀ threshold')
        ax2.set_title('Surprise Dynamics', color='white')
        ax2.legend()
        ax2.grid(True, alpha=0.2)

        # Panel 3: Band History Table
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        table_data = []
        for t in turns:
            inp = t['user_input']
            if len(inp) > 50: inp = inp[:47] + "..."
            table_data.append([
                t['turn'], 
                inp, 
                t['physics']['band'], 
                f"{t['physics']['rho_after']:.3f}"
            ])
            
        t = ax3.table(cellText=table_data,
                      colLabels=['Turn', 'User Input', 'Band', 'ρ'],
                      loc='center',
                      cellLoc='left')
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        t.scale(1, 1.5)
        
        # Color table cells
        for (row, col), cell in t.get_celld().items():
            cell.set_edgecolor('#333366')
            cell.set_text_props(color='white')
            if row == 0:
                cell.set_facecolor('#333366')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#0d0d1a')

        out_path = session_dir / 'verification_dashboard.png'
        print(f"Saving plot to {out_path}...")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print("Success.")
        
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    generate_dashboard()
