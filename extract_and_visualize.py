import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    log_path = r"c:\Users\danie\Desktop\dda_scaffold\data\brobot\20251229_090255\session_log.json"
    output_dir = r"C:\Users\danie\.gemini\antigravity\brain\e5fce599-6e91-4251-af29-6a2719fe8d63"
    os.makedirs(output_dir, exist_ok=True)

    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Extract Transcript
    transcript_path = os.path.join(output_dir, 'transcript.md')
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(f"# Session Transcript: {data.get('experiment', 'Unknown')}\n")
        f.write(f"Timestamp: {data.get('timestamp', 'N/A')}\n\n")
        f.write(f"**Model:** {data['config'].get('chat_model', 'Unknown')}\n\n")
        for turn in data.get('turns', []):
            f.write(f"### Turn {turn['turn']}\n")
            f.write(f"**User:** {turn['user_input']}\n\n")
            f.write(f"**BROBOT:** {turn['agent_response']}\n\n")
            
            m = turn.get('agent_metrics', {})
            f.write(f"**DDA-X Metrics:** ρ={m.get('rho_after',0):.3f}, ε={m.get('epsilon',0):.3f}, core_drift={m.get('core_drift',0):.3f}, band={m.get('band','N/A')}\n\n")
            f.write("---\n\n")

    # 2. Extract Metrics for Dashboard
    metrics = []
    for turn in data.get('turns', []):
        m = turn.get('agent_metrics', {})
        row = {
            'turn': turn['turn'],
            'rho_after': m.get('rho_after', 0),
            'rho_fast': m.get('rho_fast', 0),
            'rho_slow': m.get('rho_slow', 0),
            'rho_trauma': m.get('rho_trauma', 0),
            'epsilon': m.get('epsilon', 0),
            'arousal': m.get('arousal', 0),
            'core_drift': m.get('core_drift', 0),
            'g': m.get('g', 0)
        }
        metrics.append(row)

    df = pd.DataFrame(metrics)

    # Professional Grade Visualization Dashboard
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
    fig.suptitle(f"DDA-X Cognitive Dynamics Dashboard - Session {os.path.basename(os.path.dirname(log_path))}", fontsize=18, y=0.95)

    # Panel 1: Rigidity Trajectories
    ax1 = axes[0, 0]
    ax1.plot(df['turn'], df['rho_after'], label='Total Rigidity (ρ)', color='#FF00FF', linewidth=3, marker='o')
    ax1.plot(df['turn'], df['rho_fast'], label='Fast component', color='#00FFFF', linestyle='--', alpha=0.6)
    ax1.plot(df['turn'], df['rho_slow'], label='Slow component', color='#FFFF00', linestyle='--', alpha=0.6)
    ax1.fill_between(df['turn'], df['rho_after'], color='#FF00FF', alpha=0.1)
    ax1.set_title('Rigidity Evolution (Multi-timescale)', fontsize=14)
    ax1.set_ylabel('Amplitude')
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.15)

    # Panel 2: Surprise and Arousal
    ax2 = axes[0, 1]
    ax2.plot(df['turn'], df['epsilon'], label='Surprise (ε)', color='#00FF00', linewidth=2, marker='x')
    ax2.set_ylabel('Surprise', color='#00FF00')
    ax2.tick_params(axis='y', labelcolor='#00FF00')
    
    ax2b = ax2.twinx()
    ax2b.plot(df['turn'], df['arousal'], label='Arousal', color='#FFA500', linewidth=2, linestyle='-.')
    ax2b.set_ylabel('Arousal', color='#FFA500')
    ax2b.tick_params(axis='y', labelcolor='#FFA500')
    ax2b.set_ylim(0, 1.0)
    
    ax2.set_title('Surprise & Emotional Arousal', fontsize=14)
    ax2.grid(True, alpha=0.15)

    # Panel 3: Identity Stability & Core Drift
    ax3 = axes[1, 0]
    ax3.plot(df['turn'], df['core_drift'], label='Core Drift', color='#FF4500', linewidth=2)
    ax3.plot(df['turn'], df['g'], label='Conductance (g)', color='#4169E1', linewidth=2, linestyle=':')
    ax3.fill_between(df['turn'], df['core_drift'], color='#FF4500', alpha=0.1)
    ax3.set_title('Identity Stability Metrics', fontsize=14)
    ax3.set_ylabel('Metric Value')
    ax3.set_xlabel('Turn')
    ax3.legend()
    ax3.grid(True, alpha=0.15)

    # Panel 4: Phase Space (Rho vs Epsilon)
    ax4 = axes[1, 1]
    ax4.scatter(df['epsilon'], df['rho_after'], c=df['turn'], cmap='plasma', s=100, alpha=0.8, edgecolor='white')
    ax4.plot(df['epsilon'], df['rho_after'], color='white', alpha=0.3, linestyle='-')
    ax4.set_title('Phase Space: Rigidity vs Surprise', fontsize=14)
    ax4.set_xlabel('Surprise (ε)')
    ax4.set_ylabel('Rigidity (ρ)')
    ax4.grid(True, alpha=0.15)
    
    # Add turn annotations to phase space
    for i, txt in enumerate(df['turn']):
        ax4.annotate(txt, (df['epsilon'][i], df['rho_after'][i]), xytext=(5, 5), textcoords='offset points', fontsize=8, color='white')

    viz_path = os.path.join(output_dir, 'dda_x_dashboard.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Enhanced transcript saved to {transcript_path}")
    print(f"Professional Dashboard saved to {viz_path}")

if __name__ == '__main__':
    main()
