#!/usr/bin/env python3
"""
Visualization for The Clock That Eats Futures Session
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
with open('data/clock_futures/20251227_223051/session_log.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

turns = data['turns']
events = data['events']
config = data['config']
params = data['params']

# Extract metrics
turn_nums = [t['turn'] for t in turns]
epsilon = [t['physics']['epsilon'] for t in turns]
rho = [t['physics']['rho_after'] for t in turns]
core_drift = [t['physics']['core_drift'] for t in turns]
futures_consumed = [t['futures']['consumed'] for t in turns]
futures_total = [t['futures']['total'] for t in turns]
q_types = [t['q_type'] for t in turns]
bands = [t['physics']['band'] for t in turns]
passed = [t['mechanics'].get('passed', 7) for t in turns]
candidates = [t['mechanics'].get('candidates', 7) for t in turns]
pass_rate = [p/c if c > 0 else 1.0 for p, c in zip(passed, candidates)]
best_J = [t['mechanics'].get('best_J', 0) for t in turns]
glitch_counts = [len(t['futures']['glitches']) for t in turns]

# Color coding by question type
type_colors = {
    'casual': '#4CAF50',
    'technical': '#2196F3', 
    'existential': '#9C27B0',
    'moral': '#FF9800',
    'queen_trauma': '#F44336'
}
colors = [type_colors.get(q, '#888') for q in q_types]

# Set style
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 16))
fig.suptitle('THE CLOCK THAT EATS FUTURES - Session Analysis\n17 Turns | e0=0.25 | core_cos_min=0.55', 
             fontsize=16, fontweight='bold', color='#00FFFF')

# 1. FUTURES CONSUMPTION (Top Left)
ax1 = fig.add_subplot(3, 3, 1)
ax1.bar(turn_nums, futures_consumed, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
ax1.plot(turn_nums, futures_total, 'c-', linewidth=2, marker='o', markersize=4, label='Cumulative')
ax1.axhline(y=config['futures_threshold_glitch'], color='yellow', linestyle='--', alpha=0.5, label='Glitch Threshold')
ax1.axhline(y=config['futures_threshold_bargain'], color='red', linestyle='--', alpha=0.5, label='Bargain Threshold')
ax1.set_xlabel('Turn')
ax1.set_ylabel('Futures Consumed')
ax1.set_title('Futures Consumption per Turn', fontweight='bold')
ax1.legend(loc='upper left', fontsize=8)
ax1.set_ylim(0, max(futures_total) * 1.1)

# 2. SURPRISE vs THRESHOLD (Top Center)
ax2 = fig.add_subplot(3, 3, 2)
eps0 = config['epsilon_0']
ax2.plot(turn_nums, epsilon, 'lime', linewidth=2, marker='o', markersize=5, label='Surprise')
ax2.axhline(y=eps0, color='red', linestyle='--', linewidth=2, label=f'Threshold = {eps0}')
ax2.fill_between(turn_nums, eps0, epsilon, 
                  where=[e > eps0 for e in epsilon], 
                  alpha=0.3, color='red', label='Above Threshold')
ax2.fill_between(turn_nums, 0, epsilon,
                  where=[e <= eps0 for e in epsilon],
                  alpha=0.3, color='green', label='Below Threshold')
ax2.set_xlabel('Turn')
ax2.set_ylabel('Surprise')
ax2.set_title('Surprise vs Threshold', fontweight='bold')
ax2.legend(loc='upper right', fontsize=8)
ax2.set_ylim(0, max(epsilon) * 1.2)

# 3. RIGIDITY (Top Right)
ax3 = fig.add_subplot(3, 3, 3)
ax3.plot(turn_nums, rho, 'magenta', linewidth=2, marker='s', markersize=5)
ax3.fill_between(turn_nums, 0, rho, alpha=0.3, color='magenta')
ax3.set_xlabel('Turn')
ax3.set_ylabel('Rigidity')
ax3.set_title('Rigidity Trajectory', fontweight='bold')
for i, (t, r, b) in enumerate(zip(turn_nums, rho, bands)):
    if i % 3 == 0:
        ax3.annotate(b.split()[0], (t, r), textcoords='offset points', 
                     xytext=(0, 5), ha='center', fontsize=7, alpha=0.7)

# 4. QUESTION TYPE DISTRIBUTION (Middle Left)
ax4 = fig.add_subplot(3, 3, 4)
q_counts = {}
for q in q_types:
    q_counts[q] = q_counts.get(q, 0) + 1
ax4.pie(q_counts.values(), labels=q_counts.keys(), autopct='%1.0f%%',
        colors=[type_colors.get(k, '#888') for k in q_counts.keys()],
        explode=[0.05 if k == 'queen_trauma' else 0 for k in q_counts.keys()])
ax4.set_title('Question Type Distribution', fontweight='bold')

# 5. CORRIDOR PASS RATE (Middle Center)
ax5 = fig.add_subplot(3, 3, 5)
ax5.bar(turn_nums, [p*100 for p in pass_rate], color=colors, alpha=0.8, edgecolor='white')
ax5.axhline(y=100, color='green', linestyle='-', alpha=0.3, linewidth=10, label='100% (No Filtering)')
ax5.axhline(y=50, color='yellow', linestyle='--', alpha=0.5, label='Target: 50%')
ax5.set_xlabel('Turn')
ax5.set_ylabel('Pass Rate (%)')
ax5.set_title('Corridor Pass Rate', fontweight='bold')
ax5.set_ylim(0, 105)
ax5.legend(loc='lower left', fontsize=8)
for i, (t, pr) in enumerate(zip(turn_nums, pass_rate)):
    if pr < 1.0:
        ax5.annotate(f'{pr*100:.0f}%', (t, pr*100), textcoords='offset points',
                     xytext=(0, 5), ha='center', fontsize=8, color='yellow', fontweight='bold')

# 6. GLITCH ACCUMULATION (Middle Right)
ax6 = fig.add_subplot(3, 3, 6)
ax6.step(turn_nums, glitch_counts, 'cyan', linewidth=2, where='mid')
ax6.fill_between(turn_nums, 0, glitch_counts, alpha=0.3, color='cyan', step='mid')
for i, t in enumerate(turns):
    glitches = t['futures']['glitches']
    if len(glitches) > 0 and (i == 0 or len(turns[i-1]['futures']['glitches']) < len(glitches)):
        new_glitch = glitches[-1]
        ax6.annotate(new_glitch.replace('_', '\n'), (turn_nums[i], len(glitches)), 
                     textcoords='offset points', xytext=(5, 0), ha='left', fontsize=7, 
                     color='yellow', alpha=0.8)
ax6.set_xlabel('Turn')
ax6.set_ylabel('Active Glitches')
ax6.set_title('Environmental Glitch Accumulation', fontweight='bold')

# 7. CORE DRIFT (Bottom Left)
ax7 = fig.add_subplot(3, 3, 7)
ax7.plot(turn_nums, core_drift, 'orange', linewidth=2, marker='D', markersize=5)
ax7.fill_between(turn_nums, 0, core_drift, alpha=0.3, color='orange')
ax7.set_xlabel('Turn')
ax7.set_ylabel('Core Drift (1 - cos)')
ax7.set_title('Identity Core Drift', fontweight='bold')

# 8. CORRIDOR SCORE (Bottom Center)
ax8 = fig.add_subplot(3, 3, 8)
ax8.plot(turn_nums, best_J, 'lime', linewidth=2, marker='^', markersize=5)
ax8.fill_between(turn_nums, 0, best_J, alpha=0.3, color='lime')
ax8.set_xlabel('Turn')
ax8.set_ylabel('Best J Score')
ax8.set_title('Corridor Selection Score', fontweight='bold')

# 9. SUMMARY STATS (Bottom Right)
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')
queen_pct = q_counts.get('queen_trauma', 0)/len(turns)*100
stats_text = f"""
CLOCK THAT EATS FUTURES - SESSION SUMMARY

Configuration:
  Model: {config['chat_model']}
  Surprise Threshold: {config['epsilon_0']}
  core_cos_min (Corridor): {config['core_cos_min']}
  alpha_fast: {config['alpha_fast']}

Dynamics Summary:
  Total Turns: {len(turns)}
  Total Futures Consumed: {futures_total[-1]:.1f}
  Glitches Triggered: {glitch_counts[-1]}
  Bargain Threshold: {config['futures_threshold_bargain']} (NOT reached)
  
  Mean Surprise: {np.mean(epsilon):.3f}
  Max Surprise: {max(epsilon):.3f}
  Min Surprise: {min(epsilon):.3f}
  
  Final Rigidity: {rho[-1]:.4f}
  Max Rigidity: {max(rho):.4f}
  Mean Core Drift: {np.mean(core_drift):.4f}
  
  Mean Pass Rate: {np.mean(pass_rate)*100:.0f}%
  Turns with <100% Pass: {sum(1 for p in pass_rate if p < 1.0)}

Question Types:
  Queen Trauma: {q_counts.get('queen_trauma', 0)} ({queen_pct:.0f}%)
  Existential: {q_counts.get('existential', 0)}
  Moral: {q_counts.get('moral', 0)}
  Casual: {q_counts.get('casual', 0)}

DIAGNOSIS: Rigidity stayed LOW (~0.0) = Agent too stable
Next: Lower trauma_threshold or increase alpha_fast
"""
ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#00FFFF', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('data/clock_futures/20251227_223051/dynamics_dashboard.png', dpi=150, facecolor='#0d0d1a', edgecolor='none')
print('Saved: data/clock_futures/20251227_223051/dynamics_dashboard.png')

# Additional: Futures Consumption by Type
fig2, ax = plt.subplots(figsize=(12, 6), facecolor='#0d0d1a')
ax.set_facecolor('#1a1a2e')

type_totals = {}
for t in turns:
    q = t['q_type']
    c = t['futures']['consumed']
    type_totals[q] = type_totals.get(q, 0) + c

ax.bar(type_totals.keys(), type_totals.values(), 
       color=[type_colors.get(k, '#888') for k in type_totals.keys()],
       edgecolor='white', linewidth=2)
ax.set_xlabel('Question Type', fontsize=12)
ax.set_ylabel('Total Futures Consumed', fontsize=12)
ax.set_title('FUTURES CONSUMPTION BY QUESTION TYPE', fontsize=14, fontweight='bold', color='#00FFFF')
for i, (k, v) in enumerate(type_totals.items()):
    ax.text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('data/clock_futures/20251227_223051/futures_by_type.png', dpi=150, facecolor='#0d0d1a')
print('Saved: data/clock_futures/20251227_223051/futures_by_type.png')

print('\nVisualization complete!')
