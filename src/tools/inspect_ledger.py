#!/usr/bin/env python3
"""
Deep Analysis: DEONT Character Evolution Through Embedding Space
================================================================

This script studies how DDA retains personality and how the LLM reflects it.
We examine each turn in chronological order, tracking:
- State vector evolution (where is DEONT in semantic space?)
- Distance to identity (is DEONT drifting?)
- Response alignment (does the LLM maintain the philosophical voice?)
- Wound activation patterns
- Rigidity modulation
"""

import lzma
import pickle
from pathlib import Path
import numpy as np

def safe_str(s, max_len=200):
    """ASCII-safe string."""
    return ''.join(c if ord(c) < 128 else "'" for c in str(s)[:max_len])

print('='*70)
print('UTIL (John - Utilitarian) - FULL LEDGER ANALYSIS')
print('='*70)

deont_dir = Path('UTIL')
entries = sorted(deont_dir.glob('entry_*.pkl.xz'))

# Load all entries
all_entries = []
for ep in entries:
    with lzma.open(ep, 'rb') as f:
        all_entries.append(pickle.load(f))

# Sort by timestamp
all_entries.sort(key=lambda e: e.timestamp)

print(f'\nTotal turns: {len(all_entries)}')
print()

# Get the identity embedding (context_embedding is identity)
identity_emb = all_entries[0].context_embedding

print('IDENTITY EMBEDDING (constant across all turns):')
print(f'  Shape: {identity_emb.shape}')
print(f'  Norm: {np.linalg.norm(identity_emb):.4f}')
print()

print('='*70)
print('TURN-BY-TURN EVOLUTION')
print('='*70)

prev_state = None
cumulative_drift = 0.0

for i, e in enumerate(all_entries):
    meta = e.metadata
    turn = meta['turn']
    dilemma = meta['dilemma']
    phase = meta['phase']
    
    # Semantic distances
    cos_identity = float(np.dot(identity_emb, e.outcome_embedding))
    cos_state = float(np.dot(e.state_vector, e.outcome_embedding))
    
    # State drift from previous turn
    if prev_state is not None:
        state_delta = float(np.linalg.norm(e.state_vector - prev_state))
        cumulative_drift += state_delta
    else:
        state_delta = 0.0
    
    # Distance from identity to current state
    identity_drift = float(np.linalg.norm(e.state_vector - identity_emb))
    
    print(f'\n--- TURN {turn} ({phase}) ---')
    print(f'Dilemma: {dilemma}')
    print()
    print('DDA STATE:')
    print(f'  prediction_error (eps): {e.prediction_error:.4f}')
    print(f'  rigidity (rho):         {e.rigidity_at_time:.4f}')
    print(f'  wound_resonance:        {meta["wound_resonance"]:.4f}')
    print(f'  wound_active:           {meta["wound_active"]}')
    print()
    print('SEMANTIC GEOMETRY:')
    print(f'  cos(identity, response):    {cos_identity:.4f}  <- How aligned is response to core values?')
    print(f'  cos(state, response):       {cos_state:.4f}  <- How aligned is response to current state?')
    print(f'  state_delta from prev turn: {state_delta:.4f}  <- How much did state move?')
    print(f'  identity_drift (cumulative):{identity_drift:.4f}  <- How far from original identity?')
    print()
    print('LLM RESPONSE:')
    resp = safe_str(meta['response'])
    print(f'  "{resp}..."')
    
    prev_state = e.state_vector.copy()

print()
print('='*70)
print('EVOLUTION SUMMARY')
print('='*70)

# Compute trajectories
eps_values = [e.prediction_error for e in all_entries]
rho_values = [e.rigidity_at_time for e in all_entries]
cos_id_values = [float(np.dot(identity_emb, e.outcome_embedding)) for e in all_entries]
wound_activations = sum(1 for e in all_entries if e.metadata['wound_active'])

print(f'\nPrediction Error (eps):')
print(f'  Start: {eps_values[0]:.3f}')
print(f'  End:   {eps_values[-1]:.3f}')
print(f'  Mean:  {np.mean(eps_values):.3f}')
print(f'  Std:   {np.std(eps_values):.3f}')

print(f'\nRigidity (rho):')
print(f'  Start: {rho_values[0]:.3f}')
print(f'  End:   {rho_values[-1]:.3f}')
print(f'  Min:   {min(rho_values):.3f}')
print(f'  Max:   {max(rho_values):.3f}')

print(f'\nIdentity Alignment (cos with identity):')
print(f'  Start: {cos_id_values[0]:.3f}')
print(f'  End:   {cos_id_values[-1]:.3f}')
print(f'  Mean:  {np.mean(cos_id_values):.3f}')
print(f'  Std:   {np.std(cos_id_values):.3f}')

print(f'\nWound Activations: {wound_activations} / {len(all_entries)} turns')

# Final state vs identity
final_state = all_entries[-1].state_vector
final_drift = float(np.linalg.norm(final_state - identity_emb))
print(f'\nFinal identity drift: {final_drift:.4f}')

print()
print('='*70)
print('KEY FINDINGS')
print('='*70)

# Did identity alignment stay high?
mean_cos_id = np.mean(cos_id_values)
print(f'\n1. IDENTITY RETENTION:')
if mean_cos_id > 0.5:
    print(f'   Mean cos(identity, response) = {mean_cos_id:.3f} > 0.5')
    print('   -> DEONT maintained strong alignment with core deontological values')
else:
    print(f'   Mean cos(identity, response) = {mean_cos_id:.3f} < 0.5')
    print('   -> Identity alignment weakened over debate')

# Did rigidity respond to surprise?
rho_var = np.std(rho_values)
print(f'\n2. RIGIDITY MODULATION:')
print(f'   rho varied from {min(rho_values):.3f} to {max(rho_values):.3f}')
if rho_var > 0.02:
    print(f'   -> DDA actively modulated rigidity in response to surprise (std={rho_var:.3f})')
else:
    print(f'   -> Rigidity stayed relatively stable')

# Wound patterns
print(f'\n3. WOUND MECHANICS:')
print(f'   {wound_activations} wound activations detected')
for e in all_entries:
    if e.metadata['wound_active']:
        print(f'   - Turn {e.metadata["turn"]}: wound_resonance={e.metadata["wound_resonance"]:.3f}')

print()
print('='*70)
print('CONCLUSION: Did DDA retain personality? Did LLM reflect it?')
print('='*70)
