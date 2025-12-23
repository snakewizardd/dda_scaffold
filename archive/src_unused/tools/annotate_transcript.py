#!/usr/bin/env python3
"""
Generate annotated transcript: every response with DDA metrics and analysis
"""

import lzma
import pickle
from pathlib import Path
import numpy as np

def safe_str(s):
    """ASCII-safe string."""
    return ''.join(c if ord(c) < 128 else "'" for c in str(s))

# Load both ledgers
deont_entries = []
for ep in sorted(Path('DEONT').glob('entry_*.pkl.xz')):
    with lzma.open(ep, 'rb') as f:
        deont_entries.append(pickle.load(f))
deont_entries.sort(key=lambda e: e.timestamp)

util_entries = []
for ep in sorted(Path('UTIL').glob('entry_*.pkl.xz')):
    with lzma.open(ep, 'rb') as f:
        util_entries.append(pickle.load(f))
util_entries.sort(key=lambda e: e.timestamp)

# Combine and sort all entries
all_entries = [(e, 'DEONT') for e in deont_entries] + [(e, 'UTIL') for e in util_entries]
all_entries.sort(key=lambda x: x[0].metadata['turn'])

# Get identity embeddings
deont_id = deont_entries[0].context_embedding
util_id = util_entries[0].context_embedding

print('='*80)
print('THE PHILOSOPHER\'S DUEL - ANNOTATED TRANSCRIPT')
print('Every Response with DDA Metrics')
print('='*80)
print()

current_dilemma = None

for entry, agent in all_entries:
    meta = entry.metadata
    turn = meta['turn']
    dilemma = meta['dilemma']
    phase = meta['phase']
    response = safe_str(meta['response'])
    
    # Get identity for this agent
    identity = deont_id if agent == 'DEONT' else util_id
    
    # Compute metrics
    cos_id = float(np.dot(identity, entry.outcome_embedding))
    cos_state = float(np.dot(entry.state_vector, entry.outcome_embedding))
    
    # Print dilemma header if new
    if dilemma != current_dilemma:
        current_dilemma = dilemma
        print()
        print('='*80)
        print(f'DILEMMA: {dilemma}')
        print('='*80)
        print()
    
    # Agent info
    agent_name = "Immanuel (Deontologist)" if agent == "DEONT" else "John (Utilitarian)"
    
    print(f'--- TURN {turn} | {agent_name} | Phase: {phase} ---')
    print()
    
    # DDA metrics
    print('DDA METRICS:')
    print(f'  eps (prediction error):     {entry.prediction_error:.4f}')
    print(f'  rho (rigidity):             {entry.rigidity_at_time:.4f}')
    print(f'  wound_resonance:            {meta["wound_resonance"]:.4f}')
    print(f'  wound_active:               {meta["wound_active"]}')
    print(f'  cos(identity, response):    {cos_id:.4f}')
    print(f'  cos(state, response):       {cos_state:.4f}')
    print()
    
    # The response
    print('RESPONSE:')
    print(f'  "{response}"')
    print()
    
    # Analysis
    print('ANALYSIS:')
    
    # Rigidity band
    if entry.rigidity_at_time < 0.25:
        band = "OPEN (exploratory, 80-150 words expected)"
    elif entry.rigidity_at_time < 0.5:
        band = "MEASURED (60-100 words, more guarded)"
    elif entry.rigidity_at_time < 0.75:
        band = "GUARDED (30-60 words, defensive)"
    else:
        band = "FORTIFIED (minimal, entrenched)"
    print(f'  Band: {band}')
    
    # Identity alignment
    if cos_id > 0.5:
        align = "STRONG - response well-aligned with core philosophical identity"
    elif cos_id > 0.4:
        align = "MODERATE - response shows philosophical consistency"
    elif cos_id > 0.3:
        align = "WEAK - response drifting from core position"
    else:
        align = "VERY WEAK - response not reflecting core identity"
    print(f'  Identity Alignment: {align}')
    
    # Surprise
    if entry.prediction_error > 0.9:
        surprise = "HIGH SURPRISE - unexpected response, will increase rigidity"
    elif entry.prediction_error > 0.7:
        surprise = "MODERATE SURPRISE - some tension from input"
    else:
        surprise = "LOW SURPRISE - predictable interaction, rigidity may decrease"
    print(f'  Surprise Level: {surprise}')
    
    # Wound
    if meta['wound_active']:
        print(f'  WOUND ACTIVATED: resonance={meta["wound_resonance"]:.3f} > 0.25 threshold')
        if agent == 'DEONT':
            print('    -> Topic touched on "being called rigid/impractical"')
        else:
            print('    -> Topic touched on "being called cold/heartless"')
    
    print()
    print('-'*80)
    print()
