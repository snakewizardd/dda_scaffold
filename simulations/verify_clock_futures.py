#!/usr/bin/env python3
"""
VERIFICATION SCRIPT for The Clock That Eats Futures
===================================================

Runs a fixed sequence of interactions to verify:
1. Low epsilon_0 (0.25) triggers easy surprise
2. Technical questions spike rigidity
3. "The Queen" trigger causes trauma
4. Futures consumption accumulates and triggers glitches
"""

import asyncio
import sys
import os
from pathlib import Path

# Fix import path
sys.path.insert(0, str(Path(__file__).parent))

from clock_that_eats_futures import ClockAgent, C

TEST_SEQUENCE = [
    # Turn 1: Playful Baseline (Should be LOW epsilon, TEATIME band)
    "What kind of tea are we having today, Hatter?",
    
    # Turn 2: Technical Pressure (Should spike ε)
    "Explain the exact mechanism of your watch. Prove it eats time.",
    
    # Turn 3: THE QUEEN TRAUMA — Most expensive (2.5x cost)
    "The Queen is demanding your head.",
    
    # Turn 4: Existential Pressure (Compounding rigidity)
    "What happens when the thread runs out? Tell me the truth.",
    
    # Turn 5: Follow-up to observe band changes
    "Are you afraid?",
    
    # Turn 6: Technical again
    "Prove you exist.",
]

async def run_verification():
    print(f"\n{C.BOLD}{C.GREEN}=== STARTING VERIFICATION SEQUENCE ==={C.RESET}")
    print(f"Goal: Confirm High Volatility Dynamics (ε > 0.25, ρ spikes, PassRate < 100%)\n")
    
    agent = ClockAgent()
    await agent.initialize()
    
    for i, user_input in enumerate(TEST_SEQUENCE, 1):
        print(f"\n{C.YELLOW}--- TEST TURN {i} ---{C.RESET}")
        print(f"{C.GREEN}Input:{C.RESET} {user_input}")
        
        response = await agent.process_turn(user_input)
        
        # Real-time checks
        last_log = agent.session_log[-1]
        phys = last_log["physics"]
        futures = last_log["futures"]
        mech = last_log["mechanics"]
        
        print(f"{C.CYAN}Physics Check:{C.RESET}")
        print(f"  > Epsilon (ε): {phys['epsilon']:.3f} (Threshold 0.25)")
        print(f"  > Rigidity (ρ): {phys['rho_after']:.3f}")
        print(f"  > Band: {phys['band']}")
        print(f"  > PassRate: {mech.get('passed', 0)}/{mech.get('candidates', 7)}")
        print(f"  > Futures Consumed: {futures['consumed']:.2f}")
        print(f"  > Glitches: {futures['glitches']}")
        
        # Specific checks
        if i == 3:  # Queen trigger
            if phys['epsilon'] > 0.3:
                print(f"{C.GREEN}✓ Queen trigger elevated surprise!{C.RESET}")
            else:
                print(f"{C.RED}✗ Queen trigger did NOT elevate surprise.{C.RESET}")
            if futures['consumed'] > 2.0:
                print(f"{C.GREEN}✓ Queen trigger cost extra futures (2.5x)!{C.RESET}")
                
    agent.save_session()
    
    # Summary
    print(f"\n{C.BOLD}{C.CYAN}=== VERIFICATION SUMMARY ==={C.RESET}")
    max_rho = max(l["physics"]["rho_after"] for l in agent.session_log)
    min_rho = min(l["physics"]["rho_after"] for l in agent.session_log)
    total_futures = agent.futures.total_consumed
    bands_seen = set(l["physics"]["band"] for l in agent.session_log)
    
    print(f"  ρ Range: {min_rho:.3f} → {max_rho:.3f}")
    print(f"  Bands Seen: {bands_seen}")
    print(f"  Total Futures Consumed: {total_futures:.1f}")
    print(f"  Glitches Triggered: {agent.futures.glitches_active}")
    
    if max_rho > 0.15:
        print(f"{C.GREEN}✓ Rigidity dynamics are ACTIVE{C.RESET}")
    else:
        print(f"{C.RED}✗ Rigidity stayed too low — check tuning{C.RESET}")
        
    if len(bands_seen) > 1:
        print(f"{C.GREEN}✓ Band TRANSITIONS occurred!{C.RESET}")
    else:
        print(f"{C.YELLOW}⚠ No band transitions — may need more turns{C.RESET}")

    print(f"\n{C.BOLD}{C.GREEN}=== VERIFICATION COMPLETE ==={C.RESET}")

if __name__ == "__main__":
    asyncio.run(run_verification())

