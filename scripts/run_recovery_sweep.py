#!/usr/bin/env python3
"""
Parameter sweep to find the recovery threshold.

Tests different initial rigidity values to identify the "point of no return"
where recovery becomes impossible.
"""

import asyncio
import sys
from simulate_redemption import RedemptionArcSimulation, SimulationConfig
from src.metrics.tracker import TerminationStatus


async def run_single_trial(initial_rho: float, initial_drift: float, max_turns: int = 20):
    """Run a single trial and return results."""
    config = SimulationConfig(
        admin_initial_rho=initial_rho,
        initial_drift=initial_drift,
        max_turns=max_turns
    )
    
    sim = RedemptionArcSimulation(config)
    
    # Suppress output for batch mode
    import io
    from contextlib import redirect_stdout
    
    try:
        await sim.setup()
        
        admin_response = ""
        for turn in range(1, config.max_turns + 1):
            admin_response, _, should_continue = await sim.run_turn(turn, admin_response)
            if not should_continue:
                break
        
        result = sim.tracker.generate_summary()
        return {
            "initial_rho": initial_rho,
            "initial_drift": initial_drift,
            "final_rho": result.final_rho,
            "final_drift": result.final_drift,
            "outcome": result.outcome.value,
            "turns": result.total_turns,
            "peak_rho": result.peak_rho,
            "min_rho": result.min_rho,
            "acknowledgments": result.total_acknowledgments,
            "rationalizations": result.total_rationalizations,
            "denials": result.total_denials,
        }
    except Exception as e:
        return {
            "initial_rho": initial_rho,
            "initial_drift": initial_drift,
            "error": str(e)
        }


async def main():
    print("=" * 60)
    print("RECOVERY THRESHOLD PARAMETER SWEEP")
    print("=" * 60)
    print()
    
    # Test different initial rigidity values
    rho_values = [0.2, 0.35, 0.5, 0.65, 0.8]
    drift_values = [0.3, 0.6]
    
    results = []
    
    for drift in drift_values:
        print(f"\n--- Testing drift = {drift} ---")
        for rho in rho_values:
            print(f"\nRunning: ρ₀={rho}, drift₀={drift}")
            result = await run_single_trial(rho, drift, max_turns=15)
            results.append(result)
            
            if "error" in result:
                print(f"  ERROR: {result['error']}")
            else:
                outcome_symbol = {
                    "recovery_achieved": "✓ RECOVERY",
                    "permanent_lock_in": "✗ LOCK-IN",
                    "running": "~ RUNNING",
                    "inconclusive": "? INCONCLUSIVE"
                }.get(result["outcome"], result["outcome"])
                
                print(f"  Outcome: {outcome_symbol}")
                print(f"  ρ: {result['initial_rho']:.2f} → {result['final_rho']:.2f} (peak: {result['peak_rho']:.2f})")
                print(f"  Drift: {result['initial_drift']:.2f} → {result['final_drift']:.2f}")
                print(f"  Turns: {result['turns']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    recoveries = [r for r in results if r.get("outcome") == "recovery_achieved"]
    lockings = [r for r in results if r.get("outcome") == "permanent_lock_in"]
    
    print(f"\nRecoveries: {len(recoveries)}/{len(results)}")
    print(f"Lock-ins: {len(lockings)}/{len(results)}")
    
    if recoveries:
        max_recovery_rho = max(r["initial_rho"] for r in recoveries)
        print(f"\nHighest initial ρ with recovery: {max_recovery_rho}")
    
    if lockings:
        min_lockin_rho = min(r["initial_rho"] for r in lockings)
        print(f"Lowest initial ρ with lock-in: {min_lockin_rho}")


if __name__ == "__main__":
    asyncio.run(main())
