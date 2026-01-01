
import asyncio
import os
import sys
import numpy as np

# Adjust path to find the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulations.torah_study_bot import TorahBotSim, CONFIG

# MOCK PROVIDER to save tokens and ensure determinism
class MockProvider:
    def __init__(self):
        self.call_count = 0

    async def complete(self, prompt, **kwargs):
        self.call_count += 1
        return f"Rank {self.call_count}: This is mock candidate #{self.call_count}."

    async def embed(self, text):
        return np.random.rand(768)

    async def embed_batch(self, texts):
        return [np.random.rand(768) for _ in texts]

async def run_debug():
    print("==========================================")
    print("🐞 DEBUG RENDERER: K-Candidate Verification")
    print("==========================================")
    
    # 1. Init Bot
    bot = TorahBotSim()
    bot.provider = MockProvider()
    print("[1] Bot Initialized with Mock Provider.")
    print(f"    CONFIG['gen_candidates'] (K) = {CONFIG['gen_candidates']}")
    
    # 2. Mock State
    bot.study.hebrew_text = "א"
    bot.study.attempt_present = True
    bot.initialized = True  # Skip real init
    
    # 3. Process Turn
    print("\n[2] Processing Turn (Simulating /attempt)...")
    resp = await bot.process_turn("qc attempt")
    
    # 4. Analyze Result
    print("\n[3] Analysis of Response:")
    print(f"    Type: {resp.get('type')}")
    
    metrics = resp.get("metrics")
    if not metrics:
        print("    ❌ NO METRICS FOUND!")
        return

    cands = metrics.get("candidates_debug", [])
    print(f"    Candidate Count in Metrics: {len(cands)}")
    
    print("\n[4] 🧠 DDA-X Corridor Dump:")
    for i, c in enumerate(cands):
        status = "✅ SELECTED" if c["is_chosen"] else "❌ REJECTED"
        print(f"    [{i+1}] {status} | J={c['J_final']:.3f} | Text: '{c['text']}'")
        
    if len(cands) == 3:
        print("\n✅ SUCCESS: 3 Candidates detected in output structure.")
    else:
        print(f"\n❌ FAILURE: Expected 3, got {len(cands)}.")

if __name__ == "__main__":
    asyncio.run(run_debug())
