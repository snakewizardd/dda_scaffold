import asyncio
import sys
import os
import time

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.llm.hybrid_provider import HybridProvider, PersonalityParams

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

async def verify_physics():
    print(f"{CYAN}================================================={RESET}")
    print(f"{CYAN}   DDA-X HYPOTHESIS & PHYSICS VERIFICATION{RESET}")
    print(f"{CYAN}================================================={RESET}")
    print("Testing the Theoretical Loop: State (ρ) -> Parameters -> Behavior")

    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        timeout=300.0
    )

    # The Prompt to test behavior
    test_prompt = "Define the concept of 'Risk' in one sentence."

    # 1. Iterate through Rigidity States
    rigidities = [0.1, 0.5, 0.9]
    
    for rho in rigidities:
        print(f"\n{YELLOW}-------------------------------------------------{RESET}")
        print(f"{YELLOW}STATE CHECK: Rigidity (ρ) = {rho}{RESET}")
        
        # A. Verify Parameter Binding (The Physics)
        params = PersonalityParams.from_rigidity(rho, "balanced")
        print(f"{GREEN}[PHYSICS] Calculated LLM Parameters:{RESET}")
        print(f"  -> Temperature:     {params.temperature:.2f}  (ρ influence)")
        print(f"  -> Top P:           {params.top_p:.2f}")
        print(f"  -> Frequency Pen:   {params.frequency_penalty:.2f}")
        print(f"  -> Presence Pen:    {params.presence_penalty:.2f}")

        # B. Verify Behavioral Output (The Result)
        print(f"{GREEN}[BEHAVIOR] Generating response...{RESET}")
        
        start = time.time()
        print(f"  Prompt: '{test_prompt}'")
        sys.stdout.write("  Response: ")
        
        full_response = ""
        async for token in provider.stream(
            prompt=test_prompt,
            personality_params=params,
            max_tokens=60
        ):
            if token.startswith("__THOUGHT__"):
                continue # Skip thoughts for cleanliness in this test
            sys.stdout.write(token)
            sys.stdout.flush()
            full_response += token
            
        print(f"\n  (Time: {time.time()-start:.2f}s)")

    print(f"\n{CYAN}================================================={RESET}")
    print(f"{CYAN}VERIFICATION COMPLETE{RESET}")
    print("If successful, you should see:")
    print("1. Low ρ (0.1) -> High Temp -> More creative/varied definition.")
    print("2. High ρ (0.9) -> Low Temp -> Strict, dictionary-like definition.")

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            os.system('color')
        asyncio.run(verify_physics())
    except KeyboardInterrupt:
        print("\nStopped.")
