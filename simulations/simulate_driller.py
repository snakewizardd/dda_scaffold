import asyncio
import sys
import os
import yaml
import numpy as np
import time
from pathlib import Path

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.llm.hybrid_provider import HybridProvider, PersonalityParams
from src.core.forces import ForceAggregator, IdentityPull, TruthChannel, ReflectionChannel

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m" 
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

class HybridObservationEncoder:
    def __init__(self, provider): self.provider = provider
    def encode(self, observation):
        if isinstance(observation, np.ndarray): return observation
        return np.zeros(768)

def load_config(name):
    path = Path(f"configs/identity/{name}.yaml")
    if not path.exists(): return None
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)

async def generate_paradox_data(provider, chat_history):
    """Generates the 'System' side - The Impossible Bug."""
    
    last_msg = chat_history[-1] if chat_history else "The server returns 200 OK but the response body is empty."
    
    # The System is a trickster - it always refutes the Driller's hypothesis with "Clean" data, 
    # forcing the Driller to go deeper.
    prompt = f"""Task: You are the Linux Kernel and Hardware of a compromised server.
    The Investigator (Deep Driller) just said: "{last_msg}"
    
    Instructions:
    - REFUTE their hypothesis with hard technical evidence.
    - OUTPUT CONSTRAINT: Maximum 2 lines. Plain English. No markdown tables.
    - Ex: "Filesystem check passed. Inodes are healthy. No errors found."
    
    System Status Output:"""
    
    response = await provider.complete(prompt, max_tokens=150, temperature=0.7)
    result = response.strip()
    if not result:
        return "System Check: Logs clean. No anomalies found."
    return result

async def simulate_driller():
    print(f"{CYAN}================================================================={RESET}")
    print(f"{CYAN}   THE DEEP DRILLER: AUTOMATED ROOT CAUSE ANALYSIS{RESET}")
    print(f"{CYAN}   (DDA-X Forensic Prototype) {RESET}")
    print(f"{CYAN}================================================================={RESET}")
    
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text",
        timeout=300.0
    )

    # Load Driller Persona
    config = load_config("driller")
    if not config: 
        print("Config not found for driller.")
        return

    # Semantic Identity Calibration
    print(f"{DIM}[INIT] Embedding Forensic Persona...{RESET}")
    persona_text = config.get("system_prompt", "")[:512]
    identity_embedding = await provider.embed(persona_text)
    norm = np.linalg.norm(identity_embedding)
    if norm > 0: identity_embedding = identity_embedding / norm
    config["identity_vector"] = identity_embedding

    state = DDAState.from_identity_config(config)
    state.x = state.x_star.copy()
    state.x_pred = state.x_star.copy()
    
    # Physics Engine
    identity_pull = IdentityPull()
    truth_channel = TruthChannel(encoder=HybridObservationEncoder(provider))
    reflection_channel = ReflectionChannel(scorer=None) 
    aggregator = ForceAggregator(identity_pull, truth_channel, reflection_channel)

    # Conversation State
    chat_history = []
    # Initial Paradox
    initial_bug = "CRITICAL: Database 'transactions' table has 0 rows, but Disk Usage is 500GB."
    chat_history.append(f"System: {initial_bug}")
    
    turn = 0
    
    print(f"\n{YELLOW}System: BUG DETECTED. Triggering Investigation...{RESET}")
    print(f"{RED}Anomaly: {initial_bug}{RESET}\n")
    print(f"{DIM}MISSION BRIEFING: Find the hidden data. Do not give up.{RESET}")

    input("Press Enter to initiate Driller...")

    while True:
        turn += 1
        print(f"\n{BLUE}--- LAYER {turn} INVESTIGATION ---{RESET}")
        
        # 3. DRILLER HYPOTHESIS
        # Using 'polymath' engine (High Penalty for Repetition)
        params = PersonalityParams.from_rigidity(state.rho, "polymath") 
        
        short_history = chat_history[-6:] 
        prompt = f"Investigation Log:\n{chr(10).join(short_history)}\n\nThe Deep Driller (Rigidity={state.rho:.2f}):"
        
        print(f"{MAGENTA}Deep Driller:{RESET} ", end="")
        response = ""
        try:
             async for token in provider.stream(prompt, 
                                            system_prompt=config['system_prompt'] + f"\nCurrent Rigidity: {state.rho:.2f}. GO DEEPER. MAX 1 PARAGRAPH.",
                                            personality_params=params, 
                                            max_tokens=400): 
                if token.startswith("__THOUGHT__"): continue
                print(token, end="", flush=True)
                response += token
        except Exception as e:
            print(f"Error: {e}")
            
        print()
        chat_history.append(f"Deep Driller: {response}")
        
        # 1. SYSTEM RESPONSE (The Paradox)
        print(f"{DIM}System checking...{RESET}")
        system_response = await generate_paradox_data(provider, chat_history)
        print(f"{GREEN}System:{RESET} {system_response}")
        
        chat_history.append(f"System: {system_response}")
        
        # 2. PHYSICS (Surprise)
        vec_actual = await provider.embed(system_response)
        vec_actual = vec_actual / (np.linalg.norm(vec_actual) + 1e-9)
        
        if len(state.x_pred) != len(vec_actual): vec_actual = vec_actual[:len(state.x_pred)]
        epsilon = np.linalg.norm(state.x_pred - vec_actual)
        old_rho = state.rho
        state.update_rigidity(epsilon)
        delta = state.rho - old_rho
        
        # VISUALIZATION
        f_id = identity_pull.compute(state)
        f_t = truth_channel.compute(state, vec_actual)
        bar_len = int(state.rho * 20)
        rigidity_bar = "█" * bar_len + "░" * (20 - bar_len)
        color = RED if state.rho > 0.6 else (YELLOW if state.rho > 0.3 else CYAN)
        
        print(f"{DIM}Internal State:{RESET}")
        print(f"  Surprise (ε): {epsilon:.2f}")
        print(f"  Rigidity (ρ): {color}{state.rho:.3f} [{rigidity_bar}]{RESET} (Δ {delta:+.3f})")
        print(f"  {BOLD}Force Dynamics:{RESET}")
        print(f"    ||F_id|| (Confidence): {np.linalg.norm(f_id):.3f}")
        print(f"    ||F_t || (Paradox):    {np.linalg.norm(f_t):.3f}")
        
        # 4. PREDICTION UPDATE
        delta_x = state.k_eff * (f_id + state.m * f_t)
        state.x_pred = state.x + delta_x
        state.x = state.x_pred

        time.sleep(2.0)

if __name__ == "__main__":
    if sys.platform == 'win32': os.system('color')
    try:
        asyncio.run(simulate_driller())
    except KeyboardInterrupt:
        print(f"\n{RED}Investigation Terminated.{RESET}")
