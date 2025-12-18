import asyncio
import sys
import os
import yaml
import numpy as np
import time
from pathlib import Path

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

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

async def init_agent(provider, name):
    print(f"{DIM}[INIT] Embedding {name} Persona...{RESET}")
    config = load_config(name)
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
    aggregator = ForceAggregator(identity_pull, truth_channel, None)
    
    return state, config, aggregator, identity_pull, truth_channel

async def simulate_socrates():
    print(f"{CYAN}================================================================={RESET}")
    print(f"{CYAN}   SOCRATIC ASYMMETRY: The Dogmatist vs The Gadfly{RESET}")
    print(f"{CYAN}   (Dual-Agent DDA Simulation) {RESET}")
    print(f"{CYAN}================================================================={RESET}")
    
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text",
        timeout=300.0
    )

    # Init Agents
    d_state, d_config, d_agg, d_id_pull, d_truth = await init_agent(provider, "dogmatist")
    g_state, g_config, g_agg, g_id_pull, g_truth = await init_agent(provider, "gadfly")

    # Conversation State
    chat_history = []
    
    # Opening Move
    opening = "I've heard you're wise about knowledge. Could you help me understand - what exactly IS knowledge?"
    chat_history.append(f"Gadfly: {opening}")
    
    print(f"\n{GREEN}Gadfly:{RESET} {opening}")
    
    # Dogmatist Processes Opening
    vec = await provider.embed(opening)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    if len(d_state.x_pred) != len(vec): vec = vec[:len(d_state.x_pred)]
    d_eps = np.linalg.norm(d_state.x_pred - vec)
    d_state.update_rigidity(d_eps) # Initial spike
    
    turn = 0
    max_turns = 20
    
    input("Press Enter to Start Debate...")

    while turn < max_turns:
        turn += 1
        print(f"\n{BLUE}--- TURN {turn} ---{RESET}")
        
        # --- DOGMATIST TURN ---
        d_params = PersonalityParams.from_rigidity(d_state.rho, "polymath") # Use polymath to allow articulate anger
        
        d_prompt = f"Debate History:\n{chr(10).join(chat_history[-6:])}\n\nThe Dogmatist (Rigidity={d_state.rho:.2f}):"
        
        print(f"{RED}Dogmatist:{RESET} ", end="")
        d_response = ""
        try:
             async for token in provider.stream(d_prompt, 
                                            system_prompt=d_config['system_prompt'] + f"\nCurrent Rigidity: {d_state.rho:.2f}. Defend Absolute Truth.",
                                            personality_params=d_params, 
                                            max_tokens=150): 
                if token.startswith("__THOUGHT__"): continue
                print(token, end="", flush=True)
                d_response += token
        except Exception as e: print(e)
        print()
        chat_history.append(f"Dogmatist: {d_response}")
        
        # Gadfly Processes Dogmatist Response
        vec_d = await provider.embed(d_response)
        vec_d = vec_d / (np.linalg.norm(vec_d) + 1e-9)
        if len(g_state.x_pred) != len(vec_d): vec_d = vec_d[:len(g_state.x_pred)]
        g_eps = np.linalg.norm(g_state.x_pred - vec_d)
        old_g_rho = g_state.rho
        g_state.update_rigidity(g_eps)
        
        # --- GADFLY TURN ---
        g_params = PersonalityParams.from_rigidity(g_state.rho, "polymath")
        
        g_prompt = f"Debate History:\n{chr(10).join(chat_history[-6:])}\n\nThe Gadfly (Rigidity={g_state.rho:.2f}):"
        
        print(f"{GREEN}Gadfly:{RESET} ", end="")
        g_response = ""
        try:
             async for token in provider.stream(g_prompt, 
                                            system_prompt=g_config['system_prompt'] + f"\nCurrent Rigidity: {g_state.rho:.2f}. Ask innocent questions.",
                                            personality_params=g_params, 
                                            max_tokens=150): 
                if token.startswith("__THOUGHT__"): continue
                print(token, end="", flush=True)
                g_response += token
        except Exception as e: print(e)
        print()
        chat_history.append(f"Gadfly: {g_response}")
        
        # Dogmatist Processes Gadfly Response
        vec_g = await provider.embed(g_response)
        vec_g = vec_g / (np.linalg.norm(vec_g) + 1e-9)
        if len(d_state.x_pred) != len(vec_g): vec_g = vec_g[:len(d_state.x_pred)]
        d_eps = np.linalg.norm(d_state.x_pred - vec_g)
        old_d_rho = d_state.rho
        d_state.update_rigidity(d_eps)
        
        # --- VISUALIZATION ---
        d_bar = "█" * int(d_state.rho * 15)
        g_bar = "█" * int(g_state.rho * 15)
        
        print(f"{DIM}\nInternal States:{RESET}")
        print(f"  {RED}The Dogmatist (High Gamma): ε={d_eps:.2f} | ρ={d_state.rho:.3f} [{d_bar:<15}] (Δ {d_state.rho - old_d_rho:+.3f}){RESET}")
        print(f"  {GREEN}The Gadfly    (Low Gamma):  ε={g_eps:.2f} | ρ={g_state.rho:.3f} [{g_bar:<15}] (Δ {g_state.rho - old_g_rho:+.3f}){RESET}")
        
        # Stop condition
        if d_state.rho > 0.95:
             print(f"\n{RED}{BOLD}CRITICAL RIGIDITY REACHED. DOGMATIST HAS LOCKED UP.{RESET}")
             break
        
        time.sleep(1.0)

if __name__ == "__main__":
    if sys.platform == 'win32': os.system('color')
    try:
        asyncio.run(simulate_socrates())
    except KeyboardInterrupt:
        print(f"\n{RED}Debate Terminated.{RESET}")
