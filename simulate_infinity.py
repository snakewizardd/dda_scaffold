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

async def generate_troll_input(provider, chat_history):
    """Generates the 'User' side of the conversation (The Antagonist)."""
    
    # Analyze the Discordian's last message to generate a relevant counter-troll
    last_msg = chat_history[-1] if chat_history else "start the flame war"
    
    prompt = f"""Task: You are 'NormieSlayer69', an annoying internet troll / skeptic. 
    Your goal is to trigger the person you are talking to ('discordian_88').
    
    Their last message: "{last_msg}"
    
    Instructions:
    - Disagree with them.
    - Call their theories "fake" or "cringe".
    - Be short (1 sentence).
    - Use "mainstream" logic to annoy them.
    - Ask provocative questions.
    
    Your response:"""
    
    response = await provider.complete(prompt, max_tokens=60, temperature=0.9)
    return response.strip().lower()

async def simulate_infinity():
    print(f"{CYAN}================================================================={RESET}")
    print(f"{CYAN}   DDA-X INFINITY LOOP: 'THE FLAME WAR'{RESET}")
    print(f"{CYAN}   (Auto-Generating Simulations | Ad Infinitum) {RESET}")
    print(f"{CYAN}================================================================={RESET}")
    
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text",
        timeout=300.0
    )

    # Load Polymath Persona
    config = load_config("polymath")
    if not config: 
        print("Config not found.")
        return

    # Semantic Identity Calibration
    # Semantic Identity Calibration
    print(f"{DIM}[INIT] Embedding Architect Persona...{RESET}")
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
    turn = 0

    print(f"\n{YELLOW}System: Starting SUPERHUMAN Dialectic Loop... (Ctrl+C to stop){RESET}")
    print(f"{DIM}Protagonist: The Architect (Polymath Agent){RESET}")
    print(f"{DIM}Antagonist:  SkepticBot (Auto-Debater){RESET}\n")

    input("Press Enter to Ignite...")

    while True:
        turn += 1
        print(f"\n{BLUE}--- TURN {turn} ---{RESET}")
        
        # 1. AUTO-DEBATER GENERATION (Reality)
        print(f"{DIM}Generating Skeptic Input...{RESET}")
        
        last_msg = chat_history[-1] if chat_history else "start the debate"
        
        # DYNAMIC STANCE: 30% chance to partially agree (Validation), 70% conflict
        import random
        is_validation = random.random() < 0.3
        
        if is_validation:
            stance = "Partially agree with their logic, but refine it slightly. Be constructive."
            print(f"{CYAN}Skeptic (VALIDATING)...{RESET}")
        else:
            stance = "Challenge this claim aggressively. Be skeptical."
            print(f"{RED}Skeptic (ATTACKING)...{RESET}")

        prompt = f"Task: You are a philosopher on Discord. {stance} Content: '{last_msg}'. Be extremely concise (Max 2 sentences). No essays."
        
        # Increase max_tokens to 256 to stop truncation
        troll_input = (await provider.complete(prompt, max_tokens=256, temperature=0.8)).strip()
        print(f"{GREEN}Skeptic:{RESET} {troll_input}")
        
        chat_history.append(f"User: {troll_input}")
        
        # 2. PHYSICS (Surprise)
        vec_actual = await provider.embed(troll_input)
        vec_actual = vec_actual / (np.linalg.norm(vec_actual) + 1e-9)
        
        if len(state.x_pred) != len(vec_actual): vec_actual = vec_actual[:len(state.x_pred)]
        epsilon = np.linalg.norm(state.x_pred - vec_actual)
        old_rho = state.rho
        state.update_rigidity(epsilon)
        delta = state.rho - old_rho
        
        bar_len = int(state.rho * 20)
        rigidity_bar = "█" * bar_len + "░" * (20 - bar_len)
        color = RED if state.rho > 0.6 else (YELLOW if state.rho > 0.3 else CYAN)
        
        print(f"{DIM}Internal State:{RESET}")
        print(f"  Surprise (ε): {epsilon:.2f}")
        print(f"  Rigidity (ρ): {color}{state.rho:.3f} [{rigidity_bar}]{RESET} (Δ {delta:+.3f})")
        
        # 3. FORCE COMPUTATION (The Soul)
        f_id = identity_pull.compute(state)
        f_t = truth_channel.compute(state, vec_actual)
        # Note: These are already computed for the update, just visualizing magnitudes
        print(f"  {BOLD}Force Dynamics (The Soul):{RESET}")
        print(f"    ||F_id|| (Identity Pull): {np.linalg.norm(f_id):.3f} (Holding frame)")
        print(f"    ||F_t || (Reality Pull):  {np.linalg.norm(f_t):.3f} (Adapting to input)")

        # 3. POLYMATH DISCOURSE
        # USE THE NEW PERSONALITY TYPE
        params = PersonalityParams.from_rigidity(state.rho, "polymath") 
        
        short_history = chat_history[-4:] 
        prompt = f"Chat History:\n{chr(10).join(short_history)}\n\nThe Architect (Rigid={state.rho:.2f}):"
        
        print(f"{MAGENTA}The Architect:{RESET} ", end="")
        response = ""
        try:
             async for token in provider.stream(prompt, 
                                            system_prompt=config['system_prompt'] + f"\nCurrent Rigidity: {state.rho:.2f}. Be precise.",
                                            personality_params=params, 
                                            max_tokens=256): 
                if token.startswith("__THOUGHT__"): continue
                print(token, end="", flush=True)
                response += token
        except Exception as e:
            print(f"Error: {e}")
            
        print()
        chat_history.append(f"The Architect: {response}")
        
        # 4. PREDICTION UPDATE
        f_id = identity_pull.compute(state)
        f_t = truth_channel.compute(state, vec_actual)
        delta_x = state.k_eff * (f_id + state.m * f_t)
        state.x_pred = state.x + delta_x
        state.x = state.x_pred
        
        # Pace the simulation
        time.sleep(2.0)

if __name__ == "__main__":
    if sys.platform == 'win32': os.system('color')
    try:
        asyncio.run(simulate_infinity())
    except KeyboardInterrupt:
        print(f"\n{RED}Simulation Terminated.{RESET}")
