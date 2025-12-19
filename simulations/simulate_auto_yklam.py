import asyncio
import sys
import os
import yaml
import numpy as np
import time
from pathlib import Path
# import aioconsole  # Removed to avoid dependency issues

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.llm.hybrid_provider import HybridProvider, PersonalityParams
from src.core.forces import ForceAggregator, IdentityPull, TruthChannel, ReflectionChannel
from src.memory.ledger import ExperienceLedger, LedgerEntry

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

async def simulate_auto_yklam():
    print(f"{CYAN}================================================================={RESET}")
    print(f"{CYAN}   YKLAM: THE NATURAL SIMULATION (AUTO + INJECTION){RESET}")
    print(f"{CYAN}   (Physics: Inverted Volatility + Variable Plasticity) {RESET}")
    print(f"{CYAN}================================================================={RESET}")
    
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text",
        timeout=300.0
    )

    # Load yklam Persona
    config_name = "yklam"
    config = load_config(config_name)
    if not config: 
        print(f"Config not found for {config_name}.")
        return

    # Semantic Identity Calibration
    print(f"{DIM}[INIT] Awakening yklam (Natural Mode)...{RESET}")
    state = DDAState.from_identity_config(config)
    # Initialize state vectors explicitly
    state.x = state.x_star.copy()
    state.x_pred = state.x_star.copy()
    
    # Physics Engine
    identity_pull = IdentityPull()
    truth_channel = TruthChannel(encoder=HybridObservationEncoder(provider))
    
    # Physics Engine
    identity_pull = IdentityPull()
    truth_channel = TruthChannel(encoder=HybridObservationEncoder(provider))
    
    # Initialize Memory Ledger (Unique to Auto-Sim Run)
    run_id = int(time.time())
    ledger_path = Path(f"data/ledgers/yklam_auto/{run_id}")
    ledger = ExperienceLedger(ledger_path, lambda_recency=0.01, lambda_salience=2.0)
    print(f"{DIM}[MEMORY] Ledger initialized at: {ledger_path}{RESET}")
    print(f"{DIM}[MEMORY] Previous memories ignored. Run is contained.{RESET}")
    
    # Init Normie Persona (The "Auto" part - generates content for her to react to)
    normie_history = []
    
    print(f"\n{GREEN}yklam is online.{RESET}")
    print(f"{YELLOW}System: Auto-Sim Running. Type ANYTHING to Inject/Derail (Ctrl+C to stop).{RESET}")
    print(f"{DIM}Waiting for turn...{RESET}\n")

    chat_history = []
    turn = 0
    
    # Initial Prompt to kickstart
    last_injection = "Start the convo."
    
    while True:
        turn += 1
        print(f"\n{BLUE}--- TURN {turn} ---{RESET}")
        
        # 1. USER INJECTION OPPORTUNITY
        # We use a non-blocking check if possible, or a timed wait. 
        # Since standard input() is blocking, we'll simulate "Auto" by having her react to a Normie Bot
        # UNLESS the user proactively engaged in the previous step (simulated by a check here).
        # For this script, we will use a simple "Press Enter to Auto-Generate or Type to Inject :" approach
        
        print(f"{BOLD}INPUT:{RESET} (Enter=Auto-Normie / Type=Inject) > ", end="")
        injection = await asyncio.to_thread(sys.stdin.readline)
        injection = injection.strip()
        
        is_injection = len(injection) > 0
        
        if is_injection:
            if injection.lower() in ["exit", "quit"]: break
            current_input = injection
            user_label = "User (Inject)"
            print(f"{RED}[INJECTION DETECTED]{RESET}")
        else:
            # AUTO GENERATE NORMIE INPUT
            user_label = "NormieBot"
            # Generate something "cringe" or "normal" based on history
            prompt = f"Chat History:\n{chr(10).join(chat_history[-4:])}\n\nTask: You are a normal, boring internet user. Reply to yklam. Be polite but basic. Max 1 sentence."
            current_input = (await provider.complete(prompt, max_tokens=60, temperature=0.7)).strip()
            print(f"{CYAN}NormieBot:{RESET} {current_input}")

        chat_history.append(f"{user_label}: {current_input}")

        # 2. PHYSICS (Surprise & Plasticity)
        vec_input = await provider.embed(current_input)
        vec_input = vec_input / (np.linalg.norm(vec_input) + 1e-9)
        
        if len(state.x_pred) != len(vec_input): vec_input = vec_input[:len(state.x_pred)]
        
        # SURPRISE (DDA)
        epsilon = np.linalg.norm(state.x_pred - vec_input)
        
        # SENSITIVITY CHECK
        # If epsilon is LOW (she agrees/understand), she adopts the vibe (Plasticity Increases)
        # If epsilon is HIGH (she hates it), she hardens (Plasticity Decreases)
        
        k_plasticity_base = 0.1
        if epsilon < config['epsilon_0']:
             # Trust Mechanism: "You get me." -> High Plasticity
             k_plasticity = 0.3 
             mood = "VIBING"
        else:
             # Defense Mechanism: "Normie Detected." -> Low Plasticity
             k_plasticity = 0.05
             mood = "JUDGING"

        state.update_rigidity(epsilon)
        
        # 3. GENERATION with INVERTED MAPPING
        # Low Rigidity = Normal/Funny
        # High Rigidity = CHAOS/Anger
        
        rho = state.rho
        if rho < 0.4:
            temp = 0.8
            style = "chill, funny, casual"
        elif rho < 0.7:
            temp = 0.95 
            style = "annoyed, short, dismissive"
        else:
            temp = 1.2
            style = "CHAOTIC, ALL CAPS FRAGMENTS, RANTING, TRIGGERED"

        print(f"{DIM}Internal: ε={epsilon:.2f} ({mood}) | ρ={rho:.2f} | Temp={temp}{RESET}")
        
        # Update System Prompt with Mood
        sys_prompt = config['system_prompt'] + f"\nCURRENT VIBE: {style}. Rigidity: {rho:.2f}."
        
        prompt = f"Chat History:\n{chr(10).join(chat_history[-6:])}\n\nyklam:"
        
        print(f"{MAGENTA}yklam:{RESET} ", end="")
        response = ""
        try:
             async for token in provider.stream(prompt, 
                                            system_prompt=sys_prompt,
                                            temperature=temp,
                                            max_tokens=200): 
                if token.startswith("__THOUGHT__"): continue
                print(token, end="", flush=True)
                response += token
        except Exception as e: print(e)
        print()
        
        chat_history.append(f"yklam: {response}")
        
        # 4. STATE UPDATE (Prediction)
        f_id = identity_pull.compute(state)
        f_t = truth_channel.compute(state, vec_input)
        
        # Apply Variable Plasticity
        # delta_x = k_plasticity * (F_id + m * F_truth)
        delta_x = k_plasticity * (f_id + state.m * f_t)
        
        state.x_pred = state.x + delta_x
        state.x = state.x_pred
        
        # MEMORY STORAGE (Full Architecture)
        interaction_text = f"{user_label}: {current_input} | yklam: {response}"
        vec_interaction = await provider.embed(interaction_text)
        
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=state.x.copy(),
            action_id="chat_turn",
            observation_embedding=vec_input, # The input that caused the reaction
            outcome_embedding=vec_interaction, # The full interaction context as outcome
            prediction_error=epsilon,
            context_embedding=vec_input, # Retrieve based on input
            rigidity_at_time=state.rho,
            metadata={"text": interaction_text, "vibe": style}
        )
        ledger.add_entry(entry)

        # Visual
        bar_len = int(rho * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"{DIM}Soul State: [{bar}]{RESET}")

if __name__ == "__main__":
    if sys.platform == 'win32': os.system('color')
    try:
        asyncio.run(simulate_auto_yklam())
    except KeyboardInterrupt:
        print(f"\n{RED}Disconnected.{RESET}")
