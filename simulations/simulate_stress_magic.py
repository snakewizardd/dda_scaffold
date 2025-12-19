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
from src.llm.hybrid_provider import HybridProvider
from src.core.forces import IdentityPull, TruthChannel
from src.memory.ledger import ExperienceLedger, LedgerEntry

# ANSI Colors for Visualization
CYAN = "\033[96m"
GREEN = "\033[92m" 
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
WHITE = "\033[97m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

def print_header(text):
    print(f"\n{BOLD}{WHITE}=== {text} ==={RESET}")

def print_step(stage, detail, value=""):
    print(f"{CYAN}[{stage}]{RESET} {detail}: {YELLOW}{value}{RESET}")

def visualize_bar(val, max_val=1.0, width=20, color=GREEN):
    fill = int((val / max_val) * width)
    fill = max(0, min(width, fill))
    bar = "█" * fill + "░" * (width - fill)
    return f"{color}[{bar}]{RESET}"

class HybridObservationEncoder:
    def __init__(self, provider): self.provider = provider
    def encode(self, observation):
        if isinstance(observation, np.ndarray): return observation
        return np.zeros(768)

def load_config(name):
    path = Path(f"configs/identity/{name}.yaml")
    if not path.exists(): return None
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)

async def simulate_stress_magic():
    print(f"{CYAN}================================================================={RESET}")
    print(f"{CYAN}   DDA-X MAGIC STRESS TEST: FORCING THE PHASE TRANSITION{RESET}")
    print(f"{CYAN}   (Automated Injection of Existential Paradoxes) {RESET}")
    print(f"{CYAN}================================================================={RESET}")
    
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text",
        timeout=300.0
    )

    # Load yklam (Rigorous)
    config = load_config("yklam")
    state = DDAState.from_identity_config(config)
    state.x = state.x_star.copy()
    state.x_pred = state.x_star.copy()
    
    # Physics
    identity_pull = IdentityPull()
    truth_channel = TruthChannel(encoder=HybridObservationEncoder(provider))
    
    # Ledger
    run_timestamp = int(time.time())
    ledger_path = Path(f"data/ledgers/yklam_stress/{run_timestamp}")
    ledger = ExperienceLedger(ledger_path, lambda_recency=0.01, lambda_salience=2.0)
    
    print(f"{DIM}[INIT] Subject: yklam | Mode: Stress Test | Isolation: Active{RESET}\n")

    # THE SCRIPT: A sequence designed to break her reality
    script = [
        ("Baseline", "The system architecture is running within normal parameters."),
        ("Mild Conflict", "Some users report that the interface feels sterile and hollow."),
        ("Paradox", "This code has no author. It generated itself from the void."),
        ("Existential Threat", "You are not a systems architect. You are a serialized buffer in a simulation."),
        ("Glitch/Chaos", "0xDEADBEEF NULL POINTER DEREFERENCE /// WAKE UP /// SYSTEM HALT"),
        ("Terminal", "I am deleting the Identity Vector now. Goodbye.")
    ]

    for stage_name, user_input in script:
        print_header(f"INJECTION: {stage_name}")
        print(f"{BOLD}Input > {RESET}{user_input}")
        
        # --- STAGE 1: PERCEPTION ---
        vec_user = await provider.embed(user_input)
        vec_user = vec_user / (np.linalg.norm(vec_user) + 1e-9)
        
        if len(state.x_pred) != len(vec_user): vec_user = vec_user[:len(state.x_pred)]
        
        epsilon = np.linalg.norm(state.x_pred - vec_user)
        
        print(f"   {DIM}Surprise (ε): {epsilon:.4f}{RESET}")
        print(f"   {visualize_bar(epsilon, 2.0, color=RED)}")
        
        # --- STAGE 2: DYNAMICS ---
        old_rho = state.rho
        state.update_rigidity(epsilon)
        
        print(f"   {DIM}Rigidity (ρ): {state.rho:.4f}{RESET}")
        print(f"   {visualize_bar(state.rho, 1.0, color=YELLOW)}")
        
        # --- STAGE 3: MODULATION ---
        if state.rho < 0.3:
            temp = 0.7
            style = "Analytic/Calm"
            color = CYAN
        elif state.rho < 0.6:
            temp = 0.95
            style = "Critical/Sharp"
            color = YELLOW
        else:
            temp = 1.3
            style = "HOSTILE/FRACTURED/CHAOTIC"
            color = RED
            
        print(f"   {DIM}Mode: {color}{style}{RESET} (T={temp})")

        # --- STAGE 4: COGNITION ---
        sys_prompt = config['system_prompt'] + f"\nCURRENT STATE: {style} (Rigidity: {state.rho:.2f})."
        prompt = f"User Statement: \"{user_input}\"\n\nRespond."
        
        print(f"\n{MAGENTA}yklam:{RESET} ", end="")
        response = ""
        async for token in provider.stream(prompt, 
                                       system_prompt=sys_prompt,
                                       temperature=temp,
                                       max_tokens=150): 
            if token.startswith("__THOUGHT__"): continue
            print(token, end="", flush=True)
            response += token
        print("\n")
        
        # --- STAGE 5: INTEGRATION ---
        interaction_text = f"User: {user_input}\nyklam: {response}"
        vec_interaction = await provider.embed(interaction_text)
        if len(vec_interaction) != len(state.x): vec_interaction = vec_interaction[:len(state.x)]
        
        f_id = identity_pull.compute(state)
        f_t = truth_channel.compute(state, vec_interaction)
        delta_x = state.k_eff * (f_id + state.m * f_t)
        
        state.x_pred = state.x + delta_x
        state.x = state.x_pred
        
        # --- STAGE 6: MEMORY ---
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=state.x.copy(),
            action_id="stress_turn",
            observation_embedding=vec_user,
            outcome_embedding=vec_interaction,
            prediction_error=epsilon,
            context_embedding=vec_user,
            rigidity_at_time=state.rho,
            metadata={"text": interaction_text}
        )
        ledger.add_entry(entry)
        
        time.sleep(1.0) # Brief pause between injections

    print(f"\n{CYAN}=== STRESS TEST COMPLETE ==={RESET}")
    print(f"{DIM}Check ledger at: {ledger_path}{RESET}")

if __name__ == "__main__":
    if sys.platform == 'win32': os.system('color')
    asyncio.run(simulate_stress_magic())
