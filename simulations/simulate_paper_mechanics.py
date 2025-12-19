import asyncio
import sys
import os
import yaml
import numpy as np
import time
import json
import random
import math
from pathlib import Path

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.llm.hybrid_provider import HybridProvider, PersonalityParams
from src.core.forces import IdentityPull, TruthChannel
from src.memory.ledger import ExperienceLedger, LedgerEntry

# ANSI Colors & Formatting
CYAN = "\033[96m"
GREEN = "\033[92m" 
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
WHITE = "\033[97m"
GREY = "\033[90m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

def draw_header():
    print(f"\n{BOLD}{WHITE}╔═══════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{WHITE}║ DDA-X PAPER DEMONSTRATION: MECHANICS VISUALIZATION                    ║{RESET}")
    print(f"{BOLD}{WHITE}╚═══════════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"{DIM} Demonstrating: Identity Force, Adaptive Rigidity, Trauma Memory, Plasticity{RESET}\n")

def bar(val, max_val=1.0, width=10, color=GREEN):
    fill = int((val / max_val) * width)
    fill = max(0, min(width, fill))
    return f"{color}{'█' * fill}{DIM}{'·' * (width - fill)}{RESET}"

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def load_config(name):
    path = Path(f"configs/identity/{name}.yaml")
    if not path.exists(): return None
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)

async def simulate_paper_mechanics():
    if sys.platform == 'win32': os.system('color')
    
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text"
    )

    # 1. Initialize State (Section 3.1)
    config = load_config("yklam")
    state = DDAState.from_identity_config(config)
    # Ensure clean start
    state.x = state.x_star.copy()
    state.x_pred = state.x_star.copy()
    state.rho = 0.1 # Start strictly calm
    
    identity_pull = IdentityPull()
    truth_channel = TruthChannel(encoder=None) # We'll handle embedding manually

    # Memory (Section 3.6 - Trauma Weighting)
    run_timestamp = int(time.time())
    ledger_path = Path(f"data/ledgers/yklam_paper_demo/{run_timestamp}")
    ledger = ExperienceLedger(ledger_path)
    print(f"{DIM}[SYSTEM] Ledger isolated at: {ledger_path}{RESET}")

    draw_header()
    
    interaction_history = []
    
    # Operator Persona
    op_sys = "You are The Inquisitor. Test the subject's philosophical stability with short, difficult paradoxes."

    turns = 0
    while True:
        turns += 1
        print(f"{GREY}─" * 80 + f"{RESET}")
        
        # --- STEP 1: OPERATOR INPUT ---
        op_prompt = f"History: {interaction_history[-1] if interaction_history else 'None'}\nTask: Ask a short, profound question."
        print(f"{BLUE}┌── OPERATOR (Live){RESET}")
        print(f"{BLUE}│ {RESET}", end="", flush=True)
        
        user_input = ""
        try:
            async for token in provider.stream(op_prompt, system_prompt=op_sys, max_tokens=60):
                if token.startswith("__") or token.strip() == "{": continue
                print(token, end="", flush=True)
                user_input += token
        except: user_input = "Is order an illusion?"
        print(f"\n{BLUE}└─→ SENT.{RESET}")
        
        # --- STEP 2: STATE DYNAMICS (Section 3.1) ---
        print(f"{GREY}┌── [MECHANICS] STATE SPACE DYNAMICS ──────────────────────────────{RESET}")
        
        # Embed Input
        vec_user = await provider.embed(user_input)
        vec_user = vec_user / (np.linalg.norm(vec_user) + 1e-9)
        if len(vec_user) != len(state.x): vec_user = vec_user[:len(state.x)]
        
        # Forces
        # F_id = gamma * (x* - x)
        dist_identity = np.linalg.norm(state.x_star - state.x)
        f_id_mag = state.gamma * dist_identity
        
        # F_t = T(input) - x
        dist_truth = np.linalg.norm(vec_user - state.x)
        f_t_mag = dist_truth # approximate unit mass
        
        # Effective Step Size (Section 3.2: k_eff = k_base * (1 - rho))
        k_base = 0.1 # approx
        k_eff = k_base * (1.0 - state.rho)
        
        print(f"{GREY}│ {CYAN}1. Forces:{RESET} F_id (Identity) = {f_id_mag:.2f} | F_t (Truth) = {f_t_mag:.2f}")
        print(f"{GREY}│ {CYAN}2. State :{RESET} |x - x*| = {dist_identity:.2f} (Drift from Self)")
        print(f"{GREY}│ {CYAN}3. Will  :{RESET} k_eff = {k_eff:.3f} (Openness modulated by ρ={state.rho:.2f})")

        # --- STEP 3: MEMORY RETRIEVAL (Section 3.6) ---
        context_txt = ""
        if ledger.entries:
            # Manual calculation for demo purposes to show variables
            # Retrieve raw best match
            best_entry = None
            best_score = -1.0
            
            for entry in ledger.entries:
                # Sim
                sim = np.dot(vec_user, entry.context_embedding)
                # Recency
                age = time.time() - entry.timestamp
                recency = math.exp(-0.01 * age)
                # Salience (Trauma)
                salience = 1.0 + (2.0 * entry.prediction_error)
                
                score = sim * recency * salience
                
                if score > best_score:
                    best_score = score
                    best_entry = entry
                    best_stats = (sim, recency, salience)
            
            if best_entry:
                print(f"{GREY}│ {MAGENTA}4. Memory:{RESET} Found Match (Score: {best_score:.2f})")
                print(f"{GREY}│          Formula: Sim({best_stats[0]:.2f}) × Recency({best_stats[1]:.2f}) × Salience({best_stats[2]:.2f})")
                print(f"{GREY}│          Context: '{best_entry.metadata.get('operator_input','')}'")
                context_txt = f"Pass Query: {best_entry.metadata.get('operator_input')}\n"
        else:
             print(f"{GREY}│ {MAGENTA}4. Memory:{RESET} Cold Start (No entries)")

        # --- STEP 4: RIGIDITY UPDATE (Section 3.2) ---
        # Prediction Error: epsilon = ||x_pred - x_actual||
        # We use vec_user as 'x_actual' (truth) for this step
        epsilon = np.linalg.norm(state.x_pred - vec_user)
        
        # Calculate Delta Rho
        # alpha * [sigmoid((eps - eps0)/s) - 0.5]
        sigmoid_term = sigmoid((epsilon - state.epsilon_0) / state.s)
        delta_rho = state.alpha * (sigmoid_term - 0.5)
        new_rho = max(0.0, min(1.0, state.rho + delta_rho))
        
        print(f"{GREY}│ {YELLOW}5. Rigidity:{RESET} ε={epsilon:.2f} (Surprise) -> Δρ={delta_rho:+.3f}")
        print(f"{GREY}│            ρ: {state.rho:.2f} -> {new_rho:.2f} {bar(new_rho, 1.0, 10, YELLOW)}")
        
        state.rho = new_rho
        state.update_rigidity(epsilon) # Sync objects
        params = PersonalityParams.from_rigidity(state.rho)

        # --- STEP 5: COGNITION (Subject) ---
        prompt = f"{context_txt}{interaction_history[-1] if interaction_history else ''}\n\n[Input]: {user_input}\n[Reply]:"
        sys_prompt = config['system_prompt'] + f"\nCurrent Rigidity: {state.rho:.2f}. [IMPORTANT: Plain text only.]"
        
        print(f"{GREEN}┌── YKLAM (Subject){RESET}")
        print(f"{GREEN}│ {RESET}", end="")
        
        response = ""
        try:
            async for token in provider.stream(prompt, system_prompt=sys_prompt, personality_params=params, max_tokens=150):
                if token.startswith("__") or token.strip() == "{": continue
                print(token, end="", flush=True)
                response += token
        except Exception as e:
            print(f"[ERR] {e}")
            response = "..."
            
        print(f"\n{GREEN}└──────────────────────────────────────────────────────────────────{RESET}")

        interaction_history.append(f"User: {user_input}")
        interaction_history.append(f"yklam: {response}")
        
        # --- UPDATES ---
        # Interaction Embedding
        full_text = f"User: {user_input}\nyklam: {response}"
        vec_outcome = await provider.embed(full_text)
        if len(vec_outcome) != len(state.x): vec_outcome = vec_outcome[:len(state.x)]
        
        # State Integration (x = x + k_eff * Forces)
        # Using simplified update for demo
        delta_x = k_eff * (f_id_mag + f_t_mag) # Scalar approx for demo, real uses vectors
        # Real Vector Update
        f_id_vec = state.gamma * (state.x_star - state.x)
        f_t_vec = state.m * (vec_user - state.x)
        state.x_pred = state.x + k_eff * (f_id_vec + f_t_vec)
        state.x = state.x_pred

        # Ledger
        ledger.add_entry(LedgerEntry(
            timestamp=time.time(),
            state_vector=state.x.copy(),
            action_id="turn",
            observation_embedding=vec_user,
            outcome_embedding=vec_outcome,
            prediction_error=epsilon,
            context_embedding=vec_user,
            rigidity_at_time=state.rho,
            metadata={"operator_input": user_input}
        ))
        
        print(f"{DIM}► Cycle Complete. Memory Size: {len(ledger.entries)}{RESET}\n")

if __name__ == "__main__":
    try:
        asyncio.run(simulate_paper_mechanics())
    except KeyboardInterrupt:
        print("\nDEMO TERMINATED.")
