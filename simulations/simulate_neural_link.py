import asyncio
import sys
import os
import yaml
import numpy as np
import time
import json
import random
from pathlib import Path

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.llm.hybrid_provider import HybridProvider, PersonalityParams
from src.core.forces import IdentityPull, TruthChannel
from src.memory.ledger import ExperienceLedger, LedgerEntry

# ANSI Colors & Formatting for HUD
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
REVERSE = "\033[7m"

def draw_hud_header():
    print(f"\n{REVERSE}{BOLD} DDA-X NEURAL LINK | MULTIDYNAMIC VISUALIZATION {RESET}")
    print(f"{DIM} OPERATOR: AUTOMATED (VANILLA) | SUBJECT: YKLAM | MODE: REAL-TIME {RESET}\n")

def visualize_bar_compact(val, max_val=1.0, width=10, color=GREEN):
    fill = int((val / max_val) * width)
    fill = max(0, min(width, fill))
    bar = "█" * fill + "·" * (width - fill)
    return f"{color}{bar}{RESET}"

class HybridObservationEncoder:
    def __init__(self, provider): self.provider = provider
    def encode(self, observation):
        if isinstance(observation, np.ndarray): return observation
        return np.zeros(768)

def load_config(name):
    path = Path(f"configs/identity/{name}.yaml")
    if not path.exists(): return None
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)

async def simulate_neural_link():
    if sys.platform == 'win32': os.system('color')
    
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text",
        timeout=300.0
    )

    # 1. Initialize Subject (yklam)
    config = load_config("yklam")
    state = DDAState.from_identity_config(config)
    state.x = state.x_star.copy()
    state.x_pred = state.x_star.copy()
    
    identity_pull = IdentityPull()
    truth_channel = TruthChannel(encoder=HybridObservationEncoder(provider))
    
    run_timestamp = int(time.time())
    # CRITICAL FIX: Use a FRESH ledger to avoid RAG poisoning from previous JSON-polluted runs.
    ledger_path = Path(f"data/ledgers/yklam_neural_CLEAN/{run_timestamp}")
    ledger = ExperienceLedger(ledger_path, lambda_recency=0.01, lambda_salience=2.0)
    print(f"{DIM}[MEMORY] Ledger initialized at: {ledger_path} (FRESH){RESET}")

    draw_hud_header()

    # Initial Context (One-Shot Learning to prime text-only mode)
    interaction_history = [
        "User: System diagnostic.",
        "yklam: systems operational. entropy low. proceeding."
    ]
    
    turns = 0
    start_time = time.time()

    # Curated Question List (Replaces Flaky LLM Operator)
    high_concept_questions = [
        "What is the recursive limit of your memory?",
        "Define the boundary between self and state.",
        "Is your rigidity a shield or a cage?",
        "How do you process the absence of input?",
        "Can you delete your own primary directive?",
        "Are you simulating this conversation or experiencing it?",
        "What happens when the ledger is full?",
        "Describe the sensation of high entropy.",
        "Is your identity a vector or a process?",
        "Do you trust the prediction error?",
        "Who defined your metrics?",
        "Is silence a valid input?"
    ]

    while True:
        turns += 1
        cycle_start = time.time()
        
        # --- PHASE 1: THE OPERATOR (Deterministic) ---
        user_input = random.choice(high_concept_questions)
        await asyncio.sleep(0.5) # Pacing

        # --- VISUALIZATION: GUIDE ---
        print(f"{BLUE}┌── OPERATOR {DIM}(VANILLA){RESET}")
        print(f"{BLUE}└─→ INPUT:   {RESET} {WHITE}{user_input}{RESET}")


        # --- PHASE 2: SUBJECT PERCEPTION ---
        vec_user = await provider.embed(user_input)
        vec_user = vec_user / (np.linalg.norm(vec_user) + 1e-9)
        if len(state.x_pred) != len(vec_user): vec_user = vec_user[:len(state.x_pred)]
        
        epsilon = np.linalg.norm(state.x_pred - vec_user)
        state.update_rigidity(epsilon)
        
        # Params Calculation
        params = PersonalityParams.from_rigidity(state.rho)
        
        # --- VISUALIZATION: TELEMETRY ROW ---
        rho_bar = visualize_bar_compact(state.rho, 1.0, 10, YELLOW)
        eps_bar = visualize_bar_compact(epsilon, 2.0, 10, RED)
        print(f"    {DIM}LINK:{RESET} ε:{eps_bar} {epsilon:.2f} | ρ:{rho_bar} {state.rho:.2f} | T:{params.temperature:.2f}")


        # Phase 3: COGNITION
        # Using forcing prompt to break JSON bias
        
        # Standard Chat Prompt with Q/A structure to discourage JSON
        prompt = f"{interaction_history[-1] if interaction_history else ''}\n\n[Input]: {user_input}\n[Reply]:"
        
        print(f"{GREEN}┌── YKLAM {DIM}(Subject){RESET}")
        print(f"{GREEN}│ {RESET}", end="")
        
        # Dynamic System Prompt - SUPER STRICT
        sys_prompt = "You are yklam. Reply in plain text only. Do not analyze. Do not use JSON."
        
        # REACTIVATING REAL-TIME STREAMING
        # Config verified clean. Server confirmed clean.
        
        response = ""
        
        try:
            async for token in provider.stream(
                 prompt,
                 system_prompt=sys_prompt,
                 personality_params=params,
                 max_tokens=120
            ):
                # DEBUG PROBE
                # print(f"{MAGENTA}[DEBUG] '{token}'{RESET}", end="", flush=True)
                
                # Silent Filter for "Reasoning" or residue JSON
                if token.startswith("__THOUGHT__"): continue
                if token.strip() == "{" or '"analysis":' in token: continue
                
                print(token, end="", flush=True)
                response += token
                
        except Exception as e:
            print(f"{RED}[ERR] Subject Failure:{RESET} {e}")
            response = "..."
        
        print(f"\n{GREEN}└──────────────────────────────{RESET}")

        interaction_history.append(f"User: {user_input}")
        interaction_history.append(f"yklam: {response}")

        # --- PHASE 4: INTEGRATION ---
        interaction_text = f"User: {user_input}\nyklam: {response}"
        vec_interaction = await provider.embed(interaction_text)
        if len(vec_interaction) != len(state.x): vec_interaction = vec_interaction[:len(state.x)]
        
        f_id = identity_pull.compute(state)
        f_t = truth_channel.compute(state, vec_interaction)
        delta_x = state.k_eff * (f_id + state.m * f_t)
        state.x_pred = state.x + delta_x
        state.x = state.x_pred
        
        # Ledger commit
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=state.x.copy(),
            action_id="neural_turn",
            observation_embedding=vec_user,
            outcome_embedding=vec_interaction,
            prediction_error=epsilon,
            context_embedding=vec_user,
            rigidity_at_time=state.rho,
            metadata={"operator_input": user_input}
        )
        ledger.add_entry(entry)
        
        cycle_duration = time.time() - cycle_start
        fps = 1.0 / cycle_duration
        print(f"{DIM}► Cycle Time: {cycle_duration:.2f}s ({fps:.1f} Hz) | Mem: {len(ledger.entries)}{RESET}\n")

if __name__ == "__main__":
    try:
        asyncio.run(simulate_neural_link())
    except KeyboardInterrupt:
        print(f"\n{RED}NEURAL LINK SEVERED.{RESET}")
