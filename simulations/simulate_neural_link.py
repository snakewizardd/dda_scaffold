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

    # Operator Personality: The Digital Inquisitor
    operator_sys_prompt = "You are THE OPERATOR. You are a disembodied intelligence testing a digital subject. You speak in riddles, paradoxes, and high-concept queries. Your goal is to break the subject's logic or force an epiphany. Be brief. Be cryptic. Be void."  
    
    while True:
        turns += 1
        cycle_start = time.time()
        
        # --- PHASE 1: THE OPERATOR (LLM) ---
        print(f"{BLUE}┌── OPERATOR {DIM}(LIVE){RESET}")
        print(f"{BLUE}│ {RESET}", end="", flush=True)
        
        op_prompt = f"Subject Response: {interaction_history[-1] if interaction_history else 'None'}\n\nTask: Generate a single, short, profound question to test the subject's reality."
        
        user_input = ""
        try:
            # Live Operator Stream with Filter
            async for token in provider.stream(
                op_prompt,
                system_prompt=operator_sys_prompt,
                temperature=0.9, # High creativity
                max_tokens=60
            ):
                if token.startswith("__THOUGHT__"): continue
                if token.strip() == "{" or '"analysis":' in token: continue
                
                print(token, end="", flush=True)
                user_input += token
        except Exception as e:
            user_input = "System Failure. Reboot."
            print(user_input)
            
        print(f"\n{BLUE}└─→ SENT.{RESET}")
        
        # Pacing
        await asyncio.sleep(0.2) 

        # --- PHASE 2: SUBJECT PERCEPTION & PHYSICS ---
        print(f"{DIM}┌── [GLASS BOX] INTERNAL STATE MONITOR ─────────────────────{RESET}")
        
        # 2a. Embedding
        vec_user = await provider.embed(user_input)
        vec_user = vec_user / (np.linalg.norm(vec_user) + 1e-9)
        if len(state.x_pred) != len(vec_user): vec_user = vec_user[:len(state.x_pred)]
        print(f"{DIM}│ {CYAN}[INPUT]{RESET} Embedding generated. |v|={np.linalg.norm(vec_user):.2f}")

        # 2b. RAG / Memory Retrieval (New Feature)
        # Verify ledger has entries before retrieval
        context_str = ""
        if len(ledger.entries) > 0:
            relevant = ledger.retrieve(vec_user, k=1)
            if relevant:
                best_match = relevant[0]
                # Calculate similarity manually if not provided, or just show it exists
                sim_score = np.dot(vec_user, best_match.context_embedding)
                print(f"{DIM}│ {MAGENTA}[MEMORY]{RESET} RAG Retrieval: Found similar context (Sim: {sim_score:.2f})")
                print(f"{DIM}│          Ref: '{best_match.metadata.get('operator_input', 'Unknown')[:30]}...'{RESET}")
                context_str = f"Relevant Past Input: {best_match.metadata.get('operator_input')}\n"
        else:
            print(f"{DIM}│ {MAGENTA}[MEMORY]{RESET} Cold Start. No history.{RESET}")

        # 2c. DDA Physics (Forces)
        epsilon = np.linalg.norm(state.x_pred - vec_user)
        state.update_rigidity(epsilon)
        
        f_id = identity_pull.compute(state)
        f_t = truth_channel.compute(state, vec_user) # Using vec_user as proxy for truth/reality signal here
        mag_id = np.linalg.norm(f_id)
        mag_t = np.linalg.norm(f_t)
        
        # Params Calculation
        params = PersonalityParams.from_rigidity(state.rho)
        
        # Visualization: Physics
        print(f"{DIM}│ {YELLOW}[DYNAMICS]{RESET} ε:{epsilon:.2f} (Surprise) -> ρ:{state.rho:.2f} (Rigidity)")
        print(f"{DIM}│            Forces: F_id:{mag_id:.2f} vs F_t:{mag_t:.2f} | T:{params.temperature:.2f}{RESET}")

        # --- VISUALIZATION: TELEMETRY BAR (COMPACT) ---
        rho_bar = visualize_bar_compact(state.rho, 1.0, 10, YELLOW)
        eps_bar = visualize_bar_compact(epsilon, 2.0, 10, RED)
        print(f"    {DIM}LINK:{RESET} ε:{eps_bar} {epsilon:.2f} | ρ:{rho_bar} {state.rho:.2f} | T:{params.temperature:.2f}")


        # Phase 3: COGNITION
        prompt = f"{context_str}{interaction_history[-1] if interaction_history else ''}\n\n[Input]: {user_input}\n[Reply]:"
        
        print(f"{GREEN}┌── YKLAM {DIM}(Subject){RESET}")
        print(f"{GREEN}│ {RESET}", end="")
        
        # Dynamic System Prompt
        sys_prompt = config['system_prompt'] + f"\nCurrent Rigidity: {state.rho:.2f}. [IMPORTANT: Output plain text only. No JSON.]"
        
        response = ""
        try:
            async for token in provider.stream(
                 prompt,
                 system_prompt=sys_prompt,
                 personality_params=params,
                 max_tokens=150
            ):
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
        
        # Force Integration (Already computed forces above, but strictly should happen after outcome?)
        # ExACT architecture says State Update happens based on Outcome usually, or Perception. 
        # We'll stick to the existing flow where Perception drives State Update.
        
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
