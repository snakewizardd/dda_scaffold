import asyncio
import sys
import os
import yaml
import numpy as np
import time
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style

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

async def simulate_glass_box():
    print(f"{CYAN}================================================================={RESET}")
    print(f"{CYAN}   DDA-X GLASS BOX: THE MRI FOR THE DIGITAL SOUL{RESET}")
    print(f"{CYAN}   (Visualizing Cognitive Mechanics in Real-Time) {RESET}")
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
    ledger_path = Path(f"data/ledgers/yklam_glassbox/{run_timestamp}")
    ledger = ExperienceLedger(ledger_path, lambda_recency=0.01, lambda_salience=2.0)
    
    print(f"{DIM}[INIT] System Online. Identity Loaded. Ledger Active.{RESET}\n")

    session = PromptSession()

    while True:
        try:
            print(f"{BOLD}User Input > {RESET}", end="")
            user_input = await session.prompt_async()
            if user_input.lower() in ["exit", "quit"]: break
            
            # --- STAGE 1: PERCEPTION ---
            print_header("STAGE 1: PERCEPTION (Embedding & Surprise)")
            print(f"{DIM}Mapping language to semantic vector space...{RESET}")
            
            vec_user = await provider.embed(user_input)
            vec_user = vec_user / (np.linalg.norm(vec_user) + 1e-9)
            
            if len(state.x_pred) != len(vec_user): vec_user = vec_user[:len(state.x_pred)]
            
            # Calculate Surprise (Prediction Error)
            epsilon = np.linalg.norm(state.x_pred - vec_user)
            
            print_step("Input", "Vector Magnitude", f"{np.linalg.norm(vec_user):.4f}")
            print_step("Expectation", "State Prediction (x_pred)", "Ready")
            print_step("SURPRISE (ε)", "||x_pred - Input||", f"{epsilon:.4f}")
            print(f"   {visualize_bar(epsilon, 2.0, color=RED)} (Surprise Metric)")
            
            time.sleep(1.0) # Pause for effect

            # --- STAGE 2: DYNAMICS ---
            print_header("STAGE 2: DYNAMICS (Rigidity Update)")
            print(f"{DIM}Adjusting cognitive flexibility based on surprise...{RESET}")
            
            old_rho = state.rho
            state.update_rigidity(epsilon)
            delta_rho = state.rho - old_rho
            
            print_step("Previous ρ", "Rigidity", f"{old_rho:.4f}")
            print_step("Update Δρ", "f(ε)", f"{delta_rho:+.4f}")
            print_step("Current ρ", "Rigidity", f"{state.rho:.4f}")
            print(f"   {visualize_bar(state.rho, 1.0, color=YELLOW)} (Guardedness)")
            
            if state.rho > 0.5:
                print(f"   {RED}WARNING: HIGH RIGIDITY DETECTED. DEFENSIVE MODE.{RESET}")
            
            time.sleep(1.0)

            # --- STAGE 3: MODULATION ---
            print_header("STAGE 3: MODULATION (LLM Parameters)")
            print(f"{DIM}Mapping internal state to generator settings...{RESET}")
            
            # Inverted Mapping Logic
            if state.rho < 0.3:
                temp = 0.7
                style = "Analytic/Calm"
                print(f"   Mode: {CYAN}Lucid{RESET} (Low Entropy)")
            elif state.rho < 0.6:
                temp = 0.95
                style = "Critical/Sharp"
                print(f"   Mode: {YELLOW}Alert{RESET} (Medium Entropy)")
            else:
                temp = 1.2
                style = "Hostile/Abstract"
                print(f"   Mode: {RED}Volatile{RESET} (High Entropy)")
                
            print_step("Temperature", "T(ρ)", f"{temp:.2f}")
            print_step("System Prompt", "Injection", f"Vibe: {style}")
            
            time.sleep(1.0)

            # --- STAGE 4: COGNITION ---
            print_header("STAGE 4: COGNITION (Action Generation)")
            
            sys_prompt = config['system_prompt'] + f"\nCURRENT STATE: {style} (Rigidity: {state.rho:.2f})."
            prompt = f"User Statement: \"{user_input}\"\n\nRespond."
            
            print(f"{MAGENTA}yklam (Thinking...):{RESET} ", end="")
            response = ""
            async for token in provider.stream(prompt, 
                                           system_prompt=sys_prompt,
                                           temperature=temp,
                                           max_tokens=150): 
                if token.startswith("__THOUGHT__"): continue
                print(token, end="", flush=True)
                response += token
                # time.sleep(0.02) # Typewriter effect
            print("\n")
            
            time.sleep(0.5)

            # --- STAGE 5: INTEGRATION ---
            print_header("STAGE 5: INTEGRATION (State Evolution)")
            print(f"{DIM}Calculating Forces and Plasticity...{RESET}")
            
            # Embed Interaction for Truth Force
            interaction_text = f"User: {user_input}\nyklam: {response}"
            vec_interaction = await provider.embed(interaction_text)
            if len(vec_interaction) != len(state.x): vec_interaction = vec_interaction[:len(state.x)]
            
            f_id = identity_pull.compute(state)
            f_t = truth_channel.compute(state, vec_interaction)
            
            # Force Magnitudes
            mag_id = np.linalg.norm(f_id)
            mag_t = np.linalg.norm(f_t)
            
            print_step("Identity Force", "F_id (Pull to Self)", f"{mag_id:.4f}")
            print_step("Truth Force", "F_t (Pull to Reality)", f"{mag_t:.4f}")
            
            delta_x = state.k_eff * (f_id + state.m * f_t)
            shift_mag = np.linalg.norm(delta_x)
            
            print_step("Plasticity", "k_eff", f"{state.k_eff:.4f}")
            print_step("State Shift", "||Δx||", f"{shift_mag:.4f}")
            
            state.x_pred = state.x + delta_x
            state.x = state.x_pred
            
            time.sleep(1.0)

            # --- STAGE 6: MEMORY ---
            print_header("STAGE 6: MEMORY (Ledger Commit)")
            
            entry = LedgerEntry(
                timestamp=time.time(),
                state_vector=state.x.copy(),
                action_id="chat",
                observation_embedding=vec_user,
                outcome_embedding=vec_interaction,
                prediction_error=epsilon,
                context_embedding=vec_user,
                rigidity_at_time=state.rho,
                metadata={"text": interaction_text}
            )
            ledger.add_entry(entry)
            
            print(f"{GREEN}✓ Experience Committed to Ledger{RESET}")
            print(f"{DIM}Path: {ledger_path}{RESET}\n")
            print(f"{CYAN}--- Cycle Complete ---{RESET}\n")

        except KeyboardInterrupt:
            print(f"\n{RED}Disconnected.{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")

if __name__ == "__main__":
    if sys.platform == 'win32': os.system('color')
    asyncio.run(simulate_glass_box())
