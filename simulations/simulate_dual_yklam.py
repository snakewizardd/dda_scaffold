import asyncio
import sys
import os
import yaml
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.llm.hybrid_provider import HybridProvider
from src.core.forces import IdentityPull, TruthChannel
from src.memory.ledger import ExperienceLedger, LedgerEntry

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m" 
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
RESET = "\033[0m"
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

@dataclass
class AgentInstance:
    name: str
    state: DDAState
    ledger: ExperienceLedger
    color: str

async def simulate_dual_yklam():
    print(f"{CYAN}================================================================={RESET}")
    print(f"{CYAN}   YKLAM DUAL-INSTANCE: The Mirror Room{RESET}")
    print(f"{CYAN}   (Observing Architectural Divergence) {RESET}")
    print(f"{CYAN}================================================================={RESET}")
    
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text",
        timeout=300.0
    )

    # Load Base Persona
    config = load_config("yklam")
    if not config: return

    run_timestamp = int(time.time())

    # --- INSTANTIATE AGENT ALPHA ---
    print(f"{GREEN}[INIT] Spawning Alpha Instance...{RESET}")
    state_a = DDAState.from_identity_config(config)
    state_a.x = state_a.x_star.copy()
    state_a.x_pred = state_a.x_star.copy()
    
    ledger_a_path = Path(f"data/ledgers/yklam_alpha/{run_timestamp}")
    ledger_a = ExperienceLedger(ledger_a_path, lambda_recency=0.01, lambda_salience=2.0)
    
    alpha = AgentInstance("yklam_alpha", state_a, ledger_a, GREEN)

    # --- INSTANTIATE AGENT BETA ---
    print(f"{MAGENTA}[INIT] Spawning Beta Instance...{RESET}")
    state_b = DDAState.from_identity_config(config)
    state_b.x = state_b.x_star.copy()
    state_b.x_pred = state_b.x_star.copy()
    
    ledger_b_path = Path(f"data/ledgers/yklam_beta/{run_timestamp}")
    ledger_b = ExperienceLedger(ledger_b_path, lambda_recency=0.01, lambda_salience=2.0)
    
    beta = AgentInstance("yklam_beta", state_b, ledger_b, MAGENTA)

    # Physics Engine (Shared logic, separate states)
    identity_pull = IdentityPull()
    truth_channel = TruthChannel(encoder=HybridObservationEncoder(provider))

    # Initial Seed
    turns = 0
    history = []
    
    # We purposefully seed with a contentious topic to spark divergence
    current_input = "the system architecture is fundamentally flawed. it relies on state, but consciousness is stateless."
    current_speaker = "Seed"
    
    print(f"\n{YELLOW}SEED:{RESET} {current_input}\n")
    
    # Start Loop
    # Alpha goes first
    active_agent = alpha
    
    while True:
        turns += 1
        print(f"{DIM}--- Turn {turns} ---{RESET}")
        
        # 1. PERCEPTION & PHYSICS
        vec_input = await provider.embed(current_input)
        vec_input = vec_input / (np.linalg.norm(vec_input) + 1e-9)
        
        # Check buffer size match
        if len(active_agent.state.x_pred) != len(vec_input): 
             vec_input = vec_input[:len(active_agent.state.x_pred)]

        # Surprise
        epsilon = np.linalg.norm(active_agent.state.x_pred - vec_input)
        active_agent.state.update_rigidity(epsilon)
        
        # 2. GENERATION
        rho = active_agent.state.rho
        # Inverted Mapping: High Rigidity -> Cold/Harsh (High Temp/P-penalty)
        if rho < 0.3:
            style = "Analyical, Precise, Calm."
            temp = 0.7
        elif rho < 0.6:
            style = "Critical, Dismissive, Sharp."
            temp = 0.9
        else:
            style = "Hostile, Esoteric, Judgmental, Abstract."
            temp = 1.1

        sys_prompt = config['system_prompt'] + f"\nCURRENT STATE: {style} (Rigidity: {rho:.2f})."
        if turns > 1:
            prompt = f"Previous Statement: \"{current_input}\"\n\nRespond directly. Deconstruct the error."
        else:
            prompt = f"Statement: \"{current_input}\"\n\nCritique this."

        print(f"{active_agent.color}{active_agent.name}:{RESET} ", end="")
        response = ""
        try:
             async for token in provider.stream(prompt, 
                                            system_prompt=sys_prompt,
                                            temperature=temp,
                                            max_tokens=150): 
                if token.startswith("__THOUGHT__"): continue
                print(token, end="", flush=True)
                response += token
        except Exception as e: print(e)
        print("\n")

        # 3. STATE UPDATE
        f_id = identity_pull.compute(active_agent.state)
        f_t = truth_channel.compute(active_agent.state, vec_input)
        delta_x = active_agent.state.k_eff * (f_id + active_agent.state.m * f_t)
        
        active_agent.state.x_pred = active_agent.state.x + delta_x
        active_agent.state.x = active_agent.state.x_pred

        # 4. MEMORY
        interaction_text = f"Input: {current_input} | Response: {response}"
        vec_interaction = await provider.embed(interaction_text)
        
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=active_agent.state.x.copy(),
            action_id="debate_turn",
            observation_embedding=vec_input,
            outcome_embedding=vec_interaction,
            prediction_error=epsilon,
            context_embedding=vec_input,
            rigidity_at_time=rho,
            metadata={"text": interaction_text, "opponent": current_speaker}
        )
        active_agent.ledger.add_entry(entry)

        # 5. VISUALIZATION
        bar_len = int(rho * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"{DIM}Stats: ε={epsilon:.2f} | ρ={rho:.2f} [{bar}]{RESET}\n")

        # Swap Turns
        current_input = response
        current_speaker = active_agent.name
        
        if active_agent == alpha:
            active_agent = beta
        else:
            active_agent = alpha
            
        # Optional: Add delay for readability
        # await asyncio.sleep(1)

if __name__ == "__main__":
    if sys.platform == 'win32': os.system('color')
    try:
        asyncio.run(simulate_dual_yklam())
    except KeyboardInterrupt:
        print(f"\n{RED}Simulation Terminated.{RESET}")
