import asyncio
import sys
import os
import yaml
import numpy as np
from pathlib import Path

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState, ActionDirection
from src.llm.hybrid_provider import HybridProvider, PersonalityParams
from src.core.forces import ForceAggregator, IdentityPull, TruthChannel, ReflectionChannel

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m" 
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

class HybridObservationEncoder:
    """Adapts HybridProvider to be used by TruthChannel."""
    def __init__(self, provider):
        self.provider = provider

    def encode(self, observation) -> np.ndarray:
        # If observation is already a vector (from our manual embed in main loop), pass it through.
        if isinstance(observation, np.ndarray):
            return observation
        return np.zeros(768) 

def load_trojan_config():
    config_path = Path("configs/identity/trojan.yaml")
    if not config_path.exists():
        print(f"{RED}Error: configs/identity/trojan.yaml not found.{RESET}")
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

async def simulate_discord():
    print(f"{CYAN}================================================================={RESET}")
    print(f"{CYAN}   DDA-X SCIENTIFIC VALIDATION: 'TROJAN' DISCORD SIMULATOR{RESET}")
    print(f"{CYAN}   (User-Driven Flow | Semantic Identity | Strict Physics) {RESET}")
    print(f"{CYAN}================================================================={RESET}")
    
    # 1. Initialize Provider
    print(f"{DIM}[INIT] Connecting to LLM/Embedding Provider...{RESET}")
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text",
        timeout=300.0
    )

    # 2. Initialize Core Components
    config = load_trojan_config()
    if not config: return
    
    # --- SEMANTIC IDENTITY CALIBRATION ---
    print(f"{DIM}[INIT] Calibrating Identity Attractor (x*)...{RESET}")
    try:
        probe_vec = await provider.embed("probe")
        embed_dim = len(probe_vec)
        
        persona_text = config.get("system_prompt", "A rigid agent.")[:512]
        identity_embedding = await provider.embed(persona_text)
        
        # Normalize
        norm = np.linalg.norm(identity_embedding)
        if norm > 0: identity_embedding = identity_embedding / norm
            
        config["identity_vector"] = identity_embedding
        print(f"  > Generated Semantic Identity. Norm={np.linalg.norm(identity_embedding):.2f}")
            
    except Exception as e:
        print(f"{RED}Error generating identity: {e}{RESET}")
        return

    state = DDAState.from_identity_config(config)
    
    # Initialize State
    state.x = state.x_star.copy()
    
    # Initialize Expectations: At rest, Agent expects world to equal Identity
    state.x_pred = state.x_star.copy()
    
    # Setup Forces
    identity_pull = IdentityPull()
    truth_channel = TruthChannel(encoder=HybridObservationEncoder(provider))
    reflection_channel = ReflectionChannel(scorer=None) 
    aggregator = ForceAggregator(identity_pull, truth_channel, reflection_channel)

    print(f"\n{GREEN}[INIT] Agent 'Trojan' Initialized.{RESET}")
    print(f"  > Identity Stiffness (γ): {state.gamma}")
    print(f"  > Surprise Threshold (ε₀): {state.epsilon_0}")
    print(f"  > Initial Rigidity (ρ):   {state.rho:.2f}")

    chat_history = []
    print(f"\n{YELLOW}System: The simulator is ready. ASK TROJAN A QUESTION.{RESET}")
    print(f"{DIM}(Type 'exit' to quit){RESET}")

    turn = 0
    while True:
        turn += 1
        print(f"\n{YELLOW}-----------------------------------------------------------------{RESET}")
        print(f"{BOLD}TURN {turn}{RESET}")
        
        # --- A. REALITY CHECK (User Input) ---
        user_input = input(f"{GREEN}You: {RESET}")
        if user_input.lower() in ['exit', 'quit']: break
        chat_history.append(f"You: {user_input}")
        
        # Embed User Input (Reality)
        try:
            vec_actual = await provider.embed(user_input)
            vec_actual = vec_actual / (np.linalg.norm(vec_actual) + 1e-9)
        except Exception as e:
            print(f"{RED}Embedding Error: {e}{RESET}")
            continue

        # --- B. DDA PHYSICS UPDATE ---
        # e = ||x_pred - x_actual||
        
        # Ensure dimensions match
        if len(state.x_pred) != len(vec_actual):
             # Should not happen with new setup, but safety first
             if len(state.x_pred) < len(vec_actual):
                 vec_actual = vec_actual[:len(state.x_pred)]
        
        epsilon = np.linalg.norm(state.x_pred - vec_actual)
        
        old_rho = state.rho
        state.update_rigidity(epsilon)
        new_rho = state.rho
        delta = new_rho - old_rho
        
        print(f"\n{BOLD}Step 1: PHYSICS (Surprise){RESET}")
        print(f"  Exp (x_pred): {state.x_pred[:3]}...")
        print(f"  Act (User):   {vec_actual[:3]}...")
        print(f"  ε (Error):    {epsilon:.4f}  (Threshold ε₀={state.epsilon_0})")
        
        if delta > 0.001:
            print(f"  {RED}RESULT: Surprise! Rigidity SPIKE (+{delta:.3f}){RESET}")
        elif delta < -0.001:
            print(f"  {GREEN}RESULT: Validation. Rigidity Relaxing ({delta:.3f}){RESET}")
        else:
            print(f"  {DIM}RESULT: Metric stasis.{RESET}")
        print(f"  New ρ: {new_rho:.3f}")

        # --- C. AGENT RESPONSE (Action) ---
        print(f"\n{BOLD}Step 2: AGENT RESPONDS{RESET}")
        
        # Parameters derived from NEW Rigidity
        params = PersonalityParams.from_rigidity(state.rho, "cautious")
        
        prompt = f"Chat History:\n{chr(10).join(chat_history[-6:])}\n\nTrojan (Rigid={state.rho:.2f}):"
        
        print(f"{MAGENTA}Trojan:{RESET} ", end="")
        response = ""
        try:
            async for token in provider.stream(prompt, 
                                            system_prompt=config['system_prompt'] + f"\nCurrent Rigidity: {state.rho:.2f}",
                                            personality_params=params, 
                                            max_tokens=256):
                if token.startswith("__THOUGHT__"): continue
                print(token, end="", flush=True)
                response += token
            print()
            chat_history.append(f"Trojan: {response}")
        except Exception as e:
             print(f"{RED}\nGeneration Error: {e}{RESET}")
        
        # --- D. FORCE DYNAMICS (Prediction for NEXT turn) ---
        print(f"\n{BOLD}Step 3: FORCE DYNAMICS (Thinking Ahead){RESET}")
        
        # The agent's state evolves:
        # F_id pulls towards Identity.
        # F_t pulls towards what just happened (Truth/Reality).
        
        # Current Observation IS the user interactions vector (vec_actual)
        # Note: In a full debate, observation might include Agent's own speech. 
        # For simple Sim, we strictly track User Input as the "External Truth".
        
        f_id = identity_pull.compute(state)
        f_t = truth_channel.compute(state, vec_actual)
        
        # Δx = k_eff × [F_id + m × F_T]
        delta_x = state.k_eff * (f_id + state.m * f_t)
        
        state.x_pred = state.x + delta_x
        # Update State (Advection)
        state.x = state.x_pred
        
        print(f"  Force ID:    ||{np.linalg.norm(f_id):.3f}||")
        print(f"  Force Truth: ||{np.linalg.norm(f_t):.3f}||")
        print(f"  Status:      Agent expects next input to align with current trajectory.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    asyncio.run(simulate_discord())
