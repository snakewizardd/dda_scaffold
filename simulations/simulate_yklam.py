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

async def simulate_yklam():
    print(f"{CYAN}================================================================={RESET}")
    print(f"{CYAN}   YKLAM: THE SOULFUL PROXY{RESET}")
    print(f"{CYAN}   (DDA-X Interactive Persona) {RESET}")
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
    print(f"{DIM}[INIT] Awakening yklam...{RESET}")
    
    state = DDAState.from_identity_config(config)
    state.x = state.x_star.copy()
    state.x_pred = state.x_star.copy()
    
    # Physics Engine
    identity_pull = IdentityPull()
    truth_channel = TruthChannel(encoder=HybridObservationEncoder(provider))
    # No scorer needed for pure chat reflection in this simple loop
    reflection_channel = ReflectionChannel(scorer=None) 
    
    # Initialize Memory Ledger (Unique Run Isolation)
    run_id = int(time.time())
    ledger_path = Path(f"data/ledgers/yklam_manual/{run_id}")
    ledger = ExperienceLedger(ledger_path, lambda_recency=0.005, lambda_salience=2.0)
    print(f"{DIM}[MEMORY] Ledger initialized at: {ledger_path}{RESET}")
    print(f"{DIM}[MEMORY] Previous memories ignored. Run is contained.{RESET}")

    # Conversation State (Short-term buffer)
    chat_history = []
    
    # Initialize Prompt Session with custom style
    style = Style.from_dict({
        'prompt': '#0000ff bold',  # Blue
    })
    session = PromptSession(style=style)

    print(f"\n{GREEN}yklam is online.{RESET}")
    print(f"{DIM}(Type 'exit' to quit. Multiline paste supported.){RESET}\n")

    while True:
        try:
            user_input = await session.prompt_async("You: ", multiline=False)
            if user_input.strip().lower() in ["exit", "quit"]:
                break
            
            if not user_input.strip():
                continue
            
            # 1. Embed User Input for RAG
            vec_user = await provider.embed(user_input)
            
            # 2. Retrieve Relevant Memories
            memories = ledger.retrieve(vec_user, k=3, min_score=0.15)
            memory_text = ""
            if memories:
                memory_text = "\n[RELEVANT MEMORIES]:\n" + "\n".join([f"- {m.metadata.get('text', 'unknown')}" for m in memories]) + "\n"
            
            chat_history.append(f"User: {user_input}")
            
            # yklam Thinking Process
            # We override "polymath" behavior. High Rigidity (trigger) should mean HIGH ENERGY JUDGMENT, not silence.
            # So we FORCE high temperature/creativity when she's triggered (High Rho).
            
            # Base params
            temp = 0.9
            if state.rho > 0.5:
                # TRIGGERED MODE: High Temp (Chaos/Anger) + High Presence Penalty (New words/Rants)
                temp = 1.1 
            
            # Context window management
            # Short-term history + Long-term memory injection
            history_text = "\n".join(chat_history[-6:]) # Keep short-term buffer tight
            prompt = f"{memory_text}Conversation:\n{history_text}\n\nyklam:"
            
            print(f"{MAGENTA}yklam:{RESET} ", end="")
            response = ""
            
            # Stream response - giving her a bit more room than "Discord style" but still concise.
            async for token in provider.stream(
                prompt, 
                system_prompt=config['system_prompt'] + f"\nCurrent Rigidity: {state.rho:.2f}. Be casual, lowercase, honest. No metaphors. Max 2 sentences.",
                temperature=temp,
                max_tokens=150
            ): 
                if token.startswith("__THOUGHT__"): continue
                print(token, end="", flush=True)
                response += token
            
            print() # Newline
            chat_history.append(f"yklam: {response}")
            
            # DDA Dynamics Update
            # We treat the USER's input as the "Truth" or "Reality" she encounters
            interaction_text = f"User: {user_input}\nyklam: {response}"
            vec_interaction = await provider.embed(interaction_text)
            vec_interaction = vec_interaction / (np.linalg.norm(vec_interaction) + 1e-9)
            
            if len(state.x_pred) != len(vec_interaction): 
                vec_interaction = vec_interaction[:len(state.x_pred)]
                
            epsilon = np.linalg.norm(state.x_pred - vec_interaction)
            
            old_rho = state.rho
            state.update_rigidity(epsilon)
            delta = state.rho - old_rho
            
            # VISUALIZATION: Show the "Soul State" / Transparency
            bar_len = int(state.rho * 20)
            rigidity_bar = "█" * bar_len + "░" * (20 - bar_len)
            
            # Coherence / Trust Metric (Drift from Identity)
            drift = np.linalg.norm(state.x - state.x_star)
            coherence = max(0, 1.0 - drift) # Normalize roughly
            coherence_len = int(coherence * 20)
            coherence_bar = "▓" * coherence_len + "░" * (20 - coherence_len)
            
            # Color coding for rigidity/mood
            mood_color = CYAN  # Relaxed/Open
            if state.rho > 0.4: mood_color = YELLOW # Guarded
            if state.rho > 0.7: mood_color = RED    # Closed/Defensive
            
            # Surprise indicator
            surprise_len = int(min(epsilon, 1.0) * 10)
            surprise_bar = "!" * surprise_len
            
            print(f"\n{DIM}--- Soul Telemetry ---{RESET}")
            print(f"{DIM}Sensitivity (ε): {epsilon:.3f} {surprise_bar}{RESET}")
            print(f"{DIM}Guardedness (ρ): {mood_color}{state.rho:.3f} [{rigidity_bar}]{RESET} (Δ {delta:+.3f})")
            print(f"{DIM}Coherence   (C): {GREEN}{coherence:.3f} [{coherence_bar}]{RESET} (Trust/Stability)")
            
            if epsilon > 0.2:
                print(f"{YELLOW}* She felt a disturbance *{RESET}")
            elif coherence < 0.5:
                print(f"{MAGENTA}* She is drifting from herself *{RESET}")
                
            print(f"{DIM}----------------------{RESET}\n")

            # PHYSICS UPDATE: The "Stateful" part.
            f_id = identity_pull.compute(state)
            f_t = truth_channel.compute(state, vec_interaction)
            
            delta_x = state.k_eff * (f_id + state.m * f_t)
            state.x_pred = state.x + delta_x
            state.x = state.x_pred
            
            # MEMORY STORAGE
            entry = LedgerEntry(
                timestamp=time.time(),
                state_vector=state.x.copy(),
                action_id="chat",
                observation_embedding=vec_user,
                outcome_embedding=vec_interaction,
                prediction_error=epsilon,
                context_embedding=vec_user, # Key on USER input for retrieval
                rigidity_at_time=state.rho,
                metadata={"text": f"User: {user_input} | yklam: {response}"}
            )
            ledger.add_entry(entry)

        except KeyboardInterrupt:
            print(f"\n{RED}Disconnected.{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")
            

if __name__ == "__main__":
    if sys.platform == 'win32': os.system('color')
    asyncio.run(simulate_yklam())
