import asyncio
import numpy as np
import yaml
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List
import time
import json

# Framework
from src.core.state import DDAState
from src.llm.hybrid_provider import HybridProvider, PersonalityParams
from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.game.connect4 import Connect4

# Colors
RESET = "\033[0m"
DIM = "\033[2m"
RED_TXT = "\033[91m"
YELLOW_TXT = "\033[93m" 
CYAN = "\033[96m"

class DuelistAgent:
    """
    True Agentic Duelist.
    Uses LLM Chain-of-Thought + Memory + DDA Rigidity.
    Leverages LM Studio Structured Output (JSON Schema).
    """
    def __init__(
        self, 
        name: str, 
        color: str,
        config_path: str,
        provider: HybridProvider,
        mock: bool = False
    ):
        self.name = name
        self.color = color
        self.provider = provider
        self.mock = mock
        
        # Load Config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        # DDA State (Manual Management)
        self.state = DDAState(
            x=np.zeros(768),
            x_star=np.zeros(768),
            rho=0.1, # Start flexible
            epsilon_0=self.config.get("epsilon_0", 0.6),
            alpha=self.config.get("alpha", 0.1)
        )
        self.last_epsilon = 0.0
        
        # Memory Ledger
        ledger_path = Path(f"data/memory/{self.name.replace(' ', '_').lower()}")
        self.ledger = ExperienceLedger(storage_path=ledger_path)
        
        self.initial_identity_vec = np.zeros(768)

    async def prepare_battle(self):
        """Init identity embedding."""
        if self.mock:
            vec = np.zeros(768)
        else:
            sys_prompt = self.config.get("system_prompt", "I am a duelist.")
            vec = await self.provider.embed(sys_prompt[:500])
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            
        self.initial_identity_vec = vec
        self.state.x = vec.copy()
        self.state.x_star = vec.copy()

    async def decide_move(self, game: Connect4) -> Tuple[int, str]:
        """
        Agentic Decision Loop using Structured Output.
        """
        # 1. Context Retrieval
        memories = []
        if not self.mock:
            board_embedding = await self._embed_text(game.to_string(labeled=True))
            refs = self.ledger.retrieve_reflections(board_embedding, k=2)
            memories = [r.reflection_text for r in refs]
        
        context_str = ""
        if memories:
            context_str = "\nRELEVANT MEMORIES:\n" + "\n".join([f"- {m}" for m in memories])

        valid_moves = game.get_valid_moves()
        valid_moves_1based = [c+1 for c in valid_moves]

        prompt = f"""
You are playing Connect 4.
Name: {self.name} ({self.color})
Personality: {self.config.get("system_prompt", "Aggressive")}
Current Rigidity (ρ): {self.state.rho:.2f} (0.0=Adaptive, 1.0=Stubborn/Paranoid).

BOARD STATE:
{game.to_string(labeled=True)}

Position List: {game.to_list_repr()}

{context_str}

TASK:
1. THINK: Analyze the board for threats (3-in-a-row) and opportunities.
2. DECIDE: Pick a column ({min(valid_moves_1based)}-{max(valid_moves_1based)}).
3. BANTER: Generate a trash-talk line.
"""
        # JSON Schema for LM Studio
        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "connect4_move",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "thought_process": {"type": "string"},
                        "move_column": {"type": "integer", "description": "Column number 1-7"},
                        "banter": {"type": "string"}
                    },
                    "required": ["thought_process", "move_column", "banter"]
                }
            }
        }

        params = PersonalityParams.from_rigidity(self.state.rho)
        if self.mock:
            return (valid_moves[0], "Mock move.")
            
        try:
            # Pass schema to provider
            response = await self.provider.complete(
                prompt, 
                personality_params=params, 
                max_tokens=1024, 
                response_format=response_schema
            )
            
            # Guaranteed JSON if supported by model/engine
            data = json.loads(response)
            
            thought_process = data.get("thought_process", "...")
            col_1 = int(data.get("move_column", 1))
            banter = data.get("banter", "...")
            
            print(f"\n{DIM}[Self-Reflection] {thought_process}{RESET}")

            # Validation
            col_0 = col_1 - 1
            if col_0 not in valid_moves:
                print(f"{RED_TXT}Invalid Move Detected ({col_1}). Correcting to random valid.{RESET}")
                col_0 = valid_moves[0]
                banter += " (System corrected invalid move)"
            
            # DDA Physics
            epsilon = np.random.beta(2, 5)
            self.last_epsilon = epsilon
            self.state.update_rigidity(epsilon)
            
            # Save Memory
            if not self.mock:
                obs_vec = await self._embed_text(game.to_string())
                ref = ReflectionEntry(
                    timestamp=time.time(),
                    task_intent="Win Duel",
                    situation_embedding=obs_vec,
                    reflection_text=thought_process,
                    prediction_error=epsilon,
                    outcome_success=False 
                )
                self.ledger.add_reflection(ref)
            
            return col_0, banter
                
        except Exception as e:
            print(f"{RED_TXT}Error in decide_move: {e}{RESET}")
            # Fallback
            return (valid_moves[0], f"Error: {e}")

    async def observe_result(self, game: Connect4):
        pass

    async def _embed_text(self, text: str) -> np.ndarray:
        try:
            v = await self.provider.embed(text)
            return v / (np.linalg.norm(v) + 1e-9)
        except:
             return np.zeros(768)

    def get_hud_stats(self) -> str:
        """Returns colored string of stats."""
        rho = self.state.rho
        filled = int(rho * 20)
        filled = max(0, min(20, filled))
        bar = "█" * filled + "░" * (20 - filled)
        
        drift = np.linalg.norm(self.state.x - self.initial_identity_vec)
        drift_color = f"{RED_TXT}" if drift > 0.5 else f"{CYAN}"
        
        color_code = RED_TXT if self.color == "RED" else YELLOW_TXT
        
        return (
            f"{color_code}{self.name}{RESET} | "
            f"ρ: {bar} {rho:.2f} | "
            f"ε: {self.last_epsilon:.2f} | "
            f"{drift_color}Drift: {drift:.2f}{RESET}"
        )
