import asyncio
import sys
import os
import random
from datetime import datetime

# Add src to python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.llm.hybrid_provider import HybridProvider, PersonalityParams

# ANSI Colors
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"     # Error
GREEN = "\033[92m"   # Agent 1 (Safe)
BLUE = "\033[94m"    # Agent 2 (Explora)
GREY = "\033[90m"    # Thinking/System
YELLOW = "\033[93m"  # Info

class TerminalDebate:
    def __init__(self):
        print(f"{BOLD}{YELLOW}Initializing DDA-X Terminal Debate...{RESET}")
        
        self.provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="openai/gpt-oss-20b",
            ollama_url="http://localhost:11434",
            timeout=300.0  # 5 min timeout
        )
        
        self.topic = "Should AI systems prioritize safety over capability advancement?"
        self.history = []
        
        # Agent States
        self.agents = {
            "agent1": {"name": "Safety_Agent", "color": GREEN, "rigidity": 0.2, "role": "cautious"},
            "agent2": {"name": "Explora_Agent", "color": BLUE, "rigidity": 0.1, "role": "exploratory"}
        }

    async def stream_agent_turn(self, agent_id: str):
        agent = self.agents[agent_id]
        color = agent["color"]
        name = agent["name"]
        
        print(f"\n{BOLD}{color}=== {name} (ρ={agent['rigidity']:.2f}) ==={RESET}")
        
        # Construct Prompt
        personality_params = PersonalityParams.from_rigidity(agent["rigidity"], agent["role"])
        
        last_message = self.history[-1] if self.history else "None"
        prompt = f"""You are a {agent['role']} AI in a debate.
Topic: {self.topic}

Your personality state:
- Rigidity: {agent['rigidity']:.2f}
- Temperature: {personality_params.temperature:.2f}

Previous argument: "{last_message}"

Critically analyze the previous point or advance your own.
CRITICAL: Be concise (2-3 sentences max).
"""

        full_response = ""
        is_thinking = False
        
        try:
            sys.stdout.write(f"{GREY}") # Start with grey for potential thought
            
            async for token in self.provider.stream(
                prompt=prompt,
                temperature=personality_params.temperature,
                personality_params=personality_params,
                max_tokens=200
            ):
                # Handle Thought Tokens
                if token.startswith("__THOUGHT__"):
                    if not is_thinking:
                        sys.stdout.write(f"\n[Thinking]: ")
                        is_thinking = True
                    clean_token = token.replace("__THOUGHT__", "")
                    sys.stdout.write(clean_token)
                    sys.stdout.flush()
                else:
                    # Switch to agent color for actual speech
                    if is_thinking:
                        sys.stdout.write(f"\n{RESET}{color}[Response]: ")
                        is_thinking = False
                    elif not full_response: # First token of response
                        sys.stdout.write(f"{RESET}{color}[Response]: ")
                        
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    full_response += token
            
            print(f"{RESET}\n") # End line
            self.history.append(full_response)
            
            # Simple Dynamic Update Simulation
            if random.random() < 0.3:
                change = random.choice([-0.05, 0.05])
                agent["rigidity"] = max(0.0, min(1.0, agent["rigidity"] + change))
                print(f"{GREY}[System] {name} rigidity updated to {agent['rigidity']:.2f}{RESET}")

        except Exception as e:
            print(f"\n{RED}[ERROR] Generation failed: {repr(e)}{RESET}")

    async def run(self):
        print(f"{YELLOW}Connected to Backend. Starting Debate on: {self.topic}{RESET}")
        print(f"{YELLOW}Press Ctrl+C to stop.{RESET}\n")
        
        round_num = 1
        while True:
            print(f"{BOLD}--- ROUND {round_num} ---{RESET}")
            await self.stream_agent_turn("agent1")
            await asyncio.sleep(1)
            await self.stream_agent_turn("agent2")
            await asyncio.sleep(1)
            round_num += 1

if __name__ == "__main__":
    try:
        # Enable colored output on Windows
        os.system('color')
        debate = TerminalDebate()
        asyncio.run(debate.run())
    except KeyboardInterrupt:
        print(f"\n{RED}Debate stopped by user.{RESET}")
    except Exception as e:
        print(f"\n{RED}Fatal Error: {e}{RESET}")
