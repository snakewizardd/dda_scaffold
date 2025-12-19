import asyncio
import argparse
import sys
import os
import random
import time

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.game.connect4 import Connect4
from src.agents.duelist import DuelistAgent
from src.llm.hybrid_provider import HybridProvider

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

async def main():
    parser = argparse.ArgumentParser(description="DDA-X Connect 4 Duel")
    parser.add_argument("--mock", action="store_true", help="Run in offline mock mode (no LLM)")
    parser.add_argument("--speed", choices=["slow", "normal", "fast"], default="normal", help="Animation speed")
    parser.add_argument("--render", choices=["ansi", "curses"], default="ansi", help="Renderer type")
    
    args = parser.parse_args()
    
    if args.render == "curses":
        print("Curses renderer not implemented yet. Falling back to ANSI.")
        
    print(f"{CYAN}================================================================={RESET}")
    print(f"{CYAN}   CONNECT 4 DUEL: GRANDMASTER EDITION (MCTS + MEMORY){RESET}")
    print(f"{CYAN}   (High-IQ Agents with DDA-X Framework){RESET}")
    print(f"{CYAN}================================================================={RESET}")
    
    # 1. Initialize Infrastructure
    provider = HybridProvider(
        lm_studio_url="http://127.0.0.1:1234",
        lm_studio_model="openai/gpt-oss-20b",
        ollama_url="http://localhost:11434",
        embed_model="nomic-embed-text"
    )
    
    # Check connection if not mock
    if not args.mock:
        print(f"{DIM}[INIT] Checking LLM Uplink...{RESET}")
        status = provider.check_connection()
        if not status["lm_studio"]:
            print(f"{YELLOW}[WARN] LM Studio not found. Ensure server is running at port 1234.{RESET}")
        if not status["ollama"] and not args.mock:
             print(f"{YELLOW}[WARN] Ollama not found. Embeddings will fail.{RESET}")
             
    # 2. Initialize Agents
    # Using the new refactored DDAX-based DuelistAgent
    red_agent = DuelistAgent(
        name="Aggressor Red", 
        color="RED", 
        config_path="configs/identity/aggressor_red.yaml", 
        provider=provider,
        mock=args.mock
    )
    
    yellow_agent = DuelistAgent(
        name="Aggressor Yellow", 
        color="YELLOW", 
        config_path="configs/identity/aggressor_yellow.yaml", 
        provider=provider,
        mock=args.mock
    )
    
    # Async Init
    await red_agent.prepare_battle()
    await yellow_agent.prepare_battle()
    
    # 3. Game Loop
    game = Connect4()
    
    print(f"\n{GREEN}BATTLE STARTING IN 3 SECONDS...{RESET}")
    time.sleep(3)
    
    while not game.winner and not game.check_draw():
        # Determine current player
        current_agent = red_agent if game.turn == Connect4.RED else yellow_agent
        opponent_agent = yellow_agent if game.turn == Connect4.RED else red_agent
        player_color_name = "RED" if game.turn == Connect4.RED else "YELLOW"
        
        # HUD Construction
        hud = (
            f"{CYAN}-----------------------------------------------------------------\n"
            f"{red_agent.get_hud_stats()}\n"
            f"{yellow_agent.get_hud_stats()}\n"
            f"{CYAN}-----------------------------------------------------------------{RESET}"
        )
        
        # Render Initial State
        game.render_ascii(hud_data=hud, speed=args.speed)
        
        print(f"\n{BOLD}{player_color_name} TURN ({current_agent.name}){RESET}")
        print(f"{DIM}Thinking (Agentic Chain-of-Thought)...{RESET}")
        
        # DECISION TIME (Now invokes MCTS)
        # Note: we pass a COPY of the game to avoid side effects during search if not handled carefully
        # But our MCTS uses logic, not state mutation of the observed object directly usually
        # To be safe, let's pass the game. DDAX decider treats it as 'observation'.
        
        start_time = time.time()
        col_idx, banter = await current_agent.decide_move(game)
        think_time = time.time() - start_time
        
        # Execute Move
        try:
            row, col = game.drop_token(col_idx, game.turn)
            
            # Print Banter (after move decided)
            print(f"\n{BOLD}{current_agent.name} (thought for {think_time:.1f}s):{RESET} \"{banter}\"")
            time.sleep(1.0 if args.speed != "fast" else 0.2)
            
            # Animate Drop
            game.render_ascii(animate_drop=(row, col), speed=args.speed, hud_data=hud)
            
            # OBSERVE OUTCOME (Memory Update)
            # Both agents observe the board change? Or just the actor?
            # Usually the actor observes the result of their action.
            await current_agent.observe_result(game)
            if not args.mock:
                 # Opponent also observes? In DDA, yes, reality changed.
                 # But sticking to active player for now to save API calls
                 pass
            
            # Switch Turn
            game.turn = Connect4.YELLOW if game.turn == Connect4.RED else Connect4.RED
            
        except ValueError as e:
            print(f"{RED}CRITICAL ERROR: Agent attempted invalid move {col_idx}. {e}{RESET}")
            # MCTS should theoretically prevent this if implemented right
            print(f"{YELLOW}System Override: Forcing random valid move.{RESET}")
            valid = game.get_valid_moves()
            if valid:
                 # Logic for random fallback
                 r, c = game.drop_token(random.choice(valid), game.turn)
                 game.render_ascii(animate_drop=(r, c), speed=args.speed, hud_data=hud)
                 game.turn = Connect4.YELLOW if game.turn == Connect4.RED else Connect4.RED
            else:
                break 
        
        time.sleep(0.5 if args.speed != "fast" else 0.1)

    # End Game
    game.render_ascii(hud_data=hud, speed=args.speed)
    print(f"\n{CYAN}================================================================={RESET}")
    if game.winner:
        winner_name = red_agent.name if game.winner == Connect4.RED else yellow_agent.name
        print(f"{GREEN}{BOLD}GAME OVER! WINNER: {winner_name}{RESET}")
        # Notify agents of end task?
    else:
        print(f"{YELLOW}{BOLD}GAME OVER! DRAW!{RESET}")
    print(f"{CYAN}================================================================={RESET}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
