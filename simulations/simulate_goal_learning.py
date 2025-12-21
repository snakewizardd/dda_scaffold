#!/usr/bin/env python3
"""
Goal-Directed Learning Demonstration
=====================================

The CORRECT way to use DDA-X: with an actual GOAL.

Previous experiments were broken because they had no goal.
Without a goal:
- No meaningful predictions
- No real surprise
- No learning

This experiment has:
1. A CLEAR GOAL: Solve a puzzle in minimum steps
2. ACTIONS: The agent chooses moves
3. OUTCOMES: The environment responds with results
4. PREDICTION ERROR: Agent predicted X, got Y
5. LEARNING: Rigidity adapts, strategy changes, memory accumulates

The puzzle: A simple number-guessing game where the agent must
discover the hidden rule. The agent gets feedback after each guess.
The DDA dynamics should help it:
- Explore broadly when uncertain (low ρ)
- Exploit discoveries when confident (high ρ)
- Remember what worked (ledger)
- Adapt strategy based on surprise

The goal is to solve the puzzle in as few steps as possible.
"""

import asyncio
import sys
import os
import numpy as np
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.llm.hybrid_provider import HybridProvider


class C:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


@dataclass
class GameState:
    """The puzzle environment."""
    target_rule: str  # Description of hidden rule
    target_number: int  # The number to find
    history: List[Tuple[int, str]]  # (guess, feedback)
    attempts: int = 0
    solved: bool = False


class NumberPuzzle:
    """
    A goal-directed puzzle environment.
    
    The agent must discover a hidden number through feedback.
    Each guess gets feedback: "too low", "too high", "correct", 
    or more complex feedback based on the rule.
    """
    
    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self.state: Optional[GameState] = None
        self.setup_puzzle()
    
    def setup_puzzle(self):
        """Create a new puzzle."""
        if self.difficulty == "easy":
            # Simple: find a number 1-100
            target = random.randint(1, 100)
            rule = f"Find the hidden number between 1 and 100."
        elif self.difficulty == "medium":
            # Pattern: number is related to a formula
            base = random.randint(1, 10)
            target = base * base + base  # n² + n pattern
            rule = f"Find the hidden number (hint: it follows a quadratic pattern). Range: 1-500."
        else:
            # Hard: number encodes something
            digits = [random.randint(1, 9) for _ in range(3)]
            target = digits[0] * 100 + digits[1] * 10 + digits[2]
            rule = f"Find the 3-digit number where each digit is between 1-9."
        
        self.state = GameState(
            target_rule=rule,
            target_number=target,
            history=[]
        )
    
    def guess(self, number: int) -> str:
        """Process a guess and return feedback."""
        self.state.attempts += 1
        
        if number == self.state.target_number:
            self.state.solved = True
            feedback = "CORRECT! You found it!"
            self.state.history.append((number, feedback))
            return feedback
        
        diff = abs(number - self.state.target_number)
        
        if diff <= 5:
            temp = "VERY CLOSE (within 5)!"
        elif diff <= 15:
            temp = "CLOSE (within 15)"
        elif diff <= 50:
            temp = "MODERATE distance"
        else:
            temp = "FAR"
        
        direction = "too low" if number < self.state.target_number else "too high"
        
        feedback = f"{direction.upper()}, {temp}"
        self.state.history.append((number, feedback))
        
        return feedback
    
    def get_history_summary(self) -> str:
        """Get summary of past guesses."""
        if not self.state.history:
            return "No guesses yet."
        
        lines = []
        for i, (guess, feedback) in enumerate(self.state.history[-5:], 1):  # Last 5
            lines.append(f"  Guess {guess}: {feedback}")
        return "\n".join(lines)


class GoalDirectedAgent:
    """
    An agent with a GOAL: solve the puzzle in minimum steps.
    
    Uses DDA-X for:
    - Memory (ledger): remembers what it tried and learned
    - Dynamics (rigidity): adapts exploration vs exploitation
    - Identity: stays focused on the goal
    """
    
    SYSTEM_PROMPT = """You are solving a number-guessing puzzle.

GOAL: Find the hidden number in as few guesses as possible.

RULE: {rule}

CURRENT STATE:
- Attempts so far: {attempts}
- Cognitive mode: {mode} (rigidity={rho:.2f})
- Recent history:
{history}

{memory_context}

IMPORTANT: You must output EXACTLY one number as your guess.
Think step by step, then at the end output:
GUESS: [number]

Be strategic. Use the feedback to narrow down. Learn from patterns.
"""

    def __init__(self, clean_ledger: bool = True):
        self.provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="openai/gpt-oss-20b",
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            timeout=300.0
        )
        
        ledger_path = Path("data/goal_directed")
        if clean_ledger and ledger_path.exists():
            import shutil
            shutil.rmtree(ledger_path)
        
        self.ledger = ExperienceLedger(
            storage_path=ledger_path,
            lambda_recency=0.01,
            lambda_salience=2.0
        )
        
        self.rigidity = MultiTimescaleRigidity()
        self.state: Optional[DDAState] = None
        self.identity_embedding: Optional[np.ndarray] = None
        
        # Track predictions for surprise calculation
        self.last_prediction: Optional[str] = None  # What we expected
        self.strategy_history: List[str] = []
        
    async def setup(self):
        """Initialize the goal-focused identity."""
        print(f"{C.CYAN}Initializing Goal-Directed Agent...{C.RESET}")
        
        # Identity: goal-focused problem solver
        identity_text = "problem solving, goal achievement, strategic thinking, learning from feedback, minimizing attempts, efficient search"
        self.identity_embedding = await self.provider.embed(identity_text)
        self.identity_embedding /= np.linalg.norm(self.identity_embedding)
        
        self.state = DDAState(
            x=self.identity_embedding.copy(),
            x_star=self.identity_embedding.copy(),
            gamma=1.2,      # Strong goal focus
            epsilon_0=0.35, # Moderate surprise threshold
            alpha=0.12,     # Responsive learning
            s=0.12,
            rho=0.2,        # Start exploratory
            x_pred=self.identity_embedding.copy()
        )
        
        print(f"{C.GREEN}✓ Agent ready with goal focus{C.RESET}")
    
    def get_mode(self) -> str:
        if self.state.rho < 0.3:
            return "EXPLORING"
        elif self.state.rho < 0.6:
            return "NARROWING"
        else:
            return "CONVERGING"
    
    async def retrieve_relevant_strategies(self, context: np.ndarray) -> str:
        """Retrieve past strategies that worked."""
        entries = self.ledger.retrieve(context, k=3, min_score=0.1)
        
        if not entries:
            return ""
        
        lines = ["PAST LEARNINGS:"]
        for entry in entries:
            if 'strategy' in entry.metadata:
                success = "✓" if entry.metadata.get('success') else "✗"
                lines.append(f"  {success} {entry.metadata['strategy'][:100]}")
        
        return "\n".join(lines)
    
    def parse_guess(self, response: str) -> Optional[int]:
        """Extract the guess number from response."""
        import re
        
        # Look for "GUESS: number"
        match = re.search(r'GUESS:\s*(\d+)', response, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Fallback: find last number in response
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            return int(numbers[-1])
        
        return None
    
    async def make_guess(self, puzzle: NumberPuzzle, turn: int) -> Tuple[int, str, float]:
        """
        Make one guess using full DDA-X dynamics.
        Returns: (guess, reasoning, surprise)
        """
        print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BLUE}TURN {turn}: {self.get_mode()}{C.RESET}")
        print(f"{C.DIM}{'─' * 60}{C.RESET}")
        
        # Build context for retrieval
        context_text = f"puzzle attempt {turn}: {puzzle.state.target_rule}"
        context_embedding = await self.provider.embed(context_text)
        context_embedding /= np.linalg.norm(context_embedding)
        if len(context_embedding) != len(self.state.x):
            context_embedding = context_embedding[:len(self.state.x)]
        
        # Retrieve relevant past strategies
        memory_context = await self.retrieve_relevant_strategies(context_embedding)
        
        # Build prompt
        prompt = self.SYSTEM_PROMPT.format(
            rule=puzzle.state.target_rule,
            attempts=puzzle.state.attempts,
            mode=self.get_mode(),
            rho=self.state.rho,
            history=puzzle.get_history_summary(),
            memory_context=memory_context if memory_context else "[First attempt]"
        )
        
        # Temperature from rigidity
        temperature = 0.3 + 0.5 * (1 - self.state.rho)
        
        print(f"{C.DIM}  Mode: {self.get_mode()} | ρ: {self.state.rho:.3f} | T: {temperature:.2f}{C.RESET}")
        
        # Generate response
        print(f"\n{C.MAGENTA}[REASONING @ {self.get_mode()}]{C.RESET}")
        
        response = ""
        try:
            async for token in self.provider.stream(
                prompt,
                system_prompt="",  # System already in prompt
                temperature=temperature,
                max_tokens=250
            ):
                if not token.startswith("__THOUGHT__"):
                    print(token, end="", flush=True)
                    response += token
        except Exception as e:
            print(f"[Error: {e}]")
            response = "I need to make a guess. GUESS: 50"
        
        print()
        
        # Parse guess
        guess = self.parse_guess(response)
        if guess is None:
            guess = random.randint(1, 100)
            print(f"{C.YELLOW}[Fallback random guess: {guess}]{C.RESET}")
        
        # Make the guess and get feedback
        feedback = puzzle.guess(guess)
        
        print(f"\n{C.WHITE}→ GUESS: {guess}{C.RESET}")
        
        if puzzle.state.solved:
            print(f"{C.GREEN}★ {feedback}{C.RESET}")
        else:
            color = C.YELLOW if "CLOSE" in feedback else C.RED
            print(f"{color}← {feedback}{C.RESET}")
        
        # Compute surprise: did outcome match prediction?
        # Embed the outcome to compare with prediction
        outcome_text = f"guess {guess}: {feedback}"
        try:
            outcome_embedding = await self.provider.embed(outcome_text)
            outcome_embedding /= (np.linalg.norm(outcome_embedding) + 1e-9)
            if len(outcome_embedding) != len(self.state.x_pred):
                outcome_embedding = outcome_embedding[:len(self.state.x_pred)]
            
            epsilon = np.linalg.norm(self.state.x_pred - outcome_embedding)
        except:
            epsilon = 0.3  # Default surprise on error
            outcome_embedding = self.state.x_pred.copy()
        
        # Success amplifies/dampens surprise
        if puzzle.state.solved:
            epsilon *= 0.5  # Success is less surprising (converging)
        elif "VERY CLOSE" in feedback:
            epsilon *= 0.7  # Getting warmer
        elif "FAR" in feedback:
            epsilon *= 1.2  # Bad surprise
        
        # Update dynamics
        rho_before = self.state.rho
        self.state.update_rigidity(epsilon)
        self.rigidity.update(epsilon)
        
        # Update state
        k_eff = 0.2 * (1 - self.state.rho)
        self.state.x = (1 - k_eff) * self.state.x + k_eff * outcome_embedding
        self.state.x_pred = outcome_embedding
        
        # Store in ledger
        was_good = puzzle.state.solved or "CLOSE" in feedback
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=self.state.x.copy(),
            action_id=f"guess_{turn}",
            observation_embedding=context_embedding,
            outcome_embedding=outcome_embedding,
            prediction_error=epsilon,
            context_embedding=context_embedding,
            rigidity_at_time=self.state.rho,
            was_successful=was_good,
            metadata={
                'turn': turn,
                'guess': guess,
                'feedback': feedback,
                'strategy': response[:150],
                'success': was_good
            }
        )
        self.ledger.add_entry(entry)
        
        # If solved, store reflection
        if puzzle.state.solved:
            reflection = ReflectionEntry(
                timestamp=time.time(),
                task_intent="solve_puzzle",
                situation_embedding=context_embedding,
                reflection_text=f"Solved in {puzzle.state.attempts} attempts. Winning strategy: narrowing from feedback.",
                prediction_error=epsilon,
                outcome_success=True
            )
            self.ledger.add_reflection(reflection)
        
        # Display dynamics
        delta_rho = self.state.rho - rho_before
        print(f"\n{C.DIM}── Dynamics ──{C.RESET}")
        print(f"  Surprise: {epsilon:.3f}")
        color = C.GREEN if delta_rho < 0 else C.RED
        print(f"  Δρ: {color}{delta_rho:+.3f}{C.RESET} → {self.state.rho:.3f}")
        
        return guess, response, epsilon
    
    async def solve(self, puzzle: NumberPuzzle, max_attempts: int = 15) -> bool:
        """Attempt to solve the puzzle within max attempts."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}    GOAL-DIRECTED LEARNING DEMONSTRATION{C.RESET}")
        print(f"{C.BOLD}{'═' * 60}{C.RESET}")
        
        print(f"\n{C.CYAN}GOAL: {puzzle.state.target_rule}{C.RESET}")
        print(f"{C.DIM}(Target: {puzzle.state.target_number} - hidden from agent){C.RESET}")
        
        for turn in range(1, max_attempts + 1):
            guess, reasoning, surprise = await self.make_guess(puzzle, turn)
            
            if puzzle.state.solved:
                break
            
            time.sleep(0.2)
        
        # Final results
        self.display_results(puzzle)
        return puzzle.state.solved
    
    def display_results(self, puzzle: NumberPuzzle):
        """Display final results."""
        print(f"\n\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}    RESULTS{C.RESET}")
        print(f"{C.BOLD}{'═' * 60}{C.RESET}")
        
        stats = self.ledger.get_statistics()
        
        if puzzle.state.solved:
            print(f"\n{C.GREEN}✓ PUZZLE SOLVED in {puzzle.state.attempts} attempts!{C.RESET}")
        else:
            print(f"\n{C.RED}✗ Failed to solve in allowed attempts.{C.RESET}")
        
        print(f"\n{C.CYAN}Statistics:{C.RESET}")
        print(f"  Ledger entries: {stats.get('current_entries', 0)}")
        print(f"  Final rigidity: {self.state.rho:.3f}")
        print(f"  Mode: {self.get_mode()}")
        
        diag = self.rigidity.get_diagnostic()
        print(f"  Fast ρ: {diag['rho_fast']:.3f}")
        print(f"  Slow ρ: {diag['rho_slow']:.3f}")
        
        print(f"\n{C.CYAN}Guess History:{C.RESET}")
        for guess, feedback in puzzle.state.history:
            mark = "★" if "CORRECT" in feedback else "·"
            print(f"  {mark} {guess}: {feedback}")
        
        print(f"\n{C.BOLD}The DDA-X dynamics helped the agent:{C.RESET}")
        print(f"  - Start in EXPLORING mode (low ρ = high temperature)")
        print(f"  - Adapt based on feedback surprise")
        print(f"  - Converge as predictions improved")


async def main():
    # Create puzzle
    puzzle = NumberPuzzle(difficulty="easy")
    
    # Create agent and solve
    agent = GoalDirectedAgent(clean_ledger=True)
    await agent.solve(puzzle, max_attempts=12)


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.RED}Terminated.{C.RESET}")
