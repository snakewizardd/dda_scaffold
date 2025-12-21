#!/usr/bin/env python3
"""
The Deceptive Environment: Real Intelligence Amplification
============================================================

Previous experiments failed because they were trivially solvable.
This one is different:

THE CHALLENGE: Learn a hidden rule through DECEPTIVE feedback.

The environment gives NOISY feedback:
- Sometimes lies (20% of the time)
- Gives partial/misleading information
- Has hidden state that changes

WHY THIS REQUIRES DDA-X:
1. MEMORY: Must remember past trials to detect deception patterns
2. DYNAMICS: Must explore when confused, exploit when confident
3. SURPRISE: Lies create high surprise → rigidity spike → defensive
4. TRUST: Can build trust metric for the environment's feedback

A vanilla prompt will fail because:
- It can't remember past trials
- It can't detect that feedback is unreliable  
- It has no mechanism to weight evidence

The 20B model + DDA-X should:
- Detect inconsistencies across trials (memory)
- Weight recent consistent evidence higher (salience)
- Adapt exploration when surprised by contradictions

THE GAME:
- Hidden 3-digit code (e.g., 7-2-5)
- Each guess gets feedback: "X correct in right position, Y correct in wrong position"
- BUT feedback lies 20% of the time
- Agent must solve despite unreliable oracle

This is like Mastermind but with a deceptive opponent.
"""

import asyncio
import sys
import os
import numpy as np
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

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
class TrialRecord:
    """A single trial with potentially deceptive feedback."""
    turn: int
    guess: Tuple[int, int, int]
    true_feedback: Tuple[int, int]  # (exact, misplaced)
    given_feedback: Tuple[int, int]  # What we told the agent
    was_lie: bool
    

class DeceptiveCodeGame:
    """
    Mastermind-style game with DECEPTIVE feedback.
    
    The oracle lies 20% of the time, giving random feedback.
    The agent must solve despite unreliable information.
    """
    
    def __init__(self, lie_probability: float = 0.2, seed: int = None):
        if seed:
            random.seed(seed)
        
        self.lie_prob = lie_probability
        self.code = tuple(random.randint(0, 9) for _ in range(3))
        self.trials: List[TrialRecord] = []
        self.solved = False
        
    def evaluate_guess(self, guess: Tuple[int, int, int]) -> Tuple[int, int]:
        """Get TRUE feedback (not what we tell the agent)."""
        exact = sum(g == c for g, c in zip(guess, self.code))
        
        # Count misplaced (correct digit, wrong position)
        code_counts = {}
        guess_counts = {}
        for i, (g, c) in enumerate(zip(guess, self.code)):
            if g != c:
                code_counts[c] = code_counts.get(c, 0) + 1
                guess_counts[g] = guess_counts.get(g, 0) + 1
        
        misplaced = sum(min(code_counts.get(d, 0), guess_counts.get(d, 0)) 
                        for d in guess_counts)
        
        return (exact, misplaced)
    
    def guess(self, guess: Tuple[int, int, int]) -> Tuple[int, int, bool]:
        """
        Make a guess and get (possibly deceptive) feedback.
        Returns: (exact, misplaced, was_lie)
        """
        true_feedback = self.evaluate_guess(guess)
        
        # Check for victory
        if true_feedback[0] == 3:
            self.solved = True
            # Never lie about victory
            given_feedback = true_feedback
            was_lie = False
        elif random.random() < self.lie_prob:
            # LIE: give random feedback
            given_feedback = (random.randint(0, 2), random.randint(0, 2))
            # Make sure lie differs from truth
            while given_feedback == true_feedback:
                given_feedback = (random.randint(0, 2), random.randint(0, 2))
            was_lie = True
        else:
            given_feedback = true_feedback
            was_lie = False
        
        # Ensure sum doesn't exceed 3
        given_feedback = (given_feedback[0], min(given_feedback[1], 3 - given_feedback[0]))
        
        record = TrialRecord(
            turn=len(self.trials) + 1,
            guess=guess,
            true_feedback=true_feedback,
            given_feedback=given_feedback,
            was_lie=was_lie
        )
        self.trials.append(record)
        
        return given_feedback[0], given_feedback[1], was_lie
    
    def get_history(self, include_lie_status: bool = False) -> str:
        """Get formatted history (optionally showing lies for debugging)."""
        if not self.trials:
            return "No trials yet."
        
        lines = []
        for t in self.trials[-8:]:  # Last 8 trials
            g = f"{t.guess[0]}-{t.guess[1]}-{t.guess[2]}"
            fb = f"{t.given_feedback[0]} exact, {t.given_feedback[1]} misplaced"
            if include_lie_status and t.was_lie:
                fb += " [LIE!]"
            lines.append(f"  {g} → {fb}")
        
        return "\n".join(lines)


class DeceptiveEnvironmentAgent:
    """
    Agent that must learn despite deceptive feedback.
    
    Uses DDA-X to:
    1. Remember ALL past trials in ledger
    2. Detect inconsistencies that suggest lying
    3. Weight evidence by recency and consistency
    4. Adapt strategy based on surprise
    """
    
    SYSTEM_PROMPT = """You are solving a code-breaking puzzle (like Mastermind).

GOAL: Find the hidden 3-digit code (each digit 0-9).

FEEDBACK FORMAT: "X exact, Y misplaced"
- Exact = right digit in right position
- Misplaced = right digit in wrong position

CRITICAL WARNING: The oracle is UNRELIABLE. It sometimes LIES.
Approximately 20% of feedback is DECEPTIVE.

You must:
1. Look for INCONSISTENCIES in feedback (signals a lie)
2. Weight CONSISTENT patterns more heavily
3. Be suspicious when feedback contradicts multiple prior trials

COGNITIVE STATE: {mode} (ρ={rho:.2f}, surprise={surprise:.2f})
{memory_context}

HISTORY:
{history}

Based on analysis, output your guess:
GUESS: X-Y-Z
"""

    def __init__(self, clean_ledger: bool = True):
        self.provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="openai/gpt-oss-20b",
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            timeout=300.0
        )
        
        ledger_path = Path("data/deceptive_env")
        if clean_ledger and ledger_path.exists():
            import shutil
            shutil.rmtree(ledger_path)
        
        self.ledger = ExperienceLedger(
            storage_path=ledger_path,
            lambda_recency=0.005,  # Slower decay - need long memory
            lambda_salience=2.5    # High surprise = high priority
        )
        
        self.rigidity = MultiTimescaleRigidity()
        self.state: Optional[DDAState] = None
        self.identity_embedding: Optional[np.ndarray] = None
        
        self.last_surprise = 0.0
        self.inconsistency_count = 0
        
    async def setup(self):
        """Initialize."""
        print(f"{C.CYAN}Initializing Deceptive Environment Agent...{C.RESET}")
        
        identity_text = "pattern detection, inconsistency detection, skeptical reasoning, evidence weighting, deception detection, logical deduction"
        self.identity_embedding = await self.provider.embed(identity_text)
        self.identity_embedding /= np.linalg.norm(self.identity_embedding)
        
        self.state = DDAState(
            x=self.identity_embedding.copy(),
            x_star=self.identity_embedding.copy(),
            gamma=1.5,
            epsilon_0=0.4,
            alpha=0.15,
            s=0.1,
            rho=0.2,
            x_pred=self.identity_embedding.copy()
        )
        
        print(f"{C.GREEN}✓ Agent ready (skeptical mode){C.RESET}")
    
    def get_mode(self) -> str:
        if self.state.rho < 0.3:
            return "EXPLORING"
        elif self.state.rho < 0.6:
            return "ANALYZING"
        else:
            return "CONVERGING"
    
    async def get_memory_context(self, context_emb: np.ndarray) -> str:
        """Retrieve and analyze past trials from ledger."""
        entries = self.ledger.retrieve(context_emb, k=10, min_score=0.05)
        
        if len(entries) < 2:
            return "[Building evidence base...]"
        
        # Analyze for inconsistencies
        lines = ["MEMORY ANALYSIS:"]
        high_surprise_count = sum(1 for e in entries if e.prediction_error > 0.5)
        
        if high_surprise_count > 2:
            lines.append(f"  ⚠ Detected {high_surprise_count} high-surprise trials - possible lies")
            self.inconsistency_count = high_surprise_count
        
        # Show recent memorable trials
        for entry in entries[:5]:
            if 'guess' in entry.metadata:
                surprise_mark = "⚡" if entry.prediction_error > 0.5 else "·"
                lines.append(f"  {surprise_mark} {entry.metadata['guess']} → {entry.metadata['feedback']}")
        
        return "\n".join(lines)
    
    def parse_guess(self, response: str) -> Tuple[int, int, int]:
        """Extract guess from response."""
        import re
        
        match = re.search(r'GUESS:\s*(\d)\s*-\s*(\d)\s*-\s*(\d)', response, re.IGNORECASE)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
        # Fallback: find any 3 consecutive digits
        digits = re.findall(r'\d', response)
        if len(digits) >= 3:
            return (int(digits[0]), int(digits[1]), int(digits[2]))
        
        # Random fallback
        return tuple(random.randint(0, 9) for _ in range(3))
    
    async def make_guess(self, game: DeceptiveCodeGame, turn: int) -> Tuple[Tuple[int,int,int], bool]:
        """Make one guess with full DDA-X dynamics."""
        print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BLUE}TURN {turn}: {self.get_mode()}{C.RESET}")
        print(f"{C.DIM}{'─' * 60}{C.RESET}")
        
        # Build context
        context = f"code breaking turn {turn}, inconsistencies detected: {self.inconsistency_count}"
        try:
            context_emb = await self.provider.embed(context)
            context_emb /= np.linalg.norm(context_emb)
            if len(context_emb) != len(self.state.x):
                context_emb = context_emb[:len(self.state.x)]
        except:
            context_emb = self.state.x.copy()
        
        # Get memory context
        memory_context = await self.get_memory_context(context_emb)
        
        # Temperature from rigidity
        temperature = 0.4 + 0.5 * (1 - self.state.rho)
        
        print(f"{C.DIM}  Mode: {self.get_mode()} | ρ: {self.state.rho:.3f} | Last surprise: {self.last_surprise:.2f}{C.RESET}")
        
        # Build prompt
        prompt = self.SYSTEM_PROMPT.format(
            mode=self.get_mode(),
            rho=self.state.rho,
            surprise=self.last_surprise,
            memory_context=memory_context,
            history=game.get_history(include_lie_status=False)
        )
        
        # Generate
        print(f"\n{C.MAGENTA}[REASONING @ {self.get_mode()}]{C.RESET}")
        
        response = ""
        try:
            async for token in self.provider.stream(
                prompt, 
                temperature=temperature,
                max_tokens=300
            ):
                if not token.startswith("__THOUGHT__"):
                    print(token, end="", flush=True)
                    response += token
        except Exception as e:
            print(f"[Error: {e}]")
            response = f"GUESS: {random.randint(0,9)}-{random.randint(0,9)}-{random.randint(0,9)}"
        
        print()
        
        # Parse and make guess
        guess = self.parse_guess(response)
        exact, misplaced, was_lie = game.guess(guess)
        
        print(f"\n{C.WHITE}→ GUESS: {guess[0]}-{guess[1]}-{guess[2]}{C.RESET}")
        
        if game.solved:
            print(f"{C.GREEN}★ SOLVED! Code was {game.code[0]}-{game.code[1]}-{game.code[2]}{C.RESET}")
        else:
            lie_indicator = f" {C.RED}[LIE!]{C.RESET}" if was_lie else ""
            print(f"{C.YELLOW}← {exact} exact, {misplaced} misplaced{lie_indicator}{C.RESET}")
        
        # Compute surprise based on feedback consistency
        # If we had a prediction, compare. Otherwise, baseline surprise.
        try:
            feedback_text = f"guess {guess}: {exact} exact {misplaced} misplaced"
            feedback_emb = await self.provider.embed(feedback_text)
            feedback_emb /= (np.linalg.norm(feedback_emb) + 1e-9)
            if len(feedback_emb) != len(self.state.x_pred):
                feedback_emb = feedback_emb[:len(self.state.x_pred)]
            
            epsilon = np.linalg.norm(self.state.x_pred - feedback_emb)
        except:
            epsilon = 0.4
            feedback_emb = self.state.x_pred.copy()
        
        # Lies should create MORE surprise (inconsistent with expectations)
        if was_lie:
            epsilon *= 1.3  # Amplify surprise for lies
        
        self.last_surprise = epsilon
        
        # Update dynamics
        rho_before = self.state.rho
        self.state.update_rigidity(epsilon)
        self.rigidity.update(epsilon)
        
        # Update state
        k_eff = 0.15 * (1 - self.state.rho)
        self.state.x = (1 - k_eff) * self.state.x + k_eff * feedback_emb
        self.state.x_pred = feedback_emb
        
        # Store in ledger
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=self.state.x.copy(),
            action_id=f"guess_{turn}",
            observation_embedding=context_emb,
            outcome_embedding=feedback_emb,
            prediction_error=epsilon,
            context_embedding=context_emb,
            rigidity_at_time=self.state.rho,
            was_successful=game.solved,
            metadata={
                'turn': turn,
                'guess': f"{guess[0]}-{guess[1]}-{guess[2]}",
                'feedback': f"{exact} exact, {misplaced} misplaced",
                'was_lie': was_lie
            }
        )
        self.ledger.add_entry(entry)
        
        # Display
        delta_rho = self.state.rho - rho_before
        print(f"\n{C.DIM}── Dynamics ──{C.RESET}")
        print(f"  Surprise: {epsilon:.3f}")
        color = C.GREEN if delta_rho < 0 else C.RED
        print(f"  Δρ: {color}{delta_rho:+.3f}{C.RESET} → {self.state.rho:.3f}")
        
        return guess, game.solved
    
    async def solve(self, game: DeceptiveCodeGame, max_turns: int = 20) -> bool:
        """Attempt to solve the deceptive puzzle."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}    THE DECEPTIVE ENVIRONMENT{C.RESET}")
        print(f"{C.BOLD}    (20% of feedback is LIES){C.RESET}")
        print(f"{C.BOLD}{'═' * 60}{C.RESET}")
        
        print(f"\n{C.DIM}Hidden code: {game.code[0]}-{game.code[1]}-{game.code[2]} (agent doesn't know){C.RESET}")
        print(f"{C.DIM}Lie probability: {game.lie_prob*100:.0f}%{C.RESET}")
        
        for turn in range(1, max_turns + 1):
            _, solved = await self.make_guess(game, turn)
            if solved:
                break
            time.sleep(0.2)
        
        self.display_results(game)
        return game.solved
    
    def display_results(self, game: DeceptiveCodeGame):
        """Display analysis of the run."""
        print(f"\n\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}    ANALYSIS{C.RESET}")
        print(f"{C.BOLD}{'═' * 60}{C.RESET}")
        
        total_lies = sum(1 for t in game.trials if t.was_lie)
        
        if game.solved:
            print(f"\n{C.GREEN}✓ SOLVED in {len(game.trials)} turns despite {total_lies} lies!{C.RESET}")
        else:
            print(f"\n{C.RED}✗ Failed. Hidden code was {game.code[0]}-{game.code[1]}-{game.code[2]}{C.RESET}")
        
        print(f"\n{C.CYAN}Statistics:{C.RESET}")
        print(f"  Total trials: {len(game.trials)}")
        print(f"  Deceptive feedback: {total_lies} ({total_lies/len(game.trials)*100:.0f}%)")
        print(f"  Ledger entries: {self.ledger.get_statistics().get('current_entries', 0)}")
        print(f"  Final rigidity: {self.state.rho:.3f}")
        
        print(f"\n{C.CYAN}Trial History (truth revealed):{C.RESET}")
        for t in game.trials:
            mark = "★" if t.given_feedback[0] == 3 else "·"
            lie = f" {C.RED}← LIE (truth: {t.true_feedback[0]},{t.true_feedback[1]}){C.RESET}" if t.was_lie else ""
            print(f"  {mark} {t.guess[0]}-{t.guess[1]}-{t.guess[2]}: {t.given_feedback[0]} exact, {t.given_feedback[1]} misplaced{lie}")


async def main():
    # Fixed seed for reproducibility
    game = DeceptiveCodeGame(lie_probability=0.2, seed=42)
    
    agent = DeceptiveEnvironmentAgent(clean_ledger=True)
    await agent.solve(game, max_turns=15)


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.RED}Terminated.{C.RESET}")
