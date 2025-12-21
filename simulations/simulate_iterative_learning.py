#!/usr/bin/env python3
"""
Iterative Learning via Ledger + Reflection
===========================================

The DDA-X learning formula:
  Experience → Embed → Store (surprise-weighted) → Retrieve → Reflect → Adapt

Each iteration:
1. Model attempts task
2. Outcome stored in ledger with prediction error
3. High-surprise outcomes trigger reflection
4. Reflections stored separately
5. Next iteration retrieves BOTH experiences AND reflections
6. Model has accumulated insights it couldn't have in iteration 1

This is ACTUAL learning: the model in iteration N has access to
insights that only exist BECAUSE of iterations 1 through N-1.
"""

import asyncio
import sys
import os
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.llm.hybrid_provider import HybridProvider


class C:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


# The learning task: discover a hidden pattern through trial and reflection
# This is NOT trivially solvable - it requires accumulating insights
TASK = """
You are trying to understand an alien communication system.

Each iteration, you receive a message and must guess its meaning.
After each guess, you learn if you were right or wrong.

The alien system has HIDDEN RULES that you must discover:
- Certain symbols always mean certain things
- Context matters
- There's an underlying grammar

You can only learn by accumulating observations and reflecting on patterns.
"""

# Hidden rules (the model doesn't know these)
ALIEN_MESSAGES = [
    ("△○□", "greet", "greeting gesture"),
    ("□□△", "danger", "warning signal"),  
    ("○○○", "agree", "affirmation"),
    ("△□○", "question", "inquiry marker"),
    ("□○△", "gift", "offering symbol"),
    ("○△□", "farewell", "departure gesture"),
    ("△△△", "urgent", "emphasis/urgency"),
    ("□○○", "peace", "calm/safety"),
    ("○□△", "help", "assistance request"),
    ("△□□", "refuse", "negation"),
]


class IterativeLearner:
    """
    Demonstrates iterative learning through ledger + reflection.
    
    Each cycle:
    1. Present new alien message
    2. Retrieve relevant past attempts + reflections
    3. Model guesses meaning
    4. Record outcome with surprise
    5. If wrong, generate reflection on what might help
    6. Repeat with accumulated knowledge
    """
    
    SYSTEM_PROMPT = """You are learning an alien communication system.

Your accumulated knowledge so far:
{memories}

{reflections}

Current cognitive state: iteration {iteration}, confidence={confidence:.2f}

Based on your accumulated observations and reflections, make your best guess.
Output format: GUESS: [your guess for what this message means]
Then explain your reasoning briefly."""

    def __init__(self):
        self.provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="openai/gpt-oss-20b",
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            timeout=300.0
        )
        
        # Clean ledger
        ledger_path = Path("data/iterative_learning")
        if ledger_path.exists():
            import shutil
            shutil.rmtree(ledger_path)
        
        self.ledger = ExperienceLedger(
            storage_path=ledger_path,
            lambda_recency=0.01,
            lambda_salience=2.0
        )
        
        self.rigidity = MultiTimescaleRigidity()
        self.state: Optional[DDAState] = None
        self.identity_emb: Optional[np.ndarray] = None
        
        self.correct_count = 0
        self.total_count = 0
        
    async def setup(self):
        """Initialize."""
        print(f"{C.CYAN}Initializing Iterative Learner...{C.RESET}")
        
        identity = "pattern recognition, learning from experience, accumulating knowledge"
        self.identity_emb = await self.provider.embed(identity)
        self.identity_emb /= np.linalg.norm(self.identity_emb)
        
        self.state = DDAState(
            x=self.identity_emb.copy(),
            x_star=self.identity_emb.copy(),
            gamma=1.0,
            epsilon_0=0.35,
            alpha=0.1,
            s=0.1,
            rho=0.3,
            x_pred=self.identity_emb.copy()
        )
        
        print(f"{C.GREEN}✓ Ready{C.RESET}")
    
    async def retrieve_memories(self, context_emb: np.ndarray) -> str:
        """Retrieve relevant past experiences."""
        entries = self.ledger.retrieve(context_emb, k=8, min_score=0.05)
        
        if not entries:
            return "[No prior observations yet]"
        
        lines = ["PAST OBSERVATIONS:"]
        for entry in entries:
            if 'message' in entry.metadata:
                outcome = "✓" if entry.metadata.get('correct') else "✗"
                lines.append(f"  {outcome} {entry.metadata['message']} → guessed '{entry.metadata.get('guess', '?')}' (actual: '{entry.metadata.get('actual', '?')}')")
        
        return "\n".join(lines)
    
    async def retrieve_reflections(self, context_emb: np.ndarray) -> str:
        """Retrieve accumulated reflections."""
        reflections = self.ledger.retrieve_reflections(context_emb, k=5, min_score=0.05)
        
        if not reflections:
            return ""
        
        lines = ["INSIGHTS FROM REFLECTION:"]
        for ref in reflections:
            lines.append(f"  ↳ {ref.reflection_text}")
        
        return "\n".join(lines)
    
    def parse_guess(self, response: str) -> str:
        """Extract guess from response."""  
        import re
        match = re.search(r'GUESS:\s*["\']?([^"\'\n]+)["\']?', response, re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
        # Fallback
        words = response.split()[:3]
        return " ".join(words).lower()
    
    async def learn_iteration(self, iteration: int, message: str, actual_meaning: str, hint: str):
        """One learning iteration."""
        print(f"\n{C.BOLD}{'─' * 50}{C.RESET}")
        print(f"{C.BLUE}Iteration {iteration}{C.RESET}")
        print(f"{C.WHITE}Message: {message}{C.RESET}")
        
        # Embed current context
        context = f"alien message: {message}"
        try:
            context_emb = await self.provider.embed(context)
            context_emb /= np.linalg.norm(context_emb)
            if len(context_emb) != len(self.identity_emb):
                context_emb = context_emb[:len(self.identity_emb)]
        except:
            context_emb = self.identity_emb.copy()
        
        # Retrieve accumulated knowledge
        memories = await self.retrieve_memories(context_emb)
        reflections = await self.retrieve_reflections(context_emb)
        
        # Count what we're injecting
        mem_count = memories.count('\n')
        ref_count = reflections.count('\n') if reflections else 0
        print(f"{C.DIM}Retrieved: {mem_count} memories, {ref_count} reflections{C.RESET}")
        
        # Build prompt
        confidence = 1 - self.state.rho
        system = self.SYSTEM_PROMPT.format(
            memories=memories,
            reflections=reflections if reflections else "[No reflections yet]",
            iteration=iteration,
            confidence=confidence
        )
        
        prompt = f"New alien message to interpret: {message}\n\nWhat does it mean?"
        
        # Generate response
        response = ""
        try:
            async for token in self.provider.stream(
                prompt,
                system_prompt=system,
                temperature=0.3 + 0.4 * self.state.rho,
                max_tokens=150
            ):
                if not token.startswith("__THOUGHT__"):
                    response += token
        except:
            response = "GUESS: unknown"
        
        guess = self.parse_guess(response)
        correct = guess.lower() == actual_meaning.lower() or actual_meaning.lower() in guess.lower()
        
        self.total_count += 1
        if correct:
            self.correct_count += 1
        
        # Display result
        status = f"{C.GREEN}✓ CORRECT{C.RESET}" if correct else f"{C.YELLOW}✗ Wrong{C.RESET}"
        print(f"  Guess: '{guess}' → {status} (actual: '{actual_meaning}')")
        
        # Compute surprise
        try:
            outcome_text = f"message {message} means {actual_meaning}"
            outcome_emb = await self.provider.embed(outcome_text)
            outcome_emb /= (np.linalg.norm(outcome_emb) + 1e-9)
            if len(outcome_emb) != len(self.state.x_pred):
                outcome_emb = outcome_emb[:len(self.state.x_pred)]
            epsilon = np.linalg.norm(self.state.x_pred - outcome_emb)
        except:
            epsilon = 0.4
            outcome_emb = self.state.x_pred.copy()
        
        # Adjust surprise based on correctness
        if correct:
            epsilon *= 0.5  # Less surprising when right
        
        # Update dynamics
        rho_before = self.state.rho
        self.state.update_rigidity(epsilon)
        self.rigidity.update(epsilon)
        delta = self.state.rho - rho_before
        
        print(f"  {C.DIM}Surprise: {epsilon:.2f} | Δρ: {delta:+.3f} → ρ={self.state.rho:.3f}{C.RESET}")
        
        # Store in ledger
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=self.state.x.copy(),
            action_id=f"iter_{iteration}",
            observation_embedding=context_emb,
            outcome_embedding=outcome_emb,
            prediction_error=epsilon,
            context_embedding=context_emb,
            rigidity_at_time=self.state.rho,
            was_successful=correct,
            metadata={
                'iteration': iteration,
                'message': message,
                'guess': guess,
                'actual': actual_meaning,
                'correct': correct,
                'hint': hint
            }
        )
        self.ledger.add_entry(entry)
        
        # Generate reflection if wrong (learning moment)
        if not correct:
            reflect_prompt = f"""I guessed '{guess}' for message '{message}' but it actually meant '{actual_meaning}'.
The hint was: {hint}

What pattern or rule might I be missing? State ONE insight that might help next time."""
            
            reflection = await self.provider.complete(
                reflect_prompt,
                system_prompt="Extract one concise insight about the pattern.",
                max_tokens=60
            )
            reflection = reflection.strip()[:100]
            
            ref_entry = ReflectionEntry(
                timestamp=time.time(),
                task_intent="alien_communication",
                situation_embedding=context_emb,
                reflection_text=reflection,
                prediction_error=epsilon,
                outcome_success=False
            )
            self.ledger.add_reflection(ref_entry)
            print(f"  {C.MAGENTA}Reflection: {reflection}{C.RESET}")
        
        # Update state
        self.state.x_pred = outcome_emb
        self.state.x = (1 - 0.1) * self.state.x + 0.1 * outcome_emb
        
        return correct
    
    async def run(self, iterations: int = 10):
        """Run iterative learning."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═' * 50}{C.RESET}")
        print(f"{C.BOLD}  ITERATIVE LEARNING VIA LEDGER + REFLECTION{C.RESET}")
        print(f"{C.BOLD}{'═' * 50}{C.RESET}")
        print(f"\n{TASK}")
        
        # Run through examples
        for i in range(min(iterations, len(ALIEN_MESSAGES))):
            message, meaning, hint = ALIEN_MESSAGES[i]
            await self.learn_iteration(i + 1, message, meaning, hint)
            time.sleep(0.2)
        
        # Summary
        self.display_summary()
    
    def display_summary(self):
        """Show learning summary."""
        print(f"\n\n{C.BOLD}{'═' * 50}{C.RESET}")
        print(f"{C.BOLD}  LEARNING SUMMARY{C.RESET}")
        print(f"{C.BOLD}{'═' * 50}{C.RESET}")
        
        stats = self.ledger.get_statistics()
        
        accuracy = self.correct_count / self.total_count if self.total_count > 0 else 0
        print(f"\n{C.CYAN}Results:{C.RESET}")
        print(f"  Correct: {self.correct_count}/{self.total_count} ({accuracy*100:.0f}%)")
        print(f"  Ledger entries: {stats.get('current_entries', 0)}")
        print(f"  Reflections stored: {stats.get('current_reflections', 0)}")
        print(f"  Final rigidity: {self.state.rho:.3f}")
        
        diag = self.rigidity.get_diagnostic()
        print(f"  Trauma: {diag['rho_trauma']:.6f}")
        
        print(f"\n{C.CYAN}What happened:{C.RESET}")
        print(f"  Iteration 1: No memories, no reflections - guessing blind")
        print(f"  Iteration N: {stats.get('current_entries', 0)} experiences + {stats.get('current_reflections', 0)} reflections injected")
        print(f"\n{C.BOLD}The model in iteration 10 is DIFFERENT from iteration 1.{C.RESET}")
        print(f"{C.BOLD}It has accumulated knowledge that didn't exist before.{C.RESET}")


async def main():
    learner = IterativeLearner()
    await learner.run(iterations=10)


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Terminated.{C.RESET}")
