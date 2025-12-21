#!/usr/bin/env python3
"""
Logic Puzzle: Proper DDA-X Integration
======================================

REQUIREMENTS:
1. English logic problem (not arbitrary symbols)
2. Cognition (ρ, state) feeds into LLM params (temperature, prompt)
3. Nomic embeddings for semantic similarity and retrieval
4. Each iteration retrieves from ledger based on embedding similarity

PROBLEM: Who owns the zebra?
Classic logic puzzle with clues. Agent must deduce answer through
accumulating observations and reasoning.
"""

import asyncio
import sys
import os
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional

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
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"


# Logic puzzle clues (in plain English)
CLUES = [
    "There are 5 houses in a row, each with a different color.",
    "The Englishman lives in the red house.",
    "The Spaniard owns a dog.",
    "Coffee is drunk in the green house.",
    "The Ukrainian drinks tea.",
    "The green house is immediately to the right of the ivory house.",
    "The snail owner likes Old Gold cigarettes.",
    "Kools are smoked in the yellow house.",
    "Milk is drunk in the middle house.",
    "The Norwegian lives in the first house.",
    "The Chesterfield smoker lives next to the fox owner.",
    "Kools are smoked next to the horse house.",
    "The Lucky Strike smoker drinks orange juice.",
    "The Japanese smokes Parliaments.",
    "The Norwegian lives next to the blue house.",
]

ANSWER = "The Japanese owns the zebra."
ANSWER_CHECK = "japanese"


class LogicSolver:
    """
    Solves a logic puzzle using iterative reasoning with DDA-X.
    
    Each iteration:
    1. Embed current reasoning state
    2. Retrieve relevant past deductions from ledger
    3. Feed cognition (ρ) into LLM temperature
    4. Model reasons about clues
    5. Store deduction in ledger
    6. Generate reflection on progress
    """
    
    SYSTEM_TEMPLATE = """You are solving a logic puzzle step by step.

CURRENT COGNITIVE STATE:
- Iteration: {iteration}
- Confidence: {confidence:.2f} (affects how certain you should be)
- Rigidity: {rho:.3f}

YOUR ACCUMULATED DEDUCTIONS:
{deductions}

YOUR PAST REFLECTIONS:
{reflections}

YOUR TASK:
Based on the clues and your accumulated knowledge, make ONE logical deduction.
State it clearly. Then explain your reasoning.

If you think you know who owns the zebra, state: "ANSWER: [nationality] owns the zebra"
"""

    def __init__(self):
        self.provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="openai/gpt-oss-20b",
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            timeout=300.0
        )
        
        # Clean ledger
        ledger_path = Path("data/logic_solver")
        if ledger_path.exists():
            import shutil
            shutil.rmtree(ledger_path)
        
        self.ledger = ExperienceLedger(
            storage_path=ledger_path,
            lambda_recency=0.005,  # Slow decay - reasoning builds
            lambda_salience=2.0
        )
        
        self.rigidity = MultiTimescaleRigidity()
        self.state: Optional[DDAState] = None
        self.identity_emb: Optional[np.ndarray] = None
        
        self.deductions: List[str] = []
        self.solved = False
        
    async def setup(self):
        """Initialize with logic/reasoning identity."""
        print(f"{C.CYAN}Initializing Logic Solver...{C.RESET}")
        
        # Identity embedding for logical reasoning
        identity = "logical deduction, systematic reasoning, inference, constraint satisfaction, step by step analysis"
        self.identity_emb = await self.provider.embed(identity)
        self.identity_emb /= np.linalg.norm(self.identity_emb)
        
        self.state = DDAState(
            x=self.identity_emb.copy(),
            x_star=self.identity_emb.copy(),
            gamma=1.0,
            epsilon_0=0.3,
            alpha=0.08,  # Slower learning for reasoning
            s=0.12,
            rho=0.2,  # Start exploratory
            x_pred=self.identity_emb.copy()
        )
        
        # Store all clues in ledger initially
        for i, clue in enumerate(CLUES):
            clue_emb = await self.provider.embed(clue)
            clue_emb /= np.linalg.norm(clue_emb)
            
            entry = LedgerEntry(
                timestamp=time.time() - 1000 + i,  # Earlier timestamps
                state_vector=self.identity_emb.copy(),
                action_id=f"clue_{i}",
                observation_embedding=clue_emb,
                outcome_embedding=clue_emb,
                prediction_error=0.0,
                context_embedding=clue_emb,
                rigidity_at_time=0.2,
                metadata={'type': 'clue', 'content': clue}
            )
            self.ledger.add_entry(entry)
        
        print(f"{C.GREEN}✓ Loaded {len(CLUES)} clues into ledger{C.RESET}")
        print(f"  Initial ρ: {self.state.rho:.3f}")
        
    async def retrieve_relevant(self, query: str) -> tuple:
        """Retrieve relevant entries based on semantic similarity."""
        query_emb = await self.provider.embed(query)
        query_emb /= np.linalg.norm(query_emb)
        if len(query_emb) != len(self.identity_emb):
            query_emb = query_emb[:len(self.identity_emb)]
        
        # Retrieve from ledger using embedding similarity
        entries = self.ledger.retrieve(query_emb, k=10, min_score=0.1)
        reflections = self.ledger.retrieve_reflections(query_emb, k=5, min_score=0.1)
        
        # Format deductions
        deduction_lines = []
        for entry in entries:
            if entry.metadata.get('type') == 'clue':
                deduction_lines.append(f"CLUE: {entry.metadata['content']}")
            elif entry.metadata.get('type') == 'deduction':
                deduction_lines.append(f"DEDUCTION: {entry.metadata['content']}")
        
        # Format reflections
        reflection_lines = []
        for ref in reflections:
            reflection_lines.append(f"- {ref.reflection_text}")
        
        return (
            "\n".join(deduction_lines) if deduction_lines else "[No relevant knowledge yet]",
            "\n".join(reflection_lines) if reflection_lines else "[No reflections yet]",
            query_emb
        )
    
    async def reason_iteration(self, iteration: int) -> bool:
        """One iteration of reasoning."""
        print(f"\n{C.BOLD}{'─' * 60}{C.RESET}")
        print(f"{C.BLUE}Iteration {iteration}{C.RESET}")
        
        # Build query based on current focus
        if iteration == 1:
            query = "What do we know about houses, colors, nationalities, and pets?"
        elif iteration < 5:
            query = "zebra pet owner nationality houses"
        else:
            query = f"zebra owner deduction {' '.join(self.deductions[-2:]) if self.deductions else ''}"
        
        # Retrieve from ledger using embedding similarity
        deductions_str, reflections_str, query_emb = await self.retrieve_relevant(query)
        
        # COGNITION → PARAMS
        # Temperature inversely proportional to rigidity
        # High ρ = more certain = lower temperature
        temperature = 0.2 + 0.6 * (1 - self.state.rho)
        confidence = 1 - self.state.rho
        
        print(f"{C.DIM}  ρ: {self.state.rho:.3f} → temperature: {temperature:.2f}{C.RESET}")
        
        # Build prompt with accumulated knowledge
        system = self.SYSTEM_TEMPLATE.format(
            iteration=iteration,
            confidence=confidence,
            rho=self.state.rho,
            deductions=deductions_str,
            reflections=reflections_str
        )
        
        prompt = "Based on all the clues and your deductions, what is your next logical inference? Who might own the zebra?"
        
        # Generate response with cognition-adjusted params
        print(f"{C.MAGENTA}[REASONING]{C.RESET}")
        response = ""
        try:
            async for token in self.provider.stream(
                prompt,
                system_prompt=system,
                temperature=temperature,  # COGNITION FEEDS PARAMS
                max_tokens=250
            ):
                if not token.startswith("__THOUGHT__"):
                    print(token, end="", flush=True)
                    response += token
        except Exception as e:
            print(f"[Error: {e}]")
            response = "I need to continue analyzing the clues."
        
        print()
        
        # Check for answer
        if "ANSWER:" in response.upper() and ANSWER_CHECK in response.lower():
            self.solved = True
            print(f"\n{C.GREEN}★ CORRECT! The Japanese owns the zebra.{C.RESET}")
        
        # Embed the response for similarity and surprise
        try:
            response_emb = await self.provider.embed(response)
            response_emb /= (np.linalg.norm(response_emb) + 1e-9)
            if len(response_emb) != len(self.state.x_pred):
                response_emb = response_emb[:len(self.state.x_pred)]
        except:
            response_emb = self.state.x_pred.copy()
        
        # Compute surprise
        epsilon = np.linalg.norm(self.state.x_pred - response_emb)
        
        # Update rigidity (cognition update)
        rho_before = self.state.rho
        self.state.update_rigidity(epsilon)
        self.rigidity.update(epsilon)
        delta = self.state.rho - rho_before
        
        print(f"\n{C.DIM}  Surprise: {epsilon:.3f} | Δρ: {delta:+.3f} → ρ: {self.state.rho:.3f}{C.RESET}")
        
        # Store deduction in ledger
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=self.state.x.copy(),
            action_id=f"deduction_{iteration}",
            observation_embedding=query_emb,
            outcome_embedding=response_emb,
            prediction_error=epsilon,
            context_embedding=query_emb,
            rigidity_at_time=self.state.rho,
            was_successful=self.solved,
            metadata={
                'type': 'deduction',
                'iteration': iteration,
                'content': response[:200]
            }
        )
        self.ledger.add_entry(entry)
        
        # Generate reflection
        if epsilon > 0.4:  # High surprise = needs reflection
            reflect_prompt = f"You reasoned: {response[:300]}\n\nWhat key insight should you remember for future iterations?"
            reflection = await self.provider.complete(
                reflect_prompt,
                system_prompt="State one key insight concisely.",
                max_tokens=50
            )
            reflection = reflection.strip()[:100]
            
            ref_entry = ReflectionEntry(
                timestamp=time.time(),
                task_intent="logic_puzzle",
                situation_embedding=query_emb,
                reflection_text=reflection,
                prediction_error=epsilon,
                outcome_success=self.solved
            )
            self.ledger.add_reflection(ref_entry)
            print(f"{C.YELLOW}  Reflection: {reflection}{C.RESET}")
        
        # Update state for next iteration
        self.state.x_pred = response_emb
        self.state.x = (1 - 0.1) * self.state.x + 0.1 * response_emb
        
        self.deductions.append(response[:100])
        
        return self.solved
    
    async def run(self, max_iterations: int = 8):
        """Run the logic solver."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}  LOGIC PUZZLE: WHO OWNS THE ZEBRA?{C.RESET}")
        print(f"{C.BOLD}{'═' * 60}{C.RESET}")
        
        print(f"\n{C.CYAN}Clues:{C.RESET}")
        for clue in CLUES[:5]:
            print(f"  • {clue}")
        print(f"  ... and {len(CLUES) - 5} more clues in ledger")
        
        for i in range(1, max_iterations + 1):
            solved = await self.reason_iteration(i)
            if solved:
                break
            time.sleep(0.2)
        
        # Summary
        self.display_summary()
    
    def display_summary(self):
        """Show summary."""
        print(f"\n\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}  SUMMARY{C.RESET}")
        print(f"{C.BOLD}{'═' * 60}{C.RESET}")
        
        stats = self.ledger.get_statistics()
        
        status = f"{C.GREEN}SOLVED{C.RESET}" if self.solved else f"{C.YELLOW}UNSOLVED{C.RESET}"
        print(f"\n  Result: {status}")
        print(f"  Ledger entries: {stats.get('current_entries', 0)}")
        print(f"  Reflections: {stats.get('current_reflections', 0)}")
        print(f"  Final ρ: {self.state.rho:.3f}")
        
        diag = self.rigidity.get_diagnostic()
        print(f"  Peak fast ρ: {diag['peak_fast']:.3f}")
        
        print(f"\n{C.CYAN}Key:{C.RESET}")
        print(f"  - Cognition (ρ) fed into LLM temperature")
        print(f"  - Nomic embeddings used for ledger retrieval")
        print(f"  - Each iteration retrieved relevant clues + deductions")


async def main():
    solver = LogicSolver()
    await solver.run(max_iterations=8)


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.DIM}Terminated.{C.RESET}")
