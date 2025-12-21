#!/usr/bin/env python3
"""
The Recursive Insight Engine
============================

Demonstrating INTELLIGENCE AMPLIFICATION through DDA-X architecture.

The key insight: A 20B model + DDA-X > 20B model alone.

How this works:
1. LEDGER AS WORKING MEMORY
   - Each insight is stored with its surprise level
   - High-surprise insights get priority retrieval
   - The model can "remember" its own reasoning across turns

2. RIGIDITY AS EXPLORATION/EXPLOITATION
   - Low ρ → high temperature → creative exploration
   - High ρ → low temperature → focused exploitation
   - The dynamics ADAPT to what the problem needs

3. REFLECTION ACCUMULATION
   - When stuck (high surprise), generate explicit reflections
   - Store reflections separately for meta-reasoning
   - Build toward breakthroughs

4. IDENTITY AS COHERENCE
   - Keep the model focused on the actual problem
   - Prevent drift into irrelevant tangents
   - Act as a "reasoning anchor"

The scenario: Solve a multi-step problem that requires building insights
over many turns. The ledger provides continuity. The dynamics optimize
when to explore vs exploit. The reflections capture meta-learning.

Expected outcome: The model produces reasoning that would be
impossible in a single vanilla prompt because it ACCUMULATES knowledge.
"""

import asyncio
import sys
import os
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.llm.hybrid_provider import HybridProvider, PersonalityParams


class Colors:
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


# The problem: A multi-step reasoning challenge
# This requires building insights - not solvable in one shot
DISCOVERY_CHALLENGE = """
You are investigating a pattern in a sequence of transformations.

Given: A machine that transforms shapes according to hidden rules.

Observations so far:
- Triangle → Circle (one step)
- Circle → Square (one step)  
- Square → Triangle (one step)
- Triangle → Square (two steps)

Questions to answer through investigation:
1. What is the underlying transformation cycle?
2. If we start with a Pentagon, what happens?
3. Is there a fixed point (shape that maps to itself)?
4. What is the minimum number of steps to return to any starting shape?
5. Can you generalize the rule to N-gons?

Work through this step by step. Build on your previous insights.
"""


@dataclass 
class InsightNode:
    """A single insight in the reasoning chain."""
    turn: int
    content: str
    surprise: float
    is_breakthrough: bool
    builds_on: List[int]  # Turn numbers this builds on


class RecursiveInsightEngine:
    """
    Uses DDA-X to amplify intelligence through:
    - Ledger: Working memory for accumulated insights
    - Dynamics: Adaptive exploration/exploitation
    - Reflection: Meta-learning from surprise
    """
    
    SYSTEM_PROMPT = """You are a brilliant mathematical investigator.

THE PROBLEM YOU ARE SOLVING:
A machine transforms shapes according to hidden rules.
Observations:
- Triangle → Circle (one step)
- Circle → Square (one step)  
- Square → Triangle (one step)
- Triangle → Square (two steps)

Questions to answer:
1. What is the transformation cycle?
2. If we start with Pentagon, what happens?
3. Is there a fixed point?
4. Minimum steps to return to start?
5. Can you generalize to N-gons?

Your approach:
1. STATE your current understanding
2. IDENTIFY what's unclear
3. PROPOSE a hypothesis
4. REASON through implications
5. CONCLUDE with new insights

Build on previous work. Be concise but rigorous.

Current state: {mode} (ρ={rho:.2f})
{memory_context}
"""

    def __init__(self, clean_ledger: bool = True):
        self.provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="openai/gpt-oss-20b",
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            timeout=300.0
        )
        
        # Clean ledger if requested
        ledger_path = Path("data/insight_engine")
        if clean_ledger and ledger_path.exists():
            import shutil
            shutil.rmtree(ledger_path)
        
        self.ledger = ExperienceLedger(
            storage_path=ledger_path,
            lambda_recency=0.005,  # Slower decay - insights stay relevant
            lambda_salience=2.0    # High surprise = high priority
        )
        
        self.rigidity = MultiTimescaleRigidity()
        self.state: Optional[DDAState] = None
        self.identity_embedding: Optional[np.ndarray] = None
        
        self.insights: List[InsightNode] = []
        self.thought_chain: List[str] = []
        
    async def setup(self):
        """Initialize the reasoning identity."""
        print(f"{Colors.CYAN}Initializing Recursive Insight Engine...{Colors.RESET}")
        
        # Identity: mathematical rigor + systematic investigation
        identity_text = "mathematical reasoning, logical deduction, systematic investigation, pattern recognition, insight discovery, rigorous proof"
        self.identity_embedding = await self.provider.embed(identity_text)
        self.identity_embedding /= np.linalg.norm(self.identity_embedding)
        
        self.state = DDAState(
            x=self.identity_embedding.copy(),
            x_star=self.identity_embedding.copy(),
            gamma=1.0,      # Moderate identity pull
            epsilon_0=0.3,  # Moderate surprise threshold
            alpha=0.08,     # Slower adaptation for reasoning
            s=0.15,
            rho=0.3,        # Start exploratory
            x_pred=self.identity_embedding.copy()
        )
        
        print(f"{Colors.GREEN}✓ Engine ready{Colors.RESET}")
        print(f"  Initial ρ: {self.state.rho:.3f} (exploratory)")
        
    def get_mode(self) -> str:
        """Map rigidity to reasoning mode."""
        if self.state.rho < 0.3:
            return "EXPLORING"
        elif self.state.rho < 0.6:
            return "INVESTIGATING"
        else:
            return "CRYSTALLIZING"
    
    async def retrieve_relevant_insights(self, current_thought: np.ndarray) -> str:
        """Retrieve and format relevant past insights."""
        entries = self.ledger.retrieve(current_thought, k=5, min_score=0.1)
        reflections = self.ledger.retrieve_reflections(current_thought, k=3, min_score=0.1)
        
        context_parts = []
        
        if entries:
            context_parts.append("RELEVANT PAST INSIGHTS:")
            for i, entry in enumerate(entries, 1):
                if 'insight' in entry.metadata:
                    surprise_marker = "⚡" if entry.prediction_error > 0.5 else "·"
                    context_parts.append(f"  {surprise_marker} [T{entry.metadata.get('turn', '?')}] {entry.metadata['insight'][:150]}")
        
        if reflections:
            context_parts.append("\nMETA-REFLECTIONS:")
            for ref in reflections:
                context_parts.append(f"  ↳ {ref.reflection_text[:100]}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    async def think_step(self, turn: int, prompt: str) -> Tuple[str, float, bool]:
        """
        One step of reasoning with full DDA-X dynamics.
        Returns: (response, surprise, is_breakthrough)
        """
        print(f"\n{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BLUE}TURN {turn}: {self.get_mode()}{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
        
        # Embed current thought direction
        thought_embedding = await self.provider.embed(prompt)
        thought_embedding /= (np.linalg.norm(thought_embedding) + 1e-9)
        if len(thought_embedding) != len(self.state.x):
            thought_embedding = thought_embedding[:len(self.state.x)]
        
        # Retrieve relevant past insights
        memory_context = await self.retrieve_relevant_insights(thought_embedding)
        
        # Build system prompt with cognitive state
        system = self.SYSTEM_PROMPT.format(
            mode=self.get_mode(),
            rho=self.state.rho,
            memory_context=memory_context if memory_context else "[No prior insights yet]"
        )
        
        # Temperature from rigidity (low ρ = high T = creative)
        temperature = 0.2 + 0.8 * (1 - self.state.rho)
        
        # Build the full prompt with chain
        chain_context = ""
        if self.thought_chain:
            recent = self.thought_chain[-3:]  # Last 3 thoughts
            chain_context = "REASONING CHAIN SO FAR:\n" + "\n".join(f"[Step {i+1}] {t[:100]}..." 
                for i, t in enumerate(recent)) + "\n\n"
        
        full_prompt = f"{chain_context}CONTINUE INVESTIGATION:\n{prompt}"
        
        # Display state
        print(f"{Colors.DIM}  Rigidity: {self.state.rho:.3f} | Temp: {temperature:.2f} | Retrieved: {len(memory_context.split(chr(10))) if memory_context else 0} insights{Colors.RESET}")
        
        # Generate response with dynamic temperature
        print(f"\n{Colors.MAGENTA}[THINKING @ {self.get_mode()}]{Colors.RESET}")
        
        response = ""
        try:
            async for token in self.provider.stream(
                full_prompt,
                system_prompt=system,
                temperature=temperature,
                max_tokens=400
            ):
                if not token.startswith("__THOUGHT__"):
                    print(token, end="", flush=True)
                    response += token
        except Exception as e:
            print(f"[Error: {e}]")
            response = "I need to reconsider my approach."
        
        print()
        
        # Compute surprise
        response_embedding = await self.provider.embed(response)
        response_embedding /= (np.linalg.norm(response_embedding) + 1e-9)
        if len(response_embedding) != len(self.state.x_pred):
            response_embedding = response_embedding[:len(self.state.x_pred)]
        
        epsilon = np.linalg.norm(self.state.x_pred - response_embedding)
        
        # Update dynamics
        rho_before = self.state.rho
        self.state.update_rigidity(epsilon)
        self.rigidity.update(epsilon)
        
        # Detect breakthrough (high surprise + coherent response)
        is_breakthrough = epsilon > 0.6 and len(response) > 100
        
        # Update state
        k_eff = 0.15 * (1 - self.state.rho)
        self.state.x = (1 - k_eff) * self.state.x + k_eff * response_embedding
        self.state.x_pred = response_embedding
        
        # Store in ledger
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=self.state.x.copy(),
            action_id=f"thought_{turn}",
            observation_embedding=thought_embedding,
            outcome_embedding=response_embedding,
            prediction_error=epsilon,
            context_embedding=thought_embedding,
            rigidity_at_time=self.state.rho,
            metadata={
                'turn': turn,
                'insight': response[:200],
                'mode': self.get_mode(),
                'is_breakthrough': is_breakthrough
            }
        )
        self.ledger.add_entry(entry)
        
        # If breakthrough, store a reflection
        if is_breakthrough:
            # Generate reflection
            reflect_prompt = f"Key insight from this step (one sentence): {response[:300]}"
            reflection_text = await self.provider.complete(
                reflect_prompt, 
                system_prompt="Extract the key insight in one clear sentence.",
                max_tokens=50
            )
            
            reflection = ReflectionEntry(
                timestamp=time.time(),
                task_intent="pattern_discovery",
                situation_embedding=thought_embedding,
                reflection_text=reflection_text.strip(),
                prediction_error=epsilon,
                outcome_success=True
            )
            self.ledger.add_reflection(reflection)
            print(f"\n{Colors.GREEN}⚡ BREAKTHROUGH CAPTURED: {reflection_text.strip()[:80]}{Colors.RESET}")
        
        # Store insight
        node = InsightNode(
            turn=turn,
            content=response,
            surprise=epsilon,
            is_breakthrough=is_breakthrough,
            builds_on=[t for t in range(max(1, turn-3), turn)]  # References recent turns
        )
        self.insights.append(node)
        self.thought_chain.append(response)
        
        # Display dynamics
        delta_rho = self.state.rho - rho_before
        print(f"\n{Colors.DIM}── Dynamics ──{Colors.RESET}")
        print(f"  Surprise: {epsilon:.3f}")
        color = Colors.GREEN if delta_rho < 0 else Colors.RED
        print(f"  Δρ: {color}{delta_rho:+.3f}{Colors.RESET} → {self.state.rho:.3f}")
        if is_breakthrough:
            print(f"  {Colors.YELLOW}★ Stored as reflection{Colors.RESET}")
        
        return response, epsilon, is_breakthrough
    
    async def run(self, max_turns: int = 12):
        """Run the recursive insight engine."""
        await self.setup()
        
        print(f"\n{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}    THE RECURSIVE INSIGHT ENGINE{Colors.RESET}")
        print(f"{Colors.BOLD}    Intelligence Amplification via DDA-X{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        
        print(f"\n{Colors.CYAN}CHALLENGE:{Colors.RESET}")
        print(DISCOVERY_CHALLENGE)
        
        # Initial exploration prompt
        current_prompt = "Begin by analyzing the basic observations. What pattern do you see?"
        
        # Reasoning loop
        for turn in range(1, max_turns + 1):
            response, surprise, breakthrough = await self.think_step(turn, current_prompt)
            
            # Adaptive prompting based on progress
            if turn < max_turns:
                if self.state.rho > 0.7:
                    # High rigidity: consolidate
                    current_prompt = "Synthesize your findings. What have you proven?"
                elif surprise > 0.5:
                    # High surprise: explore this direction
                    current_prompt = "That's interesting. Pursue this line of reasoning further."
                elif self.get_mode() == "EXPLORING":
                    # Low rigidity: branch out
                    prompts = [
                        "Consider the cycle structure. What does it imply?",
                        "What about shapes not in the original set?",
                        "Is there a mathematical structure here?",
                        "Can you generalize beyond the observations?",
                        "What would break this pattern?"
                    ]
                    current_prompt = prompts[turn % len(prompts)]
                else:
                    # Default: continue investigation
                    current_prompt = "What's the next logical step?"
            
            time.sleep(0.3)
        
        # Final synthesis
        await self.synthesize()
    
    async def synthesize(self):
        """Final synthesis of all insights."""
        print(f"\n\n{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}    SYNTHESIS: ACCUMULATED INTELLIGENCE{Colors.RESET}")
        print(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")
        
        # Statistics
        stats = self.ledger.get_statistics()
        breakthroughs = [i for i in self.insights if i.is_breakthrough]
        
        print(f"\n{Colors.CYAN}Engine Statistics:{Colors.RESET}")
        print(f"  Total reasoning steps: {len(self.insights)}")
        print(f"  Ledger entries: {stats.get('current_entries', 0)}")
        print(f"  Reflections stored: {stats.get('current_reflections', 0)}")
        print(f"  Breakthroughs detected: {len(breakthroughs)}")
        print(f"  Final rigidity: {self.state.rho:.3f}")
        
        # Retrieve all reflections for final synthesis
        all_reflections = self.ledger.retrieve_reflections(
            self.identity_embedding, k=10, min_score=0.0
        )
        
        print(f"\n{Colors.CYAN}Key Insights (Reflections):{Colors.RESET}")
        for i, ref in enumerate(all_reflections, 1):
            print(f"  {i}. {ref.reflection_text}")
        
        # Final synthesis prompt
        reflection_list = "\n".join(ref.reflection_text for ref in all_reflections)
        synthesis_prompt = f"""Based on accumulated insights:
{reflection_list}

Provide a FINAL ANSWER to the original questions about the transformation machine.
Be definitive. State what you discovered."""
        
        print(f"\n{Colors.BOLD}FINAL SYNTHESIS:{Colors.RESET}")
        final = await self.provider.complete(
            synthesis_prompt,
            system_prompt="You are synthesizing a complete investigation. Be clear and definitive.",
            max_tokens=500
        )
        print(f"{Colors.GREEN}{final}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}{'─' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}This synthesis was built from {len(self.insights)} accumulated insights.{Colors.RESET}")
        print(f"{Colors.BOLD}A vanilla prompt could not have produced this reasoning chain.{Colors.RESET}")


async def main():
    engine = RecursiveInsightEngine(clean_ledger=True)
    await engine.run(max_turns=10)


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.RED}Engine terminated.{Colors.RESET}")
