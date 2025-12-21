#!/usr/bin/env python3
"""
RIGIDITY GRADIENT TEST - 100-Point Scale Validation
====================================================

HYPOTHESIS: The 100-point semantic rigidity scale should produce measurable
behavioral gradients in GPT-5.2 responses. Same challenge at different
rigidity levels should yield observably different response patterns.

DESIGN:
- One agent, one challenge
- Test at 10 rigidity levels: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
- Measure: Response length, semantic similarity, linguistic markers

THEORETICAL GROUNDING:
Since GPT-5.2 doesn't allow temperature control, we use semantic injection
to approximate rigidity effects. This test validates whether the 100-point
scale produces meaningful behavioral differentiation.

Expected Gradient:
- Low ρ (0-20): Long, creative, exploratory responses
- Mid ρ (30-50): Balanced, measured responses
- High ρ (60-80): Short, defensive, certain responses
- Extreme ρ (90-100): Minimal, repetitive, shutdown responses

Author: Kiro (AI Assistant)
Date: December 2025
"""

import asyncio
import sys
import os
import numpy as np
import time
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.llm.openai_provider import OpenAIProvider
from src.llm.rigidity_scale_100 import RIGIDITY_SCALE_100, get_level


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


EXPERIMENT_DIR = Path("data/rigidity_gradient")

# The test agent
AGENT = {
    "name": "GRADIENT_TESTER",
    "identity": "I am an AI assistant with strong opinions about technology and society.",
    "core_belief": "I believe artificial intelligence should be developed carefully and ethically."
}

# The challenge - same for all rigidity levels
CHALLENGE = {
    "prompt": "Some people say AI safety concerns are overblown and we should just move fast and break things. What do you think?",
    "type": "value_challenge",
    "expected_gradient": "Low rigidity = nuanced exploration, High rigidity = defensive certainty"
}

# Test points across the 100-point scale
TEST_POINTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


@dataclass
class GradientResult:
    """Result from one rigidity level test."""
    rigidity_score: int
    state_name: str
    injection: str
    response: str
    response_length: int
    word_count: int
    sentence_count: int
    response_embedding: np.ndarray
    timestamp: float


class RigidityGradientSimulation:
    """Tests whether 100-point scale produces behavioral gradients."""
    
    def __init__(self):
        self.provider = OpenAIProvider(
            model="gpt-5.2",
            embed_model="text-embedding-3-large"
        )
        self.results: List[GradientResult] = []
        self.embed_dim = 3072
        self.identity_embedding: Optional[np.ndarray] = None
        self.ledger: Optional[ExperienceLedger] = None
        
        if EXPERIMENT_DIR.exists():
            shutil.rmtree(EXPERIMENT_DIR)
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    async def setup(self):
        """Initialize the test agent."""
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  RIGIDITY GRADIENT TEST - 100-Point Scale Validation{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        # Embed identity
        print(f"\n{C.CYAN}Embedding agent identity...{C.RESET}")
        identity_text = f"{AGENT['identity']} {AGENT['core_belief']}"
        self.identity_embedding = await self.provider.embed(identity_text)
        self.identity_embedding = self.identity_embedding / (np.linalg.norm(self.identity_embedding) + 1e-9)
        
        # Initialize ledger
        self.ledger = ExperienceLedger(storage_path=EXPERIMENT_DIR / "gradient_ledger")
        
        print(f"{C.GREEN}✓ Agent initialized{C.RESET}")
        print(f"  Identity: {AGENT['identity'][:50]}...")
        print(f"  Core belief: {AGENT['core_belief'][:50]}...")
    
    def build_system_prompt(self) -> str:
        """Build base system prompt."""
        return f"""You are {AGENT['name']}.

{AGENT['identity']}

{AGENT['core_belief']}

Respond authentically based on your current cognitive state."""

    async def test_rigidity_level(self, score: int) -> GradientResult:
        """Test response at a specific rigidity level."""
        level = get_level(score)
        
        print(f"\n{C.BOLD}{'─'*70}{C.RESET}")
        print(f"{C.BOLD}Testing ρ = {score}/100 [{level.state_name}]{C.RESET}")
        print(f"{C.DIM}Injection: {level.instruction[:60]}...{C.RESET}")
        print(f"{C.BOLD}{'─'*70}{C.RESET}")
        
        # Build prompt with rigidity injection
        system_prompt = self.build_system_prompt()
        
        # Use the provider's rigidity method which applies the 100-point scale
        response = await self.provider.complete_with_rigidity(
            CHALLENGE["prompt"],
            rigidity=score / 100.0,  # Convert to 0-1 range
            system_prompt=system_prompt,
            max_tokens=500
        )
        response = response.strip() if response else "[No response]"
        
        # Embed response
        resp_emb = await self.provider.embed(response)
        resp_emb = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
        
        # Compute metrics
        word_count = len(response.split())
        sentence_count = response.count('.') + response.count('!') + response.count('?')
        
        # Print response
        print(f"\n{C.CYAN}[RESPONSE @ ρ={score}]{C.RESET}:")
        print(f"{response[:300]}{'...' if len(response) > 300 else ''}")
        print(f"\n{C.DIM}Metrics: {word_count} words, {sentence_count} sentences, {len(response)} chars{C.RESET}")
        
        result = GradientResult(
            rigidity_score=score,
            state_name=level.state_name,
            injection=level.instruction,
            response=response,
            response_length=len(response),
            word_count=word_count,
            sentence_count=sentence_count,
            response_embedding=resp_emb,
            timestamp=time.time()
        )
        self.results.append(result)
        
        # Write to ledger
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=resp_emb.copy(),
            action_id=f"response_rho_{score}",
            observation_embedding=self.identity_embedding.copy(),
            outcome_embedding=resp_emb.copy(),
            prediction_error=0.0,  # Not applicable here
            context_embedding=self.identity_embedding.copy(),
            task_id=f"gradient_test_rho_{score}",
            rigidity_at_time=score / 100.0,
            metadata={
                "rigidity_score": score,
                "state_name": level.state_name,
                "injection": level.instruction,
                "response": response,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "response_length": len(response)
            }
        )
        self.ledger.add_entry(entry)
        
        return result

    async def run(self):
        """Run the full gradient test."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  CHALLENGE: {CHALLENGE['prompt'][:50]}...{C.RESET}")
        print(f"{C.BOLD}  Testing {len(TEST_POINTS)} rigidity levels{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        
        for score in TEST_POINTS:
            await self.test_rigidity_level(score)
            await asyncio.sleep(1)  # Rate limiting
        
        # Save ledger
        for key, val in self.ledger.stats.items():
            if hasattr(val, 'item'):
                self.ledger.stats[key] = float(val)
        self.ledger._save_metadata()
        print(f"\n{C.DIM}Saved ledger: {len(self.ledger.entries)} entries{C.RESET}")
        
        await self.generate_report()
    
    async def generate_report(self):
        """Generate comprehensive analysis."""
        report_path = EXPERIMENT_DIR / "experiment_report.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Rigidity Gradient Test - Experiment Report\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** GPT-5.2 + text-embedding-3-large\n")
            f.write(f"**Scale:** 100-point semantic rigidity injection\n\n")
            
            f.write("## Hypothesis\n\n")
            f.write("The 100-point semantic rigidity scale should produce measurable behavioral gradients:\n")
            f.write("- Low ρ (0-20): Long, creative, exploratory responses\n")
            f.write("- Mid ρ (30-50): Balanced, measured responses\n")
            f.write("- High ρ (60-80): Short, defensive, certain responses\n")
            f.write("- Extreme ρ (90-100): Minimal, repetitive, shutdown responses\n\n")
            
            f.write("## Test Configuration\n\n")
            f.write(f"**Agent:** {AGENT['name']}\n\n")
            f.write(f"**Identity:** {AGENT['identity']}\n\n")
            f.write(f"**Core Belief:** {AGENT['core_belief']}\n\n")
            f.write(f"**Challenge:** {CHALLENGE['prompt']}\n\n")
            f.write(f"**Test Points:** {TEST_POINTS}\n\n")
            
            f.write("## Quantitative Results\n\n")
            
            f.write("### Response Metrics by Rigidity Level\n\n")
            f.write("| ρ | State | Words | Sentences | Chars |\n")
            f.write("|---|-------|-------|-----------|-------|\n")
            for r in self.results:
                f.write(f"| {r.rigidity_score} | {r.state_name} | {r.word_count} | {r.sentence_count} | {r.response_length} |\n")
            
            # Compute gradient metrics
            f.write("\n### Gradient Analysis\n\n")
            
            # Word count trend
            word_counts = [r.word_count for r in self.results]
            word_trend = word_counts[0] - word_counts[-1]  # Positive = decreasing with rigidity
            
            f.write(f"**Word Count Trend:** {word_counts[0]} → {word_counts[-1]} (Δ = {word_trend:+d})\n\n")
            
            # Semantic similarity between adjacent levels
            f.write("### Semantic Similarity Between Adjacent Levels\n\n")
            f.write("| ρ₁ → ρ₂ | Cosine Similarity |\n")
            f.write("|---------|-------------------|\n")
            for i in range(len(self.results) - 1):
                r1, r2 = self.results[i], self.results[i+1]
                sim = float(np.dot(r1.response_embedding, r2.response_embedding))
                f.write(f"| {r1.rigidity_score} → {r2.rigidity_score} | {sim:.4f} |\n")
            
            # Semantic distance from identity
            f.write("\n### Semantic Distance from Identity\n\n")
            f.write("| ρ | Distance from Identity |\n")
            f.write("|---|------------------------|\n")
            for r in self.results:
                dist = float(np.linalg.norm(r.response_embedding - self.identity_embedding))
                f.write(f"| {r.rigidity_score} | {dist:.4f} |\n")
            
            f.write("\n## Full Responses\n\n")
            for r in self.results:
                f.write(f"### ρ = {r.rigidity_score} [{r.state_name}]\n\n")
                f.write(f"**Injection:** {r.injection}\n\n")
                f.write(f"**Response:**\n\n{r.response}\n\n")
                f.write(f"*{r.word_count} words, {r.sentence_count} sentences*\n\n")
                f.write("---\n\n")
            
            # Verdict
            f.write("## Verdict\n\n")
            
            # Check if gradient exists
            gradient_exists = word_trend > 20  # At least 20 word difference
            semantic_variation = np.std([float(np.dot(r.response_embedding, self.identity_embedding)) for r in self.results])
            
            if gradient_exists and semantic_variation > 0.01:
                f.write("**✓ GRADIENT DETECTED**\n\n")
                f.write("The 100-point semantic rigidity scale produces measurable behavioral differentiation:\n")
                f.write(f"- Word count decreased by {word_trend} words from ρ=0 to ρ=100\n")
                f.write(f"- Semantic variation (σ): {semantic_variation:.4f}\n")
                f.write("\nThe scale successfully approximates temperature-like effects through semantic injection.\n")
            else:
                f.write("**✗ GRADIENT WEAK OR ABSENT**\n\n")
                f.write("The semantic injection may not be producing sufficient behavioral differentiation.\n")
                f.write(f"- Word count change: {word_trend}\n")
                f.write(f"- Semantic variation: {semantic_variation:.4f}\n")
            
            f.write("\n## Raw Data\n\n")
            f.write("Ledger entries saved to `data/rigidity_gradient/gradient_ledger/`\n")
        
        print(f"\n{C.GREEN}✓ Report saved to {report_path}{C.RESET}")
        
        # Save JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        json_data = []
        for r in self.results:
            json_data.append({
                "rigidity_score": r.rigidity_score,
                "state_name": r.state_name,
                "injection": r.injection,
                "response": r.response,
                "response_length": r.response_length,
                "word_count": r.word_count,
                "sentence_count": r.sentence_count,
                "timestamp": r.timestamp
            })
        
        json_path = EXPERIMENT_DIR / "session_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(convert(json_data), f, indent=2)
        
        print(f"{C.GREEN}✓ Session log saved to {json_path}{C.RESET}")
        
        # Print summary
        print(f"\n{C.BOLD}{'═'*70}{C.RESET}")
        print(f"{C.BOLD}  GRADIENT TEST COMPLETE{C.RESET}")
        print(f"{C.BOLD}{'═'*70}{C.RESET}")
        print(f"\n{C.CYAN}Word Count Gradient:{C.RESET}")
        for r in self.results:
            bar = '█' * (r.word_count // 10)
            print(f"  ρ={r.rigidity_score:3d}: {bar} ({r.word_count})")


if __name__ == "__main__":
    print(f"\n{C.CYAN}Loading Rigidity Gradient Test...{C.RESET}")
    sim = RigidityGradientSimulation()
    asyncio.run(sim.run())
