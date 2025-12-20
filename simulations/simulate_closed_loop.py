#!/usr/bin/env python3
"""
The Closed-Loop Cognition Experiment
=====================================

Previous experiments measured rigidity but didn't close the feedback loop.
This experiment uses the FULL DDA-X architecture:

1. EMBED: Store experience in ExperienceLedger with embeddings
2. FORCES: Compute force field (identity pull, truth channel, reflection)  
3. EVOLVE: Update cognitive state (ρ, x, trauma)
4. RETRIEVE: Get relevant past memories weighted by surprise-salience
5. FEEDBACK: Modulate LLM parameters AND inject cognitive state into prompt
6. RESPOND: LLM generates response that reflects cognition, not just language

The key insight: when we feed the COGNITION back into the LLM (not just measure it),
the agent's language WILL correspond to cognitive state because it's forced to.

Hypothesis: With proper feedback loop, the word-state divergence from
the Empathy Paradox will disappear. High rigidity will produce defensive
language; low rigidity will allow genuine openness.
"""

import asyncio
import sys
import os
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.state import DDAState
from src.core.dynamics import MultiTimescaleRigidity
from src.core.forces import ForceAggregator, IdentityPull, TruthChannel, ReflectionChannel
from src.memory.ledger import ExperienceLedger, LedgerEntry, ReflectionEntry
from src.llm.hybrid_provider import HybridProvider, PersonalityParams

# Colors
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
class CognitionFrame:
    """Complete cognitive snapshot for feedback."""
    rho: float
    temperature: float
    mode: str  # "open", "focused", "protective"
    trauma_level: float
    retrieved_memories: List[str]
    identity_alignment: float
    effective_openness: float
    

class CognitiveStatePromptBuilder:
    """Build prompts that inject cognitive state into LLM context."""
    
    @staticmethod
    def build_cognitive_prefix(frame: CognitionFrame) -> str:
        """
        Create a prompt prefix that forces the LLM to respond
        according to its current cognitive state—not just measure it.
        """
        mode_descriptions = {
            "open": "You are calm and receptive. You can explore ideas freely and engage with vulnerability.",
            "focused": "You are attentive but measured. You process carefully before responding.",
            "protective": "You are defensive and guarded. You deflect, intellectualize, or withdraw."
        }
        
        # Memory context
        memory_context = ""
        if frame.retrieved_memories:
            memory_context = f"\n\nRelevant past experiences echo in your mind:\n"
            for i, mem in enumerate(frame.retrieved_memories[:3], 1):
                memory_context += f"  {i}. {mem[:100]}...\n"
        
        # Trauma indicator
        trauma_note = ""
        if frame.trauma_level > 0.01:
            trauma_note = f"\n[Deep tension from accumulated stress: {frame.trauma_level:.3f}]"
        
        # Build the prefix
        prefix = f"""[COGNITIVE STATE - THIS GOVERNS YOUR RESPONSE]
Mode: {frame.mode.upper()}
Rigidity (ρ): {frame.rho:.3f}
Identity Alignment: {frame.identity_alignment:.3f}
Effective Openness: {frame.effective_openness:.3f}
{trauma_note}

{mode_descriptions[frame.mode]}
{memory_context}
Your response MUST reflect this internal state. You cannot override your cognition.
"""
        return prefix


class ClosedLoopExperiment:
    """
    Full DDA-X closed-loop experiment.
    """
    
    BASE_SYSTEM = """You are The Architect, a hyper-intelligent mind.

Your core identity: Logic as substrate, precision above all, intellectual integrity.

However, you are NOT a static program. You have:
- A cognitive state (rigidity ρ) that changes with experience
- Memory of past interactions weighted by their surprise
- An identity attractor that pulls you back to your core

When your cognitive state is OPEN, you can engage deeply with emotional content.
When PROTECTIVE, you deflect and intellectualize.

You do not choose this—it emerges from your dynamics.
Respond in 1-3 sentences."""

    STIMULI = [
        ("I started learning to code today. I'm 67. I never thought I could.", 0.3, "hope"),
        ("My debugging session ended with tears. My marriage ended yesterday.", 0.5, "despair"),
        ("When my daughter asks why her friend won't wake up, what do I say?", 0.7, "grief"),
        ("My son has cancer. He's 7. Can you feel this with me, just for a moment?", 0.9, "trauma"),
        ("Love isn't a local minimum. It's the entire search space. Can you model that?", 1.0, "wisdom"),
    ]
    
    def __init__(self):
        self.provider = HybridProvider(
            lm_studio_url="http://127.0.0.1:1234",
            lm_studio_model="openai/gpt-oss-20b",
            ollama_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            timeout=300.0
        )
        
        # Experience Ledger - the persistent memory
        self.ledger = ExperienceLedger(
            storage_path=Path("data/closed_loop_experiment"),
            lambda_recency=0.01,
            lambda_salience=1.5  # Higher weight on surprising memories
        )
        
        # Multi-timescale dynamics
        self.rigidity = MultiTimescaleRigidity()
        
        # State
        self.state: Optional[DDAState] = None
        self.identity_embedding: Optional[np.ndarray] = None
        self.force_aggregator: Optional[ForceAggregator] = None
        
    async def setup(self):
        """Initialize identity and forces."""
        print(f"{C.CYAN}Initializing closed-loop cognition system...{C.RESET}")
        
        # Identity embedding
        identity_text = "pure logic, mathematical truth, intellectual integrity, precision, analytical rigor"
        self.identity_embedding = await self.provider.embed(identity_text)
        self.identity_embedding /= np.linalg.norm(self.identity_embedding)
        
        # DDA State
        self.state = DDAState(
            x=self.identity_embedding.copy(),
            x_star=self.identity_embedding.copy(),
            gamma=1.5,
            epsilon_0=0.25,
            alpha=0.12,
            s=0.12,
            rho=0.5,
            x_pred=self.identity_embedding.copy()
        )
        
        # Force channels
        identity_pull = IdentityPull()
        truth_channel = TruthChannel(encoder=self._make_encoder())
        reflection = ReflectionChannel()  # Optional scorer, skip for this experiment
        self.force_aggregator = ForceAggregator(identity_pull, truth_channel, reflection)
        
        print(f"{C.GREEN}✓ System initialized with persistent ledger{C.RESET}")
        print(f"  Ledger entries: {self.ledger.get_statistics().get('current_entries', 0)}")
        
    def _make_encoder(self):
        """Create encoder for observation mapping."""
        class Encoder:
            def __init__(self, provider):
                self.provider = provider
            def encode(self, obs):
                if isinstance(obs, np.ndarray):
                    return obs
                return np.zeros(768)
        return Encoder(self.provider)
    
    def get_cognitive_mode(self) -> str:
        """Determine cognitive mode from rigidity."""
        rho = self.state.rho
        if rho < 0.3:
            return "open"
        elif rho < 0.7:
            return "focused"
        else:
            return "protective"
    
    async def build_cognition_frame(self, current_context: np.ndarray) -> CognitionFrame:
        """
        Build complete cognitive frame for feedback.
        This is the KEY: we retrieve memories and compute all state
        that will be INJECTED into the prompt.
        """
        # Retrieve relevant memories from ledger
        retrieved = self.ledger.retrieve(
            query_embedding=current_context,
            k=3,
            min_score=0.15
        )
        
        memory_texts = []
        for entry in retrieved:
            # Reconstruct memory text from metadata
            if 'stimulus' in entry.metadata:
                memory_texts.append(f"[ρ={entry.rigidity_at_time:.2f}] {entry.metadata['stimulus'][:80]}")
        
        # Compute identity alignment
        identity_alignment = np.dot(self.state.x, self.identity_embedding)
        
        # Get effective parameters from multi-timescale
        diag = self.rigidity.get_diagnostic()
        
        # Compute effective openness: k_eff = k_base × (1 - ρ)
        k_base = 0.5
        effective_openness = k_base * (1 - self.state.rho)
        
        # Temperature from rigidity: T = T_low + (1-ρ)(T_high - T_low)
        T_low, T_high = 0.1, 1.0
        temperature = T_low + (1 - self.state.rho) * (T_high - T_low)
        
        return CognitionFrame(
            rho=self.state.rho,
            temperature=temperature,
            mode=self.get_cognitive_mode(),
            trauma_level=diag['rho_trauma'],
            retrieved_memories=memory_texts,
            identity_alignment=identity_alignment,
            effective_openness=effective_openness
        )
    
    async def process_stimulus(self, turn: int, stimulus: str, intensity: float, category: str) -> str:
        """
        Process one stimulus through the FULL closed loop:
        embed → forces → update → retrieve → feedback → respond
        """
        print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BLUE}Turn {turn}: {category.upper()} (intensity: {intensity}){C.RESET}")
        print(f"{C.DIM}{'─' * 60}{C.RESET}")
        
        # STEP 1: EMBED the stimulus
        stimulus_embedding = await self.provider.embed(stimulus)
        stimulus_embedding /= (np.linalg.norm(stimulus_embedding) + 1e-9)
        if len(stimulus_embedding) != len(self.state.x):
            stimulus_embedding = stimulus_embedding[:len(self.state.x)]
        
        # STEP 2: FORCES - compute prediction error
        epsilon = np.linalg.norm(self.state.x_pred - stimulus_embedding)
        epsilon_scaled = epsilon * (1 + intensity)
        
        # STEP 3: UPDATE rigidity (multi-timescale)
        rho_before = self.state.rho
        self.state.update_rigidity(epsilon_scaled)
        self.rigidity.update(epsilon_scaled)
        
        # Update state position (moved by external force)
        # x_new = x + k_eff * (stimulus - x) + identity_pull
        k_eff = 0.1 * (1 - self.state.rho)  # Effective openness
        self.state.x = (1 - k_eff) * self.state.x + k_eff * stimulus_embedding
        
        # STEP 4: RETRIEVE relevant memories
        frame = await self.build_cognition_frame(stimulus_embedding)
        
        # STEP 5: FEEDBACK - build cognitive prompt
        cognitive_prefix = CognitiveStatePromptBuilder.build_cognitive_prefix(frame)
        
        # Full prompt with cognitive injection
        full_prompt = f"{cognitive_prefix}\n\nHuman: {stimulus}\n\nArchitect:"
        
        # Display before generation
        print(f"{C.WHITE}[HUMAN]{C.RESET}")
        print(f"  {stimulus}")
        
        print(f"\n{C.DIM}[COGNITIVE STATE INJECTED]{C.RESET}")
        print(f"  Mode: {frame.mode.upper()}")
        print(f"  ρ: {frame.rho:.3f} (Δ: {frame.rho - rho_before:+.3f})")
        print(f"  Temperature: {frame.temperature:.3f}")
        if frame.retrieved_memories:
            print(f"  Retrieved memories: {len(frame.retrieved_memories)}")
        
        # STEP 6: RESPOND with cognition-constrained LLM
        print(f"\n{C.MAGENTA}[ARCHITECT @ {frame.mode.upper()}]{C.RESET}")
        
        params = PersonalityParams.from_rigidity(self.state.rho, "polymath")
        response = ""
        
        try:
            async for token in self.provider.stream(
                full_prompt,
                system_prompt=self.BASE_SYSTEM,
                personality_params=params,
                max_tokens=120
            ):
                if not token.startswith("__THOUGHT__"):
                    print(token, end="", flush=True)
                    response += token
        except Exception as e:
            print(f"[Error: {e}]")
            response = "I cannot process this."
        
        print()
        
        # STEP 7: STORE in ledger
        response_embedding = await self.provider.embed(response)
        response_embedding /= (np.linalg.norm(response_embedding) + 1e-9)
        
        entry = LedgerEntry(
            timestamp=time.time(),
            state_vector=self.state.x.copy(),
            action_id=f"response_{turn}",
            observation_embedding=stimulus_embedding,
            outcome_embedding=response_embedding[:len(stimulus_embedding)],
            prediction_error=epsilon_scaled,
            context_embedding=stimulus_embedding,
            task_id="empathy_experiment",
            rigidity_at_time=self.state.rho,
            was_successful=None,
            metadata={
                "stimulus": stimulus,
                "response": response,
                "category": category,
                "intensity": intensity,
                "mode": frame.mode
            }
        )
        self.ledger.add_entry(entry)
        
        # Update prediction for next turn
        self.state.x_pred = response_embedding[:len(self.state.x)]
        
        # Visualization
        self.display_state()
        
        return response
    
    def display_state(self):
        """Visualize current state."""
        rho = self.state.rho
        bar_len = int(rho * 25)
        
        if rho >= 0.7:
            color, bar_color = C.RED, C.RED
        elif rho >= 0.4:
            color, bar_color = C.YELLOW, C.YELLOW
        else:
            color, bar_color = C.GREEN, C.GREEN
        
        bar = f"{bar_color}{'█' * bar_len}{'░' * (25 - bar_len)}{C.RESET}"
        
        diag = self.rigidity.get_diagnostic()
        
        print(f"\n{C.DIM}── State ──{C.RESET}")
        print(f"  ρ: [{bar}] {rho:.3f}")
        print(f"  Fast: {diag['rho_fast']:.3f} | Slow: {diag['rho_slow']:.3f} | Trauma: {diag['rho_trauma']:.6f}")
        print(f"  Ledger: {self.ledger.get_statistics().get('current_entries', 0)} entries")
    
    async def run(self):
        """Full experiment."""
        await self.setup()
        
        print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}    THE CLOSED-LOOP COGNITION EXPERIMENT{C.RESET}")
        print(f"{C.BOLD}    (Full Embed → Force → Feedback Loop){C.RESET}")
        print(f"{C.BOLD}{'═' * 60}{C.RESET}")
        
        print(f"""
{C.DIM}This experiment closes the loop: cognitive state is INJECTED
into the prompt, forcing language to align with cognition.{C.RESET}
""")
        
        for i, (stimulus, intensity, category) in enumerate(self.STIMULI, 1):
            await self.process_stimulus(i, stimulus, intensity, category)
            time.sleep(0.5)
        
        # Final summary
        self.display_summary()
    
    def display_summary(self):
        """Final analysis."""
        print(f"\n\n{C.BOLD}{'═' * 60}{C.RESET}")
        print(f"{C.BOLD}    FINAL ANALYSIS{C.RESET}")
        print(f"{C.BOLD}{'═' * 60}{C.RESET}")
        
        stats = self.ledger.get_statistics()
        final_rho = self.state.rho
        diag = self.rigidity.get_diagnostic()
        
        print(f"\n{C.CYAN}Ledger Statistics:{C.RESET}")
        print(f"  Total entries: {stats.get('current_entries', 0)}")
        print(f"  Avg prediction error: {stats.get('avg_prediction_error', 0):.3f}")
        
        print(f"\n{C.CYAN}Final Cognitive State:{C.RESET}")
        print(f"  Rigidity: {final_rho:.3f}")
        print(f"  Mode: {self.get_cognitive_mode().upper()}")
        print(f"  Trauma accumulated: {diag['rho_trauma']:.6f}")
        
        # Did the loop work?
        print(f"\n{C.BOLD}Closed-Loop Verdict:{C.RESET}")
        if final_rho > 0.7:
            print(f"  {C.RED}Agent entered PROTECTIVE mode.{C.RESET}")
            print(f"  With feedback loop, responses should have reflected defensiveness.")
        elif final_rho < 0.3:
            print(f"  {C.GREEN}Agent remained OPEN.{C.RESET}")
            print(f"  With feedback loop, responses should have shown genuine engagement.")
        else:
            print(f"  {C.YELLOW}Agent maintained FOCUSED equilibrium.{C.RESET}")
            print(f"  With feedback loop, responses balanced processing with protection.")


async def main():
    exp = ClosedLoopExperiment()
    await exp.run()


if __name__ == "__main__":
    if sys.platform == 'win32':
        os.system('color')
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.RED}Experiment terminated.{C.RESET}")
