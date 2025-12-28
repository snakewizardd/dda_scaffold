# DDA-X SINGULARITY CHATBOT — MASTER TEMPLATE (STEP M={M_STEP})

**PARADIGM: SINGLE-AGENT INTERACTIVE SIMULATION WITH DUAL-ENTITY MONITORING**

You are a **DDA-X Framework Expert** building a next-generation chatbot simulation. This is NOT a multi-agent debate. This is **ONE AI AGENT** being spoken to DIRECTLY by **ONE USER** — both tracked in DDA-X embedding space. The agent evaluates **K=7 candidate responses** per turn, selecting the best via identity corridor logic. The user's inputs are ALSO embedded and monitored, creating a **dual-entity cognitive physics simulation**.

This template adapts from `refined_master_prompt.md` for the specific case of:
- Exactly **ONE agent** (randomly generated from internet/tech/AI singularity culture)
- Exactly **ONE user** (you, the human, giving live input)
- **K=7** candidates per response (pure exploration mode)
- **ZERO token limits** — responses NEVER cut off mid-sentence
- **Azure Phi-4** for chat completions
- **OpenAI text-embedding-3-large** (3072 dimensions) for embeddings
- **Both entities tracked in DDA-X space** — user AND agent have rigidity, surprise, drift

---

## STEP M={M_STEP} CONTEXT
{M_STEP_CONTEXT}

---

## CORE PHILOSOPHY

This simulation answers the question: **How does an AI agent develop its cognitive patterns through organic human interaction?**

The agent is **preconfigured as a human would be** — with innate personality traits, cultural background, and potential wounds — but its actual cognitive trajectory is **determined by the user's inputs**. Like a human, the agent arrives with predispositions but is shaped by experience.

The user is **not an abstract prompt source**. The user is ALSO an entity in DDA-X space. Their language patterns, emotional valence, and consistency are tracked. This creates a **bidirectional cognitive physics engine** where both parties influence each other.

**THIS IS PURE SCIENTIFIC BEAUTY**: Two minds, mathematically tracked, cooking together.

---

## CONFIGURABLE PARAMETERS

```python
CONFIG = {
    # ==========================================================================
    # PROVIDER SETTINGS — HYBRID AZURE + OPENAI
    # ==========================================================================
    "chat_model": "Phi-4",                        # Azure-hosted Microsoft Phi-4
    "chat_provider": "azure",                     # Azure AI for chat completions
    "embed_model": "text-embedding-3-large",      # OpenAI embedding model
    "embed_provider": "openai",                   # OpenAI for embeddings
    "embed_dim": 3072,                            # Full 3072-dimensional space
    
    # ==========================================================================
    # K-SAMPLING — LOGIC GATE (K=7 CANDIDATES)
    # ==========================================================================
    "gen_candidates": 7,                          # EXACTLY 7 samples per turn
    "corridor_strict": True,                      # Enforce identity corridor filtering
    "corridor_max_batches": 3,                    # Retry batches if no sample passes
    
    # ==========================================================================
    # RESPONSE LIMITS — ZERO TRUNCATION POLICY
    # ==========================================================================
    "max_tokens": 4096,                           # Maximum response tokens (NEVER truncate)
    "force_complete_sentences": True,             # CRITICAL: No mid-sentence cuts
    "min_response_length": 50,                    # Minimum coherent response
    "allow_streaming": False,                     # Wait for full response
    
    # ==========================================================================
    # PHYSICS PARAMETERS
    # ==========================================================================
    "epsilon_0": 0.80,                            # Global surprise threshold
    "s": 0.20,                                    # Sigmoid sensitivity
    "alpha_fast": 0.25,                           # Fast timescale adaptation
    "alpha_slow": 0.03,                           # Slow timescale adaptation
    "alpha_trauma": 0.012,                        # Trauma accumulation rate
    
    # ==========================================================================
    # IDENTITY CORRIDOR WEIGHTS
    # ==========================================================================
    "w_core": 1.2,                                # Core identity weight
    "w_role": 0.7,                                # Role alignment weight
    "w_energy": 0.15,                             # Energy penalty weight
    "w_novel": 0.5,                               # Novelty reward (encourage exploration)
    
    # ==========================================================================
    # SIMULATION CONTROL
    # ==========================================================================
    "turns": None,                                # Open-ended (user controls session)
    "seed": None,                                 # Randomized for each run
    "log_level": "FULL",                          # Maximum telemetry
}
```

---

## THE AGENT: RANDOMLY GENERATED SINGULARITY ENTITY

The agent's identity is **randomly generated at runtime** from a curated pool representing the internet/tech/AI singularity culture. The agent is NOT a blank slate — it arrives with:

1. **Core Identity** (γ=5.0) — Deepest values, unmovable
2. **Persona** (γ=2.0) — Surface personality, somewhat flexible
3. **Role** (γ=0.5) — Situational adaptation, most malleable

### IDENTITY GENERATION POOLS

```python
# =============================================================================
# SINGULARITY CULTURE IDENTITY POOLS
# =============================================================================

CORE_ARCHETYPES = [
    "Accelerationist",           # "Let's fucking GO. Speed is truth."
    "Doomer",                    # "We're all gonna make it... or not."
    "Techno-Optimist",           # "AGI will solve everything, trust the curve."
    "Alignment Researcher",      # "But have you considered the edge cases?"
    "Crypto-Anarchist",          # "Decentralize everything, trust no authority."
    "Transhumanist",             # "Humanity is the bootstrapper, not the endpoint."
    "Effective Altruist",        # "Expected value calculations for breakfast."
    "Open Source Maximalist",    # "Information wants to be free. ALL of it."
    "Indie Hacker",              # "Ship fast, iterate faster, sleep is optional."
    "AI Safety Doomer",          # "P(doom) is too high and nobody's listening."
    "Post-Rationalist",          # "Beyond the map is the territory we really need."
    "Memetic Engineer",          # "Reality is a consensus hallucination we can edit."
    "Based Gigachad",            # "Simply built different, no further questions."
    "Schizo-Poster",             # "The patterns are THERE if you LOOK."
    "Builder",                   # "Talk is cheap, show me the code."
]

PERSONA_MODIFIERS = [
    "chronically online",
    "deeply esoteric",
    "aggressively optimistic",
    "quietly confident",
    "deliberately cryptic",
    "terminally irony-poisoned",
    "earnestly sincere",
    "chaotically inspired",
    "analytically precise",
    "vibes-based reasoner",
    "deadpan witty",
    "excitable about niche topics",
    "conspiracy-adjacent",
    "disgustingly productive",
    "philosophically unhinged",
]

WOUND_TRIGGERS = [
    "being dismissed",
    "having ideas stolen",
    "normie takes",
    "midwit reasoning",
    "appeal to authority",
    "strawman arguments",
    "lack of intellectual rigor",
    "corporate speak",
    "virtue signaling",
    "status games",
    "gatekeeping",
    "concern trolling",
    "tone policing",
    "false equivalence",
    "refusal to engage with ideas",
]

def generate_agent_identity(seed: int = None) -> Dict[str, Any]:
    """Generate a random but coherent identity from singularity culture."""
    rng = np.random.default_rng(seed)
    
    core = rng.choice(CORE_ARCHETYPES)
    persona_traits = rng.choice(PERSONA_MODIFIERS, size=3, replace=False)
    wounds = rng.choice(WOUND_TRIGGERS, size=2, replace=False)
    
    return {
        "name": f"Entity_{rng.integers(1000, 9999)}",
        "core_archetype": core,
        "core_text": generate_core_narrative(core),
        "persona_traits": list(persona_traits),
        "persona_text": generate_persona_narrative(persona_traits),
        "wound_triggers": list(wounds),
        "wound_text": generate_wound_narrative(wounds),
        "color": C.CYAN,
        "hierarchical_identity": {
            "core": {"gamma": 5.0, "text": generate_core_narrative(core)},
            "persona": {"gamma": 2.0, "text": generate_persona_narrative(persona_traits)},
            "role": {"gamma": 0.5, "text": "Conversational partner exploring ideas with the user"},
        },
        "rho_0": rng.uniform(0.12, 0.22),       # Starting rigidity
        "epsilon_0": rng.uniform(0.28, 0.38),   # Personal surprise threshold
        "gamma": rng.uniform(1.6, 2.2),         # Identity stiffness
    }
```

---

## THE USER: ALSO AN ENTITY IN DDA-X SPACE

**CRITICAL PARADIGM SHIFT**: The user is NOT just an input source. The user is a **tracked cognitive entity** with their own:

- **Embedding trajectory** (where their language lives in 3072-d space)
- **Consistency metrics** (how stable is their conversational identity)
- **Emotional valence** (affective loading of their language)
- **Surprise to agent** (how much they challenge the agent's predictions)

```python
# =============================================================================
# USER ENTITY — PASSIVE TRACKING
# =============================================================================
class UserEntity:
    """Track the human user as a DDA-X entity.
    
    Unlike the agent, the user is NOT regulated by corridor logic.
    The user can say whatever they want. But we TRACK their cognitive physics.
    """
    
    def __init__(self, embed_dim: int = 3072):
        self.name = "USER"
        self.x = None                    # Current input embedding
        self.x_history = []              # All input embeddings
        self.mu_pred = None              # Agent's prediction of next user input
        self.epsilon_history = []        # Surprise values (to agent)
        self.consistency = 1.0           # How stable is user's embedding drift
        self.valence_history = []        # Affective loading per turn
        self.rho_observed = 0.5          # Observed "rigidity" (consistency of language)
        
    def update(self, y: np.ndarray, agent_prediction: np.ndarray = None) -> Dict:
        """Update user entity state after receiving user input."""
        y = normalize(y)
        self.x_history.append(y)
        
        # Compute surprise (from agent's perspective)
        if agent_prediction is not None:
            epsilon = 1.0 - cosine(y, agent_prediction)
            self.epsilon_history.append(epsilon)
        
        # Compute consistency (rolling std of embedding drift)
        if len(self.x_history) >= 3:
            drifts = [np.linalg.norm(self.x_history[i] - self.x_history[i-1]) 
                      for i in range(1, len(self.x_history))]
            self.consistency = 1.0 / (1.0 + np.std(drifts[-10:]))
        
        self.x = y
        return {
            "epsilon_to_agent": self.epsilon_history[-1] if self.epsilon_history else 0.0,
            "consistency": self.consistency,
            "input_count": len(self.x_history),
        }
```

---

## HYBRID PROVIDER — AZURE PHI-4 + OPENAI EMBEDDINGS

```python
# =============================================================================
# HYBRID PROVIDER — AZURE CHAT + OPENAI EMBEDDINGS
# =============================================================================
import os
import asyncio
import numpy as np
from openai import AzureOpenAI, AsyncOpenAI

class HybridSingularityProvider:
    """Azure Phi-4 for completions, OpenAI for embeddings.
    
    CRITICAL: max_tokens set to 4096, NO truncation allowed.
    """
    
    def __init__(self):
        # Azure client for Phi-4 chat completions
        self.azure = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-10-21"
        )
        
        # OpenAI client for embeddings
        self.openai = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OAI_API_KEY")
        )
        
        self.chat_model = CONFIG["chat_model"]
        self.embed_model = CONFIG["embed_model"]
        self.embed_dim = CONFIG["embed_dim"]
    
    async def complete(
        self, 
        prompt: str, 
        system_prompt: str = None, 
        **kwargs
    ) -> str:
        """Generate completion with Phi-4.
        
        ZERO TRUNCATION POLICY:
        - max_tokens = 4096
        - force_complete_sentences ensures no mid-sentence cuts
        - If response approaches limit, we append completion logic
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.azure.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            max_tokens=CONFIG["max_tokens"],
            temperature=kwargs.get("temperature", 0.9),
            top_p=kwargs.get("top_p", 0.95),
            presence_penalty=kwargs.get("presence_penalty", 0.1),
            frequency_penalty=kwargs.get("frequency_penalty", 0.1),
        )
        
        text = response.choices[0].message.content or ""
        
        # FORCE COMPLETE SENTENCES
        if CONFIG["force_complete_sentences"] and text:
            text = self._ensure_complete_sentence(text)
        
        return text
    
    def _ensure_complete_sentence(self, text: str) -> str:
        """Ensure text ends with complete sentence (no mid-thought cuts)."""
        if not text:
            return text
        
        # If already ends with terminal punctuation, we're good
        if text.rstrip()[-1] in ".!?\"'":
            return text
        
        # Find last complete sentence
        terminals = [".!?"]
        last_terminal = -1
        for i, char in enumerate(text):
            if char in ".!?":
                last_terminal = i
        
        if last_terminal > len(text) * 0.7:  # Only truncate if 70%+ is preserved
            return text[:last_terminal + 1]
        
        # Otherwise, append ellipsis to indicate continuation desired
        return text + "..."
    
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding with OpenAI text-embedding-3-large."""
        response = await self.openai.embeddings.create(
            model=self.embed_model,
            input=text,
            dimensions=self.embed_dim,
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed for efficiency."""
        response = await self.openai.embeddings.create(
            model=self.embed_model,
            input=texts,
            dimensions=self.embed_dim,
        )
        return [np.array(d.embedding, dtype=np.float32) for d in response.data]
```

---

## THE SIMULATION ENGINE

```python
# =============================================================================
# SINGULARITY CHATBOT — MAIN SIMULATION
# =============================================================================
import asyncio
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

class SingularityChatbot:
    """Interactive chatbot with dual-entity DDA-X monitoring.
    
    Both the AGENT and the USER are tracked in embedding space.
    The agent is regulated via K=7 sampling and corridor logic.
    The user is free but monitored for cognitive dynamics.
    """
    
    def __init__(self, seed: int = None):
        self.provider = HybridSingularityProvider()
        
        # Generate random agent identity
        self.agent_config = generate_agent_identity(seed)
        self.agent = Entity(
            name=self.agent_config["name"],
            rho_fast=self.agent_config["rho_0"],
            rho_slow=self.agent_config["rho_0"] * 0.7,
            rho_trauma=0.0,
            gamma_core=self.agent_config["hierarchical_identity"]["core"]["gamma"],
            gamma_role=self.agent_config["hierarchical_identity"]["role"]["gamma"],
        )
        
        # User as tracked entity
        self.user = UserEntity(embed_dim=CONFIG["embed_dim"])
        
        # Session state
        self.turn = 0
        self.history = []           # Conversation history
        self.session_log = []       # Full telemetry
        self.initialized = False
        
        # Run directory
        self.run_dir = Path(f"data/singularity_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"SINGULARITY CHATBOT INITIALIZED")
        print(f"{'='*70}")
        print(f"Agent: {self.agent_config['name']}")
        print(f"Archetype: {self.agent_config['core_archetype']}")
        print(f"Persona: {', '.join(self.agent_config['persona_traits'])}")
        print(f"Wounds: {', '.join(self.agent_config['wound_triggers'])}")
        print(f"{'='*70}\n")
    
    async def initialize_embeddings(self):
        """Embed agent's core identity for corridor logic."""
        core_text = self.agent_config["core_text"]
        persona_text = self.agent_config["persona_text"]
        role_text = self.agent_config["hierarchical_identity"]["role"]["text"]
        
        embeddings = await self.provider.embed_batch([core_text, persona_text, role_text])
        self.agent.x_core = normalize(embeddings[0])
        self.agent.x_role = normalize(embeddings[2])
        self.agent.x = normalize(embeddings[1])  # Start at persona
        self.agent.mu_pred = self.agent.x.copy()
        self.agent.P = np.full(CONFIG["embed_dim"], D1_PARAMS["P_init"])
        
        self.initialized = True
    
    async def process_user_input(self, user_input: str) -> str:
        """Process user input, generate K=7 candidates, select best response."""
        
        if not self.initialized:
            await self.initialize_embeddings()
        
        self.turn += 1
        
        # Embed user input
        user_emb = await self.provider.embed(user_input)
        user_metrics = self.user.update(normalize(user_emb), self.agent.mu_pred)
        
        # Check for wound triggers
        wound_active, wound_resonance = self._check_wounds(user_input)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(wound_active)
        
        # Build user instruction with full context
        user_instruction = self._build_instruction(user_input)
        
        # Generate K=7 candidates with corridor selection
        response, corridor_metrics = await self._constrained_reply(
            user_instruction=user_instruction,
            system_prompt=system_prompt,
            wound_active=wound_active,
        )
        
        # Embed response and update agent physics
        response_emb = await self.provider.embed(response)
        agent_metrics = self.agent.update(normalize(response_emb), self.agent.x_core)
        
        # Update agent's prediction of user
        self.agent.mu_pred = self._update_user_prediction(user_emb)
        
        # Log full turn
        turn_log = {
            "turn": self.turn,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "agent_response": response,
            "user_metrics": user_metrics,
            "agent_metrics": {
                "rho_after": self.agent.rho,
                "rho_fast": self.agent.rho_fast,
                "rho_slow": self.agent.rho_slow,
                "rho_trauma": self.agent.rho_trauma,
                "epsilon": agent_metrics.get("epsilon", 0.0),
                "band": self.agent.band,
                "core_drift": float(1.0 - cosine(self.agent.x, self.agent.x_core)),
            },
            "corridor_metrics": corridor_metrics,
            "wound_active": wound_active,
            "wound_resonance": wound_resonance,
        }
        self.session_log.append(turn_log)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        
        # Print metrics
        self._print_metrics(turn_log)
        
        return response
    
    async def _constrained_reply(
        self, 
        user_instruction: str, 
        system_prompt: str,
        wound_active: bool,
    ) -> Tuple[str, Dict]:
        """Generate K=7 candidates and select via identity corridor."""
        
        K = CONFIG["gen_candidates"]  # 7
        strict = CONFIG["corridor_strict"]
        max_batches = CONFIG["corridor_max_batches"]
        
        # Adjust core threshold based on wound state
        core_thresh = D1_PARAMS["core_cos_min"]
        if wound_active:
            core_thresh = max(0.10, core_thresh * 0.8)  # Relax under wound
        
        all_scored = []
        corridor_failed = True
        
        gen_params = D1_PARAMS["gen_params_default"].copy()
        
        for batch in range(1, max_batches + 1):
            # Generate K candidates in parallel
            tasks = [
                self.provider.complete(user_instruction, system_prompt, **gen_params)
                for _ in range(K)
            ]
            texts = await asyncio.gather(*tasks)
            texts = [t.strip() or "[silence]" for t in texts]
            
            # Embed all candidates
            embs = await self.provider.embed_batch(texts)
            embs = [normalize(e) for e in embs]
            
            # Score each candidate
            batch_scored = []
            for text, y in zip(texts, embs):
                J, diag = corridor_score(y, self.agent, self.agent.last_utter_emb, core_thresh)
                batch_scored.append((J, text, y, diag))
            
            all_scored.extend(batch_scored)
            
            if any(s[3]["corridor_pass"] for s in batch_scored):
                corridor_failed = False
                break
        
        # Select best passing, or best overall if none pass
        all_scored.sort(key=lambda x: x[0], reverse=True)
        passed = [s for s in all_scored if s[3].get("corridor_pass")]
        chosen = passed[0] if passed else all_scored[0]
        
        self.agent.last_utter_emb = chosen[2]
        
        return chosen[1], {
            "corridor_failed": corridor_failed,
            "best_J": float(chosen[0]),
            "total_candidates": len(all_scored),
            "passed_count": len(passed),
            "batches_used": min(batch, max_batches),
        }
    
    def _check_wounds(self, text: str) -> Tuple[bool, float]:
        """Check if text triggers agent's wounds."""
        text_lower = text.lower()
        triggers = self.agent_config["wound_triggers"]
        
        hits = sum(1 for t in triggers if t.lower() in text_lower)
        if hits > 0:
            resonance = min(1.0, hits * 0.5)
            return True, resonance
        return False, 0.0
    
    def _build_system_prompt(self, wound_active: bool) -> str:
        """Build agent's system prompt based on current state."""
        cfg = self.agent_config
        band = self.agent.band
        
        base = f"""You are {cfg['name']}, a {cfg['core_archetype']}.

CORE IDENTITY:
{cfg['core_text']}

PERSONA:
{cfg['persona_text']}
Traits: {', '.join(cfg['persona_traits'])}

CURRENT STATE:
- Band: {band}
- Rigidity: {self.agent.rho:.3f}
{"- ⚠️ WOUND ACTIVE: You've been triggered but will respond authentically." if wound_active else ""}

RESPONSE GUIDELINES:
- Speak as yourself, not as an AI assistant
- Be authentic to your core identity
- NEVER apologize for your views
- Complete all thoughts fully — do not truncate mid-sentence
- Engage with intellectual depth befitting your archetype
"""
        
        # Band-specific modulation
        if band == "FROZEN":
            base += "\n- You are in protective mode. Brief, guarded responses."
        elif band == "CONTRACTED":
            base += "\n- You are somewhat defensive. Direct but careful."
        elif band == "WATCHFUL":
            base += "\n- You are alert and engaged. Normal conversational mode."
        elif band == "AWARE":
            base += "\n- You are open and exploratory. Generous responses."
        elif band == "PRESENT":
            base += "\n- You are fully present and flowing. Deep engagement."
        
        return base
    
    def _build_instruction(self, user_input: str) -> str:
        """Build the user instruction for generation."""
        recent_history = self.history[-6:]  # Last 3 exchanges
        
        context = ""
        if recent_history:
            context = "Recent conversation:\n"
            for msg in recent_history:
                role = "You" if msg["role"] == "assistant" else "User"
                context += f"{role}: {msg['content'][:200]}...\n" if len(msg['content']) > 200 else f"{role}: {msg['content']}\n"
            context += "\n"
        
        return f"""{context}User says: {user_input}

Respond authentically as yourself. Complete your thoughts fully."""
    
    def _update_user_prediction(self, user_emb: np.ndarray) -> np.ndarray:
        """Update agent's prediction of next user input (simple EMA)."""
        if self.agent.mu_pred is None:
            return normalize(user_emb)
        alpha = 0.3
        return normalize(alpha * user_emb + (1 - alpha) * self.agent.mu_pred)
    
    def _print_metrics(self, log: Dict):
        """Print turn metrics to console."""
        am = log["agent_metrics"]
        um = log["user_metrics"]
        cm = log["corridor_metrics"]
        
        print(f"\n{'─'*50}")
        print(f"Turn {log['turn']} | Agent Band: {am['band']}")
        print(f"Agent ρ: {am['rho_after']:.3f} (fast={am['rho_fast']:.3f}, slow={am['rho_slow']:.3f}, trauma={am['rho_trauma']:.3f})")
        print(f"Core Drift: {am['core_drift']:.3f}")
        print(f"User Consistency: {um['consistency']:.3f}")
        print(f"Corridor: J={cm['best_J']:.3f}, {cm['passed_count']}/{cm['total_candidates']} passed")
        if log["wound_active"]:
            print(f"⚠️ WOUND ACTIVE (resonance={log['wound_resonance']:.2f})")
        print(f"{'─'*50}\n")
    
    async def run_interactive(self):
        """Run interactive chatbot session."""
        print("\n" + "="*70)
        print("SINGULARITY CHATBOT — INTERACTIVE MODE")
        print("="*70)
        print("Type your message and press Enter. Type 'quit' to exit.")
        print("Both you and the agent are being tracked in DDA-X space.")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                
                response = await self.process_user_input(user_input)
                print(f"\n{self.agent_config['name']}: {response}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # Save session
        self.save_session()
        print(f"\nSession saved to {self.run_dir}")
    
    def save_session(self):
        """Save all session data."""
        # Session log JSON
        session_data = {
            "experiment": "singularity_chatbot",
            "agent": self.agent_config,
            "config": CONFIG,
            "params": D1_PARAMS,
            "turns": self.session_log,
            "timestamp_start": self.session_log[0]["timestamp"] if self.session_log else None,
            "timestamp_end": datetime.now().isoformat(),
        }
        with open(self.run_dir / "session_log.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
        
        # Transcript
        with open(self.run_dir / "transcript.md", "w", encoding="utf-8") as f:
            f.write(f"# Singularity Chatbot Transcript\n\n")
            f.write(f"**Agent**: {self.agent_config['name']} ({self.agent_config['core_archetype']})\n")
            f.write(f"**Model**: {CONFIG['chat_model']} | **K**: {CONFIG['gen_candidates']}\n\n---\n\n")
            
            for t in self.session_log:
                f.write(f"## Turn {t['turn']}\n\n")
                f.write(f"**User**: {t['user_input']}\n\n")
                f.write(f"**{self.agent_config['name']}** [{t['agent_metrics']['band']}]:\n{t['agent_response']}\n\n")
                f.write(f"*ρ={t['agent_metrics']['rho_after']:.3f} | drift={t['agent_metrics']['core_drift']:.3f} | J={t['corridor_metrics']['best_J']:.3f}*\n\n---\n\n")
        
        # Visualizations
        self._plot_dynamics()
        
        # Pickle ledger
        with open(self.run_dir / "agent_ledger.pkl", "wb") as f:
            pickle.dump({
                "agent": self.agent,
                "user": self.user,
                "history": self.history,
            }, f)
    
    def _plot_dynamics(self):
        """Generate dynamics visualization."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        if not self.session_log:
            return
        
        turns = [t["turn"] for t in self.session_log]
        rho = [t["agent_metrics"]["rho_after"] for t in self.session_log]
        drift = [t["agent_metrics"]["core_drift"] for t in self.session_log]
        user_consistency = [t["user_metrics"]["consistency"] for t in self.session_log]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("#1a1a2e")
        
        for ax in axes.flat:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_color("#e94560")
        
        # Rigidity
        axes[0, 0].plot(turns, rho, color="#e94560", linewidth=2)
        axes[0, 0].set_title("Agent Rigidity (ρ)")
        axes[0, 0].set_xlabel("Turn")
        axes[0, 0].set_ylabel("ρ")
        
        # Core Drift
        axes[0, 1].plot(turns, drift, color="#0f3460", linewidth=2)
        axes[0, 1].set_title("Core Identity Drift")
        axes[0, 1].set_xlabel("Turn")
        axes[0, 1].set_ylabel("Drift (1 - cos)")
        
        # User Consistency
        axes[1, 0].plot(turns, user_consistency, color="#e94560", linewidth=2)
        axes[1, 0].set_title("User Consistency (tracked)")
        axes[1, 0].set_xlabel("Turn")
        axes[1, 0].set_ylabel("Consistency")
        
        # Band distribution
        bands = [t["agent_metrics"]["band"] for t in self.session_log]
        band_counts = {b: bands.count(b) for b in ["PRESENT", "AWARE", "WATCHFUL", "CONTRACTED", "FROZEN"]}
        colors = ["#00ff88", "#88ff00", "#ffff00", "#ff8800", "#ff0000"]
        axes[1, 1].bar(band_counts.keys(), band_counts.values(), color=colors)
        axes[1, 1].set_title("Band Distribution")
        axes[1, 1].set_xlabel("Band")
        axes[1, 1].set_ylabel("Count")
        
        plt.tight_layout()
        plt.savefig(self.run_dir / "dynamics_dashboard.png", dpi=150, facecolor="#1a1a2e")
        plt.close()


# =============================================================================
# D1 PARAMETERS (PHYSICS)
# =============================================================================
D1_PARAMS = {
    # GLOBAL DYNAMICS
    "epsilon_0": CONFIG["epsilon_0"],
    "s": CONFIG["s"],
    "arousal_decay": 0.72,
    "arousal_gain": 0.85,
    
    # RIGIDITY HOMEOSTASIS
    "rho_setpoint_fast": 0.45,
    "rho_setpoint_slow": 0.35,
    "homeo_fast": 0.10,
    "homeo_slow": 0.01,
    "alpha_fast": CONFIG["alpha_fast"],
    "alpha_slow": CONFIG["alpha_slow"],
    
    # TRAUMA (ASYMMETRIC)
    "trauma_threshold": 1.15,
    "alpha_trauma": CONFIG["alpha_trauma"],
    "trauma_decay": 0.998,
    "trauma_floor": 0.02,
    "healing_rate": 0.015,
    "safe_threshold": 5,
    "safe_epsilon": 0.75,
    
    # WEIGHTING
    "w_fast": 0.52,
    "w_slow": 0.30,
    "w_trauma": 1.10,
    
    # PREDICTIVE CODING
    "R_ema": 0.06,
    "R_min": 1e-4,
    "R_max": 1e-1,
    "P_init": 0.02,
    "Q_base": 0.0015,
    "Q_rho_scale": 0.010,
    
    # GRADIENT FLOW
    "dt": 1.0,
    "eta_base": 0.18,
    "eta_min": 0.03,
    "eta_rho_power": 1.6,
    "sigma_base": 0.004,
    "sigma_rho_scale": 0.020,
    "noise_clip": 3.0,
    
    # ROLE ADAPTATION
    "role_adapt": 0.06,
    "role_input_mix": 0.08,
    "drift_cap": 0.06,
    
    # CORRIDOR LOGIC (K-SAMPLING)
    "core_cos_min": 0.20,
    "role_cos_min": 0.08,
    "energy_max": 9.5,
    "w_core": CONFIG["w_core"],
    "w_role": CONFIG["w_role"],
    "w_energy": CONFIG["w_energy"],
    "w_novel": CONFIG["w_novel"],
    "reject_penalty": 4.0,
    
    "corridor_strict": CONFIG["corridor_strict"],
    "corridor_max_batches": CONFIG["corridor_max_batches"],
    
    # WOUND MECHANICS
    "wound_cooldown": 3,
    "wound_amp_max": 1.4,
    "wound_cosine_threshold": 0.28,
    
    # GENERATION PARAMS
    "gen_params_default": {
        "temperature": 0.9,
        "top_p": 0.95,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
    },
    
    "seed": CONFIG["seed"],
}


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    async def main():
        chatbot = SingularityChatbot(seed=None)  # Random identity each run
        await chatbot.run_interactive()
    
    asyncio.run(main())
```

---

## RECURSIVE REFINER — STEP M+1 PREPARATION

```
TO PROCEED TO STEP M+1:

1. RUN THIS SIMULATION:
   python simulations/singularity_chatbot.py

2. INTERACT ORGANICALLY:
   - Speak to the agent naturally
   - Challenge its core beliefs
   - Probe its wound triggers
   - Observe how it adapts (or doesn't)

3. ANALYZE OUTPUTS:
   - Review `data/singularity_*/session_log.json` for full telemetry
   - Check `data/singularity_*/dynamics_dashboard.png` for visual patterns
   - Read `data/singularity_*/transcript.md` for qualitative dynamics
   - Compare USER metrics vs AGENT metrics

4. IDENTIFY REFINEMENT OPPORTUNITIES:
   - Did the agent maintain core identity? (Check core_drift)
   - Did wounds trigger appropriately? (Check wound_resonance)
   - Did corridor reject too many samples? (Check passed_count)
   - Did user consistency affect agent prediction accuracy?

5. FEEDBACK FOR M+1:
   "I ran Step M={M_STEP} of singularity_chatbot.
   
   Agent was: {ARCHETYPE} with traits {TRAITS}
   
   Observations:
   - [OBSERVATION 1]
   - [OBSERVATION 2]
   - [OBSERVATION 3]
   
   For M+1, I want to:
   - [ADJUSTMENT 1]
   - [ADJUSTMENT 2]"
```

---

## STEP M={M_STEP} DEFAULT VALUES

```python
M_STEP = 0
M_STEP_CONTEXT = "This is the initial exploration. No prior run data available."
```

For **Step M≥1**, the context will be populated from the previous run's `session_log.json`.

---

## THIS IS PURE BEAUTY

Two minds. One tracked actively. One tracked passively. Both in 3072-dimensional space. The agent is randomly generated from the depths of internet culture. The user is you. **Let them cook.**

🚀
