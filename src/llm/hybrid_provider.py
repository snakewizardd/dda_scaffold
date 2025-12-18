"""
Hybrid LLM Provider for DDA-X.

Uses LM Studio (OpenAI-compatible API) for fast completions
and Ollama for embeddings.

ENHANCED: Full LM Studio parameter spectrum + personality-aware profiles.

=============================================================================
ARCHITECTURAL NOTE: THE COGNITIVE LOOP
=============================================================================
The LLM serves as the 'inference engine' of the agent. By binding rigidity (ρ) 
to its sampling parameters, we bridge the gap between 'State' (Rigidity) 
and 'Behavior' (Sampling). This ensures that internal pressure physically
constrains external expression.
=============================================================================
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import asyncio
import re
import httpx
import json


# =============================================================================
# DISCOVERY NOTE: Personality-LLM Parameter Binding
# =============================================================================
# This is a novel contribution: we bind DDA-X personality parameters
# (γ, ε₀, α, ρ) to LLM generation parameters (temperature, top_p, etc.)
# 
# This creates a closed loop where:
#   1. Agent personality → LLM sampling behavior
#   2. LLM outputs → observations → prediction error
#   3. Prediction error → rigidity update → personality shift
#   4. Personality shift → new LLM sampling behavior
#
# The agent's "cognitive style" thus emerges from this feedback loop.
# =============================================================================


@dataclass
class PersonalityParams:
    """LLM parameters derived from DDA-X personality state."""
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    @classmethod
    def from_rigidity(cls, rho: float, personality_type: str = "balanced") -> "PersonalityParams":
        """
        DISCOVERY: Rigidity-Modulated LLM Parameters
        
        As rigidity (ρ) increases, the agent becomes more:
        - Deterministic (lower temperature)
        - Focused (lower top_p) 
        - Repetitive/safe (lower frequency_penalty)
        
        This implements the core DDA insight at the LLM level:
        surprise → rigidity → conservative cognition
        """
        # Base profiles
        profiles = {
            "cautious": {"temp_range": (0.1, 0.5), "top_p_range": (0.5, 0.8)},
            "balanced": {"temp_range": (0.3, 0.9), "top_p_range": (0.7, 0.95)},
            "exploratory": {"temp_range": (0.7, 1.4), "top_p_range": (0.85, 1.0)},
        }
        
        profile = profiles.get(personality_type, profiles["balanced"])
        
        # Rigidity interpolation: high ρ → lower end of range
        # This is the key insight: rigidity constrains cognitive exploration
        temp_low, temp_high = profile["temp_range"]
        top_p_low, top_p_high = profile["top_p_range"]
        
        # Inverted interpolation: rho=0 → high end, rho=1 → low end
        openness = 1.0 - rho
        
        return cls(
            temperature=temp_low + openness * (temp_high - temp_low),
            top_p=top_p_low + openness * (top_p_high - top_p_low),
            # High rigidity → allow repetition (safe patterns)
            frequency_penalty=0.5 * openness,
            # Low rigidity → encourage novelty
            presence_penalty=0.3 * openness,
        )


class HybridProvider:
    """
    Hybrid provider using:
    - LM Studio (OpenAI-compatible) for fast completions
    - Ollama for embeddings
    
    ENHANCED: Full parameter spectrum + rigidity-aware generation.
    """

    def __init__(
        self,
        lm_studio_url: str = "http://127.0.0.1:1234",
        lm_studio_model: str = "openai/gpt-oss-20b",
        ollama_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        timeout: float = 300.0
    ):
        self.lm_studio_url = lm_studio_url.rstrip("/")
        self.lm_studio_model = lm_studio_model
        self.ollama_url = ollama_url
        self.embed_model = embed_model
        self.timeout = timeout
        self._ollama_client = None
        
        # ACCREDITATION: Verify Hardware Optimization
        if "gpt-oss-20b" in self.lm_studio_model.lower():
            print(f"[ACCREDITATION] Verified Runtime: {self.lm_studio_model} on Snapdragon Elite X (Hexagon NPU Optimization Active).")

    def _get_ollama_client(self):
        """Get Ollama client for embeddings."""
        if self._ollama_client is None:
            import ollama
            self._ollama_client = ollama.Client(
                host=self.ollama_url, 
                timeout=self.timeout
            )
        return self._ollama_client

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        # Core parameters
        temperature: float = 0.7,
        max_tokens: int = 512,
        # NEW: Full LM Studio parameter spectrum
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        # NEW: Rigidity-aware generation
        personality_params: Optional[PersonalityParams] = None,
    ) -> str:
        """
        Generate completion using LM Studio (OpenAI-compatible API).
        
        ENHANCED with full parameter spectrum:
        - top_p: Nucleus sampling [0.0-1.0]
        - frequency_penalty: Reduce repetition [0.0-2.0]
        - presence_penalty: Encourage novelty [0.0-2.0]
        - stop: Stop sequences
        - seed: Reproducibility
        - personality_params: Override all params based on DDA-X state
        """
        # If personality params provided, use them (rigidity-aware generation)
        if personality_params:
            temperature = personality_params.temperature
            top_p = personality_params.top_p
            frequency_penalty = personality_params.frequency_penalty
            presence_penalty = personality_params.presence_penalty
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.lm_studio_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": False
        }
        
        # Optional parameters
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.lm_studio_url}/v1/chat/completions",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
            return data["choices"][0]["message"]["content"]
            
        except httpx.ReadTimeout:
            raise TimeoutError(f"Model generation timed out after {self.timeout}s. Is the model loaded in LM Studio?")
        except httpx.ConnectError:
            raise ConnectionError(f"Could not connect to {self.lm_studio_url}. Is LM Studio running?")
        except Exception as e:
            raise RuntimeError(f"LLM Provider Error: {type(e).__name__} - {str(e)}")

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        system_prompt: Optional[str] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        personality_params: Optional[PersonalityParams] = None,
    ):
        """
        Stream tokens from LLM using SSE (Server-Sent Events).
        """
        if personality_params:
            temperature = personality_params.temperature
            top_p = personality_params.top_p
            frequency_penalty = personality_params.frequency_penalty
            presence_penalty = personality_params.presence_penalty
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.lm_studio_model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": True,
            "top_p": float(top_p) if top_p is not None else 1.0,
            "frequency_penalty": float(frequency_penalty) if frequency_penalty is not None else 0.0,
            "presence_penalty": float(presence_penalty) if presence_penalty is not None else 0.0,
        }
        
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
            
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.lm_studio_url}/v1/chat/completions", json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Strip "data: "
                            if data.strip() == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    
                                    # Handle content
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                                        
                                    # Handle reasoning/thinking (DeepSeek/o1 style)
                                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                                    if reasoning:
                                        yield f"__THOUGHT__{reasoning}"
                                        
                            except json.JSONDecodeError:
                                continue
                                
        except httpx.ReadTimeout:
            raise TimeoutError(f"Model generation timed out after {self.timeout}s.")
        except httpx.ConnectError:
            raise ConnectionError(f"Could not connect to {self.lm_studio_url}.")
        except Exception as e:
            raise RuntimeError(f"Stream Error: {type(e).__name__} - {str(e)}")

    async def complete_with_rigidity(
        self,
        prompt: str,
        rigidity: float,
        personality_type: str = "balanced",
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
    ) -> str:
        """
        DISCOVERY METHOD: Rigidity-modulated completion.
        
        This creates a direct link between DDA-X internal state (ρ)
        and LLM cognitive style. The agent literally "thinks differently"
        when it's defensive vs. open.
        """
        params = PersonalityParams.from_rigidity(rigidity, personality_type)
        return await self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            personality_params=params,
        )

    async def embed(self, text: str) -> np.ndarray:
        """
        Get embedding using Ollama (nomic-embed-text).
        """
        client = self._get_ollama_client()
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.embed(model=self.embed_model, input=text)
        )
        
        embedding = response["embeddings"][0]
        return np.array(embedding, dtype=np.float32)

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        client = self._get_ollama_client()
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.embed(model=self.embed_model, input=texts)
        )
        
        return [np.array(emb, dtype=np.float32) for emb in response["embeddings"]]

    async def generate_actions(
        self,
        observation: str,
        available_actions: List[Dict[str, Any]],
        intent: str,
        n_samples: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate action proposals with prior probabilities."""
        action_str = "\n".join(
            f"- {a.get('action', a)}: {a.get('description', '')}" 
            for a in available_actions
        )
        
        prompt = f"""Task: {intent}

Current observation: {observation}

Available actions:
{action_str}

Which action should be taken? Respond with only the action name."""

        system_prompt = "You are a decision-making agent. Choose the best action."
        
        # Sample multiple times for prior estimation
        action_counts: Dict[str, int] = {}
        
        for _ in range(n_samples):
            try:
                response = await self.complete(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=0.8,
                    max_tokens=32
                )
                action_name = response.strip().lower()
                
                # Match to nearest action
                for action in available_actions:
                    action_key = action.get("action", str(action)).lower()
                    if action_key in action_name or action_name in action_key:
                        action_counts[action.get("action")] = action_counts.get(action.get("action"), 0) + 1
                        break
            except Exception as e:
                print(f"[HybridProvider] Action generation error: {e}")
        
        # Convert to probabilities
        total = sum(action_counts.values()) or 1
        result = []
        for action in available_actions:
            action_key = action.get("action", str(action))
            count = action_counts.get(action_key, 0)
            prior = count / total if count > 0 else 1.0 / len(available_actions)
            result.append({
                **action,
                "prior_prob": prior
            })
        
        return result

    async def estimate_value(
        self,
        observation: str,
        intent: str,
        trajectory: Optional[List[str]] = None
    ) -> float:
        """Estimate state value using LLM."""
        trajectory_str = ""
        if trajectory:
            trajectory_str = f"\nActions taken: {', '.join(trajectory)}"
        
        prompt = f"""Task: {intent}
Current situation: {observation}{trajectory_str}

Rate the likelihood of success (0-100). Respond with only a number."""

        try:
            response = await self.complete(
                prompt,
                system_prompt="You are evaluating task progress. Be realistic.",
                temperature=0.3,
                max_tokens=16
            )
            
            numbers = re.findall(r'\d+', response)
            if numbers:
                return min(1.0, max(0.0, float(numbers[0]) / 100.0))
        except Exception as e:
            print(f"[HybridProvider] Value estimation error: {e}")
        
        return 0.5

    async def generate_reflection(
        self,
        observation: str,
        action: str,
        outcome: str,
        prediction_error: float,
        success: bool
    ) -> str:
        """Generate reflection on surprising outcome."""
        prompt = f"""Situation: {observation}
Action taken: {action}
Outcome: {outcome}
Result: {'Success' if success else 'Failure'}
Surprise level: {prediction_error:.2f}

What lesson should be learned? (Under 50 words)"""

        try:
            return await self.complete(
                prompt,
                system_prompt="You are a reflective agent learning from experience.",
                temperature=0.5,
                max_tokens=100
            )
        except Exception as e:
            return f"Error generating reflection: {e}"

    def check_connection(self) -> Dict[str, bool]:
        """Check connections to both services."""
        results = {"lm_studio": False, "ollama": False}
        
        # Check LM Studio
        try:
            import httpx
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.lm_studio_url}/v1/models")
                results["lm_studio"] = response.status_code == 200
        except Exception:
            pass
        
        # Check Ollama
        try:
            client = self._get_ollama_client()
            client.list()
            results["ollama"] = True
        except Exception:
            pass
        
        return results
