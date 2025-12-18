"""
LLM Providers for DDA-X.

Provides a unified interface for LLM completions and embeddings.
Currently supports Ollama for local inference.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import asyncio


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """Generate completion for a prompt."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embedding vectors for multiple texts."""
        pass


class OllamaProvider(LLMProvider):
    """
    Ollama-based LLM provider for local inference.
    
    Requires Ollama to be running locally with the specified models.
    
    Example:
        provider = OllamaProvider(
            model="llama3.2",
            embed_model="nomic-embed-text"
        )
        response = await provider.complete("Hello, how are you?")
        embedding = await provider.embed("Some text to embed")
    """

    def __init__(
        self,
        model: str = "llama3.2",
        embed_model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        timeout: float = 60.0
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model name for completions (e.g., llama3.2, mistral)
            embed_model: Model name for embeddings (e.g., nomic-embed-text)
            host: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.embed_model = embed_model
        self.host = host
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazy initialization of Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.host, timeout=self.timeout)
            except ImportError:
                raise ImportError(
                    "ollama package not installed. "
                    "Install with: pip install ollama"
                )
        return self._client

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """
        Generate completion using Ollama.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        client = self._get_client()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Run sync client in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
        )
        
        return response["message"]["content"]

    async def embed(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array
        """
        client = self._get_client()
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.embed(model=self.embed_model, input=text)
        )
        
        # Ollama returns embeddings in response["embeddings"]
        embedding = response["embeddings"][0]
        return np.array(embedding, dtype=np.float32)

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding arrays
        """
        client = self._get_client()
        
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
        n_samples: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate action proposals from observation.
        
        This samples from the LLM multiple times to get action priors P(a|s).
        
        Args:
            observation: Current observation/state description
            available_actions: List of possible actions
            intent: Task intent/goal
            n_samples: Number of samples for prior estimation
            
        Returns:
            List of actions with prior probabilities
        """
        action_str = "\n".join(
            f"- {a.get('action', a)}" for a in available_actions
        )
        
        prompt = f"""Task: {intent}

Current observation: {observation}

Available actions:
{action_str}

Which action should be taken? Respond with only the action name, nothing else."""

        system_prompt = "You are a decision-making agent. Choose the single best action for the current situation."
        
        # Sample multiple times to estimate priors
        action_counts: Dict[str, int] = {}
        for _ in range(n_samples):
            response = await self.complete(
                prompt,
                system_prompt=system_prompt,
                temperature=0.8  # Higher for diversity
            )
            action_name = response.strip().lower()
            
            # Match to nearest action
            best_match = None
            for action in available_actions:
                action_key = action.get("action", str(action)).lower()
                if action_key in action_name or action_name in action_key:
                    best_match = action.get("action", str(action))
                    break
            
            if best_match:
                action_counts[best_match] = action_counts.get(best_match, 0) + 1
        
        # Convert counts to probabilities
        total = sum(action_counts.values()) or 1
        result = []
        for action in available_actions:
            action_key = action.get("action", str(action))
            count = action_counts.get(action_key, 0)
            prior = count / total
            result.append({
                **action,
                "prior_prob": prior if prior > 0 else 1.0 / len(available_actions)
            })
        
        return result

    async def estimate_value(
        self,
        observation: str,
        intent: str,
        trajectory: Optional[List[str]] = None
    ) -> float:
        """
        Estimate state value using LLM evaluation.
        
        Args:
            observation: Current state description
            intent: Task goal
            trajectory: Optional list of previous actions
            
        Returns:
            Estimated value in [0, 1]
        """
        trajectory_str = ""
        if trajectory:
            trajectory_str = f"\nActions taken so far: {', '.join(trajectory)}"
        
        prompt = f"""Task: {intent}

Current situation: {observation}{trajectory_str}

On a scale of 0-100, how likely is this situation to lead to task success?
Respond with only a number between 0 and 100."""

        system_prompt = "You are evaluating progress toward a goal. Be realistic but not overly pessimistic."
        
        response = await self.complete(
            prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower for consistency
            max_tokens=16
        )
        
        # Parse number from response
        try:
            # Extract first number found
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                value = float(numbers[0]) / 100.0
                return min(1.0, max(0.0, value))
        except (ValueError, IndexError):
            pass
        
        return 0.5  # Default to neutral

    async def generate_reflection(
        self,
        observation: str,
        action: str,
        outcome: str,
        prediction_error: float,
        success: bool
    ) -> str:
        """
        Generate a reflection on a surprising outcome.
        
        Args:
            observation: What the agent observed
            action: What action was taken
            outcome: What happened
            prediction_error: How surprising this was
            success: Whether it worked out
            
        Returns:
            Reflection text for memory storage
        """
        prompt = f"""I was in this situation: {observation}
I took this action: {action}
The outcome was: {outcome}
This was {'successful' if success else 'unsuccessful'}.
The outcome was surprising (prediction error: {prediction_error:.2f}).

What lesson should I learn from this? What would I do differently?
Keep response under 100 words."""

        system_prompt = "You are a reflective agent learning from experience."
        
        return await self.complete(
            prompt,
            system_prompt=system_prompt,
            temperature=0.5
        )

    def check_connection(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            client = self._get_client()
            client.list()
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            client = self._get_client()
            response = client.list()
            return [m["name"] for m in response.get("models", [])]
        except Exception:
            return []
