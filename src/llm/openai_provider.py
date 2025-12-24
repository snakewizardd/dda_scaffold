"""
OpenAI Provider for DDA-X.

Integrates:
- GPT-5.2 (or latest available) for synthesis and logic.
- text-embedding-3-large for high-dimensional conceptual space (3072 dim).
- Full DDA-X Rigidity-Binding for personality modulation.
- Cost tracking for embeddings and chat completions (no credentials exposed).
"""

import os
import numpy as np
from typing import List, Optional, Dict, Any, Union
import asyncio
import json
from dataclasses import dataclass, field
import httpx
from dotenv import load_dotenv

# Load env to get OAI_API_KEY safely
load_dotenv()

from src.llm.hybrid_provider import PersonalityParams
from src.llm.providers import LLMProvider


# OpenAI pricing (USD per 1K tokens) - updated Dec 2024
# NOTE: No API keys, org IDs, or credentials stored here
PRICING = {
    # Chat models (input/output per 1K tokens)
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-5.2": {"input": 0.01, "output": 0.03},  # Estimated
    "o1": {"input": 0.015, "output": 0.06},
    "o1-preview": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    # Embedding models (per 1K tokens)
    "text-embedding-3-large": {"input": 0.00013},
    "text-embedding-3-small": {"input": 0.00002},
    "text-embedding-ada-002": {"input": 0.0001},
}


@dataclass
class CostTracker:
    """Tracks API usage and estimated costs without exposing credentials."""
    
    # Embedding stats
    embed_requests: int = 0
    embed_tokens: int = 0
    embed_model: str = ""
    
    # Chat completion stats
    chat_requests: int = 0
    chat_input_tokens: int = 0
    chat_output_tokens: int = 0
    chat_model: str = ""
    
    # Per-model breakdown
    model_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def record_embedding(self, model: str, tokens: int):
        """Record an embedding request."""
        self.embed_requests += 1
        self.embed_tokens += tokens
        self.embed_model = model
        
        if model not in self.model_usage:
            self.model_usage[model] = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        self.model_usage[model]["requests"] += 1
        self.model_usage[model]["input_tokens"] += tokens
    
    def record_chat(self, model: str, input_tokens: int, output_tokens: int):
        """Record a chat completion request."""
        self.chat_requests += 1
        self.chat_input_tokens += input_tokens
        self.chat_output_tokens += output_tokens
        self.chat_model = model
        
        if model not in self.model_usage:
            self.model_usage[model] = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        self.model_usage[model]["requests"] += 1
        self.model_usage[model]["input_tokens"] += input_tokens
        self.model_usage[model]["output_tokens"] += output_tokens
    
    def estimate_cost(self) -> Dict[str, Any]:
        """Calculate estimated cost based on usage. No credentials exposed."""
        total_cost = 0.0
        breakdown = {}
        
        for model, usage in self.model_usage.items():
            pricing = PRICING.get(model, {"input": 0.01, "output": 0.03})
            
            input_cost = (usage["input_tokens"] / 1000) * pricing.get("input", 0.01)
            output_cost = (usage["output_tokens"] / 1000) * pricing.get("output", 0.0)
            model_cost = input_cost + output_cost
            
            breakdown[model] = {
                "requests": usage["requests"],
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "input_cost_usd": round(input_cost, 6),
                "output_cost_usd": round(output_cost, 6),
                "total_cost_usd": round(model_cost, 6),
            }
            total_cost += model_cost
        
        return {
            "total_cost_usd": round(total_cost, 4),
            "total_requests": self.embed_requests + self.chat_requests,
            "total_tokens": self.embed_tokens + self.chat_input_tokens + self.chat_output_tokens,
            "embedding": {
                "requests": self.embed_requests,
                "tokens": self.embed_tokens,
            },
            "chat": {
                "requests": self.chat_requests,
                "input_tokens": self.chat_input_tokens,
                "output_tokens": self.chat_output_tokens,
            },
            "by_model": breakdown,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export cost report as dict (for JSON serialization)."""
        return self.estimate_cost()
    
    def reset(self):
        """Reset all counters."""
        self.embed_requests = 0
        self.embed_tokens = 0
        self.chat_requests = 0
        self.chat_input_tokens = 0
        self.chat_output_tokens = 0
        self.model_usage = {}


class OpenAIProvider:
    """
    Connects to OpenAI API for high-fidelity simulation.
    
    Features:
    - Text Embedding 3 Large (3072 dim)
    - GPT-5.2 (Preview/Beta)
    - Async operation
    - DDA-X Parameter Binding
    - Cost tracking (no credentials exposed)
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        embed_model: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        timeout: float = 60.0
    ):
        self.model = model
        self.embed_model = embed_model
        self.timeout = timeout
        self.api_key = api_key or os.getenv("OAI_API_KEY")
        
        if not self.api_key:
            print("⚠️ WARNING: No OpenAI API Key found in environment.")
            
        self._client = None
        
        # Cost tracking (no credentials stored)
        self.cost_tracker = CostTracker()
        
    def _get_client(self):
        """Lazy init of AsyncOpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout
            )
        return self._client

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1024,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        personality_params: Optional[PersonalityParams] = None,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Generate completion using OpenAI.
        """
        if personality_params:
            temperature = personality_params.temperature
            top_p = personality_params.top_p
            frequency_penalty = personality_params.frequency_penalty
            presence_penalty = personality_params.presence_penalty

        client = self._get_client()
        
        # gpt-5-nano requires the Responses API, not Chat Completions
        if "gpt-5-nano" in self.model:
            try:
                # Build input with system prompt if provided
                full_input = prompt
                if system_prompt:
                    full_input = f"[SYSTEM]: {system_prompt}\n\n[USER]: {prompt}"
                
                kwargs = {
                    "model": self.model,
                    "input": full_input,
                }
                if max_tokens is not None:
                    kwargs["max_output_tokens"] = max(500, max_tokens)

                response = await client.responses.create(**kwargs)
                
                # Track usage
                if hasattr(response, 'usage') and response.usage:
                    self.cost_tracker.record_chat(
                        self.model,
                        response.usage.input_tokens or 0,
                        response.usage.output_tokens or 0
                    )
                
                # Extract text from message items
                if response.output:
                    for item in response.output:
                        if item.type == 'message':
                            for content in item.content:
                                if hasattr(content, 'text') and content.text:
                                    return content.text
                return ""
            except Exception as e:
                raise RuntimeError(f"OpenAI Responses API Error: {str(e)}")
        
        # Standard Chat Completions API for other models
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
            }
            
            # Reasoning models (o1/gpt-5.2) often don't support standard sampling params
            if "gpt-5" in self.model or "o1" in self.model:
                 if max_tokens is not None:
                     kwargs["max_completion_tokens"] = max_tokens
                 # Explicitly exclude temp, top_p, penalties as they cause 400 errors
            else:
                # Standard GPT-4/3.5/4o parameters
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                kwargs["temperature"] = temperature
                kwargs["top_p"] = top_p
                kwargs["frequency_penalty"] = frequency_penalty
                kwargs["presence_penalty"] = presence_penalty
            
            if response_format:
                kwargs["response_format"] = response_format

            response = await client.chat.completions.create(**kwargs)
            
            # Track usage (no credentials exposed)
            if hasattr(response, 'usage') and response.usage:
                self.cost_tracker.record_chat(
                    self.model,
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Fallback handling or re-raise
            raise RuntimeError(f"OpenAI API Error: {str(e)}")

    async def embed(self, text: str) -> np.ndarray:
        """
        Get 3072-dim embedding.
        """
        client = self._get_client()
        
        # Replace newlines for best performance
        text = text.replace("\n", " ")
        
        try:
            response = await client.embeddings.create(
                input=[text],
                model=self.embed_model,
                dimensions=3072  # Explicitly request full dimensionality
            )
            
            # Track usage (no credentials exposed)
            if hasattr(response, 'usage') and response.usage:
                self.cost_tracker.record_embedding(
                    self.embed_model,
                    response.usage.total_tokens or 0
                )
            
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"OpenAI Embedding Error: {str(e)}")

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get batch embeddings.
        """
        client = self._get_client()
        processed_texts = [t.replace("\n", " ") for t in texts]
        
        try:
            response = await client.embeddings.create(
                input=processed_texts,
                model=self.embed_model,
                dimensions=3072
            )
            
            # Track usage (no credentials exposed)
            if hasattr(response, 'usage') and response.usage:
                self.cost_tracker.record_embedding(
                    self.embed_model,
                    response.usage.total_tokens or 0
                )
            
            # Ensure order is preserved (OpenAI guarantees this but good to be safe)
            embeddings = [np.array(d.embedding, dtype=np.float32) for d in response.data]
            return embeddings
        except Exception as e:
            raise RuntimeError(f"OpenAI Batch Embedding Error: {str(e)}")

    async def complete_with_rigidity(
        self,
        prompt: str,
        rigidity: float,
        personality_type: str = "balanced",
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        """
        DDA-X Core: Bind internal state (rigidity) to external behavior.
        
        For Standard Models: Uses Sampling Parameters (Temp, Top-P).
        For Reasoning Models (GPT-5.2): Uses Semantic Injection.
        """
        params = PersonalityParams.from_rigidity(rigidity, personality_type)
        
        # Semantic Adapter for Reasoning Models
        if "gpt-5.2" in self.model or "o1" in self.model:
            # Translate Rigidity (Physics) -> Instruction (Semantics)
            semantic_instruction = self._get_semantic_rigidity_instruction(rigidity)
            
            # Inject into system prompt or user prompt
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n[COGNITIVE STATE]: {semantic_instruction}"
            else:
                # If no system prompt, prepend to user prompt (some o1-preview models restrict system)
                prompt = f"[COGNITIVE STATE]: {semantic_instruction}\n\n{prompt}"
                
        return await self.complete(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            personality_params=params
        )

    def _get_semantic_rigidity_instruction(self, rho: float) -> str:
        """
        Translates mathematical rigidity (0-1) into behavioral instructions
        to compensate for lack of sampling parameter control.
        
        Uses the 100-point RigidityScale for fine-grained semantic injection.
        """
        try:
            from src.llm.rigidity_scale_100 import get_rigidity_injection
            return get_rigidity_injection(rho)
        except ImportError:
            # Fallback to original 5-bucket system if scale not available
            if rho < 0.2:
                return "Cognitive State: FLUID. Be highly creative, abstract, and metaphorical. Make intuitive leaps. Ignore conventions."
            elif rho < 0.4:
                return "Cognitive State: OPEN. Explore new ideas but maintain coherence. Be curious and questioning."
            elif rho < 0.6:
                return "Cognitive State: BALANCED. Weigh evidence carefully. Be pragmatic and structural."
            elif rho < 0.8:
                return "Cognitive State: RIGID. Be skeptical, concise, and literal. Rely only on established facts. Reject speculation."
            else:
                return "Cognitive State: FROZEN. Be extremely dogmatic, repetitive, and defensive. Refuse to change your mind. Short, clipped responses."

    async def generate_actions(
        self,
        observation: str,
        available_actions: List[Dict[str, Any]],
        intent: str,
        n_samples: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate actions. For GPT-5.2, we trust a single zero-shot call 
        with high reasoning capability rather than sampling multiple times,
        unless n_samples > 1 is strictly requested for distribution analysis.
        """
        pass 

    def check_connection(self) -> bool:
        return self.api_key is not None
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Get cost report for current session. No credentials exposed."""
        return self.cost_tracker.to_dict()
    
    def reset_cost_tracker(self):
        """Reset cost tracking for new session."""
        self.cost_tracker.reset()
