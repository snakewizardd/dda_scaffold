"""
OpenAI Provider for DDA-X.

Integrates:
- GPT-5.2 (or latest available) for synthesis and logic.
- text-embedding-3-large for high-dimensional conceptual space (3072 dim).
- Full DDA-X Rigidity-Binding for personality modulation.
"""

import os
import numpy as np
from typing import List, Optional, Dict, Any, Union
import asyncio
import json
from dataclasses import dataclass
import httpx
from dotenv import load_dotenv

# Load env to get OAI_API_KEY safely
load_dotenv()

from src.llm.hybrid_provider import PersonalityParams
from src.llm.providers import LLMProvider

class OpenAIProvider:
    """
    Connects to OpenAI API for high-fidelity simulation.
    
    Features:
    - Text Embedding 3 Large (3072 dim)
    - GPT-5.2 (Preview/Beta)
    - Async operation
    - DDA-X Parameter Binding
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
        max_tokens: int = 1024,
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
            if "gpt-5.2" in self.model or "o1" in self.model:
                 kwargs["max_completion_tokens"] = max_tokens
                 # Explicitly exclude temp, top_p, penalties as they cause 400 errors
            else:
                # Standard GPT-4/3.5/4o parameters
                kwargs["max_tokens"] = max_tokens
                kwargs["temperature"] = temperature
                kwargs["top_p"] = top_p
                kwargs["frequency_penalty"] = frequency_penalty
                kwargs["presence_penalty"] = presence_penalty
            
            if response_format:
                kwargs["response_format"] = response_format

            response = await client.chat.completions.create(**kwargs)
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
        """
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
