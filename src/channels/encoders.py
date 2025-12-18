"""
Encoders for DDA-X channels.

Converts observations, actions, and outcomes into state-space vectors.
Uses LLM embeddings for semantic representation.
"""

import numpy as np
from typing import Any, Dict, Optional, List
import asyncio
import hashlib


class ObservationEncoder:
    """
    Encodes observations into state-space vectors.
    
    Maps observation text → ℝ^d via LLM embeddings.
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        target_dim: int = 64,
        cache_embeddings: bool = True
    ):
        """
        Initialize observation encoder.
        
        Args:
            llm_provider: LLM provider for embeddings
            target_dim: Target dimension for state vectors
            cache_embeddings: Whether to cache embeddings
        """
        self.llm_provider = llm_provider
        self.target_dim = target_dim
        self.cache_embeddings = cache_embeddings
        self._cache: Dict[str, np.ndarray] = {}
        self._projection_matrix: Optional[np.ndarray] = None
        self._embed_dim: Optional[int] = None

    def _get_cache_key(self, text: str) -> str:
        """Create cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _project_to_target_dim(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project embedding to target dimension.
        
        Uses random projection to reduce/expand dimensions while
        preserving approximate distances (Johnson-Lindenstrauss).
        """
        if len(embedding) == self.target_dim:
            return embedding
        
        # Initialize projection matrix on first use
        if self._projection_matrix is None or self._embed_dim != len(embedding):
            self._embed_dim = len(embedding)
            # Random projection with proper scaling
            np.random.seed(42)  # For reproducibility
            self._projection_matrix = np.random.randn(
                len(embedding), self.target_dim
            ) / np.sqrt(self.target_dim)
        
        # Project and normalize
        projected = embedding @ self._projection_matrix
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm
        
        return projected

    async def encode_async(self, observation: Any) -> np.ndarray:
        """
        Encode observation asynchronously.
        
        Args:
            observation: Text observation or dict with text
            
        Returns:
            State vector in ℝ^target_dim
        """
        # Convert to text
        if isinstance(observation, dict):
            text = str(observation.get("text", observation))
        else:
            text = str(observation)
        
        # Check cache
        cache_key = self._get_cache_key(text)
        if self.cache_embeddings and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get embedding from LLM
        if self.llm_provider:
            embedding = await self.llm_provider.embed(text)
            result = self._project_to_target_dim(embedding)
        else:
            # Fallback: deterministic hash-based embedding
            result = self._hash_embedding(text)
        
        # Cache result
        if self.cache_embeddings:
            self._cache[cache_key] = result
        
        return result

    def encode(self, observation: Any) -> np.ndarray:
        """
        Synchronous wrapper for encode_async.
        
        For compatibility with existing code that expects sync.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, can't use run_until_complete
                # Return hash embedding as fallback
                if isinstance(observation, dict):
                    text = str(observation.get("text", observation))
                else:
                    text = str(observation)
                return self._hash_embedding(text)
            return loop.run_until_complete(self.encode_async(observation))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.encode_async(observation))

    def _hash_embedding(self, text: str) -> np.ndarray:
        """
        Create deterministic embedding from text hash.
        
        Used when no LLM provider is available.
        """
        # Use hash to seed random state for reproducibility
        text_hash = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        rng = np.random.RandomState(text_hash % (2**32))
        embedding = rng.randn(self.target_dim)
        return embedding / np.linalg.norm(embedding)


class ActionEncoder:
    """
    Encodes actions into direction vectors.
    
    Maps action text/dict → d̂(a) ∈ ℝ^d (unit vector).
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        target_dim: int = 64,
        cache_embeddings: bool = True
    ):
        """
        Initialize action encoder.
        
        Args:
            llm_provider: LLM provider for embeddings
            target_dim: Target dimension for direction vectors
            cache_embeddings: Whether to cache embeddings
        """
        self.llm_provider = llm_provider
        self.target_dim = target_dim
        self.cache_embeddings = cache_embeddings
        self._cache: Dict[str, np.ndarray] = {}
        self._projection_matrix: Optional[np.ndarray] = None
        self._embed_dim: Optional[int] = None

    def _get_cache_key(self, text: str) -> str:
        """Create cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _project_to_target_dim(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding to target dimension."""
        if len(embedding) == self.target_dim:
            return embedding
        
        if self._projection_matrix is None or self._embed_dim != len(embedding):
            self._embed_dim = len(embedding)
            np.random.seed(43)  # Different seed than observation encoder
            self._projection_matrix = np.random.randn(
                len(embedding), self.target_dim
            ) / np.sqrt(self.target_dim)
        
        projected = embedding @ self._projection_matrix
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm
        
        return projected

    def _action_to_text(self, action: Any) -> str:
        """Convert action to text representation."""
        if isinstance(action, dict):
            # Extract meaningful fields
            parts = []
            if "action" in action:
                parts.append(str(action["action"]))
            if "description" in action:
                parts.append(str(action["description"]))
            if "message" in action:
                parts.append(str(action["message"]))
            return " ".join(parts) if parts else str(action)
        return str(action)

    async def encode_async(self, action: Any) -> np.ndarray:
        """
        Encode action to direction vector asynchronously.
        
        Args:
            action: Action dict or text
            
        Returns:
            Unit direction vector in ℝ^target_dim
        """
        text = self._action_to_text(action)
        
        # Check cache
        cache_key = self._get_cache_key(text)
        if self.cache_embeddings and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get embedding from LLM
        if self.llm_provider:
            embedding = await self.llm_provider.embed(text)
            result = self._project_to_target_dim(embedding)
        else:
            # Fallback: hash-based
            result = self._hash_embedding(text)
        
        # Cache result
        if self.cache_embeddings:
            self._cache[cache_key] = result
        
        return result

    def encode(self, action: Any) -> np.ndarray:
        """Synchronous wrapper for encode_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                text = self._action_to_text(action)
                return self._hash_embedding(text)
            return loop.run_until_complete(self.encode_async(action))
        except RuntimeError:
            return asyncio.run(self.encode_async(action))

    def _hash_embedding(self, text: str) -> np.ndarray:
        """Create deterministic embedding from text hash."""
        text_hash = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        rng = np.random.RandomState(text_hash % (2**32))
        embedding = rng.randn(self.target_dim)
        return embedding / np.linalg.norm(embedding)


class OutcomeEncoder:
    """
    Encodes outcomes into state-space vectors.
    
    Similar to ObservationEncoder but for outcomes/transitions.
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        target_dim: int = 64,
        cache_embeddings: bool = True
    ):
        """
        Initialize outcome encoder.
        
        Args:
            llm_provider: LLM provider for embeddings
            target_dim: Target dimension for state vectors
            cache_embeddings: Whether to cache embeddings
        """
        self.llm_provider = llm_provider
        self.target_dim = target_dim
        self.cache_embeddings = cache_embeddings
        self._cache: Dict[str, np.ndarray] = {}
        self._projection_matrix: Optional[np.ndarray] = None
        self._embed_dim: Optional[int] = None

    def _get_cache_key(self, text: str) -> str:
        """Create cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _project_to_target_dim(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding to target dimension."""
        if len(embedding) == self.target_dim:
            return embedding
        
        if self._projection_matrix is None or self._embed_dim != len(embedding):
            self._embed_dim = len(embedding)
            np.random.seed(44)  # Different seed
            self._projection_matrix = np.random.randn(
                len(embedding), self.target_dim
            ) / np.sqrt(self.target_dim)
        
        projected = embedding @ self._projection_matrix
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected = projected / norm
        
        return projected

    async def encode_async(self, outcome: Any) -> np.ndarray:
        """
        Encode outcome asynchronously.
        
        Args:
            outcome: Outcome text or dict
            
        Returns:
            State vector in ℝ^target_dim
        """
        if isinstance(outcome, dict):
            text = str(outcome.get("text", outcome.get("result", outcome)))
        else:
            text = str(outcome)
        
        # Check cache
        cache_key = self._get_cache_key(text)
        if self.cache_embeddings and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get embedding from LLM
        if self.llm_provider:
            embedding = await self.llm_provider.embed(text)
            result = self._project_to_target_dim(embedding)
        else:
            result = self._hash_embedding(text)
        
        if self.cache_embeddings:
            self._cache[cache_key] = result
        
        return result

    def encode(self, outcome: Any) -> np.ndarray:
        """Synchronous wrapper for encode_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                if isinstance(outcome, dict):
                    text = str(outcome.get("text", outcome.get("result", outcome)))
                else:
                    text = str(outcome)
                return self._hash_embedding(text)
            return loop.run_until_complete(self.encode_async(outcome))
        except RuntimeError:
            return asyncio.run(self.encode_async(outcome))

    def _hash_embedding(self, text: str) -> np.ndarray:
        """Create deterministic embedding from text hash."""
        text_hash = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        rng = np.random.RandomState(text_hash % (2**32))
        embedding = rng.randn(self.target_dim)
        return embedding / np.linalg.norm(embedding)
