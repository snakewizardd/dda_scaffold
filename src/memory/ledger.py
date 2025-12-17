"""
Experience Ledger for DDA-X

Stores and retrieves experiences with surprise-weighted salience.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import time
from pathlib import Path
import pickle
import lzma
import json


@dataclass
class LedgerEntry:
    """Single entry in the experience ledger."""

    timestamp: float                       # When this happened
    state_vector: np.ndarray               # x_t at decision time
    action_id: str                         # Action taken
    observation_embedding: np.ndarray      # Encoded observation
    outcome_embedding: np.ndarray          # Encoded outcome
    prediction_error: float                # ε_t = ||x_pred - x_actual||
    context_embedding: np.ndarray          # For retrieval similarity

    # Metadata
    task_id: Optional[str] = None
    rigidity_at_time: float = 0.0
    was_successful: Optional[bool] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ReflectionEntry:
    """A learned lesson from experience."""

    timestamp: float
    task_intent: str                       # What we were trying to do
    situation_embedding: np.ndarray        # Embedded (state, action)
    reflection_text: str                   # LLM-generated lesson
    prediction_error: float                # How surprising this was
    outcome_success: bool                  # Did it work?
    metadata: Dict = field(default_factory=dict)


class ExperienceLedger:
    """
    DDA's memory system.

    Retrieval score = sim(c_now, c_t) × e^{-λ_r(now-t)} × (1 + λ_ε × ε_t)
    """

    def __init__(
        self,
        storage_path: Path,
        lambda_recency: float = 0.01,      # Recency decay
        lambda_salience: float = 1.0,      # Salience (surprise) weight
        max_entries: int = 10000,          # Maximum entries to keep in memory
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.lambda_r = lambda_recency
        self.lambda_e = lambda_salience
        self.max_entries = max_entries

        self.entries: List[LedgerEntry] = []
        self.reflections: List[ReflectionEntry] = []

        # Metadata
        self.metadata_file = self.storage_path / "ledger_metadata.json"
        self.stats = {
            "total_entries": 0,
            "total_reflections": 0,
            "avg_prediction_error": 0.0,
            "last_updated": time.time()
        }

        self._load()

    def add_entry(self, entry: LedgerEntry) -> None:
        """Add new experience to ledger."""
        self.entries.append(entry)
        self._save_entry(entry)

        # Update stats
        self.stats["total_entries"] += 1
        self._update_avg_error(entry.prediction_error)

        # Prune if necessary
        if len(self.entries) > self.max_entries:
            self._prune_old_entries()

    def add_reflection(self, reflection: ReflectionEntry) -> None:
        """Add new reflection to memory."""
        self.reflections.append(reflection)
        self._save_reflection(reflection)
        self.stats["total_reflections"] += 1

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        min_score: float = 0.2,
        max_age: Optional[float] = None
    ) -> List[LedgerEntry]:
        """
        Retrieve top-k relevant experiences.

        score = similarity × recency × salience
        """
        now = time.time()
        scored_entries = []

        for entry in self.entries:
            # Skip if too old
            if max_age and (now - entry.timestamp) > max_age:
                continue

            # Cosine similarity
            sim = self._cosine_similarity(query_embedding, entry.context_embedding)

            # Recency decay
            age = now - entry.timestamp
            recency = np.exp(-self.lambda_r * age)

            # Salience (surprise) boost
            salience = 1 + self.lambda_e * entry.prediction_error

            # Combined score
            score = sim * recency * salience

            if score >= min_score:
                scored_entries.append((score, entry))

        # Sort by score descending
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        return [entry for _, entry in scored_entries[:k]]

    def retrieve_reflections(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        min_score: float = 0.25
    ) -> List[ReflectionEntry]:
        """Retrieve relevant learned lessons."""
        scored = []

        for ref in self.reflections:
            sim = self._cosine_similarity(query_embedding, ref.situation_embedding)

            # Boost reflections from surprising situations
            salience = 1 + self.lambda_e * ref.prediction_error
            score = sim * salience

            if score >= min_score:
                scored.append((score, ref))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]

    def get_statistics(self) -> Dict:
        """Get ledger statistics."""
        self.stats["last_updated"] = time.time()
        self.stats["current_entries"] = len(self.entries)
        self.stats["current_reflections"] = len(self.reflections)

        if self.entries:
            recent_errors = [e.prediction_error for e in self.entries[-100:]]
            self.stats["recent_avg_error"] = np.mean(recent_errors)

        return self.stats.copy()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        return np.dot(a, b) / (norm_a * norm_b)

    def _update_avg_error(self, new_error: float) -> None:
        """Update running average of prediction error."""
        n = self.stats["total_entries"]
        old_avg = self.stats["avg_prediction_error"]
        self.stats["avg_prediction_error"] = (old_avg * (n - 1) + new_error) / n

    def _prune_old_entries(self) -> None:
        """Remove oldest entries when over limit."""
        # Sort by timestamp and keep most recent
        self.entries.sort(key=lambda e: e.timestamp)
        to_remove = len(self.entries) - self.max_entries
        self.entries = self.entries[to_remove:]

    def _save_entry(self, entry: LedgerEntry) -> None:
        """Save entry to disk."""
        filename = f"entry_{int(entry.timestamp * 1000)}.pkl.xz"
        path = self.storage_path / filename
        with lzma.open(path, "wb") as f:
            pickle.dump(entry, f)

    def _save_reflection(self, reflection: ReflectionEntry) -> None:
        """Save reflection to disk."""
        filename = f"reflection_{int(reflection.timestamp * 1000)}.pkl.xz"
        path = self.storage_path / filename
        with lzma.open(path, "wb") as f:
            pickle.dump(reflection, f)

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.stats, f, indent=2)

    def _load(self) -> None:
        """Load all entries from storage."""
        # Load entries
        for path in self.storage_path.glob("entry_*.pkl.xz"):
            try:
                with lzma.open(path, "rb") as f:
                    self.entries.append(pickle.load(f))
            except Exception as e:
                print(f"Error loading {path}: {e}")

        # Load reflections
        for path in self.storage_path.glob("reflection_*.pkl.xz"):
            try:
                with lzma.open(path, "rb") as f:
                    self.reflections.append(pickle.load(f))
            except Exception as e:
                print(f"Error loading {path}: {e}")

        # Load metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.stats = json.load(f)

    def __del__(self):
        """Save metadata on cleanup."""
        try:
            self._save_metadata()
        except:
            pass