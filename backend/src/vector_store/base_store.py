"""
Base Vector Store — abstract interface for all vector store backends.

All concrete implementations (FAISS, NumPy, Chroma) must satisfy this contract
so that the retrieval pipeline is backend-agnostic and swappable at runtime
via the ``config.yaml`` ``vector_store.backend`` setting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SearchResult:
    """A single retrieval result returned by any vector store backend."""

    chunk_id: str
    score: float                       # cosine similarity (0–1, higher is better)
    rank: int                          # 1-based rank in this result list
    text: str = ""
    source: str = ""
    section: str = ""
    metadata: Dict[str, Any] = None    # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "rank": self.rank,
            "text": self.text,
            "source": self.source,
            "section": self.section,
            "metadata": self.metadata,
        }


class BaseVectorStore(ABC):
    """
    Abstract interface that every vector store backend must implement.

    The store maps a sequential integer index (0-based) to a chunk.
    ``build_index`` populates the underlying ANN structure; ``search``
    performs similarity lookup; ``save_index`` / ``load_index`` handle
    persistence.
    """

    def __init__(self) -> None:
        self._chunks: List[Any] = []    # List[Chunk] — avoids circular import
        self._is_built: bool = False

    # ── Abstract methods ──────────────────────────────────────────────────────

    @abstractmethod
    def build_index(self, vectors: np.ndarray) -> None:
        """
        Build (or rebuild) the index from the supplied *vectors*.

        Parameters
        ----------
        vectors:
            2-D float32 array of shape (N, dimension) — one row per chunk.
            Vectors must already be L2-normalised.
        """

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Search for the *top_k* most similar chunks.

        Parameters
        ----------
        query_vector:
            1-D or 2-D float32 array (will be reshaped as needed).
        top_k:
            Number of results to return.

        Returns
        -------
        List[SearchResult]
            Ordered from most to least similar.
        """

    @abstractmethod
    def save_index(self, path: str) -> None:
        """Persist the index to *path*."""

    @abstractmethod
    def load_index(self, path: str) -> None:
        """Load a previously saved index from *path*."""

    # ── Concrete helpers ──────────────────────────────────────────────────────

    def register_chunks(self, chunks: List[Any]) -> None:
        """
        Store the list of :class:`~src.ingestion.chunking_engine.Chunk`
        objects that correspond (index-for-index) to the vectors passed to
        :py:meth:`build_index`.
        """
        self._chunks = chunks

    def get_chunk_by_index(self, index: int) -> Optional[Any]:
        """Return the chunk at *index*, or ``None`` if out of range."""
        if 0 <= index < len(self._chunks):
            return self._chunks[index]
        return None

    def _index_to_search_result(
        self, idx: int, score: float, rank: int
    ) -> SearchResult:
        """Convert an index integer + score into a :class:`SearchResult`."""
        chunk = self.get_chunk_by_index(idx)
        if chunk is None:
            return SearchResult(
                chunk_id=f"unknown_{idx}",
                score=score,
                rank=rank,
            )
        return SearchResult(
            chunk_id=chunk.chunk_id,
            score=score,
            rank=rank,
            text=chunk.text,
            source=chunk.source,
            section=getattr(chunk, "section", ""),
            metadata=getattr(chunk, "metadata", {}),
        )

    def ensure_built(self) -> None:
        """Raise :class:`RuntimeError` if the index has not been built yet."""
        if not self._is_built:
            raise RuntimeError(
                "Vector index has not been built. Call build_index() first."
            )

    @property
    def num_chunks(self) -> int:
        """Number of chunks registered in the store."""
        return len(self._chunks)

    def __len__(self) -> int:
        return self.num_chunks
