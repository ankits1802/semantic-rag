"""
Strategy A — Raw Vector Search (Direct Semantic Retrieval).

Flow
----
::

    User Query
        → Embedding Service (normalised dense vector)
        → FAISS Index (cosine similarity)
        → Top-K Results with scores

This is the baseline retrieval strategy.  No query modification is applied;
the raw query is embedded and compared directly against the corpus vectors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.embeddings.embedding_service import EmbeddingService
from src.utils.logger import get_logger
from src.vector_store.base_store import BaseVectorStore, SearchResult

logger = get_logger(__name__)


@dataclass
class StrategyAResult:
    """Complete result from Strategy A retrieval."""

    query: str
    top_k: int
    retrieved_chunks: List[SearchResult]
    latency_ms: float
    embedding_dim: int = 0
    strategy: str = "strategy_a"

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "query": self.query,
            "top_k": self.top_k,
            "latency_ms": round(self.latency_ms, 3),
            "embedding_dim": self.embedding_dim,
            "retrieved_chunks": [r.to_dict() for r in self.retrieved_chunks],
        }


class StrategyA:
    """
    Direct semantic retrieval — embeds the query as-is and searches the
    vector index for the closest chunks.

    Parameters
    ----------
    embedding_service:
        Pre-initialised :class:`~src.embeddings.embedding_service.EmbeddingService`.
    vector_store:
        Pre-built :class:`~src.vector_store.base_store.BaseVectorStore`.
    semantic_threshold:
        Minimum cosine similarity score for a result to be included.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: BaseVectorStore,
        semantic_threshold: float = 0.0,
        config: Optional[dict] = None,
    ) -> None:
        self._embed = embedding_service
        self._store = vector_store
        # Allow config dict to override semantic_threshold
        if config:
            self.semantic_threshold = config.get("retrieval", {}).get(
                "semantic_threshold", semantic_threshold
            )
        else:
            self.semantic_threshold = semantic_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> StrategyAResult:
        """
        Execute a direct vector search for *query*.

        Parameters
        ----------
        query:
            Raw user query string.
        top_k:
            Number of top results to return.

        Returns
        -------
        StrategyAResult
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        self._store.ensure_built()
        t0 = time.perf_counter()

        # Embed query
        query_vector: np.ndarray = self._embed.embed_text(query)

        # Vector search
        results: List[SearchResult] = self._store.search(query_vector, top_k=top_k)

        # Apply semantic threshold filter
        if self.semantic_threshold > 0:
            before = len(results)
            results = [r for r in results if r.score >= self.semantic_threshold]
            if len(results) < before:
                logger.debug(
                    "Threshold filter removed %d low-score results (threshold=%.2f)",
                    before - len(results),
                    self.semantic_threshold,
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[StrategyA] query='%s...' top_k=%d results=%d latency=%.1fms",
            query[:60], top_k, len(results), elapsed_ms,
        )

        return StrategyAResult(
            query=query,
            top_k=top_k,
            retrieved_chunks=results,
            latency_ms=elapsed_ms,
            embedding_dim=self._embed.dimension,
        )
