"""
Strategy B — AI-Enhanced Retrieval (Query Expansion + Optional Reranking).

Flow
----
::

    User Query
        → Query Expansion Engine (mock Gemini)
            ├─ Synonym Expansion
            ├─ Technical Context Injection
            ├─ Multi-query Variant Generation
            └─ (Optional) HyDE passage generation
        → Embedding of expanded query / variants
        → FAISS Vector Search per variant
        → Reciprocal Rank Fusion (RRF) to merge multi-variant results
        → (Optional) Cross-encoder Re-ranking
        → Top-K Final Results

This strategy is the differentiator over Strategy A.  By enriching the query
before embedding, it retrieves chunks that are semantically relevant but may
not share exact terminology with the original query, substantially improving
recall on technical domain queries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.query_expansion import ExpandedQuery, QueryExpansionEngine
from src.utils.logger import get_logger
from src.vector_store.base_store import BaseVectorStore, SearchResult

logger = get_logger(__name__)


@dataclass
class StrategyBResult:
    """Complete result from Strategy B retrieval."""

    query: str
    expanded_query: ExpandedQuery
    top_k: int
    retrieved_chunks: List[SearchResult]
    latency_ms: float
    embedding_dim: int = 0
    reranked: bool = False
    strategy: str = "strategy_b"

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "query": self.query,
            "expanded_query": self.expanded_query.to_dict(),
            "top_k": self.top_k,
            "latency_ms": round(self.latency_ms, 3),
            "embedding_dim": self.embedding_dim,
            "reranked": self.reranked,
            "retrieved_chunks": [r.to_dict() for r in self.retrieved_chunks],
        }


class StrategyB:
    """
    AI-enhanced retrieval with query expansion, multi-query fusion, and
    optional cross-encoder re-ranking.

    Parameters
    ----------
    embedding_service:
        Pre-initialised embedding service.
    vector_store:
        Pre-built vector store.
    expansion_engine:
        Query expansion engine (wraps mock Gemini).
    reranker:
        Optional :class:`~src.retrieval.reranker.CrossEncoderReranker`.
    semantic_threshold:
        Minimum similarity score threshold.
    use_multi_query:
        When ``True``, embed all query variants and merge via RRF.
    rrf_k:
        RRF constant (default 60 per the original paper).
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: BaseVectorStore,
        expansion_engine: Optional[QueryExpansionEngine] = None,
        query_expansion: Optional[QueryExpansionEngine] = None,  # alias for expansion_engine
        reranker=None,
        semantic_threshold: float = 0.0,
        use_multi_query: bool = True,
        rrf_k: int = 60,
        config: Optional[dict] = None,
    ) -> None:
        self._embed = embedding_service
        self._store = vector_store
        # Support both parameter names for backwards compatibility
        engine = expansion_engine or query_expansion
        self._expansion = engine or QueryExpansionEngine()
        self._reranker = reranker
        # Allow config dict to override semantic_threshold
        if config:
            self.semantic_threshold = config.get("retrieval", {}).get(
                "semantic_threshold", semantic_threshold
            )
        else:
            self.semantic_threshold = semantic_threshold
        self.use_multi_query = use_multi_query
        self.rrf_k = rrf_k

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        expansion_mode: Optional[str] = None,
    ) -> StrategyBResult:
        """
        Execute AI-enhanced retrieval for *query*.

        Parameters
        ----------
        query:
            Raw user query string.
        top_k:
            Number of final results to return.
        expansion_mode:
            Override expansion mode for this call.

        Returns
        -------
        StrategyBResult
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        self._store.ensure_built()
        t0 = time.perf_counter()

        # 1. Expand query
        expanded: ExpandedQuery = self._expansion.expand(query, mode=expansion_mode)
        logger.debug(
            "Expanded query: '%s' → '%s...'",
            query[:50], expanded.expanded_query[:80],
        )

        # 2. Retrieve — single expanded query or multi-variant fusion
        if self.use_multi_query and expanded.variants:
            all_queries = list(dict.fromkeys([expanded.expanded_query] + expanded.variants))
            results = self._multi_query_retrieve(all_queries, top_k=top_k * 2)
        else:
            query_vector = self._embed.embed_text(expanded.expanded_query)
            results = self._store.search(query_vector, top_k=top_k * 2)

        # 3. Optional re-ranking
        reranked = False
        if self._reranker is not None and results:
            try:
                results = self._reranker.rerank(query, results, top_n=top_k)
                reranked = True
            except Exception as exc:
                logger.warning("Re-ranking failed, using raw results: %s", exc)

        # 4. Apply threshold and truncate
        if self.semantic_threshold > 0:
            results = [r for r in results if r.score >= self.semantic_threshold]
        results = results[:top_k]

        # Re-assign ranks after fusion/reranking
        for i, r in enumerate(results, start=1):
            r.rank = i

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[StrategyB] query='%s...' expanded='%s...' top_k=%d results=%d latency=%.1fms",
            query[:40], expanded.expanded_query[:40], top_k, len(results), elapsed_ms,
        )

        return StrategyBResult(
            query=query,
            expanded_query=expanded,
            top_k=top_k,
            retrieved_chunks=results,
            latency_ms=elapsed_ms,
            embedding_dim=self._embed.dimension,
            reranked=reranked,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _multi_query_retrieve(
        self, queries: List[str], top_k: int
    ) -> List[SearchResult]:
        """
        Embed each query variant, search independently, then fuse results
        with Reciprocal Rank Fusion (RRF).
        """
        # Embed all variants in one batched call
        vectors = self._embed.embed_batch(queries)

        # Individual result lists
        all_results: List[List[SearchResult]] = []
        for vec in vectors:
            results = self._store.search(vec, top_k=top_k)
            all_results.append(results)

        logger.debug(
            "Multi-query RRF: %d variants, merging result lists",
            len(all_results),
        )
        return self._reciprocal_rank_fusion(all_results, top_k=top_k, rrf_k=self.rrf_k)

    @staticmethod
    def _reciprocal_rank_fusion(
        result_lists: List[List[SearchResult]],
        top_k: int,
        rrf_k: int = 60,
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion (Cormack et al., 2009).

        RRF score for chunk *d* across result lists:

            RRF(d) = Σ  1 / (k + rank(d, list_i))

        where k = 60 (constant preventing high ranks dominating).

        This is a ``@staticmethod`` so it can be tested independently without
        instantiating the full strategy object.
        """
        if not result_lists:
            return []

        rrf_scores: Dict[str, float] = {}
        chunk_objects: Dict[str, Any] = {}

        for result_list in result_lists:
            for result in result_list:
                cid = result.chunk_id
                rank = result.rank
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
                # Keep the instance with the highest raw score as canonical
                if cid not in chunk_objects or result.score > chunk_objects[cid].score:
                    chunk_objects[cid] = result

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        fused: List[SearchResult] = []
        for rank, cid in enumerate(sorted_ids[:top_k], start=1):
            obj = chunk_objects[cid]
            # Use RRF score (normalised) as the display score
            normalised_rrf = min(rrf_scores[cid] / max(len(result_lists), 1), 1.0)
            fused.append(SearchResult(
                chunk_id=getattr(obj, "chunk_id", cid),
                score=normalised_rrf,
                rank=rank,
                text=getattr(obj, "text", ""),
                source=getattr(obj, "source", ""),
                section=getattr(obj, "section", ""),
                metadata=getattr(obj, "metadata", {}),
            ))
        return fused
