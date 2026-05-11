"""
Hybrid Search — combines dense vector retrieval with BM25 sparse retrieval.

Motivation
----------
Pure dense retrieval (Strategy A / B) excels at semantic similarity but can
miss exact keyword matches.  BM25 is excellent at exact term matching but
misses semantic paraphrases.  Hybrid search combines both signals for
superior retrieval, especially for technical queries with domain jargon.

Fusion Methods
--------------
1. Weighted Linear Combination:
       hybrid_score(d) = α × dense_score(d) + (1-α) × bm25_score(d)

2. Reciprocal Rank Fusion (RRF):
       rrf_score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_bm25(d))

The method is selected via ``fusion_method`` parameter.

Dependencies
------------
* ``rank-bm25`` — pip install rank-bm25
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.embeddings.embedding_service import EmbeddingService
from src.utils.logger import get_logger
from src.vector_store.base_store import BaseVectorStore, SearchResult

logger = get_logger(__name__)


@dataclass
class HybridSearchResult:
    """Result from hybrid search including component scores."""

    chunk_id: str
    final_score: float
    dense_score: float
    bm25_score: float
    dense_rank: int
    bm25_rank: int
    rank: int
    text: str = ""
    source: str = ""
    section: str = ""
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def to_search_result(self) -> SearchResult:
        return SearchResult(
            chunk_id=self.chunk_id,
            score=self.final_score,
            rank=self.rank,
            text=self.text,
            source=self.source,
            section=self.section,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "final_score": round(self.final_score, 6),
            "dense_score": round(self.dense_score, 6),
            "bm25_score": round(self.bm25_score, 6),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rank": self.rank,
            "text": self.text,
            "source": self.source,
        }


class HybridSearch:
    """
    Hybrid dense + sparse retrieval.

    The BM25 index is built lazily on first call from the chunks registered in
    the vector store, so no additional setup is required.

    Parameters
    ----------
    embedding_service:
        Dense embedding service.
    vector_store:
        Pre-built FAISS/NumPy vector store.
    dense_weight:
        Weight for dense scores in linear fusion (0–1).
    fusion_method:
        ``"linear"`` or ``"rrf"``.
    rrf_k:
        RRF constant (default 60).
    candidate_multiplier:
        Retrieve this many candidates from each source before merging.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: BaseVectorStore,
        dense_weight: float = 0.7,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
        config: Optional[dict] = None,
    ) -> None:
        self._embed = embedding_service
        self._store = vector_store
        # Allow config dict to override constructor defaults
        _cfg = config or {}
        _hs_cfg = _cfg.get("hybrid_search", {})
        self.dense_weight = _hs_cfg.get("dense_weight", dense_weight)
        self.bm25_weight = 1.0 - self.dense_weight
        self.fusion_method = _hs_cfg.get("fusion", fusion_method)
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

        self._bm25 = None            # BM25Okapi instance
        self._bm25_corpus: List[str] = []
        self._bm25_chunk_ids: List[str] = []
        self._extra_corpus: Optional[list] = None  # from register_corpus()

    def register_corpus(self, chunks) -> None:
        """
        Pre-register a list of :class:`~src.ingestion.chunking_engine.Chunk`
        objects for BM25 indexing.

        This is optional — if not called, the BM25 index is built lazily from
        the chunks already registered in the vector store.  Calling this method
        forces an immediate (re)build of the BM25 index from the provided corpus.

        Parameters
        ----------
        chunks:
            List of ``Chunk`` objects to index.
        """
        self._extra_corpus = chunks
        # Force rebuild on next search
        self._bm25 = None
        self._bm25_corpus = [c.text for c in chunks]
        self._bm25_chunk_ids = [c.chunk_id for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            tokenised = [self._tokenise(text) for text in self._bm25_corpus]
            self._bm25 = BM25Okapi(tokenised)
            logger.info("BM25 index built (register_corpus): %d chunks", len(chunks))
        except ImportError:
            logger.warning("rank-bm25 not installed; BM25 index deferred.")

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Execute hybrid search and return the top *top_k* merged results.

        Parameters
        ----------
        query:
            Raw search query.
        top_k:
            Final number of results to return.
        """
        self._store.ensure_built()
        self._ensure_bm25_index()

        t0 = time.perf_counter()
        candidates = top_k * self.candidate_multiplier

        # ── Dense retrieval ───────────────────────────────────────────────────
        query_vec = self._embed.embed_text(query)
        dense_results = self._store.search(query_vec, top_k=candidates)

        # ── BM25 sparse retrieval ─────────────────────────────────────────────
        bm25_results = self._bm25_search(query, top_k=candidates)

        # ── Fusion ────────────────────────────────────────────────────────────
        if self.fusion_method == "rrf":
            merged = self._rrf_fusion(dense_results, bm25_results, top_k=top_k)
        else:
            merged = self._linear_fusion(dense_results, bm25_results, top_k=top_k)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[HybridSearch] query='%s...' fusion=%s top_k=%d results=%d latency=%.1fms",
            query[:50], self.fusion_method, top_k, len(merged), elapsed_ms,
        )
        return [r.to_search_result() for r in merged]

    def search_with_scores(
        self, query: str, top_k: int = 5
    ) -> List[HybridSearchResult]:
        """Like :py:meth:`search` but returns :class:`HybridSearchResult` instances."""
        self._store.ensure_built()
        self._ensure_bm25_index()
        candidates = top_k * self.candidate_multiplier
        query_vec = self._embed.embed_text(query)
        dense_results = self._store.search(query_vec, top_k=candidates)
        bm25_results = self._bm25_search(query, top_k=candidates)

        if self.fusion_method == "rrf":
            return self._rrf_fusion(dense_results, bm25_results, top_k=top_k)
        return self._linear_fusion(dense_results, bm25_results, top_k=top_k)

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _ensure_bm25_index(self) -> None:
        """Build the BM25 index lazily from registered or stored chunks."""
        if self._bm25 is not None:
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is not installed. Run: pip install rank-bm25"
            ) from exc

        # Prefer externally-registered corpus (register_corpus) over store chunks
        if self._extra_corpus:
            chunks = self._extra_corpus
        else:
            chunks = self._store._chunks
        if not chunks:
            raise RuntimeError("No chunks registered in the vector store.")

        self._bm25_corpus = [c.text for c in chunks]
        self._bm25_chunk_ids = [c.chunk_id for c in chunks]

        tokenised = [self._tokenise(text) for text in self._bm25_corpus]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("BM25 index built over %d chunks.", len(chunks))

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Return top-*top_k* BM25 results as :class:`SearchResult` objects."""
        tokenised_query = self._tokenise(query)
        scores = self._bm25.get_scores(tokenised_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        max_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 1.0
        if max_score == 0:
            max_score = 1.0

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            # Normalise BM25 scores to [0, 1]
            normalised = score / max_score
            results.append(
                self._store._index_to_search_result(int(idx), normalised, rank)
            )
        return results

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Simple lowercase whitespace tokenisation with stopword removal."""
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall",
            "this", "that", "these", "those", "it", "its", "as", "if",
        }
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if t not in stopwords and len(t) > 2]

    # ── Fusion methods ────────────────────────────────────────────────────────

    def _rrf_fusion(
        self,
        dense: List[SearchResult],
        bm25: List[SearchResult],
        top_k: int,
    ) -> List[HybridSearchResult]:
        """Reciprocal Rank Fusion of two ranked lists."""
        rrf_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {r.chunk_id: r.rank for r in dense}
        bm25_ranks: Dict[str, int] = {r.chunk_id: r.rank for r in bm25}
        dense_score_map: Dict[str, float] = {r.chunk_id: r.score for r in dense}
        bm25_score_map: Dict[str, float] = {r.chunk_id: r.score for r in bm25}
        chunk_objects: Dict[str, SearchResult] = {
            r.chunk_id: r for r in dense + bm25
        }

        all_ids = set(dense_ranks) | set(bm25_ranks)
        k = self.rrf_k
        for cid in all_ids:
            dr = dense_ranks.get(cid, len(dense) + k)
            br = bm25_ranks.get(cid, len(bm25) + k)
            rrf_scores[cid] = 1.0 / (k + dr) + 1.0 / (k + br)

        sorted_ids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)

        results: List[HybridSearchResult] = []
        for rank, cid in enumerate(sorted_ids[:top_k], start=1):
            obj = chunk_objects[cid]
            # Normalise RRF score
            max_rrf = 2.0 / (k + 1)
            normalised = min(rrf_scores[cid] / max_rrf, 1.0)
            results.append(HybridSearchResult(
                chunk_id=cid,
                final_score=normalised,
                dense_score=dense_score_map.get(cid, 0.0),
                bm25_score=bm25_score_map.get(cid, 0.0),
                dense_rank=dense_ranks.get(cid, 9999),
                bm25_rank=bm25_ranks.get(cid, 9999),
                rank=rank,
                text=obj.text,
                source=obj.source,
                section=obj.section,
                metadata=obj.metadata,
            ))
        return results

    def _linear_fusion(
        self,
        dense: List[SearchResult],
        bm25: List[SearchResult],
        top_k: int,
    ) -> List[HybridSearchResult]:
        """Weighted linear combination of normalised scores."""
        dense_scores = {r.chunk_id: r.score for r in dense}
        bm25_scores = {r.chunk_id: r.score for r in bm25}
        dense_ranks = {r.chunk_id: r.rank for r in dense}
        bm25_ranks = {r.chunk_id: r.rank for r in bm25}
        chunk_objects = {r.chunk_id: r for r in dense + bm25}

        all_ids = set(dense_scores) | set(bm25_scores)
        combined: Dict[str, float] = {}
        for cid in all_ids:
            d = dense_scores.get(cid, 0.0)
            b = bm25_scores.get(cid, 0.0)
            combined[cid] = self.dense_weight * d + self.bm25_weight * b

        sorted_ids = sorted(combined, key=lambda c: combined[c], reverse=True)

        results: List[HybridSearchResult] = []
        for rank, cid in enumerate(sorted_ids[:top_k], start=1):
            obj = chunk_objects[cid]
            results.append(HybridSearchResult(
                chunk_id=cid,
                final_score=combined[cid],
                dense_score=dense_scores.get(cid, 0.0),
                bm25_score=bm25_scores.get(cid, 0.0),
                dense_rank=dense_ranks.get(cid, 9999),
                bm25_rank=bm25_ranks.get(cid, 9999),
                rank=rank,
                text=obj.text,
                source=obj.source,
                section=obj.section,
                metadata=obj.metadata,
            ))
        return results
