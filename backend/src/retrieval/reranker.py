"""
Cross-Encoder Re-ranking Layer.

After initial retrieval (dense / hybrid) produces a candidate set, the
re-ranker scores each (query, chunk) pair with a cross-encoder model that
jointly encodes both texts and produces a relevance score.  Cross-encoders
are slower than bi-encoders but far more accurate.

Architecture
------------
::

    Top-N candidates (from initial retrieval)
        → Cross-encoder: score(query, chunk_i) for each i
        → Sort by cross-encoder score descending
        → Return Top-K

Supported models
----------------
* ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (default, ~22MB, fast)
* ``cross-encoder/ms-marco-MiniLM-L-12-v2`` (larger, more accurate)
* ``BAAI/bge-reranker-base``
* Any HuggingFace cross-encoder model

Graceful degradation
--------------------
If the cross-encoder model is not available or fails, the module falls back
to the original similarity scores from the initial retrieval stage.
"""

from __future__ import annotations

import time
from typing import List, Optional

from src.utils.logger import get_logger
from src.vector_store.base_store import SearchResult

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder based re-ranker for search result refinement.

    Parameters
    ----------
    model_name:
        HuggingFace cross-encoder model identifier.
    max_length:
        Maximum token length for the cross-encoder input.
    device:
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    batch_size:
        Number of (query, passage) pairs to score per forward pass.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        device: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.batch_size = batch_size
        self._model = None  # lazy load

    def _load(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading cross-encoder: '%s'", self.model_name)
                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load cross-encoder '%s': %s. Reranking disabled.",
                    self.model_name, exc,
                )
                self._model = None

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_n: int = 5,
    ) -> List[SearchResult]:
        """
        Re-rank *candidates* for *query* and return top *top_n*.

        Parameters
        ----------
        query:
            Original user query (not expanded).
        candidates:
            Initial retrieval results (may be more than *top_n*).
        top_n:
            Number of re-ranked results to return.

        Returns
        -------
        List[SearchResult]
            Re-ranked and truncated results.
        """
        if not candidates:
            return []

        self._load()

        if self._model is None:
            logger.warning("Cross-encoder unavailable — returning original ranking.")
            return candidates[:top_n]

        t0 = time.perf_counter()

        # Build (query, passage) pairs
        pairs = [(query, r.text) for r in candidates]

        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.warning("Cross-encoder prediction failed: %s", exc)
            return candidates[:top_n]

        # Attach cross-encoder scores and sort
        import numpy as np
        scores_np = np.array(scores, dtype=float)

        # Normalise cross-encoder scores to [0, 1] using sigmoid
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        normalised = sigmoid(scores_np)

        ranked_indices = np.argsort(normalised)[::-1][:top_n]

        reranked: List[SearchResult] = []
        for new_rank, orig_idx in enumerate(ranked_indices, start=1):
            candidate = candidates[int(orig_idx)]
            reranked.append(SearchResult(
                chunk_id=candidate.chunk_id,
                score=float(normalised[orig_idx]),
                rank=new_rank,
                text=candidate.text,
                source=candidate.source,
                section=candidate.section,
                metadata=dict(candidate.metadata, cross_encoder_score=float(normalised[orig_idx])),
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Cross-encoder reranked %d→%d candidates in %.1fms",
            len(candidates), len(reranked), elapsed_ms,
        )
        return reranked

    def is_available(self) -> bool:
        """Return ``True`` if the cross-encoder loaded successfully."""
        self._load()
        return self._model is not None


class NoOpReranker:
    """
    Passthrough reranker that returns candidates unchanged.
    Used when re-ranking is disabled in config.
    """

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_n: int = 5,
    ) -> List[SearchResult]:
        return candidates[:top_n]

    def is_available(self) -> bool:
        return True


def create_reranker(config: Optional[dict] = None):
    """
    Create the appropriate reranker from config.

    Parameters
    ----------
    config:
        ``reranking`` subsection of ``config.yaml``.
    """
    cfg = config or {}
    if not cfg.get("enabled", True):
        return NoOpReranker()
    return CrossEncoderReranker(
        model_name=cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        batch_size=cfg.get("batch_size", 16),
    )
