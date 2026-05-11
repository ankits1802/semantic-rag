"""
Embedding Service — the central embedding API consumed by the retrieval pipeline.

Responsibilities
----------------
* Wrap the :class:`~src.embeddings.mock_vertexai.TextEmbeddingModel` (or the
  real Vertex AI SDK in production) behind a clean application-level interface.
* Provide single-text and batch embedding with transparent caching.
* Enforce L2 normalisation so that inner-product similarity equals cosine
  similarity (compatible with FAISS ``IndexFlatIP``).
* Expose latency telemetry for benchmarking.

Production migration
--------------------
Replace::

    from src.embeddings.mock_vertexai import TextEmbeddingModel

with::

    from vertexai.language_models import TextEmbeddingModel

The rest of this module stays the same.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.embeddings.embedding_cache import BaseEmbeddingCache, create_cache, _make_cache_key
from src.embeddings.mock_vertexai import TextEmbeddingModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    High-level embedding service with caching and normalisation.

    Parameters
    ----------
    model_name:
        Vertex AI (or sentence-transformers) model identifier.
    batch_size:
        Maximum number of texts to embed in a single model call.
    cache_backend:
        ``"sqlite"`` | ``"pickle"`` | ``"json"`` | ``None`` (disables cache).
    cache_kwargs:
        Additional keyword arguments forwarded to the cache constructor.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        cache_backend: Optional[str] = "sqlite",
        cache_kwargs: Optional[dict] = None,
        # Legacy / test-compatible kwargs
        model=None,             # Accept pre-built TextEmbeddingModel directly
        dimension: Optional[int] = None,   # Hint; derived from model when None
        cache=None,             # Alias for cache_backend when set to None
        normalize: bool = True, # Kept for API parity; normalisation always on
    ) -> None:
        # If a pre-built model object is passed, use it directly
        if model is not None:
            self._model = model
            self.model_name = getattr(model, "model_name", model_name)
        else:
            self._model = TextEmbeddingModel.from_pretrained(model_name)
            self.model_name = model_name

        self.batch_size = batch_size

        # Resolve cache: explicit cache_backend wins; ``cache=None`` disables
        effective_cache_backend: Optional[str] = cache_backend
        if model is not None:
            # When called with legacy kwargs, use ``cache`` param
            effective_cache_backend = None if cache is None else (cache_backend or "sqlite")

        self._cache: Optional[BaseEmbeddingCache] = None
        if effective_cache_backend:
            self._cache = create_cache(effective_cache_backend, **(cache_kwargs or {}))
            logger.info(
                "Embedding cache initialised: backend='%s', entries=%d",
                effective_cache_backend,
                self._cache.size(),
            )

        # telemetry
        self._embed_calls: int = 0
        self._cache_hits: int = 0
        self._total_embed_ms: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Returns
        -------
        List[float]
            1-D normalised float32 vector as a Python list.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")
        results = self.embed_batch([text])
        row = results[0]
        # Return as a Python list for JSON-serialisability and test compatibility
        return row.tolist() if isinstance(row, np.ndarray) else list(row)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts efficiently, using the cache to skip already-
        computed vectors.

        Returns
        -------
        np.ndarray
            2-D normalised float32 array of shape (len(texts), dimension).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        keys = [_make_cache_key(self.model_name, t) for t in texts]
        results: Dict[int, np.ndarray] = {}
        missing_indices: List[int] = []

        # ── Cache lookup ─────────────────────────────────────────────────────
        if self._cache:
            cached = self._cache.get_batch(keys)
            for idx, key in enumerate(keys):
                val = cached.get(key)
                if val is not None:
                    results[idx] = np.array(val, dtype=np.float32)
                    self._cache_hits += 1
                else:
                    missing_indices.append(idx)
        else:
            missing_indices = list(range(len(texts)))

        # ── Embed uncached texts ──────────────────────────────────────────────
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            vectors = self._call_model_batched(missing_texts)
            vectors = self.normalize_embeddings(vectors)

            # Store in cache
            if self._cache:
                cache_batch: dict = {}
                for local_idx, global_idx in enumerate(missing_indices):
                    cache_batch[keys[global_idx]] = vectors[local_idx].tolist()
                self._cache.set_batch(cache_batch, model_name=self.model_name)

            for local_idx, global_idx in enumerate(missing_indices):
                results[global_idx] = vectors[local_idx]

        # ── Assemble in original order ────────────────────────────────────────
        ordered = np.stack([results[i] for i in range(len(texts))], axis=0)
        return ordered.astype(np.float32)

    def normalize_embeddings(self, vectors) -> np.ndarray:
        """
        L2-normalise *vectors* and return as a float32 ndarray.

        Accepts either a 2-D ``np.ndarray`` or a ``List[List[float]]``.
        After normalisation, inner-product similarity equals cosine similarity,
        making FAISS ``IndexFlatIP`` equivalent to cosine search.
        """
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        norms = np.where(norms == 0, 1.0, norms)
        return (arr / norms).astype(np.float32)

    @property
    def dimension(self) -> int:
        """Embedding dimensionality of the underlying model."""
        return self._model.embedding_dimension

    def telemetry(self) -> dict:
        """Return a snapshot of embedding service telemetry counters."""
        total_requests = self._embed_calls
        cache_hit_rate = (
            self._cache_hits / total_requests if total_requests > 0 else 0.0
        )
        avg_ms = (
            self._total_embed_ms / total_requests if total_requests > 0 else 0.0
        )
        return {
            "model": self.model_name,
            "dimension": self.dimension,
            "total_embed_calls": total_requests,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": round(cache_hit_rate, 4),
            "avg_embed_latency_ms": round(avg_ms, 2),
            "cache_entries": self._cache.size() if self._cache else 0,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_model_batched(self, texts: List[str]) -> np.ndarray:
        """
        Call the underlying model in batches of ``self.batch_size`` and
        concatenate the results.
        """
        all_vectors: List[np.ndarray] = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start: batch_start + self.batch_size]
            start = time.perf_counter()
            embeddings = self._model.get_embeddings(batch)
            elapsed_ms = (time.perf_counter() - start) * 1000

            batch_vectors = np.array([e.values for e in embeddings], dtype=np.float32)
            all_vectors.append(batch_vectors)

            self._embed_calls += len(batch)
            self._total_embed_ms += elapsed_ms

            logger.debug(
                "Embedded batch %d–%d in %.1f ms",
                batch_start,
                batch_start + len(batch) - 1,
                elapsed_ms,
            )

        return np.vstack(all_vectors)


# ── Convenience factory ───────────────────────────────────────────────────────

def create_embedding_service(config: Optional[dict] = None) -> EmbeddingService:
    """
    Build an :class:`EmbeddingService` from a config dict (or ``config.yaml``
    if *config* is ``None``).
    """
    if config is None:
        import pathlib
        import yaml
        config_path = pathlib.Path(__file__).resolve().parents[2] / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as fh:
                full_cfg = yaml.safe_load(fh) or {}
            config = full_cfg.get("embedding", {})
        else:
            config = {}

    return EmbeddingService(
        model_name=config.get("model", "all-MiniLM-L6-v2"),
        batch_size=config.get("batch_size", 32),
        cache_backend=config.get("cache_backend", "sqlite"),
    )
