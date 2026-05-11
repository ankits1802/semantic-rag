"""
NumPy Vector Store — pure NumPy brute-force cosine similarity backend.

This store is provided as a zero-dependency fallback when FAISS is not
available, and as a reference implementation for unit-testing correctness.
For datasets under ~10,000 chunks it performs adequately; beyond that, the
FAISS backend should be preferred.

Algorithm
---------
Cosine similarity between a query vector **q** and a corpus matrix **M**
(both L2-normalised) reduces to a simple dot product:

    scores = M @ q.T

which NumPy executes via highly optimised BLAS routines.
"""

from __future__ import annotations

import json
import pathlib
import pickle
from typing import Any, List, Optional

import numpy as np

from src.utils.logger import get_logger
from src.vector_store.base_store import BaseVectorStore, SearchResult

logger = get_logger(__name__)


class NumpyVectorStore(BaseVectorStore):
    """
    Pure-NumPy brute-force cosine similarity store.

    All vectors are stored as a 2-D float32 array in memory.  The matrix-
    vector product ``M @ q`` simultaneously computes the cosine similarity
    to every stored vector in one vectorised operation, making this backend
    surprisingly competitive for small corpora.

    Parameters
    ----------
    similarity_metric:
        ``"cosine"`` (default) or ``"euclidean"``.
    """

    def __init__(self, similarity_metric: str = "cosine") -> None:
        super().__init__()
        self.similarity_metric = similarity_metric
        self._vectors: Optional[np.ndarray] = None   # (N, D) float32

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_index(self, vectors: np.ndarray) -> None:
        """
        Store *vectors* in memory.  For cosine search, vectors should be
        pre-normalised (L2) before calling this method.

        Parameters
        ----------
        vectors:
            float32 array of shape (N, D).
        """
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {arr.shape}")

        if self.similarity_metric == "cosine":
            # Normalise for cosine similarity
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._vectors = (arr / norms).astype(np.float32)
        else:
            self._vectors = arr

        self._is_built = True
        logger.info(
            "NumPy index built: metric=%s, vectors=%d, dimension=%d",
            self.similarity_metric, arr.shape[0], arr.shape[1],
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Return the *top_k* most similar chunks to *query_vector*.
        """
        self.ensure_built()

        qv = np.asarray(query_vector, dtype=np.float32).flatten()

        if self.similarity_metric == "cosine":
            # Normalise query
            norm = np.linalg.norm(qv)
            if norm > 0:
                qv = qv / norm
            # Cosine similarity = dot product of normalised vectors
            scores = self._vectors @ qv  # type: ignore[operator]
        else:
            # Euclidean distance (negated so higher = better)
            diffs = self._vectors - qv   # type: ignore[operator]
            scores = -np.linalg.norm(diffs, axis=1)

        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            raw_score = float(scores[idx])
            # Clip cosine similarity to [0, 1]
            if self.similarity_metric == "cosine":
                raw_score = min(max(raw_score, 0.0), 1.0)
            results.append(self._index_to_search_result(int(idx), raw_score, rank))

        logger.debug(
            "NumPy search: top_k=%d, results=%d, top_score=%.4f",
            top_k, len(results), results[0].score if results else 0.0,
        )
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_index(self, path: str) -> None:
        """
        Save vectors and chunk metadata to ``<path>.npy`` and
        ``<path>.meta.pkl``.
        """
        self.ensure_built()
        out_path = pathlib.Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(str(out_path) + ".npy", self._vectors)

        with open(str(out_path) + ".meta.pkl", "wb") as fh:
            pickle.dump(
                {
                    "chunks": self._chunks,
                    "similarity_metric": self.similarity_metric,
                },
                fh,
            )
        logger.info("NumPy index saved to '%s'", path)

    def load_index(self, path: str) -> None:
        """
        Load vectors and chunk metadata from a previously saved path.
        """
        npy_file = str(path) + ".npy"
        meta_file = str(path) + ".meta.pkl"

        if not pathlib.Path(npy_file).exists():
            raise FileNotFoundError(f"NumPy vector file not found: {npy_file}")
        if not pathlib.Path(meta_file).exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        self._vectors = np.load(npy_file).astype(np.float32)
        with open(meta_file, "rb") as fh:
            meta = pickle.load(fh)

        self._chunks = meta["chunks"]
        self.similarity_metric = meta["similarity_metric"]
        self._is_built = True
        logger.info(
            "NumPy index loaded: vectors=%d, dimension=%d",
            self._vectors.shape[0], self._vectors.shape[1],
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        shape = self._vectors.shape if self._vectors is not None else "(not built)"
        return f"NumpyVectorStore(metric={self.similarity_metric}, shape={shape})"


# ── Factory ───────────────────────────────────────────────────────────────────

def create_vector_store(config: Optional[dict] = None) -> BaseVectorStore:
    """
    Instantiate the configured vector store backend.

    Parameters
    ----------
    config:
        ``vector_store`` subsection of ``config.yaml``.  When ``None``,
        defaults to FAISS with flat_ip.
    """
    from src.vector_store.faiss_store import FAISSVectorStore

    cfg = config or {}
    backend = cfg.get("backend", "faiss").lower()
    index_type = cfg.get("index_type", "flat_ip")

    if backend == "faiss":
        return FAISSVectorStore(index_type=index_type)
    elif backend == "numpy":
        metric = cfg.get("similarity_metric", "cosine")
        return NumpyVectorStore(similarity_metric=metric)
    else:
        raise ValueError(f"Unknown vector store backend '{backend}'. Choose: faiss | numpy")
