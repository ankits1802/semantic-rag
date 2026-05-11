"""
FAISS Vector Store — primary high-performance vector index backend.

Uses Facebook AI Similarity Search (FAISS) with ``IndexFlatIP`` (exact inner
product) on L2-normalised vectors, which is mathematically equivalent to
cosine similarity and is recommended for transformer embeddings.

Index types supported (configured via ``config.yaml``)
------------------------------------------------------
* ``flat_ip``   — ``IndexFlatIP`` — exact brute-force, best recall (default)
* ``flat_l2``   — ``IndexFlatL2`` — exact L2 distance search
* ``ivf_flat``  — ``IndexIVFFlat`` — approximate, fast for >100K vectors

Persistence
-----------
``save_index()`` writes both the FAISS binary index file and a companion JSON
metadata file so chunk mappings survive process restarts.

Production migration note
-------------------------
In production, replace this store with Vertex AI Matching Engine (Vector Search)
for billion-scale ANN retrieval with autoscaling and sub-10ms p99 latency.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, List, Optional

import numpy as np

from src.utils.logger import get_logger
from src.vector_store.base_store import BaseVectorStore, SearchResult

logger = get_logger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-backed vector store supporting three index types.

    Parameters
    ----------
    index_type:
        ``"flat_ip"`` | ``"flat_l2"`` | ``"ivf_flat"``
    nlist:
        Number of Voronoi cells for IVF index (ignored for flat indexes).
    nprobe:
        Number of cells to visit during IVF search (higher → better recall,
        slower).
    """

    def __init__(
        self,
        index_type: str = "flat_ip",
        nlist: int = 100,
        nprobe: int = 10,
    ) -> None:
        super().__init__()
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self._index: Optional[Any] = None  # faiss.Index
        self._dimension: Optional[int] = None

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_index(self, vectors: np.ndarray) -> None:
        """
        Construct the FAISS index from *vectors*.

        Parameters
        ----------
        vectors:
            float32 array of shape (N, D) — must already be L2-normalised for
            ``flat_ip`` to produce cosine similarity scores.
        """
        try:
            import faiss  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from exc

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {vectors.shape}")

        n, d = vectors.shape
        self._dimension = d

        if self.index_type == "flat_ip":
            self._index = faiss.IndexFlatIP(d)
        elif self.index_type == "flat_l2":
            self._index = faiss.IndexFlatL2(d)
        elif self.index_type == "ivf_flat":
            if n < self.nlist:
                logger.warning(
                    "IVF nlist (%d) > num_vectors (%d) — falling back to flat_ip",
                    self.nlist, n,
                )
                self._index = faiss.IndexFlatIP(d)
            else:
                quantiser = faiss.IndexFlatIP(d)
                self._index = faiss.IndexIVFFlat(quantiser, d, self.nlist)
                self._index.train(vectors)  # type: ignore[union-attr]
                self._index.nprobe = self.nprobe  # type: ignore[union-attr]
        else:
            raise ValueError(f"Unknown index_type '{self.index_type}'")

        self._index.add(vectors)  # type: ignore[union-attr]
        self._is_built = True
        logger.info(
            "FAISS index built: type=%s, vectors=%d, dimension=%d",
            self.index_type, n, d,
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Find the *top_k* nearest neighbours of *query_vector*.

        The query vector must be L2-normalised before passing (to get cosine
        similarity rather than raw inner product magnitude).
        """
        self.ensure_built()

        qv = np.asarray(query_vector, dtype=np.float32)
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)

        # Normalise query for cosine similarity
        norm = np.linalg.norm(qv, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        qv = (qv / norm).astype(np.float32)

        k = min(top_k, self._index.ntotal)  # type: ignore[union-attr]
        scores, indices = self._index.search(qv, k)  # type: ignore[union-attr]

        results: List[SearchResult] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:   # FAISS returns -1 for padded results
                continue
            # Clip cosine similarity to [0, 1] (can slightly exceed 1 due to FP)
            clipped_score = float(min(max(score, 0.0), 1.0))
            results.append(
                self._index_to_search_result(int(idx), clipped_score, rank)
            )

        logger.debug(
            "FAISS search: top_k=%d, results=%d, top_score=%.4f",
            top_k, len(results), results[0].score if results else 0.0,
        )
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_index(self, path: str) -> None:
        """
        Save the FAISS index and chunk metadata to *path*.

        Creates two files:
        * ``<path>.index`` — binary FAISS index
        * ``<path>.meta.json`` — JSON chunk metadata
        """
        self.ensure_built()
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu is required for save_index") from exc

        out_path = pathlib.Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        index_file = str(out_path) + ".index"
        meta_file = str(out_path) + ".meta.json"

        faiss.write_index(self._index, index_file)  # type: ignore[arg-type]
        logger.info("Saved FAISS index to '%s'", index_file)

        # Serialise chunk metadata (avoid storing the full Chunk objects)
        meta = {
            "index_type": self.index_type,
            "dimension": self._dimension,
            "num_chunks": len(self._chunks),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "text": c.text,
                    "section": getattr(c, "section", ""),
                    "tokens": getattr(c, "tokens", 0),
                    "metadata": getattr(c, "metadata", {}),
                }
                for c in self._chunks
            ],
        }
        with open(meta_file, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        logger.info("Saved chunk metadata to '%s'", meta_file)

    def load_index(self, path: str) -> None:
        """
        Load a previously saved FAISS index and chunk metadata from *path*.
        """
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu is required for load_index") from exc

        index_file = str(path) + ".index"
        meta_file = str(path) + ".meta.json"

        if not pathlib.Path(index_file).exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")
        if not pathlib.Path(meta_file).exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        self._index = faiss.read_index(index_file)
        logger.info(
            "Loaded FAISS index from '%s' (%d vectors)", index_file, self._index.ntotal
        )

        with open(meta_file, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        self._dimension = meta["dimension"]
        self.index_type = meta["index_type"]

        # Reconstruct lightweight chunk proxies
        from types import SimpleNamespace
        self._chunks = [
            SimpleNamespace(**c) for c in meta["chunks"]
        ]
        self._is_built = True
        logger.info(
            "Loaded %d chunk mappings from '%s'", len(self._chunks), meta_file
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        """Number of vectors currently in the FAISS index."""
        if self._index is None:
            return 0
        return self._index.ntotal  # type: ignore[union-attr]

    def __repr__(self) -> str:
        return (
            f"FAISSVectorStore(type={self.index_type}, "
            f"vectors={self.total_vectors}, "
            f"chunks={self.num_chunks})"
        )
