"""
test_vector_search.py — Tests for vector store backends.

Covers:
* FAISS flat index similarity search
* NumPy brute-force search
* Top-K result ordering
* Semantic threshold filtering
* Save/load round-trip
* Cosine similarity correctness
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import List

import numpy as np
import pytest

DIM = 384


def _unit_vec(seed: int, dim: int = DIM) -> List[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ── NumPy store ───────────────────────────────────────────────────────────────

class TestNumpyVectorStore:

    def test_search_returns_top_k_results(self, built_numpy_store) -> None:
        query_vec = _unit_vec(0)  # seed 0 == first chunk embedding
        results = built_numpy_store.search(query_vec, top_k=3)
        assert len(results) == 3

    def test_search_result_is_ranked(self, built_numpy_store) -> None:
        query_vec = _unit_vec(0)
        results = built_numpy_store.search(query_vec, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results must be descending by score"

    def test_first_result_is_best_match(self, built_numpy_store) -> None:
        """Querying with the embedding of chunk 0 should return chunk 0 first."""
        query_vec = _unit_vec(0)
        results = built_numpy_store.search(query_vec, top_k=5)
        assert results[0].chunk_id == "test_doc_chunk_0"

    def test_cosine_score_in_valid_range(self, built_numpy_store) -> None:
        query_vec = _unit_vec(7)
        results = built_numpy_store.search(query_vec, top_k=5)
        for r in results:
            assert -1.0 <= r.score <= 1.0, f"Score out of range: {r.score}"

    def test_top_k_respected(self, built_numpy_store) -> None:
        for k in [1, 3, 5, 10]:
            results = built_numpy_store.search(_unit_vec(0), top_k=k)
            assert len(results) <= k

    def test_ranks_are_sequential(self, built_numpy_store) -> None:
        results = built_numpy_store.search(_unit_vec(0), top_k=5)
        for i, r in enumerate(results, start=1):
            assert r.rank == i

    def test_search_with_threshold_filters_low_scores(self, sample_chunks) -> None:
        from src.vector_store.numpy_store import NumpyVectorStore

        store = NumpyVectorStore()
        store.register_chunks(sample_chunks)
        # Use orthogonal-ish vectors so scores are low
        rng = np.random.default_rng(99)
        vecs = [rng.standard_normal(DIM).astype(np.float32) for _ in sample_chunks]
        vecs = [(v / np.linalg.norm(v)).tolist() for v in vecs]
        store.build_index(vecs)

        query = _unit_vec(999)
        results_no_thresh = store.search(query, top_k=5)
        results_thresh = store.search(query, top_k=5, threshold=0.99)  # very high threshold
        assert len(results_thresh) < len(results_no_thresh) or len(results_thresh) == 0

    def test_save_and_load_round_trip(self, built_numpy_store) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(pathlib.Path(tmpdir) / "store")
            built_numpy_store.save_index(path)

            from src.vector_store.numpy_store import NumpyVectorStore
            loaded = NumpyVectorStore()
            loaded.load_index(path)

            original = built_numpy_store.search(_unit_vec(3), top_k=3)
            reloaded = loaded.search(_unit_vec(3), top_k=3)

            assert [r.chunk_id for r in original] == [r.chunk_id for r in reloaded]


# ── FAISS store ───────────────────────────────────────────────────────────────

class TestFAISSVectorStore:

    def test_faiss_search_returns_top_k(self, built_faiss_store) -> None:
        results = built_faiss_store.search(_unit_vec(0), top_k=3)
        assert len(results) == 3

    def test_faiss_first_result_correct(self, built_faiss_store) -> None:
        results = built_faiss_store.search(_unit_vec(0), top_k=5)
        assert results[0].chunk_id == "test_doc_chunk_0"

    def test_faiss_scores_descending(self, built_faiss_store) -> None:
        results = built_faiss_store.search(_unit_vec(2), top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_faiss_cosine_score_valid_range(self, built_faiss_store) -> None:
        results = built_faiss_store.search(_unit_vec(1), top_k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"FAISS cosine score out of [0,1]: {r.score}"

    def test_faiss_save_and_load(self, built_faiss_store) -> None:
        pytest.importorskip("faiss")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(pathlib.Path(tmpdir) / "faiss_store")
            built_faiss_store.save_index(path)

            from src.vector_store.faiss_store import FAISSVectorStore
            loaded = FAISSVectorStore()
            loaded.load_index(path)

            original = built_faiss_store.search(_unit_vec(4), top_k=3)
            reloaded = loaded.search(_unit_vec(4), top_k=3)
            assert [r.chunk_id for r in original] == [r.chunk_id for r in reloaded]


# ── SearchResult dataclass ────────────────────────────────────────────────────

class TestSearchResult:

    def test_to_dict_keys(self, built_numpy_store) -> None:
        results = built_numpy_store.search(_unit_vec(0), top_k=1)
        d = results[0].to_dict()
        assert "chunk_id" in d
        assert "score" in d
        assert "rank" in d
        assert "text" in d
