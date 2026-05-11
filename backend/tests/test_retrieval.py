"""
test_retrieval.py — Integration-level tests for Strategy A, B, and Hybrid search.

These tests use a NumpyVectorStore (no FAISS required) with deterministic
embeddings to validate the full retrieval pipeline end-to-end.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

DIM = 384


def _unit_vec(seed: int, dim: int = DIM) -> List[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ── Strategy A ────────────────────────────────────────────────────────────────

class TestStrategyA:

    @pytest.fixture()
    def strategy_a(self, built_numpy_store, mock_embedding_service, sample_chunks, mock_config):
        from src.retrieval.strategy_a import StrategyA
        return StrategyA(
            vector_store=built_numpy_store,
            embedding_service=mock_embedding_service,
            config=mock_config,
        )

    def test_strategy_a_returns_result(self, strategy_a) -> None:
        from src.retrieval.strategy_a import StrategyAResult
        result = strategy_a.retrieve("How does autoscaling work?", top_k=3)
        assert isinstance(result, StrategyAResult)

    def test_strategy_a_top_k_respected(self, strategy_a) -> None:
        result = strategy_a.retrieve("load balancing", top_k=3)
        assert len(result.retrieved_chunks) <= 3

    def test_strategy_a_results_sorted_descending(self, strategy_a) -> None:
        result = strategy_a.retrieve("rate limiting", top_k=5)
        scores = [r.score for r in result.retrieved_chunks]
        assert scores == sorted(scores, reverse=True)

    def test_strategy_a_to_dict(self, strategy_a) -> None:
        result = strategy_a.retrieve("circuit breaker", top_k=3)
        d = result.to_dict()
        assert "query" in d
        assert "retrieved_chunks" in d
        assert "latency_ms" in d

    def test_strategy_a_latency_positive(self, strategy_a) -> None:
        result = strategy_a.retrieve("test", top_k=1)
        assert result.latency_ms >= 0.0

    def test_strategy_a_strategy_field(self, strategy_a) -> None:
        result = strategy_a.retrieve("test", top_k=1)
        assert result.strategy == "strategy_a"


# ── Strategy B ────────────────────────────────────────────────────────────────

class TestStrategyB:

    @pytest.fixture()
    def strategy_b(self, built_numpy_store, mock_embedding_service, mock_config):
        from src.retrieval.query_expansion import QueryExpansionEngine
        from src.retrieval.reranker import NoOpReranker
        from src.retrieval.strategy_b import StrategyB

        return StrategyB(
            vector_store=built_numpy_store,
            embedding_service=mock_embedding_service,
            query_expansion=QueryExpansionEngine(),
            reranker=NoOpReranker(),
            config=mock_config,
        )

    def test_strategy_b_returns_result(self, strategy_b) -> None:
        from src.retrieval.strategy_b import StrategyBResult
        result = strategy_b.retrieve("How does autoscaling work?", top_k=3)
        assert isinstance(result, StrategyBResult)

    def test_strategy_b_has_expanded_query(self, strategy_b) -> None:
        from src.retrieval.query_expansion import ExpandedQuery
        result = strategy_b.retrieve("peak load", top_k=3)
        assert isinstance(result.expanded_query, ExpandedQuery)
        assert result.expanded_query.original_query == "peak load"

    def test_strategy_b_top_k_respected(self, strategy_b) -> None:
        result = strategy_b.retrieve("latency", top_k=3)
        assert len(result.retrieved_chunks) <= 3

    def test_strategy_b_to_dict(self, strategy_b) -> None:
        result = strategy_b.retrieve("consensus", top_k=2)
        d = result.to_dict()
        assert "expanded_query" in d
        assert "retrieved_chunks" in d

    def test_strategy_b_strategy_field(self, strategy_b) -> None:
        result = strategy_b.retrieve("test", top_k=1)
        assert result.strategy == "strategy_b"


# ── Hybrid Search ─────────────────────────────────────────────────────────────

class TestHybridSearch:

    @pytest.fixture()
    def hybrid(self, built_numpy_store, mock_embedding_service, sample_chunks, mock_config):
        from src.retrieval.hybrid_search import HybridSearch

        h = HybridSearch(
            vector_store=built_numpy_store,
            embedding_service=mock_embedding_service,
            config=mock_config,
        )
        h.register_corpus(sample_chunks)
        return h

    def test_hybrid_search_returns_results(self, hybrid) -> None:
        results = hybrid.search("autoscaling load balancing", top_k=5)
        assert len(results) > 0

    def test_hybrid_search_top_k_respected(self, hybrid) -> None:
        results = hybrid.search("rate limiting", top_k=3)
        assert len(results) <= 3

    def test_hybrid_results_have_final_score(self, hybrid) -> None:
        results = hybrid.search("circuit breaker", top_k=5)
        for r in results:
            assert hasattr(r, "final_score") or hasattr(r, "score")

    def test_hybrid_result_sorted_descending(self, hybrid) -> None:
        results = hybrid.search("kubernetes pod scaling", top_k=5)
        if len(results) > 1:
            scores = [getattr(r, "final_score", r.score) for r in results]
            assert scores == sorted(scores, reverse=True)


# ── RRF fusion ────────────────────────────────────────────────────────────────

class TestRRFFusion:

    def test_rrf_combines_rankings(self) -> None:
        from src.retrieval.strategy_b import StrategyB

        # Minimal mock: just test the static RRF helper
        list1 = [
            SimpleNamespace(chunk_id="a", score=0.9, rank=1, text="a", source="s", section=""),
            SimpleNamespace(chunk_id="b", score=0.7, rank=2, text="b", source="s", section=""),
        ]
        list2 = [
            SimpleNamespace(chunk_id="b", score=0.8, rank=1, text="b", source="s", section=""),
            SimpleNamespace(chunk_id="c", score=0.6, rank=2, text="c", source="s", section=""),
        ]
        # StrategyB._reciprocal_rank_fusion is a method; invoke via the class
        fused = StrategyB._reciprocal_rank_fusion([list1, list2], top_k=3)
        ids = [r.chunk_id for r in fused]
        # "b" appears in both lists — should rank highest
        assert "b" in ids
        assert ids[0] == "b"

    def test_rrf_top_k_respected(self) -> None:
        from src.retrieval.strategy_b import StrategyB

        lists = [
            [SimpleNamespace(chunk_id=str(i), score=1.0 / (i + 1), rank=i + 1, text=f"t{i}", source="s", section="")
             for i in range(5)]
        ]
        fused = StrategyB._reciprocal_rank_fusion(lists, top_k=3)
        assert len(fused) <= 3

    def test_rrf_handles_empty_lists(self) -> None:
        from src.retrieval.strategy_b import StrategyB

        fused = StrategyB._reciprocal_rank_fusion([], top_k=5)
        assert fused == []
