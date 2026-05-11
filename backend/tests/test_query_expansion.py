"""
test_query_expansion.py — Unit tests for query expansion & rewriting.

Covers:
* Query rewriting (synonym / technical / full / hyde modes)
* Keyword injection is non-empty
* HyDE passage generation
* Multiple variant generation
* Expansion does not corrupt the original query
* ExpandedQuery dataclass fields
"""

from __future__ import annotations

import pytest

from src.retrieval.query_expansion import ExpandedQuery, QueryExpansionEngine


@pytest.fixture(scope="module")
def engine() -> QueryExpansionEngine:
    return QueryExpansionEngine()


# ── Basic expansion ───────────────────────────────────────────────────────────

class TestQueryExpansion:

    def test_expand_returns_expanded_query_instance(self, engine: QueryExpansionEngine) -> None:
        result = engine.expand("How does autoscaling work?")
        assert isinstance(result, ExpandedQuery)

    def test_expanded_query_original_preserved(self, engine: QueryExpansionEngine) -> None:
        q = "rate limiting strategies"
        result = engine.expand(q)
        assert result.original_query == q

    def test_expanded_query_is_non_empty(self, engine: QueryExpansionEngine) -> None:
        result = engine.expand("peak load handling")
        assert len(result.expanded_query) > 0

    def test_synonym_expansion_adds_keywords(self, engine: QueryExpansionEngine) -> None:
        """For a known synonym query, the expanded text should contain extra terms."""
        result = engine.expand("latency", mode="synonyms")
        # should add synonyms like "response time", "delay", etc.
        assert len(result.keywords_added) > 0 or len(result.expanded_query) > len("latency")

    def test_technical_expansion_works(self, engine: QueryExpansionEngine) -> None:
        result = engine.expand("kubernetes scaling", mode="technical")
        assert isinstance(result, ExpandedQuery)
        assert result.expanded_query  # non-empty

    def test_full_expansion_mode(self, engine: QueryExpansionEngine) -> None:
        result = engine.expand("How does Raft consensus work?", mode="full")
        assert isinstance(result, ExpandedQuery)

    def test_expansion_mode_stored_in_result(self, engine: QueryExpansionEngine) -> None:
        for mode in ["synonyms", "technical", "full", "hyde"]:
            result = engine.expand("test query", mode=mode)
            assert result.expansion_mode == mode

    def test_expansion_not_empty_for_various_queries(self, engine: QueryExpansionEngine) -> None:
        queries = [
            "peak load",
            "node failure",
            "consistent hashing",
            "circuit breaker pattern",
            "kubernetes autoscaling",
        ]
        for q in queries:
            result = engine.expand(q)
            assert result.expanded_query, f"Expansion was empty for: {q}"


# ── Variant generation ────────────────────────────────────────────────────────

class TestVariantGeneration:

    def test_generate_variants_returns_list(self, engine: QueryExpansionEngine) -> None:
        variants = engine.generate_variants("How does caching improve performance?")
        assert isinstance(variants, list)

    def test_generate_variants_non_empty(self, engine: QueryExpansionEngine) -> None:
        variants = engine.generate_variants("database replication")
        assert len(variants) >= 1

    def test_generate_variants_max_count(self, engine: QueryExpansionEngine) -> None:
        variants = engine.generate_variants("load balancing", max_variants=2)
        assert len(variants) <= 2

    def test_variants_are_strings(self, engine: QueryExpansionEngine) -> None:
        variants = engine.generate_variants("service mesh")
        for v in variants:
            assert isinstance(v, str)
            assert len(v) > 0


# ── HyDE ─────────────────────────────────────────────────────────────────────

class TestHyDE:

    def test_hyde_generates_passage(self, engine: QueryExpansionEngine) -> None:
        result = engine.expand("What is a circuit breaker?", mode="hyde")
        assert result.hyde_passage is not None
        assert len(result.hyde_passage) > 0

    def test_hyde_passage_is_string(self, engine: QueryExpansionEngine) -> None:
        result = engine.expand("fault tolerance techniques", mode="hyde")
        assert isinstance(result.hyde_passage, str)

    def test_hyde_expanded_query_non_empty(self, engine: QueryExpansionEngine) -> None:
        result = engine.expand("rate limiting algorithms", mode="hyde")
        assert result.expanded_query


# ── Rewrite (simple string) ────────────────────────────────────────────────────

class TestQueryRewrite:

    def test_rewrite_returns_string(self, engine: QueryExpansionEngine) -> None:
        result = engine.rewrite("How does autoscaling work?")
        assert isinstance(result, str)

    def test_rewrite_non_empty(self, engine: QueryExpansionEngine) -> None:
        result = engine.rewrite("consensus protocol")
        assert result
