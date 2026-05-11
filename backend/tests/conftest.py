"""
conftest.py — Shared pytest fixtures for the entire test suite.

All fixtures are designed to be fast (no I/O, fully in-memory) so
the test suite can run offline without any external dependencies.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings.embedding_service import EmbeddingService
from src.ingestion.chunking_engine import Chunk
from src.vector_store.base_store import SearchResult


# ── Constants ────────────────────────────────────────────────────────────────

DIM = 384  # sentence-transformers/all-MiniLM-L6-v2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unit_vector(seed: int, dim: int = DIM) -> List[float]:
    """Reproducible unit vector for a given seed."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()


def _make_chunk(index: int, text: str, source: str = "test_doc.txt") -> Chunk:
    return Chunk(
        chunk_id=f"test_doc_chunk_{index}",
        source=source,
        text=text,
        tokens=len(text.split()),
        char_start=0,
        char_end=len(text),
        chunk_index=index,
        section="",
        metadata={},
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_chunk_texts() -> List[str]:
    return [
        "Autoscaling adjusts the number of running instances based on traffic load.",
        "Load balancing distributes requests across multiple backend servers.",
        "Circuit breakers prevent cascading failures in distributed systems.",
        "Rate limiting controls the number of requests a client can make per second.",
        "Consensus algorithms like Raft and Paxos ensure distributed agreement.",
        "Kubernetes uses HPA to automatically scale pods based on CPU utilisation.",
        "Caching with Redis reduces database query latency significantly.",
        "Service discovery allows microservices to find each other at runtime.",
        "BM25 is a term-frequency based sparse retrieval algorithm.",
        "Dense vector retrieval uses cosine similarity in high-dimensional space.",
    ]


@pytest.fixture(scope="session")
def sample_chunks(sample_chunk_texts: List[str]) -> List[Chunk]:
    return [_make_chunk(i, text) for i, text in enumerate(sample_chunk_texts)]


@pytest.fixture(scope="session")
def sample_embeddings(sample_chunk_texts: List[str]) -> List[List[float]]:
    """Deterministic unit vectors, one per chunk text."""
    return [_unit_vector(i) for i in range(len(sample_chunk_texts))]


@pytest.fixture()
def mock_embedding_service(sample_chunk_texts: List[str], sample_embeddings: List[List[float]]) -> EmbeddingService:
    """
    An EmbeddingService backed by a MagicMock model that returns
    pre-computed deterministic unit vectors.
    """
    text_to_vec = {text: vec for text, vec in zip(sample_chunk_texts, sample_embeddings)}

    service = MagicMock(spec=EmbeddingService)
    service.dimension = DIM

    def _embed_text(text: str, **_kwargs) -> List[float]:
        return text_to_vec.get(text, _unit_vector(hash(text) % 1000))

    def _embed_batch(texts: List[str], **_kwargs) -> List[List[float]]:
        return [_embed_text(t) for t in texts]

    def _normalize(vecs: List[List[float]]) -> List[List[float]]:
        result = []
        for v in vecs:
            arr = np.array(v, dtype=np.float32)
            norm = np.linalg.norm(arr)
            result.append((arr / norm if norm > 0 else arr).tolist())
        return result

    service.embed_text.side_effect = _embed_text
    service.embed_batch.side_effect = _embed_batch
    service.normalize_embeddings.side_effect = _normalize
    service.telemetry.return_value = {
        "model": "mock-model",
        "dimension": DIM,
        "cache_entries": 0,
        "cache_hit_rate": 0.0,
    }
    return service


@pytest.fixture()
def built_numpy_store(sample_chunks: List[Chunk], sample_embeddings: List[List[float]]):
    """A NumpyVectorStore pre-built with sample chunks."""
    from src.vector_store.numpy_store import NumpyVectorStore

    store = NumpyVectorStore()
    store.register_chunks(sample_chunks)
    store.build_index(sample_embeddings)
    return store


@pytest.fixture()
def built_faiss_store(sample_chunks: List[Chunk], sample_embeddings: List[List[float]]):
    """A FAISSVectorStore pre-built with sample chunks."""
    pytest.importorskip("faiss")
    from src.vector_store.faiss_store import FAISSVectorStore

    store = FAISSVectorStore()
    store.register_chunks(sample_chunks)
    store.build_index(sample_embeddings)
    return store


@pytest.fixture()
def mock_config() -> dict:
    return {
        "embedding": {
            "model": "textembedding-gecko@003",
            "batch_size": 5,
            "dimension": DIM,
            "normalize": True,
            "cache_backend": "none",
        },
        "chunking": {"chunk_size": 256, "chunk_overlap": 32, "min_chunk_size": 20},
        "retrieval": {"top_k": 5, "similarity_metric": "cosine", "semantic_threshold": 0.0},
        "vector_store": {"backend": "numpy", "index_type": "flat_ip"},
        "query_expansion": {
            "enabled": True,
            "model": "gemini-3.1-pro-preview",
            "expansion_type": "full",
            "max_variants": 2,
        },
        "hybrid_search": {"fusion": "rrf", "dense_weight": 0.7, "bm25_weight": 0.3},
        "reranking": {"enabled": False, "model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "top_n_rerank": 10},
        "benchmarking": {"metrics": ["precision", "recall", "mrr", "ndcg"], "k_values": [1, 3, 5]},
    }
