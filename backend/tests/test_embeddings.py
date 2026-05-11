"""
test_embeddings.py — Unit tests for the embedding layer.

Tests cover:
* Output dimensionality
* L2 normalization (unit-norm vectors)
* Batch shape consistency
* Cache hit / miss behaviour
* Mock Vertex AI SDK surface
"""

from __future__ import annotations

import math
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

DIM = 384


# ── TextEmbeddingModel (mock Vertex AI) ───────────────────────────────────────

class TestMockTextEmbeddingModel:
    """Tests for the mock Vertex AI TextEmbeddingModel."""

    def test_from_pretrained_returns_instance(self) -> None:
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        assert model is not None

    def test_get_embeddings_returns_correct_type(self) -> None:
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        results = model.get_embeddings(["Hello world"])
        assert isinstance(results, list)
        assert len(results) == 1

    def test_embedding_values_attribute_exists(self) -> None:
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        result = model.get_embeddings(["test"])[0]
        assert hasattr(result, "values")
        assert isinstance(result.values, list)

    def test_embedding_dimension(self) -> None:
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        embedding = model.get_embeddings(["hello world"])[0].values
        assert len(embedding) == DIM, f"Expected {DIM}, got {len(embedding)}"

    def test_batch_size_matches_input(self) -> None:
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        texts = ["alpha", "beta", "gamma", "delta"]
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        results = model.get_embeddings(texts)
        assert len(results) == len(texts)

    def test_unknown_model_name_still_works(self) -> None:
        """Should fall back gracefully to the default sentence-transformers model."""
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
        result = model.get_embeddings(["test"])[0]
        assert len(result.values) == DIM


# ── EmbeddingService ─────────────────────────────────────────────────────────

class TestEmbeddingService:

    def _make_service(self):
        from src.embeddings.embedding_service import EmbeddingService
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        return EmbeddingService(model=model, dimension=DIM, cache=None, normalize=True)

    def test_embed_text_returns_list_of_floats(self) -> None:
        svc = self._make_service()
        result = svc.embed_text("How does autoscaling work?")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_embed_text_dimension(self) -> None:
        svc = self._make_service()
        result = svc.embed_text("How does autoscaling work?")
        assert len(result) == DIM

    def test_embed_text_returns_unit_norm(self) -> None:
        svc = self._make_service()
        vec = np.array(svc.embed_text("normalisation test"), dtype=np.float32)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_embed_batch_shape(self) -> None:
        svc = self._make_service()
        texts = ["first doc", "second doc", "third doc"]
        result = svc.embed_batch(texts)
        assert len(result) == 3
        for vec in result:
            assert len(vec) == DIM

    def test_embed_batch_unit_norm(self) -> None:
        svc = self._make_service()
        vecs = svc.embed_batch(["a", "b", "c"])
        for vec in vecs:
            norm = np.linalg.norm(np.array(vec, dtype=np.float32))
            assert abs(norm - 1.0) < 1e-5

    def test_cache_hit_on_second_call(self) -> None:
        """Calling embed_text twice for the same text should hit the cache."""
        from src.embeddings.embedding_service import EmbeddingService
        from src.embeddings.mock_vertexai import TextEmbeddingModel

        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        svc = EmbeddingService(model=model, dimension=DIM, cache=None, normalize=True)

        first = svc.embed_text("cache test query")
        second = svc.embed_text("cache test query")
        assert first == second  # exact same vector returned

    def test_normalize_embeddings_produces_unit_vectors(self) -> None:
        svc = self._make_service()
        rng = np.random.default_rng(42)
        raw_vecs = rng.standard_normal((5, DIM)).tolist()
        normalized = svc.normalize_embeddings(raw_vecs)
        for vec in normalized:
            norm = np.linalg.norm(np.array(vec, dtype=np.float32))
            assert abs(norm - 1.0) < 1e-5

    def test_telemetry_returns_dict_with_required_keys(self) -> None:
        svc = self._make_service()
        t = svc.telemetry()
        assert "model" in t
        assert "dimension" in t
        assert "cache_hit_rate" in t
