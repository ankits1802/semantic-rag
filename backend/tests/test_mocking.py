"""
test_mocking.py — Tests for the mock Vertex AI SDK surface.

Validates that the mock implementation is a true drop-in replacement for
the real `vertexai` SDK.  Only the import path changes in production.

Covers:
* TextEmbeddingModel API parity with Vertex AI SDK
* GenerativeModel.generate_content() returns GenerationResponse with .text
* HyDE passage generation
* Query expansion via mock GenerativeModel
* Embedding values are real floats (not strings, ints, etc.)
* No external network calls are made
"""

from __future__ import annotations

import socket
from unittest.mock import patch

import numpy as np
import pytest


# ── API surface parity ────────────────────────────────────────────────────────

class TestVertexAIMockParity:
    """Verify the mock mirrors the real SDK public surface."""

    def test_text_embedding_model_has_from_pretrained(self) -> None:
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        assert callable(TextEmbeddingModel.from_pretrained)

    def test_text_embedding_model_has_get_embeddings(self) -> None:
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        assert callable(model.get_embeddings)

    def test_text_embedding_result_has_values(self) -> None:
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        result = model.get_embeddings(["hello"])[0]
        assert hasattr(result, "values")

    def test_generative_model_has_generate_content(self) -> None:
        from src.embeddings.mock_vertexai import GenerativeModel
        gm = GenerativeModel("gemini-3.1-pro-preview")
        assert callable(gm.generate_content)

    def test_generation_response_has_text(self) -> None:
        from src.embeddings.mock_vertexai import GenerativeModel
        gm = GenerativeModel("gemini-3.1-pro-preview")
        response = gm.generate_content("expand: test query")
        assert hasattr(response, "text")
        assert isinstance(response.text, str)

    def test_generation_response_class(self) -> None:
        from src.embeddings.mock_vertexai import GenerationResponse, GenerativeModel
        gm = GenerativeModel("gemini-3.1-pro-preview")
        response = gm.generate_content("some prompt")
        assert isinstance(response, GenerationResponse)


# ── Embedding values ──────────────────────────────────────────────────────────

class TestMockModelEmbeddings:

    def test_embedding_values_are_floats(self) -> None:
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        values = model.get_embeddings(["hello world"])[0].values
        assert all(isinstance(v, float) for v in values), "All values should be float"

    def test_embedding_values_are_finite(self) -> None:
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        values = model.get_embeddings(["finite check"])[0].values
        assert all(np.isfinite(v) for v in values), "All values must be finite"

    def test_different_texts_give_different_embeddings(self) -> None:
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        v1 = model.get_embeddings(["autoscaling"])[0].values
        v2 = model.get_embeddings(["circuit breaker"])[0].values
        assert v1 != v2, "Different texts should produce different embeddings"

    def test_same_text_gives_same_embedding(self) -> None:
        """Deterministic: same input must produce same output."""
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        text = "deterministic embedding test"
        v1 = model.get_embeddings([text])[0].values
        v2 = model.get_embeddings([text])[0].values
        assert v1 == v2

    def test_no_network_call_on_embed(self) -> None:
        """The mock must never open a real TCP connection."""
        from src.embeddings.mock_vertexai import TextEmbeddingModel

        original_connect = socket.socket.connect

        def fail_connect(*args, **kwargs):
            raise AssertionError("Network call detected — mock should not call the internet!")

        with patch.object(socket.socket, "connect", fail_connect):
            model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
            _ = model.get_embeddings(["no network please"])  # must not raise


# ── GenerativeModel mock (query expansion) ────────────────────────────────────

class TestGenerativeModelMock:

    def test_generate_content_expand_query(self) -> None:
        from src.embeddings.mock_vertexai import GenerativeModel
        gm = GenerativeModel("gemini-3.1-pro-preview")
        response = gm.generate_content("expand: How does load balancing work?")
        assert response.text
        assert "load balanc" in response.text.lower() or len(response.text) > 10

    def test_generate_content_hyde(self) -> None:
        from src.embeddings.mock_vertexai import GenerativeModel
        gm = GenerativeModel("gemini-3.1-pro-preview")
        response = gm.generate_content("hyde: What is rate limiting?")
        assert len(response.text) > 20, "HyDE response should be a multi-word passage"

    def test_generate_content_variants(self) -> None:
        from src.embeddings.mock_vertexai import GenerativeModel
        gm = GenerativeModel("gemini-3.1-pro-preview")
        response = gm.generate_content("variants: node failure recovery")
        assert response.text

    def test_no_network_call_on_generate(self) -> None:
        from src.embeddings.mock_vertexai import GenerativeModel

        def fail_connect(*args, **kwargs):
            raise AssertionError("Unexpected network call in GenerativeModel mock")

        with patch.object(socket.socket, "connect", fail_connect):
            gm = GenerativeModel("gemini-3.1-pro-preview")
            _ = gm.generate_content("expand: consensus algorithms")  # must not raise


# ── Mock API parity smoke test ────────────────────────────────────────────────

class TestMockApiParity:
    """Quick smoke tests matching the Vertex AI SDK public contract."""

    def test_from_pretrained_accepts_model_id(self) -> None:
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        for model_id in ["textembedding-gecko@003", "text-embedding-004", "textembedding-gecko@001"]:
            model = TextEmbeddingModel.from_pretrained(model_id)
            assert model is not None

    def test_get_embeddings_accepts_string_list(self) -> None:
        from src.embeddings.mock_vertexai import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        texts = ["a", "b", "c"]
        results = model.get_embeddings(texts)
        assert len(results) == 3

    def test_generative_model_init_with_model_name(self) -> None:
        from src.embeddings.mock_vertexai import GenerativeModel
        for name in ["gemini-3.1-pro-preview", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]:
            gm = GenerativeModel(name)
            assert gm is not None
