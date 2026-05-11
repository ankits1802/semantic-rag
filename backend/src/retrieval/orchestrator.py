"""
ContextAwareRetriever — central pipeline orchestrator.

This class is the single entry point for the entire RAG system.  It ties
together document ingestion, chunking, embedding, vector indexing, retrieval
(both strategies), hybrid search, and benchmarking into a cohesive API that
can be driven from the CLI, the FastAPI server, or tests.

Responsibilities
----------------
* ingest_documents()  — load, preprocess, and chunk source documents
* build_embeddings()  — embed all chunks using the embedding service
* create_index()      — build the FAISS vector index
* retrieve_raw()      — Strategy A retrieval
* retrieve_enhanced() — Strategy B retrieval
* retrieve_hybrid()   — Hybrid BM25 + dense retrieval
* benchmark_queries() — run the full benchmark suite
* save_results()      — persist benchmark output as JSON
* save_index() / load_index() — persist/restore the vector index
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from src.embeddings.embedding_service import EmbeddingService, create_embedding_service
from src.ingestion.chunking_engine import Chunk, ChunkingEngine
from src.ingestion.document_loader import DocumentLoader, RawDocument
from src.ingestion.preprocessor import TextPreprocessor
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.query_expansion import QueryExpansionEngine
from src.retrieval.reranker import CrossEncoderReranker, NoOpReranker, create_reranker
from src.retrieval.strategy_a import StrategyA, StrategyAResult
from src.retrieval.strategy_b import StrategyB, StrategyBResult
from src.utils.logger import get_logger
from src.vector_store.base_store import BaseVectorStore, SearchResult
from src.vector_store.faiss_store import FAISSVectorStore
from src.vector_store.numpy_store import NumpyVectorStore, create_vector_store

logger = get_logger(__name__)


@dataclass
class HybridResult:
    """Result wrapper for hybrid search, providing the same interface as Strategy A/B results."""

    query: str
    top_k: int
    retrieved_chunks: List[SearchResult]
    latency_ms: float
    strategy: str = "hybrid"

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "query": self.query,
            "top_k": self.top_k,
            "latency_ms": round(self.latency_ms, 3),
            "retrieved_chunks": [r.to_dict() for r in self.retrieved_chunks],
        }


class ContextAwareRetriever:
    """
    End-to-end context-aware retrieval pipeline.

    Parameters
    ----------
    config_path:
        Path to ``config.yaml``.  When ``None``, default values are used.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._cfg = self._load_config(config_path)
        self._chunks: List[Chunk] = []
        self._vectors: Optional[np.ndarray] = None

        # ── Sub-components ───────────────────────────────────────────────────
        self._loader = DocumentLoader()
        self._preprocessor = TextPreprocessor()
        self._chunker = ChunkingEngine(
            chunk_size=self._cfg["chunking"]["chunk_size"],
            chunk_overlap=self._cfg["chunking"]["chunk_overlap"],
            min_chunk_size=self._cfg["chunking"]["min_chunk_size"],
        )
        self._embedding_service: EmbeddingService = create_embedding_service(
            self._cfg.get("embedding")
        )
        self._vector_store: BaseVectorStore = create_vector_store(
            self._cfg.get("vector_store")
        )
        self._expansion_engine = QueryExpansionEngine(
            model_name=self._cfg["query_expansion"].get("model", "gemini-3.1-pro-preview"),
            expansion_type=self._cfg["query_expansion"].get("expansion_type", "full"),
            max_variants=self._cfg["query_expansion"].get("max_variants", 3),
        )
        reranker_cfg = self._cfg.get("reranking", {})
        self._reranker = create_reranker(reranker_cfg)

        # Build strategy objects (populated after index is created)
        self._strategy_a: Optional[StrategyA] = None
        self._strategy_b: Optional[StrategyB] = None
        self._hybrid: Optional[HybridSearch] = None

        logger.info(
            "ContextAwareRetriever initialised: model=%s store=%s",
            self._cfg["embedding"]["model"],
            self._cfg["vector_store"]["backend"],
        )

    # ── Pipeline stages ──────────────────────────────────────────────────────

    def ingest_documents(
        self,
        paths: Optional[List[Union[str, pathlib.Path]]] = None,
        inline_texts: Optional[List[str]] = None,
        documents: Optional[List[dict]] = None,
    ) -> List[Chunk]:
        """
        Load, preprocess, and chunk source documents.

        Parameters
        ----------
        paths:
            List of file paths or directory paths.
        inline_texts:
            List of raw strings to ingest directly.
        documents:
            List of dicts with ``content``/``text`` keys (JSON-style).

        Returns
        -------
        List[Chunk]
            All produced chunks across all documents.
        """
        raw_docs: List[RawDocument] = []

        # Load from files/directories
        if paths:
            for p in paths:
                p = pathlib.Path(p)
                if p.is_dir():
                    raw_docs.extend(self._loader.load_directory(p))
                else:
                    raw_docs.extend(self._loader.load_file(p))

        # Load inline texts
        if inline_texts:
            for i, text in enumerate(inline_texts):
                raw_docs.append(
                    self._loader.load_text(text, doc_id=f"inline_{i}")
                )

        # Load JSON-style document list
        if documents:
            raw_docs.extend(self._loader.load_documents_from_list(documents))

        if not raw_docs:
            raise ValueError("No documents provided for ingestion.")

        logger.info("Ingesting %d raw documents...", len(raw_docs))

        # Preprocess each document
        for doc in raw_docs:
            doc.text = self._preprocessor.preprocess(doc.text)

        # Chunk
        self._chunks = self._chunker.chunk_documents(raw_docs)
        logger.info("Total chunks after ingestion: %d", len(self._chunks))
        return self._chunks

    def build_embeddings(self) -> np.ndarray:
        """
        Embed all ingested chunks and cache the resulting vectors.

        Returns
        -------
        np.ndarray
            float32 array of shape (num_chunks, embedding_dim).
        """
        if not self._chunks:
            raise RuntimeError("No chunks found. Call ingest_documents() first.")

        logger.info("Building embeddings for %d chunks...", len(self._chunks))
        t0 = time.perf_counter()

        texts = [c.text for c in self._chunks]
        self._vectors = self._embedding_service.embed_batch(texts)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Embeddings built: shape=%s, time=%.2fs (telemetry: %s)",
            self._vectors.shape, elapsed,
            self._embedding_service.telemetry(),
        )
        return self._vectors

    def create_index(self) -> None:
        """
        Build the vector index from precomputed embeddings and register
        chunk metadata.
        """
        if self._vectors is None:
            raise RuntimeError("No embeddings found. Call build_embeddings() first.")

        logger.info("Building vector index...")
        self._vector_store.build_index(self._vectors)
        self._vector_store.register_chunks(self._chunks)

        # Initialise strategy objects
        threshold = self._cfg["retrieval"].get("semantic_threshold", 0.0)
        self._strategy_a = StrategyA(
            embedding_service=self._embedding_service,
            vector_store=self._vector_store,
            semantic_threshold=threshold,
        )
        self._strategy_b = StrategyB(
            embedding_service=self._embedding_service,
            vector_store=self._vector_store,
            expansion_engine=self._expansion_engine,
            reranker=self._reranker,
            semantic_threshold=threshold,
            use_multi_query=True,
        )
        self._hybrid = HybridSearch(
            embedding_service=self._embedding_service,
            vector_store=self._vector_store,
            dense_weight=self._cfg["hybrid_search"].get("dense_weight", 0.7),
            fusion_method="rrf",
        )
        logger.info("Vector index ready. %d vectors indexed.", len(self._chunks))

    def setup(self, data_dir: Optional[str] = None) -> None:
        """
        Convenience method: ingest → embed → index in one call.

        Parameters
        ----------
        data_dir:
            Directory containing documents.  Defaults to ``./data/documents``.
        """
        if data_dir is None:
            data_dir = str(pathlib.Path(__file__).resolve().parents[2] / "data" / "documents")
        self.ingest_documents(paths=[data_dir])
        self.build_embeddings()
        self.create_index()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_raw(
        self, query: str, top_k: Optional[int] = None
    ) -> StrategyAResult:
        """Strategy A — direct vector search."""
        self._require_index()
        k = top_k or self._cfg["retrieval"]["top_k"]
        return self._strategy_a.retrieve(query, top_k=k)  # type: ignore[union-attr]

    def retrieve_enhanced(
        self, query: str, top_k: Optional[int] = None, mode: Optional[str] = None
    ) -> StrategyBResult:
        """Strategy B — query expansion + vector search + reranking."""
        self._require_index()
        k = top_k or self._cfg["retrieval"]["top_k"]
        return self._strategy_b.retrieve(query, top_k=k, expansion_mode=mode)  # type: ignore[union-attr]

    def retrieve_hybrid(
        self, query: str, top_k: Optional[int] = None
    ) -> HybridResult:
        """Hybrid dense + BM25 search with RRF fusion."""
        self._require_index()
        k = top_k or self._cfg["retrieval"]["top_k"]
        t0 = time.perf_counter()
        chunks = self._hybrid.search(query, top_k=k)  # type: ignore[union-attr]
        latency_ms = (time.perf_counter() - t0) * 1000
        return HybridResult(
            query=query,
            top_k=k,
            retrieved_chunks=chunks,
            latency_ms=latency_ms,
        )

    # ── Benchmarking ─────────────────────────────────────────────────────────

    def benchmark_queries(self, queries: List[str]) -> List[dict]:
        """
        Run both strategies on each query and return structured comparison dicts.

        Parameters
        ----------
        queries:
            List of benchmark query strings.

        Returns
        -------
        List[dict]
            One dict per query containing side-by-side results from both strategies.
        """
        self._require_index()
        results = []
        for query in queries:
            logger.info("Benchmarking query: '%s'", query)
            t0 = time.perf_counter()
            result_a = self.retrieve_raw(query)
            result_b = self.retrieve_enhanced(query)
            elapsed = (time.perf_counter() - t0) * 1000
            results.append({
                "query": query,
                "total_latency_ms": round(elapsed, 2),
                "strategy_a": result_a.to_dict(),
                "strategy_b": result_b.to_dict(),
            })
        return results

    def save_results(self, results: List[dict], output_path: str) -> None:
        """Persist benchmark results as a JSON file."""
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        logger.info("Benchmark results saved to '%s'", output_path)

    # ── Index persistence ─────────────────────────────────────────────────────

    def save_index(self, path: Optional[str] = None) -> None:
        """Persist the vector index to disk."""
        self._require_index()
        persist_path = path or self._cfg["vector_store"].get(
            "persist_path", "./data/indices"
        )
        full_path = str(pathlib.Path(persist_path) / "rag_index")
        self._vector_store.save_index(full_path)
        logger.info("Index saved to '%s'", full_path)

    def load_index(self, path: Optional[str] = None) -> None:
        """Load a previously saved index and re-initialise strategy objects."""
        persist_path = path or self._cfg["vector_store"].get(
            "persist_path", "./data/indices"
        )
        full_path = str(pathlib.Path(persist_path) / "rag_index")
        self._vector_store.load_index(full_path)

        # Reconstruct chunks list from the loaded metadata
        self._chunks = self._vector_store._chunks  # type: ignore[assignment]

        threshold = self._cfg["retrieval"].get("semantic_threshold", 0.0)
        self._strategy_a = StrategyA(
            embedding_service=self._embedding_service,
            vector_store=self._vector_store,
            semantic_threshold=threshold,
        )
        self._strategy_b = StrategyB(
            embedding_service=self._embedding_service,
            vector_store=self._vector_store,
            expansion_engine=self._expansion_engine,
            reranker=self._reranker,
            semantic_threshold=threshold,
        )
        self._hybrid = HybridSearch(
            embedding_service=self._embedding_service,
            vector_store=self._vector_store,
            dense_weight=self._cfg["hybrid_search"].get("dense_weight", 0.7),
        )
        logger.info("Index loaded and strategies re-initialised.")

    # ── Config ────────────────────────────────────────────────────────────────

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def embedding_service(self) -> EmbeddingService:
        return self._embedding_service

    @property
    def vector_store(self) -> BaseVectorStore:
        return self._vector_store

    def _require_index(self) -> None:
        if self._strategy_a is None:
            raise RuntimeError(
                "Index not ready. Call setup() or create_index() first."
            )

    @staticmethod
    def _load_config(config_path: Optional[str]) -> dict:
        if config_path is None:
            config_path = str(
                pathlib.Path(__file__).resolve().parents[2] / "config" / "config.yaml"
            )
        cfg_file = pathlib.Path(config_path)
        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        # Sensible defaults when config file is absent
        return {
            "embedding": {"model": "all-MiniLM-L6-v2", "batch_size": 32, "cache_backend": "sqlite"},
            "chunking": {"chunk_size": 512, "chunk_overlap": 64, "min_chunk_size": 50},
            "retrieval": {"top_k": 5, "semantic_threshold": 0.0},
            "vector_store": {"backend": "faiss", "index_type": "flat_ip"},
            "query_expansion": {"model": "gemini-3.1-pro-preview", "expansion_type": "full", "max_variants": 3},
            "hybrid_search": {"enabled": True, "dense_weight": 0.7},
            "reranking": {"enabled": False},
        }
