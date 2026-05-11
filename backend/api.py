"""
FastAPI backend — Context-Aware Retrieval Engine REST API.

Endpoints
---------
GET  /api/health              — Liveness / readiness check
GET  /api/info                — Pipeline statistics
POST /api/ingest              — Load new documents into the index
GET  /api/documents           — List all indexed document sources
POST /api/search              — Run a semantic query
POST /api/benchmark           — Execute full benchmark suite
GET  /api/benchmark/results   — List saved benchmark result files

CORS is enabled for http://localhost:3000 (Next.js dev server) and
http://localhost:8000 (self, for the API docs fetch).
"""

from __future__ import annotations

import json
import pathlib
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.benchmarking.benchmark_engine import BenchmarkEngine, DEFAULT_QUERY_BANK
from src.retrieval.orchestrator import ContextAwareRetriever
from src.utils.logger import configure_root_logger, get_logger

configure_root_logger()
logger = get_logger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────

_retriever: Optional[ContextAwareRetriever] = None
_DATA_DIR = str(pathlib.Path(__file__).parent / "data" / "documents")


# ── Lifespan — load retriever at startup ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global _retriever
    logger.info("Warming up retrieval pipeline …")
    t0 = time.perf_counter()
    _retriever = ContextAwareRetriever()
    _retriever.setup(data_dir=_DATA_DIR)
    elapsed = time.perf_counter() - t0
    logger.info("Pipeline ready in %.2fs — %d chunks indexed", elapsed, _retriever.num_chunks)
    yield
    logger.info("Shutting down …")


def get_retriever() -> ContextAwareRetriever:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")
    return _retriever


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Context-Aware Retrieval Engine",
    description=(
        "RAG system implementing dual retrieval strategies, hybrid BM25+dense search, "
        "cross-encoder reranking, and a comprehensive benchmarking suite."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048, example="How does autoscaling work?")
    top_k: int = Field(default=5, ge=1, le=50)
    strategy: str = Field(
        default="both",
        description="One of: 'a', 'b', 'hybrid', 'both'",
        pattern="^(a|b|hybrid|both)$",
    )
    expansion_mode: str = Field(
        default="full",
        pattern="^(full|synonyms|technical|hyde)$",
    )


class SearchResult(BaseModel):
    chunk_id: str
    rank: int
    score: float
    source: str
    section: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = {}


class StrategyResult(BaseModel):
    strategy: str
    latency_ms: float
    results: List[SearchResult]
    expanded_query: Optional[str] = None
    keywords_added: Optional[List[str]] = None


class SearchResponse(BaseModel):
    query: str
    strategy_a: Optional[StrategyResult] = None
    strategy_b: Optional[StrategyResult] = None
    hybrid: Optional[StrategyResult] = None


class IngestRequest(BaseModel):
    texts: Optional[List[str]] = Field(default=None, description="Inline text documents")
    paths: Optional[List[str]] = Field(default=None, description="File paths to load")


class BenchmarkRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    queries: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Custom query bank (uses default bank if omitted)"
    )
    save_report: bool = True


class HealthResponse(BaseModel):
    status: str
    num_chunks: int
    model: str
    uptime_s: float


# ── Startup timestamp ─────────────────────────────────────────────────────────

_start_time = time.perf_counter()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Liveness + readiness probe."""
    retriever = get_retriever()
    telemetry = retriever.embedding_service.telemetry()
    return HealthResponse(
        status="ok",
        num_chunks=retriever.num_chunks,
        model=telemetry.get("model", "unknown"),
        uptime_s=round(time.perf_counter() - _start_time, 2),
    )


@app.get("/api/info", tags=["System"])
async def pipeline_info() -> dict:
    """Return detailed pipeline statistics."""
    retriever = get_retriever()
    telemetry = retriever.embedding_service.telemetry()
    return {
        "num_chunks": retriever.num_chunks,
        "embedding": telemetry,
        "config": retriever._cfg,
    }


@app.get("/api/documents", tags=["Ingestion"])
async def list_documents() -> dict:
    """List all unique document sources in the index."""
    retriever = get_retriever()
    sources = sorted({c.source for c in retriever._chunks})
    return {"count": len(sources), "sources": sources}


@app.post("/api/ingest", status_code=201, tags=["Ingestion"])
async def ingest(req: IngestRequest) -> dict:
    """Ingest new texts or file paths into the index."""
    retriever = get_retriever()
    before = retriever.num_chunks
    retriever.ingest_documents(
        paths=req.paths or [],
        inline_texts=req.texts or [],
    )
    retriever.build_embeddings()
    retriever.create_index()
    return {"added_chunks": retriever.num_chunks - before, "total_chunks": retriever.num_chunks}


@app.post("/api/search", response_model=SearchResponse, tags=["Retrieval"])
async def search(req: SearchRequest) -> SearchResponse:
    """
    Run a semantic search query.

    * ``strategy="a"`` — direct vector search only
    * ``strategy="b"`` — AI-enhanced with query expansion + RRF
    * ``strategy="hybrid"`` — BM25 + dense fusion
    * ``strategy="both"`` — run A and B, return side-by-side
    """
    retriever = get_retriever()
    strategy = req.strategy

    response = SearchResponse(query=req.query)

    if strategy in ("a", "both"):
        r = retriever.retrieve_raw(req.query, top_k=req.top_k)
        response.strategy_a = StrategyResult(
            strategy="strategy_a",
            latency_ms=round(r.latency_ms, 2),
            results=[_to_search_result(c) for c in r.retrieved_chunks],
        )

    if strategy in ("b", "both"):
        r = retriever.retrieve_enhanced(req.query, top_k=req.top_k, mode=req.expansion_mode)
        exp = r.expanded_query
        response.strategy_b = StrategyResult(
            strategy="strategy_b",
            latency_ms=round(r.latency_ms, 2),
            results=[_to_search_result(c) for c in r.retrieved_chunks],
            expanded_query=exp.expanded_query,
            keywords_added=exp.keywords_added,
        )

    if strategy == "hybrid":
        r = retriever.retrieve_hybrid(req.query, top_k=req.top_k)
        response.hybrid = StrategyResult(
            strategy="hybrid",
            latency_ms=round(r.latency_ms, 2),
            results=[_to_search_result(c) for c in r.retrieved_chunks],
        )

    return response


@app.post("/api/benchmark", tags=["Benchmarking"])
async def run_benchmark(req: BenchmarkRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Run the full benchmark suite.

    This may take 20-60 seconds depending on the corpus size.
    Returns aggregated metrics immediately; saves the full report in the background.
    """
    retriever = get_retriever()
    query_bank = req.queries or DEFAULT_QUERY_BANK

    engine = BenchmarkEngine(retriever, top_k=req.top_k, query_bank=query_bank)
    report = engine.run()

    if req.save_report:
        background_tasks.add_task(
            engine.save_report,
            report,
            output_dir="./outputs/benchmark_results",
        )

    return {
        "timestamp": report.timestamp,
        "num_queries": report.num_queries,
        "aggregate_metrics_a": report.aggregate_metrics_a,
        "aggregate_metrics_b": report.aggregate_metrics_b,
        "aggregate_comparison": report.aggregate_comparison,
        "overall_analysis": report.overall_analysis,
        "query_results": [r.to_dict() for r in report.query_results],
    }


@app.get("/api/benchmark/results", tags=["Benchmarking"])
async def list_benchmark_results() -> dict:
    """List all saved benchmark result JSON files."""
    out_dir = pathlib.Path("./outputs/benchmark_results")
    if not out_dir.exists():
        return {"count": 0, "files": []}
    files = sorted(out_dir.glob("benchmark_*.json"), reverse=True)
    return {
        "count": len(files),
        "files": [f.name for f in files[:20]],
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_search_result(chunk: Any) -> SearchResult:
    return SearchResult(
        chunk_id=chunk.chunk_id,
        rank=chunk.rank,
        score=round(chunk.score, 6),
        source=chunk.source,
        section=getattr(chunk, "section", None),
        text=chunk.text,
        metadata=getattr(chunk, "metadata", {}),
    )


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
