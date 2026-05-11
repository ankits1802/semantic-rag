"""
Mock Vertex AI SDK
==================

Provides drop-in mock classes that mirror the public API of the real
Google Cloud Vertex AI Python SDK so that the rest of the codebase can
import and call them identically to production code.

Mocked classes
--------------
* :class:`TextEmbeddingModel`    — mirrors ``vertexai.language_models.TextEmbeddingModel``
* :class:`TextEmbedding`         — return type of ``get_embeddings()``
* :class:`GenerativeModel`       — mirrors ``vertexai.generative_models.GenerativeModel``
* :class:`GenerationResponse`    — return type of ``generate_content()``

The ``TextEmbeddingModel`` internally delegates to a ``sentence-transformers``
model, making the mock produce real, meaningful embeddings while removing the
GCP project / credentials requirement.

The ``GenerativeModel`` uses a deterministic heuristic query-expansion engine
to produce realistic output for the query-rewriting use-case evaluated in this
assessment.  It never calls an external API.

Production migration
--------------------
Replace these imports with the real SDK::

    # Local (mock)
    from src.embeddings.mock_vertexai import TextEmbeddingModel, GenerativeModel

    # GCP Production
    from vertexai.language_models import TextEmbeddingModel
    from vertexai.generative_models import GenerativeModel

No other code changes are necessary.
"""

from __future__ import annotations

import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── TextEmbedding ─────────────────────────────────────────────────────────────

@dataclass
class TextEmbedding:
    """Mirrors ``vertexai.language_models.TextEmbedding``."""

    values: List[float]
    statistics: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.values)


# ── TextEmbeddingModel ────────────────────────────────────────────────────────

class TextEmbeddingModel:
    """
    Mock of ``vertexai.language_models.TextEmbeddingModel``.

    Internally wraps a ``sentence_transformers.SentenceTransformer`` to
    produce genuine dense vector embeddings.  The public API surface is
    identical to the real Vertex AI SDK so that no callers need to change
    when migrating to production.

    Usage (identical to real SDK)::

        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        embeddings = model.get_embeddings(["hello world"])
        print(embeddings[0].values[:3])   # [-0.021, 0.043, 0.012, ...]
    """

    # Maps Vertex AI model names to equivalent sentence-transformer identifiers.
    _MODEL_MAP: dict[str, str] = {
        "textembedding-gecko@003":  "all-MiniLM-L6-v2",
        "textembedding-gecko@001":  "all-MiniLM-L6-v2",
        "text-embedding-004":       "all-MiniLM-L6-v2",
        "text-multilingual-embedding-002": "paraphrase-multilingual-MiniLM-L12-v2",
        # Allow passing ST model names directly
        "all-MiniLM-L6-v2":        "all-MiniLM-L6-v2",
        "bge-small-en-v1.5":       "BAAI/bge-small-en-v1.5",
        "intfloat/e5-base-v2":     "intfloat/e5-base-v2",
        "e5-base-v2":               "intfloat/e5-base-v2",
    }

    def __init__(self, model_name: str) -> None:
        self._vertex_model_name = model_name
        self._st_model_name = self._MODEL_MAP.get(model_name, "all-MiniLM-L6-v2")
        self._model = None  # lazy load

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(
                "Loading embedding model '%s' (Vertex AI mock: '%s')",
                self._st_model_name,
                self._vertex_model_name,
            )
            self._model = SentenceTransformer(self._st_model_name)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "TextEmbeddingModel":
        """
        Factory method mirroring the real SDK.

        Parameters
        ----------
        model_name:
            Vertex AI model ID (e.g. ``"textembedding-gecko@003"``) or a
            sentence-transformers model ID.
        """
        instance = cls(model_name)
        return instance

    def get_embeddings(
        self,
        texts: List[str],
        auto_truncate: bool = True,
    ) -> List[TextEmbedding]:
        """
        Embed a list of texts.

        Parameters
        ----------
        texts:
            List of strings to embed.
        auto_truncate:
            Ignored (present for API parity with the real SDK).  The
            underlying sentence-transformer handles truncation internally.

        Returns
        -------
        List[TextEmbedding]
        """
        self._load()
        if not texts:
            return []
        import numpy as np
        start = time.perf_counter()
        raw: "np.ndarray" = self._model.encode(  # type: ignore[union-attr]
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
            convert_to_numpy=True,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "Embedded %d text(s) in %.1f ms (dim=%d)",
            len(texts), elapsed_ms, raw.shape[1],
        )
        return [TextEmbedding(values=row.tolist()) for row in raw]

    @property
    def model_name(self) -> str:
        return self._vertex_model_name

    @property
    def embedding_dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]


# ── GenerationResponse ────────────────────────────────────────────────────────

@dataclass
class GenerationResponse:
    """Mirrors ``vertexai.generative_models.GenerationResponse``."""

    _text: str

    @property
    def text(self) -> str:
        return self._text

    def __str__(self) -> str:  # noqa: D105
        return self._text


# ── GenerativeModel ───────────────────────────────────────────────────────────

class GenerativeModel:
    """
    Mock of ``vertexai.generative_models.GenerativeModel``.

    Implements a deterministic query-expansion engine that produces rich,
    domain-aware expansions without calling any external API.  This makes
    benchmarks reproducible and entirely offline.

    Strategy
    --------
    When ``generate_content`` is called with a query-expansion prompt, the
    model:

    1. Extracts the original query from the prompt.
    2. Expands key technical terms using a domain synonym dictionary.
    3. Injects relevant technical context keywords.
    4. Formats the result as a single enriched query string.

    Usage (identical to real SDK)::

        model = GenerativeModel("gemini-3.1-pro-preview")
        response = model.generate_content("Expand: How does autoscaling work?")
        print(response.text)
    """

    # ── Technical synonym dictionary ─────────────────────────────────────────
    _SYNONYMS: dict[str, List[str]] = {
        "peak load":         ["high traffic", "traffic spike", "load surge", "concurrent requests", "heavy load"],
        "high traffic":      ["peak load", "traffic spike", "load surge", "request flood"],
        "autoscaling":       ["horizontal scaling", "auto scaling", "elastic scaling", "dynamic scaling", "scale out"],
        "scaling":           ["autoscaling", "horizontal scaling", "vertical scaling", "scale out", "elasticity"],
        "latency":           ["response time", "processing delay", "round-trip time", "RTT", "end-to-end delay"],
        "failure":           ["fault", "outage", "downtime", "crash", "unavailability", "service interruption"],
        "fault":             ["failure", "error", "exception", "crash", "node failure"],
        "cache":             ["in-memory store", "Redis", "buffer", "caching layer", "memory cache"],
        "caching":           ["in-memory caching", "Redis caching", "cache layer", "cache-aside", "write-through"],
        "load balancing":    ["traffic distribution", "request routing", "server selection", "round-robin", "least connections"],
        "rate limiting":     ["throttling", "quota enforcement", "API limits", "request throttling", "traffic shaping"],
        "distributed":       ["multi-node", "clustered", "sharded", "replicated", "federated"],
        "monitoring":        ["observability", "metrics collection", "tracing", "logging", "alerting"],
        "kubernetes":        ["k8s", "container orchestration", "pod management", "cluster management", "HPA"],
        "microservices":     ["service mesh", "distributed services", "API gateway", "service decomposition"],
        "database":          ["data store", "persistence layer", "storage backend", "DB", "RDBMS"],
        "concurrency":       ["parallelism", "multi-threading", "async processing", "concurrent execution"],
        "fault tolerance":   ["resilience", "high availability", "disaster recovery", "failover", "redundancy"],
        "throughput":        ["requests per second", "RPS", "TPS", "transactions per second", "capacity"],
        "replication":       ["data replication", "replica", "primary-replica", "synchronous replication", "async replication"],
        "consensus":         ["raft", "paxos", "leader election", "quorum", "distributed agreement"],
        "partition":         ["sharding", "data partitioning", "horizontal partitioning", "shard"],
        "circuit breaker":   ["resilience pattern", "fault isolation", "open circuit", "half-open", "hystrix"],
        "retry":             ["retry policy", "exponential backoff", "jitter", "retry mechanism"],
        "service discovery": ["consul", "etcd", "DNS-based discovery", "registry", "health check"],
        "observability":     ["monitoring", "tracing", "metrics", "logs", "distributed tracing", "OpenTelemetry"],
        "container":         ["docker", "pod", "image", "containerised workload", "OCI image"],
        "deployment":        ["rollout", "release", "blue-green deployment", "canary deployment", "rolling update"],
        "api gateway":       ["ingress", "reverse proxy", "rate limiting", "authentication", "routing"],
        "message queue":     ["kafka", "pub/sub", "rabbitmq", "async messaging", "event streaming"],
        "cdn":               ["content delivery network", "edge cache", "PoP", "static asset serving"],
        "security":          ["authentication", "authorisation", "TLS", "mTLS", "encryption", "RBAC"],
        "node failure":      ["server crash", "instance termination", "hardware failure", "pod eviction"],
        "data consistency":  ["strong consistency", "eventual consistency", "linearizability", "CAP theorem"],
        "traffic spike":     ["peak load", "burst traffic", "sudden load increase", "flash crowd"],
        "horizontal scaling":["scale out", "add instances", "autoscaling", "replica count increase"],
    }

    # ── Context injection templates by domain ─────────────────────────────────
    _DOMAIN_CONTEXT: dict[str, List[str]] = {
        "scaling": ["load balancing", "autoscaling", "horizontal scaling", "Kubernetes HPA", "traffic management"],
        "load":    ["CPU utilisation", "memory pressure", "request queue depth", "throughput", "latency"],
        "failure": ["circuit breaker", "retry mechanism", "fallback", "bulkhead", "fault tolerance"],
        "cache":   ["TTL expiration", "cache invalidation", "eviction policy", "Redis", "cache hit rate"],
        "data":    ["replication", "partitioning", "consistency", "durability", "ACID", "BASE"],
        "network": ["TCP", "TLS", "CDN", "load balancer", "firewall", "latency"],
        "deploy":  ["blue-green deployment", "canary", "rolling update", "rollback", "health check"],
        "monitor": ["Prometheus", "Grafana", "alerting", "SLO", "error budget", "tracing"],
        "security":["authentication", "RBAC", "encryption", "audit logging", "zero trust"],
        "queue":   ["Kafka", "Pub/Sub", "consumer group", "dead letter queue", "backpressure"],
    }

    def __init__(self, model_name: str = "gemini-3.1-pro-preview") -> None:
        self._model_name = model_name
        logger.info("MockGenerativeModel initialised: '%s'", model_name)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "GenerativeModel":
        return cls(model_name)

    def generate_content(
        self,
        prompt: str,
        generation_config: Optional[dict] = None,
    ) -> GenerationResponse:
        """
        Generate a response for *prompt*.

        When the prompt contains a query-expansion instruction (detected by
        keywords like "expand", "rewrite", "enhance", or "variants"), the
        mock runs its deterministic expansion engine.  Otherwise it echoes
        the prompt for testing purposes.

        Parameters
        ----------
        prompt:
            The prompt string sent by the application.
        generation_config:
            Ignored (present for API parity).

        Returns
        -------
        GenerationResponse
        """
        # Extract query from common prompt patterns
        query = self._extract_query_from_prompt(prompt)

        if self._is_expansion_request(prompt):
            expanded = self._expand_query(query)
            return GenerationResponse(_text=expanded)

        if self._is_multi_query_request(prompt):
            variants = self._generate_variants(query)
            return GenerationResponse(_text="\n".join(variants))

        if self._is_hyde_request(prompt):
            hypothesis = self._generate_hypothesis(query)
            return GenerationResponse(_text=hypothesis)

        # Fallback: return original query unchanged
        return GenerationResponse(_text=query)

    # ── Private expansion logic ──────────────────────────────────────────────

    def _is_expansion_request(self, prompt: str) -> bool:
        keywords = ["expand", "rewrite", "enhance", "rephrase", "enrich", "elaborate"]
        return any(k in prompt.lower() for k in keywords)

    def _is_multi_query_request(self, prompt: str) -> bool:
        keywords = ["variants", "multiple queries", "generate queries", "alternative queries"]
        return any(k in prompt.lower() for k in keywords)

    def _is_hyde_request(self, prompt: str) -> bool:
        keywords = ["hypothetical document", "hyde", "hypothetical answer", "generate passage"]
        return any(k in prompt.lower() for k in keywords)

    def _extract_query_from_prompt(self, prompt: str) -> str:
        """
        Extract the original user query from a wrapping prompt template.
        Falls back to returning the full prompt if no markers found.
        """
        # Look for patterns like: "Query: <q>" or "Question: <q>" or "Original: <q>"
        for pattern in [
            r"(?:Query|Question|Original query|Input|User query)[:\s]+(.+?)(?:\n|$)",
            r'"(.+?)"',
            r"'(.+?)'",
        ]:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1).strip().rstrip("?.")

        # If the prompt is short enough, treat the whole thing as the query
        lines = [l.strip() for l in prompt.strip().splitlines() if l.strip()]
        return lines[-1] if lines else prompt

    def _expand_query(self, query: str) -> str:
        """
        Produce an enriched query string by:
        1. Expanding matched synonym phrases
        2. Injecting domain-specific context keywords
        """
        query_lower = query.lower()
        expansion_terms: List[str] = []

        # Synonym expansion
        for phrase, synonyms in self._SYNONYMS.items():
            if phrase in query_lower:
                expansion_terms.extend(synonyms[:3])

        # Domain context injection
        for domain_key, context_words in self._DOMAIN_CONTEXT.items():
            if domain_key in query_lower:
                expansion_terms.extend(context_words[:2])

        # Deduplicate while preserving order
        seen: set = set()
        unique_terms: List[str] = []
        for t in expansion_terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_terms.append(t)

        if not unique_terms:
            # Generic technical expansion
            unique_terms = ["scalability", "performance", "reliability", "system design"]

        expansion_clause = ", ".join(unique_terms[:8])
        return f"{query} including {expansion_clause}"

    def _generate_variants(self, query: str) -> List[str]:
        """Generate 3 semantically related query variants."""
        base = self._expand_query(query)
        variants = [query]

        # Variant 1: Focus on mechanism
        variants.append(f"What mechanisms are used for {query.lower().replace('how does', '').replace('how is', '').strip()}?")

        # Variant 2: Focus on challenges/problems
        variants.append(f"What are the challenges and solutions for {query.lower().replace('how', '').strip()}?")

        # Variant 3: The expanded base
        variants.append(base)

        return variants[:4]  # return at most 4

    def _generate_hypothesis(self, query: str) -> str:
        """
        Generate a HyDE (Hypothetical Document Embedding) passage — a fake
        but plausible answer that, when embedded, lands near real answers.
        """
        query_lower = query.lower()
        expanded = self._expand_query(query)

        # Build a plausible short technical paragraph
        hypothesis = textwrap.dedent(f"""\
            In modern distributed systems, {expanded}.
            The system handles this through a combination of horizontal scaling,
            load balancing, and fault-tolerant design patterns.
            Autoscaling policies adjust the number of running instances based on
            observed metrics such as CPU utilisation and request queue depth.
            This ensures consistent performance during traffic spikes and peak load
            conditions while maintaining low latency and high availability.
        """).strip()
        return hypothesis
