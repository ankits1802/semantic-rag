"""
Query Expansion Engine — AI-enhanced query rewriting for Strategy B.

This module uses the :class:`~src.embeddings.mock_vertexai.GenerativeModel`
(mock Gemini) to transform a raw user query into an embedding-rich expanded
query.  Four expansion modes are supported:

Expansion Modes
---------------
* ``full``       — synonym expansion + context injection + enriched phrasing
* ``synonyms``   — synonym expansion only
* ``technical``  — domain keyword injection only
* ``hyde``       — HyDE: generate a hypothetical document passage to use as
                    the query embedding (Gao et al., 2022)

Multi-query Generation
----------------------
:py:meth:`QueryExpansionEngine.generate_variants` produces multiple semantically
diverse query strings.  The retrieval pipeline can embed all variants and
aggregate their results via Reciprocal Rank Fusion (RRF).

Design rationale
----------------
Raw user queries are often short and ambiguous (e.g. "peak load handling").
After expansion they become embedding-rich strings that land closer to the
document vectors in the embedding space, dramatically improving recall for
technical topics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.embeddings.mock_vertexai import GenerativeModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExpandedQuery:
    """
    Result of a query expansion operation.

    Attributes
    ----------
    original_query:
        The unmodified input query.
    expanded_query:
        The single expanded query string used for embedding.
    variants:
        Optional list of alternative query phrasings for multi-query retrieval.
    expansion_mode:
        Which expansion strategy was applied.
    keywords_added:
        List of new keywords/phrases injected during expansion.
    hyde_passage:
        Hypothetical document passage (only populated in ``hyde`` mode).
    """

    original_query: str
    expanded_query: str
    variants: List[str] = field(default_factory=list)
    expansion_mode: str = "full"
    keywords_added: List[str] = field(default_factory=list)
    hyde_passage: str = ""

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "expanded_query": self.expanded_query,
            "variants": self.variants,
            "expansion_mode": self.expansion_mode,
            "keywords_added": self.keywords_added,
            "hyde_passage": self.hyde_passage,
        }


class QueryExpansionEngine:
    """
    Orchestrates all query enhancement functionality.

    Parameters
    ----------
    model_name:
        Vertex AI / mock model identifier.
    expansion_type:
        Default expansion mode (``full`` | ``synonyms`` | ``technical`` | ``hyde``).
    max_variants:
        Maximum number of query variants to generate.
    """

    # ── Technical synonym dictionary ─────────────────────────────────────────
    _SYNONYMS: Dict[str, List[str]] = {
        "peak load":          ["high traffic", "traffic spike", "load surge", "heavy concurrency"],
        "high traffic":       ["peak load", "traffic spike", "load surge", "flash crowd"],
        "autoscaling":        ["auto scaling", "horizontal scaling", "elastic scaling", "dynamic provisioning"],
        "scaling":            ["autoscaling", "horizontal scaling", "vertical scaling", "elasticity"],
        "latency":            ["response time", "processing delay", "round-trip time", "end-to-end delay"],
        "failure":            ["fault", "outage", "downtime", "crash", "service disruption"],
        "fault":              ["failure", "error", "crash", "node failure"],
        "cache":              ["in-memory store", "Redis", "buffer", "memory cache"],
        "caching":            ["in-memory caching", "Redis caching", "cache-aside", "write-through cache"],
        "load balancing":     ["traffic distribution", "request routing", "round-robin", "least connections"],
        "rate limiting":      ["throttling", "quota enforcement", "API rate control", "request throttling"],
        "distributed":        ["multi-node", "clustered", "sharded", "replicated", "federated"],
        "monitoring":         ["observability", "metrics collection", "distributed tracing", "alerting"],
        "kubernetes":         ["k8s", "container orchestration", "pod management", "HPA", "cluster management"],
        "microservices":      ["service mesh", "distributed services", "API gateway", "service decomposition"],
        "database":           ["data store", "persistence layer", "storage backend", "RDBMS"],
        "concurrency":        ["parallelism", "multi-threading", "async processing", "parallel execution"],
        "fault tolerance":    ["resilience", "high availability", "disaster recovery", "failover"],
        "throughput":         ["requests per second", "RPS", "TPS", "capacity", "bandwidth"],
        "replication":        ["data replication", "replica sync", "primary-replica", "synchronous replication"],
        "consensus":          ["raft protocol", "paxos", "leader election", "quorum", "distributed agreement"],
        "partition":          ["sharding", "data partitioning", "hash partitioning", "range partitioning"],
        "circuit breaker":    ["resilience pattern", "fault isolation", "hystrix", "open circuit state"],
        "retry":              ["retry policy", "exponential backoff", "jitter", "retry mechanism"],
        "service discovery":  ["consul", "etcd", "DNS discovery", "health check registry"],
        "observability":      ["monitoring", "distributed tracing", "metrics dashboard", "OpenTelemetry"],
        "container":          ["docker container", "pod", "container image", "OCI image"],
        "deployment":         ["rollout", "release", "blue-green deployment", "canary release", "rolling update"],
        "api gateway":        ["ingress controller", "reverse proxy", "rate limiting gateway", "routing layer"],
        "message queue":      ["kafka", "pub/sub", "rabbitmq", "async messaging", "event streaming"],
        "traffic spike":      ["peak load", "burst traffic", "sudden load increase", "flash crowd"],
        "node failure":       ["server crash", "instance termination", "hardware failure", "pod eviction"],
        "consistency":        ["strong consistency", "eventual consistency", "linearizability", "CAP theorem"],
        "horizontal scaling": ["scale out", "add instances", "autoscaling", "replica count increase"],
        "availability":       ["uptime", "SLA", "high availability", "99.99%", "fault tolerance"],
        "security":           ["authentication", "authorisation", "TLS encryption", "RBAC", "zero trust"],
        "cdn":                ["content delivery network", "edge cache", "PoP", "static asset caching"],
    }

    # ── Domain-specific context injection terms ───────────────────────────────
    _DOMAIN_CONTEXT: Dict[str, List[str]] = {
        "scal":    ["autoscaling", "load balancing", "horizontal scaling", "Kubernetes HPA", "traffic management"],
        "load":    ["CPU utilisation", "memory pressure", "request queue", "throughput", "traffic distribution"],
        "fail":    ["circuit breaker", "retry policy", "fallback strategy", "bulkhead pattern", "fault tolerance"],
        "cache":   ["TTL expiration", "cache invalidation", "eviction policy", "Redis", "cache hit rate"],
        "data":    ["replication", "sharding", "consistency model", "durability", "ACID transactions"],
        "network": ["TCP connection", "TLS handshake", "CDN", "load balancer", "network latency"],
        "deploy":  ["blue-green deployment", "canary release", "rolling update", "rollback", "health check"],
        "monitor": ["Prometheus metrics", "Grafana dashboard", "alerting rules", "SLO", "error budget"],
        "securit": ["authentication", "RBAC", "TLS encryption", "audit logging", "zero trust network"],
        "queue":   ["Kafka", "Pub/Sub", "consumer group", "dead letter queue", "backpressure"],
        "distrib": ["consensus protocol", "leader election", "quorum", "replication lag", "CAP theorem"],
        "concur":  ["thread pool", "async I/O", "event loop", "lock contention", "race condition"],
        "latenc":  ["p99 latency", "response time", "network RTT", "connection pooling", "timeout policy"],
        "throughp":["RPS", "TPS", "burst capacity", "rate limiting", "connection pooling"],
        "kubernetes": ["pod scheduling", "Horizontal Pod Autoscaler", "node affinity", "resource limits"],
    }

    def __init__(
        self,
        model_name: str = "gemini-3.1-pro-preview",
        expansion_type: str = "full",
        max_variants: int = 3,
    ) -> None:
        self.expansion_type = expansion_type
        self.max_variants = max_variants
        self._model = GenerativeModel(model_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def expand(
        self,
        query: str,
        mode: Optional[str] = None,
    ) -> ExpandedQuery:
        """
        Expand *query* using the specified (or default) *mode*.

        Parameters
        ----------
        query:
            Raw user query.
        mode:
            Override the default expansion mode for this call.

        Returns
        -------
        ExpandedQuery
        """
        if not query or not query.strip():
            raise ValueError("Cannot expand an empty query.")

        active_mode = mode or self.expansion_type
        query = query.strip()

        logger.debug("Expanding query '%s' (mode=%s)", query, active_mode)

        if active_mode == "hyde":
            return self._hyde_expansion(query)
        elif active_mode == "synonyms":
            return self._synonym_expansion(query)
        elif active_mode == "technical":
            return self._technical_expansion(query)
        else:  # "full" — default
            return self._full_expansion(query)

    def generate_variants(self, query: str) -> List[str]:
        """
        Generate up to ``self.max_variants`` semantically diverse reformulations
        of *query* for multi-query retrieval.
        """
        prompt = (
            f"Generate {self.max_variants} alternative queries for: '{query}'. "
            "Focus on technical variants for distributed systems."
        )
        response = self._model.generate_content(prompt)
        lines = [l.strip() for l in response.text.splitlines() if l.strip()]

        variants = [query] + lines[: self.max_variants - 1]
        logger.debug("Generated %d query variants for '%s'", len(variants), query)
        return variants

    def rewrite(self, query: str) -> str:
        """
        Simple single-call rewrite: return the expanded query string only.
        """
        return self.expand(query).expanded_query

    # ── Private expansion strategies ─────────────────────────────────────────

    def _full_expansion(self, query: str) -> ExpandedQuery:
        """Synonym expansion + domain context injection via mock GenerativeModel."""
        prompt = (
            f"Expand the following technical query with synonyms, related concepts, "
            f"and domain-specific terms for better semantic retrieval. "
            f"Query: {query}"
        )
        expanded_text = self._model.generate_content(prompt).text
        keywords_added = self._extract_keywords_added(query, expanded_text)
        variants = self.generate_variants(query)

        return ExpandedQuery(
            original_query=query,
            expanded_query=expanded_text,
            variants=variants,
            expansion_mode="full",
            keywords_added=keywords_added,
        )

    def _synonym_expansion(self, query: str) -> ExpandedQuery:
        """Pure synonym replacement without context injection."""
        query_lower = query.lower()
        expansion_terms: List[str] = []

        for phrase, synonyms in self._SYNONYMS.items():
            if phrase in query_lower:
                expansion_terms.extend(synonyms[:2])

        if not expansion_terms:
            # Fallback: use full expansion
            return self._full_expansion(query)

        deduped = list(dict.fromkeys(expansion_terms))
        expanded = f"{query}, {', '.join(deduped[:6])}"
        return ExpandedQuery(
            original_query=query,
            expanded_query=expanded,
            expansion_mode="synonyms",
            keywords_added=deduped[:6],
        )

    def _technical_expansion(self, query: str) -> ExpandedQuery:
        """Inject domain-specific technical context keywords."""
        query_lower = query.lower()
        context_terms: List[str] = []

        for key_prefix, terms in self._DOMAIN_CONTEXT.items():
            if key_prefix.lower() in query_lower:
                context_terms.extend(terms[:3])

        if not context_terms:
            return self._full_expansion(query)

        deduped = list(dict.fromkeys(context_terms))
        expanded = f"{query} — related concepts: {', '.join(deduped[:6])}"
        return ExpandedQuery(
            original_query=query,
            expanded_query=expanded,
            expansion_mode="technical",
            keywords_added=deduped[:6],
        )

    def _hyde_expansion(self, query: str) -> ExpandedQuery:
        """
        Hypothetical Document Embedding (HyDE):
        Generate a plausible answer passage and use it as the search text.
        """
        prompt = (
            f"Generate a hypothetical technical document passage that would directly "
            f"answer the following question. Be specific and use domain terminology. "
            f"Question: {query}"
        )
        passage = self._model.generate_content(prompt).text

        return ExpandedQuery(
            original_query=query,
            expanded_query=passage,   # embed the passage, not the query
            expansion_mode="hyde",
            keywords_added=self._extract_keywords_added(query, passage),
            hyde_passage=passage,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_keywords_added(self, original: str, expanded: str) -> List[str]:
        """Identify new words present in *expanded* but not in *original*."""
        original_words = set(re.findall(r"\b\w{4,}\b", original.lower()))
        expanded_words = re.findall(r"\b\w{4,}\b", expanded.lower())
        new_words = [w for w in expanded_words if w not in original_words]
        # Deduplicate while preserving order
        seen: set = set()
        unique = []
        for w in new_words:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique[:10]
