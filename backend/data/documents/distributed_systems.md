# Distributed Systems Engineering — Comprehensive Reference

## 1. Consensus Algorithms and Leader Election

Distributed consensus is the problem of getting multiple nodes to agree on a
single value or sequence of values, even in the presence of node failures and
network partitions. Consensus is foundational to distributed databases,
leader election, and distributed lock services.

### Raft Consensus Protocol

Raft decomposes consensus into three relatively independent subproblems:
leader election, log replication, and safety. Raft guarantees that at any
given time, at most one leader exists per term.

**Leader Election**: Nodes begin as followers. If a follower receives no
heartbeat within an election timeout (randomised 150–300ms), it transitions
to candidate and requests votes from peers. A candidate becomes leader upon
receiving a majority quorum of votes.

**Log Replication**: The leader appends client commands to its log and
replicates entries to followers via AppendEntries RPCs. An entry is
"committed" once acknowledged by a majority; the leader then applies the
entry to its state machine and responds to the client.

**Safety**: Raft's election restriction ensures only candidates with logs
at least as up-to-date as the majority can win, preventing data loss on
leader failover.

### Paxos and Multi-Paxos

Paxos achieves consensus in two phases: Prepare/Promise and Accept/Accepted.
Multi-Paxos elects a stable leader to skip Phase 1 for subsequent proposals,
reducing round-trip latency. Paxos is used by Google Chubby (distributed
lock service) and underpins Google Spanner's TrueTime-based transactions.

### PBFT (Practical Byzantine Fault Tolerance)

PBFT tolerates Byzantine (arbitrarily malicious) failures, not just crash
failures. It requires 3f+1 nodes to tolerate f Byzantine failures. Used in
permissioned blockchain systems and safety-critical distributed applications.

---

## 2. Data Replication and Consistency Models

Replication maintains copies of data across multiple nodes to achieve
fault tolerance, high availability, and read scalability.

### Replication Strategies

**Synchronous Replication**: The primary waits for all replicas to acknowledge
before confirming the write to the client. Guarantees strong consistency but
increases write latency proportional to the slowest replica (tail latency).

**Asynchronous Replication**: The primary acknowledges immediately after local
write; replicas catch up asynchronously. Low write latency but risk of data
loss if the primary fails before replicas synchronise.

**Semi-Synchronous Replication**: Requires acknowledgement from at least one
replica before confirming to the client. Balances durability and latency.

### Consistency Models

**Strong Consistency (Linearizability)**: All reads reflect the most recent
write. Every operation appears to execute atomically at a single point in time.
Achieved via synchronous replication or consensus protocols. High latency.

**Sequential Consistency**: All nodes see operations in the same order, but
that order may not match real-time ordering. Weaker than linearizability.

**Causal Consistency**: Causally related operations are seen in the same order
by all nodes; concurrent operations may be seen in different orders.
Implemented using vector clocks or hybrid logical clocks.

**Eventual Consistency**: All replicas will converge to the same value given
sufficient time and no new updates. Used by systems like Amazon DynamoDB,
Apache Cassandra, and CouchDB for high availability.

**Read-Your-Writes / Monotonic Reads**: Session-level consistency guarantees
that a client always reads the latest value it wrote, and that subsequent reads
never return older versions.

---

## 3. Data Partitioning and Sharding

Partitioning (sharding) distributes data across multiple nodes to overcome
the storage and throughput limitations of a single machine.

### Partitioning Strategies

**Hash Partitioning**: A hash function is applied to the partition key (e.g.,
user_id), and the result modulo the number of partitions determines placement.
Provides uniform distribution but makes range queries inefficient.

**Range Partitioning**: Data is partitioned by contiguous key ranges (e.g.,
users A–M on shard 1, N–Z on shard 2). Supports efficient range scans but
may cause hotspots if traffic is concentrated in one range.

**Consistent Hashing**: Maps both data keys and nodes onto a ring. Each key
is assigned to the next clockwise node. Adding or removing nodes only
redistributes 1/N of keys on average, minimising rebalancing overhead.
Used by Apache Cassandra, Amazon Dynamo, and distributed caches.

**Directory-Based Partitioning**: A centralised lookup table maps keys to
partition nodes. Flexible but introduces a bottleneck and single point of
failure in the directory service.

### Rebalancing

When nodes are added or removed, data must be moved to maintain balance.
Strategies include fixed number of partitions (pre-created), dynamic
partitioning (split/merge when size thresholds are crossed), and
virtual nodes (vnodes) in consistent hashing.

---

## 4. Fault Tolerance and High Availability

### Failure Modes

**Crash failures**: A node stops responding abruptly without warning.
**Omission failures**: A node fails to send or receive specific messages.
**Byzantine failures**: A node behaves arbitrarily, possibly maliciously.
**Network partition**: A subset of nodes cannot communicate with another subset.

### CAP Theorem

Eric Brewer's CAP theorem states that a distributed system cannot simultaneously
guarantee all three of: Consistency, Availability, and Partition Tolerance.

- **CP systems** (consistent and partition-tolerant): Sacrifice availability
  during partitions. Examples: HBase, ZooKeeper, etcd, Consul.
- **AP systems** (available and partition-tolerant): Sacrifice consistency
  during partitions, offering eventual consistency. Examples: Cassandra,
  DynamoDB, CouchDB.
- **CA systems**: Not practically achievable in distributed systems that must
  tolerate network partitions.

### PACELC Model

PACELC extends CAP: even in the absence of partitions (P), there is a
tradeoff between latency (L) and consistency (C). Most real systems must
choose between low latency (accept stale reads) and strong consistency
(pay replication round-trip latency).

### Failure Detection

The **Phi Accrual Failure Detector** (used by Cassandra) outputs a suspicion
value φ that increases continuously when heartbeats are not received.
Applications set a threshold φ_threshold to classify a node as down.

**SWIM (Scalable Weakly-consistent Infection-style Membership)** protocol uses
gossip-based failure detection with indirect probe: if a direct probe to node X
fails, the detector enlists K random nodes to probe X indirectly, reducing
false positives caused by transient network loss.

### Bulkhead Pattern

Isolates resource pools (thread pools, connection pools, memory) for different
subsystems. A failure in one subsystem consumes only its allocated resources,
preventing resource exhaustion from cascading to other parts of the system.

---

## 5. Distributed Transactions

### Two-Phase Commit (2PC)

**Phase 1 (Prepare)**: Coordinator sends prepare to all participants. Each
participant executes the transaction locally (but does not commit), writes to
a redo log, and responds VOTE_COMMIT or VOTE_ABORT.

**Phase 2 (Commit/Rollback)**: If all participants voted COMMIT, the coordinator
sends COMMIT to all; otherwise, sends ROLLBACK. Participants apply the decision.

2PC is blocking: if the coordinator crashes after Phase 1 but before Phase 2,
participants remain blocked in uncertain state until the coordinator recovers.

### Saga Pattern

A saga is a sequence of local transactions where each step publishes an event
triggering the next. If a step fails, compensating transactions undo the
previous steps. Sagas avoid distributed locks but require idempotent
compensating transactions and careful failure scenario analysis.

**Choreography-based Saga**: Services react to events; no central coordinator.
Decoupled but harder to monitor and debug.

**Orchestration-based Saga**: A central saga orchestrator sends commands to
services and handles failures. More visible but creates coupling to orchestrator.

---

## 6. Distributed Caching and Content Delivery

### Cache Coherence in Distributed Systems

When multiple nodes cache the same data, write updates must be propagated to
prevent stale reads. Cache invalidation can be push-based (pub/sub
notifications) or pull-based (TTL expiry with background refresh).

Thundering herd / cache stampede occurs when many clients simultaneously
request the same uncached item, flooding the database. Mitigations:
- Probabilistic early expiration (PER): refresh before TTL expires
- Request coalescing: deduplicate concurrent cache miss requests
- Distributed lock on cache miss: only one request populates the cache

### CDN Edge Caching

Content Delivery Networks cache static and dynamic content at geographically
distributed Points of Presence (PoPs), reducing origin load and minimising
latency for end users. Cache-Control headers (max-age, stale-while-revalidate,
surrogate-control) govern CDN caching behaviour.

---

## 7. Observability and Distributed Tracing

The three pillars of observability are metrics, logs, and traces.

**Distributed Tracing** (OpenTelemetry, Jaeger, Zipkin): A trace represents
the end-to-end journey of a single request across microservices. Each service
emits spans with start/end timestamps, operation names, and tags. Spans are
linked by trace IDs propagated via HTTP headers (B3, W3C TraceContext).

**Metrics** (Prometheus, Datadog, Cloud Monitoring): Time-series numeric data
collected at regular intervals. The RED method tracks Request Rate, Error Rate,
and Duration per service. The USE method tracks Utilisation, Saturation,
and Errors per resource.

**Structured Logging** (JSON over text): Machine-parseable log entries with
fields for severity, timestamp, trace_id, request_id, service, and message.
Aggregated in Elasticsearch, Cloud Logging, or Splunk for full-text search
and analytical queries.

**Alerting**: SLI (Service Level Indicator) metrics are evaluated against SLO
(Service Level Objective) thresholds. Alert fatigue is reduced by correlating
related alerts, using multi-window / multi-burn-rate error budget alerting
(Google SRE approach), and routing to the appropriate on-call rotation.
