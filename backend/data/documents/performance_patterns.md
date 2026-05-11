# Performance Engineering and Concurrency Patterns

## 1. Concurrency Models and Thread Management

Concurrency enables multiple tasks to make progress simultaneously, increasing
CPU utilisation and throughput. Understanding concurrency models is critical
for designing high-performance systems.

### Thread-Based Concurrency

Traditional multi-threaded applications create one OS thread per request or task.
Threads are heavyweight: each thread consumes 1–8 MB of stack memory and carries
significant context-switching overhead. Thread pools (Java ExecutorService,
Python ThreadPoolExecutor) bound the maximum number of threads to prevent
resource exhaustion and the "too many threads" bottleneck.

Work-stealing schedulers (Go runtime, Java ForkJoinPool) allow idle threads to
steal tasks from busy thread queues, improving load balancing across cores
without global locking.

### Asynchronous I/O and Event Loops

For I/O-bound workloads, asynchronous programming significantly outperforms
threads by avoiding the blocking wait during I/O operations. An event loop runs
on a single thread and processes I/O completion events:

- **Python asyncio**: Uses `async/await` coroutines. The event loop runs until
  all scheduled coroutines complete. `asyncio.gather()` runs coroutines
  concurrently. `asyncio.Semaphore` bounds concurrent operations.
- **Node.js libuv**: Single-threaded event loop with non-blocking I/O via libuv.
  CPU-bound tasks are offloaded to a worker thread pool.
- **Java NIO/Netty**: Non-blocking I/O channels multiplexed over a small number
  of event loop threads (typically one per CPU core).

### Actor Model

The actor model (Akka, Erlang OTP, Microsoft Orleans) represents concurrent
computation as independent actors that communicate solely by passing immutable
messages. Actors have private state and process messages sequentially from their
mailbox, eliminating race conditions without explicit synchronisation.

### Green Threads and Coroutines

Go goroutines are multiplexed over OS threads by the Go runtime scheduler (GOMAXPROCS
threads). Goroutines are lightweight (~2KB initial stack) and can number in the
hundreds of thousands. Channels provide synchronised communication between
goroutines, replacing shared-memory concurrency.

---

## 2. Lock-Free and Wait-Free Algorithms

Lock contention is a major bottleneck in concurrent systems. Lock-free data
structures use atomic hardware instructions (Compare-And-Swap, Fetch-And-Add)
to achieve progress without mutual exclusion.

**CAS (Compare-And-Swap)**: Atomically updates a memory location if its current
value matches an expected value. Used in lock-free queues (Michael-Scott queue),
lock-free stacks, and atomic reference updates.

**ABA Problem**: A value may change from A to B and back to A between a CAS
read and write, causing the CAS to succeed incorrectly. Mitigated by tagging
pointers with a version counter (stamped references) or using hazard pointers
for safe memory reclamation.

---

## 3. Performance Optimisation Patterns

### Connection Pooling

Establishing TCP connections and TLS handshakes incur significant latency (1–3 RTTs).
Connection pools maintain persistent connections to databases, HTTP services, and
message brokers. Key parameters:
- minIdle: Minimum idle connections kept alive
- maxTotal: Maximum connections in the pool
- connectionTimeout: Time to wait for an available connection
- validationQuery: SQL query to verify connection health before use

HikariCP (Java) is the highest-performing JDBC connection pool, measuring
average acquisition time in microseconds through careful lock elision and
atomic operations.

### Request Batching and Microbatching

Batching multiple small requests into a single large request amortises per-request
overhead (network RTT, serialisation, database query planning). Microbatching
collects requests over a short time window (1–10ms) before dispatching, trading
a small latency increase for significant throughput improvement. Used in Kafka
producer batching (linger.ms, batch.size), Redis pipeline, and database bulk inserts.

### Zero-Copy I/O

Traditional I/O involves copying data between kernel and user space buffers.
Zero-copy techniques (sendfile syscall, mmap, Direct I/O) transfer data directly
between kernel buffers without intermediate user-space copies, reducing CPU usage
and increasing throughput for file serving and network transfer workloads.

### NUMA-Aware Memory Allocation

In multi-socket servers, memory access to a remote NUMA node has 2–3x higher
latency than local memory. NUMA-aware applications pin threads and memory
allocations to the same NUMA node (numactl, jemalloc with NUMA support),
reducing cross-NUMA traffic and improving cache locality.

---

## 4. Rate Limiting in Application Code

### Token Bucket Implementation

```python
class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity,
                          self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
```

For distributed systems, Redis Lua scripts implement atomic token bucket checks
across multiple application instances without race conditions.

---

## 5. Monitoring, SLOs, and Error Budgets

### The Four Golden Signals (Google SRE)

1. **Latency**: Time to service a request. Track p50, p95, p99, p999 percentiles.
   High p99 latency indicates tail latency affecting a significant fraction of users.
2. **Traffic**: Request rate (RPS), bytes per second, active connections.
3. **Errors**: Error rate (5xx responses, exception rates, timeout counts).
4. **Saturation**: Measure of how "full" a resource is — CPU queue depth,
   memory pressure, disk I/O utilisation, database connection pool exhaustion.

### Service Level Indicators and Objectives

SLI: A quantitative measure of a service behaviour (e.g., 95th percentile
request latency < 200ms over a 5-minute window).

SLO: A target range or threshold for an SLI (e.g., 99.9% of requests served
within 200ms over a 30-day rolling window).

Error Budget: The allowed amount of SLO violation (100% - SLO). A 99.9% SLO
provides a 0.1% error budget, equivalent to ~43.8 minutes per month.

### Apdex Score

Apdex (Application Performance Index) quantifies user satisfaction as a single
score from 0 to 1. For a threshold T:
- Satisfied: response time ≤ T
- Tolerating: T < response time ≤ 4T
- Frustrated: response time > 4T
- Apdex = (Satisfied + 0.5 × Tolerating) / Total

### Percentile Latency vs. Average

Averages mask tail latency outliers. P99 latency (99th percentile) represents the
worst experience for 1 in 100 requests. P999 represents 1 in 1000 requests — critical
for identifying resource contention, GC pauses, and lock starvation.

Histograms (Prometheus histogram_quantile, HdrHistogram) provide space-efficient
approximation of latency distributions without storing every individual measurement.

---

## 6. Caching for Performance

### CPU Cache Hierarchy

Modern CPUs have L1 (~32KB, ~1ns), L2 (~256KB, ~4ns), and L3 (~8MB, ~12ns)
caches. Cache-friendly data structures (arrays over linked lists) and sequential
memory access patterns maximise cache utilisation and avoid cache line
invalidation in multi-core scenarios.

### Memory-Mapped Files

mmap() maps file contents into the process's virtual address space, allowing
file I/O via pointer dereferences. The OS page cache handles actual I/O.
Applications like RocksDB, LMDB, and Lucene use mmap for high-performance
read-heavy workloads.

### Application-Level Caching Tiers

- **Memoization**: Cache pure function results by input hash.
- **Query Result Cache**: Cache expensive database query results.
- **Fragment Cache**: Cache rendered HTML fragments or API response segments.
- **Object Cache**: Cache hydrated domain objects to avoid repeated
  deserialisation and computation.

---

## 7. Chaos Engineering and Resilience Testing

Chaos engineering is the practice of deliberately injecting failures into a
system to verify its resilience and uncover hidden weaknesses before they
manifest in production.

Netflix Chaos Monkey randomly terminates virtual machine instances in production.
Chaos Toolkit, Litmus Chaos, and Gremlin provide structured chaos experiments
including network latency injection, packet loss, resource exhaustion, and
process killing.

Gamedays are planned exercises where teams simulate disaster scenarios (datacenter
failure, dependency outage, data corruption) to validate runbooks, test alerting,
and practice incident response. Results feed back into system improvements and
SRE capacity planning.
