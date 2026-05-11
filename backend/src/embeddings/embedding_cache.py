"""
Embedding Cache — persists computed embeddings to avoid recomputation.

Three backends are supported and selected via the ``config.yaml``
``embedding.cache_backend`` field:

* ``sqlite`` (default) — stores vectors as JSON blobs in a local SQLite DB
* ``pickle``            — shelve-based key/value binary store
* ``json``              — plain JSON file (simple but slow for large datasets)

Cache keys are SHA-256 hashes of ``(model_name, text)`` making them stable
across process restarts and insensitive to surrounding whitespace.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import pickle
import shelve
import sqlite3
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_cache_key(model_name: str, text: str) -> str:
    """Return a deterministic 64-char hex SHA-256 key for (model, text)."""
    payload = f"{model_name}::{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseEmbeddingCache(ABC):
    """Abstract interface for embedding caches."""

    @abstractmethod
    def get(self, key: str) -> Optional[List[float]]:
        """Return cached vector or ``None`` on miss."""

    @abstractmethod
    def set(self, key: str, vector: List[float]) -> None:
        """Store *vector* under *key*."""

    @abstractmethod
    def get_batch(self, keys: List[str]) -> Dict[str, Optional[List[float]]]:
        """Return a dict mapping each key to its cached vector (or None)."""

    @abstractmethod
    def set_batch(self, items: Dict[str, List[float]]) -> None:
        """Store multiple key → vector pairs."""

    @abstractmethod
    def size(self) -> int:
        """Return number of cached entries."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all cached entries."""


# ── SQLite backend ────────────────────────────────────────────────────────────

class SQLiteEmbeddingCache(BaseEmbeddingCache):
    """
    SQLite-backed embedding cache.

    Stores each embedding as a JSON-serialised float array in a table with
    a TEXT primary key.  Thread-safe for single-process use (SQLite serialised
    mode).
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS embeddings (
            cache_key   TEXT PRIMARY KEY,
            model_name  TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            created_at  REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model_name);
    """

    def __init__(self, db_path: str = "data/cache/embeddings.db") -> None:
        self._db_path = pathlib.Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(self._CREATE_TABLE)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path), check_same_thread=False
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")
        return self._conn

    def get(self, key: str) -> Optional[List[float]]:
        row = self._get_conn().execute(
            "SELECT vector_json FROM embeddings WHERE cache_key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, vector: List[float], model_name: str = "") -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (cache_key, model_name, vector_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (key, model_name, json.dumps(vector), time.time()),
        )
        conn.commit()

    def get_batch(self, keys: List[str]) -> Dict[str, Optional[List[float]]]:
        if not keys:
            return {}
        placeholders = ",".join("?" * len(keys))
        rows = self._get_conn().execute(
            f"SELECT cache_key, vector_json FROM embeddings WHERE cache_key IN ({placeholders})",
            keys,
        ).fetchall()
        found = {r[0]: json.loads(r[1]) for r in rows}
        return {k: found.get(k) for k in keys}

    def set_batch(self, items: Dict[str, List[float]], model_name: str = "") -> None:
        if not items:
            return
        conn = self._get_conn()
        ts = time.time()
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (cache_key, model_name, vector_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            [(k, model_name, json.dumps(v), ts) for k, v in items.items()],
        )
        conn.commit()

    def size(self) -> int:
        return self._get_conn().execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0]

    def clear(self) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM embeddings")
        conn.commit()


# ── Pickle / shelve backend ───────────────────────────────────────────────────

class PickleEmbeddingCache(BaseEmbeddingCache):
    """shelve-backed embedding cache using Python's native serialisation."""

    def __init__(self, db_path: str = "data/cache/embeddings_shelve") -> None:
        self._db_path = pathlib.Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _open(self):  # noqa: ANN201
        return shelve.open(str(self._db_path), flag="c", writeback=False)

    def get(self, key: str) -> Optional[List[float]]:
        with self._open() as db:
            return db.get(key)

    def set(self, key: str, vector: List[float], **_) -> None:
        with self._open() as db:
            db[key] = vector

    def get_batch(self, keys: List[str]) -> Dict[str, Optional[List[float]]]:
        with self._open() as db:
            return {k: db.get(k) for k in keys}

    def set_batch(self, items: Dict[str, List[float]], **_) -> None:
        with self._open() as db:
            for k, v in items.items():
                db[k] = v

    def size(self) -> int:
        with self._open() as db:
            return len(db)

    def clear(self) -> None:
        with self._open() as db:
            db.clear()


# ── JSON backend ──────────────────────────────────────────────────────────────

class JSONEmbeddingCache(BaseEmbeddingCache):
    """Simple JSON-file-backed cache — easy to inspect, slower for large corpora."""

    def __init__(self, file_path: str = "data/cache/embeddings_cache.json") -> None:
        self._file_path = pathlib.Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, List[float]] = {}
        self._load()

    def _load(self) -> None:
        if self._file_path.exists():
            with open(self._file_path, "r", encoding="utf-8") as fh:
                self._data = json.load(fh)

    def _save(self) -> None:
        with open(self._file_path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh)

    def get(self, key: str) -> Optional[List[float]]:
        return self._data.get(key)

    def set(self, key: str, vector: List[float], **_) -> None:
        self._data[key] = vector
        self._save()

    def get_batch(self, keys: List[str]) -> Dict[str, Optional[List[float]]]:
        return {k: self._data.get(k) for k in keys}

    def set_batch(self, items: Dict[str, List[float]], **_) -> None:
        self._data.update(items)
        self._save()

    def size(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()
        self._save()


# ── Factory ───────────────────────────────────────────────────────────────────

def create_cache(backend: str = "sqlite", **kwargs) -> BaseEmbeddingCache:
    """
    Instantiate and return the requested cache backend.

    Parameters
    ----------
    backend:
        ``"sqlite"`` | ``"pickle"`` | ``"json"``
    **kwargs:
        Passed to the cache constructor (e.g., ``db_path=...``).
    """
    _registry = {
        "sqlite":  SQLiteEmbeddingCache,
        "pickle":  PickleEmbeddingCache,
        "json":    JSONEmbeddingCache,
    }
    if backend not in _registry:
        raise ValueError(f"Unknown cache backend '{backend}'. Choose from: {list(_registry)}")
    return _registry[backend](**kwargs)
