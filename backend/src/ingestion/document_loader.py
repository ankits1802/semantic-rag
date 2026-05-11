"""
Document Loader — ingests raw documents from multiple formats.

Supported formats:
    * Plain text  (.txt)
    * Markdown    (.md)
    * JSON list   (.json) — expects a list of objects with a ``content`` field
    * Inline text — strings passed directly to :py:meth:`DocumentLoader.load_text`

Every loaded document is returned as a :class:`RawDocument` dataclass that
carries the original text plus lightweight provenance metadata.
"""

from __future__ import annotations

import json
import pathlib
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class RawDocument:
    """Represents a single loaded document before chunking."""

    doc_id: str
    source: str                        # filename or "inline"
    text: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError(f"Document '{self.doc_id}' has empty text content.")


# ── Loader ───────────────────────────────────────────────────────────────────

class DocumentLoader:
    """
    Loads documents from disk or inline strings into :class:`RawDocument`
    instances ready for the chunking pipeline.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".json"}

    def __init__(self) -> None:
        self._loaded_sources: set[str] = set()

    # ── public API ───────────────────────────────────────────────────────────

    def load_directory(
        self, directory: Union[str, pathlib.Path], recursive: bool = True
    ) -> List[RawDocument]:
        """
        Load all supported documents from *directory*.

        Parameters
        ----------
        directory:
            Path to the directory to scan.
        recursive:
            When ``True``, subdirectories are also scanned.

        Returns
        -------
        List[RawDocument]
            Ordered list of loaded documents.
        """
        dir_path = pathlib.Path(directory)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        glob_pattern = "**/*" if recursive else "*"
        docs: List[RawDocument] = []

        for file_path in sorted(dir_path.glob(glob_pattern)):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    loaded = self.load_file(file_path)
                    docs.extend(loaded)
                except Exception as exc:
                    logger.warning("Skipping %s — %s", file_path, exc)

        logger.info("Loaded %d documents from '%s'", len(docs), directory)
        return docs

    def load_file(self, path: Union[str, pathlib.Path]) -> List[RawDocument]:
        """
        Load a single file.  Returns a list because a JSON file can contain
        multiple document objects.
        """
        file_path = pathlib.Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        if suffix == ".json":
            docs = self._load_json(file_path)
        else:
            docs = [self._load_text_file(file_path)]

        for doc in docs:
            if doc.source in self._loaded_sources:
                logger.debug("Duplicate source skipped: %s", doc.source)
            else:
                self._loaded_sources.add(doc.source)

        logger.debug("Loaded %d document(s) from '%s'", len(docs), file_path.name)
        return docs

    def load_text(
        self,
        text: str,
        doc_id: str = "inline_doc",
        metadata: Optional[dict] = None,
    ) -> RawDocument:
        """
        Create a :class:`RawDocument` directly from an inline string.
        Useful for testing and API ingestion endpoints.
        """
        clean = self._normalise_text(text)
        doc = RawDocument(
            doc_id=doc_id,
            source="inline",
            text=clean,
            metadata=metadata or {},
        )
        logger.debug("Created inline document '%s' (%d chars)", doc_id, len(clean))
        return doc

    def load_documents_from_list(
        self, documents: List[dict]
    ) -> List[RawDocument]:
        """
        Accept a list of dicts — each requires at least ``content`` or ``text``
        and optionally ``id``, ``source``, and ``metadata`` keys.
        """
        results: List[RawDocument] = []
        for idx, item in enumerate(documents):
            text = item.get("content") or item.get("text", "")
            if not text:
                logger.warning("Skipping document at index %d — no content found", idx)
                continue
            doc_id = item.get("id", f"doc_{idx}")
            source = item.get("source", "inline_list")
            metadata = item.get("metadata", {})
            metadata.update({k: v for k, v in item.items()
                             if k not in {"content", "text", "id", "source", "metadata"}})
            results.append(RawDocument(
                doc_id=doc_id,
                source=source,
                text=self._normalise_text(text),
                metadata=metadata,
            ))
        return results

    # ── private helpers ──────────────────────────────────────────────────────

    def _load_text_file(self, path: pathlib.Path) -> RawDocument:
        text = path.read_text(encoding="utf-8", errors="replace")
        clean = self._normalise_text(text)
        return RawDocument(
            doc_id=path.stem,
            source=path.name,
            text=clean,
            metadata={"file_type": path.suffix.lstrip("."), "path": str(path)},
        )

    def _load_json(self, path: pathlib.Path) -> List[RawDocument]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            # Accept {"documents": [...]} wrapper
            items = raw.get("documents", [raw])
        else:
            raise ValueError(f"Unexpected JSON structure in {path.name}")

        docs: List[RawDocument] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            text = item.get("content") or item.get("text") or ""
            if not text.strip():
                logger.warning("Empty content at JSON index %d in %s", idx, path.name)
                continue
            doc_id = item.get("id", f"{path.stem}_{idx}")
            metadata = {
                "file_type": "json",
                "path": str(path),
                "source_item_index": idx,
            }
            # Preserve any extra fields as metadata
            for key, val in item.items():
                if key not in {"id", "content", "text"}:
                    metadata[key] = val
            docs.append(RawDocument(
                doc_id=doc_id,
                source=path.name,
                text=self._normalise_text(text),
                metadata=metadata,
            ))
        return docs

    @staticmethod
    def _normalise_text(text: str) -> str:
        """Apply Unicode NFKC normalisation and collapse whitespace."""
        normalised = unicodedata.normalize("NFKC", text)
        # Collapse multiple blank lines to at most two
        import re
        normalised = re.sub(r"\n{3,}", "\n\n", normalised)
        # Replace tab characters with spaces
        normalised = normalised.replace("\t", "    ")
        return normalised.strip()
