"""
Chunking Engine — splits pre-processed documents into overlapping text chunks.

Features
--------
* Fixed-size character-level chunking with configurable overlap
* Sentence-boundary aware splitting (avoids cutting mid-sentence when possible)
* Rich metadata preserved on every chunk (source, position, tokens, section)
* Deterministic chunk IDs: ``<doc_id>_chunk_<index>``
* Minimum chunk size guard

Data model
----------
The :class:`Chunk` dataclass is the unit of storage for the vector index.
Every chunk carries enough provenance to reconstruct its origin and to
display meaningful context in retrieval results.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk ready for embedding and vector indexing."""

    chunk_id: str
    source: str
    text: str
    tokens: int                          # approximate token count
    char_start: int                      # start position in original document
    char_end: int                        # end position in original document
    chunk_index: int                     # 0-based index within the document
    section: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "text": self.text,
            "tokens": self.tokens,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "chunk_index": self.chunk_index,
            "section": self.section,
            "metadata": self.metadata,
        }


# ── Engine ───────────────────────────────────────────────────────────────────

class ChunkingEngine:
    """
    Splits a document text into overlapping fixed-size chunks.

    Parameters
    ----------
    chunk_size:
        Target chunk size in characters.
    chunk_overlap:
        Number of characters shared between consecutive chunks.
    min_chunk_size:
        Chunks shorter than this threshold are discarded.
    """

    # Sentence-ending patterns used to find the nearest sentence boundary
    _SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
    # Section header heuristics (markdown / ALLCAPS lines / numbered sections)
    _SECTION_HEADER_RE = re.compile(
        r"^(?:#{1,6}\s+(.+)|([A-Z][A-Z 0-9]{3,}):?|(?:SECTION|Chapter)\s+\d+[.:]\s*(.+))$",
        re.MULTILINE,
    )

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    # ── Public API ───────────────────────────────────────────────────────────

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        source: str,
        metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        """
        Chunk a single document.

        Parameters
        ----------
        text:
            Pre-processed document text.
        doc_id:
            Identifier for the parent document (used to build chunk IDs).
        source:
            Source filename or label (persisted in every chunk).
        metadata:
            Extra key/value pairs forwarded to every chunk's metadata.

        Returns
        -------
        List[Chunk]
        """
        if not text or not text.strip():
            logger.warning("Empty text for document '%s' — skipping.", doc_id)
            return []

        sections = self._extract_sections(text)
        chunks: List[Chunk] = []
        chunk_index = 0

        for section_name, section_text, section_start in sections:
            for raw_chunk, c_start, c_end in self._sliding_window(
                section_text, global_offset=section_start
            ):
                if len(raw_chunk.strip()) < self.min_chunk_size:
                    continue

                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                approximate_tokens = self._approximate_tokens(raw_chunk)
                chunk_meta = dict(metadata or {})
                chunk_meta["doc_id"] = doc_id

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source=source,
                        text=raw_chunk.strip(),
                        tokens=approximate_tokens,
                        char_start=c_start,
                        char_end=c_end,
                        chunk_index=chunk_index,
                        section=section_name,
                        metadata=chunk_meta,
                    )
                )
                chunk_index += 1

        logger.debug(
            "Chunked document '%s' into %d chunks (chunk_size=%d, overlap=%d)",
            doc_id,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def chunk_documents(
        self,
        documents: List,  # List[RawDocument]
    ) -> List[Chunk]:
        """
        Chunk a list of :class:`~src.ingestion.document_loader.RawDocument`
        instances.
        """
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self.chunk_document(
                text=doc.text,
                doc_id=doc.doc_id,
                source=doc.source,
                metadata=doc.metadata,
            )
            all_chunks.extend(chunks)
        logger.info(
            "Total chunks produced from %d documents: %d", len(documents), len(all_chunks)
        )
        return all_chunks

    # ── Private helpers ──────────────────────────────────────────────────────

    def _sliding_window(
        self, text: str, global_offset: int = 0
    ) -> Iterator[tuple[str, int, int]]:
        """
        Yield (chunk_text, start, end) using a sliding window approach.
        Tries to align boundaries to the nearest sentence end.
        """
        step = self.chunk_size - self.chunk_overlap
        pos = 0
        length = len(text)

        while pos < length:
            end = min(pos + self.chunk_size, length)
            # Try to extend to the nearest sentence boundary within a tolerance
            if end < length:
                # Look for sentence end within 20% of chunk_size ahead
                tolerance = min(int(self.chunk_size * 0.2), 80)
                search_end = min(end + tolerance, length)
                snippet = text[end:search_end]
                match = self._SENTENCE_END_RE.search(snippet)
                if match:
                    end = end + match.start() + 1

            raw = text[pos:end]
            yield raw, global_offset + pos, global_offset + end

            pos += step
            if pos >= length:
                break

    def _extract_sections(self, text: str) -> List[tuple[str, str, int]]:
        """
        Split a document into (section_name, section_text, start_offset).
        Falls back to a single section if no headers are found.
        """
        matches = list(self._SECTION_HEADER_RE.finditer(text))
        if not matches:
            return [("general", text, 0)]

        sections: List[tuple[str, str, int]] = []
        for i, m in enumerate(matches):
            header_text = (m.group(1) or m.group(2) or m.group(3) or "unknown").strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_body = text[start:end].strip()
            if section_body:
                sections.append((header_text.lower()[:60], section_body, start))

        # Prepend any text before the first header
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.insert(0, ("preamble", preamble, 0))

        return sections if sections else [("general", text, 0)]

    @staticmethod
    def _approximate_tokens(text: str) -> int:
        """Estimate token count using a simple whitespace split heuristic."""
        return len(text.split())
