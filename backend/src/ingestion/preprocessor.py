"""
Text Preprocessor — cleans and normalises raw document text before chunking.

Transformations applied (all individually toggleable):
    1. Lowercasing
    2. Whitespace normalisation
    3. Unicode NFKC normalisation
    4. Duplicate sentence / paragraph removal
    5. Boilerplate filtering (very short lines, repeated dashes, etc.)

Designed to be called on :py:attr:`RawDocument.text` *before* the
:class:`~src.ingestion.chunking_engine.ChunkingEngine` splits the text into
chunks, so that every downstream chunk is already clean.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Set

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessorConfig:
    """Configuration flags for the preprocessor pipeline."""

    lowercase: bool = True
    normalise_unicode: bool = True
    collapse_whitespace: bool = True
    remove_duplicate_lines: bool = True
    remove_boilerplate: bool = True
    min_line_length: int = 10          # discard lines shorter than this
    remove_repeated_punctuation: bool = True


class TextPreprocessor:
    """
    Stateless text-cleaning pipeline.

    Usage::

        preprocessor = TextPreprocessor()
        clean_text = preprocessor.preprocess("Raw   document  text...")
    """

    # Regex patterns compiled once at class load
    _WHITESPACE_RE = re.compile(r"[ \t]+")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    _REPEATED_DASH_RE = re.compile(r"-{4,}")
    _REPEATED_EQUALS_RE = re.compile(r"={4,}")
    _REPEATED_PUNCT_RE = re.compile(r"([!?.,:;])\1{2,}")
    _MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
    _MARKDOWN_BOLD_ITALIC_RE = re.compile(r"\*{1,3}(.+?)\*{1,3}")
    _MARKDOWN_LINK_RE = re.compile(r"\[(.+?)\]\(.+?\)")
    _CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    _INLINE_CODE_RE = re.compile(r"`([^`]+)`")

    def __init__(self, config: Optional[PreprocessorConfig] = None) -> None:
        self.config = config or PreprocessorConfig()

    # ── Public API ───────────────────────────────────────────────────────────

    def preprocess(self, text: str) -> str:
        """
        Run the full preprocessing pipeline on *text* and return the cleaned string.

        Parameters
        ----------
        text:
            Raw input text.

        Returns
        -------
        str
            Cleaned, normalised text.
        """
        if not text or not text.strip():
            return ""

        cfg = self.config

        # 1. Unicode normalisation
        if cfg.normalise_unicode:
            text = unicodedata.normalize("NFKC", text)

        # 2. Strip Markdown syntax while preserving content
        text = self._strip_markdown(text)

        # 3. Remove boilerplate lines (heavy separators, very short lines)
        if cfg.remove_boilerplate:
            text = self._remove_boilerplate_lines(text, cfg.min_line_length)

        # 4. Remove duplicate consecutive lines / paragraphs
        if cfg.remove_duplicate_lines:
            text = self._remove_duplicate_lines(text)

        # 5. Lowercase (applied after duplicate removal so deduplication is
        #    case-sensitive as intended for section headers)
        if cfg.lowercase:
            text = text.lower()

        # 6. Remove repeated punctuation
        if cfg.remove_repeated_punctuation:
            text = self._REPEATED_PUNCT_RE.sub(r"\1", text)

        # 7. Whitespace collapse
        if cfg.collapse_whitespace:
            text = self._WHITESPACE_RE.sub(" ", text)
            text = self._MULTI_NEWLINE_RE.sub("\n\n", text)

        return text.strip()

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Apply :py:meth:`preprocess` to each element in *texts*."""
        return [self.preprocess(t) for t in texts]

    def remove_duplicates(self, texts: List[str]) -> List[str]:
        """
        Deduplicate a list of text chunks preserving insertion order.

        Two chunks are considered duplicates when their normalised lowercase
        representation is identical.
        """
        seen: Set[str] = set()
        unique: List[str] = []
        for t in texts:
            key = " ".join(t.lower().split())
            if key not in seen:
                seen.add(key)
                unique.append(t)
        duplicates_removed = len(texts) - len(unique)
        if duplicates_removed:
            logger.debug("Removed %d duplicate text chunks.", duplicates_removed)
        return unique

    # ── Private helpers ──────────────────────────────────────────────────────

    def _strip_markdown(self, text: str) -> str:
        """Replace common Markdown syntax with its plain text equivalent."""
        # Remove code fences but keep the code content
        text = self._CODE_FENCE_RE.sub(lambda m: m.group(0).strip("```").strip(), text)
        # Inline code → plain text
        text = self._INLINE_CODE_RE.sub(r"\1", text)
        # Bold/italic → plain text
        text = self._MARKDOWN_BOLD_ITALIC_RE.sub(r"\1", text)
        # Links → link text only
        text = self._MARKDOWN_LINK_RE.sub(r"\1", text)
        # Remove header markers but keep text
        text = self._MARKDOWN_HEADER_RE.sub("", text)
        # Remove heavy separator lines
        text = self._REPEATED_DASH_RE.sub("", text)
        text = self._REPEATED_EQUALS_RE.sub("", text)
        return text

    @staticmethod
    def _remove_boilerplate_lines(text: str, min_length: int) -> str:
        """
        Remove lines that are too short to be meaningful (navigation links,
        single-word headings, empty dividers, etc.).
        """
        lines = text.splitlines()
        filtered: List[str] = []
        for line in lines:
            stripped = line.strip()
            # Keep empty lines as paragraph separators
            if not stripped:
                filtered.append("")
                continue
            # Drop lines that are purely punctuation / symbols
            if re.fullmatch(r"[^a-zA-Z0-9 ]+", stripped):
                continue
            if len(stripped) >= min_length:
                filtered.append(line)
        return "\n".join(filtered)

    @staticmethod
    def _remove_duplicate_lines(text: str) -> str:
        """Remove consecutive identical lines (e.g. repeated headings)."""
        lines = text.splitlines()
        result: List[str] = []
        prev = None
        for line in lines:
            normalised = line.strip().lower()
            if normalised != prev:
                result.append(line)
                prev = normalised
        return "\n".join(result)
