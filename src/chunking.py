from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        # Split on sentence-ending punctuation followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        return self._split(text, list(self.separators))

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        """Recursively split current_text using the next available separator."""
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # No separators left — character-level fallback
        if not remaining_separators:
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        sep = remaining_separators[0]
        rest = remaining_separators[1:]

        if sep == "":
            # Empty separator: character-level split
            return self._split(current_text, rest) if rest else [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        parts = current_text.split(sep)
        chunks: list[str] = []
        current_piece = ""
        for part in parts:
            candidate = current_piece + (sep if current_piece else "") + part
            if len(candidate) <= self.chunk_size:
                current_piece = candidate
            else:
                if current_piece:
                    chunks.append(current_piece)
                # If part itself is too large, recurse with remaining separators
                if len(part) > self.chunk_size:
                    chunks.extend(self._split(part, rest))
                    current_piece = ""
                else:
                    current_piece = part
        if current_piece:
            chunks.append(current_piece)
        return chunks if chunks else [current_text]


class HeadingChunker:
    """
    Split text based on document headings (#, ##, ###, Điều ...).

    Preserves semantic section boundaries from university regulations.
    When a section exceeds max_chunk_size, recursively splits it while
    re-injecting the heading into each sub-chunk to prevent context loss.
    """

    def __init__(self, max_chunk_size: int = 500) -> None:
        self.max_chunk_size = max_chunk_size
        self._fallback = RecursiveChunker(chunk_size=max_chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        lines = text.splitlines()
        sections: list[tuple[str, list[str]]] = []
        current_heading = ""
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            # Detect Markdown headings (#, ##, ...) or Điều sections
            if stripped.startswith(("# ", "## ", "### ", "#### ")) or re.match(r"^Điều\s+\d+", stripped):
                if current_lines:
                    sections.append((current_heading, current_lines))
                current_heading = stripped
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, current_lines))

        if not sections:
            sections = [("", lines)]

        chunks: list[str] = []
        for heading, s_lines in sections:
            sec_text = "\n".join(s_lines).strip()
            if not sec_text:
                continue
            if len(sec_text) <= self.max_chunk_size:
                chunks.append(sec_text)
            else:
                # Sub-chunk section that is too long, re-inject heading for context
                sub_chunks = self._fallback.chunk(sec_text)
                for sc in sub_chunks:
                    sc_clean = sc.strip()
                    if heading and not sc_clean.startswith(heading):
                        chunks.append(f"{heading}\n{sc_clean}")
                    else:
                        chunks.append(sc_clean)

        return chunks if chunks else [text.strip()]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0).chunk(text),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3).chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }
        result = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
        return result
