"""Chunking strategies. Every chunk keeps character offsets so retrieval can be scored
against gold evidence spans regardless of how the text was split."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .data import Doc


@dataclass
class Chunk:
    doc_id: str
    start: int
    end: int
    text: str
    header: str = ""

    def index_text(self, contextual: bool) -> str:
        """Text that is actually indexed. Contextual chunking prepends the document header
        (client/provider names) so a clause that never mentions the client is still findable."""
        return f"{self.header}\n{self.text}" if contextual else self.text


def fixed_window(doc: Doc, size: int = 50, overlap: int = 0) -> list[Chunk]:
    toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", doc.text)]
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(toks), step):
        window = toks[i:i + size]
        if not window:
            break
        s, e = window[0][0], window[-1][1]
        chunks.append(Chunk(doc.doc_id, s, e, doc.text[s:e], doc.header))
        if i + size >= len(toks):
            break
    return chunks


def paragraph(doc: Doc) -> list[Chunk]:
    chunks, pos = [], 0
    for part in doc.text.split("\n\n"):
        s = doc.text.index(part, pos) if part else pos
        e = s + len(part)
        pos = e
        if part.strip():
            chunks.append(Chunk(doc.doc_id, s, e, part.strip(), doc.header))
    return chunks


STRATEGIES = {
    "fixed-50": lambda d: fixed_window(d, 50, 0),
    "fixed-50+overlap": lambda d: fixed_window(d, 50, 20),
    "paragraph": paragraph,
}


def is_relevant(chunk: Chunk, doc_id: str, evidence: tuple[int, int], min_cover: float = 0.5) -> bool:
    """A chunk is relevant if it is from the right document and covers >= `min_cover`
    of the gold evidence span."""
    if chunk.doc_id != doc_id:
        return False
    s, e = evidence
    overlap = max(0, min(chunk.end, e) - max(chunk.start, s))
    return overlap / max(1, e - s) >= min_cover
