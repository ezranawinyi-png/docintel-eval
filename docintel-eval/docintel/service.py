"""Framework-free service layer (easy to test); `api.py` is a thin FastAPI wrapper around it."""
from __future__ import annotations

import pathlib
from dataclasses import asdict

from .data import Doc, generate_docs
from .extraction import EXTRACTORS, LLMExtractor
from .pipeline import QAPipeline


def docs_from_directory(path: str) -> list[Doc]:
    """Load every .txt file in a directory as an (unlabelled) document."""
    docs = []
    for f in sorted(pathlib.Path(path).glob("*.txt")):
        text = f.read_text(encoding="utf-8", errors="replace")
        docs.append(Doc(doc_id=f.stem, client="", vendor="", text=text, gold={}, evidence={},
                        header=f"Document: {f.stem}"))
    if not docs:
        raise ValueError(f"no .txt files found in {path}")
    return docs


class Service:
    def __init__(self, docs: list[Doc] | None = None):
        self.docs = docs or generate_docs(60, seed=7)
        self.pipeline = QAPipeline(self.docs)

    def extract(self, text: str, method: str = "regex-v2") -> dict:
        if method == "llm":
            return LLMExtractor()(text)
        if method not in EXTRACTORS:
            raise ValueError(f"unknown method '{method}'; choose from {sorted(EXTRACTORS) + ['llm']}")
        return EXTRACTORS[method](text)

    def ask(self, question: str) -> dict:
        return asdict(self.pipeline.ask(question))

    def stats(self) -> dict:
        return {"documents": len(self.docs), "chunks": len(self.pipeline.chunks)}
