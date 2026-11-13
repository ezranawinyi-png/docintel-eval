"""End-to-end question answering: classify -> retrieve -> extract -> cite."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .chunking import Chunk, STRATEGIES
from .data import Doc
from .extraction import extract_v2_with_evidence
from .retrieval import build_retriever

# Ordered rules: first match wins. Word-boundary regexes (a bare substring test once matched
# the "cap" in company names such as "Falcon Capital" - found via the evaluation harness).
_FIELD_RULES = [
    ("liability_cap", r"\bliab|\bcap(?:ped)?\b"),
    ("governing_law", r"\bjurisdiction|\bgovern|\blaws?\b"),
    ("notice_days", r"\bnotice\b|\bterminat|\bwarning\b|\bend the\b"),
    ("payment_days", r"\bpay"),
    ("term_months", r"\binitial term\b|\bhow long\b|\brun\b"),
    ("effective_date", r"\beffective\b|\bstart\b|\btake effect\b"),
]


# Domain vocabulary appended to the query once the target field is known (query expansion).
FIELD_HINTS = {
    "effective_date": "effective date entered into as of dated",
    "term_months": "initial term months years remain in force continues",
    "payment_days": "payment invoices payable due days net",
    "notice_days": "terminate termination written notice days",
    "governing_law": "governing law governed laws jurisdiction",
    "liability_cap": "liability limited cap exceed aggregate",
}


def classify_field(question: str) -> str | None:
    q = question.lower()
    for fld, pat in _FIELD_RULES:
        if re.search(pat, q):
            return fld
    return None


@dataclass
class Answer:
    question: str
    field: str | None
    value: object
    citations: list = field(default_factory=list)   # [{doc_id, start, end, text, score}]
    abstained: bool = False
    reason: str = ""


class QAPipeline:
    """Defaults to the best configuration found in the ablation study."""

    def __init__(self, docs: list[Doc], strategy: str = "fixed-50+overlap",
                 retriever: str = "hybrid-rrf", contextual: bool = True,
                 top_k: int = 5, min_score: float = 0.0, scope_by_entity: bool = True,
                 expand_query: bool = True):
        self.chunks: list[Chunk] = [c for d in docs for c in STRATEGIES[strategy](d)]
        self.retriever = build_retriever(retriever, self.chunks, contextual)
        self.top_k, self.min_score, self.scope = top_k, min_score, scope_by_entity
        self.docs = {d.doc_id: d for d in docs}
        self.expand = expand_query
        # entity name -> doc_id, used to resolve which document a question is about
        self.entities = {}
        for d in docs:
            for name in (d.client, d.vendor):
                if name.strip():          # documents without entity metadata are simply unscoped
                    self.entities[name.lower()] = d.doc_id

    def resolve_doc(self, question: str) -> str | None:
        """Longest entity name mentioned in the question -> its document (None if no match)."""
        q = question.lower()
        found = [name for name in self.entities if name in q]
        return self.entities[max(found, key=len)] if found else None

    def ask(self, question: str) -> Answer:
        fld = classify_field(question)
        target = self.resolve_doc(question) if self.scope else None
        query = f"{question} {FIELD_HINTS[fld]}" if (fld and self.expand) else question
        if target:
            hits = [(i, sc) for i, sc in self.retriever.search(query, 200)
                    if self.chunks[i].doc_id == target][: self.top_k]
        else:
            hits = self.retriever.search(query, self.top_k)
        if fld is None:
            return Answer(question, None, None, abstained=True, reason="could not map question to a known field")
        if not hits or hits[0][1] <= self.min_score:
            return Answer(question, fld, None, abstained=True, reason="retrieval confidence too low")
        for doc_id, start, end, score in self._merge(hits):
            text = self.docs[doc_id].text[start:end]
            values, evidence = extract_v2_with_evidence(text)
            val = values.get(fld)
            if val is not None:
                sent = evidence.get(fld, "")
                at = text.find(sent) if sent else -1
                s0, e0 = (start + at, start + at + len(sent)) if at >= 0 else (start, end)
                cite = {"doc_id": doc_id, "start": s0, "end": e0,
                        "text": self.docs[doc_id].text[s0:e0], "score": round(score, 4)}
                return Answer(question, fld, val, [cite])
        return Answer(question, fld, None, abstained=True,
                      reason="no retrieved passage contained a parseable value (flag for human review)")

    def _merge(self, hits):
        """Merge overlapping/adjacent retrieved chunks of the same document into single spans so a
        clause that straddles a chunk boundary is reassembled. Spans are ordered by best score."""
        by_doc: dict = {}
        for idx, score in hits:
            c = self.chunks[idx]
            by_doc.setdefault(c.doc_id, []).append((c.start, c.end, score))
        spans = []
        for doc_id, items in by_doc.items():
            items.sort()
            cur_s, cur_e, cur_sc = items[0]
            for st, en, sc in items[1:]:
                if st <= cur_e + 1:
                    cur_e, cur_sc = max(cur_e, en), max(cur_sc, sc)
                else:
                    spans.append((doc_id, cur_s, cur_e, cur_sc))
                    cur_s, cur_e, cur_sc = st, en, sc
            spans.append((doc_id, cur_s, cur_e, cur_sc))
        return sorted(spans, key=lambda x: -x[3])
