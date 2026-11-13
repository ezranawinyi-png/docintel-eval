"""Pure-Python retrievers: BM25, character-n-gram TF-IDF, and reciprocal-rank-fusion hybrid."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .chunking import Chunk

_STOP = set("the a an of to in is are for and or on at by with what which does do how much many "
            "this that be it its as from under".split())


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s", "e"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]


class BM25:
    name = "bm25"

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = [sum(c.values()) for c in self.docs]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df: Counter = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.inv = defaultdict(list)
        for i, c in enumerate(self.docs):
            for t in c:
                self.inv[t].append(i)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(tokenize(query)):
            if t not in self.idf:
                continue
            for i in self.inv[t]:
                tf = self.docs[i][t]
                norm = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
                scores[i] += self.idf[t] * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda x: -x[1])[:k]


def _ngrams(text: str, n: int = 3) -> Counter:
    grams: Counter = Counter()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP:
            continue
        w = f" {w} "
        for i in range(len(w) - n + 1):
            grams[w[i:i + n]] += 1
    return grams


class CharNgramTfidf:
    """Cosine similarity over TF-IDF weighted character 3-grams. Robust to morphology
    ('terminate' / 'terminating' / 'termination') where word-level BM25 is brittle."""
    name = "char-tfidf"

    def __init__(self, texts: list[str]):
        grams = [_ngrams(t) for t in texts]
        df: Counter = Counter()
        for g in grams:
            df.update(g.keys())
        n = len(texts)
        self.idf = {t: math.log((1 + n) / (1 + f)) + 1 for t, f in df.items()}
        self.vecs, self.norms = [], []
        self.inv = defaultdict(list)
        for i, g in enumerate(grams):
            v = {t: (1 + math.log(c)) * self.idf[t] for t, c in g.items()}
            self.vecs.append(v)
            self.norms.append(math.sqrt(sum(x * x for x in v.values())) or 1.0)
            for t in v:
                self.inv[t].append(i)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        q = {t: (1 + math.log(c)) * self.idf.get(t, 0) for t, c in _ngrams(query).items()}
        qn = math.sqrt(sum(x * x for x in q.values())) or 1.0
        scores: dict[int, float] = defaultdict(float)
        for t, w in q.items():
            for i in self.inv.get(t, ()):
                scores[i] += w * self.vecs[i][t]
        return sorted(((i, s / (qn * self.norms[i])) for i, s in scores.items()),
                      key=lambda x: -x[1])[:k]


class Hybrid:
    """Reciprocal Rank Fusion of several retrievers."""
    name = "hybrid-rrf"

    def __init__(self, retrievers: list, rrf_k: int = 60, depth: int = 50):
        self.retrievers, self.rrf_k, self.depth = retrievers, rrf_k, depth

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        fused: dict[int, float] = defaultdict(float)
        for r in self.retrievers:
            for rank, (i, _) in enumerate(r.search(query, self.depth)):
                fused[i] += 1.0 / (self.rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]


def build_retriever(kind: str, chunks: list[Chunk], contextual: bool):
    texts = [c.index_text(contextual) for c in chunks]
    if kind == "bm25":
        return BM25(texts)
    if kind == "char-tfidf":
        return CharNgramTfidf(texts)
    if kind == "hybrid-rrf":
        return Hybrid([BM25(texts), CharNgramTfidf(texts)])
    raise ValueError(f"unknown retriever {kind}")
