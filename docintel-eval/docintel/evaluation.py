"""Evaluation harness: retrieval (recall@k, MRR), extraction (accuracy, P/R/F1) and end-to-end QA."""
from __future__ import annotations

import time
from collections import defaultdict

from .chunking import STRATEGIES, is_relevant
from .data import FIELDS, Doc, Query
from .pipeline import QAPipeline
from .retrieval import build_retriever


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else 0.0


def retrieval_eval(docs: list[Doc], queries: list[Query], strategy: str, retriever: str,
                   contextual: bool, ks=(1, 3, 5)) -> dict:
    by_id = {d.doc_id: d for d in docs}
    chunks = [c for d in docs for c in STRATEGIES[strategy](d)]
    t0 = time.perf_counter()
    r = build_retriever(retriever, chunks, contextual)
    build_ms = (time.perf_counter() - t0) * 1000
    hits = {k: 0 for k in ks}
    variant_hits: dict = defaultdict(lambda: [0, 0])
    rr_sum, lat = 0.0, []
    for q in queries:
        ev = by_id[q.doc_id].evidence[q.field]
        t = time.perf_counter()
        res = r.search(q.text, 10)
        lat.append((time.perf_counter() - t) * 1000)
        rank = next((i + 1 for i, (ci, _) in enumerate(res) if is_relevant(chunks[ci], q.doc_id, ev)), None)
        if rank:
            rr_sum += 1 / rank
            for k in ks:
                hits[k] += rank <= k
        variant = "lexical-overlap" if q.qid.endswith("-0") else "paraphrased"
        variant_hits[variant][1] += 1
        variant_hits[variant][0] += bool(rank and rank <= 5)
    n = len(queries)
    return {
        "strategy": strategy, "retriever": retriever, "contextual": contextual,
        "n_chunks": len(chunks), "n_queries": n,
        **{f"recall@{k}": hits[k] / n for k in ks},
        "mrr@10": rr_sum / n,
        "recall@5_by_variant": {k: v[0] / v[1] for k, v in variant_hits.items()},
        "build_ms": build_ms, "query_ms_mean": sum(lat) / n, "query_ms_p95": _pct(lat, 0.95),
    }


def extraction_eval(docs: list[Doc], extractor) -> dict:
    correct = defaultdict(int)
    predicted = defaultdict(int)
    wrong, missed, examples = 0, 0, []
    for d in docs:
        pred = extractor(d.text)
        for f in FIELDS:
            p, g = pred.get(f), d.gold[f]
            if p is not None:
                predicted[f] += 1
            if p == g:
                correct[f] += 1
            elif p is None:
                missed += 1
            else:
                wrong += 1
                if len(examples) < 8:
                    examples.append({"doc": d.doc_id, "field": f, "gold": g, "pred": p})
    n_docs = len(docs)
    tot_c, tot_p, tot_g = sum(correct.values()), sum(predicted.values()), n_docs * len(FIELDS)
    prec = tot_c / tot_p if tot_p else 0.0
    rec = tot_c / tot_g
    return {
        "accuracy_by_field": {f: correct[f] / n_docs for f in FIELDS},
        "accuracy": tot_c / tot_g, "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "wrong_values": wrong, "missed_values": missed, "error_examples": examples,
    }


def e2e_eval(docs: list[Doc], queries: list[Query], **pipeline_kwargs) -> dict:
    by_id = {d.doc_id: d for d in docs}
    pipe = QAPipeline(docs, **pipeline_kwargs)
    correct = wrong = abstained = 0
    lat = []
    per_field = {f: [0, 0] for f in FIELDS}
    for q in queries:
        t = time.perf_counter()
        a = pipe.ask(q.text)
        lat.append((time.perf_counter() - t) * 1000)
        gold = by_id[q.doc_id].gold[q.field]
        right = (not a.abstained) and a.value == gold and a.citations and a.citations[0]["doc_id"] == q.doc_id
        per_field[q.field][1] += 1
        if a.abstained:
            abstained += 1
        elif right:
            correct += 1
            per_field[q.field][0] += 1
        else:
            wrong += 1
    n = len(queries)
    answered = correct + wrong
    return {
        "n_queries": n, "accuracy": correct / n, "wrong_rate": wrong / n, "abstain_rate": abstained / n,
        "accuracy_when_answered": correct / answered if answered else 0.0,
        "accuracy_by_field": {f: c / t for f, (c, t) in per_field.items()},
        "query_ms_mean": sum(lat) / n, "query_ms_p95": _pct(lat, 0.95),
    }


def avg_tokens_per_doc(docs: list[Doc], overhead_in: int = 250, out_tokens: int = 80) -> dict:
    """Rough LLM-extraction token budget (chars/4 heuristic) so cost can be estimated."""
    n_in = sum(len(d.text) / 4 + overhead_in for d in docs) / len(docs)
    return {"input_tokens_per_doc": round(n_in), "output_tokens_per_doc": out_tokens}
