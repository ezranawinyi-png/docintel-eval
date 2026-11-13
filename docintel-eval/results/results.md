# Benchmark results

60 synthetic contracts, 720 questions (6 fields x 2 phrasings per document). Retrieval is corpus-wide: every question must find its answer among *all* documents' chunks.

## 1. Retrieval ablation (standard set)

| Chunking | Contextual header | Retriever | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Query ms (mean) |
|---|---|---|---|---|---|---|---|
| fixed-50 | no | bm25 | 0.261 | 0.307 | 0.311 | 0.285 | 0.09 |
| fixed-50 | no | char-tfidf | 0.256 | 0.310 | 0.314 | 0.283 | 0.57 |
| fixed-50 | no | hybrid-rrf | 0.263 | 0.307 | 0.311 | 0.286 | 0.71 |
| fixed-50 | yes | bm25 | 0.308 | 0.640 | 0.831 | 0.514 | 0.11 |
| fixed-50 | yes | char-tfidf | 0.233 | 0.569 | 0.765 | 0.446 | 0.69 |
| fixed-50 | yes | hybrid-rrf | 0.293 | 0.649 | 0.817 | 0.504 | 0.95 |
| fixed-50+overlap | no | bm25 | 0.261 | 0.307 | 0.311 | 0.285 | 0.13 |
| fixed-50+overlap | no | char-tfidf | 0.260 | 0.308 | 0.311 | 0.285 | 1.08 |
| fixed-50+overlap | no | hybrid-rrf | 0.263 | 0.307 | 0.311 | 0.286 | 1.56 |
| fixed-50+overlap | yes | bm25 | 0.314 | 0.681 | 0.833 | 0.530 | 0.17 |
| fixed-50+overlap | yes | char-tfidf | 0.265 | 0.661 | 0.843 | 0.497 | 1.34 |
| fixed-50+overlap | yes | hybrid-rrf | 0.300 | 0.701 | 0.871 | 0.532 | 1.54 |
| paragraph | no | bm25 | 0.000 | 0.136 | 0.169 | 0.075 | 0.14 |
| paragraph | no | char-tfidf | 0.000 | 0.117 | 0.150 | 0.067 | 1.13 |
| paragraph | no | hybrid-rrf | 0.000 | 0.124 | 0.161 | 0.071 | 1.38 |
| paragraph | yes | bm25 | 0.319 | 0.532 | 0.665 | 0.470 | 0.31 |
| paragraph | yes | char-tfidf | 0.301 | 0.574 | 0.697 | 0.471 | 1.94 |
| paragraph | yes | hybrid-rrf | 0.361 | 0.601 | 0.704 | 0.514 | 2.47 |

**Best config:** `fixed-50+overlap` + contextual header + `hybrid-rrf` (recall@5 = 0.871). Recall@5 by question type for that config: lexical-overlap = 0.922, paraphrased = 0.819.

## 2. Field extraction

`standard` matches the phrasings the rule-based v2 extractor was written against (so its score is optimistic by construction). `shifted` is a held-out set with unseen date formats, number formats and clause wordings.

### standard

| Extractor | Accuracy | Precision | Recall | F1 | Wrong | Missed |
|---|---|---|---|---|---|---|
| regex-v1 | 0.411 | 0.614 | 0.411 | 0.493 | 93 | 119 |
| regex-v2 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |

Per-field accuracy: **regex-v1**: effective 0.30, term 0.48, payment 0.50, notice 0.28, governing 0.53, liability 0.37; **regex-v2**: effective 1.00, term 1.00, payment 1.00, notice 1.00, governing 1.00, liability 1.00

### shifted (held-out phrasing)

| Extractor | Accuracy | Precision | Recall | F1 | Wrong | Missed |
|---|---|---|---|---|---|---|
| regex-v1 | 0.292 | 0.461 | 0.292 | 0.357 | 123 | 132 |
| regex-v2 | 0.300 | 0.720 | 0.300 | 0.424 | 42 | 210 |

Per-field accuracy: **regex-v1**: effective 0.18, term 0.90, payment 0.13, notice 0.30, governing 0.15, liability 0.08; **regex-v2**: effective 0.33, term 0.25, payment 0.35, notice 0.25, governing 0.32, liability 0.30

## 3. End-to-end question answering

Pipeline: classify field -> (resolve document) -> hybrid retrieval over contextual chunks -> merge adjacent chunks -> extract -> cite, or abstain for human review. A *wrong* answer is a confident error; an *abstention* is a safe failure.

| Dataset | Pipeline variant | Accuracy | Wrong | Abstain | Acc. when answered | Query ms (mean / p95) |
|---|---|---|---|---|---|---|
| standard | baseline (no entity scoping, no expansion) | 0.710 | 0.219 | 0.071 | 0.764 | 2.06 / 3.20 |
| standard | + entity scoping | 0.963 | 0.000 | 0.037 | 1.000 | 1.87 / 2.79 |
| standard | + entity scoping + query expansion | 1.000 | 0.000 | 0.000 | 1.000 | 2.85 / 4.14 |
| shifted (held-out phrasing) | baseline (no entity scoping, no expansion) | 0.232 | 0.375 | 0.393 | 0.382 | 2.03 / 3.15 |
| shifted (held-out phrasing) | + entity scoping | 0.289 | 0.117 | 0.594 | 0.712 | 2.29 / 3.55 |
| shifted (held-out phrasing) | + entity scoping + query expansion | 0.300 | 0.117 | 0.583 | 0.720 | 3.02 / 4.45 |

## 4. LLM extraction budget (estimate)

About 568 input + 80 output tokens per document (chars/4 heuristic).
