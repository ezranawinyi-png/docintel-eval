#!/usr/bin/env python
"""Run the full benchmark and write results/results.json, results/results.md and charts.

    python scripts/run_eval.py
    python scripts/run_eval.py --llm --price-in 3 --price-out 15   # include the LLM extractor (needs API key)
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from docintel.data import FIELDS, generate_docs, generate_queries  # noqa: E402
from docintel.evaluation import avg_tokens_per_doc, e2e_eval, extraction_eval, retrieval_eval  # noqa: E402
from docintel.extraction import EXTRACTORS, LLMExtractor  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=60)
    ap.add_argument("--llm", action="store_true", help="also evaluate the LLM extractor (needs API key)")
    ap.add_argument("--price-in", type=float, help="USD per million input tokens (for cost estimate)")
    ap.add_argument("--price-out", type=float, help="USD per million output tokens")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    std = generate_docs(args.n_docs, seed=7)
    shifted = generate_docs(args.n_docs, seed=11, shifted=True)
    std_q, shifted_q = generate_queries(std), generate_queries(shifted)

    extractors = dict(EXTRACTORS)
    if args.llm:
        extractors["llm"] = LLMExtractor()

    results = {"config": {"n_docs": args.n_docs, "n_queries": len(std_q)}, "retrieval": [], "extraction": {}, "e2e": {}}

    # 1) retrieval ablation
    for strat in ("fixed-50", "fixed-50+overlap", "paragraph"):
        for ctx in (False, True):
            for kind in ("bm25", "char-tfidf", "hybrid-rrf"):
                results["retrieval"].append(retrieval_eval(std, std_q, strat, kind, ctx))

    # 2) extraction: in-distribution vs held-out phrasing
    for ds_name, ds in (("standard", std), ("shifted (held-out phrasing)", shifted)):
        results["extraction"][ds_name] = {name: extraction_eval(ds, fn) for name, fn in extractors.items()}

    # 3) end-to-end QA
    variants = {
        "baseline (no entity scoping, no expansion)": dict(scope_by_entity=False, expand_query=False),
        "+ entity scoping": dict(scope_by_entity=True, expand_query=False),
        "+ entity scoping + query expansion": dict(scope_by_entity=True, expand_query=True),
    }
    for ds_name, ds, qs in (("standard", std, std_q), ("shifted (held-out phrasing)", shifted, shifted_q)):
        results["e2e"][ds_name] = {v: e2e_eval(ds, qs, **kw) for v, kw in variants.items()}

    results["llm_token_budget"] = avg_tokens_per_doc(std)
    if args.price_in is not None and args.price_out is not None:
        tb = results["llm_token_budget"]
        per_1k = 1000 * (tb["input_tokens_per_doc"] * args.price_in + tb["output_tokens_per_doc"] * args.price_out) / 1e6
        results["llm_token_budget"]["usd_per_1000_docs"] = round(per_1k, 2)

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    (OUT / "results.md").write_text(render_markdown(results))
    try:
        make_charts(results)
    except ImportError:
        print("matplotlib not installed; skipping charts")
    print((OUT / "results.md").read_text())


def render_markdown(r: dict) -> str:
    L = ["# Benchmark results", "",
         f"{r['config']['n_docs']} synthetic contracts, {r['config']['n_queries']} questions "
         "(6 fields x 2 phrasings per document). Retrieval is corpus-wide: every question must find "
         "its answer among *all* documents' chunks.", "",
         "## 1. Retrieval ablation (standard set)", "",
         "| Chunking | Contextual header | Retriever | Recall@1 | Recall@3 | Recall@5 | MRR@10 | Query ms (mean) |",
         "|---|---|---|---|---|---|---|---|"]
    for x in r["retrieval"]:
        L.append(f"| {x['strategy']} | {'yes' if x['contextual'] else 'no'} | {x['retriever']} | "
                 f"{x['recall@1']:.3f} | {x['recall@3']:.3f} | {x['recall@5']:.3f} | {x['mrr@10']:.3f} | "
                 f"{x['query_ms_mean']:.2f} |")
    best = max(r["retrieval"], key=lambda x: x["recall@5"])
    L += ["", f"**Best config:** `{best['strategy']}` + contextual header + `{best['retriever']}` "
              f"(recall@5 = {best['recall@5']:.3f}). Recall@5 by question type for that config: "
              + ", ".join(f"{k} = {v:.3f}" for k, v in best["recall@5_by_variant"].items()) + ".", ""]

    L += ["## 2. Field extraction", "",
          "`standard` matches the phrasings the rule-based v2 extractor was written against (so its score is "
          "optimistic by construction). `shifted` is a held-out set with unseen date formats, number formats "
          "and clause wordings.", ""]
    for ds, per in r["extraction"].items():
        L += [f"### {ds}", "", "| Extractor | Accuracy | Precision | Recall | F1 | Wrong | Missed |", "|---|---|---|---|---|---|---|"]
        for name, m in per.items():
            L.append(f"| {name} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                     f"{m['wrong_values']} | {m['missed_values']} |")
        L += ["", "Per-field accuracy: " + "; ".join(
            f"**{n}**: " + ", ".join(f"{f.split('_')[0]} {v:.2f}" for f, v in m["accuracy_by_field"].items())
            for n, m in per.items()), ""]

    L += ["## 3. End-to-end question answering", "",
          "Pipeline: classify field -> (resolve document) -> hybrid retrieval over contextual chunks -> "
          "merge adjacent chunks -> extract -> cite, or abstain for human review. A *wrong* answer is a "
          "confident error; an *abstention* is a safe failure.", "",
          "| Dataset | Pipeline variant | Accuracy | Wrong | Abstain | Acc. when answered | Query ms (mean / p95) |",
          "|---|---|---|---|---|---|---|"]
    for ds, variants in r["e2e"].items():
        for v, m in variants.items():
            L.append(f"| {ds} | {v} | {m['accuracy']:.3f} | {m['wrong_rate']:.3f} | {m['abstain_rate']:.3f} | "
                     f"{m['accuracy_when_answered']:.3f} | {m['query_ms_mean']:.2f} / {m['query_ms_p95']:.2f} |")
    tb = r["llm_token_budget"]
    L += ["", "## 4. LLM extraction budget (estimate)", "",
          f"About {tb['input_tokens_per_doc']} input + {tb['output_tokens_per_doc']} output tokens per document "
          "(chars/4 heuristic)."]
    if "usd_per_1000_docs" in tb:
        L.append(f" At the supplied prices: **~${tb['usd_per_1000_docs']} per 1,000 documents**.")
    return "\n".join(L) + "\n"


def make_charts(r: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # chart 1: retrieval recall@5, contextual vs not, per retriever (fixed-50+overlap)
    rows = [x for x in r["retrieval"] if x["strategy"] == "fixed-50+overlap"]
    kinds = ["bm25", "char-tfidf", "hybrid-rrf"]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.35
    for i, ctx in enumerate((False, True)):
        vals = [next(x["recall@5"] for x in rows if x["retriever"] == k and x["contextual"] == ctx) for k in kinds]
        bars = ax.bar([j + (i - 0.5) * w for j in range(3)], vals, w,
                      label="with contextual header" if ctx else "no context")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), kinds)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Recall@5")
    ax.set_title("Retrieval: effect of contextual chunk headers (fixed-50+overlap)")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "retrieval_recall.png", dpi=150)
    plt.close(fig)

    # chart 2: extraction accuracy standard vs shifted
    names = list(next(iter(r["extraction"].values())).keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (ds, per) in enumerate(r["extraction"].items()):
        vals = [per[n]["accuracy"] for n in names]
        bars = ax.bar([j + (i - 0.5) * w for j in range(len(names))], vals, w, label=ds)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Field accuracy")
    ax.set_title("Extraction: in-distribution vs held-out phrasing")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "extraction_accuracy.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
