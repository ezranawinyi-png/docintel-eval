#!/usr/bin/env python
"""30-second live demo: ask questions over the demo corpus and print answers with citations."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from docintel.data import generate_docs  # noqa: E402
from docintel.service import Service  # noqa: E402

docs = generate_docs(60, seed=7)
svc = Service(docs)
d1, d2 = docs[3], docs[17]
questions = [
    f"What notice period applies to terminating the {d1.client} agreement?",
    f"What is the liability cap in the {d2.client} agreement?",
    f"Which jurisdiction's law governs the {d1.client} agreement?",
    "What is the airspeed velocity of an unladen swallow?",
]
for q in questions:
    a = svc.ask(q)
    print(f"\nQ: {q}")
    if a["abstained"]:
        print(f"   -> ABSTAINED ({a['reason']})")
    else:
        c = a["citations"][0]
        print(f"   -> {a['field']} = {a['value']}")
        print(f"   source: {c['doc_id']} chars {c['start']}-{c['end']}")
        print(f"   \"{c['text'][:160].strip()}...\"")
print("\nExtraction on raw text (regex-v2):")
print(svc.extract(docs[0].text))
