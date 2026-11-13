"""Synthetic contract generator with ground-truth labels and evidence spans.

Why synthetic? It gives a fully reproducible, label-complete benchmark with no licensing issues.
The generator deliberately injects (a) several phrasings per field, (b) number/date format
variation, (c) distractor clauses that contain look-alike numbers, and (d) random clause order.
Swap in real documents + labels by producing the same `Doc` objects.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .textutil import MONTHS, num_to_words

FIELDS = ["effective_date", "term_months", "payment_days", "notice_days",
          "governing_law", "liability_cap"]

_ADJ = ["Acme", "Brightwave", "Cedar Ridge", "Delta", "Evergreen", "Falcon", "Granite Peak",
        "Harbor", "Ionic", "Juniper", "Kestrel", "Lumen", "Meridian", "Northwind", "Orchid",
        "Pinnacle", "Quartz", "Riverbend", "Summit", "Tidewater"]
_NOUN = ["Logistics", "Analytics", "Foods", "Freight", "Energy", "Media", "Mining", "Software",
         "Labs", "Health", "Capital", "Textiles", "Robotics", "Foundry", "Retail"]
_STATES = ["Delaware", "New York", "California", "Texas", "Kenya", "England and Wales",
           "Singapore", "Ontario"]


@dataclass
class Doc:
    doc_id: str
    client: str
    vendor: str
    text: str
    gold: dict                      # field -> normalised value
    evidence: dict                  # field -> (char_start, char_end) of the clause paragraph
    header: str = ""                # short context string used for contextual chunking


@dataclass
class Query:
    qid: str
    doc_id: str
    field: str
    text: str


def _fmt_date(rng: random.Random, y: int, m: int, d: int, shifted: bool = False):
    iso = f"{y:04d}-{m:02d}-{d:02d}"
    if shifted and rng.random() < 0.7:  # unseen formats: 'Mar 3, 2024' / 'March 3rd, 2024'
        if rng.random() < 0.5:
            return f"{MONTHS[m - 1][:3]} {d}, {y}", iso
        suf = "th" if 10 <= d % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
        return f"{MONTHS[m - 1]} {d}{suf}, {y}", iso
    style = rng.choice(["us", "uk", "iso"])
    if style == "us":
        return f"{MONTHS[m - 1]} {d}, {y}", iso
    if style == "uk":
        return f"{d} {MONTHS[m - 1]} {y}", iso
    return iso, iso


def _term_clause(rng, n, shifted=False):
    if shifted and rng.random() < 0.7:
        return rng.choice([f"The Agreement runs for a period of {n} months.",
                           f"Services will be provided for {n} months starting on the Effective Date."])
    if n % 12 == 0 and rng.random() < 0.5:
        y = n // 12
        unit = "year" if y == 1 else "years"
        return f"This Agreement shall remain in force for {num_to_words(y)} ({y}) {unit} from the Effective Date."
    return rng.choice([
        f"The initial term of this Agreement is {n} months from the Effective Date.",
        f"The Agreement continues for {num_to_words(n)} ({n}) months from the Effective Date.",
        f"The initial term of this Agreement is {n} months from the Effective Date.",
    ])


def _payment_clause(rng, n, shifted=False):
    if shifted and rng.random() < 0.7:
        return rng.choice([f"Net-{n} terms apply to all invoices.",
                           f"Invoices must be settled no later than {n} calendar days after issue."])
    return rng.choice([
        f"Provider's invoices are payable within {n} days of receipt.",
        f"Client shall pay each invoice net {n}.",
        f"Payment is due {num_to_words(n)} ({n}) days after the invoice date.",
        f"Provider's invoices are payable within {n} days of receipt.",
    ])


def _notice_clause(rng, n, shifted=False):
    if shifted and rng.random() < 0.7:
        return rng.choice([f"Either party may end this Agreement by giving {n} days' written notice.",
                           f"Either party may terminate this Agreement by providing {n} calendar days' notice."])
    return rng.choice([
        f"Either party may terminate this Agreement upon {n} days' written notice.",
        f"Termination requires {n} days prior written notice to the other party.",
        f"A party wishing to terminate must give the other not less than {num_to_words(n)} ({n}) days' notice.",
        f"Either party may terminate this Agreement upon {n} days' written notice.",
    ])


def _law_clause(rng, s, shifted=False):
    if shifted and rng.random() < 0.7:
        return f"This Agreement shall be construed in accordance with the laws of the State of {s}."
    return rng.choice([
        f"This Agreement is governed by the laws of {s}.",
        f"The laws of {s} shall govern this Agreement.",
        f"Governing Law: {s}.",
        f"This Agreement is governed by the laws of {s}.",
    ])


def _cap_clause(rng, amt, shifted=False):
    if shifted and rng.random() < 0.7:
        return f"Provider's maximum aggregate liability is limited to ${amt // 1000}k."
    return rng.choice([
        f"Provider's total liability shall not exceed ${amt:,}.",
        f"Liability under this Agreement is capped at {amt:,} US dollars.",
        f"In no event shall aggregate liability exceed USD {amt:,}.",
        f"Provider's total liability shall not exceed ${amt:,}.",
    ])


def _distractors(rng):
    """Clauses with look-alike numbers that naive extractors confuse with real fields."""
    pool = [
        ("Confidentiality", f"Each party shall keep Confidential Information secret for {rng.choice([2, 3, 5])} years after termination of this Agreement."),
        ("Late Fees", f"Overdue invoices accrue a late fee of 1.5% per month after {rng.choice([10, 15])} days."),
        ("Audit", f"Client may audit Provider's records upon {rng.choice([5, 7, 10])} days' written notice."),
        ("Cure", f"A party in material breach has {rng.choice([20, 30])} days to cure after written notice before the other party may terminate."),
        ("Insurance", f"Provider shall maintain insurance coverage of at least ${rng.choice([1, 2, 5]) * 1_000_000:,}."),
        ("Data Retention", f"Provider will delete Client data within {rng.choice([7, 14, 21])} days of request."),
        ("Intellectual Property", "All deliverables become the property of Client upon full payment."),
        ("Force Majeure", "Neither party is liable for delays caused by events beyond its reasonable control."),
        ("Assignment", "Neither party may assign this Agreement without the other party's prior written consent."),
        ("Entire Agreement", "This Agreement is the entire agreement between the parties and supersedes all prior discussions."),
    ]
    k = rng.randint(6, 8)
    return rng.sample(pool, k)


def generate_docs(n: int = 60, seed: int = 7, shifted: bool = False) -> list[Doc]:
    """`shifted=True` yields a held-out set using phrasings/formats the rule-based extractor
    was NOT designed around (use it to measure generalisation)."""
    rng = random.Random(seed)
    names = [f"{a} {b}" for a in _ADJ for b in _NOUN]
    rng.shuffle(names)
    docs: list[Doc] = []
    for i in range(n):
        client, vendor = names[2 * i], names[2 * i + 1]
        y, m, d = rng.randint(2022, 2026), rng.randint(1, 12), rng.randint(1, 28)
        date_txt, date_iso = _fmt_date(rng, y, m, d, shifted)
        term = rng.choice([6, 12, 18, 24, 36])
        pay = rng.choice([15, 30, 45, 60])
        notice = rng.choice([14, 30, 45, 60, 90])
        law = rng.choice(_STATES)
        cap = rng.choice([50_000, 100_000, 250_000, 500_000, 1_000_000])

        header = f"Service Agreement | Client: {client} | Provider: {vendor}"
        preamble = (f"This Service Agreement is entered into as of {date_txt} between "
                    f"{client} (\"Client\") and {vendor} (\"Provider\").")

        key_clauses = [
            ("Term", _term_clause(rng, term, shifted), "term_months"),
            ("Payment", _payment_clause(rng, pay, shifted), "payment_days"),
            ("Termination", _notice_clause(rng, notice, shifted), "notice_days"),
            ("Governing Law", _law_clause(rng, law, shifted), "governing_law"),
            ("Liability", _cap_clause(rng, cap, shifted), "liability_cap"),
        ]
        clauses = [(t, b, f) for t, b, f in key_clauses] + [(t, b, None) for t, b in _distractors(rng)]
        rng.shuffle(clauses)

        text = f"SERVICE AGREEMENT\nClient: {client}\nProvider: {vendor}\n\n"
        evidence: dict = {}
        start = len(text)
        text += preamble
        evidence["effective_date"] = (start, len(text))
        text += "\n\n"
        for idx, (title, body, fld) in enumerate(clauses, start=1):
            para = f"{idx}. {title}. {body}"
            start = len(text)
            text += para
            if fld:
                evidence[fld] = (start, len(text))
            text += "\n\n"

        gold = {"effective_date": date_iso, "term_months": term, "payment_days": pay,
                "notice_days": notice, "governing_law": law, "liability_cap": cap}
        docs.append(Doc(f"doc{i:03d}", client, vendor, text.rstrip() + "\n", gold, evidence, header))
    return docs


_QUERY_TEMPLATES = {
    "effective_date": ["What is the effective date of the {client} agreement?",
                       "When does the contract with {client} start?"],
    "term_months": ["How long is the initial term of the {client} agreement?",
                    "For how long does the {client} contract run?"],
    "payment_days": ["How many days does {client} have to pay invoices?",
                     "What are the payment terms for {client}?"],
    "notice_days": ["What notice period applies to terminating the {client} agreement?",
                    "How much warning is needed to end the {client} contract?"],
    "governing_law": ["Which jurisdiction's law governs the {client} agreement?",
                      "Where is the {client} contract legally governed?"],
    "liability_cap": ["What is the liability cap in the {client} agreement?",
                      "What is the maximum amount {vendor} can be liable for under the {client} contract?"],
}


def generate_queries(docs: list[Doc]) -> list[Query]:
    out = []
    for d in docs:
        for f in FIELDS:
            for j, tpl in enumerate(_QUERY_TEMPLATES[f]):
                out.append(Query(f"{d.doc_id}-{f}-{j}", d.doc_id, f,
                                 tpl.format(client=d.client, vendor=d.vendor)))
    return out
