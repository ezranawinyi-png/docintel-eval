"""Field extractors.

* ``extract_v1``  - naive first-match regexes (the baseline).
* ``extract_v2``  - sentence-scoped extraction with exclusion rules, number-word and
                    multi-format date/money handling.
* ``LLMExtractor``- optional LLM-based extractor behind a ``complete(prompt) -> str`` callable
                    (works with any provider; an Anthropic default is included).

All extractors return ``{field: normalised_value | None}`` so they can be scored by the same
harness and compared head-to-head on your own documents.
"""
from __future__ import annotations

import json
import os
import re

from .data import FIELDS
from .textutil import NUM_WORD_RE, parse_date, parse_money, words_to_num

# --------------------------------------------------------------------------- v1 (baseline)
def extract_v1(text: str) -> dict:
    def first(pattern, conv=lambda x: x):
        m = re.search(pattern, text)
        return conv(m.group(1)) if m else None

    return {
        "effective_date": first(r"as of ([A-Z][a-z]+ \d{1,2}, \d{4})", parse_date),
        "term_months": first(r"(\d+) months", int),
        "payment_days": first(r"within (\d+) days", int),
        "notice_days": first(r"(\d+) days' written notice", int),
        "governing_law": first(r"governed by the laws of ([A-Z][A-Za-z ]+?)\.", str),
        "liability_cap": first(r"\$([\d,]+)", lambda s: int(s.replace(",", ""))),
    }


# --------------------------------------------------------------------------- v2
_DATE_RE = (r"(\d{4}-\d{2}-\d{2}|[A-Z][a-z]+ \d{1,2}, \d{4}|\d{1,2} [A-Z][a-z]+ \d{4})")
_STATE = r"[A-Z][a-z]+(?:(?: and)? [A-Z][a-z]+)*"
_MONEY_RE = r"(?:\$|USD\s*)(\d{1,3}(?:,\d{3})+|\d+)|(\d{1,3}(?:,\d{3})+|\d+)\s+US dollars"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _quantity(sentence: str, units: str):
    """Return (n, unit) for '45 days', 'forty-five (45) days', '2 years'; None if absent."""
    m = re.search(
        rf"(?:\b(\d+)\b|\b({NUM_WORD_RE})\b)(?:\s*\((\d+)\))?\s+({units})s?\b", sentence, re.I)
    if not m:
        return None
    if m.group(3):
        n = int(m.group(3))
    elif m.group(1):
        n = int(m.group(1))
    else:
        n = words_to_num(m.group(2))
    return (n, m.group(4).lower()) if n is not None else None


def _first_sentence(text, require, exclude=None):
    for s in _sentences(text):
        if re.search(require, s, re.I) and not (exclude and re.search(exclude, s, re.I)):
            yield s


def extract_v2(text: str) -> dict:
    return extract_v2_with_evidence(text)[0]


def extract_v2_with_evidence(text: str) -> tuple[dict, dict]:
    """Like extract_v2 but also returns {field: supporting sentence} for citations."""
    out: dict = {f: None for f in FIELDS}
    ev: dict = {}

    for s in _first_sentence(text, r"entered into|effective"):
        m = re.search(_DATE_RE, s)
        if m and parse_date(m.group(1)):
            out["effective_date"] = parse_date(m.group(1))
            ev["effective_date"] = s
            break

    for s in _first_sentence(text, r"initial term|remain in force|continues for"):
        q = _quantity(s, "month|year")
        if q:
            out["term_months"] = q[0] * 12 if q[1].startswith("year") else q[0]
            ev["term_months"] = s
            break

    for s in _first_sentence(text, r"invoice|payment|\bpay\b", exclude=r"late|overdue|accru"):
        m = re.search(r"\bnet\s+(\d+)\b", s, re.I)
        q = _quantity(s, "day")
        if m or q:
            out["payment_days"] = int(m.group(1)) if m else q[0]
            ev["payment_days"] = s
            break

    for s in _first_sentence(text, r"terminat", exclude=r"breach|cure|after termination"):
        if not re.search(r"notice", s, re.I):
            continue
        q = _quantity(s, "day")
        if q:
            out["notice_days"] = q[0]
            ev["notice_days"] = s
            break

    for pat in (rf"[Ll]aws of ({_STATE})(?=\s+shall govern|\s*[.,])", rf"Governing Law:\s*({_STATE})"):
        m = re.search(pat, text)
        if m:
            out["governing_law"] = m.group(1)
            ev["governing_law"] = next(
                (x for x in _sentences(text) if m.group(1) in x and re.search("law", x, re.I)), m.group(0))
            break

    for s in _first_sentence(text, r"liabilit"):
        m = re.search(_MONEY_RE, s)
        if m:
            out["liability_cap"] = parse_money(m.group(1) or m.group(2))
            ev["liability_cap"] = s
            break
    return out, ev


# --------------------------------------------------------------------------- LLM extractor
_PROMPT = """Extract these fields from the contract below. Respond with ONLY a JSON object.
Use null for any field that is not stated. Do not guess.

Fields and formats:
- effective_date: ISO date string YYYY-MM-DD
- term_months: integer, the initial term expressed in months
- payment_days: integer, days allowed to pay an invoice
- notice_days: integer, days of notice required to TERMINATE for convenience
  (not audit notice, not breach cure periods)
- governing_law: jurisdiction name only, e.g. "Delaware"
- liability_cap: integer amount in dollars, the cap on the provider's liability
  (not insurance coverage)

CONTRACT:
{text}
"""


def _anthropic_complete(prompt: str) -> str:  # pragma: no cover - needs network + API key
    import anthropic  # lazy import: optional dependency

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    model = os.environ.get("DOCINTEL_MODEL", "claude-sonnet-5-5")
    msg = client.messages.create(model=model, max_tokens=500,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class LLMExtractor:
    def __init__(self, complete=None):
        self.complete = complete or _anthropic_complete

    def __call__(self, text: str) -> dict:
        raw = self.complete(_PROMPT.format(text=text))
        return self.parse(raw)

    @staticmethod
    def parse(raw: str) -> dict:
        """Tolerant JSON parsing + type validation. Anything malformed becomes None."""
        out: dict = {f: None for f in FIELDS}
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return out
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return out
        if isinstance(data.get("effective_date"), str):
            out["effective_date"] = parse_date(data["effective_date"])
        for f in ("term_months", "payment_days", "notice_days", "liability_cap"):
            v = data.get(f)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and v == int(v):
                out[f] = int(v)
            elif isinstance(v, str) and v.replace(",", "").isdigit():
                out[f] = int(v.replace(",", ""))
        if isinstance(data.get("governing_law"), str) and data["governing_law"].strip():
            out["governing_law"] = data["governing_law"].strip().rstrip(".")
        return out


EXTRACTORS = {"regex-v1": extract_v1, "regex-v2": extract_v2}
