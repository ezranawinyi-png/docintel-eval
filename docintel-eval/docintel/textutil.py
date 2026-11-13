"""Small helpers: number words, date and money normalisation."""
from __future__ import annotations

import re
from datetime import date

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
         80: "eighty", 90: "ninety"}
_WORD2NUM = {w: i for i, w in enumerate(_ONES)}
_WORD2NUM.update({w: n for n, w in _TENS.items()})

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
_MONTH2NUM = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}


def num_to_words(n: int) -> str:
    """0-99 to words, e.g. 45 -> 'forty-five'."""
    if n < 20:
        return _ONES[n]
    tens, ones = (n // 10) * 10, n % 10
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"


def words_to_num(text: str) -> int | None:
    """'forty-five' -> 45, 'sixty' -> 60. Returns None if not a number word."""
    parts = re.split(r"[\s-]+", text.strip().lower())
    total, seen = 0, False
    for p in parts:
        if p in _WORD2NUM:
            total += _WORD2NUM[p]
            seen = True
        else:
            return None
    return total if seen else None


NUM_WORD_RE = r"(?:%s)(?:[\s-](?:%s))?" % (
    "|".join(sorted(list(_TENS.values()) + _ONES[1:], key=len, reverse=True)),
    "|".join(_ONES[1:10]),
)


def parse_date(text: str) -> str | None:
    """Parse 'March 3, 2024' / '3 March 2024' / '2024-03-03' to ISO format."""
    t = text.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return t
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", t)
    if m and m.group(1).lower() in _MONTH2NUM:
        return date(int(m.group(3)), _MONTH2NUM[m.group(1).lower()], int(m.group(2))).isoformat()
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", t)
    if m and m.group(2).lower() in _MONTH2NUM:
        return date(int(m.group(3)), _MONTH2NUM[m.group(2).lower()], int(m.group(1))).isoformat()
    return None


def parse_money(text: str) -> int | None:
    """'$250,000' / 'USD 250,000' / '250,000 US dollars' -> 250000."""
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)", text)
    return int(m.group(1).replace(",", "")) if m else None
