"""Price parsing (Serbian/English formats) and currency conversion."""
import re
from dataclasses import dataclass
from typing import Optional

_CUR = [
    (re.compile(r"(rsd|din\.?|dinara|srdin|дин\.?|динара)", re.I), "RSD"),
    (re.compile(r"(€|eur|evra|евра|евро)", re.I), "EUR"),
    (re.compile(r"(\$|usd)", re.I), "USD"),
]
_NUM = re.compile(r"\d[\d.,\s\u00a0]*\d|\d")


@dataclass
class Price:
    amount: float
    currency: str

    def to_usd(self, rates: dict) -> Optional[float]:
        return to_usd(self.amount, self.currency, rates)


def parse_number(s: str) -> Optional[float]:
    s = s.replace("\u00a0", "").replace(" ", "")
    if not s:
        return None
    if "." in s and "," in s:
        # last separator is the decimal one
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        parts = s.split(sep)
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            s = s.replace(sep, "")  # thousands separator
        else:
            s = s.replace(sep, ".")
    try:
        return float(s)
    except ValueError:
        return None


def detect_currency(text: str, default: Optional[str] = None) -> Optional[str]:
    for rx, cur in _CUR:
        if rx.search(text or ""):
            return cur
    return default


def parse_price(text: str, default_currency: Optional[str] = None) -> Optional[Price]:
    """Parse the first price in text. Returns None for 'Dogovor', 'Kontakt', etc."""
    if not text:
        return None
    m = _NUM.search(text)
    if not m:
        return None
    amount = parse_number(m.group(0))
    cur = detect_currency(text, default_currency)
    if amount is None or cur is None or amount <= 0:
        return None
    return Price(amount, cur)


def to_usd(amount: float, currency: str, rates: dict) -> Optional[float]:
    cur = (currency or "").upper()
    if cur in ("SRDIN", "DIN"):
        cur = "RSD"
    usd_per_eur = float(rates.get("usd_per_eur", 1.16))
    rsd_per_eur = float(rates.get("rsd_per_eur", 117.2))
    if cur == "USD":
        return amount
    if cur == "EUR":
        return amount * usd_per_eur
    if cur == "RSD":
        return amount / rsd_per_eur * usd_per_eur
    return None
