"""Text normalisation: lowercase, Cyrillic to Latin, strip diacritics."""
import re
import unicodedata

# Serbian + Russian Cyrillic. Mapped so that Cyrillic text lands on the same
# string as diacritic-stripped Serbian Latin (š -> s, č/ć -> c, ž -> z).
_CYR = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "dj", "е": "e", "ё": "e",
    "ж": "z", "з": "z", "и": "i", "й": "j", "ј": "j", "к": "k", "л": "l", "љ": "lj",
    "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "ћ": "c", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "c", "џ": "dz", "ш": "s",
    "щ": "sc", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "ju", "я": "ja",
    "і": "i", "ї": "ji", "є": "je", "ґ": "g",
}
_LAT = {"đ": "dj", "ß": "ss", "ø": "o", "ł": "l"}
_PUNCT = re.compile(r"[\u2010-\u2015\u2212_|/\\,;:!?()\[\]{}\"'«»„“”‘’`*+=<>~]+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = "".join(_CYR.get(ch, _LAT.get(ch, ch)) for ch in t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.replace("″", '"').replace("”", '"')
    t = _PUNCT.sub(" ", t)
    # "a 2179", "a-2179", "a.2179" -> "a2179"; "emc 3302" -> "emc3302"
    t = re.sub(r"\b(a|emc)[\s.\-]{0,2}(\d{4})\b", r"\1\2", t)
    return _WS.sub(" ", t).strip()
