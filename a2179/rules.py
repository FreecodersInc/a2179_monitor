"""Regex rules for A2179 display / lid listings.

Terms are written in natural script (Latin, Cyrillic, English) and run through
normalize() at import time, so one rule covers 'ekran', 'екран', 'экран'.
"""
import re
from dataclasses import dataclass, field

from .normalize import normalize


def _alt(terms):
    return "|".join(sorted({re.escape(normalize(t)) for t in terms}, key=len, reverse=True))


def _rx(terms, prefix=r"\b", suffix=""):
    return re.compile(prefix + "(?:" + _alt(terms) + ")" + suffix)


# Model identifiers ----------------------------------------------------------
STRONG_MODEL = re.compile(r"\ba2179\b|\bemc3302\b|\b661[\s\-]?15389\b|\b661[\s\-]?15390\b")
A1932 = re.compile(r"\ba1932\b|\bemc3184\b|\bemc3331\b")
A2337 = re.compile(r"\ba2337\b|\bemc3598\b|\bm1\b|\bapple silicon\b")
MACBOOK_AIR = re.compile(r"\b(mac\s?book|macbook|makbuk|mekbuk|mak\s?buk|mek\s?buk)\s?(air|ejr|er|eir|ejr)\b|\bmba\b")
YEAR_2020 = re.compile(r"\b2020\b")
INTEL = re.compile(r"\bintel\b|\bcore\s?i[357]\b|\bi[357]\b")
RANGE_2018_2020 = re.compile(r"\b2018\s?-?\s?2020\b|\b2019\s?-?\s?2020\b")

# Part terms -----------------------------------------------------------------
PART = _rx([
    "ekran", "ekrana", "ekranom", "displej", "displeja", "display", "lcd", "panel", "matrica",
    "matricu", "screen", "lid", "assembly", "sklop", "retina display",
    "екран", "дисплеј", "дисплей", "экран", "матрица", "матрицу", "крышка", "крышку", "в сборе",
    "лцд", "панел", "панель", "kuciste", "kucistem", "kucista", "gornji deo", "gornji poklopac", "top lid", "komplet ekran",
])
# A part word that is the subject of the listing, not a description of a laptop.
STRONG_PART = re.compile(
    r"\b(ekran|displej|display|lcd|panel|matric[au]|screen|lid|ekrana)\b"
    r"(\s+\S+){0,2}?\s+(za|for|dl\s?ja|dlja|assembly|komplet|kompletu|sklop|v sbore|panel|ekran|lcd)\b"
    r"|^\s*(\d+[.,]?\d*\s*(\"|inca|incni|inch|in)?\s*)?(lcd|ekran|displej|display|panel|matric[au]|screen)\b"
    r"|\b(lcd|display|screen|ekran|displej)\s+(panel|assembly|komplet|sklop)\b"
    r"|\bkomplet\b.{0,30}\b(ekran|lcd|displej)\b|\b(ekran|lcd|displej)\b.{0,30}\bkomplet\b"
    r"|\bkrysk[au]\b|\bv sbore\b|\bgornji (deo|poklopac)\b|\btop lid\b"
)
DONOR = _rx([
    "u delove", "za delove", "na delove", "delovi", "dijelovi", "делови", "for parts", "parts only",
    "donor", "донор", "на запчасти", "запчасти", "на детали", "neispravan", "neispravna", "ne radi",
    "pokvaren", "pokvarena", "ne pali", "не работает", "не включается", "ne ukljucuje",
    "ispravan ekran", "ekran ispravan", "ekran radi", "zdrav ekran", "rabotaet ekran", "экран целый",
])
# Always a negative, even with a part word ("zastita za ekran", "lcd kabl", "zamena ekrana").
HARD_EXCL = re.compile(
    r"\b(zastita|zastitna|zastitno|zastitni|folija|folije|staklo|kaljeno|protector|protektor|plenka|"
    r"zascitnaja|zascitnoe|steklo|nalepnica|stiker|skin|lepak|traka|adhesive|backlight kabl)\b"
    r"|\b(flet|flex|flat|fleks)\s?(kabl|kabal|cable|kabel)?\b|\b(lcd|edp|lvds)\s+(kabl|kabal|cable|kabel|sajl)\b"
    r"|\b(kabl|kabal|cable|kabel)\s+(za\s+)?(lcd|ekran|edp|lvds|displej)\b"
    r"|\b(zamena|zamenu|popravka|popravak|servis|service|repair|ugradnja|zamina|zamena ekrana|"
    r"remont|zamena displeja)\b"
    r"|\b(trazim|kupujem|otkup)\b"
    r"|\b(futrola|futrolu|maska|torba|case|sleeve|cehol|cohol|cexol|navlaka)\b"
    r"|\bcover\b(?!.{0,15}\b(lcd|display|ekran)\b)"
)
# Other parts / accessories: excluded unless the listing is clearly a display or donor.
SOFT_EXCL = _rx([
    "tastatura", "tastaturu", "keyboard", "palmrest", "topcase", "top case", "baterija", "bateriju",
    "battery", "punjac", "charger", "adapter", "srafovi", "srafove", "screws", "konektor", "connector",
    "kuler", "fan", "ventilator", "antena", "antene", "trackpad", "touchpad", "tacpad", "zvucnik",
    "zvucnici", "speaker", "kamera", "camera", "maticna", "maticnu", "motherboard", "logic board",
    "ploca", "plocu", "ssd", "ram", "donji poklopac", "bottom case", "donje kuciste", "sarke",
    "hinge", "hinges", "клавиатура", "аккумулятор", "батарея", "зарядка", "зарядно", "кулер",
    "тастатура", "батерија", "пуњач",
])
# Whole-laptop descriptors.
LAPTOP_SPECS = re.compile(
    r"\b\d{1,4}\s?(gb|tb)\b|\bssd\b|\bram\b|\bciklus\w*|\bcycle\w*|\bcore\s?i[357]\b|\bi[357]\b|"
    r"\bstanje\b|\bkao nov\w*|\bkutij\w*|\bgarancij\w*\s+\d|\bmacos\b|\bodlicn\w*"
)


@dataclass
class RuleResult:
    match: bool
    model: str            # strong | a1932 | weak | none
    gate: bool            # listing is about an A2179-compatible machine (used by the similarity channel)
    reasons: list = field(default_factory=list)


def classify(title: str, snippet: str = "", accept_a1932: bool = True) -> RuleResult:
    """Positive signals (model, part, donor) are read from title + snippet.
    Exclusions are read from the title only, so a real panel listing whose
    description says 'ugradnja moguca' or 'servis' is not lost (recall first)."""
    tt = normalize(title)
    t = (tt + " " + normalize(snippet)).strip()
    r = []
    strong = bool(STRONG_MODEL.search(t))
    a1932 = bool(A1932.search(t))
    a2337 = bool(A2337.search(t))
    air = bool(MACBOOK_AIR.search(t))
    weak = air and bool(
        (YEAR_2020.search(t) and INTEL.search(t))
        or RANGE_2018_2020.search(t)
        or (YEAR_2020.search(t) and not a2337)
        or ("retina" in t and not a2337)
    )
    if strong:
        model = "strong"
    elif a1932 and accept_a1932:
        model = "a1932"
    elif weak:
        model = "weak"
    else:
        model = "none"
    r.append(f"model={model}")
    gate = model != "none" or (air and not a2337)
    if model == "none":
        return RuleResult(False, model, gate, r + ["no A2179-compatible model"])

    hard = HARD_EXCL.search(tt)
    if hard:
        return RuleResult(False, model, gate, r + [f"hard_excl={hard.group(0).strip()}"])
    donor = DONOR.search(t)
    part = PART.search(t)
    strong_part = STRONG_PART.search(tt) or STRONG_PART.search(t)
    soft = SOFT_EXCL.search(tt)
    specs = LAPTOP_SPECS.search(tt)
    if strong_part and not (soft and not PART.search(tt)):
        return RuleResult(True, model, gate, r + [f"strong_part={strong_part.group(0).strip()}"])
    if donor:
        return RuleResult(True, model, gate, r + [f"donor={donor.group(0).strip()}"])
    if part and not soft and not specs:
        return RuleResult(True, model, gate, r + [f"part={part.group(0).strip()}"])
    why = "no part term" if not part else (f"soft_excl={soft.group(0)}" if soft else f"laptop_specs={specs.group(0)}")
    return RuleResult(False, model, gate, r + [why])
