"""Regex filter, driven by a target profile (a2179/target.py, targets/<name>.json).

The default profile (targets/a2179_screen.json) holds the A2179 display / lid rules that used
to be written here; the module level names below are kept as views of that profile.
"""
from dataclasses import dataclass, field

from .normalize import normalize
from .target import default_target, load_target, terms_rx

_R = default_target().rules
STRONG_MODEL = _R.models[0].rx
A1932 = next((m.rx for m in _R.models if m.name == "a1932"), None)
A2337 = _R.conflict
MACBOOK_AIR = _R.family
PART, STRONG_PART, DONOR = _R.part, _R.strong_part, _R.donor
HARD_EXCL, SOFT_EXCL, LAPTOP_SPECS = _R.hard_excl, _R.soft_excl, _R.specs
_rx = terms_rx  # backwards compatible helper name


@dataclass
class RuleResult:
    match: bool
    model: str            # a tier name from the profile (strong | a1932 for A2179) | weak | none
    gate: bool            # listing is about a compatible machine (used by the similarity channel)
    reasons: list = field(default_factory=list)


def _accepted(tier, options: dict) -> bool:
    return not tier.accept_option or bool(options.get(tier.accept_option, True))


def classify(title: str, snippet: str = "", accept_a1932: bool = True, target=None, options: dict = None) -> RuleResult:
    """Positive signals (model, part, donor) are read from title + snippet.
    Exclusions are read from the title only, so a real panel listing whose
    description says 'ugradnja moguca' or 'servis' is not lost (recall first).
    target: a Target (default profile if None). options: accept_* switches of the model tiers,
    accept_a1932 is the old name of options['accept_a1932']."""
    R = load_target(target).rules if target is not None else _R
    opts = {"accept_a1932": accept_a1932, **(options or {})}
    tt = normalize(title)
    t = (tt + " " + normalize(snippet)).strip()
    r = []
    conflict = bool(R.conflict.search(t))
    fam = bool(R.family.search(t))
    weak = fam and any(all(p.search(t) for p in w.all) and not (w.unless_conflict and conflict) for w in R.weak)
    model = next((m.name for m in R.models if m.rx.search(t) and _accepted(m, opts)), None)
    if model is None:
        model = "weak" if weak else "none"
    r.append(f"model={model}")
    gate = model != "none" or (fam and not conflict)
    if model == "none":
        return RuleResult(False, model, gate, r + [R.no_model_reason])

    hard = R.hard_excl.search(tt)
    if hard:
        return RuleResult(False, model, gate, r + [f"hard_excl={hard.group(0).strip()}"])
    donor = R.donor.search(t)
    part = R.part.search(t)
    strong_part = R.strong_part.search(tt) or R.strong_part.search(t)
    soft = R.soft_excl.search(tt)
    specs = R.specs.search(tt)
    if strong_part and not (soft and not R.part.search(tt)):
        return RuleResult(True, model, gate, r + [f"strong_part={strong_part.group(0).strip()}"])
    if donor:
        return RuleResult(True, model, gate, r + [f"donor={donor.group(0).strip()}"])
    if part and not soft and not specs:
        return RuleResult(True, model, gate, r + [f"part={part.group(0).strip()}"])
    why = "no part term" if not part else (f"soft_excl={soft.group(0)}" if soft else f"laptop_specs={specs.group(0)}")
    return RuleResult(False, model, gate, r + [why])
