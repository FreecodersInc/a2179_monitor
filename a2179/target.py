"""Target profiles: everything that is specific to one search case lives in targets/<name>.json.

A profile has
  rules       the regex filter: model tiers, family + conflict (gate for the similarity channel),
              weak model evidence, part / donor / exclusion terms, laptop-spec pattern
  prototypes  positive / negative phrases for the similarity channel (TF-IDF, BM25, embedder)
  defaults    config values merged under config.json: sources, target price, thresholds, options
  label       short name for alerts
The default profile is a2179_screen (MacBook Air 2020 Intel display / lid / donor in Serbia).
Pick another with "target": "<name or path.json>" in config.json or A2179_TARGET.

Term lists are written in natural script and run through normalize() when compiled, so one
term covers 'ekran', 'екран', 'экран'. Regex strings apply to normalized text as they are.
"""
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import PROJECT_ROOT
from .normalize import normalize

TARGETS_DIR = PROJECT_ROOT / "targets"
DEFAULT_TARGET = "a2179_screen"
NEVER = re.compile(r"(?!x)x")  # matches nothing


class TargetError(ValueError):
    pass


def _alt(terms):
    # longest first (as the old rules._rx), ties alphabetical so the pattern is the same in every
    # process (the old set order depended on PYTHONHASHSEED; equal length literals cannot both
    # match at one position, so the order among them never changed a match)
    return "|".join(sorted({re.escape(normalize(t)) for t in terms}, key=lambda x: (-len(x), x)))


def terms_rx(terms, prefix=r"\b", suffix=""):
    """Same construction as the original rules._rx: longest alternative first."""
    if not terms:
        return NEVER
    return re.compile(prefix + "(?:" + _alt(terms) + ")" + suffix)


def _rx(p):
    if p in (None, ""):
        return NEVER
    try:
        return re.compile(p)
    except re.error as e:
        raise TargetError(f"bad regex {p!r}: {e}") from None


@dataclass
class ModelTier:
    name: str
    rx: re.Pattern
    accept_option: str = ""     # config key that can switch this tier off (e.g. accept_a1932)
    alert_note: str = ""        # appended to alerts for listings matched through this tier


@dataclass
class WeakRule:
    all: list                   # every pattern must be found
    unless_conflict: bool = False


@dataclass
class Rules:
    models: list
    family: re.Pattern
    conflict: re.Pattern
    weak: list
    part: re.Pattern
    strong_part: re.Pattern
    donor: re.Pattern
    hard_excl: re.Pattern
    soft_excl: re.Pattern
    specs: re.Pattern
    no_model_reason: str = "no compatible model"


@dataclass
class Target:
    name: str
    label: str
    description: str
    rules: Rules
    positive: list
    negative: list
    defaults: dict = field(default_factory=dict)
    source: str = ""
    eval: dict = field(default_factory=dict)   # optional: data, features {display, other}, kinds

    def tier(self, name):
        return next((m for m in self.rules.models if m.name == name), None)


def compile_rules(r: dict) -> Rules:
    for k in ("models", "family"):
        if k not in r:
            raise TargetError(f"rules.{k} is required")
    models = [ModelTier(m["name"], _rx(m["regex"]), m.get("accept_option", ""), m.get("alert_note", ""))
              for m in r["models"]]
    names = [m.name for m in models]
    if len(set(names)) != len(names) or {"weak", "none"} & set(names):
        raise TargetError(f"model tier names must be unique and not weak/none: {names}")
    return Rules(
        models=models,
        family=_rx(r["family"]),
        conflict=_rx(r.get("conflict")),
        weak=[WeakRule([_rx(p) for p in w["all"]], bool(w.get("unless_conflict"))) for w in r.get("weak", [])],
        part=terms_rx(r.get("part_terms", [])),
        strong_part=_rx(r.get("strong_part")),
        donor=terms_rx(r.get("donor_terms", [])),
        hard_excl=_rx(r.get("hard_excl")),
        soft_excl=terms_rx(r.get("soft_excl_terms", [])),
        specs=_rx(r.get("specs")),
        no_model_reason=r.get("no_model_reason", "no compatible model"),
    )


def from_dict(d: dict, source: str = "") -> Target:
    if not isinstance(d, dict) or "rules" not in d:
        raise TargetError(f"target {source or '?'}: needs a 'rules' section")
    pr = d.get("prototypes") or {}
    pos, neg = list(pr.get("positive") or []), list(pr.get("negative") or [])
    if not pos or not neg:
        raise TargetError(f"target {source or '?'}: prototypes.positive and prototypes.negative must not be empty")
    name = d.get("name") or Path(source).stem or "custom"
    return Target(name=name, label=d.get("label") or name, description=d.get("description", ""),
                  rules=compile_rules(d["rules"]), positive=pos, negative=neg,
                  defaults=dict(d.get("defaults") or {}), source=source, eval=dict(d.get("eval") or {}))


def resolve(spec) -> Path:
    s = str(spec or DEFAULT_TARGET)
    p = Path(s)
    if s.endswith(".json"):
        return p if p.is_absolute() else PROJECT_ROOT / p
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", s):
        raise TargetError(f"bad target name {s!r}")
    return TARGETS_DIR / f"{s}.json"


@lru_cache(maxsize=16)
def _load_path(path: str) -> Target:
    p = Path(path)
    if not p.is_file():
        raise TargetError(f"target profile not found: {p}")
    return from_dict(json.loads(p.read_text(encoding="utf-8")), str(p))


def load_target(spec=None) -> Target:
    """spec: profile name (targets/<name>.json), a path to .json, a dict, or a Target."""
    if isinstance(spec, Target):
        return spec
    if isinstance(spec, dict):
        return from_dict(spec, spec.get("name", "inline"))
    return _load_path(str(resolve(spec)))


def eval_data_path(target) -> Path:
    """Labeled set of a target (T-002): profile eval.data, else eval/<name>/labeled.jsonl.
    The A2179 set stays at eval/labeled.jsonl (its profile names it), so its numbers do not move."""
    t = load_target(target)
    rel = t.eval.get("data") or f"eval/{t.name}/labeled.jsonl"
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def default_target() -> Target:
    return load_target(DEFAULT_TARGET)


def target_of(cfg) -> Target:
    return load_target((cfg or {}).get("target") or DEFAULT_TARGET)


def list_targets():
    # *.local.json is gitignored personal data and is not a public target name
    return sorted(p.stem for p in TARGETS_DIR.glob("*.json") if not p.name.endswith(".local.json"))
