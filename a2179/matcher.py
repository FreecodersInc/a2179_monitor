"""Hybrid filter: regex OR (gate AND similarity). Recall first."""
from dataclasses import dataclass

from .rules import classify

# Tuned by eval/run_eval.py for recall 1.0 on eval/labeled.jsonl (see reports/).
DEFAULT_THRESHOLDS = {
    "tfidf": {"pos": 0.38, "margin": 0.18, "calibrated": True},
    "embed": {"pos": 0.60, "margin": 0.03, "calibrated": False},
}


@dataclass
class Decision:
    match: bool
    regex: bool
    sim: bool
    sim_pos: float
    sim_neg: float
    model: str
    reasons: list


class Matcher:
    def __init__(self, sim=None, mode: str = "regex", thresholds: dict = None, accept_a1932: bool = True):
        self.sim = sim
        self.mode = mode if sim is not None else "regex"
        th = dict(DEFAULT_THRESHOLDS)
        th.update(thresholds or {})
        self.th = th.get(self.mode, {"pos": 1.1, "margin": 1.1})
        self.accept_a1932 = accept_a1932

    def decide(self, title: str, snippet: str = "", scores=None) -> Decision:
        rr = classify(title, snippet, self.accept_a1932)
        sp = sn = 0.0
        sim_hit = False
        if self.sim is not None:
            sp, sn = scores if scores is not None else self.sim.score(f"{title} {snippet}".strip())
            sim_hit = rr.gate and sp >= self.th["pos"] and (sp - sn) >= self.th["margin"]
        reasons = list(rr.reasons)
        if sim_hit and not rr.match:
            reasons.append(f"{self.mode}: pos={sp:.2f} neg={sn:.2f}")
        return Decision(rr.match or sim_hit, rr.match, sim_hit, sp, sn, rr.model, reasons)
