#!/usr/bin/env python3
"""EXP-021 / H-24 on the runner: dense channel (sim_pos - sim_neg to the profile prototypes) with the
title and the snippet embedded together (base) or apart: title only, max(title, snippet), mean.
P@R1 on all 430 rows and on the real rows, paired bootstrap 95% of the difference to the base.
The model must already be in the HF cache (local_only), nothing is downloaded here."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VARIANTS = ("base", "title", "max", "mean")


def p_at_r1(scores, y):
    npos = sum(y)
    if not npos:
        return float("nan")
    order = sorted(range(len(y)), key=lambda i: -scores[i])
    hits = 0
    for k, i in enumerate(order, 1):
        hits += y[i]
        if hits == npos:
            return npos / k
    return float("nan")


def variants(rows, score_many):
    """score_many(texts) -> [(sim_pos, sim_neg)]. Rows without a snippet use the title for it."""
    ts = [f"{r['title']} {r.get('snippet', '')}".strip() for r in rows]
    t = [r["title"] for r in rows]
    s = [(r.get("snippet") or "").strip() or r["title"] for r in rows]
    d = lambda xs: [p - n for p, n in score_many(xs)]  # noqa: E731
    dts, dt, ds = d(ts), d(t), d(s)
    return {"base": dts, "title": dt, "max": [max(a, b) for a, b in zip(dt, ds)],
            "mean": [(a + b) / 2 for a, b in zip(dt, ds)]}


def evaluate(rows, V, reps=1000, seed=0):
    out = {}
    for part, idx in (("all", list(range(len(rows)))),
                      ("real", [i for i, r in enumerate(rows) if not str(r.get("source", "")).startswith("synth")])):
        y = [int(rows[i]["label"] == 1) for i in idx]
        res = {"n": len(idx), "pos": sum(y), "with_snippet": sum(bool((rows[i].get("snippet") or "").strip()) for i in idx)}
        for v in VARIANTS:
            res[v] = round(p_at_r1([V[v][i] for i in idx], y), 3)
        rnd = random.Random(seed)
        diffs = {v: [] for v in VARIANTS[1:]}
        for _ in range(reps):
            b = [idx[rnd.randrange(len(idx))] for _ in idx]
            yb = [int(rows[i]["label"] == 1) for i in b]
            if not sum(yb):
                continue
            base = p_at_r1([V["base"][i] for i in b], yb)
            for v in diffs:
                diffs[v].append(p_at_r1([V[v][i] for i in b], yb) - base)
        for v, xs in diffs.items():
            xs = sorted(x for x in xs if not math.isnan(x))
            res[f"{v}_minus_base"] = round(res[v] - res["base"], 3)
            res[f"{v}_minus_base_ci95"] = [round(xs[int(0.025 * len(xs))], 3), round(xs[int(0.975 * len(xs)) - 1], 3)]
        out[part] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    from a2179.config import DEFAULTS
    from a2179.embedder import EmbedSim
    model = a.model or DEFAULTS["embedder"]["model"]
    rows = [json.loads(l) for l in (ROOT / "eval" / "labeled.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sim = EmbedSim(model, local_only=True, device=a.device)
    res = {"model": model, **evaluate(rows, variants(rows, sim.score_many))}
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "granularity.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
