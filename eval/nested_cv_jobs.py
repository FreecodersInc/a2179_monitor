#!/usr/bin/env python3
"""EXP-023 / H-44: nested CV for cv_score (H-35) on the 462 vacancies with full-text labels.

Inside each outer training fold: part weights on a 0.1 simplex grid, the domain scale (train positive
rate per rules class, smoothed) and the stack saturation, picked by train AUC. The hand rules, the
required-skills list and the stack list stay fixed (written with these rows in view), so a separate
check fits on all 462 and scores the fresh batch3 (EXP-011). Strata check: AUC inside match / rest.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import cv_distance as CV  # noqa: E402
from a2179.rules import classify  # noqa: E402
from a2179.target import load_target  # noqa: E402

POS = {"E", "S"}
PARTS = ("skills", "stack", "domain", "seniority")
SATS = (5, 10, 15, 20)


def fast_auc(s, y):
    """Mann-Whitney AUC with ties, O(n log n)."""
    o = sorted(range(len(s)), key=lambda i: s[i])
    r = [0.0] * len(s)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and s[o[j + 1]] == s[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    npos = sum(y)
    nneg = len(y) - npos
    if not npos or not nneg:
        return float("nan")
    return (sum(r[i] for i in range(len(y)) if y[i]) - npos * (npos + 1) / 2) / (npos * nneg)


def raw(rows, texts, tg, stack):
    out = []
    for r in rows:
        title, text = r["title"], texts.get(r["url"], "")
        full = f"{title} {text}"
        out.append({"skills": sum(1 for rx in CV.REQUIRED if re.search(rx, full, re.I)) / len(CV.REQUIRED),
                    "nstack": sum(1 for rx in stack if rx.search(full)),
                    "cls": classify(title, "", target=tg).model,
                    "match": bool(classify(title, "", target=tg).match),
                    "seniority": CV.seniority(title)})
    return out


GRID = [w for w in itertools.product(range(11), repeat=4) if sum(w) == 10]


def score(F, w, dom, sat):
    return [w[0] * f["skills"] + w[1] * min(1.0, f["nstack"] / sat) + w[2] * dom.get(f["cls"], 0.0)
            + w[3] * f["seniority"] for f in F]


def fit(F, y):
    cls = {f["cls"] for f in F}
    dom = {c: (sum(y[i] for i, f in enumerate(F) if f["cls"] == c) + 0.5)
           / (sum(1 for f in F if f["cls"] == c) + 1) for c in cls}
    best = (-1, None)
    for sat in SATS:
        for w in GRID:
            a = fast_auc(score(F, w, dom, sat), y)
            if a > best[0]:
                best = (a, (w, dom, sat))
    return best[1], best[0]


def hand(F):
    return [CV.W["skills"] * f["skills"] + CV.W["stack"] * min(1.0, f["nstack"] / CV.STACK_SAT)
            + CV.W["domain"] * CV.DOMAIN.get(f["cls"], 0.0) + CV.W["seniority"] * f["seniority"] for f in F]


def boot_auc(s, y, reps=1000, seed=0):
    rnd, v = random.Random(seed), []
    n = len(s)
    for _ in range(reps):
        ix = [rnd.randrange(n) for _ in range(n)]
        a = fast_auc([s[i] for i in ix], [y[i] for i in ix])
        if a == a:
            v.append(a)
    v.sort()
    return [round(v[int(0.025 * len(v))], 3), round(v[int(0.975 * len(v)) - 1], 3)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache" / "jobs"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "exp023"))
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args(argv)
    tg = load_target("de_jobs")
    rd = lambda p: [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]  # noqa: E731
    rows = [r for r in rd(ROOT / "eval" / "de_jobs" / "labeled.jsonl") if r.get("esci_text")]
    fresh = [r for r in rd(ROOT / "eval" / "de_jobs" / "batch3.jsonl") if r.get("esci_text")]
    texts = CV.job_texts(Path(a.cache))
    stack = CV.stack_terms(json.loads((ROOT / "targets" / "de_jobs.json").read_text(encoding="utf-8")).get("skills", {}))
    F, Ff = raw(rows, texts, tg, stack), raw(fresh, texts, tg, stack)
    y = [r["esci_text"] in POS for r in rows]
    yf = [r["esci_text"] in POS for r in fresh]
    res = {"rows": len(rows), "fresh": len(fresh), "fresh_with_text": sum(bool(texts.get(r["url"])) for r in fresh),
           "pos": sum(y), "fresh_pos": sum(yf)}
    nested, handed, picks = [], [], []
    for rep in range(a.repeats):
        rnd = random.Random(rep)
        pos = [i for i in range(len(rows)) if y[i]]
        neg = [i for i in range(len(rows)) if not y[i]]
        rnd.shuffle(pos), rnd.shuffle(neg)
        folds = [pos[f::5] + neg[f::5] for f in range(5)]
        sc, sh = [0.0] * len(rows), hand(F)
        for f in range(5):
            tr = [i for g in range(5) if g != f for i in folds[g]]
            (w, dom, sat), tra = fit([F[i] for i in tr], [y[i] for i in tr])
            picks.append({"w": w, "sat": sat, "train_auc": round(tra, 3)})
            s = score([F[i] for i in folds[f]], w, dom, sat)
            for i, v in zip(folds[f], s):
                sc[i] = v
        nested.append(fast_auc(sc, y))
        handed.append(fast_auc(sh, y))
    res["nested_auc_mean"] = round(sum(nested) / len(nested), 3)
    res["nested_auc_range"] = [round(min(nested), 3), round(max(nested), 3)]
    res["hand_auc"] = round(handed[0], 3)
    res["train_auc_mean"] = round(sum(p["train_auc"] for p in picks) / len(picks), 3)
    (w, dom, sat), full_auc = fit(F, y)
    res["fit_all"] = {"w": w, "sat": sat, "train_auc": round(full_auc, 3), "domain": {k: round(v, 3) for k, v in dom.items()}}
    sf = score(Ff, w, dom, sat)
    res["fresh_auc_fitted"] = round(fast_auc(sf, yf), 3)
    res["fresh_auc_fitted_ci95"] = boot_auc(sf, yf)
    res["fresh_auc_hand"] = round(fast_auc(hand(Ff), yf), 3)
    strata = {}
    for name, FF, yy, s in (("462", F, y, hand(F)), ("fresh", Ff, yf, hand(Ff))):
        for st in (True, False):
            ix = [i for i, f in enumerate(FF) if f["match"] == st]
            strata[f"{name}_{'match' if st else 'rest'}"] = {
                "n": len(ix), "pos": sum(yy[i] for i in ix),
                "auc_hand": round(fast_auc([s[i] for i in ix], [yy[i] for i in ix]), 3)}
    res["strata"] = strata
    # post hoc (not preregistered): positive rate per rules class in both sets
    pc = {}
    for name, FF, yy in (("462", F, y), ("fresh", Ff, yf)):
        for c in sorted({f["cls"] for f in FF}):
            ix = [i for i, f in enumerate(FF) if f["cls"] == c]
            pc.setdefault(c, {})[name] = f"{sum(yy[i] for i in ix)}/{len(ix)}"
    res["post_hoc_pos_rate_by_class"] = pc
    ph = {}
    for name, FF, yy, s in (("462", F, y, hand(F)), ("fresh", Ff, yf, hand(Ff))):
        for c in ("ml_sre_infra", "devops", "none"):
            ix = [i for i, f in enumerate(FF) if f["cls"] == c]
            ph[f"{name}_{c}"] = round(fast_auc([s[i] for i in ix], [yy[i] for i in ix]), 3)
        ix = [i for i, f in enumerate(FF) if f["cls"] != "strong"]
        ph[f"{name}_without_strong"] = {"n": len(ix), "auc_hand": round(fast_auc([s[i] for i in ix], [yy[i] for i in ix]), 3)}
    res["post_hoc_auc_within_class"] = ph
    res["picks_head"] = picks[:5]
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "nested_cv_jobs.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "picks_head"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
