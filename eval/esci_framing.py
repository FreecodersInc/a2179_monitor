#!/usr/bin/env python3
"""EXP-021 H-27: binary vs ESCI centroid framing on real A2179 rows, FP at R 1.0, repeated stratified 5-fold."""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from a2179.similarity import char_ngrams  # noqa: E402

KIND = {"a2337": "S", "accessory": "C", "other_part": "C", "service": "C"}


def cls(r):
    return "E" if r["label"] == 1 else KIND.get(r.get("kind"), "I")


def vec(t, idf, d):
    c = Counter(char_ngrams(t))
    v = {g: (1 + math.log(n)) * idf.get(g, d) for g, n in c.items()}
    s = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {g: x / s for g, x in v.items()}


def centroid(vs):
    c = Counter()
    for v in vs:
        c.update(v)
    s = math.sqrt(sum(x * x for x in c.values())) or 1.0
    return {g: x / s for g, x in c.items()}


def cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(g, 0.0) for g, x in a.items())


def fit(train, texts):
    df = Counter()
    for i in train:
        df.update(set(char_ngrams(texts[i])))
    n = len(train)
    idf = {g: math.log((1 + n) / (1 + c)) + 1 for g, c in df.items()}
    return idf, math.log(1 + n) + 1


def run(rows, reps=10, k=5):
    texts = [f"{r['title']} {r.get('snippet', '')}" for r in rows]
    C = [cls(r) for r in rows]
    y = [c == "E" for c in C]
    out = []
    for rep in range(reps):
        rnd = random.Random(rep)
        pos = [i for i in range(len(rows)) if y[i]]
        neg = [i for i in range(len(rows)) if not y[i]]
        rnd.shuffle(pos), rnd.shuffle(neg)
        folds = [pos[f::k] + neg[f::k] for f in range(k)]
        res = {"bin": [0, 0, 0], "esci": [0, 0, 0]}  # fp, tp, n_test_pos
        for f in range(k):
            test = folds[f]
            train = [i for g in range(k) if g != f for i in folds[g]]
            idf, d = fit(train, texts)
            V = {i: vec(texts[i], idf, d) for i in range(len(rows))}
            cen = {c: centroid([V[i] for i in train if C[i] == c]) for c in "ESCI" if any(C[i] == c for i in train)}
            nonE = centroid([V[i] for i in train if C[i] != "E"])

            def sb(i):
                return cos(V[i], cen["E"]) - cos(V[i], nonE)

            def se(i):
                return cos(V[i], cen["E"]) - max(cos(V[i], cen[c]) for c in cen if c != "E")

            for name, fn in (("bin", sb), ("esci", se)):
                t = min(fn(i) for i in train if y[i])
                pred = [i for i in test if fn(i) >= t]
                res[name][0] += sum(not y[i] for i in pred)
                res[name][1] += sum(y[i] for i in pred)
                res[name][2] += sum(y[i] for i in test)
        out.append(res)
    return out, Counter(C)


def main():
    rows = [json.loads(l) for l in (ROOT / "eval" / "labeled.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("source") != "synthetic" and not str(r.get("source", "")).startswith("synth")]
    out, cc = run(rows)
    fb = [r["bin"][0] for r in out]
    fe = [r["esci"][0] for r in out]
    rb = [r["bin"][1] / r["bin"][2] for r in out]
    re_ = [r["esci"][1] / r["esci"][2] for r in out]
    diff = [b - e for b, e in zip(fb, fe)]
    rnd = random.Random(0)
    bs = sorted(sum(rnd.choice(diff) for _ in diff) / len(diff) for _ in range(1000))
    mb = sum(fb) / len(fb)
    res = {"rows": len(rows), "classes": dict(cc), "fp_bin_mean": round(mb, 2), "fp_esci_mean": round(sum(fe) / len(fe), 2),
           "fp_reduction": round((mb - sum(fe) / len(fe)) / mb, 3) if mb else None,
           "diff_ci95": [round(bs[25], 2), round(bs[974], 2)],
           "recall_bin_mean": round(sum(rb) / len(rb), 3), "recall_esci_mean": round(sum(re_) / len(re_), 3),
           "per_rep": out}
    o = ROOT / "reports" / "exp021"
    o.mkdir(parents=True, exist_ok=True)
    (o / "h27_esci_framing.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "per_rep"}))


if __name__ == "__main__":
    main()
