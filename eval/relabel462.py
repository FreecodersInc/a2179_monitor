#!/usr/bin/env python3
"""EXP-020: H-35 and H-37 on all 462 old vacancies with full-text labels (esci_text)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import cv_distance as CV  # noqa: E402
from a2179.target import load_target  # noqa: E402

POS = {"E", "S"}
GRADE = {"E": 3, "S": 2, "C": 1, "I": 0}


def boot(fn, n, reps=1000, seed=0):
    rnd, v = random.Random(seed), []
    for _ in range(reps):
        idx = [rnd.randrange(n) for _ in range(n)]
        try:
            v.append(fn(idx))
        except (ZeroDivisionError, ValueError):
            pass
    v.sort()
    return [round(v[int(0.025 * len(v))], 3), round(v[int(0.975 * len(v)) - 1], 3)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default=str(ROOT / "eval" / "de_jobs" / "labeled.jsonl"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache" / "jobs"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "exp020"))
    a = ap.parse_args(argv)
    tg = load_target("de_jobs")
    rows = [json.loads(l) for l in Path(a.labeled).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("esci_text")]
    texts = CV.job_texts(Path(a.cache))
    prof = CV.stack_terms(json.loads((ROOT / "targets" / "de_jobs.json").read_text(encoding="utf-8")).get("skills", {}))
    cvst = CV.stack_terms({"plus": CV.cv_stack_terms((ROOT / "cv" / "master_cv.md").read_text(encoding="utf-8"))})
    have = [bool(texts.get(r["url"], "")) for r in rows]
    res = {"rows": len(rows), "with_text": sum(have)}
    ch = Counter((r["esci"], r["esci_text"]) for r in rows)
    flips = sum((r["esci"] in POS) != (r["esci_text"] in POS) for r in rows)
    grade = sum(r["esci"] != r["esci_text"] for r in rows)
    res["label_change"] = {"any_grade": grade, "rate": round(grade / len(rows), 3), "binary_flip": flips,
                           "binary_rate": round(flips / len(rows), 3),
                           "rate_ci95": boot(lambda ix: sum(rows[i]["esci"] != rows[i]["esci_text"] for i in ix) / len(ix), len(rows)),
                           "matrix": {f"{a}->{b}": n for (a, b), n in sorted(ch.items())},
                           "text_counts": dict(Counter(r["esci_text"] for r in rows)), "title_counts": dict(Counter(r["esci"] for r in rows))}
    for name, st in (("A_profile", prof), ("B_cv", cvst)):
        P = [CV.parts(r["title"], texts.get(r["url"], ""), tg, st) for r in rows]
        sc = [p["cv_score"] for p in P]
        out = {}
        for f in ("esci", "esci_text"):
            y = [r[f] in POS for r in rows]
            g = [GRADE[r[f]] for r in rows]
            out[f] = {"auc": round(CV.auc(sc, y), 3),
                      "auc_ci95": boot(lambda ix: CV.auc([sc[i] for i in ix], [y[i] for i in ix]), len(rows)),
                      "spearman": round(CV.spearman(sc, g), 3),
                      "spearman_ci95": boot(lambda ix: CV.spearman([sc[i] for i in ix], [g[i] for i in ix]), len(rows)),
                      "parts_auc": {k: round(CV.auc([p[k] for p in P], y), 3) for k in CV.W}}
        d = [CV.auc(sc, [r["esci_text"] in POS for r in rows]) - CV.auc(sc, [r["esci"] in POS for r in rows])]
        out["delta_auc_text_minus_title"] = round(d[0], 3)
        out["delta_ci95"] = boot(lambda ix: CV.auc([sc[i] for i in ix], [rows[i]["esci_text"] in POS for i in ix])
                                 - CV.auc([sc[i] for i in ix], [rows[i]["esci"] in POS for i in ix]), len(rows))
        res[name] = out
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "relabel462.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
