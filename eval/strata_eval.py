#!/usr/bin/env python3
"""Matcher quality on a stratified de_jobs batch, reweighted back to the pool (EXP-005).

The capped sheet (eval/jobs_label.py sheet --max) keeps every row the matcher flagged and a seeded
uniform draw of the rest, and writes <sheet>_strata.json with pool and kept counts per stratum.
Here: precision from the matcher stratum, positives the matcher missed from the random stratum
scaled by pool / kept (Wilson 95% interval), recall = found / (found + missed). Two label views:
E+S (label 1, the ESCI rule) and E only (a strict DE reading). Label sensitivity: all review_unsure
rows counted positive, then all negative (the sample itself is never thinned).

Usage:
  python eval/strata_eval.py [--labeled eval/de_jobs/labeled.jsonl]
                             [--strata eval/de_jobs/to_label_strata.json] [--out reports/exp005]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


PATTERNS = [   # title families for the error table, first hit wins
    ("software engineer + data platform/infra", r"software engineer.*data (platform|infrastructure)|data (platform|infrastructure).*software engineer"),
    ("site reliability / devops", r"site reliability|\bsre\b|devops"),
    ("machine learning / ai engineer", r"machine learning|\bml\b|\bai engineer|applied ai|ai fde|forward deployed"),
    ("platform / infrastructure / cloud", r"platform|infrastructure|cloud"),
    ("solutions architect", r"solutions? architect"),
    ("data engineer / analytics engineer", r"data engineer|analytics engineer|data platform engineer"),
    ("data scientist / analyst", r"scien|analyst"),
    ("manager / director", r"manager|director|head of"),
]


def family(title: str) -> str:
    t = title.lower()
    for name, rx in PATTERNS:
        if re.search(rx, t):
            return name
    return "other"


def flip_unsure(rows, to: str):
    """Label sensitivity: every review_unsure row set to `to` (S = counted positive, I = negative).
    Keeps the sample intact, so the reweighting stays valid (dropping unsure rows would not:
    they are not a random subsample of their stratum)."""
    return [{**r, "esci": to} if r.get("review_unsure") else r for r in rows]


def view(rows, strata, pos):
    m = [r for r in rows if r["stratum"] == "matcher"]
    tp = sum(1 for r in m if r["esci"] in pos)
    out = {"matcher_rows": len(m), "tp": tp, "fp": len(m) - tp,
           "precision": round(tp / len(m), 3) if m else None, "precision_ci95": [round(x, 3) for x in wilson(tp, len(m))]}
    missed, lo_m, hi_m = 0.0, 0.0, 0.0
    for s in ("random", "hard_neg"):
        rs = [r for r in rows if r["stratum"] == s]
        st = strata.get(s, {"pool": 0, "kept": 0})
        if not rs:
            continue
        k = sum(1 for r in rs if r["esci"] in pos)
        lo, hi = wilson(k, len(rs))
        w = st["pool"]
        missed += w * k / len(rs)
        lo_m += w * lo
        hi_m += w * hi
        out[f"{s}_pos"] = f"{k}/{len(rs)} of a pool of {w}"
    pool_tp = tp * strata["matcher"]["pool"] / max(1, strata["matcher"]["kept"])
    out["missed_est"] = round(missed)
    out["missed_ci95"] = [round(lo_m), round(hi_m)]
    out["recall_est"] = round(pool_tp / (pool_tp + missed), 3) if pool_tp + missed else None
    out["recall_ci95"] = [round(pool_tp / (pool_tp + hi_m), 3), round(pool_tp / (pool_tp + lo_m), 3) if pool_tp + lo_m else 1.0]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default=str(ROOT / "eval" / "de_jobs" / "labeled.jsonl"))
    ap.add_argument("--strata", default=str(ROOT / "eval" / "de_jobs" / "to_label_strata.json"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "exp005"))
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in Path(a.labeled).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("stratum")]
    meta = json.loads(Path(a.strata).read_text(encoding="utf-8"))
    strata = meta["strata"]
    res = {"rows": len(rows), "labeler": dict(Counter(r.get("labeler", "human") for r in rows)),
           "unsure": sum(1 for r in rows if r.get("review_unsure")), "strata": strata, "seed": meta.get("seed"),
           "esci_by_stratum": {s: dict(Counter(r["esci"] for r in rows if r["stratum"] == s)) for s in strata},
           "E+S": view(rows, strata, {"E", "S"}), "E": view(rows, strata, {"E"}),
           "E+S_unsure_as_pos": view(flip_unsure(rows, "S"), strata, {"E", "S"}),
           "E+S_unsure_as_neg": view(flip_unsure(rows, "I"), strata, {"E", "S"})}
    fam = {}
    for r in rows:
        f = fam.setdefault(family(r["title"]), Counter())
        f[(r["stratum"] == "matcher", r["esci"] in ("E", "S"))] += 1
    res["families"] = {k: {"matcher_pos": v[(True, True)], "matcher_neg": v[(True, False)],
                           "missed_pos": v[(False, True)], "ok_neg": v[(False, False)]} for k, v in sorted(fam.items())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "strata_eval.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    L = [f"# EXP-005: matcher de_jobs на второй пачке ({res['rows']} строк, все labeler = bot, {res['unsure']} unsure)", "",
         "| вид меток | P (95%) | пропущено в пуле, оценка (95%) | R оценка (95%) |", "|---|---|---|---|"]
    for k in ("E+S", "E", "E+S_unsure_as_pos", "E+S_unsure_as_neg"):
        v = res[k]
        L.append(f"| {k} | {v['precision']} ({v['precision_ci95'][0]}–{v['precision_ci95'][1]}) | {v['missed_est']} "
                 f"({v['missed_ci95'][0]}–{v['missed_ci95'][1]}) | {v['recall_est']} ({v['recall_ci95'][0]}–{v['recall_ci95'][1]}) |")
    L += ["", "| семейство заголовков | matcher верно | matcher ложно | пропущено (в выборке) | верно отсеяно |", "|---|---|---|---|---|"]
    for k, v in res["families"].items():
        L.append(f"| {k} | {v['matcher_pos']} | {v['matcher_neg']} | {v['missed_pos']} | {v['ok_neg']} |")
    (out / "strata_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("E+S", "E")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
