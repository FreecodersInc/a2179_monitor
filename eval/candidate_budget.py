#!/usr/bin/env python3
"""EXP-024 H-12: candidate recall at a fixed overall budget, no threshold tuning."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from a2179.similarity import TfidfSim  # noqa: E402


def collect(a, b, k):
    out = list(a[:math.ceil(k / 2)])
    for i in b:
        if i not in out:
            out.append(i)
        if len(out) >= k:
            break
    return out


def main():
    rows = [json.loads(l) for l in (ROOT / "eval" / "labeled.jsonl").read_text().splitlines()]
    rows = [r for r in rows if not str(r.get("source", "")).startswith("synth")]
    dense = {r["id"]: r["pos"] - r["neg"] for r in map(json.loads, (ROOT / "results" / "2026-11-20_a43706f" / "dense_scores.jsonl").read_text().splitlines())}
    sim = TfidfSim()
    lexical = {r["id"]: (lambda p, n: p - n)(*sim.score(f"{r['title']} {r.get('snippet', '')}")) for r in rows}
    a = sorted([r["id"] for r in rows], key=lambda i: -lexical[i])
    b = sorted(a, key=lambda i: -dense[i])
    pos = {r["id"] for r in rows if r["label"] == 1}
    k = 3 * len(pos)
    picks = {"tfidf": a[:k], "dense": b[:k], "union_tfidf_first": collect(a, b, k), "union_dense_first": collect(b, a, k),
             "union_double_budget": list(dict.fromkeys(a[:k] + b[:k]))}
    res = {"rows": len(rows), "npos": len(pos), "budget": k,
           "variants": {n: {"k": len(xs), "hits": len(pos.intersection(xs)), "recall": round(len(pos.intersection(xs)) / len(pos), 3),
                             "missed_ids": sorted(pos.difference(xs))} for n, xs in picks.items()}}
    out = ROOT / "reports" / "exp024"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
