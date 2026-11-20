#!/usr/bin/env python3
"""EXP-025 H-03: the recall-first TF-IDF threshold of one target applied to the other."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from a2179.similarity import TfidfSim  # noqa: E402
from a2179.target import load_target  # noqa: E402


def scores(rows, tg):
    sim = TfidfSim(tg.positive, tg.negative)
    return [(lambda p, n: p - n)(*sim.score(f"{r['title']} {r.get('snippet', '')}")) for r in rows]


def pr(s, y, t):
    pred = [v >= t for v in s]
    tp = sum(p and l for p, l in zip(pred, y))
    return {"t": round(t, 4), "R": round(tp / sum(y), 3), "P": round(tp / max(1, sum(pred)), 3), "selected": sum(pred)}


def main():
    rd = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]  # noqa: E731
    a = [r for r in rd(ROOT / "eval" / "labeled.jsonl") if not str(r.get("source", "")).startswith("synth")]
    j = [r for r in rd(ROOT / "eval" / "de_jobs" / "labeled.jsonl") if r.get("esci_text")]
    ya = [r["label"] == 1 for r in a]
    yj = [r["esci_text"] in ("E", "S") for r in j]
    sa, sj = scores(a, load_target("a2179_screen")), scores(j, load_target("de_jobs"))
    ta = min(v for v, l in zip(sa, ya) if l)
    tj = min(v for v, l in zip(sj, yj) if l)
    res = {"a2179": {"rows": len(a), "pos": sum(ya), "own": pr(sa, ya, ta), "with_jobs_t": pr(sa, ya, tj)},
           "jobs": {"rows": len(j), "pos": sum(yj), "own": pr(sj, yj, tj), "with_a2179_t": pr(sj, yj, ta)}}
    res["recall_loss_jobs_pp"] = round(100 * (res["jobs"]["own"]["R"] - res["jobs"]["with_a2179_t"]["R"]), 1)
    res["recall_loss_a2179_pp"] = round(100 * (res["a2179"]["own"]["R"] - res["a2179"]["with_jobs_t"]["R"]), 1)
    out = ROOT / "reports" / "exp025"
    out.mkdir(parents=True, exist_ok=True)
    (out / "threshold_transfer.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
