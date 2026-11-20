#!/usr/bin/env python3
"""EXP-026 / T-010 / H-45: BM25 k1 and b chosen inside the training folds (protocol of EXP-013)."""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import nested_cv as N  # noqa: E402
import proto_select as PS  # noqa: E402
import sweep as S  # noqa: E402
from a2179.normalize import normalize  # noqa: E402
from a2179.target import eval_data_path, load_target  # noqa: E402

K1 = (0.5, 0.9, 1.2, 1.5, 2.0)
B = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT = (1.5, 0.75)


def grid_scores(tg, texts):
    out = {}
    for k1 in K1:
        for b in B:
            m = S.BM25(tg.positive, tg.negative, k1=k1, b=b)
            name = "full" if (k1, b) == DEFAULT else f"k1={k1},b={b}"
            out[name] = [p - q for p, q in (m.score(t) for t in texts)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="a2179_screen")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "reports" / "exp026"))
    a = ap.parse_args(argv)
    tg = load_target(a.target)
    min_recall = N.operating_point(tg)
    rows = [json.loads(l) for l in eval_data_path(tg).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if not str(r.get("source", "")).startswith("synth")]
    for r in rows:
        r.setdefault("snippet", "")
        if r.get("esci_text"):
            r["label"] = int(r["esci_text"] in {"E", "S"})
    sw = S.Sweep(rows, target=tg)
    y = [bool(r["label"]) for r in rows]
    groups = [normalize(r["title"])[:60] for r in rows]
    texts = [f"{r['title']} {r['snippet']}".strip() for r in rows]
    frx = S.feature_rx(tg)
    feat = [(S.features(r["title"], r["snippet"], frx)[0] + 1) / 2 for r in rows]
    scores = grid_scores(tg, texts)
    res, picks = {}, {}
    for use_feat in (False, True):
        ch = "bm25" + ("+feat" if use_feat else "")
        for mode in ("default", "selected"):
            reps, names = [], Counter()
            for rep in range(a.repeats):
                F = N.folds_grouped(y, groups, 5, random.Random(rep))
                pred, chosen = PS.select_run(y, sw.regex, sw.gate, feat, scores, use_feat, F, min_recall, mode == "selected")
                idx = sorted(pred)
                reps.append(S.prf([y[i] for i in idx], [pred[i] for i in idx]))
                names.update(chosen)
            res[f"{ch} {mode}"] = {m: {"mean": round(st.mean(r[m] for r in reps), 3), "sd": round(st.pstdev([r[m] for r in reps]), 3)}
                                   for m in ("P", "R", "F1", "fp", "fn")}
            if mode == "selected":
                picks[ch] = dict(names.most_common())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"meta": {"target": tg.name, "n": len(rows), "positives": sum(y), "repeats": a.repeats, "grid": {"k1": K1, "b": B},
                    "default": DEFAULT, "min_recall": min_recall}, "summary": res, "chosen": picks}
    (out / f"{tg.name}.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
