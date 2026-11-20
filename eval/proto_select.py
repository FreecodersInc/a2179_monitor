"""EXP-013 / T-027 / H-39: choose the BM25 (or tfidf) prototype set inside the training folds.

Candidates: the full prototype set plus N random halves (eval/nested_cv.subsample, frac 0.5, seeds 0..N-1).
On every fold the candidate with the best training F1 (same pick() and operating point as nested_cv) is
carried with its threshold to the test fold. Baseline: the full set. Nothing is chosen on test rows.

  python eval/proto_select.py --target a2179_screen --repeats 5 --out reports/exp013
"""
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
import sweep as S  # noqa: E402
from a2179.normalize import normalize  # noqa: E402
from a2179.target import eval_data_path, load_target  # noqa: E402


def candidates(tg, n: int) -> dict:
    out = {"full": (list(tg.positive), list(tg.negative))}
    for s in range(n):
        out[f"half{s}"] = (N.subsample(tg.positive, 0.5, s), N.subsample(tg.negative, 0.5, s + 1000))
    return out


def channel_scores(kind: str, cands: dict, texts: list[str]) -> dict:
    cls = S.BM25 if kind == "bm25" else S.TfidfSim
    res = {}
    for name, (pos, neg) in cands.items():
        m = cls(pos, neg)
        res[name] = [p - q for p, q in (m.score(t) for t in texts)]
    return res


def select_run(y, regex, gate, feat, scores: dict, use_feat: bool, folds, min_recall: float, select: bool):
    """Pooled test predictions; select=False keeps the full set (baseline). Returns (pred, chosen names)."""
    pred, chosen = {}, []
    k = len(folds)
    for f in range(k):
        te = folds[f]
        tr = [i for g in range(k) if g != f for i in folds[g]]
        best = None
        for name, base in (scores.items() if select else [("full", scores["full"])]):
            ch = N.pick(y, regex, gate, base, feat, tr, use_feat, "sweep", min_recall)
            if ch is None:
                continue
            lam, t, lo, hi = ch
            p = N.predict(regex, gate, base, feat, tr, lam, t, lo, hi)
            f1 = S.prf([y[i] for i in tr], [p[i] for i in tr])["F1"]
            if best is None or f1 > best[0] or (f1 == best[0] and name == "full"):
                best = (f1, name, ch)
        if best is None:
            pred.update({i: True for i in te})
            chosen.append("none")
            continue
        _, name, ch = best
        chosen.append(name)
        pred.update(N.predict(regex, gate, scores[name], feat, te, *ch))
    return pred, chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="a2179_screen")
    ap.add_argument("--halves", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "reports" / "exp013"))
    a = ap.parse_args(argv)
    tg = load_target(a.target)
    min_recall = N.operating_point(tg)
    rows = [json.loads(l) for l in eval_data_path(tg).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r.setdefault("snippet", "")
    sw = S.Sweep(rows, target=tg)
    y = [bool(r["label"]) for r in rows]
    groups = [normalize(r["title"])[:60] for r in rows]
    texts = [f"{r['title']} {r['snippet']}".strip() for r in rows]
    frx = S.feature_rx(tg)
    feat = [(S.features(r["title"], r["snippet"], frx)[0] + 1) / 2 for r in rows]
    cands = candidates(tg, a.halves)
    res, picks = {}, {}
    for kind in ("bm25", "tfidf"):
        scores = channel_scores(kind, cands, texts)
        for use_feat in (False, True):
            ch = kind + ("+feat" if use_feat else "")
            for mode in ("full", "selected"):
                reps, names = [], Counter()
                for rep in range(a.repeats):
                    F = N.folds_grouped(y, groups, a.folds, random.Random(a.seed + rep))
                    pred, chosen = select_run(y, sw.regex, sw.gate, feat, scores, use_feat, F, min_recall, mode == "selected")
                    idx = sorted(pred)
                    reps.append(S.prf([y[i] for i in idx], [pred[i] for i in idx]))
                    names.update(chosen)
                res[f"{ch} {mode}"] = {m: {"mean": round(st.mean(r[m] for r in reps), 3), "sd": round(st.pstdev([r[m] for r in reps]), 3)}
                                       for m in ("P", "R", "F1", "fp", "fn")}
                if mode == "selected":
                    picks[ch] = dict(names.most_common())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"target": tg.name, "n": len(rows), "positives": sum(y), "halves": a.halves, "folds": a.folds,
            "repeats": a.repeats, "seed": a.seed, "min_recall": min_recall}
    (out / f"{tg.name}.json").write_text(json.dumps({"meta": meta, "summary": res, "chosen": picks}, indent=1) + "\n", encoding="utf-8")
    L = [f"# EXP-013, {tg.label}: {len(rows)} rows, {sum(y)} positives, {a.halves} halves + full set, "
         f"{a.folds} folds x {a.repeats} repeats, train recall >= {min_recall}", "",
         "| channel | prototypes | P | R | F1 | FP | FN |", "|---|---|---|---|---|---|---|"]
    for k_, v in res.items():
        ch, mode = k_.split(" ")
        L.append(f"| {ch} | {mode} | {v['P']['mean']:.3f} | {v['R']['mean']:.3f} | {v['F1']['mean']:.3f} ± {v['F1']['sd']:.3f} | "
                 f"{v['fp']['mean']:.1f} | {v['fn']['mean']:.1f} |")
    L += ["", "Chosen on the training folds (count over folds x repeats):", ""]
    L += [f"* {ch}: " + ", ".join(f"{n} {c}" for n, c in v.items()) for ch, v in picks.items()]
    (out / f"{tg.name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
