#!/usr/bin/env python3
"""Clean held-out estimate for the hybrid channels of the sweep (EXP-004, follow up of EXP-003 / ADR-0020).

The sweep picks the threshold t and the fusion weight lam on all 430 rows and reports numbers on
those same rows. Here every choice is made on the training part only:
  * outer: K folds, stratified by label, grouped by near-duplicate title (normalised title), R repeats
  * min-max of every channel fitted on the training part, applied (clipped) to the test part
  * lam (grid 0.0..1.0) and t picked on the training part: recall 1.0 there, then best F1, ties: smallest lam (as the sweep), then stricter t
  * test part scored once with the frozen (lo, hi, lam, t)
Output: pooled P / R / F1 / FP / FN over the test folds, mean and sd over repeats, the same on the
real rows of the test folds only, and paired differences between channels on identical folds.
Dense scores come from a CI run (results/<base>/dense_scores*.jsonl), nothing is embedded here.

Usage: python eval/nested_cv.py --dense-dir results/2026-11-19_247d60c [--repeats 20] [--out reports/exp004]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import sweep as S  # noqa: E402
from a2179.normalize import normalize  # noqa: E402
from a2179.target import eval_data_path, load_target  # noqa: E402

LAMS = [round(x * 0.1, 1) for x in range(11)]


def load_dense(path: Path, rows):
    if not path.is_file():
        return None
    d = {j["id"]: j["pos"] - j["neg"] for j in map(json.loads, path.read_text().splitlines()) if j}
    return [d[r["id"]] for r in rows] if all(r["id"] in d for r in rows) else None


def folds_grouped(y, groups, k, rnd):
    """Stratified by label, a duplicate group never split across folds."""
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    gl = list(by.values())
    rnd.shuffle(gl)
    gl.sort(key=lambda g: -max(y[i] for i in g))   # groups with a positive first, dealt round robin
    pos = [g for g in gl if any(y[i] for i in g)]
    neg = [g for g in gl if not any(y[i] for i in g)]
    out = [[] for _ in range(k)]
    for j, g in enumerate(pos):
        out[j % k].extend(g)
    for j, g in enumerate(neg):
        out[(j + len(pos)) % k].extend(g)
    return out


def subsample(protos, frac: float, seed: int) -> list:
    """Seeded share of a prototype list, at least one, order kept (EXP-007)."""
    protos = list(protos)
    if frac >= 1.0:
        return protos
    k = max(1, round(len(protos) * frac))
    keep = set(random.Random(seed).sample(range(len(protos)), k))
    return [p for i, p in enumerate(protos) if i in keep]


def cap_train(tr, y, cap: int, seed: int) -> list:
    """Stratified random `cap` rows of the training part (EXP-008). Own RNG, so the folds stay the
    same as without a cap."""
    if cap >= len(tr):
        return tr
    rnd = random.Random(seed)
    pos = [i for i in tr if y[i]]
    neg = [i for i in tr if not y[i]]
    kp = max(1, round(cap * len(pos) / len(tr)))
    return sorted(rnd.sample(pos, min(kp, len(pos))) + rnd.sample(neg, min(cap - kp, len(neg))))


def fit_mm(v, idx):
    lo, hi = min(v[i] for i in idx), max(v[i] for i in idx)
    return lo, hi


def apply_mm(x, lo, hi):
    return min(1.0, max(0.0, (x - lo) / (hi - lo))) if hi > lo else 0.0


RRF_K = 60


def _rank(v, ref):
    """1 + number of reference values strictly above v (rank against the training part)."""
    return 1 + sum(r > v for r in ref)


def rrf(base, feat, tr, idx):
    """EXP-012 / H-14: reciprocal rank fusion of a channel and feat, ranks relative to the training part, no lam."""
    rb, rf = [base[i] for i in tr], [feat[i] for i in tr]
    return {i: 1 / (RRF_K + _rank(base[i], rb)) + 1 / (RRF_K + _rank(feat[i], rf)) for i in idx}


def pick_rrf(y, regex, gate, base, feat, tr, tie="sweep", min_recall=1.0):
    s = rrf(base, feat, tr, tr)
    best = None
    for t in sorted(set(s.values()), reverse=True):
        m = S.prf([y[i] for i in tr], [regex[i] or (gate[i] and s[i] >= t) for i in tr])
        if m["R"] < min_recall:
            continue
        key = (m["F1"], t)
        if best is None or key > best[0]:
            best = (key, t)
    return None if best is None else best[1]


def pick(y, regex, gate, base, feat, tr, use_feat, tie="sweep", min_recall=1.0):
    """(lam, t, lo, hi) chosen on tr only. base: raw channel or None (feat only).
    tie on F1: "sweep" = smallest lam, then strictest t (what eval/sweep.py does), "strict" = strictest t
    across all lam (larger lam wins, the feature channel dominates). EXP-004 reports both."""
    lo, hi = fit_mm(base, tr) if base is not None else (0.0, 1.0)
    best = None
    for lam in (LAMS if use_feat and base is not None else ([1.0] if base is None else [0.0])):
        s = {i: (1 - lam) * (apply_mm(base[i], lo, hi) if base is not None else 0.0) + lam * feat[i] for i in tr}
        for t in sorted({s[i] for i in tr}, reverse=True):
            p = [regex[i] or (gate[i] and s[i] >= t) for i in tr]
            m = S.prf([y[i] for i in tr], p)
            if m["R"] < min_recall:   # 1.0 = the monitor rule; lower values are an ablation (EXP-006)
                continue
            key = (m["F1"], -lam, t) if tie == "sweep" else (m["F1"], t, -lam)
            if best is None or key > best[0]:
                best = (key, lam, t)
    if best is None:
        return None
    return best[1], best[2], lo, hi


def predict(regex, gate, base, feat, te, lam, t, lo, hi):
    out = {}
    for i in te:
        s = (1 - lam) * (apply_mm(base[i], lo, hi) if base is not None else 0.0) + lam * feat[i]
        out[i] = bool(regex[i] or (gate[i] and s >= t))
    return out


def run(rows, channels, feat, regex, gate, k=5, repeats=20, seed=0, tie="sweep", min_recall=1.0, train_cap=0):
    y = [bool(r["label"]) for r in rows]
    real = [r.get("source") != "synthetic" for r in rows]
    groups = [normalize(r["title"])[:60] for r in rows]
    res = defaultdict(list)       # name -> per repeat pooled dict
    for rep in range(repeats):
        rnd = random.Random(seed + rep)
        F = folds_grouped(y, groups, k, rnd)
        for name, (base, use_feat) in channels.items():
            pred = {}
            for f in range(k):
                te = F[f]
                tr = [i for g in range(k) if g != f for i in F[g]]
                if train_cap:
                    tr = cap_train(tr, y, train_cap, (seed + rep) * 100 + f)
                if name == "regex":
                    pred.update({i: bool(regex[i]) for i in te})
                    continue
                if name.endswith(":rrf"):
                    t = pick_rrf(y, regex, gate, base, feat, tr, tie, min_recall)
                    if t is None:
                        pred.update({i: True for i in te})
                        continue
                    s_ = rrf(base, feat, tr, te)
                    pred.update({i: bool(regex[i] or (gate[i] and s_[i] >= t)) for i in te})
                    continue
                ch = pick(y, regex, gate, base, feat, tr, use_feat, tie, min_recall)
                if ch is None:
                    pred.update({i: True for i in te})
                    continue
                pred.update(predict(regex, gate, base, feat, te, *ch))
            idx = sorted(pred)
            allm = S.prf([y[i] for i in idx], [pred[i] for i in idx])
            ridx = [i for i in idx if real[i]]
            realm = S.prf([y[i] for i in ridx], [pred[i] for i in ridx])
            res[name].append({"all": allm, "real": realm})
    return res


def summarise(res):
    out = {}
    for name, reps in res.items():
        d = {}
        for part in ("all", "real"):
            for m in ("P", "R", "F1", "fp", "fn"):
                v = [r[part][m] for r in reps]
                d[f"{part}_{m}"] = {"mean": round(st.mean(v), 3), "sd": round(st.pstdev(v), 3),
                                    "min": min(v), "max": max(v)}
        out[name] = d
    return out


def paired(res, a, b, part="real", metric="fp"):
    diff = [x[part][metric] - y[part][metric] for x, y in zip(res[a], res[b])]
    return {"mean": round(st.mean(diff), 2), "sd": round(st.pstdev(diff), 2), "a_better": sum(d < 0 for d in diff),
            "tie": sum(d == 0 for d in diff), "b_better": sum(d > 0 for d in diff), "n": len(diff)}


def operating_point(tg) -> float:
    """T-026: the working point lives in the profile (eval.operating_point.min_recall), 1.0 if absent."""
    return float((tg.eval.get("operating_point") or {}).get("min_recall", 1.0))


def build_text(title: str, body: str, title_weight: int = 1) -> str:
    """H-16: title repeated title_weight times, then the body."""
    return " ".join([title] * max(1, title_weight) + ([body] if body else [])).strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="a2179_screen")
    ap.add_argument("--dense-dir", required=True, help="results/<base> with dense_scores*.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "reports" / "exp004"))
    ap.add_argument("--tie", choices=["sweep", "strict"], default="sweep")
    ap.add_argument("--proto-frac", type=float, default=1.0,
                    help="share of the profile prototypes (each side) used by tfidf and bm25 (EXP-007)")
    ap.add_argument("--proto-seed", type=int, default=0)
    ap.add_argument("--train-cap", type=int, default=0, help="choose t, lam, min-max on this many training rows (EXP-008)")
    ap.add_argument("--relabel-rx", default="", help="variant: rows whose title matches this regex count as positive (EXP-009 Data Science variant)")
    ap.add_argument("--min-recall", type=float, default=None,
                    help="recall the threshold must keep on the training part; default: profile eval.operating_point.min_recall, else 1.0")
    ap.add_argument("--rrf", action="store_true", help="add X+feat:rrf channels, reciprocal rank fusion (EXP-012, H-14)")
    ap.add_argument("--title-weight", type=int, default=1, help="repeat the title this many times in the tfidf/bm25 text (EXP-012, H-16)")
    ap.add_argument("--text-from-cache", action="store_true",
                    help="tfidf/bm25 text = title + first 2000 chars of the cached description (jobs, by url)")
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache" / "jobs"))
    ap.add_argument("--name", default="nested_cv", help="output file stem")
    a = ap.parse_args(argv)
    tg = load_target(a.target)
    if a.min_recall is None:
        a.min_recall = operating_point(tg)
    rows = [json.loads(l) for l in eval_data_path(tg).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r.setdefault("snippet", "")
    if a.relabel_rx:
        import re
        rx = re.compile(a.relabel_rx, re.I)
        for r in rows:
            if rx.search(r["title"]):
                r["label"] = 1
    sw = S.Sweep(rows, target=tg)
    body = [r["snippet"] for r in rows]
    n_text = 0
    if a.text_from_cache:
        from a2179.jobs import collect_jobs
        jobs, _ = collect_jobs(Path(a.cache), sources=("remotive", "remoteok", "ats"), fetch="never")
        by = {j.url: (j.text or j.snippet) for j in jobs}
        body = [(by.get(r.get("url", ""), "") or r["snippet"])[:2000] for r in rows]
        n_text = sum(bool(by.get(r.get("url", ""))) for r in rows)
    texts = [build_text(r["title"], b, a.title_weight) for r, b in zip(rows, body)]
    pos_p, neg_p = subsample(tg.positive, a.proto_frac, a.proto_seed), subsample(tg.negative, a.proto_frac, a.proto_seed + 1000)
    tf, bm = S.TfidfSim(pos_p, neg_p), S.BM25(pos_p, neg_p)
    frx = S.feature_rx(tg)
    feat = [(S.features(r["title"], r["snippet"], frx)[0] + 1) / 2 for r in rows]
    raw = {"tfidf": [p - n for p, n in (tf.score(t) for t in texts)],
           "bm25": [p - n for p, n in (bm.score(t) for t in texts)]}
    dd = Path(a.dense_dir)
    for p in sorted(dd.glob("dense_scores*.jsonl")):
        short = p.stem[len("dense_scores"):].lstrip("_") or "minilm"
        v = load_dense(p, rows)
        if v is not None:
            raw[f"dense:{short}"] = v
    channels = {"regex": (None, False), "feat only": (None, True)}
    for k_, v in raw.items():
        channels[k_] = (v, False)
        channels[f"{k_}+feat"] = (v, True)
        if a.rrf:
            channels[f"{k_}+feat:rrf"] = (v, True)
    res = run(rows, channels, feat, sw.regex, sw.gate, a.folds, a.repeats, a.seed, a.tie, a.min_recall, a.train_cap)
    summ = summarise(res)
    pairs = {}
    for b in [c for c in channels if c.endswith("+feat") and c != "tfidf+feat"]:
        pairs[f"tfidf+feat vs {b}"] = {"real_fp": paired(res, "tfidf+feat", b), "all_F1": paired(res, "tfidf+feat", b, "all", "F1")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"target": tg.name, "n": len(rows), "positives": sum(sw.y), "real_rows": sum(r.get("source") != "synthetic" for r in rows),
            "real_positives": sum(1 for r in rows if r.get("source") != "synthetic" and r["label"]),
            "folds": a.folds, "repeats": a.repeats, "seed": a.seed, "tie": a.tie, "min_recall": a.min_recall, "proto_frac": a.proto_frac, "train_cap": a.train_cap, "relabel_rx": a.relabel_rx, "proto_seed": a.proto_seed,
            "rrf": a.rrf, "title_weight": a.title_weight, "text_from_cache": a.text_from_cache, "rows_with_cached_text": n_text,
            "prototypes": [len(pos_p), len(neg_p)], "dense_dir": str(dd)}
    (out / f"{a.name}.json").write_text(json.dumps({"meta": meta, "summary": summ, "paired": pairs}, indent=1), encoding="utf-8")
    lines = [f"# Nested CV, {tg.label}: {meta['n']} rows, {meta['positives']} positives "
             f"({meta['real_positives']} of them real), {a.folds} folds x {a.repeats} repeats, seed {a.seed}, F1 ties: {a.tie}, train recall >= {a.min_recall}", "",
             "Hybrid mode (regex OR gate and score >= t). t, lam and min-max chosen on the training folds only. "
             "mean ± sd over repeats of the pooled test folds.", "",
             "| channel | P | R | F1 | FP | FN | real P | real R | real FP | real FN |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(summ, key=lambda n: -summ[n]["all_F1"]["mean"])
    for n in order:
        s = summ[n]
        f = lambda k: f"{s[k]['mean']:.3f} ± {s[k]['sd']:.3f}"  # noqa: E731
        g = lambda k: f"{s[k]['mean']:.1f} ± {s[k]['sd']:.1f}"  # noqa: E731
        lines.append(f"| {n} | {f('all_P')} | {f('all_R')} | {f('all_F1')} | {g('all_fp')} | {g('all_fn')} | "
                     f"{f('real_P')} | {f('real_R')} | {g('real_fp')} | {g('real_fn')} |")
    lines += ["", "Paired on identical folds (difference tfidf+feat minus other, negative FP = tfidf+feat better):", "",
              "| pair | real FP diff mean ± sd | tfidf+feat better / tie / other better | F1 diff mean ± sd |", "|---|---|---|---|"]
    for k_, v in pairs.items():
        r_, f_ = v["real_fp"], v["all_F1"]
        lines.append(f"| {k_} | {r_['mean']} ± {r_['sd']} | {r_['a_better']} / {r_['tie']} / {r_['b_better']} | "
                     f"{f_['mean']} ± {f_['sd']} |")
    (out / f"{a.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
