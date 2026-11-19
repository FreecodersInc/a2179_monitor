#!/usr/bin/env python3
"""Charts and ranking metrics for a sweep run, exploratory data analysis style.

Reads what eval/sweep.py wrote (the per row channel scores in sweep.json) plus the labelled set,
and writes PNGs, plots.json and plots.md into --out. Nothing is fitted here: every number is
recomputed from the same scores the sweep tables use, so a chart and its table always agree.

Principles (Tukey's exploratory data analysis, Tufte's small multiples, see the lecture
"Exploratory data analysis and visualization" in MIT OCW 11.220, Spring 2009):
  look at the raw distribution before any summary, split every distribution by label and by class,
  same axes across a row of panels so they compare at a glance, label every axis with what and unit,
  show the chosen operating point on the curve it was chosen from.

Charts
  01_dataset         rows per kind and per source, split by label, title length by label
Channels: tfidf, bm25, dense (if scored), feat, rrf(tfidf,bm25), rrf(dense,bm25) and the sweep's best
"<base>+feat lam=" fusion per base, all min-max normalised like the sweep.
  02_score_hist      score distribution per channel, positives vs negatives, chosen threshold
  03_pr_curves       precision vs recall per channel and fusion, pure and hybrid (regex OR score)
  04_threshold       P, R, F1 against the threshold, one panel per channel (hybrid)
  05_confusion       2x2 confusion matrix per variant at its tuned threshold
  06_tfidf_2d        F1 and FP over the monitor's own 2D grid (pos x margin), recall < 1 marked
  07_fusion_weights  F1 and FP against the fusion weight w and the feature weight lam
  08_recall_budget   recall@k and precision@k when you read the top k listings of a ranking
  09_errors_by_kind  false positives and false negatives per listing kind for each tuned variant

Ranking metrics (plots.json "ranking", table in plots.md), per channel, pure and hybrid order:
  R@npos, R@2npos   recall when you read as many (twice as many) listings as there are positives
  P@R1, P@R95       precision at the depth where recall first reaches 1.0 (0.95)
  AP, nDCG@npos     average precision, binary nDCG cut at the number of positives
  depth@R1          how many listings you read to see every positive

Thresholds are retuned on the rebuilt scores with the sweep's rule, so they can differ from the
sweep table in the 4th decimal (sweep.json rounds the scores).

Usage
  python eval/plots.py --sweep reports/sweep_<date>.json [--data eval/labeled.jsonl] --out DIR
  python eval/sweep.py --plots DIR       (same thing right after the sweep)
Without matplotlib the ranking metrics and plots.md are still written, charts are skipped.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import sweep as S  # noqa: E402

POS_C, NEG_C = "#1b7837", "#b2182b"   # positives green, negatives red, everywhere
CH_ORDER = ("tfidf", "bm25", "dense", "feat", "rrf(tfidf,bm25)", "rrf(dense,bm25)")


def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def channels(sw_res, rows):
    """{name: normalised scalar per row} rebuilt from sweep.json the same way sweep.py does."""
    sc = sw_res["scores"]
    ids = [str(r["id"]) for r in rows]
    missing = [i for i in ids if i not in sc and int(i) not in sc]
    if missing:
        raise SystemExit(f"sweep.json has no scores for {len(missing)} rows of the data (e.g. {missing[:5]}), "
                         "run the sweep on the same file")
    get = lambda i: sc.get(i, sc.get(int(i)) if i.isdigit() else None)  # noqa: E731
    raw = {k: [get(i)[k] for i in ids] for k in ("tfidf", "bm25", "dense") if k in get(ids[0])}
    D = {k: S.minmax([p - n for p, n in v]) for k, v in raw.items()}
    D["feat"] = [(get(i)["feat"] + 1) / 2 for i in ids]
    for a, b in (("tfidf", "bm25"), ("dense", "bm25")):
        if a in D and b in D:
            ra, rb = S.ranks(D[a]), S.ranks(D[b])
            D[f"rrf({a},{b})"] = S.minmax([1 / (60 + x) + 1 / (60 + z) for x, z in zip(ra, rb)])
    # the sweep's best feature fusion per base (hybrid), the strongest transparent variants so far
    for v in sw_res.get("variants", []):
        name = v.get("variant", "")
        if v.get("hybrid") and "+feat lam=" in name and name.split("+")[0] in D:
            base, lam = name.split("+")[0], float(name.rsplit("=", 1)[1])
            D[f"{base}+feat lam={lam}"] = [(1 - lam) * x + lam * f for x, f in zip(D[base], D["feat"])]
    return raw, {k: D[k] for k in list(CH_ORDER) + sorted(D) if k in D and (k in CH_ORDER or "+feat" in k)}


# ------------------------------------------------------------------ curves and ranking metrics
def pr_curve(sw, s, hybrid):
    """(recall, precision, threshold) for every distinct threshold, high to low."""
    out = []
    for t in sorted(set(s), reverse=True):
        m = S.prf(sw.y, sw.pred(s, t, hybrid))
        if m["tp"] + m["fp"]:
            out.append((m["R"], m["P"], t))
    return out


def order(sw, s, hybrid):
    """Reading order. pure: by score. hybrid: regex hits first, then gated rows by score,
    rows the gate drops last (the monitor never shows them)."""
    idx = range(len(s))
    if hybrid:
        return sorted(idx, key=lambda i: (-int(sw.regex[i]), -int(sw.gate[i]), -s[i], i))
    return sorted(idx, key=lambda i: (-s[i], i))


def ranking_metrics(y, ordr):
    rel = [bool(y[i]) for i in ordr]
    npos = sum(rel)
    if not npos:
        return {}
    hits, ap, first = 0, 0.0, None
    depth = {}
    for k, r in enumerate(rel, 1):
        if r:
            hits += 1
            ap += hits / k
            first = first or k
            for lvl in (0.95, 1.0):
                if lvl not in depth and hits >= math.ceil(lvl * npos - 1e-9):
                    depth[lvl] = k
    dcg = sum(1 / math.log2(k + 1) for k, r in enumerate(rel[:npos], 1) if r)
    idcg = sum(1 / math.log2(k + 1) for k in range(1, npos + 1))
    return {"npos": npos, "R@npos": round(sum(rel[:npos]) / npos, 3),
            "R@2npos": round(sum(rel[:2 * npos]) / npos, 3),
            "P@R95": round(math.ceil(0.95 * npos - 1e-9) / depth[0.95], 3),
            "P@R1": round(npos / depth[1.0], 3), "depth@R1": depth[1.0],
            "AP": round(ap / npos, 3), "nDCG@npos": round(dcg / idcg, 3), "MRR": round(1 / first, 3)}


def recall_at(y, ordr, ks):
    npos = sum(1 for v in y if v)
    cum, out, h = 0, [], 0
    rel = [bool(y[i]) for i in ordr]
    for k in ks:
        h = sum(rel[:k])
        out.append((h / npos if npos else 0.0, h / k if k else 0.0))
    return out


def tuned(sw, s, hybrid=True):
    """Threshold by the sweep's own rule (recall 1.0 first, best F1, ties stricter), retuned on the
    scores rebuilt here: sweep.json stores them rounded to 4 places, so reusing its t can flip a row."""
    b = sw.tune(s, hybrid)
    return b[1] if b else None


# ------------------------------------------------------------------ charts
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})
    return plt


def _save(plt, fig, out, name, title):
    fig.suptitle(title, fontsize=11, x=0.01, y=1.03, ha="left")
    p = out / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    return p.name


def ch_dataset(plt, out, rows, label):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2.2, 1.3, 1.2]})
    for a, key in ((ax[0], "kind"), (ax[1], "source")):
        c = Counter((r[key], r["label"]) for r in rows)
        cats = sorted({r[key] for r in rows}, key=lambda k: -(c[(k, 0)] + c[(k, 1)]))
        pos = [c[(k, 1)] for k in cats]
        neg = [c[(k, 0)] for k in cats]
        a.barh(cats, neg, color=NEG_C, label="label 0 (not wanted)")
        a.barh(cats, pos, left=neg, color=POS_C, label="label 1 (wanted)")
        a.invert_yaxis()
        a.set_xscale("symlog", linthresh=10)
        a.set_xlim(0, max(p_ + n_ for p_, n_ in zip(pos, neg)) * 4)
        a.set_xlabel("listings (symlog scale)")
        a.set_title(f"rows per {key}")
        for i, (p_, n_) in enumerate(zip(pos, neg)):
            a.text(p_ + n_, i, f" {n_}/{p_}", va="center", fontsize=7)
    ax[0].legend(loc="lower right", fontsize=7)
    L = {v: [len(r["title"]) for r in rows if r["label"] == v] for v in (0, 1)}
    bins = range(0, max(L[0] + L[1] + [10]) + 10, 5)
    ax[2].hist(L[0], bins=bins, color=NEG_C, alpha=0.55, label=f"label 0 (n={len(L[0])})", density=True)
    ax[2].hist(L[1], bins=bins, color=POS_C, alpha=0.55, label=f"label 1 (n={len(L[1])})", density=True)
    ax[2].set_xlabel("title length, characters")
    ax[2].set_ylabel("share of rows per label")
    ax[2].set_title("title length by label")
    ax[2].legend(fontsize=7)
    return _save(plt, fig, out, "01_dataset", f"{label}: what is in the labelled set (numbers are neg/pos)")


def ch_hist(plt, out, sw, D, res, label):
    n = len(D)
    cols = min(n, 3)
    rws = math.ceil(n / cols)
    fig, ax = plt.subplots(rws, cols, figsize=(4.2 * cols, 3.0 * rws), sharex=True, squeeze=False)
    bins = [i / 25 for i in range(26)]
    for a, (k, s) in zip(ax.flat, D.items()):
        neg = [x for x, v in zip(s, sw.y) if not v]
        pos = [x for x, v in zip(s, sw.y) if v]
        a.hist(neg, bins=bins, color=NEG_C, alpha=0.55, label=f"label 0 (n={len(neg)})",
               weights=[1 / len(neg)] * len(neg) if neg else None)
        a.hist(pos, bins=bins, color=POS_C, alpha=0.55, label=f"label 1 (n={len(pos)})",
               weights=[1 / len(pos)] * len(pos) if pos else None)
        t = tuned(sw, s, True)
        if t is not None:
            a.axvline(t, color="k", ls="--", lw=1)
            a.text(t, a.get_ylim()[1] * 0.9, f" t={t:.2f}", fontsize=7)
        a.set_title(k)
        a.set_xlabel("normalised score (0..1)")
        a.set_ylabel("share of the label")
    for a in list(ax.flat)[n:]:
        a.axis("off")
    ax[0][0].legend(fontsize=7)
    return _save(plt, fig, out, "02_score_hist",
                 f"{label}: score by label per channel (each label sums to 1), dashed = tuned hybrid threshold")


def ch_pr(plt, out, sw, D, label):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    reg = S.prf(sw.y, sw.regex)
    for a, hybrid in zip(ax, (False, True)):
        for k, s in D.items():
            c = pr_curve(sw, s, hybrid)
            a.plot([x[0] for x in c], [x[1] for x in c], marker=".", ms=3, lw=1.2, label=k,
                   ls="--" if k.startswith("rrf") else "-")
        a.scatter([reg["R"]], [reg["P"]], color="k", marker="*", s=90, zorder=5, label="regex alone")
        a.axvline(0.95, color="grey", lw=0.8, ls=":")
        a.set_xlabel("recall")
        a.set_title("hybrid: regex OR (gate AND score >= t)" if hybrid else "pure: gate AND score >= t")
        a.set_xlim(0, 1.02)
        a.set_ylim(0, 1.02)
        if hybrid:   # the regex already gives most of the recall, zoom to where the curves differ
            lo = min([reg["R"]] + [x[0] for k, s in D.items() for x in pr_curve(sw, s, True)])
            plo = min([reg["P"]] + [x[1] for k, s in D.items() for x in pr_curve(sw, s, True)])
            a.set_xlim(max(0, lo - 0.02), 1.005)
            a.set_ylim(max(0, plo - 0.05), 1.01)
            a.set_title(a.get_title() + "\n(axes zoomed)", fontsize=9)
    ax[0].set_ylabel("precision")
    ax[1].legend(fontsize=7, loc="lower left")
    return _save(plt, fig, out, "03_pr_curves", f"{label}: precision vs recall, dotted line = recall 0.95")


def ch_threshold(plt, out, res, label):
    st = {k: v for k, v in res.get("steps", {}).items() if not k.startswith("tfidf_2d")}
    if not st:
        return None
    n = len(st)
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.2), sharey=True, squeeze=False)
    for a, (k, steps) in zip(ax[0], st.items()):
        ts = [float(t) for t in steps]
        for m, c in (("P", "#2166ac"), ("R", POS_C), ("F1", "k")):
            a.plot(ts, [steps[t][m] for t in steps], label=m, color=c, lw=1.4)
        a.set_title(k)
        a.set_xlabel("threshold t (normalised)")
    ax[0][0].set_ylabel("metric")
    ax[0][0].legend(fontsize=7)
    return _save(plt, fig, out, "04_threshold", f"{label}: P, R, F1 against the threshold (hybrid mode)")


def ch_confusion(plt, out, sw, D, res, label):
    items = [("regex", sw.regex)]
    for k, s in D.items():
        t = tuned(sw, s, True)
        if t is not None:
            items.append((f"{k} hybrid t={t:.2f}", sw.pred(s, t, True)))
    n = len(items)
    fig, ax = plt.subplots(1, n, figsize=(2.5 * n, 2.8), squeeze=False)
    for a, (name, p) in zip(ax[0], items):
        m = S.prf(sw.y, p)
        M = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        a.imshow([[1, 0], [0, 1]], cmap="Greens", alpha=0.25, vmin=0, vmax=1.6)
        for i in range(2):
            for j in range(2):
                a.text(j, i, str(M[i][j]), ha="center", va="center", fontsize=12,
                       color=NEG_C if i != j and M[i][j] else "k")
        a.set_xticks([0, 1], ["pred 0", "pred 1"])
        a.set_yticks([0, 1], ["true 0", "true 1"])
        a.grid(False)
        a.set_title(f"{name}\nP={m['P']:.2f} R={m['R']:.2f}", fontsize=8)
    return _save(plt, fig, out, "05_confusion", f"{label}: confusion at the tuned threshold (red = errors)")


def grid_2d(sw, sc):
    F, FP, R = [], [], []
    for pos in S.GRID_POS:
        f, fp, r = [], [], []
        for mg in S.GRID_MARGIN:
            p = [sw.regex[i] or (sw.gate[i] and a >= pos and a - b >= mg) for i, (a, b) in enumerate(sc)]
            m = S.prf(sw.y, p)
            f.append(m["F1"])
            fp.append(m["fp"])
            r.append(m["R"])
        F.append(f)
        FP.append(fp)
        R.append(r)
    return F, FP, R


def ch_2d(plt, out, sw, raw, res, label):
    if "tfidf" not in raw:
        return None
    best = res.get("tfidf_2d", {}).get("hybrid")
    F, FP, R = grid_2d(sw, raw["tfidf"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a, M, name, cmap in ((ax[0], F, "F1", "viridis"), (ax[1], FP, "false positives", "magma_r")):
        im = a.imshow(M, origin="lower", aspect="auto", cmap=cmap)
        fig.colorbar(im, ax=a, shrink=0.85, label=name)
        a.contourf([[1 if r < 1.0 else 0 for r in rr] for rr in R], levels=[0.5, 1.5], colors="none",
                   hatches=["///"])
        a.contour([[1.0 if r >= 1.0 else 0.0 for r in rr] for rr in R], levels=[0.5], colors="w", linewidths=1.2)
        if best:
            a.scatter([S.GRID_MARGIN.index(best["margin"])], [S.GRID_POS.index(best["pos"])], marker="*", s=120,
                      color="w", edgecolors="k", zorder=5)
        a.set_xticks(range(len(S.GRID_MARGIN))[::2], [f"{m:+.2f}" for m in S.GRID_MARGIN][::2], rotation=60)
        a.set_yticks(range(len(S.GRID_POS))[::2], [f"{p:.2f}" for p in S.GRID_POS][::2])
        a.set_xlabel("margin = sim_pos - sim_neg, at least")
        a.set_ylabel("sim_pos, at least")
        a.set_title(name)
        a.grid(False)
    return _save(plt, fig, out, "06_tfidf_2d", f"{label}: tfidf 2D rule (hybrid), hatched = recall below 1.0, star = chosen")


def ch_weights(plt, out, res, label):
    curves = [(f"w: {k}", "w", v) for k, v in res.get("w_curves", {}).items()]
    curves += [(f"lam: {k}", "lam", v) for k, v in res.get("lam_curves", {}).items()]
    if not curves:
        return None
    cols = 4
    rows_ = math.ceil(len(curves) / cols)
    fig, ax = plt.subplots(rows_, cols, figsize=(3.4 * cols, 2.9 * rows_), sharey=True, squeeze=False)
    for a, (name, key, c) in zip(ax.flat, curves):
        xs = [x[key] for x in c if "F1" in x]
        a.plot(xs, [x["F1"] for x in c if "F1" in x], color="k", lw=1.4, label="F1 at recall 1.0")
        a.plot(xs, [x["cv5"]["F1"] for x in c if "F1" in x], color="#2166ac", lw=1, ls="--", label="5-fold F1")
        a2 = a.twinx()
        a2.bar(xs, [x["fp"] for x in c if "F1" in x], width=0.06, color=NEG_C, alpha=0.3)
        a2.set_ylabel("FP", color=NEG_C, fontsize=7)
        a.set_title(name, fontsize=8)
        a.set_xlabel(key)
    for a in list(ax.flat)[len(curves):]:
        a.axis("off")
    ax[0][0].legend(fontsize=6)
    return _save(plt, fig, out, "07_fusion_weights", f"{label}: fusion weight vs F1 (lines) and false positives (bars)")


def ch_budget(plt, out, sw, D, label):
    N = len(sw.y)
    npos = sum(sw.y)
    ks = list(range(1, min(N, max(4 * npos, 20)) + 1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, hybrid in zip(ax, (False, True)):
        for k, s in D.items():
            c = recall_at(sw.y, order(sw, s, hybrid), ks)
            a.plot(ks, [x[0] for x in c], lw=1.3, label=f"{k} recall", ls="--" if k.startswith("rrf") else "-")
        a.plot(ks, [min(k / npos, 1.0) for k in ks], color="grey", lw=0.8, ls=":", label="perfect ranking")
        a.axvline(npos, color="grey", ls=":", lw=1)
        a.text(npos, 0.03, f" k = positives ({npos})", fontsize=7)
        a.set_xlabel("listings read from the top (k)")
        a.set_title("hybrid order" if hybrid else "pure order")
    ax[0].set_ylabel("recall@k")
    ax[1].legend(fontsize=7, loc="lower right")
    return _save(plt, fig, out, "08_recall_budget", f"{label}: how much you read to see every wanted listing")


def ch_errors(plt, out, sw, D, res, rows, label):
    items = [("regex", sw.regex)]
    for k, s in D.items():
        t = tuned(sw, s, True)
        if t is not None:
            items.append((f"{k} t={t:.2f}", sw.pred(s, t, True)))
    kinds = sorted({r["kind"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.32 * len(kinds) + 1.6), sharey=True)
    for a, (want, title) in zip(ax, ((False, "false positives (label 0 shown)"), (True, "false negatives (label 1 missed)"))):
        M = []
        for _, p in items:
            c = Counter(r["kind"] for r, yy, pp in zip(rows, sw.y, p) if yy == want and yy != pp)
            M.append([c[k] for k in kinds])
        T = list(map(list, zip(*M)))
        im = a.imshow(T, aspect="auto", cmap="Reds", vmin=0, vmax=max(1, max(map(max, T))))
        for i in range(len(kinds)):
            for j in range(len(items)):
                if T[i][j]:
                    a.text(j, i, T[i][j], ha="center", va="center", fontsize=7)
        a.set_xticks(range(len(items)), [n for n, _ in items], rotation=35, ha="right", fontsize=7)
        a.set_yticks(range(len(kinds)), kinds, fontsize=7)
        a.set_title(title)
        a.grid(False)
    return _save(plt, fig, out, "09_errors_by_kind", f"{label}: where each tuned variant goes wrong, per listing kind")


# ------------------------------------------------------------------ main
def build(sweep_path, data_path, out, label=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads(Path(sweep_path).read_text(encoding="utf-8"))
    rows = load_rows(data_path)
    sw = S.Sweep(rows)
    raw, D = channels(res, rows)
    srcs = Counter("synthetic" if r["source"] == "synthetic" else "real" for r in rows)
    label = label or f"{Path(data_path).name}, {len(rows)} rows ({srcs['real']} real, {srcs['synthetic']} synthetic)"
    rk = {}
    for k, s in D.items():
        for hybrid in (False, True):
            rk[f"{k} {'hybrid' if hybrid else 'pure'}"] = ranking_metrics(sw.y, order(sw, s, hybrid))
    pngs, note = [], ""
    try:
        plt = _plt()
    except ImportError:
        plt, note = None, "matplotlib is not installed, charts skipped (pip install -r requirements.txt)"
    if plt:
        for fn in (lambda: ch_dataset(plt, out, rows, label), lambda: ch_hist(plt, out, sw, D, res, label),
                   lambda: ch_pr(plt, out, sw, D, label), lambda: ch_threshold(plt, out, res, label),
                   lambda: ch_confusion(plt, out, sw, D, res, label), lambda: ch_2d(plt, out, sw, raw, res, label),
                   lambda: ch_weights(plt, out, res, label), lambda: ch_budget(plt, out, sw, D, label),
                   lambda: ch_errors(plt, out, sw, D, res, rows, label)):
            p = fn()
            if p:
                pngs.append(p)
    meta = {"label": label, "n": len(rows), "positives": sum(sw.y), "channels": list(D),
            "dense": "dense" in D, "ranking": rk, "charts": pngs, "note": note}
    (out / "plots.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "plots.md").write_text(markdown(meta), encoding="utf-8")
    return meta


CAPTIONS = {
    "01_dataset": "What the labelled set holds: rows per kind and source split by label, title length by label.",
    "02_score_hist": "Raw score distribution by label per channel. Overlap = rows no threshold can separate.",
    "03_pr_curves": "Precision against recall when the threshold slides. Star = regex alone.",
    "04_threshold": "How P, R and F1 move with the threshold (hybrid). The flat P line is where the regex dominates.",
    "05_confusion": "Confusion matrix per variant at the threshold the sweep picked (recall first).",
    "06_tfidf_2d": "The monitor's own tfidf rule over its full grid: how pos and margin interact. Hatched cells miss a "
                   "positive, the white line bounds recall 1.0, the star is the setting the sweep picked.",
    "07_fusion_weights": "Fusion weight against F1 (black), 5-fold F1 (blue dashed) and false positives (bars).",
    "08_recall_budget": "Recall when you read the top k listings. The dotted line is k = number of positives.",
    "09_errors_by_kind": "Errors per listing kind, which kinds each channel confuses with the product.",
}


def markdown(meta, prefix=""):
    L = [f"<!-- generated by python eval/plots.py, do not edit by hand -->", "",
         f"## Charts: {meta['label']}", ""]
    if meta["note"]:
        L += [f"_{meta['note']}_", ""]
    if not meta["dense"]:
        L += ["_dense channel not in this sweep (no dense scores), only the transparent channels are drawn_", ""]
    L += ["### Ranking metrics (read the list from the top)", "",
          f"{meta['positives']} positives. pure = sorted by the score alone, hybrid = regex hits first, then rows the "
          "model gate lets through by score. R@n = recall after reading n listings, P@R1 / P@R95 = precision at the "
          "depth where recall reaches 1.0 / 0.95, depth@R1 = listings read to see all positives.", "",
          "| channel | order | R@npos | R@2npos | P@R95 | P@R1 | depth@R1 | AP | nDCG@npos |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, m in meta["ranking"].items():
        if m:
            ch, mode = k.rsplit(" ", 1)
            L.append(f"| {ch} | {mode} | {m['R@npos']:.3f} | {m['R@2npos']:.3f} | {m['P@R95']:.3f} | {m['P@R1']:.3f} | "
                     f"{m['depth@R1']} | {m['AP']:.3f} | {m['nDCG@npos']:.3f} |")
    L.append("")
    for p in meta["charts"]:
        stem = Path(p).stem
        L += [f"### {stem}", "", CAPTIONS.get(stem, ""), "", f"![{stem}]({prefix}{p})", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="sweep.json written by eval/sweep.py")
    ap.add_argument("--data", default=str(ROOT / "eval" / "labeled.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None, help="title prefix on every chart")
    a = ap.parse_args(argv)
    meta = build(a.sweep, a.data, a.out, a.label)
    print(f"{len(meta['charts'])} charts -> {a.out}" + (f" ({meta['note']})" if meta["note"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
