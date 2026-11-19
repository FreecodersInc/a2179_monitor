#!/usr/bin/env python3
"""F1 report for the A2179 filter on eval/labeled.jsonl.

Modes: regex | tfidf (similarity only, gated) | hybrid_tfidf (regex OR tfidf)
       embed | hybrid_embed (only if sentence-transformers + model are available).
Thresholds of the similarity channel are tuned for recall 1.0 (max F1 among those).
A 2-fold check (tune on odd ids, test on even ids and back) shows how much of that is overfit.

Usage: python eval/run_eval.py [--target NAME] [--no-embed] [--model NAME] [--local-only]
--target picks the profile (targets/<name>.json): its rules, prototypes and labeled set
(profile eval.data, else eval/<name>/labeled.jsonl). Default: A2179_TARGET or a2179_screen.
Exit 4 when the target has no labeled set yet (nothing to measure).
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from a2179.matcher import Matcher  # noqa: E402
from a2179.rules import classify  # noqa: E402
from a2179.similarity import TfidfSim  # noqa: E402
from a2179.embedder import EmbedSim, EmbedderUnavailable, DEFAULT_MODEL  # noqa: E402
from a2179.target import DEFAULT_TARGET, eval_data_path, load_target  # noqa: E402

NO_LABELS = 4

GRID_POS = [round(x * 0.02, 2) for x in range(0, 50)]
GRID_MARGIN = [round(-0.2 + x * 0.02, 2) for x in range(0, 26)]


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def prf(y, p):
    tp = sum(1 for a, b in zip(y, p) if a and b)
    fp = sum(1 for a, b in zip(y, p) if not a and b)
    fn = sum(1 for a, b in zip(y, p) if a and not b)
    tn = sum(1 for a, b in zip(y, p) if not a and not b)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def preds(rows, regex, gates, scores, pos, margin, use_regex=True):
    out = []
    for r, g, (sp, sn) in zip(regex, gates, scores):
        s = g and sp >= pos and (sp - sn) >= margin
        out.append((r or s) if use_regex else s)
    return out


def tune(y, regex, gates, scores, use_regex):
    best = None
    for pos in GRID_POS:
        for m in GRID_MARGIN:
            p = preds(None, regex, gates, scores, pos, m, use_regex)
            met = prf(y, p)
            if met["recall"] < 1.0:
                continue
            key = (met["f1"], pos, m)  # tie-break: stricter thresholds
            if best is None or key > best[0]:
                best = (key, {"pos": pos, "margin": m}, met)
    return best


def twofold(rows, y, regex, gates, scores, use_regex):
    idx = [list(range(0, len(rows), 2)), list(range(1, len(rows), 2))]
    agg = Counter()
    for a, b in ((0, 1), (1, 0)):
        tr, te = idx[a], idx[b]
        sel = lambda L, I: [L[i] for i in I]  # noqa: E731
        best = tune(sel(y, tr), sel(regex, tr), sel(gates, tr), sel(scores, tr), use_regex)
        th = best[1] if best else {"pos": 0.0, "margin": -1.0}
        p = preds(None, sel(regex, te), sel(gates, te), sel(scores, te), th["pos"], th["margin"], use_regex)
        m = prf(sel(y, te), p)
        for k in ("tp", "fp", "fn", "tn"):
            agg[k] += m[k]
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(pr, 4), "recall": round(rc, 4),
            "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0, **dict(agg)}


def run_sim(name, sim, rows, y, regex, gates, report, errors):
    t0 = time.time()
    texts = [f"{r['title']} {r['snippet']}".strip() for r in rows]
    scores = sim.score_many(texts) if hasattr(sim, "score_many") else [sim.score(t) for t in texts]
    ms = (time.time() - t0) * 1000 / len(rows)
    for use_regex, mode in ((False, name), (True, f"hybrid_{name}")):
        best = tune(y, regex, gates, scores, use_regex)
        if best is None:
            report["modes"][mode] = {"note": "no threshold reaches recall 1.0"}
            continue
        th, met = best[1], best[2]
        p = preds(None, regex, gates, scores, th["pos"], th["margin"], use_regex)
        report["modes"][mode] = {**met, "thresholds": th, "ms_per_item": round(ms, 2),
                                 "twofold": twofold(rows, y, regex, gates, scores, use_regex)}
        errors[mode] = [(r["id"], r["label"], r["title"], round(s[0], 3), round(s[1], 3))
                        for r, a, b, s in zip(rows, y, p, scores) if a != b]
    return scores


def write_thresholds(path: Path, report: dict, model: str):
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    th = cfg.setdefault("thresholds", {})
    for mode in ("tfidf", "embed"):
        m = report["modes"].get(f"hybrid_{mode}", {})
        if "thresholds" in m:
            th[mode] = {**m["thresholds"], "calibrated": True}
            if mode == "embed":
                th[mode]["model"] = model
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"thresholds written to {path.name}: {th}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=os.environ.get("A2179_TARGET") or DEFAULT_TARGET)
    ap.add_argument("--data", default=None, help="labeled JSONL (default: the target's labeled set)")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--write-thresholds", action="store_true",
                    help="save the tuned embed (and tfidf) thresholds into config.json")
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports"))
    ap.add_argument("--no-a1932", action="store_true", help="treat A1932-only listings as negatives")
    a = ap.parse_args(argv)

    tg = load_target(a.target)
    data = Path(a.data) if a.data else eval_data_path(tg)
    if not data.is_file():
        print(f"target {tg.name}: no labeled set at {data} (label one first, see work/T-019)", file=sys.stderr)
        return NO_LABELS
    rows = load(data)
    if a.no_a1932:
        for r in rows:
            if r["kind"] == "a1932_display":
                r["label"] = 0
    y = [bool(r["label"]) for r in rows]
    rr = [classify(r["title"], r["snippet"], not a.no_a1932, target=tg) for r in rows]
    regex = [x.match for x in rr]
    gates = [x.gate for x in rr]

    report = {
        "target": tg.name,
        "dataset": {"n": len(rows), "positives": sum(y), "negatives": len(y) - sum(y),
                    "by_source": dict(Counter(r.get("source") for r in rows)),
                    "by_lang": dict(Counter(r.get("lang") for r in rows)),
                    "by_kind": dict(Counter(r.get("kind") for r in rows))},
        "modes": {"regex": prf(y, regex)},
        "generated": time.strftime("%Y-%m-%d %H:%M %Z"),
    }
    errors = {"regex": [(r["id"], r["label"], r["title"], "; ".join(x.reasons))
                        for r, x, t in zip(rows, rr, y) if x.match != t]}

    run_sim("tfidf", TfidfSim(tg.positive, tg.negative), rows, y, regex, gates, report, errors)
    if a.no_embed:
        report["embedder"] = "skipped (--no-embed)"
    else:
        try:
            sim = EmbedSim(a.model, a.local_only, pos=tg.positive, neg=tg.negative,
                           device=os.environ.get("A2179_DEVICE") or "auto")
            report["embedder"] = a.model
            run_sim("embed", sim, rows, y, regex, gates, report, errors)
        except EmbedderUnavailable as e:
            report["embedder"] = f"unavailable ({e.stage}): {e.cause}"
            print(e.console(), file=sys.stderr)

    if a.write_thresholds:
        write_thresholds(Path(a.config), report, a.model)

    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    (out / "eval_report.json").write_text(json.dumps({**report, "errors": errors}, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    lines = [f"# Eval report: {tg.label} ({report['generated']})", "",
             f"Set: {report['dataset']['n']} items, {report['dataset']['positives']} positive, "
             f"{report['dataset']['negatives']} negative. Sources: {report['dataset']['by_source']}. "
             f"Languages: {report['dataset']['by_lang']}.", "",
             f"Embedder: {report['embedder']}", "",
             "| mode | precision | recall | F1 | TP | FP | FN | TN | thresholds | 2-fold P/R/F1 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for mode, m in report["modes"].items():
        if "f1" not in m:
            lines.append(f"| {mode} | {m.get('note')} |||||||||")
            continue
        tf = m.get("twofold")
        lines.append(f"| {mode} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['tp']} | {m['fp']} | "
                     f"{m['fn']} | {m['tn']} | {m.get('thresholds', '')} | "
                     + (f"{tf['precision']:.3f}/{tf['recall']:.3f}/{tf['f1']:.3f}" if tf else "") + " |")
    lines += ["", f"Confusion: TP/FP/FN/TN above (positive = a match for {tg.label}: {tg.description}).", ""]
    for mode, errs in errors.items():
        lines.append(f"## Errors: {mode} ({len(errs)})")
        lines += [f"- id {e[0]} label={e[1]}: {e[2]}  ({', '.join(map(str, e[3:]))})" for e in errs]
        lines.append("")
    (out / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8 + len(report["modes"])]))


if __name__ == "__main__":
    sys.exit(main())
