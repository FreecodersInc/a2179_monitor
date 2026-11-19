#!/usr/bin/env python3
"""Transparent threshold / fusion sweep on eval/labeled.jsonl. No vector DB, no learned weights.

Channels (each gives sim_pos, sim_neg = max similarity to the positive / negative prototypes
of a2179/prototypes.py, and the scalar d = sim_pos - sim_neg):
  tfidf   char 3..5 gram TF-IDF cosine (a2179/similarity.py, what the monitor uses today)
  bm25    word BM25 (k1=1.5, b=0.75, IDF from the 33 prototypes only), listing = query
  dense   MiniLM cosine (paraphrase-multilingual-MiniLM-L12-v2). Needs real embeddings:
          python eval/sweep.py --compute-dense   (writes eval/dense_scores.jsonl, then rerun)
  feat    explicit product features: +1 display words, -1 per other-part / accessory /
          service / wanted word group, clipped to [-1, 1] (lists below, fixed before the sweep)

Fusions (each scalar min-max normalised over the eval set first):
  lin(a,b,w)        w*a + (1-w)*b, w = 0.0 .. 1.0 step 0.1, threshold swept per w
  rrf(a,b)          sum 1/(60 + rank), the usual reciprocal rank fusion, as a control
  +feat(lam)        (1-lam)*base + lam*feat, lam = 0.0 .. 1.0 step 0.1, the weight is shown

Every similarity decision is gated like the monitor (A2179-compatible model in the text).
pure = similarity alone, hybrid = regex OR similarity (what the monitor does).
Threshold choice = recall first: best F1 among thresholds with recall 1.0, ties -> stricter.
Honesty checks: 2-fold (odd/even rows, like run_eval.py) and 5-fold (row index mod 5), the
threshold tuned on the train part only, plus the full-set threshold applied to real rows only.

Usage: python eval/sweep.py [--dense-scores eval/dense_scores.jsonl] [--compute-dense] [--plots DIR]
Charts and ranking metrics (recall@k, precision at recall 1.0 / 0.95, nDCG): eval/plots.py
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from a2179 import prototypes  # noqa: E402
from a2179.normalize import normalize  # noqa: E402
from a2179.rules import classify  # noqa: E402
from a2179.similarity import TfidfSim  # noqa: E402
from a2179.target import DEFAULT_TARGET, eval_data_path, load_target  # noqa: E402

NO_LABELS = 4   # the target has no labeled set yet

DATE = "2026-11-19"
WS = [round(x * 0.1, 1) for x in range(11)]
GRID_POS = [round(x * 0.02, 2) for x in range(0, 50)]
GRID_MARGIN = [round(-0.2 + x * 0.02, 2) for x in range(0, 26)]

# ------------------------------------------------------------------ features (fixed a priori)
F_DISPLAY = ["ekran", "ekrana", "ekranom", "displej", "displeja", "display", "lcd", "panel", "matrica",
             "matricu", "screen", "lid", "assembly", "retina display", "gornji deo", "gornji poklopac",
             "экран", "дисплей", "дисплеј", "екран", "матрица", "крышка", "в сборе"]
F_OTHER = {
    "hinge": ["sarke", "sarka", "hinge", "hinges", "петли"],
    "cooling": ["kuler", "cooler", "fan", "ventilator", "кулер"],
    "keyboard": ["tastatura", "tastaturu", "keyboard", "palmrest", "topcase", "клавиатура", "тастатура"],
    "battery_power": ["baterija", "bateriju", "battery", "punjac", "charger", "adapter", "аккумулятор",
                      "батарея", "зарядка", "батерија", "пуњач"],
    "case_accessory": ["futrola", "futrolu", "maska", "case", "cover", "sleeve", "torba", "чехол", "navlaka"],
    "protector": ["folija", "staklo", "zastita", "zastitna", "protector", "kaljeno", "пленка", "стекло"],
    "cable_small": ["kabl", "kabal", "cable", "flex", "flet", "konektor", "konektori", "connector", "antena",
                    "antene", "srafovi", "screws", "kamera", "camera", "lepak", "traka", "adhesive"],
    "board_bottom": ["maticna", "ploca", "motherboard", "logic board", "donji poklopac", "bottom case"],
    "service": ["zamena", "servis", "service", "repair", "popravka", "ugradnja", "ремонт", "замена"],
    "wanted": ["trazim", "kupujem", "otkup", "куплю"],
}


def _words_rx(words):
    return re.compile(r"\b(?:" + "|".join(sorted({re.escape(normalize(w)) for w in words}, key=len, reverse=True))
                      + r")\b")


RX_DISPLAY = _words_rx(F_DISPLAY)
RX_OTHER = {k: _words_rx(v) for k, v in F_OTHER.items()}


def feature_rx(target=None):
    """(display rx, {group: rx}) of a target: profile eval.features {"display": [...], "other": {...}},
    else the fixed A2179 groups above (so the A2179 numbers stay the same)."""
    tg = load_target(target) if target is not None else None
    fe = (tg.eval.get("features") if tg else None) or {}
    if not fe:
        if tg is None or tg.name == DEFAULT_TARGET:
            return RX_DISPLAY, RX_OTHER
        return _words_rx([]), {}   # another target without groups: the feat channel stays flat
    return _words_rx(fe.get("display") or []), {k: _words_rx(v) for k, v in (fe.get("other") or {}).items()}


def features(title: str, snippet: str = "", rx=None):
    """(value in [-1, 1], which groups fired). Display words from title + snippet, the negative
    groups from the title only (same split as the regex rules: a description may mention a service).
    rx: feature_rx(target), default the A2179 groups. For another target "display" means its
    positive part words (profile eval.features.display)."""
    rx_display, rx_other = rx or (RX_DISPLAY, RX_OTHER)
    tt, t = normalize(title), normalize(f"{title} {snippet}")
    disp = 1 if rx_display.pattern != r"\b(?:)\b" and rx_display.search(t) else 0
    other = [k for k, r_ in rx_other.items() if r_.search(tt)]
    return max(-1.0, min(1.0, disp - len(other))), (["display"] if disp else []) + other


# ------------------------------------------------------------------ BM25 (transparent)
class BM25:
    def __init__(self, pos, neg, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [normalize(d).split() for d in list(pos) + list(neg)]
        self.npos = len(pos)
        n = len(self.docs)
        self.avgdl = sum(map(len, self.docs)) / n
        df = Counter(w for d in self.docs for w in set(d))
        self.idf = {w: math.log((n - c + 0.5) / (c + 0.5) + 1.0) for w, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def _one(self, q, i):
        d, tf = self.docs[i], self.tf[i]
        s = 0.0
        for w in q:
            f = tf.get(w, 0)
            if f:
                s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
        return s

    def score(self, text):
        q = set(normalize(text).split())
        sc = [self._one(q, i) for i in range(len(self.docs))]
        return max(sc[:self.npos]), max(sc[self.npos:])


# ------------------------------------------------------------------ metrics
def prf(y, p):
    tp = sum(1 for a, b in zip(y, p) if a and b)
    fp = sum(1 for a, b in zip(y, p) if not a and b)
    fn = sum(1 for a, b in zip(y, p) if a and not b)
    tn = len(y) - tp - fp - fn
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"P": round(pr, 3), "R": round(rc, 3), "F1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def minmax(v):
    lo, hi = min(v), max(v)
    return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in v]


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: -v[i])
    r = [0] * len(v)
    for k, i in enumerate(order, 1):
        r[i] = k
    return r


class Sweep:
    def __init__(self, rows, accept_a1932=True, target=None):
        self.rows = rows
        self.y = [bool(r["label"]) for r in rows]
        rr = [classify(r["title"], r.get("snippet", ""), accept_a1932, target=target) for r in rows]
        self.regex = [x.match for x in rr]
        self.gate = [x.gate for x in rr]

    def pred(self, s, t, hybrid, idx=None):
        idx = range(len(self.rows)) if idx is None else idx
        return [(self.regex[i] if hybrid else False) or (self.gate[i] and s[i] >= t) for i in idx]

    def tune(self, s, hybrid, idx=None):
        idx = list(range(len(self.rows))) if idx is None else idx
        y = [self.y[i] for i in idx]
        cands = sorted({s[i] for i in idx}, reverse=True)
        best = None
        for t in cands:
            m = prf(y, self.pred(s, t, hybrid, idx))
            if m["R"] < 1.0:
                continue
            key = (m["F1"], t)
            if best is None or key > best[0]:
                best = (key, t, m)
        return best

    def kfold(self, s, hybrid, k):
        idx = list(range(len(self.rows)))
        folds = [[i for i in idx if i % k == f] for f in range(k)]
        agg = Counter()
        for f in range(k):
            tr = [i for i in idx if i % k != f]
            b = self.tune(s, hybrid, tr)
            t = b[1] if b else -1e9
            m = prf([self.y[i] for i in folds[f]], self.pred(s, t, hybrid, folds[f]))
            agg.update({x: m[x] for x in ("tp", "fp", "fn", "tn")})
        tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        return {"P": round(pr, 3), "R": round(rc, 3), "F1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0}

    def subset(self, s, t, hybrid, real):
        idx = [i for i, r in enumerate(self.rows) if (r.get("source") != "synthetic") == real]
        return prf([self.y[i] for i in idx], self.pred(s, t, hybrid, idx))

    def evaluate(self, name, s, hybrid, extra=None):
        b = self.tune(s, hybrid)
        if b is None:
            return {"variant": name, "hybrid": hybrid, "note": "no threshold reaches recall 1.0", **(extra or {})}
        t, m = b[1], b[2]
        errs = [self.rows[i]["id"] for i, (a, p) in enumerate(zip(self.y, self.pred(s, t, hybrid))) if a != p]
        return {"variant": name, "hybrid": hybrid, "t": round(t, 4), **m, "cv2": self.kfold(s, hybrid, 2),
                "cv5": self.kfold(s, hybrid, 5), "real": self.subset(s, t, hybrid, True),
                "synthetic": self.subset(s, t, hybrid, False), "errors": errs, **(extra or {})}


def tfidf_2d(sw, sc, hybrid):
    """The monitor's own rule: sim_pos >= pos and margin >= m, on run_eval.py's grid."""
    best, table = None, {}
    for pos in GRID_POS:
        for mg in GRID_MARGIN:
            p = [(sw.regex[i] if hybrid else False) or (sw.gate[i] and a >= pos and a - b >= mg)
                 for i, (a, b) in enumerate(sc)]
            m = prf(sw.y, p)
            table[(pos, mg)] = m
            if m["R"] == 1.0 and (best is None or (m["F1"], pos, mg) > best[0]):
                best = ((m["F1"], pos, mg), pos, mg, m)
    return best, table


def compute_dense(rows, out: Path, model: str, proto_prefix: str = "", text_prefix: str = "", local_only=False,
                  device="auto", pos=None, neg=None):
    from a2179.embedder import EmbedSim
    texts = [f"{r['title']} {r.get('snippet', '')}".strip() for r in rows]
    try:
        sim = EmbedSim(model, local_only=local_only, device=device, proto_prefix=proto_prefix, text_prefix=text_prefix,
                       pos=pos, neg=neg)
        t0 = time.time()
        sc = sim.score_many(texts)
    except Exception as e:  # noqa: BLE001
        # a big model on a small GPU (bge-m3 or e5-large on a 2 GB MX130): same model on the CPU
        why = f"{e} {getattr(e, 'detail', '')} {e.__cause__ or ''}".lower()   # EmbedderUnavailable wraps the torch error
        if device == "cpu" or "out of memory" not in why:
            raise
        print(f"dense {model}: CUDA out of memory, rerun on cpu")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        sim = EmbedSim(model, local_only=local_only, device="cpu", proto_prefix=proto_prefix, text_prefix=text_prefix,
                       pos=pos, neg=neg)
        t0 = time.time()
        sc = sim.score_many(texts)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r, (a, b) in zip(rows, sc):
            f.write(json.dumps({"id": r["id"], "pos": round(a, 5), "neg": round(b, 5)}) + "\n")
    print(f"dense scores for {len(rows)} rows -> {out} ({time.time() - t0:.1f} s, device {sim.device}, "
          f"model {model}, prefixes {proto_prefix!r} / {text_prefix!r})")


# ------------------------------------------------------------------ several embedders (EXP-003, H-30)
REGISTRY = ROOT / "eval" / "embedders.json"


def registry(path=REGISTRY):
    return json.loads(Path(path).read_text(encoding="utf-8"))["models"]


def hf_cached(model_id: str) -> bool:
    """Is the model in the local Hugging Face cache? Looks at the cache folder only, no network."""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"), "hub")
    snap = Path(hub) / ("models--" + model_id.replace("/", "--")) / "snapshots"
    return snap.is_dir() and any(snap.iterdir())


def select_embedders(spec, reg=None, default_model=None):
    """spec: none | all | cached | comma list of short names or ids. Returns registry entries,
    the default model (the plain "dense" channel) left out, so it is never scored twice."""
    reg = registry() if reg is None else reg
    spec = (spec or "none").strip()
    if spec in ("", "none", "default"):
        return []
    if spec == "all":
        out = list(reg)
    elif spec == "cached":
        out = [m for m in reg if hf_cached(m["id"])]
    else:
        want = [x.strip() for x in spec.split(",") if x.strip()]
        by = {m["short"]: m for m in reg} | {m["id"]: m for m in reg if m["id"] not in {x["short"] for x in reg}}
        unknown = [w for w in want if w not in by]
        if unknown:
            raise SystemExit(f"unknown embedders {unknown}, known: {[m['short'] for m in reg]} (eval/embedders.json)")
        out = [by[w] for w in want]
    return [m for m in out if not (m["id"] == default_model and not m["proto_prefix"] and not m["text_prefix"])]


def extra_scores_path(dense_scores: Path, short: str) -> Path:
    return dense_scores.with_name(f"{dense_scores.stem}_{short}{dense_scores.suffix}")


def read_scores(path: Path, rows):
    if not path.exists():
        return None
    d = {j["id"]: (j["pos"], j["neg"]) for j in map(json.loads, path.read_text().splitlines()) if j}
    return [d[r["id"]] for r in rows] if all(r["id"] in d for r in rows) else None


def fmt(m):
    if "note" in m:
        return f"| {m['variant']} | {'hybrid' if m['hybrid'] else 'pure'} | {m['note']} |||||||||"
    return (f"| {m['variant']} | {'hybrid' if m['hybrid'] else 'pure'} | {m['t']} | {m['P']:.3f} | {m['R']:.3f} | "
            f"{m['F1']:.3f} | {m['fp']} | {m['fn']} | {m['cv2']['P']:.3f}/{m['cv2']['R']:.3f} | "
            f"{m['cv5']['P']:.3f}/{m['cv5']['R']:.3f} | {m['real']['P']:.3f}/{m['real']['R']:.3f} "
            f"(fp {m['real']['fp']}, fn {m['real']['fn']}) | {', '.join(map(str, m['errors'][:12]))}"
            f"{' …' if len(m['errors']) > 12 else ''} |")


HEAD = ("| variant | mode | threshold | P | R | F1 | FP | FN | 2-fold P/R | 5-fold P/R | real rows only P/R | "
        "error ids |\n|---|---|---|---|---|---|---|---|---|---|---|---|")


def committed_dense(rows):
    """What his committed run (reports/eval_report.json, real MiniLM on his machine) tells us.
    Only the misclassified rows carry scores there, so this is an error list, not a sweep."""
    p = ROOT / "reports" / "eval_report.json"
    if not p.exists():
        return None
    rep = json.loads(p.read_text(encoding="utf-8"))
    kind = {r["id"]: r["kind"] for r in rows}
    src = {r["id"]: r.get("source") for r in rows}
    out = {"modes": {k: rep["modes"].get(k) for k in ("embed", "hybrid_embed")}, "errors": {}}
    for k in ("embed", "hybrid_embed"):
        out["errors"][k] = [{"id": e[0], "label": e[1], "title": e[2], "pos": e[3], "neg": e[4],
                             "margin": round(e[3] - e[4], 3), "kind": kind.get(e[0]), "source": src.get(e[0])}
                            for e in rep.get("errors", {}).get(k, [])]
    return out


def _row(m):
    return fmt(m) if "cv2" in m or "note" in m else (
        f"| regex | pure | | {m['P']:.3f} | {m['R']:.3f} | {m['F1']:.3f} | {m['fp']} | {m['fn']} | | | | "
        f"{', '.join(map(str, m['errors']))} |")


def tables_md(res, rows, sw):
    L = [f"<!-- generated by python eval/sweep.py ({res['generated']}), do not edit by hand -->", ""]
    L += ["### Variants, best threshold each (recall first)", "", HEAD] + [_row(m) for m in res["variants"]]
    L += ["", f"Dense channel: {res['dense'] or 'not computed here, every dense row is missing from this table'}", ""]
    L += ["### tfidf, the monitor's own 2D rule (hybrid), one step at a time", "",
          "| pos (margin 0.18) | P | R | FP | FN |", "|---|---|---|---|---|"]
    L += [f"| {k} | {m['P']:.3f} | {m['R']:.3f} | {m['fp']} | {m['fn']} |" for k, m in res["steps"]["tfidf_2d_pos"].items()]
    L += ["", "| margin (pos 0.38) | P | R | FP | FN |", "|---|---|---|---|---|"]
    L += [f"| {k} | {m['P']:.3f} | {m['R']:.3f} | {m['fp']} | {m['fn']} |" for k, m in res["steps"]["tfidf_2d_margin"].items()]
    L += ["", f"Best 2D: pure {res['tfidf_2d'].get('pure')}, hybrid {res['tfidf_2d'].get('hybrid')}", ""]
    chans = [k for k in res["steps"] if not k.startswith("tfidf_2d")]
    L += ["### Normalised scalar threshold steps (hybrid)", "",
          "| t | " + " | ".join(f"{k} P/R (FP,FN)" for k in chans) + " |", "|---" * (len(chans) + 1) + "|"]
    for t in res["steps"][chans[0]]:
        L.append(f"| {t} | " + " | ".join(
            f"{res['steps'][k][t]['P']:.3f}/{res['steps'][k][t]['R']:.3f} ({res['steps'][k][t]['fp']},{res['steps'][k][t]['fn']})"
            for k in chans) + " |")
    for title, curves, key in (("w curves: w*A + (1-w)*B", res["w_curves"], "w"),
                               ("feature weight curves: (1-lam)*base + lam*feat", res["lam_curves"], "lam")):
        L += ["", f"### {title}", ""]
        for name, c in curves.items():
            L += [f"{name}", "", f"| {key} | threshold | P | R | F1 | FP | 5-fold F1 |", "|---|---|---|---|---|---|---|"]
            L += [f"| {x[key]} | {x.get('t', '')} | {x.get('P', '')} | {x.get('R', '')} | {x.get('F1', '')} | "
                  f"{x.get('fp', '')} | {x.get('cv5', {}).get('F1', '')} |" for x in c]
            L.append("")
    dv = [m for m in res["variants"] if m["variant"].startswith(("dense", "dense:")) and "cv5" in m]
    if any(m["variant"].startswith("dense:") for m in dv):
        L += ["### Dense models side by side (EXP-003, H-30)", "",
              f"default dense = {res.get('dense_model')}, others from eval/embedders.json. P at R 1.0 with the "
              "recall first threshold, 5-fold P/R, real rows only. Bootstrap CIs for P@R1 / P@R95: plots.md.", "",
              "| model | mode | threshold | P | FP | 5-fold P/R | real only P |", "|---|---|---|---|---|---|---|"]
        for m in sorted(dv, key=lambda m: (not m["hybrid"], -m["P"])):
            L.append(f"| {m['variant']} | {'hybrid' if m['hybrid'] else 'pure'} | {m['t']} | {m['P']:.3f} | {m['fp']} | "
                     f"{m['cv5']['P']:.3f}/{m['cv5']['R']:.3f} | {m['real']['P']:.3f} |")
        if res.get("embedders_skipped"):
            L += ["", "Skipped: " + ", ".join(f"{k} ({v})" for k, v in res["embedders_skipped"].items())]
        L.append("")
    cd = committed_dense(rows)
    if cd:
        L += ["### MiniLM from the committed run (reports/eval_report.json, computed on Leonid's machine)", "",
              "| mode | P | R | F1 | FP | thresholds | 2-fold P/R |", "|---|---|---|---|---|---|---|"]
        for k, m in cd["modes"].items():
            if m:
                L.append(f"| {k} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['fp']} | "
                         f"{m.get('thresholds')} | {m['twofold']['precision']:.3f}/{m['twofold']['recall']:.3f} |")
        for k, errs in cd["errors"].items():
            L += ["", f"{k}: misclassified rows with their cosine to the nearest positive / negative prototype", "",
                  "| id | kind | source | title | cos pos | cos neg | margin |", "|---|---|---|---|---|---|---|"]
            L += [f"| {e['id']} | {e['kind']} | {e['source']} | {e['title']} | {e['pos']} | {e['neg']} | {e['margin']:+.3f} |"
                  for e in sorted(errs, key=lambda e: -e["pos"])]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=os.environ.get("A2179_TARGET") or DEFAULT_TARGET,
                    help="profile in targets/: rules, prototypes, features and the labeled set")
    ap.add_argument("--data", default=None, help="labeled JSONL (default: the target's labeled set)")
    ap.add_argument("--dense-scores", default=str(ROOT / "eval" / "dense_scores.jsonl"))
    ap.add_argument("--compute-dense", action="store_true")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--out", default=str(ROOT / "reports" / f"sweep_{DATE}.json"))
    ap.add_argument("--plots", default=None, metavar="DIR", help="also draw the charts (eval/plots.py) into DIR")
    ap.add_argument("--embedders", default=os.environ.get("A2179_EMBEDDERS", "none"),
                    help="more dense channels from eval/embedders.json: none | all | cached | e5-base,bge-m3,... "
                         "each is scored into <dense-scores>_<short>.jsonl and becomes channel dense:<short>")
    a = ap.parse_args(argv)
    tg = load_target(a.target)
    a.data = a.data or str(eval_data_path(tg))
    if not Path(a.data).is_file():
        print(f"target {tg.name}: no labeled set at {a.data} (label one first, see work/T-019)", file=sys.stderr)
        return NO_LABELS
    rows = [json.loads(l) for l in Path(a.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r.setdefault("snippet", "")
    extras = select_embedders(a.embedders, default_model=a.model)
    skipped = {}
    if a.compute_dense:
        compute_dense(rows, Path(a.dense_scores), a.model, pos=tg.positive, neg=tg.negative)
        for m in extras:
            try:   # one model that is missing or fails must not cost the others
                compute_dense(rows, extra_scores_path(Path(a.dense_scores), m["short"]), m["id"],
                              m["proto_prefix"], m["text_prefix"],
                              local_only=os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "yes"),
                              pos=tg.positive, neg=tg.negative)
            except Exception as e:  # noqa: BLE001
                skipped[m["short"]] = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"dense {m['short']} skipped: {skipped[m['short']]}")
    sw = Sweep(rows, target=tg)
    texts = [f"{r['title']} {r['snippet']}".strip() for r in rows]
    tf = TfidfSim(tg.positive, tg.negative)
    bm = BM25(tg.positive, tg.negative)
    ch = {"tfidf": [tf.score(t) for t in texts], "bm25": [bm.score(t) for t in texts]}
    dense_src = None
    dp = Path(a.dense_scores)
    if dp.exists():
        d = {j["id"]: (j["pos"], j["neg"]) for j in map(json.loads, dp.read_text().splitlines()) if j}
        if all(r["id"] in d for r in rows):
            ch["dense"] = [d[r["id"]] for r in rows]
            dense_src = str(dp.relative_to(ROOT) if dp.is_relative_to(ROOT) else dp)
    embedders = {}
    for m in extras:
        sc = read_scores(extra_scores_path(dp, m["short"]), rows)
        if sc is None:
            skipped.setdefault(m["short"], "no scores file for every row (run with --compute-dense)")
            continue
        ch[f"dense:{m['short']}"] = sc
        embedders[m["short"]] = {k: m.get(k) for k in ("id", "proto_prefix", "text_prefix", "params_m", "weights_mb")}
    frx = feature_rx(tg)
    feats = [features(r["title"], r["snippet"], frx) for r in rows]
    F = [(f + 1) / 2 for f, _ in feats]
    D = {k: minmax([p - n for p, n in v]) for k, v in ch.items()}

    res = {"generated": time.strftime("%Y-%m-%d %H:%M %Z"), "target": tg.name, "n": len(rows), "dense": dense_src,
           "dense_model": a.model if dense_src else None, "embedders": embedders, "embedders_skipped": skipped,
           "variants": [], "w_curves": {}, "lam_curves": {}, "tfidf_2d": {}, "steps": {}}
    V = res["variants"]
    V.append({"variant": "regex", "hybrid": False, **prf(sw.y, sw.regex),
              "errors": [r["id"] for r, a_, p in zip(rows, sw.y, sw.regex) if a_ != p]})
    for hybrid in (False, True):
        b, table = tfidf_2d(sw, ch["tfidf"], hybrid)
        if b:
            res["tfidf_2d"]["hybrid" if hybrid else "pure"] = {"pos": b[1], "margin": b[2], **b[3]}
        if hybrid:
            res["steps"]["tfidf_2d_pos"] = {f"{p:.2f}": table[(p, 0.18)] for p in GRID_POS if 0.28 <= p <= 0.50}
            res["steps"]["tfidf_2d_margin"] = {f"{m:+.2f}": table[(0.38, m)] for m in GRID_MARGIN if -0.04 <= m <= 0.30}
        for k in D:
            V.append(sw.evaluate(f"{k} (d=pos-neg)", D[k], hybrid))
        V.append(sw.evaluate("feat only", F, hybrid))
        pairs = [("dense", "bm25"), ("tfidf", "bm25")]
        for A, B in pairs:
            if A not in D:
                continue
            curve = []
            for w in WS:
                s = [w * x + (1 - w) * z for x, z in zip(D[A], D[B])]
                curve.append({"w": w, **sw.evaluate(f"lin({A},{B})", s, hybrid)})
            res["w_curves"][f"{A}+{B} {'hybrid' if hybrid else 'pure'}"] = curve
            ok = [c for c in curve if "F1" in c]
            if ok:
                bw = max(ok, key=lambda c: (c["F1"], c["cv5"]["F1"], -abs(c["w"] - 0.5)))
                V.append({**bw, "variant": f"lin({A},{B}) w={bw['w']}"})
            ra, rb = ranks(D[A]), ranks(D[B])
            rrf = minmax([1 / (60 + x) + 1 / (60 + z) for x, z in zip(ra, rb)])
            V.append(sw.evaluate(f"rrf({A},{B})", rrf, hybrid))
        for base in [k for k in ("bm25", "tfidf", "dense") if k in D] + [k for k in D if k.startswith("dense:")]:
            curve = []
            for lam in WS:
                s = [(1 - lam) * x + lam * f for x, f in zip(D[base], F)]
                curve.append({"lam": lam, **sw.evaluate(f"{base}+feat", s, hybrid)})
            res["lam_curves"][f"{base}+feat {'hybrid' if hybrid else 'pure'}"] = curve
            ok = [c for c in curve if "F1" in c and c["lam"] not in (0.0, 1.0)]
            if ok:
                bl = max(ok, key=lambda c: (c["F1"], c["cv5"]["F1"], -c["lam"]))
                V.append({**bl, "variant": f"{base}+feat lam={bl['lam']}"})
    # per-threshold steps for the transparent scalars (hybrid, the monitor's mode)
    for k in list(D) + ["feat"]:
        s = F if k == "feat" else D[k]
        res["steps"][k] = {f"{t:.2f}": prf(sw.y, sw.pred(s, t, True)) for t in [x * 0.05 for x in range(4, 19)]}
    res["features"] = {r["id"]: g for r, (_, g) in zip(rows, feats)}
    res["scores"] = {r["id"]: {k: [round(v[i][0], 4), round(v[i][1], 4)] for k, v in ch.items()}
                     | {"feat": feats[i][0]} for i, r in enumerate(rows)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    md = Path(a.out).with_suffix(".tables.md")
    md.write_text(tables_md(res, rows, sw), encoding="utf-8")
    print(HEAD)
    for m in V:
        if "P" in m and "cv2" not in m:
            print(f"| regex | pure | | {m['P']:.3f} | {m['R']:.3f} | {m['F1']:.3f} | {m['fp']} | {m['fn']} | | | | "
                  f"{', '.join(map(str, m['errors']))} |")
        else:
            print(fmt(m))
    print(f"\ndense channel: {dense_src or 'MISSING (no eval/dense_scores.jsonl), dense rows skipped'}")
    print(f"json -> {a.out}\ntables -> {md}")
    if a.plots:
        import plots
        meta = plots.build(a.out, a.data, a.plots)
        print(f"{len(meta['charts'])} charts -> {a.plots}" + (f" ({meta['note']})" if meta["note"] else ""))


if __name__ == "__main__":
    sys.exit(main())
