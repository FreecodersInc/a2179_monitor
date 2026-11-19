#!/usr/bin/env python3
"""T-023 / ADR-0021: freeze the hybrid settings once, then score a fresh hand labelled batch with them.

freeze: pick (lam, t) and the min-max bounds (lo, hi) of every frozen variant on the whole old set, the
        same rule as the monitor and EXP-004 (recall 1.0, then best F1, ties as eval/sweep.py), and write them
        with fingerprints of the data and the profile. Commit the file before the fresh batch is collected.
score:  apply the frozen file to a new jsonl unchanged. Nothing is refit on it. Raw scores are
        min-maxed with the frozen bounds (clipped), so the threshold means the same thing as at freeze
        time. Labelled rows give P / R / F1 / FP / FN with 95% bootstrap intervals, all and real rows.

  python eval/freeze.py freeze --dense-dir results/2026-11-19_247d60c --out eval/frozen/a2179_screen.json
  python eval/freeze.py score --frozen eval/frozen/a2179_screen.json --data eval/holdout/fresh.jsonl \
      [--dense mpnet=path/to/dense_scores_mpnet.jsonl] [--out reports/exp005]
Dense variants need scores for the new rows (CI sweep or eval/sweep.py --compute-dense on the runner),
without them the variant is reported as skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import nested_cv as N  # noqa: E402
import sweep as S  # noqa: E402
from a2179.target import eval_data_path, load_target  # noqa: E402

VARIANTS = ["regex", "tfidf+feat", "tfidf+feat@strict", "dense:mpnet+feat"]   # @strict: F1 ties broken by t (EXP-004)
VERSION = 1


def load_rows(path: Path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r.setdefault("snippet", "")
    return rows


def profile_hash(tg):
    blob = json.dumps({"pos": list(tg.positive), "neg": list(tg.negative), "rules": getattr(tg, "rules", None),
                       "features": (tg.eval or {}).get("features")}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def channels(rows, tg, dense: dict):
    """raw score per variant base, feat in [0, 1], regex and gate lists."""
    sw = S.Sweep(rows, target=tg)
    texts = [f"{r['title']} {r['snippet']}".strip() for r in rows]
    tf = S.TfidfSim(tg.positive, tg.negative)
    frx = S.feature_rx(tg)
    feat = [(S.features(r["title"], r["snippet"], frx)[0] + 1) / 2 for r in rows]
    raw = {"tfidf": [p - n for p, n in (tf.score(t) for t in texts)]}
    for short, path in dense.items():
        v = N.load_dense(Path(path), rows)
        if v is not None:
            raw[f"dense:{short}"] = v
    return sw, raw, feat


def do_freeze(a):
    tg = load_target(a.target)
    data = eval_data_path(tg)
    rows = load_rows(data)
    dd = Path(a.dense_dir)
    sw, raw, feat = channels(rows, tg, {"mpnet": dd / "dense_scores_mpnet.jsonl"})
    y = [bool(r["label"]) for r in rows]
    allidx = list(range(len(rows)))
    out = {"version": VERSION, "target": tg.name, "profile_sha": profile_hash(tg),
           "data": {"path": str(data.relative_to(ROOT)), "rows": len(rows), "positives": sum(y),
                    "sha256": hashlib.sha256(data.read_bytes()).hexdigest()},
           "rule": "recall 1.0 on the freeze set, then best F1, ties: smallest lam then strictest t (@strict: strictest t)", "variants": {}}
    emb = {m["short"]: m for m in json.loads((ROOT / "eval" / "embedders.json").read_text())["models"]}
    for v in VARIANTS:
        if v == "regex":
            out["variants"][v] = {"kind": "regex"}
            continue
        name, _, tie = v.partition("@")
        tie = tie or "sweep"
        base = name[:-len("+feat")]
        if base not in raw:
            out["variants"][v] = {"kind": "skipped", "why": f"no scores for {base} in {dd}"}
            continue
        ch = N.pick(y, sw.regex, sw.gate, raw[base], feat, allidx, True, tie)
        lam, t, lo, hi = ch
        spec = {"kind": "hybrid", "base": base, "tie": tie, "lam": lam, "t": round(t, 6), "lo": lo, "hi": hi}
        if base.startswith("dense:"):
            m = emb[base.split(":", 1)[1]]
            spec["model"] = {k: m[k] for k in ("id", "proto_prefix", "text_prefix")}
        out["variants"][v] = spec
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


def boot(y, p, idx, n=1000, seed=0):
    rnd = random.Random(seed)
    vals = {"P": [], "R": [], "F1": []}
    for _ in range(n):
        s = [idx[rnd.randrange(len(idx))] for _ in idx]
        m = S.prf([y[i] for i in s], [p[i] for i in s])
        for k in vals:
            vals[k].append(m[k])
    ci = {}
    for k, v in vals.items():
        v.sort()
        ci[k] = [v[int(0.025 * n)], v[int(0.975 * n) - 1]]
    return ci


def do_score(a):
    fz = json.loads(Path(a.frozen).read_text(encoding="utf-8"))
    tg = load_target(fz["target"])
    if profile_hash(tg) != fz["profile_sha"]:
        print(f"WARNING: profile {tg.name} changed since freeze (rules, prototypes or features). "
              "The frozen numbers then do not describe the same matcher.", file=sys.stderr)
    rows = load_rows(Path(a.data))
    old = {json.loads(l)["id"] for l in (ROOT / fz["data"]["path"]).read_text(encoding="utf-8").splitlines() if l.strip()} \
        if (ROOT / fz["data"]["path"]).is_file() else set()
    overlap = sorted(r["id"] for r in rows if r["id"] in old)
    dense = dict(x.split("=", 1) for x in a.dense)
    sw, raw, feat = channels(rows, tg, dense)
    labelled = [i for i, r in enumerate(rows) if r.get("label") is not None]
    y = [bool(r.get("label")) for r in rows]
    real = [i for i in labelled if rows[i].get("source") != "synthetic"]
    res = {"frozen": a.frozen, "data": a.data, "rows": len(rows), "labelled": len(labelled), "real": len(real),
           "positives": sum(y[i] for i in labelled), "id_overlap_with_freeze_set": overlap, "variants": {}}
    allidx = list(range(len(rows)))
    for v, spec in fz["variants"].items():
        if spec["kind"] == "skipped":
            res["variants"][v] = spec
            continue
        if spec["kind"] == "regex":
            pred = [bool(x) for x in sw.regex]
        else:
            if spec["base"] not in raw:
                res["variants"][v] = {"kind": "skipped", "why": f"no scores for {spec['base']} (pass --dense)"}
                continue
            pd = N.predict(sw.regex, sw.gate, raw[spec["base"]], feat, allidx, spec["lam"], spec["t"], spec["lo"], spec["hi"])
            pred = [pd[i] for i in allidx]
        r_ = {"flagged": sum(pred)}
        for part, idx in (("all", labelled), ("real", real)):
            if idx:
                r_[part] = S.prf([y[i] for i in idx], [pred[i] for i in idx])
                r_[part]["ci95"] = boot(y, pred, idx)
        r_["flagged_ids"] = [rows[i]["id"] for i in allidx if pred[i]]
        res["variants"][v] = r_
    if a.out:
        o = Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "frozen_score.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 3 if overlap else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--target", default="a2179_screen")
    f.add_argument("--dense-dir", required=True)
    f.add_argument("--out", default=str(ROOT / "eval" / "frozen" / "a2179_screen.json"))
    s = sub.add_parser("score")
    s.add_argument("--frozen", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--dense", action="append", default=[], help="short=path, e.g. mpnet=dense_scores_mpnet.jsonl")
    s.add_argument("--out", default="")
    a = ap.parse_args(argv)
    return do_freeze(a) if a.cmd == "freeze" else do_score(a)


if __name__ == "__main__":
    sys.exit(main())
