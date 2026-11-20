#!/usr/bin/env python3
"""EXP-011 / T-025 / T-032: batch 3 of vacancies from rows nobody has looked at, labels by title and by full text.

sample: pool = review all.csv of one collection day minus everything in eval/de_jobs/labeled.jsonl
        (url, else normalised title + company). Strata by the current profile rules on the title:
        match (rules say yes) and rest, up to --per rows each, seed fixed. Writes eval/de_jobs/batch3.jsonl
        (no description text: ADR-0032, texts are not republished) and batch3_strata.json.
text:   writes the description text of every batch row into data/scratch (gitignored) for reading.
score:  Horvitz-Thompson P and R of the rules on the batch for the title labels (esci) and the full text
        labels (esci_text), strict E recall, label change rate, AUC of cv_score (variant A profile,
        variant B master CV stack) on both label sets.

  python eval/batch3.py sample --pool data/scratch_reports/2026-11-20/de_jobs/all.csv --seed 11 --per 75
  python eval/batch3.py text
  python eval/batch3.py score --out reports/exp011
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from a2179.normalize import normalize  # noqa: E402
from a2179.rules import classify  # noqa: E402
from a2179.target import load_target  # noqa: E402

LABELED = ROOT / "eval" / "de_jobs" / "labeled.jsonl"
BATCH = ROOT / "eval" / "de_jobs" / "batch3.jsonl"
STRATA = ROOT / "eval" / "de_jobs" / "batch3_strata.json"
POS = {"E", "S"}


def key(title: str, company: str) -> str:
    return normalize(f"{title} {company}")[:120]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def pool_rows(pool: Path, labeled: list[dict]) -> list[dict]:
    seen_u = {r.get("url", "").lower() for r in labeled if r.get("url")}
    seen_k = {key(r["title"], r.get("company", "")) for r in labeled}
    out, dup = [], set()
    for r in csv.DictReader(pool.open(encoding="utf-8")):
        k = key(r["title"], r.get("company", ""))
        if r["url"].lower() in seen_u or k in seen_k or k in dup:
            continue
        dup.add(k)
        out.append(r)
    return out


def sample(rows: list[dict], per: int, seed: int, tg) -> tuple[list[dict], dict]:
    by = {"match": [], "rest": []}
    for r in rows:
        by["match" if classify(r["title"], "", target=tg).match else "rest"].append(r)
    rnd = random.Random(seed)
    out, strata = [], {}
    for s, xs in by.items():
        pick = rnd.sample(xs, min(per, len(xs)))
        strata[s] = {"pool": len(xs), "kept": len(pick)}
        out += [{"stratum": s, **r} for r in pick]
    batch = []
    for i, r in enumerate(out, 1):
        batch.append({"id": f"b3-{i:03d}", "title": r["title"], "company": r.get("company", ""),
                      "location": r.get("location", ""), "url": r["url"], "source": f"real:{r.get('source', '')}",
                      "fetched_at": r.get("fetched_at", ""), "stratum": r["stratum"], "esci": "", "esci_text": "",
                      "labeler": "bot", "note": "", "note_text": ""})
    return batch, strata


def texts_for(batch: list[dict], cache: Path) -> dict:
    from a2179.jobs import collect_jobs
    jobs, _ = collect_jobs(cache, sources=("remotive", "remoteok", "ats"), fetch="never")
    by = {j.url: (j.text or j.snippet) for j in jobs}
    return {r["id"]: by.get(r["url"], "") for r in batch}


def ht(rows: list[dict], strata: dict, field: str, pos: set, tg) -> dict:
    """Horvitz-Thompson P and R of the rules over the pool, bootstrap 95% inside strata."""
    by = {}
    for r in rows:
        if r.get(field) not in ("E", "S", "C", "I"):
            continue
        by.setdefault(r["stratum"], []).append((classify(r["title"], "", target=tg).match, r[field] in pos))

    def est(b):
        tp = fp = fn = 0.0
        for s, xs in b.items():
            w = strata[s]["pool"] / len(xs)
            tp += w * sum(m and y for m, y in xs)
            fp += w * sum(m and not y for m, y in xs)
            fn += w * sum((not m) and y for m, y in xs)
        return (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)

    p, r = est(by)
    rnd = random.Random(0)
    bs = [est({s: [rnd.choice(xs) for _ in xs] for s, xs in by.items()}) for _ in range(1000)]
    q = lambda v, a: sorted(v)[int(a * (len(v) - 1))]  # noqa: E731
    return {"P": round(p, 3), "P_ci95": [round(q([b[0] for b in bs], .025), 3), round(q([b[0] for b in bs], .975), 3)],
            "R": round(r, 3), "R_ci95": [round(q([b[1] for b in bs], .025), 3), round(q([b[1] for b in bs], .975), 3)],
            "n": sum(len(x) for x in by.values()), "pos_in_sample": sum(y for xs in by.values() for _, y in xs)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--pool", default=str(ROOT / "data" / "scratch_reports" / "2026-11-20" / "de_jobs" / "all.csv"))
    s.add_argument("--per", type=int, default=75)
    s.add_argument("--seed", type=int, default=11)
    t = sub.add_parser("text")
    t.add_argument("--cache", default=str(ROOT / "data" / "cache" / "jobs"))
    t.add_argument("--out", default=str(ROOT / "data" / "scratch" / "batch3_text.jsonl"))
    c = sub.add_parser("score")
    c.add_argument("--cache", default=str(ROOT / "data" / "cache" / "jobs"))
    c.add_argument("--out", default=str(ROOT / "reports" / "exp011"))
    a = ap.parse_args(argv)
    tg = load_target("de_jobs")
    if a.cmd == "sample":
        if BATCH.exists():
            print("batch3.jsonl exists, not resampling", file=sys.stderr)
            return 3
        rows = pool_rows(Path(a.pool), read_jsonl(LABELED))
        batch, strata = sample(rows, a.per, a.seed, tg)
        BATCH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in batch), encoding="utf-8")
        STRATA.write_text(json.dumps({"seed": a.seed, "pool_file": Path(a.pool).name, "pool_after_dedup": len(rows),
                                      "strata": strata}, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(strata))
        return 0
    batch = read_jsonl(BATCH)
    if a.cmd == "text":
        tx = texts_for(batch, Path(a.cache))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for r in batch:
                f.write(json.dumps({"id": r["id"], "title": r["title"], "company": r["company"], "text": tx[r["id"]]},
                                   ensure_ascii=False) + "\n")
        print(f"{sum(bool(v) for v in tx.values())} of {len(batch)} rows have text")
        return 0
    import cv_distance as CV
    strata = json.loads(STRATA.read_text(encoding="utf-8"))["strata"]
    tx = texts_for(batch, Path(a.cache))
    prof = CV.stack_terms(json.loads((ROOT / "targets" / "de_jobs.json").read_text(encoding="utf-8")).get("skills", {}))
    cvst = CV.stack_terms({"plus": CV.cv_stack_terms((ROOT / "cv" / "master_cv.md").read_text(encoding="utf-8"))})
    res = {"strata": strata, "rows": len(batch), "with_text": sum(bool(v) for v in tx.values())}
    for f in ("esci", "esci_text"):
        res[f] = {"E+S": ht(batch, strata, f, POS, tg), "E": ht(batch, strata, f, {"E"}, tg)}
    both = [r for r in batch if r["esci"] and r["esci_text"]]
    flips = [r["id"] for r in both if (r["esci"] in POS) != (r["esci_text"] in POS)]
    res["label_change"] = {"rows": len(both), "flipped": len(flips), "rate": round(len(flips) / len(both), 3) if both else None,
                           "any_grade_change": sum(r["esci"] != r["esci_text"] for r in both), "flipped_ids": flips}
    for name, st in (("A_profile", prof), ("B_cv", cvst)):
        sc = [CV.parts(r["title"], tx[r["id"]], tg, st)["cv_score"] for r in both]
        res[f"auc_{name}"] = {f: round(CV.auc(sc, [r[f] in POS for r in both]), 3) for f in ("esci", "esci_text")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "batch3.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    L = [f"# EXP-011: batch 3, {res['rows']} unseen vacancies ({res['with_text']} with description)", "",
         "Rules v2 on the title, Horvitz-Thompson over the pool, bootstrap 95%.", "",
         "| labels | positive | P (95%) | R (95%) | labeled rows | positives in sample |", "|---|---|---|---|---|---|"]
    for f in ("esci", "esci_text"):
        for k, v in res[f].items():
            L.append(f"| {'title' if f == 'esci' else 'full text'} | {k} | {v['P']} ({v['P_ci95'][0]}–{v['P_ci95'][1]}) | "
                     f"{v['R']} ({v['R_ci95'][0]}–{v['R_ci95'][1]}) | {v['n']} | {v['pos_in_sample']} |")
    lc = res["label_change"]
    L += ["", f"Label change title → full text: {lc['flipped']} of {lc['rows']} flip positive/negative ({lc['rate']}), "
          f"{lc['any_grade_change']} change grade.", "",
          "| cv_score AUC | title labels | full-text labels |", "|---|---|---|",
          f"| A profile stack | {res['auc_A_profile']['esci']} | {res['auc_A_profile']['esci_text']} |",
          f"| B master CV stack | {res['auc_B_cv']['esci']} | {res['auc_B_cv']['esci_text']} |"]
    (out / "batch3.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
