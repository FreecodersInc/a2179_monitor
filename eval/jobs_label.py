#!/usr/bin/env python3
"""Labeled set for the vacancies case (T-019, ADR-0014 rules): a sheet to review, then merge.

sheet: reads every reports/<date>/de_jobs/all.csv (the CI review step commits them), drops
       duplicates (url, else normalised title + company) and rows already in the labeled set,
       pre-labels each row by the profile rules only (no model, no LLM) with the reason, and
       writes eval/de_jobs/to_label.csv in review order: rules and matcher disagree first,
       then hard negatives (matcher says no but scores high), then the rest.
merge: appends rows with reviewed = 1 and esci in E/S/C/I to eval/de_jobs/labeled.jsonl.
       label = 1 for E (fits) and S (fits with a stretch: adjacent role or seniority),
       0 for C (same domain, not it: DS, analyst) and I (miss). New ids, nothing deleted.

Then CI measures the case by itself (eval[de_jobs], sweep[de_jobs]), no CI change.

Usage:
  python eval/jobs_label.py sheet [--reports reports] [--out eval/de_jobs/to_label.csv]
  python eval/jobs_label.py merge [--sheet eval/de_jobs/to_label.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from a2179.normalize import normalize  # noqa: E402
from a2179.rules import classify  # noqa: E402
from a2179.jobs import fix_text  # noqa: E402
from a2179.target import eval_data_path, load_target  # noqa: E402

CASE = "de_jobs"
ESCI = {"E": 1, "S": 1, "C": 0, "I": 0}
COLS = ["cand_id", "reviewed", "esci", "pre_esci", "title", "company", "location", "rule_match", "rule_model",
        "matcher", "score", "reason", "url", "source", "fetched_at", "note", "labeler", "unsure", "stratum"]
# labeler: who set esci. "bot" = proposed by the assistant from title, company and location, not an
# independent human label (eval/de_jobs/README.md). It is carried into labeled.jsonl so EXP records can
# split or drop those rows. unsure = 1 goes in as review_unsure.


def key(r: dict) -> str:
    return (r.get("url") or "").strip() or f"{normalize(r.get('title') or '')}|{normalize(r.get('company') or '')}"


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def labeled_keys(path: Path) -> set:
    if not path.is_file():
        return set()
    return {key(json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def pre_esci(res) -> str:
    """Transparent guess from the profile tiers, the human decides."""
    if not res.match:
        return "I" if res.model == "none" else "C"
    return {"strong": "E", "lead": "S", "ai_platform": "S", "weak": "S"}.get(res.model, "C")


def stratum(r: dict) -> str:
    """Why a row is in the batch. matcher: the matcher says match. disagree: profile rules say match,
    matcher says no. hard_neg: matcher says no with score >= 0.3. random: the rest."""
    if r["matcher"]:
        return "matcher"
    if r["rule_match"]:
        return "disagree"
    return "hard_neg" if float(r["score"] or 0) >= 0.3 else "random"


def sample(rows: list[dict], max_rows: int, seed: int) -> tuple[list[dict], dict]:
    """Keep every matcher, disagree and hard_neg row, fill up to max_rows with a uniform random draw
    of the rest (seeded). Returns the kept rows and per stratum population / kept counts, so an EXP
    record can reweight the random stratum back to the pool."""
    import random
    pop = {}
    for r in rows:
        pop.setdefault(r["stratum"], []).append(r)
    keep = [r for r in rows if r["stratum"] != "random"]
    rest = pop.get("random", [])
    k = max(0, min(len(rest), max_rows - len(keep))) if max_rows else len(rest)
    keep += random.Random(seed).sample(rest, k)
    kept = {}
    for r in keep:
        kept[r["stratum"]] = kept.get(r["stratum"], 0) + 1
    return keep, {s: {"pool": len(v), "kept": kept.get(s, 0)} for s, v in sorted(pop.items())}


def sheet(reports: Path, out: Path, target=CASE, max_rows: int = 0, seed: int = 0) -> list[dict]:
    tg = load_target(target)
    seen = labeled_keys(eval_data_path(tg))
    rows, keys = [], set()
    for p in sorted(reports.glob(f"????-??-??/{CASE}/all.csv")):
        for r in read_csv(p):
            k = key(r)
            if not r.get("title") or k in keys or k in seen:
                continue
            keys.add(k)
            res = classify(r["title"], "", target=tg)
            matcher = r.get("decision") == "match"
            rows.append({**{c: r.get(c, "") for c in ("title", "company", "location", "url", "source",
                                                       "fetched_at", "score")},
                         "rule_match": int(res.match), "rule_model": res.model, "matcher": int(matcher),
                         "reason": "; ".join(res.reasons), "pre_esci": pre_esci(res), "esci": "",
                         "reviewed": 0, "note": ""})
    for r in rows:
        r["stratum"] = stratum(r)
    strata = None
    if max_rows:
        rows, strata = sample(rows, max_rows, seed)

    def order(r):
        s = float(r["score"] or 0)
        disagree = r["rule_match"] != r["matcher"]
        hard_neg = not r["matcher"] and s >= 0.3
        return (0 if disagree else 1 if hard_neg else 2, -s, r["title"])
    rows.sort(key=order)
    for i, r in enumerate(rows, 1):
        r["cand_id"] = i
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    if strata is not None:
        out.with_name(out.stem + "_strata.json").write_text(
            json.dumps({"seed": seed, "max_rows": max_rows, "strata": strata}, indent=2) + "\n", encoding="utf-8")
    return rows


def jsonl_line(r: dict) -> str:
    """One JSON line that every reader's splitlines() keeps whole: escape the separators it splits on."""
    s = json.dumps(r, ensure_ascii=False)
    for c in ("\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"):
        s = s.replace(c, f"\\u{ord(c):04x}")
    return s + "\n"


def merge(sheet_path: Path, target=CASE) -> list[dict]:
    tg = load_target(target)
    dest = eval_data_path(tg)
    seen = labeled_keys(dest)
    old = [json.loads(l) for l in dest.read_text(encoding="utf-8").splitlines() if l.strip()] if dest.is_file() else []
    nid = max((r["id"] for r in old), default=0)
    new = []
    for r in read_csv(sheet_path):
        e = (r.get("esci") or "").strip().upper()
        if str(r.get("reviewed", "")).strip() not in ("1", "true", "yes") or e not in ESCI or key(r) in seen:
            continue
        nid += 1
        seen.add(key(r))
        new.append({"id": nid, "title": fix_text(r["title"]), "snippet": "", "label": ESCI[e], "esci": e, "lang": "en",
                    "kind": f"esci_{e}", "source": f"real:{r.get('source') or 'jobs'}",
                    "company": fix_text(r.get("company", "")), "location": fix_text(r.get("location", "")), "url": r.get("url", ""), "fetched_at": r.get("fetched_at", ""),
                    **({"note": r["note"]} if r.get("note") else {}),
                    **({"labeler": r["labeler"].strip()} if (r.get("labeler") or "").strip() else {}),
                    **({"stratum": r["stratum"].strip()} if (r.get("stratum") or "").strip() else {}),
                    **({"review_unsure": True} if str(r.get("unsure", "")).strip() in ("1", "true", "yes") else {})})
    if new:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8") as f:
            for r in new:
                f.write(jsonl_line(r))
    return new


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--reports", default=str(ROOT / "reports"))
    s.add_argument("--out", default=str(ROOT / "eval" / CASE / "to_label.csv"))
    s.add_argument("--max", type=int, default=0, help="cap the batch: all matcher/disagree/hard_neg rows plus a "
                   "seeded random draw of the rest (writes <out>_strata.json)")
    s.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--sheet", default=str(ROOT / "eval" / CASE / "to_label.csv"))
    a = ap.parse_args(argv)
    if a.cmd == "sheet":
        rows = sheet(Path(a.reports), Path(a.out), max_rows=a.max, seed=a.seed)
        print(f"{len(rows)} rows -> {a.out}")
        print("next: put reviewed 1 and esci E / S / C / I (pre_esci is only a hint), then "
              "python eval/jobs_label.py merge")
        return 0 if rows else 3
    new = merge(Path(a.sheet))
    print(f"+{len(new)} rows -> {eval_data_path(CASE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
