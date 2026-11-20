#!/usr/bin/env python3
"""How close is a vacancy to the candidate's experience (H-35, T-024, EXP-010).

Reference: the public skills profile `targets/de_jobs.json` (skills.required, skills.plus,
skills.groups, written from the approved CV, no personal data). The full CV stays in private/
and is never read here. Each opening gets four parts in [0, 1], fixed before any run:
  skills     share of the 3 required items found in title + description
             (SQL / Python or Scala / one of Spark, Airflow, ETL, dbt)
  stack      profile terms (skills.plus + every skills.groups term) found, saturating at 15
  domain     role family of the title from the profile rules: strong DE 1.0, adjacent roles
             (ai_platform, ml_sre_infra, devops, lead) 0.8, data_scientist 0.5, weak 0.3, none 0
  seniority  level of the title: senior / staff / lead / principal / plain 1.0 (0.9 for plain),
             manager / director / head / vp 0.4, intern / junior / graduate 0.2
  cv_score = 0.30 skills + 0.30 stack + 0.25 domain + 0.15 seniority
Labels (E/S/C/I) are only the check, they never set the weights.

Text comes from the job cache (`data/cache/jobs`, fetch never): ATS jobs keep the plain
description (`text`), Remotive and Remote OK are parsed again from their cached bodies.

Usage:
  python eval/cv_distance.py [--labeled eval/de_jobs/labeled.jsonl] [--out reports/exp010]
Writes <out>/dense_scores_cv.jsonl (id, pos = cv_score, neg = 0, readable by eval/nested_cv.py
as channel dense:cv), <out>/cv_parts.jsonl (parts per id, no text) and <out>/cv_distance.json/.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from a2179.rules import classify  # noqa: E402
from a2179.target import load_target  # noqa: E402

W = {"skills": 0.30, "stack": 0.30, "domain": 0.25, "seniority": 0.15}
DOMAIN = {"strong": 1.0, "ai_platform": 0.8, "ml_sre_infra": 0.8, "devops": 0.8, "lead": 0.8,
          "data_scientist": 0.5, "weak": 0.3, "none": 0.0}
STACK_SAT = 15
REQUIRED = [r"\bsql\b", r"\bpython\b|\bscala\b", r"\bspark\b|\bairflow\b|\betl\b|\belt\b|\bdbt\b"]
LOW = re.compile(r"\b(intern|internship|junior|graduate|new\s+grad|entry\s+level)\b", re.I)
MGMT = re.compile(r"\b(manager|director|head\s+of|vp|vice\s+president)\b", re.I)
SENIOR = re.compile(r"\b(senior|sr\.?|staff|lead|principal)\b", re.I)


def term_rx(term: str) -> re.Pattern:
    t = re.escape(term.lower()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){t}(?![a-z0-9])", re.I)


def cv_stack_terms(md: str) -> list[str]:
    """Variant B (EXP-010): stack terms from the "Technical and Delivery Skills" section of the master CV."""
    a = md.index("## Technical and Delivery Skills")
    b = md.index("\n## ", a + 5)
    out = []
    for line in md[a:b].splitlines():
        if not line.startswith("- "):
            continue
        body = line.split(":**", 1)[-1]
        for t in re.split(r"[,;]| and |/", body):
            t = re.sub(r"\(.*?\)|\.$", "", t).strip(" .")
            if 1 < len(t) <= 30:
                out.append(t)
    return out


def stack_terms(skills: dict) -> list[re.Pattern]:
    sk = skills or {}
    terms = list(sk.get("plus", []))
    for v in sk.get("groups", {}).values():
        terms += v
    seen, out = set(), []
    for t in terms:
        k = t.lower()
        if k not in seen and len(k) > 1:
            seen.add(k)
            out.append(term_rx(t))
    return out


def seniority(title: str) -> float:
    if LOW.search(title):
        return 0.2
    if MGMT.search(title):
        return 0.4
    return 1.0 if SENIOR.search(title) else 0.9


def parts(title: str, text: str, tg, stack) -> dict:
    full = f"{title} {text}"
    skills = sum(1 for rx in REQUIRED if re.search(rx, full, re.I)) / len(REQUIRED)
    st = min(1.0, sum(1 for rx in stack if rx.search(full)) / STACK_SAT)
    dom = DOMAIN.get(classify(title, "", target=tg).model, 0.0)
    p = {"skills": round(skills, 3), "stack": round(st, 3), "domain": dom, "seniority": seniority(title)}
    p["cv_score"] = round(sum(W[k] * p[k] for k in W), 4)
    return p


def auc(scores, y) -> float:
    pos = [s for s, l in zip(scores, y) if l]
    neg = [s for s, l in zip(scores, y) if not l]
    if not pos or not neg:
        return float("nan")
    win = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return win / (len(pos) * len(neg))


def spearman(a, b) -> float:
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def job_texts(cache: Path) -> dict:
    from a2179.jobs import collect_jobs
    jobs, _ = collect_jobs(cache, sources=("remotive", "remoteok", "ats"), fetch="never")
    return {j.url: j.text or j.snippet for j in jobs}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default=str(ROOT / "eval" / "de_jobs" / "labeled.jsonl"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache" / "jobs"))
    ap.add_argument("--target", default="de_jobs")
    ap.add_argument("--out", default=str(ROOT / "reports" / "exp010"))
    ap.add_argument("--stack-from", default="", help="variant B: stack terms from this CV markdown (cv/master_cv.md)")
    a = ap.parse_args(argv)
    tg = load_target(a.target)
    if a.stack_from:
        stack = stack_terms({"plus": cv_stack_terms(Path(a.stack_from).read_text(encoding="utf-8"))})
    else:
        stack = stack_terms(json.loads((ROOT / "targets" / f"{a.target}.json").read_text(encoding="utf-8")).get("skills", {}))
    rows = [json.loads(l) for l in Path(a.labeled).read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = job_texts(Path(a.cache))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P, have = [], 0
    for r in rows:
        t = texts.get(r.get("url", ""), "")
        have += bool(t)
        P.append({"id": r["id"], "has_text": bool(t), **parts(r["title"], t, tg, stack)})
    with (out / "dense_scores_cv.jsonl").open("w", encoding="utf-8") as f:
        for p in P:
            f.write(json.dumps({"id": p["id"], "pos": p["cv_score"], "neg": 0.0}) + "\n")
    with (out / "cv_parts.jsonl").open("w", encoding="utf-8") as f:
        for p in P:
            f.write(json.dumps(p) + "\n")
    y = [bool(r["label"]) for r in rows]
    grade = [{"E": 3, "S": 2, "C": 1, "I": 0}[r["esci"]] for r in rows]
    res = {"stack_from": a.stack_from or "targets profile", "stack_terms": len(stack), "rows": len(rows), "with_text": have, "weights": W,
           "auc": {k: round(auc([p[k] for p in P], y), 3) for k in (*W, "cv_score")},
           "auc_with_text_only": round(auc([p["cv_score"] for p in P if p["has_text"]],
                                           [l for l, p in zip(y, P) if p["has_text"]]), 3),
           "spearman_grade": round(spearman([p["cv_score"] for p in P], grade), 3)}
    (out / "cv_distance.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    L = [f"# EXP-010: CV distance score on {res['rows']} labeled vacancies ({have} with description text)", "",
         "| part | AUC (E+S vs C+I) |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in res["auc"].items()]
    L += ["", f"AUC on rows with text only: {res['auc_with_text_only']}. Spearman cv_score vs grade E>S>C>I: {res['spearman_grade']}."]
    (out / "cv_distance.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
