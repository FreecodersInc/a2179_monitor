#!/usr/bin/env python3
"""T-038 CI job: build companies.csv on the runner, fetch vacancies, geo+CV filter, ensemble rank.

Run only on Leonid's runner (network + model cache). Tests call the library with fixtures.
No Telegram alerts. Output lands in --out and is committed under results/<base>/reports/.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from a2179 import ats as A  # noqa: E402
from a2179 import company_base as C  # noqa: E402
from a2179 import jobs as J  # noqa: E402
from a2179.jobs import Job  # noqa: E402

_spec = importlib.util.spec_from_file_location("geo_filter", ROOT / "scripts" / "geo_filter.py")
geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geo)


def load_geo():
    return geo


def cv_scorer(cv_path: Path = ROOT / "cv" / "master_cv.md"):
    import cv_distance as CV
    from a2179.target import load_target
    tg = load_target("de_jobs")
    if cv_path.is_file():
        terms = CV.cv_stack_terms(cv_path.read_text(encoding="utf-8"))
        stack = CV.stack_terms({"plus": terms}) if terms else CV.stack_terms(
            json.loads((ROOT / "targets" / "de_jobs.json").read_text(encoding="utf-8")).get("skills", {}))
        src = "master_cv"
    else:
        stack = CV.stack_terms(json.loads((ROOT / "targets" / "de_jobs.json").read_text(encoding="utf-8")).get("skills", {}))
        src = "de_jobs_profile"
    def score(job: Job) -> dict:
        p = CV.parts(job.title, job.text or job.snippet, tg, stack)
        return p
    return score, src


def norm_ranks(vals: list[float]) -> list[float]:
    """0-based average ranks for ties, scaled to [0, 1], same as EXP-030 (rankdata-1)/(n-1)."""
    n = len(vals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    denom = max(1, n - 1)
    return [r / denom for r in ranks]


def ensemble_weights(channels: list[str], model_ids: dict) -> dict:
    """EXP-030: equal family weights. cv_score, tinybert, and dense each get 1/groups.
    Prefix variants of one checkpoint share that model's slice of the dense family."""
    has_cv = "cv_score" in channels
    has_tb = "tinybert" in channels
    dense = [c for c in channels if c not in {"cv_score", "tinybert"}]
    ids = {model_ids.get(c, c) for c in dense}
    groups = int(has_cv) + int(has_tb) + int(bool(dense))
    groups = max(1, groups)
    w = {}
    if has_cv:
        w["cv_score"] = 1 / groups
    if has_tb:
        w["tinybert"] = 1 / groups
    for c in dense:
        variants = sum(1 for x in dense if model_ids.get(x, x) == model_ids.get(c, c))
        w[c] = 1 / (groups * max(1, len(ids)) * max(1, variants))
    return w


def apply_ensemble(rows: list[dict], raw: dict, weights: dict) -> list[dict]:
    keys = [r["url"] for r in rows]
    matrix = {}
    for ch, scores in raw.items():
        if ch not in weights:
            continue
        vals = [float(scores.get(k, 0.0) or 0.0) for k in keys]
        ranks = norm_ranks(vals)
        matrix[ch] = ranks
        for r, v, rk in zip(rows, vals, ranks):
            r["raw_" + ch] = round(float(v), 6)
            r["rank_" + ch] = round(float(rk), 6)
    for i, r in enumerate(rows):
        r["ensemble"] = round(sum(weights[ch] * matrix[ch][i] for ch in weights if ch in matrix), 6)
    return sorted(rows, key=lambda r: -r["ensemble"])


def try_dense_and_rerank(titles_by_url: dict, out: Path) -> tuple[dict, dict, dict]:
    """Best-effort EXP-030 channels. Missing models are recorded, never fatal."""
    raw, errors, model_ids = {}, {}, {}
    try:
        import sweep
        models = sweep.registry()
    except Exception as e:  # noqa: BLE001
        return raw, {"sweep": f"{type(e).__name__}: {e}"[:400]}, model_ids
    query = ""
    cv_path = ROOT / "cv" / "master_cv.md"
    if cv_path.is_file():
        md = cv_path.read_text(encoding="utf-8")
        if "## Professional Summary" in md:
            query = md.split("## Professional Summary", 1)[1].split("\n## ", 1)[0].strip()
    if not query:
        query = ("Data Engineer job: build data pipelines and a data platform with SQL, Python, Spark, "
                 "Airflow, dbt, Kafka, Snowflake, Databricks on AWS")
    urls = list(titles_by_url)
    titles = [titles_by_url[u] for u in urls]
    for model in models:
        name = model["short"]
        try:
            import exp030_cv_ensemble as E30
            scores = E30.cosine(model, query, titles)
            raw[name] = {u: float(s) for u, s in zip(urls, scores)}
            model_ids[name] = model["id"]
        except Exception as e:  # noqa: BLE001
            errors[name] = f"{type(e).__name__}: {e}"[:400]
    try:
        from sentence_transformers import CrossEncoder
        import rerank_jobs
        ce = CrossEncoder(rerank_jobs.MODEL, device="cpu", local_files_only=True)
        scores = ce.predict([(rerank_jobs.QUERY, t) for t in titles], batch_size=32, show_progress_bar=False)
        raw["tinybert"] = {u: float(s) for u, s in zip(urls, scores)}
        del ce
    except Exception as e:  # noqa: BLE001
        errors["tinybert"] = f"{type(e).__name__}: {e}"[:400]
    return raw, errors, model_ids


def write_jobs_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


JOB_FIELDS = ["title", "company", "location", "geo", "geo_reason", "cv_score", "ensemble",
              "vacancy_url", "company_url", "board_url", "ats", "slug", "source", "published_at"]


def job_row(job: Job, companies_by: dict, extra: dict) -> dict:
    key = (job.source, (job.company or "").lower())
    slug_key = (job.source, "")
    info = companies_by.get(key) or companies_by.get((job.source, job.company)) or {}
    rec = {
        "title": job.title, "company": job.company, "location": job.location,
        "vacancy_url": job.url, "company_url": info.get("company_url") or "",
        "board_url": info.get("board_url") or C.public_board_url(job.source, info.get("slug") or ""),
        "ats": job.source, "slug": info.get("slug") or "", "source": job.source,
        "published_at": job.published_at or "",
        "url": job.url,
    }
    rec.update(extra)
    return rec


def index_companies(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        if r.get("ats") and r.get("name"):
            by[(r["ats"], r["name"].lower())] = r
        if r.get("ats") and r.get("slug"):
            by[(r["ats"], r["slug"].lower())] = r
    return by


def probe_candidates(rows: list[dict], cache: Path, get_fn, budget_s: float, delay_s: float,
                     sleep_fn, clock, state_path: Path) -> dict:
    """Flatten slug variants and reuse a2179.ats.probe (resumable, stops at the first 200)."""
    cands = []
    seen = set()
    for r in rows:
        if C.sleeping(r):
            continue
        for c in C.candidates_for(r):
            k = (c["slug"], c.get("ats") or "")
            if k in seen:
                continue
            seen.add(k)
            cands.append(c)
    return A.probe(cands, cache, get_fn=get_fn, budget_s=budget_s, delay_s=delay_s,
                   sleep_fn=sleep_fn, clock=clock, state_path=state_path)


def summary_counts(rows, jobs, geo_kept, cv_kept, ranked, notes, errors, weights, src_notes, cv_src, extra=None):
    by_ats = {}
    for r in rows:
        if r.get("status") == C.STATUS_LIVE:
            by_ats[r.get("ats") or "?"] = by_ats.get(r.get("ats") or "?", 0) + 1
    d = {
        "companies": len(rows),
        "live_boards": sum(1 for r in rows if r.get("status") == C.STATUS_LIVE),
        "live_by_ats": by_ats,
        "sleeping": sum(1 for r in rows if r.get("status") == C.STATUS_SLEEP),
        "no_board": sum(1 for r in rows if r.get("status") == C.STATUS_NONE),
        "vacancies_fetched": len(jobs),
        "geo_keep_check": len(geo_kept),
        "cv_match": len(cv_kept),
        "ranked": len(ranked),
        "weights": weights,
        "cv_source": cv_src,
        "source_notes": src_notes,
        "fetch_notes": notes,
        "ensemble_errors": errors,
        "cv_path_present": (ROOT / "cv" / "master_cv.md").is_file(),
    }
    if extra:
        d.update(extra)
    return d


def run_build(out: Path, cache: Path, req: dict, get_fn=None, sleep_fn=time.sleep, clock=time.monotonic,
              now=None, collect=None, score_fn=None, dense_fn=None) -> dict:
    t0 = clock()
    budget = float(req.get("time_budget_s") or 7200)
    delay = float(req.get("delay_s") or 0.35)
    target = int(req.get("target_companies") or 1000)
    cv_min = float(req.get("cv_min") or 0.25)
    checked = J.iso(now)[:10]
    src_notes = {}
    rows, src_notes = C.collect_sources(get_fn, target=target, include_known=True)
    if req.get("pilot"):
        n = int(req.get("pilot_n") or 40)
        known = [r for r in rows if r.get("source") in {"ats_companies", "t038-seed", "ats_companies+t038-seed"}
                 or "ats_companies" in (r.get("source") or "")]
        rest = [r for r in rows if r not in known]
        rows = known + rest[:n]
        src_notes["pilot"] = {"known": len(known), "extra": min(n, len(rest))}
    left = max(30.0, budget - (clock() - t0))
    state_path = cache / "ats_probe_t038.json"
    state = probe_candidates(rows, cache, get_fn, left * 0.55, delay, sleep_fn, clock, state_path)
    rows = [C.apply_probe(r, state, checked) for r in rows]
    C.write_csv(out / "companies.csv", rows)
    live = C.live_boards(rows)
    left = max(30.0, budget - (clock() - t0))
    if collect is not None:
        jobs, notes = collect(live)
    else:
        jobs, notes = A.collect_ats(cache, companies=live, fetch="auto", get_fn=get_fn, now=now,
                                    budget_s=min(left * 0.7, left - 20), delay_s=delay,
                                    sleep_fn=sleep_fn, clock=clock, prefilter=True)
        jobs = A.dedup(jobs)
    by = index_companies(rows)
    score, cv_src = (score_fn, "injected") if score_fn else cv_scorer()
    fetched_rows = []
    geo_kept, cv_kept = [], []
    geo_counts = {"keep": 0, "check": 0, "cut": 0}
    for j in jobs:
        p = score(j)
        bucket, why = geo.verdict(j.location or "", j.title or "")
        geo_counts[bucket] = geo_counts.get(bucket, 0) + 1
        extra = {"geo": bucket, "geo_reason": why, "cv_score": p["cv_score"],
                 "skills": p.get("skills"), "stack": p.get("stack"), "domain": p.get("domain")}
        rec = job_row(j, by, extra)
        fetched_rows.append(rec)
        if bucket in {"keep", "check"}:
            geo_kept.append(rec)
            if float(rec["cv_score"]) >= cv_min:
                cv_kept.append(dict(rec))
    fetch_fields = [f for f in JOB_FIELDS if f != "ensemble"] + ["geo", "geo_reason", "cv_score"]
    # unique fields
    fetch_fields = list(dict.fromkeys(fetch_fields))
    write_jobs_csv(out / "vacancies_fetched.csv", fetched_rows, fetch_fields)
    write_jobs_csv(out / "vacancies_geo.csv", geo_kept, fetch_fields)
    write_jobs_csv(out / "vacancies_cv.csv", cv_kept, fetch_fields)
    raw = {"cv_score": {r["url"]: float(r["cv_score"]) for r in cv_kept}}
    errors, model_ids, weights = {}, {}, {"cv_score": 1.0}
    if cv_kept and not req.get("skip_ensemble"):
        dense_fn = dense_fn or try_dense_and_rerank
        more, errors, model_ids = dense_fn({r["url"]: r["title"] for r in cv_kept}, out)
        raw.update(more)
        weights = ensemble_weights(list(raw), model_ids)
    ranked = apply_ensemble([dict(r) for r in cv_kept], raw, weights) if cv_kept else []
    rank_fields = list(dict.fromkeys(JOB_FIELDS + [k for r in ranked for k in r if k.startswith(("raw_", "rank_"))]))
    write_jobs_csv(out / "vacancies_ranked.csv", ranked, rank_fields)
    if weights:
        write_jobs_csv(out / "weights.csv", [{"channel": k, "weight": v} for k, v in weights.items()],
                       ["channel", "weight"])
    extra = {"geo_counts": geo_counts, "cv_min": cv_min, "checked": checked,
             "probe_keys": len(state), "live_seeded_known": sum(1 for r in live if True),
             "budget_s": budget, "elapsed_s": round(clock() - t0, 2)}
    summary = summary_counts(rows, fetched_rows, geo_kept, cv_kept, ranked, notes, errors, weights, src_notes, cv_src, extra)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    date = checked
    live_dir = out.parent / "liveness" / date
    # stamp today's liveness so the daily request does not redo this work
    if out.parent.name == "company_base":
        live_dir.mkdir(parents=True, exist_ok=True)
        C.write_csv(live_dir / "companies.csv", rows)
        (live_dir / "summary.json").write_text(json.dumps({"from": "build", **summary}, indent=1) + "\n",
                                               encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("companies", "live_boards", "live_by_ats", "vacancies_fetched",
                                              "geo_keep_check", "cv_match", "ranked")}))
    return summary


def run_liveness(out: Path, cache: Path, req: dict, companies_path: Path | None, get_fn=None,
                 sleep_fn=time.sleep, clock=time.monotonic, now=None) -> dict:
    t0 = clock()
    budget = float(req.get("time_budget_s") or 2400)
    delay = float(req.get("delay_s") or 0.35)
    fails = int(req.get("fails_to_sleep") or C.FAILS_TO_SLEEP)
    checked = J.iso(now)[:10]
    src = companies_path or C.latest_companies_csv(ROOT)
    if src is None or not Path(src).is_file():
        summary = {"error": "no companies.csv yet", "companies": 0, "live_boards": 0}
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(summary))
        return summary
    rows = C.read_csv(Path(src))
    cands = [{"name": r["name"], "slug": r["slug"], "ats": r["ats"]}
             for r in rows if r.get("slug") and r.get("ats") in A.PARSE and r.get("status") != C.STATUS_SLEEP]
    state = A.probe(cands, cache, get_fn=get_fn, budget_s=max(30.0, budget * 0.6), delay_s=delay,
                    sleep_fn=sleep_fn, clock=clock, state_path=cache / "ats_liveness.json")
    rows = C.liveness_update(rows, state, checked, fails_to_sleep=fails)
    C.write_csv(out / "companies.csv", rows)
    live = C.live_boards(rows)
    left = max(20.0, budget - (clock() - t0))
    jobs, notes = A.collect_ats(cache, companies=live, fetch="auto", get_fn=get_fn, now=now,
                                budget_s=left, delay_s=delay, sleep_fn=sleep_fn, clock=clock, prefilter=True)
    geo_counts = {"keep": 0, "check": 0, "cut": 0}
    for j in jobs:
        b, _ = geo.verdict(j.location or "", j.title or "")
        geo_counts[b] = geo_counts.get(b, 0) + 1
    summary = {
        "mode": "liveness", "source_csv": str(src), "checked": checked,
        "companies": len(rows), "live_boards": len(live),
        "live_by_ats": {a: sum(1 for r in rows if r.get("status") == C.STATUS_LIVE and r.get("ats") == a)
                        for a in A.PARSE},
        "sleeping": sum(1 for r in rows if r.get("status") == C.STATUS_SLEEP),
        "vacancies_fetched": len(jobs), "geo_counts": geo_counts, "fetch_notes": notes,
        "elapsed_s": round(clock() - t0, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("companies", "live_boards", "vacancies_fetched", "geo_counts", "sleeping")}))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["build", "liveness"], default="build")
    ap.add_argument("--request", default="")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "a2179-ci" / "jobs"))
    ap.add_argument("--companies", default="")
    args = ap.parse_args(argv)
    req = {}
    if args.request:
        req = json.loads(Path(args.request).read_text(encoding="utf-8"))
    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if args.mode == "liveness":
        run_liveness(out, cache, req, Path(args.companies) if args.companies else None)
    else:
        run_build(out, cache, req)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
