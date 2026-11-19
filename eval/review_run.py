#!/usr/bin/env python3
"""Fetch (or reuse cache) + score both cases and write reports/<date>/<case>/.

Laptop screen (a2179_screen): scores the existing candidate / labeled listings
(with source URLs). This command does not crawl the shops or KP.

DE vacancies (de_jobs): Remotive and Remote OK official APIs, cached 24 hours,
at most 4 Remotive calls per day, jobs newer than 24 hours dropped (their terms).

Usage:
  python eval/review_run.py
  python eval/review_run.py --fetch never --jobs-cache tests/fixtures/jobs_cache
  python eval/review_run.py --cases de_jobs --fetch auto
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from a2179.jobs import collect_jobs, iso  # noqa: E402
from a2179.review import (  # noqa: E402
    case_dir, load_listings, make_matcher, row_from_job, score_rows, today, write_case_reports,
)
from a2179.target import load_target  # noqa: E402


DEFAULT_SCREEN = ROOT / "eval" / "candidates_2026-11-19.jsonl"
DEFAULT_LABELED = ROOT / "eval" / "labeled.jsonl"


def screen_listings(path: Path | None, now=None) -> tuple[list[dict], str]:
    """Prefer --listings, then the candidates dump (has URLs), then labeled.jsonl."""
    candidates = []
    note = ""
    if path:
        rows = load_listings(path, fetched_at=iso(now))
        return rows, f"listings={path} n={len(rows)}"
    if DEFAULT_SCREEN.is_file():
        # Use the candidates file mtime date as fetched_at when rows have none.
        fetched = iso(now)
        try:
            import datetime as dt
            fetched = dt.datetime.fromtimestamp(DEFAULT_SCREEN.stat().st_mtime, dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
        except OSError:
            pass
        rows = load_listings(DEFAULT_SCREEN, fetched_at=fetched)
        return rows, f"listings={DEFAULT_SCREEN.relative_to(ROOT)} n={len(rows)}"
    if DEFAULT_LABELED.is_file():
        rows = [r for r in load_listings(DEFAULT_LABELED, fetched_at=iso(now)) if r.get("url")]
        return rows, f"listings={DEFAULT_LABELED.relative_to(ROOT)} (rows with a url) n={len(rows)}"
    return [], "no screen listings file found"


def copy_tree(src: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dest / p.name)
        elif p.is_dir():
            copy_tree(p, dest / p.name)


def run(cases, out_root: Path, date: str, fetch: str, jobs_cache: Path, listings=None,
        get_fn=None, copy_to: Path = None, now=None) -> dict:
    out_root = Path(out_root)
    date = date or today(now)
    summaries = []
    for case in cases:
        target = load_target(case)
        matcher, _mode = make_matcher(target)
        notes = []
        if case == "de_jobs":
            apis = tuple(((getattr(target, "defaults", None) or {}).get("sources") or {}).get("job_apis")
                         or ("remotive", "remoteok"))
            jobs, jnotes = collect_jobs(Path(jobs_cache), sources=apis, fetch=fetch, get_fn=get_fn, now=now)
            rows = [row_from_job(j) for j in jobs]
            for n in jnotes:
                bit = f"{n['label']}: {n['n']} jobs"
                if n.get("from_cache"):
                    bit += " (cache)"
                if n.get("note"):
                    bit += f" ({n['note']})"
                if not n.get("ok") and not n.get("n"):
                    bit += " — nothing to score"
                notes.append(bit)
            if not jobs:
                notes.append("no vacancies loaded: add a cache or allow one official API call (--fetch auto)")
        elif case == "a2179_screen":
            rows, note = screen_listings(Path(listings) if listings else None, now=now)
            notes.append(note)
        else:
            rows, notes = [], [f"unknown case {case}"]
        scored = score_rows(rows, target, matcher)
        d = case_dir(out_root, date, case)
        summaries.append(write_case_reports(d, case, scored, notes=notes, date=date))
    if copy_to:
        copy_tree(out_root / date, Path(copy_to) / date)
    report = {"date": date, "out": str(out_root / date), "cases": summaries}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="a2179_screen,de_jobs",
                    help="comma-separated target names, default: both public cases")
    ap.add_argument("--out-root", default=str(ROOT / "reports"))
    ap.add_argument("--date", default="")
    ap.add_argument("--fetch", default="auto", choices=["auto", "never", "always"],
                    help="job APIs: auto uses a 24h cache, never is cache only, always still respects Remotive's daily cap")
    ap.add_argument("--jobs-cache", default=os.environ.get("A2179_JOBS_CACHE", str(ROOT / "data" / "cache" / "jobs")))
    ap.add_argument("--listings", default="", help="optional JSONL/JSON of laptop listings (default: eval/candidates_*.jsonl)")
    ap.add_argument("--copy-to", default="", help="optional second tree (ci_out/reports)")
    a = ap.parse_args(argv)
    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    return 0 if run(cases, Path(a.out_root), a.date, a.fetch, Path(a.jobs_cache),
                    listings=a.listings or None, copy_to=Path(a.copy_to) if a.copy_to else None) else 1


if __name__ == "__main__":
    sys.exit(main())
