#!/usr/bin/env python3
"""Which CI steps a push needs (docs/ci.md, ADR-0026). stdlib only, prints shell `export` lines.

base = the newest earlier commit that already has a results/<date>_<short sha>.json (the last one CI
really tested), else HEAD~1. changed = git diff --name-only base HEAD.

  skip      every changed path is docs: docs/, experiments/, work/, reports/, results/, *.md.
            run.sh exits 0 before the venv, nothing is committed.
  full      shared code changed: a2179/, ci/, monitor.py, requirements*, targets/, eval/run_eval.py,
            eval/sweep.py, eval/plots.py, eval/embedders.json, eval/labeled.jsonl, tests/helpers.py,
            tests/__init__.py, tests/fixtures/. Full regression, every experiment step.
  targeted  anything else: lint + only the tests for the touched files + only the experiment the
            commit touched (e.g. eval/de_jobs/ -> the de_jobs steps).
A2179_CI_MODE=full (repo variable) forces a full run. Nothing changed or git missing -> full.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = re.compile(r"^(docs/|experiments/|work/|reports/|results/)|\.md$")
SHARED = re.compile(r"^(a2179/|ci/|monitor\.py$|requirements|targets/|tests/helpers\.py$|tests/__init__\.py$|"
                    r"tests/fixtures/|eval/(run_eval|sweep|plots)\.py$|eval/embedders\.json$|eval/labeled\.jsonl$)")
# experiment steps a touched path asks for (targeted mode)
STEPS = [(re.compile(r"^eval/de_jobs/|^eval/jobs_label\.py$"), {"targets", "review"}),
         (re.compile(r"^eval/(review_run)\.py$"), {"review"}),
         (re.compile(r"^eval/(candidates_.*|enrich\.py)"), {"eval", "sweep", "plots"})]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def base_commit() -> str:
    tested = {p.stem.rsplit("_", 1)[-1] for p in (ROOT / "results").glob("*_*.json")}
    try:
        for sha in git("rev-list", "--max-count=200", "HEAD~1").split():
            if sha[:7] in tested:
                return sha
        return git("rev-parse", "HEAD~1").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def tests_for(paths) -> list[str]:
    mods = set()
    for p in paths:
        if p.startswith("tests/test_") and p.endswith(".py"):
            mods.add("tests." + Path(p).stem)
            continue
        stem = Path(p).stem
        for t in (ROOT / "tests").glob(f"test_{stem}*.py"):
            mods.add("tests." + t.stem)
        for t in (ROOT / "tests").glob("test_*.py"):   # tests that import the module by name
            try:
                if re.search(rf"\b(import|from)\s+[\w.]*\b{re.escape(stem)}\b", t.read_text(encoding="utf-8")):
                    mods.add("tests." + t.stem)
            except OSError:
                pass
    return sorted(mods)


def decide(changed: list[str], forced: str = "") -> dict:
    if forced == "full" or not changed:
        return {"mode": "full", "tests": [], "steps": [], "why": "forced" if forced else "no diff, run everything"}
    code = [p for p in changed if not DOCS.search(p)]
    if not code:
        return {"mode": "skip", "tests": [], "steps": [], "why": "docs only"}
    shared = [p for p in code if SHARED.search(p)]
    if shared:
        return {"mode": "full", "tests": [], "steps": [], "why": "shared code: " + ", ".join(shared[:5])}
    steps = set()
    for p in code:
        for rx, s in STEPS:
            if rx.search(p):
                steps |= s
    return {"mode": "targeted", "tests": tests_for(code), "steps": sorted(steps), "why": ", ".join(code[:8])}


def main(argv=None):
    forced = (os.environ.get("A2179_CI_MODE") or "").strip().lower()
    base = base_commit()
    try:
        changed = [l for l in git("diff", "--name-only", base, "HEAD").splitlines() if l] if base else []
    except (subprocess.CalledProcessError, FileNotFoundError):
        changed = []
    d = decide(changed, forced)
    print(f"export A2179_CI_SELECT_MODE={d['mode']}")
    print(f"export A2179_CI_SELECT_TESTS='{' '.join(d['tests'])}'")
    print(f"export A2179_CI_SELECT_STEPS='{','.join(d['steps'])}'")
    print(f"export A2179_CI_SELECT_BASE='{base[:12]}'")
    why = d["why"].replace("'", "")
    print(f"export A2179_CI_SELECT_WHY='{why}'")
    print(f"echo 'ci select: {d['mode']} (base {base[:7] or 'none'}): {why}' >&2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
