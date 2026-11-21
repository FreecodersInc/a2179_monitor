"""Build results/<UTC date>_<shortsha>.json from a CI run and (optionally) commit + push it.

Input: the folder written by ci/run_experiments.py (steps.json, eval/eval_report.json,
sweep/sweep.json). Output: one JSON per run with the commit, the runner, every command
with exit code and duration, and the metrics. The commit message carries [skip ci] and the
workflow ignores results/**, so pushing a result never starts another run.

Called by ci/run.sh with the system python: stdlib only, so it still writes a result when
the venv setup broke. Git auth comes from the environment the workflow sets (GIT_CONFIG_*),
nothing is stored. Push: git push to the branch, on rejection `git pull --rebase` and try again (a few times,
with a short growing pause). Only git and the token the checkout left behind are used.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = 3  # 2: secrets (present / absent), skipped steps, values scrubbed. 3: metrics.ranking, advisory steps
sys.path.insert(0, str(ROOT / "ci"))
import redact  # noqa: E402
EVAL_KEYS = ("precision", "recall", "f1", "tp", "fp", "fn", "tn", "thresholds", "twofold", "ms_per_item", "note")
SWEEP_KEYS = ("variant", "hybrid", "t", "w", "lam", "P", "R", "F1", "tp", "fp", "fn", "tn", "cv2", "cv5", "real",
              "note")


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def git(*args, cwd=ROOT, check=True):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip() or p.stdout.strip()}")
    return p


def commit_info(env=os.environ, cwd=ROOT):
    sha = env.get("GITHUB_SHA") or git("rev-parse", "HEAD", cwd=cwd, check=False).stdout.strip()
    msg = git("log", "-1", "--format=%s", sha, cwd=cwd, check=False).stdout.strip() if sha else ""
    c = {"sha": sha, "short": sha[:7], "subject": msg, "ref": env.get("GITHUB_REF"),
         "repository": env.get("GITHUB_REPOSITORY"), "run_id": env.get("GITHUB_RUN_ID"),
         "run_attempt": env.get("GITHUB_RUN_ATTEMPT"), "workflow": env.get("GITHUB_WORKFLOW"),
         "actor": env.get("GITHUB_ACTOR")}
    if env.get("GITHUB_SERVER_URL") and env.get("GITHUB_REPOSITORY") and env.get("GITHUB_RUN_ID"):
        c["run_url"] = f"{env['GITHUB_SERVER_URL']}/{env['GITHUB_REPOSITORY']}/actions/runs/{env['GITHUB_RUN_ID']}"
    return c


def eval_metrics(rep):
    if not rep:
        return None
    return {"dataset": {k: rep.get("dataset", {}).get(k) for k in ("n", "positives", "negatives", "by_source")},
            "embedder": rep.get("embedder"),
            "modes": {m: {k: v for k, v in d.items() if k in EVAL_KEYS} for m, d in rep.get("modes", {}).items()},
            "errors": {m: [e[0] for e in errs] for m, errs in rep.get("errors", {}).items()}}


def sweep_metrics(sw):
    if not sw:
        return None
    V = [{k: v for k, v in m.items() if k in SWEEP_KEYS} | {"errors": m.get("errors", [])[:20]}
         for m in sw.get("variants", [])]
    scored = [v for v in V if "F1" in v and v.get("hybrid")]
    best = max(scored, key=lambda v: (v["R"] == 1.0, v["F1"], -v["fp"]), default=None)
    return {"n": sw.get("n"), "dense_source": sw.get("dense"), "tfidf_2d": sw.get("tfidf_2d"),
            "best_hybrid": best, "variants": V}


def review_metrics(reports_root: Path):
    """Per-case n / n_matches from summary.json written by eval/review_run.py."""
    if not reports_root.is_dir():
        return None
    out = {}
    for p in sorted(reports_root.glob("????-??-??/*/summary.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        case = s.get("case") or p.parent.name
        out[case] = {"n": s.get("n"), "n_matches": s.get("n_matches"), "date": s.get("date"),
                     "notes": s.get("notes") or []}
    return out or None



def target_metrics(ci_out: Path, steps):
    """metrics.targets.<name>: eval + sweep of every non-default target (T-004), or why it was skipped."""
    out = {}
    for s in steps:
        n = s.get("name") or ""
        if n.startswith("eval[") and n.endswith("]") and s.get("skipped"):
            out[n[5:-1]] = {"skipped": s["skipped"]}
    root = ci_out / "targets"
    if root.is_dir():
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            ev, sw = _load(d / "eval" / "eval_report.json"), _load(d / "sweep" / "sweep.json")
            if ev or sw:
                out[d.name] = {"eval": eval_metrics(ev), "sweep": sweep_metrics(sw)}
    return out or None


def build(ci_out: Path, env=os.environ, now=None, cwd=ROOT):
    now = now or dt.datetime.now(dt.timezone.utc)
    steps = _load(ci_out / "steps.json") or {"env": {}, "dense": {}, "steps": []}
    setup = _load(ci_out / "setup.json")  # written by ci/run.sh
    st = ([{**setup, "stderr_tail": setup.get("log_tail")}] if setup else []) + steps.get("steps", [])
    res = {
        "schema": SCHEMA,
        "generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit_info(env, cwd),
        "runner": {"name": env.get("RUNNER_NAME"), "os": env.get("RUNNER_OS"), "arch": env.get("RUNNER_ARCH"),
                   **(steps.get("env") or {})},
        "dense": steps.get("dense") or {},
        "secrets": steps.get("secrets") or (_load(ci_out / "secrets.json") or redact.presence(env)),
        "steps": [{k: s.get(k) for k in ("name", "cmd", "exit_code", "signal", "timed_out", "duration_s", "ok",
                                         "fallback", "skipped", "advisory", "stderr_tail")} for s in st],
        "metrics": {"eval": eval_metrics(_load(ci_out / "eval" / "eval_report.json")),
                    "sweep": sweep_metrics(_load(ci_out / "sweep" / "sweep.json")),
                    "ranking": (_load(ci_out / PLOTS / "plots.json") or {}).get("ranking"),
                    "review": review_metrics(ci_out / "reports"),
                    "targets": target_metrics(ci_out, st)},
    }
    for s in res["steps"]:  # keep only a short tail of stderr for failed steps
        s["stderr_tail"] = (s.get("stderr_tail") or "")[-1500:] if not s.get("ok") else None
    tests = [s for s in st if s.get("name") == "tests"]
    last = {}
    for s in st:  # a rerun (fallback) of a step replaces its first try
        last[s.get("name")] = s
    # check_embedder only decides whether dense runs, its failure is reported under "dense",
    # advisory steps (plots) are reported but never make the run red
    res["ok"] = bool(last) and all(s.get("ok") for k, s in last.items()
                                   if k != "check_embedder" and not s.get("advisory"))
    res["tests_ok"] = tests[-1].get("ok") if tests else None
    active, source = Path(cwd) / ".github" / "workflows" / "experiments.yml", Path(cwd) / "ci" / "experiments.yml"
    res["workflow_in_sync"] = (active.read_bytes() == source.read_bytes()) if active.is_file() and source.is_file() else None
    res["total_duration_s"] = round(sum(s.get("duration_s") or 0 for s in st), 2)
    return redact.scrub_obj(res, env)


def target_path(results_dir: Path, res: dict) -> Path:
    base = f"{res['generated_utc'][:10]}_{res['commit']['short'] or 'nosha'}"
    p = results_dir / f"{base}.json"
    n = 2
    while p.exists():
        p = results_dir / f"{base}_{n}.json"
        n += 1
    return p


def write(results_dir: Path, res: dict) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    p = target_path(results_dir, res)
    p.write_text(redact.scrub(json.dumps(res, ensure_ascii=False, indent=1)) + "\n", encoding="utf-8")
    return p


REPORTS = ("eval/eval_report.json", "eval/eval_report.md", "sweep/sweep.tables.md", "sweep/dense_scores.jsonl",
           "setup.log")


TARGET_REPORTS = ("eval_report.json", "eval_report.md", "sweep.tables.md", "dense_scores.jsonl")
PLOTS = "plots"   # ci_out/plots/: *.png, plots.md, plots.json from eval/plots.py


def attach_reports(ci_out: Path, result: Path) -> Path | None:
    """Copy the human readable reports (and dense scores, if computed) next to the result JSON:
    results/<base>/. The big sweep.json is left out (recomputable from dense_scores.jsonl).
    Charts from eval/plots.py land in the same folder and are embedded at the end of
    sweep.tables.md, so the tables and their charts read as one report."""
    files = [ci_out / r for r in REPORTS if (ci_out / r).is_file()]
    files += sorted((ci_out / "sweep").glob("dense_scores_*.jsonl")) if (ci_out / "sweep").is_dir() else []
    pdir = ci_out / PLOTS
    pngs = sorted(pdir.glob("*.png")) if pdir.is_dir() else []
    pmd = pdir / "plots.md"
    review = (ci_out / "reports").is_dir() and any(f.is_file() for f in (ci_out / "reports").rglob("*"))
    tfiles = [p for p in sorted((ci_out / "targets").rglob("*")) if p.is_file() and (p.name in TARGET_REPORTS
                                                                      or (p.name.startswith("dense_scores_") and p.suffix == ".jsonl"))] \
        if (ci_out / "targets").is_dir() else []
    if not files and not pngs and not review and not tfiles:
        return None
    d = result.with_suffix("")
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        (d / f.name).write_text(redact.scrub(f.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    for f in pngs:   # pixels only, rendered from scores and labels, no environment text in them
        shutil.copyfile(f, d / f.name)
    for name in ("plots.md", "plots.json"):
        if (pdir / name).is_file():
            (d / name).write_text(redact.scrub((pdir / name).read_text(encoding="utf-8", errors="replace")),
                                  encoding="utf-8")
    if pmd.is_file() and (d / "sweep.tables.md").is_file():
        with (d / "sweep.tables.md").open("a", encoding="utf-8") as fh:
            fh.write("\n" + redact.scrub(pmd.read_text(encoding="utf-8", errors="replace")))
    if pmd.is_file() and (d / "eval_report.md").is_file():
        with (d / "eval_report.md").open("a", encoding="utf-8") as fh:
            fh.write("\nCharts and ranking metrics (recall@k, precision at recall 1.0 / 0.95, nDCG): "
                     "[plots.md](plots.md), also embedded at the end of [sweep.tables.md](sweep.tables.md)\n")
    for f in tfiles:   # results/<base>/targets/<name>/eval_report.md, sweep.tables.md, dense scores
        dest = d / "targets" / f.relative_to(ci_out / "targets").parts[0] / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(redact.scrub(f.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    # two-case manual review files: ci_out/reports/<date>/<case>/matches.csv
    src = ci_out / "reports"
    if src.is_dir():
        for f in src.rglob("*"):
            if f.is_file():
                dest = d / "reports" / f.relative_to(src)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(redact.scrub(f.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    return d


def commit_and_push(path: Path, branch: str, cwd=ROOT, remote="origin", tries=5, pause=2.0, push=True,
                    extra=()):
    git("config", "user.name", "a2179-ci", cwd=cwd)
    git("config", "user.email", "a2179-ci@users.noreply.github.com", cwd=cwd)
    root = Path(cwd).resolve()
    rels = [str(x.resolve().relative_to(root)) for x in (path, *extra) if x]
    git("add", "--", *rels, cwd=cwd)
    git("commit", "-m", f"results: {path.stem} [skip ci]", "--", *rels, cwd=cwd)
    if not push:
        return 0
    for i in range(1, tries + 1):
        p = git("push", remote, f"HEAD:{branch}", cwd=cwd, check=False)
        if p.returncode == 0:
            print(f"pushed {' '.join(rels)} to {remote}/{branch} (try {i})")
            return i
        print(f"push rejected (try {i}): {p.stderr.strip()[-300:]}", file=sys.stderr)
        r = git("pull", "--rebase", remote, branch, cwd=cwd, check=False)
        if r.returncode:
            git("rebase", "--abort", cwd=cwd, check=False)
            raise RuntimeError(f"rebase failed: {r.stderr.strip()[-300:]}")
        time.sleep(pause * i)
    raise RuntimeError(f"push failed after {tries} tries")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci-out", default=str(ROOT / "ci_out"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--attach-reports", action="store_true", help="copy reports into results/<base>/")
    a = ap.parse_args(argv)
    res = build(Path(a.ci_out))
    p = write(Path(a.results_dir), res)
    extra = []
    attached = attach_reports(Path(a.ci_out), p) if a.attach_reports else None
    if attached:
        extra.append(attached)
    # copy CI review reports into reports/<YYYY-MM-DD>/ so they live in the repo too
    if a.commit:
        src = Path(a.ci_out) / "reports"
        if src.is_dir():
            for dated in src.glob("????-??-??"):
                if dated.is_dir() and any(dated.glob("*/matches.csv")):
                    dest = ROOT / "reports" / dated.name
                    for f in dated.rglob("*"):
                        if f.is_file():
                            q = dest / f.relative_to(dated)
                            q.parent.mkdir(parents=True, exist_ok=True)
                            q.write_text(redact.scrub(f.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
                    extra.append(dest)
    print(f"results -> {p}" + (f" + {', '.join(str(x) for x in extra)}" if extra else ""))
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"results_file={p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}\n")
    if a.commit:
        commit_and_push(p, a.branch, push=a.push, extra=extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
