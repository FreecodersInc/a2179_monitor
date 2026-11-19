"""Build results/<UTC date>_<shortsha>.json from a CI run and (optionally) commit + push it.

Input: the folder written by ci/run_experiments.py (steps.json, eval/eval_report.json,
sweep/sweep.json). Output: one JSON per run with the commit, the runner, every command
with exit code and duration, and the metrics. The commit message carries [skip ci] and the
workflow ignores results/**, so pushing a result never starts another run.

Push: git push to the branch, on rejection `git pull --rebase` and try again (a few times,
with a short growing pause). Only git and the token the checkout left behind are used.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = 1
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


def build(ci_out: Path, env=os.environ, now=None, cwd=ROOT):
    now = now or dt.datetime.now(dt.timezone.utc)
    steps = _load(ci_out / "steps.json") or {"env": {}, "dense": {}, "steps": []}
    st = steps.get("steps", [])
    res = {
        "schema": SCHEMA,
        "generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit_info(env, cwd),
        "runner": {"name": env.get("RUNNER_NAME"), "os": env.get("RUNNER_OS"), "arch": env.get("RUNNER_ARCH"),
                   **(steps.get("env") or {})},
        "dense": steps.get("dense") or {},
        "steps": [{k: s.get(k) for k in ("name", "cmd", "exit_code", "signal", "timed_out", "duration_s", "ok",
                                         "fallback", "stderr_tail")} for s in st],
        "metrics": {"eval": eval_metrics(_load(ci_out / "eval" / "eval_report.json")),
                    "sweep": sweep_metrics(_load(ci_out / "sweep" / "sweep.json"))},
    }
    for s in res["steps"]:  # keep only a short tail of stderr for failed steps
        s["stderr_tail"] = (s.get("stderr_tail") or "")[-1500:] if not s.get("ok") else None
    tests = [s for s in st if s.get("name") == "tests"]
    last = {}
    for s in st:  # a rerun (fallback) of a step replaces its first try
        last[s.get("name")] = s
    # check_embedder only decides whether dense runs, its failure is reported under "dense"
    res["ok"] = bool(last) and all(s.get("ok") for k, s in last.items() if k != "check_embedder")
    res["tests_ok"] = tests[-1].get("ok") if tests else None
    res["total_duration_s"] = round(sum(s.get("duration_s") or 0 for s in st), 2)
    return res


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
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return p


def commit_and_push(path: Path, branch: str, cwd=ROOT, remote="origin", tries=5, pause=2.0, push=True):
    git("config", "user.name", "a2179-ci", cwd=cwd)
    git("config", "user.email", "a2179-ci@users.noreply.github.com", cwd=cwd)
    rel = path.resolve().relative_to(Path(cwd).resolve())
    git("add", "--", str(rel), cwd=cwd)
    git("commit", "-m", f"results: {path.stem} [skip ci]", "--", str(rel), cwd=cwd)
    if not push:
        return 0
    for i in range(1, tries + 1):
        p = git("push", remote, f"HEAD:{branch}", cwd=cwd, check=False)
        if p.returncode == 0:
            print(f"pushed {rel} to {remote}/{branch} (try {i})")
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
    a = ap.parse_args(argv)
    res = build(Path(a.ci_out))
    p = write(Path(a.results_dir), res)
    print(f"results -> {p}")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"results_file={p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}\n")
    if a.commit:
        commit_and_push(p, a.branch, push=a.push)
    return 0


if __name__ == "__main__":
    sys.exit(main())
