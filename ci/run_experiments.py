"""CI step runner: tests, embedder check, eval and sweep, each in its own process.

Every step records command, exit code, duration and the tail of its output into
<out>/steps.json, so ci/write_results.py can build one results file even when a step
crashed. The dense (embedder) steps run only when torch + sentence-transformers import
and `monitor.py --check-embedder` passes (that check loads the model in a child process
first, see ADR-0005). If a dense step still fails, the same step is rerun without the
embedder and marked as a fallback. Exit code: 1 when the unit tests or the Telegram integration
test fail, else 0. Secrets reach only the integration step, all output is scrubbed (ci/redact.py).

Nothing here talks to the network by itself. With HF_HUB_OFFLINE=1 (the workflow default)
the model must already be in the Hugging Face cache of the runner user.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAIL = 4000
sys.path.insert(0, str(ROOT / "ci"))
import redact  # noqa: E402


# Compute limits for every step (set by main from A2179_CI_DEVICE, default cpu). A 2 GB MX130
# with several embedders in a row overflowed VRAM into shared RAM and hung the whole machine, so
# CI embeds on the CPU by default: CUDA hidden, few threads, small batches, one model at a time.
COMPUTE_ENV = {}


def compute_env(env=os.environ):
    device = (env.get("A2179_CI_DEVICE") or "cpu").strip().lower()
    threads = str(env.get("A2179_CI_THREADS") or max(1, min(4, (os.cpu_count() or 2) // 2)))
    out = {"A2179_DEVICE": device, "A2179_EMBED_BATCH": str(env.get("A2179_CI_EMBED_BATCH") or 16),
           "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
           "TOKENIZERS_PARALLELISM": "false",
           # dense scores reused across runs for the same model + prefixes + prototypes + rows
           "A2179_DENSE_CACHE": env.get("A2179_CI_DENSE_CACHE") or str(Path.home() / ".cache" / "a2179-ci" / "dense")}
    if device == "cpu":
        out["CUDA_VISIBLE_DEVICES"] = ""   # nothing can touch the GPU, torch reports cuda_available False
    return out


def step_env(with_secrets=False):
    """Environment for a step. Secrets reach only the steps that need them (the TG integration
    test); tests, eval and sweep never see TG_BOT_TOKEN, so no test can send by accident."""
    env = {**os.environ, **COMPUTE_ENV}
    if not with_secrets:
        for n in redact.names():
            env.pop(n, None)
        env.pop("A2179_CI_SECRET_NAMES", None)
    return env


def run(name, cmd, timeout, env=None, with_secrets=False):
    t0 = time.time()
    rec = {"name": name, "cmd": cmd, "exit_code": None, "signal": None, "timed_out": False}
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                           env={**step_env(with_secrets), **(env or {})})
        rec["exit_code"] = p.returncode
        if p.returncode < 0:
            try:
                rec["signal"] = signal.Signals(-p.returncode).name
            except ValueError:
                rec["signal"] = str(-p.returncode)
        out, err = p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rec["timed_out"] = True
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    rec["duration_s"] = round(time.time() - t0, 2)
    out, err = redact.scrub(out), redact.scrub(err)
    rec["stdout_tail"] = out[-TAIL:]
    rec["stderr_tail"] = err[-TAIL:]
    rec["ok"] = rec["exit_code"] == 0
    print(f"[{name}] exit={rec['exit_code']} signal={rec['signal']} timeout={rec['timed_out']} "
          f"{rec['duration_s']}s", flush=True)
    if not rec["ok"]:
        print((err or out)[-1500:], flush=True)
    return rec


def env_info(py, timeout=300):
    rec = run("env_probe", [py, str(ROOT / "ci" / "env_probe.py")], timeout)
    info = {}
    if rec["ok"]:
        try:
            info = json.loads(rec["stdout_tail"].strip().splitlines()[-1])
        except (ValueError, IndexError):
            info = {}
    return rec, info


def dense_decision(mode, info, check_rec):
    """(enabled, reason). mode: auto | off | on."""
    if mode == "off":
        return False, "disabled (A2179_CI_DENSE=off)"
    if not info.get("torch"):
        return False, "torch not importable in the CI venv"
    if not info.get("sentence_transformers"):
        return False, "sentence-transformers not importable in the CI venv"
    if check_rec is None or not check_rec["ok"]:
        why = "crashed" if check_rec and (check_rec["signal"] or check_rec["timed_out"]) else "failed"
        return False, f"monitor.py --check-embedder {why}"
    return True, "cuda" if info.get("cuda_available") else "cpu (no CUDA device visible)"


DEFAULT_TARGET = "a2179_screen"
NO_LABELS = 4   # eval/run_eval.py and eval/sweep.py: the target has no labeled set yet


def other_targets(spec):
    names = []
    for n in (spec or "").split(","):
        n = n.strip()
        if n and n != DEFAULT_TARGET and n not in names and all(c.isalnum() or c in "_-" for c in n):
            names.append(n)
    return names


def target_steps(name, py, out, dense, offline, T):
    d = out / "targets" / name
    (d / "eval").mkdir(parents=True, exist_ok=True)
    (d / "sweep").mkdir(parents=True, exist_ok=True)
    ev = [py, "eval/run_eval.py", "--target", name, "--out-dir", str(d / "eval"),
          "--config", str(d / "eval" / "config.unused.json")] + (["--local-only"] if offline else [])
    rec = run(f"eval[{name}]", ev if dense else ev + ["--no-embed"], T)
    if rec["exit_code"] == NO_LABELS:
        why = (rec["stderr_tail"].strip().splitlines() or ["no labeled set"])[-1][:300]
        return [{**rec, "ok": True, "advisory": True, "skipped": why}]
    rec["advisory"] = True
    steps = [rec]
    if dense and not rec["ok"]:
        rec = run(f"eval[{name}]", ev + ["--no-embed"], T)
        rec.update(advisory=True, fallback="rerun without the embedder")
        steps.append(rec)
    sw = [py, "eval/sweep.py", "--target", name, "--out", str(d / "sweep" / "sweep.json"),
          "--dense-scores", str(d / "sweep" / "dense_scores.jsonl")]
    # T-029: same extra embedders as the default target (cached-small, nothing downloaded),
    # so de_jobs gets dense_scores_<model>.jsonl besides MiniLM
    emb = ["--embedders", os.environ.get("A2179_CI_EMBEDDERS", "cached-small"),
           "--device", COMPUTE_ENV.get("A2179_DEVICE", "cpu")]
    rec = run(f"sweep[{name}]", sw + ["--compute-dense"] + emb if dense else sw, T)
    rec["advisory"] = True
    steps.append(rec)
    if dense and not rec["ok"]:
        (d / "sweep" / "dense_scores.jsonl").unlink(missing_ok=True)
        rec = run(f"sweep[{name}]", sw, T)
        rec.update(advisory=True, fallback="rerun without dense scores")
        steps.append(rec)
    return steps


REQUESTS = ROOT / "ci" / "requests"


def fetch_request(root=ROOT):
    """ADR-0022: a committed request ci/requests/fetch_listings.json {"id": ..., "max_pages": ..,
    "time_budget_s": ..} asks the runner to collect fresh A2179 listings once. Fulfilled when some
    results/<base>/reports/fresh_a2179/<id>/ is already in the repo. Returns the request or None."""
    p = Path(root) / "ci" / "requests" / "fetch_listings.json"
    if os.environ.get("A2179_CI_FETCH_LISTINGS", "auto") in ("off", "0", "no") or not p.is_file():
        return None
    req = json.loads(p.read_text(encoding="utf-8"))
    rid = str(req.get("id", "")).strip()
    if not rid or any((Path(root) / "results").glob(f"*/reports/fresh_a2179/{rid}/candidates.jsonl")):
        return None
    return {**req, "id": rid}


def fetch_step(py, out: Path, T, req):
    """Collect fresh listings with the monitor's own polite fetcher (eval/enrich.py collect: per host
    pacer, KP stops at the first refusal, rows already labeled dropped). Raw rows land in
    <out>/reports/fresh_a2179/<id>/ and write_results commits them under results/<base>/reports/.
    Advisory: a refused site must not turn CI red."""
    d = out / "reports" / "fresh_a2179" / req["id"]
    d.mkdir(parents=True, exist_ok=True)
    cmd = [py, "eval/enrich.py", "collect", "--out", str(d / "candidates.jsonl"), "--csv",
           "--max-pages", str(int(req.get("max_pages", 3))),
           "--time-budget", str(float(req.get("time_budget_s", 600)))]
    if req.get("queries_file"):   # T-023: wider query set, a file committed in the repo
        cmd += ["--queries", str(ROOT / req["queries_file"])]
    rec = run("fetch_listings", cmd, min(T, int(float(req.get("time_budget_s", 600))) + 300))
    rec["advisory"] = True
    return rec


def probe_request(root=ROOT):
    """T-033..T-035: ci/requests/probe_sources.json {"id": .., "urls": [..]} asks the runner for one polite
    probe of new sources (robots.txt first, one GET per URL, blocked stays blocked). Once per id."""
    p = Path(root) / "ci" / "requests" / "probe_sources.json"
    if os.environ.get("A2179_CI_FETCH_LISTINGS", "auto") in ("off", "0", "no") or not p.is_file():
        return None
    req = json.loads(p.read_text(encoding="utf-8"))
    rid = str(req.get("id", "")).strip()
    if not rid or any((Path(root) / "results").glob(f"*/reports/source_probe/{rid}/probe.json")):
        return None
    return {**req, "id": rid}


def probe_step(py, out: Path, T, req):
    d = out / "reports" / "source_probe" / req["id"]
    d.mkdir(parents=True, exist_ok=True)
    rec = run("probe_sources", [py, "eval/source_probe.py", "--request", str(ROOT / "ci" / "requests" / "probe_sources.json"),
                                "--out", str(d / "probe.json")], min(T, 300))
    rec["advisory"] = True
    return rec


def used_request(root=ROOT, date=None):
    """T-033 / EXP-017: ci/requests/used_sources_daily.json {"id", "pages": [..]} collects the Limundo
    and KupindO category pages once per UTC day (CI runs only on push, so: the first run of the day).
    Fulfilled for the day when some results/<base>/reports/used_sources/<date>/summary.json exists."""
    p = Path(root) / "ci" / "requests" / "used_sources_daily.json"
    if os.environ.get("A2179_CI_FETCH_LISTINGS", "auto") in ("off", "0", "no") or not p.is_file():
        return None
    req = json.loads(p.read_text(encoding="utf-8"))
    date = date or time.strftime("%Y-%m-%d", time.gmtime())
    if not req.get("pages") or any((Path(root) / "results").glob(f"*/reports/used_sources/{date}/summary.json")):
        return None
    return {**req, "date": date}


def used_step(py, out: Path, T, req):
    d = out / "reports" / "used_sources" / req["date"]
    rec = run("used_sources", [py, "eval/used_sources.py", "--request", str(ROOT / "ci" / "requests" / "used_sources_daily.json"),
                               "--out", str(d)], min(T, int(float(req.get("time_budget_s", 240))) + 120))
    rec["advisory"] = True
    return rec


def kp_daily_request(root=ROOT, date=None):
    """T-031 / H-38: ci/requests/kp_daily.json {"id", "queries_file", "max_pages", "time_budget_s"} collects
    KP once per UTC day (first CI run of the day), same polite fetcher as fetch_listings, KP only.
    Fulfilled for the day when some results/<base>/reports/kp_daily/<date>/candidates.jsonl exists."""
    p = Path(root) / "ci" / "requests" / "kp_daily.json"
    if os.environ.get("A2179_CI_FETCH_LISTINGS", "auto") in ("off", "0", "no") or not p.is_file():
        return None
    req = json.loads(p.read_text(encoding="utf-8"))
    date = date or time.strftime("%Y-%m-%d", time.gmtime())
    if not req.get("queries_file") or any((Path(root) / "results").glob(f"*/reports/kp_daily/{date}/candidates.jsonl")):
        return None
    return {**req, "date": date}


def kp_daily_step(py, out: Path, T, req):
    d = out / "reports" / "kp_daily" / req["date"]
    d.mkdir(parents=True, exist_ok=True)
    cmd = [py, "eval/enrich.py", "collect", "--out", str(d / "candidates.jsonl"), "--shops", "none",
           "--max-pages", str(int(req.get("max_pages", 1))), "--time-budget", str(float(req.get("time_budget_s", 300))),
           "--queries", str(ROOT / req["queries_file"]), "--summary", str(d / "summary.json")]
    rec = run("kp_daily", cmd, min(T, int(float(req.get("time_budget_s", 300))) + 120))
    rec["advisory"] = True
    return rec


RUNNER_OUT = {"granularity": "granularity.json", "rerank_jobs": "rerank_jobs.json", "exp028": "exp028.json"}


def runner_request(root=ROOT):
    """EXP-021/022/028: ci/requests/runner_experiments.json {"id", "steps"} runs CPU experiments
    that need the runner's model cache. Once per id and step: a step is done when its output file is in
    some results/<base>/reports/runner_exp/<id>/."""
    p = Path(root) / "ci" / "requests" / "runner_experiments.json"
    if os.environ.get("A2179_CI_RUNNER_EXP", "auto") in ("off", "0", "no") or not p.is_file():
        return None
    req = json.loads(p.read_text(encoding="utf-8"))
    rid = str(req.get("id", "")).strip()
    todo = [s for s in req.get("steps", []) if s in RUNNER_OUT
            and not any((Path(root) / "results").glob(f"*/reports/runner_exp/{rid}/{RUNNER_OUT[s]}"))]
    return {**req, "id": rid, "steps": todo} if rid and todo else None


def runner_steps(py, out: Path, T, req, dense):
    d = out / "reports" / "runner_exp" / req["id"]
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for s in req["steps"]:
        if s in ("granularity", "rerank_jobs"):
            if not dense:
                recs.append({"name": f"runner_exp[{s}]", "cmd": None, "exit_code": None, "signal": None,
                             "timed_out": False, "duration_s": 0, "ok": True, "skipped": "dense not available",
                             "stdout_tail": "", "stderr_tail": "", "advisory": True})
                continue
        if s == "granularity":
            rec = run("runner_exp[granularity]", [py, "eval/granularity.py", "--out", str(d),
                                                  "--device", COMPUTE_ENV.get("A2179_DEVICE", "cpu")], min(T, 900))
        elif s == "exp028":
            # EXP-028: CPU only, one model at a time, runner may fetch weights. GPU stays hidden.
            rec = run("runner_exp[exp028]", [py, "eval/exp028_blocked_models.py", "--out", str(d), "--device", "cpu"],
                      min(max(T, 5400), 7200),
                      env={"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0", "CUDA_VISIBLE_DEVICES": "",
                           "A2179_DEVICE": "cpu"})
        else:
            # the only download outside setup: eval/rerank_jobs.py checks the summed file size (Hub
            # metadata) against 50 MiB first and writes blocked instead of downloading when over
            rec = run("runner_exp[rerank_jobs]", [py, "eval/rerank_jobs.py", "--out", str(d)], min(T, 900),
                      env={"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"})
        rec["advisory"] = True
        recs.append(rec)
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "ci_out"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dense", default=os.environ.get("A2179_CI_DENSE", "auto"), choices=["auto", "on", "off"])
    ap.add_argument("--skip-tests", action="store_true", default=bool(os.environ.get("A2179_CI_SKIP_TESTS")))
    ap.add_argument("--step-timeout", type=int, default=int(os.environ.get("A2179_CI_STEP_TIMEOUT", "1800")))
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    (out / "sweep").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    py, T = a.python, a.step_timeout
    COMPUTE_ENV.clear()
    COMPUTE_ENV.update(compute_env())
    print(f"compute: device={COMPUTE_ENV['A2179_DEVICE']} threads={COMPUTE_ENV['OMP_NUM_THREADS']} "
          f"batch={COMPUTE_ENV['A2179_EMBED_BATCH']} embedders={os.environ.get('A2179_CI_EMBEDDERS', 'cached-small')}",
          flush=True)
    offline = os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "yes")

    steps = []
    probe, info = env_info(py)
    steps.append(probe)
    # ADR-0026: A2179_CI_SELECT_* from ci/select_steps.py. full = everything (default), targeted =
    # lint + the tests of the touched files + only the experiment steps the commit touched
    mode = os.environ.get("A2179_CI_SELECT_MODE", "full")
    sel_tests = os.environ.get("A2179_CI_SELECT_TESTS", "").split()
    sel_steps = {s for s in os.environ.get("A2179_CI_SELECT_STEPS", "").split(",") if s}
    want = (lambda s: True) if mode != "targeted" else (lambda s: s in sel_steps)
    lint = run("lint", [py, "-m", "compileall", "-q", "a2179", "eval", "ci", "tests", "monitor.py"], T)
    steps.append(lint)
    if not a.skip_tests:
        if mode == "targeted":
            if sel_tests:
                steps.append(run("tests", [py, "-m", "unittest", *sel_tests], T))
        else:
            steps.append(run("tests", [py, "-m", "unittest", "discover", "-s", "tests", "-t", "."], T))
    if mode == "targeted":
        print(f"targeted run ({os.environ.get('A2179_CI_SELECT_WHY', '')}): tests {sel_tests or 'none'}, "
              f"steps {sorted(sel_steps) or 'none'}", flush=True)

    check = None
    if mode == "targeted" and not ({"eval", "sweep", "targets"} & sel_steps):
        a.dense = "off"
    if a.dense != "off" and info.get("torch") and info.get("sentence_transformers"):
        check = run("check_embedder", [py, "monitor.py", "--check-embedder"], T)
        steps.append(check)
    dense, reason = dense_decision(a.dense, info, check)
    if a.dense == "on" and not dense:
        reason += " (A2179_CI_DENSE=on, but not available)"

    if want("eval") or want("sweep"):
        ev = [py, "eval/run_eval.py", "--out-dir", str(out / "eval"), "--config", str(out / "eval" / "config.unused.json")]
        if offline:
            ev.append("--local-only")
        rec = run("eval", ev if dense else ev + ["--no-embed"], T)
        steps.append(rec)
        if dense and not rec["ok"]:
            rec = run("eval", ev + ["--no-embed"], T)
            rec["fallback"] = "rerun without the embedder"
            steps.append(rec)

        sw = [py, "eval/sweep.py", "--out", str(out / "sweep" / "sweep.json"),
              "--dense-scores", str(out / "sweep" / "dense_scores.jsonl")]
        # extra embedders for the side by side table (EXP-003): by default every model of
        # eval/embedders.json that is already in the HF cache, nothing is downloaded here
        # (cached-small: cached and at most A2179_MAX_PARAMS_M, default 300 M params, so bge-m3,
        # e5-large, LaBSE only when listed by name in A2179_CI_EMBEDDERS)
        emb = ["--embedders", os.environ.get("A2179_CI_EMBEDDERS", "cached-small"),
               "--device", COMPUTE_ENV.get("A2179_DEVICE", "cpu")]
        rec = run("sweep", sw + ["--compute-dense"] + emb if dense else sw, T)
        steps.append(rec)
        if dense and not rec["ok"]:
            (out / "sweep" / "dense_scores.jsonl").unlink(missing_ok=True)
            rec = run("sweep", sw, T)
            rec["fallback"] = "rerun without dense scores"
            steps.append(rec)

    # the other targets (T-004, ADR-0017): same eval + sweep per profile into <out>/targets/<name>/,
    # skipped with a reason while a target has no labeled set (exit 4). Advisory: a new case
    # must not turn the A2179 run red. The default target above keeps its old paths.
    for name in other_targets(os.environ.get("A2179_CI_TARGETS", "a2179_screen,de_jobs")) if want("targets") else []:
        steps.extend(target_steps(name, py, out, dense, offline, T))

    req = fetch_request()
    if req and (want("fetch") or mode != "targeted"):
        steps.append(fetch_step(py, out, T, req))
    preq = probe_request()
    if preq and (want("fetch") or mode != "targeted"):
        steps.append(probe_step(py, out, T, preq))
    ureq = used_request()
    if ureq:   # once a day in any mode, also on a targeted run (light: robots + 5 pages)
        steps.append(used_step(py, out, T, ureq))
    kreq = kp_daily_request()
    if kreq:   # once a day in any mode, KP only, one page per query
        steps.append(kp_daily_step(py, out, T, kreq))

    # charts + ranking metrics from the sweep scores, advisory: a failure here never turns CI red
    if want("plots") and os.environ.get("A2179_CI_PLOTS", "1") not in ("0", "false", "no") and (out / "sweep" / "sweep.json").is_file():
        rec = run("plots", [py, "eval/plots.py", "--sweep", str(out / "sweep" / "sweep.json"),
                            "--out", str(out / "plots")], T)
        rec["advisory"] = True
        steps.append(rec)

    # two-case review reports (ADR-0017): reports/<date>/<case>/matches.csv + all.csv, with source links
    if want("review") and os.environ.get("A2179_CI_REVIEW", "1") not in ("0", "false", "no"):
        date = time.strftime("%Y-%m-%d", time.gmtime())
        cache = os.environ.get("A2179_JOBS_CACHE") or str(Path.home() / ".cache" / "a2179-ci" / "jobs")
        rv = [py, "eval/review_run.py",
              "--out-root", str(out / "reports"),
              "--date", date,
              "--cases", os.environ.get("A2179_CI_TARGETS", "a2179_screen,de_jobs"),
              "--fetch", os.environ.get("A2179_CI_FETCH_JOBS", "auto"),
              "--jobs-cache", cache]
        rec = run("review", rv, T)
        rec["advisory"] = True  # a missing cache or a failed API call must not turn CI red
        steps.append(rec)

    # one-off CPU experiments on the runner's model cache (after review, so the jobs cache is fresh)
    rreq = runner_request()
    if rreq:
        steps.extend(runner_steps(py, out, T, rreq, bool(info.get("torch") and info.get("sentence_transformers"))))

    # Telegram integration: only with both secrets, the only step that gets them
    have = redact.presence()
    if have.get("TG_BOT_TOKEN", "absent") != "absent" and have.get("TG_CHAT_ID", "absent") != "absent":
        steps.append(run("tg_integration", [py, "-m", "unittest", "-v", "integration.test_telegram"], 300,
                         with_secrets=True))
    else:
        steps.append({"name": "tg_integration", "cmd": None, "exit_code": None, "signal": None,
                      "timed_out": False, "duration_s": 0, "ok": True,
                      "skipped": "TG_BOT_TOKEN / TG_CHAT_ID absent", "stdout_tail": "", "stderr_tail": ""})

    summary = {"select": {"mode": mode, "tests": sel_tests, "steps": sorted(sel_steps),
                          "base": os.environ.get("A2179_CI_SELECT_BASE", ""), "why": os.environ.get("A2179_CI_SELECT_WHY", "")},
               "env": info, "secrets": have, "dense": {"enabled": dense, "reason": reason, "mode": a.dense, "hf_offline": offline},
               "steps": steps}
    (out / "steps.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"dense: {dense} ({reason}) -> {out / 'steps.json'}")
    gating = [s for s in steps if s["name"] in ("lint", "tests", "tg_integration")]
    return 1 if any(not s["ok"] for s in gating) else 0


if __name__ == "__main__":
    sys.exit(main())
