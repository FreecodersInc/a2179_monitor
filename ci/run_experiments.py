"""CI step runner: tests, embedder check, eval and sweep, each in its own process.

Every step records command, exit code, duration and the tail of its output into
<out>/steps.json, so ci/write_results.py can build one results file even when a step
crashed. The dense (embedder) steps run only when torch + sentence-transformers import
and `monitor.py --check-embedder` passes (that check loads the model in a child process
first, see ADR-0005). If a dense step still fails, the same step is rerun without the
embedder and marked as a fallback. Exit code: 1 when the unit tests fail, else 0.

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


def run(name, cmd, timeout, env=None):
    t0 = time.time()
    rec = {"name": name, "cmd": cmd, "exit_code": None, "signal": None, "timed_out": False}
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, **(env or {})})
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
    py, T = a.python, a.step_timeout
    offline = os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "yes")

    steps = []
    probe, info = env_info(py)
    steps.append(probe)
    if not a.skip_tests:
        steps.append(run("tests", [py, "-m", "unittest", "discover", "-s", "tests", "-t", "."], T))

    check = None
    if a.dense != "off" and info.get("torch") and info.get("sentence_transformers"):
        check = run("check_embedder", [py, "monitor.py", "--check-embedder"], T)
        steps.append(check)
    dense, reason = dense_decision(a.dense, info, check)
    if a.dense == "on" and not dense:
        reason += " (A2179_CI_DENSE=on, but not available)"

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
    rec = run("sweep", sw + ["--compute-dense"] if dense else sw, T)
    steps.append(rec)
    if dense and not rec["ok"]:
        (out / "sweep" / "dense_scores.jsonl").unlink(missing_ok=True)
        rec = run("sweep", sw, T)
        rec["fallback"] = "rerun without dense scores"
        steps.append(rec)

    summary = {"env": info, "dense": {"enabled": dense, "reason": reason, "mode": a.dense, "hf_offline": offline},
               "steps": steps}
    (out / "steps.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"dense: {dense} ({reason}) -> {out / 'steps.json'}")
    tests = [s for s in steps if s["name"] == "tests"]
    return 1 if tests and not tests[-1]["ok"] else 0


if __name__ == "__main__":
    sys.exit(main())
