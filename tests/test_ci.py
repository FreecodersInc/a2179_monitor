"""ci/run_experiments.py and ci/write_results.py: step records, dense guard, results file, commit + push."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import datetime as dt
from pathlib import Path

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "ci"))
import run_experiments as rx  # noqa: E402
import write_results as wr  # noqa: E402

GIT = shutil.which("git")
NOW = dt.datetime(2026, 11, 19, 23, 30, tzinfo=dt.timezone.utc)
ENV = {"GITHUB_SHA": "0123456789abcdef0123456789abcdef01234567", "GITHUB_REPOSITORY": "FreecodersInc/a2179_monitor",
       "GITHUB_RUN_ID": "42", "GITHUB_SERVER_URL": "https://github.com", "GITHUB_REF": "refs/heads/main",
       "RUNNER_NAME": "leonid-wsl", "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64"}


def step(name, ok=True, **kw):
    return {"name": name, "cmd": ["python", name], "exit_code": 0 if ok else 1, "signal": None, "timed_out": False,
            "duration_s": 1.5, "ok": ok, "stdout_tail": "", "stderr_tail": "" if ok else "boom", **kw}


def ci_out(tmp: Path, steps, dense=None, env=None, eval_report=True, sweep=True):
    (tmp / "eval").mkdir(parents=True, exist_ok=True)
    (tmp / "sweep").mkdir(parents=True, exist_ok=True)
    (tmp / "steps.json").write_text(json.dumps({"env": env or {"python": "3.14.4", "torch": "2.15.0+cu132",
                                                                "cuda_available": True, "device_name": "RTX"},
                                                "dense": dense or {"enabled": True, "reason": "cuda"},
                                                "steps": steps}))
    if eval_report:
        shutil.copy(ROOT / "reports" / "eval_report.json", tmp / "eval" / "eval_report.json")
    if sweep:
        shutil.copy(ROOT / "reports" / "sweep_2026-11-19.json", tmp / "sweep" / "sweep.json")


class TestRun(unittest.TestCase):
    def test_records_exit_and_duration(self):
        r = rx.run("x", [sys.executable, "-c", "import sys; print('hi'); sys.exit(3)"], 30)
        self.assertEqual((r["exit_code"], r["ok"], r["stdout_tail"].strip()), (3, False, "hi"))
        self.assertGreaterEqual(r["duration_s"], 0)

    def test_crashing_child_is_recorded(self):
        r = rx.run("x", [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"], 30)
        self.assertEqual(r["signal"], "SIGSEGV")
        self.assertFalse(r["ok"])

    def test_timeout(self):
        r = rx.run("x", [sys.executable, "-c", "import time; time.sleep(5)"], 0.5)
        self.assertTrue(r["timed_out"])
        self.assertFalse(r["ok"])

    def test_env_probe_is_json(self):
        rec, info = rx.env_info(sys.executable)
        self.assertTrue(rec["ok"])
        self.assertIn("python", info)
        self.assertIn("cuda_available", info)


class TestDense(unittest.TestCase):
    ok = {"name": "check_embedder", "ok": True, "signal": None, "timed_out": False}

    def test_off(self):
        self.assertFalse(rx.dense_decision("off", {"torch": "2", "sentence_transformers": "3"}, self.ok)[0])

    def test_no_torch(self):
        en, why = rx.dense_decision("auto", {"torch": None}, None)
        self.assertFalse(en)
        self.assertIn("torch", why)

    def test_check_crashed(self):
        bad = {**self.ok, "ok": False, "signal": "SIGSEGV"}
        en, why = rx.dense_decision("auto", {"torch": "2", "sentence_transformers": "3"}, bad)
        self.assertFalse(en)
        self.assertIn("crashed", why)

    def test_cuda(self):
        self.assertEqual(rx.dense_decision("auto", {"torch": "2", "sentence_transformers": "3",
                                                     "cuda_available": True}, self.ok), (True, "cuda"))

    def test_main_without_torch_skips_dense(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []

            def fake_run(name, cmd, timeout, env=None):
                calls.append((name, cmd))
                return step(name)
            orig_run, orig_env = rx.run, rx.env_info
            rx.run, rx.env_info = fake_run, lambda py: (step("env_probe"), {"python": "3.14", "torch": None})
            try:
                code = rx.main(["--out", d])
            finally:
                rx.run, rx.env_info = orig_run, orig_env
            names = [c[0] for c in calls]
            self.assertEqual(code, 0)
            self.assertEqual(names, ["tests", "eval", "sweep"])
            self.assertIn("--no-embed", calls[1][1])
            self.assertNotIn("--compute-dense", calls[2][1])
            s = json.loads((Path(d) / "steps.json").read_text())
            self.assertFalse(s["dense"]["enabled"])

    def test_dense_failure_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []

            def fake_run(name, cmd, timeout, env=None):
                calls.append((name, cmd))
                bad = name in ("eval", "sweep") and ("--no-embed" not in cmd and "--compute-dense" in cmd
                                                     or name == "eval" and "--no-embed" not in cmd)
                return step(name, ok=not bad)
            orig_run, orig_env = rx.run, rx.env_info
            rx.run = fake_run
            rx.env_info = lambda py: (step("env_probe"), {"torch": "2", "sentence_transformers": "3",
                                                         "cuda_available": True})
            try:
                rx.main(["--out", d])
            finally:
                rx.run, rx.env_info = orig_run, orig_env
            names = [c[0] for c in calls]
            self.assertEqual(names, ["tests", "check_embedder", "eval", "eval", "sweep", "sweep"])
            s = json.loads((Path(d) / "steps.json").read_text())
            self.assertEqual([x.get("fallback") is not None for x in s["steps"]][-4:], [False, True, False, True])

    def test_tests_fail_exit_1(self):
        with tempfile.TemporaryDirectory() as d:
            orig_run, orig_env = rx.run, rx.env_info
            rx.run = lambda name, cmd, timeout, env=None: step(name, ok=name != "tests")
            rx.env_info = lambda py: (step("env_probe"), {})
            try:
                self.assertEqual(rx.main(["--out", d]), 1)
            finally:
                rx.run, rx.env_info = orig_run, orig_env


class TestBuild(unittest.TestCase):
    def test_full_record(self):
        with tempfile.TemporaryDirectory() as d:
            ci_out(Path(d), [step("env_probe"), step("tests"), step("check_embedder"), step("eval"), step("sweep")])
            r = wr.build(Path(d), env=ENV, now=NOW)
        self.assertEqual(r["commit"]["short"], "0123456")
        self.assertEqual(r["commit"]["run_url"], "https://github.com/FreecodersInc/a2179_monitor/actions/runs/42")
        self.assertEqual(r["runner"]["name"], "leonid-wsl")
        self.assertTrue(r["runner"]["cuda_available"])
        self.assertTrue(r["ok"])
        self.assertTrue(r["tests_ok"])
        self.assertEqual(r["total_duration_s"], 7.5)
        ev = r["metrics"]["eval"]
        self.assertEqual(ev["dataset"]["n"], 99)
        self.assertEqual(ev["modes"]["hybrid_tfidf"]["recall"], 1.0)
        sw = r["metrics"]["sweep"]
        self.assertEqual(sw["best_hybrid"]["R"], 1.0)
        self.assertTrue(all("F1" in v or "note" in v or "P" in v for v in sw["variants"]))
        json.dumps(r)

    def test_failed_step_and_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            ci_out(Path(d), [step("tests"), step("eval", ok=False), {**step("eval"), "fallback": "no embed"},
                             step("check_embedder", ok=False)], eval_report=False, sweep=False)
            r = wr.build(Path(d), env=ENV, now=NOW)
        self.assertTrue(r["ok"])  # eval rerun passed, check_embedder only gates dense
        self.assertEqual(r["steps"][1]["stderr_tail"], "boom")
        self.assertIsNone(r["steps"][0]["stderr_tail"])
        self.assertIsNone(r["metrics"]["eval"])

    def test_tests_failed_not_ok(self):
        with tempfile.TemporaryDirectory() as d:
            ci_out(Path(d), [step("tests", ok=False), step("eval")])
            r = wr.build(Path(d), env=ENV, now=NOW)
        self.assertFalse(r["ok"])
        self.assertFalse(r["tests_ok"])

    def test_missing_ci_out(self):
        with tempfile.TemporaryDirectory() as d:
            r = wr.build(Path(d) / "nothing", env=ENV, now=NOW)
        self.assertFalse(r["ok"])

    def test_path_and_collision(self):
        with tempfile.TemporaryDirectory() as d:
            res = {"generated_utc": "2026-11-19T23:30:00Z", "commit": {"short": "0123456"}}
            p1 = wr.write(Path(d), res)
            p2 = wr.write(Path(d), res)
            self.assertEqual((p1.name, p2.name), ("2026-11-19_0123456.json", "2026-11-19_0123456_2.json"))
            self.assertEqual(json.loads(p1.read_text())["commit"]["short"], "0123456")


@unittest.skipUnless(GIT, "git not installed")
class TestPush(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = {**os.environ, "GIT_CONFIG_GLOBAL": str(self.tmp / "gitconfig"), "GIT_CONFIG_NOSYSTEM": "1",
                    "HOME": str(self.tmp)}
        (self.tmp / "gitconfig").write_text("[user]\n\tname = t\n\temail = t@t\n[init]\n\tdefaultBranch = main\n")
        self._old = dict(os.environ)
        os.environ.update(self.env)
        self.g = lambda *a, cwd: subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True)
        self.remote = self.tmp / "remote.git"
        self.g("init", "--bare", "-b", "main", str(self.remote), cwd=self.tmp)
        self.a = self.tmp / "a"
        self.g("clone", str(self.remote), str(self.a), cwd=self.tmp)
        (self.a / "README").write_text("x\n")
        self.g("add", "README", cwd=self.a)
        self.g("commit", "-m", "init", cwd=self.a)
        self.g("push", "origin", "HEAD:main", cwd=self.a)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def result(self, repo):
        return wr.write(repo / "results", {"generated_utc": "2026-11-19T23:30:00Z", "commit": {"short": "abc1234"}})

    def test_commit_message_and_push(self):
        p = self.result(self.a)
        self.assertEqual(wr.commit_and_push(p, "main", cwd=self.a, pause=0), 1)
        log = self.g("log", "-1", "--format=%s%n%an", "main", cwd=self.remote).stdout.split("\n")
        self.assertEqual(log[0], "results: 2026-11-19_abc1234 [skip ci]")
        self.assertEqual(log[1], "a2179-ci")
        files = self.g("show", "--name-only", "--format=", "main", cwd=self.remote).stdout.split()
        self.assertEqual(files, ["results/2026-11-19_abc1234.json"])

    def test_rejected_push_rebases_and_retries(self):
        b = self.tmp / "b"
        self.g("clone", str(self.remote), str(b), cwd=self.tmp)
        (b / "other.txt").write_text("someone pushed meanwhile\n")
        self.g("add", "other.txt", cwd=b)
        self.g("commit", "-m", "meanwhile", cwd=b)
        self.g("push", "origin", "HEAD:main", cwd=b)
        p = self.result(self.a)
        self.assertEqual(wr.commit_and_push(p, "main", cwd=self.a, pause=0), 2)
        subjects = self.g("log", "--format=%s", "main", cwd=self.remote).stdout.split("\n")[:2]
        self.assertEqual(subjects, ["results: 2026-11-19_abc1234 [skip ci]", "meanwhile"])

    def test_only_the_result_file_is_committed(self):
        (self.a / "ci_out").mkdir()
        (self.a / "ci_out" / "junk").write_text("x")
        (self.a / "README").write_text("changed by a step\n")
        p = self.result(self.a)
        wr.commit_and_push(p, "main", cwd=self.a, pause=0, push=False)
        files = self.g("show", "--name-only", "--format=", "HEAD", cwd=self.a).stdout.split()
        self.assertEqual(files, ["results/2026-11-19_abc1234.json"])


class TestWorkflowText(unittest.TestCase):
    """Structural checks without a YAML dependency (the box parses it with PyYAML separately).

    The workflow ships as ci/experiments.yml (our token cannot write .github/workflows), Leonid
    moves it into .github/workflows/ with one git mv. Whichever exists is checked."""
    _active = ROOT / ".github" / "workflows" / "experiments.yml"
    wf = (_active if _active.exists() else ROOT / "ci" / "experiments.yml").read_text()

    def test_shipped_in_ci_or_activated(self):
        self.assertTrue((ROOT / "ci" / "experiments.yml").exists() or self._active.exists())

    def test_triggers(self):
        on = self.wf.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
        for bad in ("pull_request", "workflow_run", "schedule", "issue_comment", "workflow_dispatch"):
            self.assertNotIn(bad, on)
        self.assertIn("  push:\n    branches: [main]", on)
        self.assertIn('"results/**"', on)
        self.assertIn("!contains(github.event.head_commit.message, '[skip ci]')", self.wf)

    def test_permissions_and_runner(self):
        self.assertIn("permissions:\n  contents: write\n\n", self.wf)
        self.assertIn("runs-on: [self-hosted, linux, a2179-gpu]", self.wf)
        self.assertIn("timeout-minutes:", self.wf)
        self.assertIn("concurrency:", self.wf)
        self.assertNotIn("secrets.", self.wf)
        self.assertNotIn("sudo", self.wf)


if __name__ == "__main__":
    unittest.main()
