"""ci/run_experiments.py and ci/write_results.py: step records, dense guard, results file, commit + push."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
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
        shutil.copy(ROOT / "tests" / "fixtures" / "eval_report_99.json", tmp / "eval" / "eval_report.json")
    if sweep:
        shutil.copy(ROOT / "tests" / "fixtures" / "sweep_99.json", tmp / "sweep" / "sweep.json")


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
            self.assertEqual(names, ["lint", "tests", "eval", "sweep", "eval[de_jobs]", "sweep[de_jobs]", "review"])
            self.assertIn("--no-embed", calls[4][1])   # the second target follows the same dense decision
            self.assertIn("--no-embed", calls[2][1])
            self.assertNotIn("--compute-dense", calls[3][1])
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
            self.assertEqual(names, ["lint", "tests", "check_embedder", "eval", "eval", "sweep", "sweep",
                                     "eval[de_jobs]", "sweep[de_jobs]", "review"])
            s = json.loads((Path(d) / "steps.json").read_text())
            ev_sw = [x for x in s["steps"] if x["name"] in ("eval", "sweep")]
            self.assertEqual([x.get("fallback") is not None for x in ev_sw], [False, True, False, True])
            self.assertEqual(s["steps"][-1]["skipped"], "TG_BOT_TOKEN / TG_CHAT_ID absent")

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

    def test_workflow_in_sync(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "ci").mkdir()
            (repo / ".github" / "workflows").mkdir(parents=True)
            (repo / "ci" / "experiments.yml").write_text("new\n")
            (repo / ".github" / "workflows" / "experiments.yml").write_text("old\n")
            self.assertFalse(wr.build(repo / "none", env=ENV, now=NOW, cwd=repo)["workflow_in_sync"])
            (repo / ".github" / "workflows" / "experiments.yml").write_text("new\n")
            self.assertTrue(wr.build(repo / "none", env=ENV, now=NOW, cwd=repo)["workflow_in_sync"])

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

    def test_reports_dir_committed_with_result(self):
        p = self.result(self.a)
        co = self.tmp / "co"
        (co / "eval").mkdir(parents=True)
        (co / "eval" / "eval_report.md").write_text("# r\n")
        (co / "setup.log").write_text("ok\n")
        d = wr.attach_reports(co, p)
        wr.commit_and_push(p, "main", cwd=self.a, pause=0, extra=[d])
        files = sorted(self.g("show", "--name-only", "--format=", "main", cwd=self.remote).stdout.split())
        self.assertEqual(files, ["results/2026-11-19_abc1234.json", "results/2026-11-19_abc1234/eval_report.md",
                                 "results/2026-11-19_abc1234/setup.log"])

    def test_review_reports_are_attached(self):
        p = self.result(self.a)
        co = self.tmp / "co2"
        case = co / "reports" / "2026-11-19" / "de_jobs"
        case.mkdir(parents=True)
        (case / "matches.csv").write_text("title,company,location,score,decision,url,fetched_at\n")
        (case / "summary.json").write_text('{"case":"de_jobs","n":3,"n_matches":1,"date":"2026-11-19"}')
        d = wr.attach_reports(co, p)
        self.assertTrue((d / "reports" / "2026-11-19" / "de_jobs" / "matches.csv").is_file())
        met = wr.review_metrics(co / "reports")
        self.assertEqual(met["de_jobs"]["n_matches"], 1)

    def test_only_the_result_file_is_committed(self):
        (self.a / "ci_out").mkdir()
        (self.a / "ci_out" / "junk").write_text("x")
        (self.a / "README").write_text("changed by a step\n")
        p = self.result(self.a)
        wr.commit_and_push(p, "main", cwd=self.a, pause=0, push=False)
        files = self.g("show", "--name-only", "--format=", "HEAD", cwd=self.a).stdout.split()
        self.assertEqual(files, ["results/2026-11-19_abc1234.json"])


WF = ROOT / "ci" / "experiments.yml"  # source of truth, Leonid copies it to .github/workflows/


def run_block(text: str) -> str:
    """The shell script under `run: |` of the only step (no YAML dependency)."""
    lines = text.split("run: |\n", 1)[1].splitlines()
    ind = len(lines[0]) - len(lines[0].lstrip())
    out = []
    for ln in lines:
        if ln.strip() and len(ln) - len(ln.lstrip()) < ind:
            break
        out.append(ln[ind:])
    return "\n".join(out) + "\n"


class TestWorkflowText(unittest.TestCase):
    """Structural checks of the thin wrapper (the box parses it with PyYAML separately)."""
    wf = WF.read_text()

    def test_triggers(self):
        on = self.wf.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
        for bad in ("pull_request", "workflow_run", "schedule", "issue_comment", "workflow_dispatch"):
            self.assertNotIn(bad, on)
        self.assertIn("  push:\n    branches: [main]", on)
        self.assertIn('"results/**"', on)
        self.assertIn("!contains(github.event.head_commit.message, '[skip ci]')", self.wf)
        self.assertIn("github.repository == 'FreecodersInc/a2179_monitor'", self.wf)

    def test_permissions_and_runner(self):
        self.assertIn("permissions:\n  contents: write\n\n", self.wf)
        self.assertIn("runs-on: [self-hosted, linux, a2179-gpu]", self.wf)
        self.assertIn("timeout-minutes:", self.wf)
        self.assertIn("concurrency:", self.wf)
        self.assertNotIn("sudo", self.wf)

    def test_no_actions_only_named_secrets(self):
        # org policy: only FreecodersInc actions; run #1 died on actions/checkout + upload-artifact
        code = [ln for ln in self.wf.splitlines() if not ln.lstrip().startswith("#")]
        self.assertFalse([ln for ln in code if "uses:" in ln])
        secrets = {w.rstrip("}") for w in self.wf.replace("${{", " ").split() if w.startswith("secrets.")}
        self.assertEqual(secrets, {"secrets.GITHUB_TOKEN", "secrets.TG_BOT_TOKEN", "secrets.TG_CHAT_ID",
                                   "secrets.A2179_CI_SECRETS"})
        self.assertFalse([ln for ln in code if "toJSON(secrets)" in ln])
        self.assertIn("    environment: experiment\n", self.wf)
        self.assertNotIn(".git/config", self.wf)

    def test_thin(self):
        # one thin job: the wrapper only execs ci/run.sh. H-11 is a runner_experiments step, not a job.
        self.assertEqual(self.wf.count("- name:"), 1)
        self.assertIn("A2179_CI_VARS: ${{ toJSON(vars) }}", self.wf)
        self.assertTrue(run_block(self.wf).rstrip().endswith("exec bash ci/run.sh"))
        self.assertNotIn("git config --", run_block(self.wf))  # token never written to .git/config
        self.assertNotIn("exec bash ci/run_h11.sh", self.wf)
        self.assertNotIn("  h11:\n", self.wf)
        self.assertEqual(self.wf.count("runs-on: [self-hosted, linux, a2179-gpu]"), 1)

    @unittest.skipUnless(GIT, "git not installed")
    def test_bootstrap_checks_out_pushed_sha_and_execs_run_sh(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            env = {**os.environ, "GIT_CONFIG_GLOBAL": str(tmp / "gc"), "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(tmp)}
            (tmp / "gc").write_text("[user]\n\tname = t\n\temail = t@t\n[init]\n\tdefaultBranch = main\n")
            g = lambda *a, cwd: subprocess.run(["git", *a], cwd=cwd, env=env, check=True, capture_output=True,
                                               text=True).stdout.strip()
            bare = tmp / "org" / "repo.git"
            bare.parent.mkdir()
            g("init", "--bare", "-b", "main", str(bare), cwd=tmp)
            src = tmp / "src"
            g("clone", str(bare), str(src), cwd=tmp)
            (src / "ci").mkdir()
            (src / "ci" / "run.sh").write_text('echo "run.sh at $(git rev-parse HEAD) on $(git branch --show-current)"\n'
                                               'test -z "$(git config --get http.extraheader)" && echo clean\n')
            g("add", ".", cwd=src)
            g("commit", "-m", "one", cwd=src)
            g("push", "origin", "HEAD:main", cwd=src)
            sha = g("rev-parse", "HEAD", cwd=src)
            ws = tmp / "ws"
            ws.mkdir()
            (ws / "stale.txt").write_text("from an old run")
            run_env = {**env, "GITHUB_WORKSPACE": str(ws), "GITHUB_SERVER_URL": f"file://{tmp}",
                       "GITHUB_REPOSITORY": "org/repo", "GITHUB_SHA": sha, "GITHUB_TOKEN": "t0ken"}
            for _ in range(2):  # second run reuses the workspace
                p = subprocess.run(["bash", "-c", run_block(self.wf)], env=run_env, capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn(f"run.sh at {sha} on main", p.stdout)
                self.assertIn("clean", p.stdout)
                self.assertFalse((ws / "stale.txt").exists())
                self.assertNotIn("t0ken", (ws / ".git" / "config").read_text())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


import redact  # noqa: E402
import secrets_env  # noqa: E402


class TestRedact(unittest.TestCase):
    ENV = {"TG_BOT_TOKEN": "123456:ABC-secret-token", "TG_CHAT_ID": "206751209", "GITHUB_TOKEN": "",
           "A2179_CI_SECRET_NAMES": "SECRET_X TG_SHORT", "SECRET_X": "hunter22", "TG_SHORT": "7"}

    def test_presence_only(self):
        self.assertEqual(redact.presence(self.ENV), {
            "GITHUB_TOKEN": "absent", "TG_BOT_TOKEN": "present", "TG_CHAT_ID": "present",
            "SECRET_X": "present", "TG_SHORT": "present (short, not scrubbed)"})

    def test_scrub_everywhere(self):
        text = "GET https://api.telegram.org/bot123456:ABC-secret-token/getChat?chat_id=206751209 pw hunter22"
        out = redact.scrub(text, self.ENV)
        for v in ("123456:ABC-secret-token", "206751209", "hunter22"):
            self.assertNotIn(v, out)
        self.assertIn("***", out)
        obj = redact.scrub_obj({"a": [text, 3], "b": {"c": text}}, self.ENV)
        self.assertNotIn("hunter22", json.dumps(obj))
        self.assertEqual(obj["a"][1], 3)

    def test_results_file_never_contains_values(self):
        env = {**ENV, **self.ENV}
        with tempfile.TemporaryDirectory() as d:
            ci_out(Path(d), [step("tests"), {**step("eval", ok=False),
                                              "stderr_tail": "boom bot123456:ABC-secret-token chat 206751209"}])
            (Path(d) / "setup.log").write_text("token 123456:ABC-secret-token\n")
            res = wr.build(Path(d), env=env, now=NOW)
            old = dict(os.environ)
            os.environ.update(self.ENV)
            try:
                p = wr.write(Path(d) / "results", res)
                rep = wr.attach_reports(Path(d), p)
                blob = p.read_text() + "".join(f.read_text() for f in rep.iterdir())
            finally:
                os.environ.clear()
                os.environ.update(old)
        for v in ("123456:ABC-secret-token", "206751209", "hunter22"):
            self.assertNotIn(v, blob)
        self.assertEqual(res["secrets"]["TG_BOT_TOKEN"], "present")

    def test_steps_do_not_get_secrets(self):
        old = dict(os.environ)
        os.environ.update(self.ENV)
        try:
            plain, full = rx.step_env(), rx.step_env(with_secrets=True)
        finally:
            os.environ.clear()
            os.environ.update(old)
        for n in ("TG_BOT_TOKEN", "TG_CHAT_ID", "SECRET_X"):
            self.assertNotIn(n, plain)
            self.assertIn(n, full)

    def test_run_scrubs_output(self):
        old = dict(os.environ)
        os.environ.update(self.ENV)
        try:
            r = rx.run("x", [sys.executable, "-c", "print('leak 123456:ABC-secret-token')"], 30, with_secrets=True)
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertNotIn("ABC-secret-token", r["stdout_tail"])


class TestSecretsEnv(unittest.TestCase):
    def test_blob_masked_and_exported(self):
        out = secrets_env.lines(json.dumps({"SECRET_DB": "pa ss", "TG_OTHER": "x1y2", "PATH": "/evil",
                                            "GITHUB_TOKEN": "no", "A2179_CI_DENSE": "off"}), env={})
        self.assertIn("echo '::add-mask::pa ss'", out)
        self.assertIn("export SECRET_DB='pa ss'", out)
        self.assertIn("export TG_OTHER=x1y2", out)
        self.assertFalse([ln for ln in out if ln.startswith(("export PATH", "export GITHUB_TOKEN", "export A2179_CI_DENSE"))])
        self.assertEqual(out[-1], "export A2179_CI_SECRET_NAMES='SECRET_DB TG_OTHER'")
        self.assertEqual(sum("ignored" in ln for ln in out), 3)
        # the warning about ignored keys never carries a value
        self.assertFalse([ln for ln in out if "/evil" in ln])

    def test_explicit_secret_wins(self):
        out = secrets_env.lines(json.dumps({"TG_BOT_TOKEN": "fromblob"}), env={"TG_BOT_TOKEN": "explicit"})
        self.assertFalse([ln for ln in out if "fromblob" in ln])

    def test_empty_and_bad(self):
        self.assertEqual(secrets_env.lines(""), ["export A2179_CI_SECRET_NAMES=''"])
        self.assertTrue(secrets_env.lines("[1]")[0].startswith("echo '::warning::"))


class TestTelegramIntegration(unittest.TestCase):
    def test_skips_without_secrets(self):
        env = {k: v for k, v in os.environ.items() if k not in ("TG_BOT_TOKEN", "TG_CHAT_ID")}
        p = subprocess.run([sys.executable, "-m", "unittest", "-v", "integration.test_telegram"], cwd=ROOT,
                           env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("skipped", p.stderr)

    def test_failure_message_has_no_token(self):
        import importlib
        old = dict(os.environ)
        os.environ.update({"TG_BOT_TOKEN": "999:SECRETTOKEN", "TG_CHAT_ID": "42424242"})
        try:
            import integration.test_telegram as tt
            tt = importlib.reload(tt)

            class R:
                status_code = 401

                def json(self):
                    return {"ok": False, "description": "Unauthorized for bot999:SECRETTOKEN"}
            import requests
            orig = requests.get
            requests.get = lambda url, **kw: R()
            try:
                status, ok, desc = tt.call("getMe")
            finally:
                requests.get = orig
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual((status, ok), (401, False))
        self.assertNotIn("SECRETTOKEN", desc)


class TestCacheModel(unittest.TestCase):
    def run_with(self, fake, env, argv=None):
        import importlib
        import types
        import cache_model
        mod = types.ModuleType("huggingface_hub")
        mod.snapshot_download = fake
        old_mod, old_env = sys.modules.get("huggingface_hub"), dict(os.environ)
        sys.modules["huggingface_hub"] = mod
        os.environ.update(env)
        try:
            return importlib.reload(cache_model).main(argv)
        finally:
            os.environ.clear()
            os.environ.update(old_env)
            if old_mod is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = old_mod

    def test_cached_no_download(self):
        calls = []

        def fake(name, local_files_only=False, **kw):
            calls.append(local_files_only)
            return "/hf/x"
        self.assertEqual(self.run_with(fake, {"A2179_CI_MODEL": "m", "HF_HUB_OFFLINE": "0"}), 0)
        self.assertEqual(calls, [True])

    def test_missing_downloads_only_when_online(self):
        calls = []

        def fake(name, local_files_only=False, **kw):
            calls.append(local_files_only)
            if local_files_only:
                raise FileNotFoundError(name)
            return "/hf/x"
        self.assertEqual(self.run_with(fake, {"A2179_CI_MODEL": "m", "HF_HUB_OFFLINE": "1"}), 4)
        self.assertEqual(calls, [True])
        self.assertEqual(self.run_with(fake, {"A2179_CI_MODEL": "m", "HF_HUB_OFFLINE": "0"}), 0)
        self.assertEqual(calls, [True, True, False])

    def test_model_list_only_needed_files(self):
        seen = []

        def fake(name, local_files_only=False, allow_patterns=None):
            seen.append((name, local_files_only, tuple(allow_patterns or ())))
            if local_files_only and name == "BAAI/bge-m3":
                raise FileNotFoundError(name)
            return "/hf/x"
        rc = self.run_with(fake, {"A2179_CI_MODELS": "e5-base, BAAI/bge-m3", "HF_HUB_OFFLINE": "0"})
        self.assertEqual(rc, 0)
        self.assertEqual([x[0] for x in seen], ["intfloat/multilingual-e5-base", "BAAI/bge-m3", "BAAI/bge-m3"])
        self.assertIn("model.safetensors", seen[0][2])
        self.assertNotIn("*.onnx", seen[0][2])
        self.assertIn("pytorch_model.bin", seen[-1][2])     # bge-m3 has no safetensors
        self.assertFalse(seen[-1][1])                       # downloaded only the missing one

    def test_one_failing_model_does_not_stop_the_rest(self):
        seen = []

        def fake(name, local_files_only=False, **kw):
            seen.append(name)
            if name == "bad/model":
                raise RuntimeError("401 gated")
            return "/hf/x"
        self.assertEqual(self.run_with(fake, {"A2179_CI_MODELS": "bad/model,e5-small", "HF_HUB_OFFLINE": "0"}), 5)
        self.assertIn("intfloat/multilingual-e5-small", seen)

    def test_default_model_from_config(self):
        import cache_model
        old = os.environ.pop("A2179_CI_MODEL", None)
        try:
            self.assertEqual(cache_model.model_name(), "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        finally:
            if old:
                os.environ["A2179_CI_MODEL"] = old

    def test_groups_and_dry_run_do_not_download(self):
        import cache_model
        ids = cache_model.resolve_specs(["multilingual", "english"])
        self.assertEqual(ids, [
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "intfloat/multilingual-e5-small",
            "intfloat/multilingual-e5-base",
            "BAAI/bge-m3",
            "BAAI/bge-base-en-v1.5",
            "intfloat/e5-base-v2",
        ])
        # unique ids: three e5-base shorts collapse, groups plus an id do not repeat
        self.assertEqual(len(cache_model.resolve_specs(["e5-base", "multilingual"])), 5)
        calls = []

        def fake(name, local_files_only=False, allow_patterns=None):
            calls.append((name, local_files_only))
            raise FileNotFoundError(name)

        rc = self.run_with(fake, {"HF_HUB_OFFLINE": "0", "A2179_CI_MODELS": ""},
                           ["--dry-run", "english"])
        self.assertEqual(rc, 0)
        self.assertEqual([c[0] for c in calls], ["BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"])
        self.assertTrue(all(c[1] for c in calls))  # only local_files_only probes


class TestSetupVenvTorchFlag(unittest.TestCase):
    def test_bad_torch_value_refused(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            venv = tmp / "venv"
            (venv / "bin").mkdir(parents=True)
            # -S: no site-packages, so a torch in the runner's own venv cannot leak into this fake one
            (venv / "bin" / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n')
            (venv / "bin" / "python").chmod(0o755)
            import hashlib
            (venv / ".stamp_requirements.txt").write_text(
                hashlib.sha256((ROOT / "requirements.txt").read_bytes()).hexdigest() + "\n")
            env = {**os.environ, "A2179_CI_VENV": str(venv), "A2179_CI_INSTALL_TORCH": "cu126; curl evil"}
            p = subprocess.run(["bash", "ci/setup_venv.sh"], cwd=ROOT, env=env, capture_output=True, text=True,
                               timeout=60)
            self.assertEqual(p.returncode, 3, p.stdout + p.stderr)
            self.assertIn("expected cu126", p.stdout)
            env["A2179_CI_INSTALL_TORCH"] = ""
            p = subprocess.run(["bash", "ci/setup_venv.sh"], cwd=ROOT, env=env, capture_output=True, text=True,
                               timeout=60)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("A2179_CI_INSTALL_TORCH=cu126", p.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestOwnVenvWithTorch(unittest.TestCase):
    def test_install_flag_is_noop_when_torch_imports(self):
        # A2179_CI_VENV pointing at an existing venv that already has torch (Leonid's .venv)
        tmp = Path(tempfile.mkdtemp())
        try:
            venv, site = tmp / ".venv", tmp / "site"
            (venv / "bin").mkdir(parents=True)
            (site / "torch").mkdir(parents=True)
            (site / "torch" / "__init__.py").write_text("__version__ = '9.9+fake'\n")
            (venv / "bin" / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
            (venv / "bin" / "python").chmod(0o755)
            import hashlib
            for req in ("requirements.txt", "requirements-embed.txt"):
                (venv / f".stamp_{req}").write_text(hashlib.sha256((ROOT / req).read_bytes()).hexdigest() + "\n")
            env = {**os.environ, "A2179_CI_VENV": str(venv), "A2179_CI_INSTALL_TORCH": "cu126",
                   "A2179_CI_PYTHON": "/nonexistent/python3.14", "PYTHONPATH": str(site), "A2179_CI_MODEL": "m",
                   # hermetic: no model download from a test (red run 3b678fa: the runner's venv has the real
                   # huggingface_hub, the cache step went online for the fake model "m" and failed setup)
                   "A2179_CI_CACHE_MODEL": "0"}
            p = subprocess.run(["bash", "ci/setup_venv.sh"], cwd=ROOT, env=env, capture_output=True, text=True,
                               timeout=60)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("torch already in", p.stdout)
            self.assertNotIn("installing torch", p.stdout)
            self.assertNotIn("creating venv", p.stdout)   # existing venv kept, A2179_CI_PYTHON not used
            self.assertIn("requirements-embed.txt unchanged", p.stdout)
            self.assertIn("model cache step off", p.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_failed_model_download_does_not_fail_setup(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            venv, site = tmp / ".venv", tmp / "site"
            (venv / "bin").mkdir(parents=True)
            (site / "torch").mkdir(parents=True)
            (site / "torch" / "__init__.py").write_text("__version__ = '9.9+fake'\n")
            # a huggingface_hub that has nothing cached and fails every download, no network touched
            (site / "huggingface_hub").mkdir()
            (site / "huggingface_hub" / "__init__.py").write_text(
                "def snapshot_download(name, local_files_only=False, **kw):\n"
                "    raise OSError('offline test: ' + name)\n")
            (venv / "bin" / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
            (venv / "bin" / "python").chmod(0o755)
            import hashlib
            for req in ("requirements.txt", "requirements-embed.txt"):
                (venv / f".stamp_{req}").write_text(hashlib.sha256((ROOT / req).read_bytes()).hexdigest() + "\n")
            env = {**os.environ, "A2179_CI_VENV": str(venv), "PYTHONPATH": str(site), "A2179_CI_MODEL": "m",
                   "A2179_CI_CACHE_MODEL": "1"}
            p = subprocess.run(["bash", "ci/setup_venv.sh"], cwd=ROOT, env=env, capture_output=True, text=True,
                               timeout=60)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("model cache step failed", p.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_sh_passes_ci_python_to_setup(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            env = {**os.environ, "A2179_CI_VARS": json.dumps({"A2179_CI_VENV": str(tmp / "new" / "venv"),
                                                              "A2179_CI_PYTHON": "/nonexistent/python3.14"}),
                   "A2179_CI_OUT": str(tmp / "out"), "A2179_CI_RESULTS_DIR": str(tmp / "results"),
                   "A2179_CI_NO_PUSH": "1"}
            env.pop("A2179_CI_PYTHON", None)
            subprocess.run(["bash", str(ROOT / "ci" / "run.sh")], cwd=tmp, env=env, capture_output=True,
                           text=True, timeout=60)
            log = (tmp / "out" / "setup.log").read_text()
            self.assertIn("/nonexistent/python3.14", log)   # setup_venv.sh tried exactly that python
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestHomePath(unittest.TestCase):
    def test_leading_tilde_expands_without_eval(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            p = subprocess.run(["bash", "-c", 'source ci/home_path.sh; expand_home_path "$1"; echo; '
                                              'expand_home_path "$2"; echo; expand_home_path "$3"',
                                "x", "~/a2179/.venv", "/abs/~/x", "~other/x"],
                               cwd=ROOT, env={**os.environ, "HOME": str(tmp)}, capture_output=True, text=True)
            self.assertEqual(p.stdout.splitlines(), [f"{tmp}/a2179/.venv", "/abs/~/x", "~other/x"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestVarsEnv(unittest.TestCase):
    def test_only_ci_keys_quoted(self):
        import vars_env
        out = vars_env.lines(json.dumps({"A2179_CI_DENSE": "off", "HF_HOME": "/data/hf", "OTHER": "x",
                                         "A2179_CI_PYTHON": "python3.14; rm -rf /"}))
        self.assertEqual(out, ["export A2179_CI_DENSE=off", "export A2179_CI_PYTHON='python3.14; rm -rf /'",
                               "export HF_HOME=/data/hf"])

    def test_bad_json(self):
        import vars_env
        self.assertEqual(len(vars_env.lines("not json")), 1)
        self.assertTrue(vars_env.lines("not json")[0].startswith("echo"))
        self.assertEqual(vars_env.lines("null"), [])


class TestRunSh(unittest.TestCase):
    """ci/run.sh end to end with a ready venv (this python), no pip, no tests, no push."""

    def test_writes_result_and_reports(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            venv = tmp / "venv"
            (venv / "bin").mkdir(parents=True)
            # a wrapper, not a symlink: a symlink to a venv python outside its venv starts the bare
            # base interpreter without the venv packages (seen on the CI runner, red run ebaed9c)
            # hermetic and fast on any runner: torch, sentence-transformers and matplotlib are hidden
            # from the steps (red run 42447e5: on the MX130 runner the real torch probe plus real
            # charts pushed this test past its 90 s timeout). The plumbing of run.sh is what is
            # tested here, the real steps run in CI itself.
            block = tmp / "block"
            block.mkdir()
            (block / "sitecustomize.py").write_text(
                "import sys\n"
                "HIDE = ('torch', 'sentence_transformers', 'transformers', 'matplotlib')\n"
                "class _Hide:\n"
                "    def find_spec(self, name, path=None, target=None):\n"
                "        if name.split('.')[0] in HIDE:\n"
                "            raise ModuleNotFoundError(f'hidden in test: {name}', name=name)\n"
                "        return None\n"
                "sys.meta_path.insert(0, _Hide())\n")
            (venv / "bin" / "python").write_text(
                f'#!/bin/sh\nPYTHONPATH="{block}${{PYTHONPATH:+:$PYTHONPATH}}" exec "{sys.executable}" "$@"\n')
            (venv / "bin" / "python").chmod(0o755)
            import hashlib
            (venv / ".stamp_requirements.txt").write_text(
                hashlib.sha256((ROOT / "requirements.txt").read_bytes()).hexdigest() + "\n")
            env = {**os.environ, "A2179_CI_VARS": json.dumps({"A2179_CI_VENV": str(venv), "A2179_CI_DENSE": "off",
                                                             "A2179_CI_FETCH_JOBS": "never"}),
                   "A2179_CI_OUT": str(tmp / "out"), "A2179_CI_RESULTS_DIR": str(tmp / "results"),
                   "A2179_CI_NO_PUSH": "1", "A2179_CI_SKIP_TESTS": "1", "GITHUB_SHA": ENV["GITHUB_SHA"],
                   # plumbing only: do not launch the one-shot runner experiments (they may fetch models)
                   "A2179_CI_RUNNER_EXP": "off"}
            env.pop("GITHUB_PATH", None)
            p = subprocess.run(["bash", str(ROOT / "ci" / "run.sh")], cwd=tmp, env=env, capture_output=True,
                               text=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stdout[-2000:] + p.stderr[-2000:])
            self.assertIn("hidden in test: torch", (tmp / "out" / "steps.json").read_text())
            res = list((tmp / "results").glob("*_0123456.json"))
            self.assertEqual(len(res), 1)
            r = json.loads(res[0].read_text())
            names = [s["name"] for s in r["steps"]]
            # sweep[de_jobs] only runs once eval/de_jobs/labeled.jsonl exists (eval[de_jobs] skipped otherwise)
            de = ["sweep[de_jobs]"] if (ROOT / "eval" / "de_jobs" / "labeled.jsonl").is_file() else []
            self.assertEqual(names, ["setup_venv", "env_probe", "lint", "eval", "sweep", "eval[de_jobs]", *de, "plots", "review",
                                     "tg_integration"])
            self.assertTrue(r["ok"], [(s["name"], s["exit_code"], s.get("stderr_tail")) for s in r["steps"]
                                      if not s.get("ok")])
            self.assertEqual(r["secrets"]["TG_BOT_TOKEN"], "absent")
            self.assertEqual(r["dense"]["reason"], "disabled (A2179_CI_DENSE=off)")
            self.assertTrue(r["dense"]["hf_offline"])
            rep = res[0].with_suffix("")
            self.assertTrue((rep / "eval_report.md").is_file())
            self.assertTrue((rep / "sweep.tables.md").is_file())
            # charts (or, without matplotlib, the ranking table) land next to the tables and inside them
            self.assertTrue((rep / "plots.md").is_file())
            self.assertIn("Ranking metrics", (rep / "sweep.tables.md").read_text())
            self.assertIn("tfidf hybrid", r["metrics"]["ranking"])
            self.assertTrue(any((rep / "reports").rglob("a2179_screen/matches.csv")))
            self.assertIsNotNone(r["metrics"].get("review"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_setup_failure_still_writes_result(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            env = {**os.environ, "A2179_CI_VARS": json.dumps({"A2179_CI_VENV": str(tmp / "nope" / "venv"),
                                                              "A2179_CI_PYTHON": "/nonexistent/python"}),
                   "A2179_CI_OUT": str(tmp / "out"), "A2179_CI_RESULTS_DIR": str(tmp / "results"),
                   "A2179_CI_NO_PUSH": "1", "GITHUB_SHA": ENV["GITHUB_SHA"]}
            p = subprocess.run(["bash", str(ROOT / "ci" / "run.sh")], cwd=tmp, env=env, capture_output=True,
                               text=True, timeout=60)
            self.assertNotEqual(p.returncode, 0)
            r = json.loads(next((tmp / "results").glob("*.json")).read_text())
            self.assertFalse(r["ok"])
            self.assertEqual(r["steps"][0]["name"], "setup_venv")
            self.assertFalse(r["steps"][0]["ok"])
            self.assertIn("nonexistent", r["steps"][0]["stderr_tail"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestSelectSteps(unittest.TestCase):
    """ci/select_steps.py (ADR-0026): docs skip, shared code full, the rest targeted."""
    def setUp(self):
        import select_steps
        self.S = select_steps

    def test_docs_only_skips(self):
        d = self.S.decide(["docs/ci.md", "docs/adr/0026-x.md", "experiments/EXP-005.md", "reports/exp004/a.json",
                           "work/T-019.md", "README.md"])
        self.assertEqual(d["mode"], "skip")

    def test_shared_code_runs_everything(self):
        for p in ("a2179/ats.py", "ci/run.sh", "eval/sweep.py", "targets/de_jobs.json", "tests/__init__.py"):
            self.assertEqual(self.S.decide([p, "docs/x.md"])["mode"], "full", p)

    def test_targeted_picks_tests_and_the_experiment(self):
        d = self.S.decide(["eval/jobs_label.py", "eval/de_jobs/labeled.jsonl", "docs/x.md"])
        self.assertEqual(d["mode"], "targeted")
        self.assertIn("tests.test_jobs_label", d["tests"])
        self.assertEqual(d["steps"], ["review", "targets"])
        d = self.S.decide(["eval/nested_cv.py"])
        self.assertEqual((d["tests"], d["steps"]), (["tests.test_nested_cv", "tests.test_nested_cv_fusion"], []))

    def test_forced_or_empty_is_full(self):
        self.assertEqual(self.S.decide(["docs/x.md"], forced="full")["mode"], "full")
        self.assertEqual(self.S.decide([])["mode"], "full")


if __name__ == "__main__":
    unittest.main()


class TestFetchRequest(unittest.TestCase):
    def test_request_runs_once_and_never_in_tests_by_default(self):
        import run_experiments as rx
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "ci" / "requests").mkdir(parents=True)
            self.assertIsNone(rx.fetch_request(tmp))
            (tmp / "ci" / "requests" / "fetch_listings.json").write_text('{"id": "r1", "max_pages": 2}')
            self.assertIsNone(rx.fetch_request(tmp))          # tests: A2179_CI_FETCH_LISTINGS=off
            with mock.patch.dict(os.environ, {"A2179_CI_FETCH_LISTINGS": "auto"}):
                self.assertEqual(rx.fetch_request(tmp)["id"], "r1")
                done = tmp / "results" / "2026-11-20_abc" / "reports" / "fresh_a2179" / "r1"
                done.mkdir(parents=True)
                (done / "candidates.jsonl").write_text("")
                self.assertIsNone(rx.fetch_request(tmp))      # fulfilled
            calls = []
            with mock.patch.object(rx, "run", lambda name, cmd, timeout, env=None: calls.append((name, cmd, timeout)) or {"name": name, "ok": True}):
                rec = rx.fetch_step("py", tmp / "out", 1800, {"id": "r2", "max_pages": 2, "time_budget_s": 60})
            self.assertTrue(rec["advisory"])
            name, cmd, timeout = calls[0]
            self.assertEqual((name, timeout), ("fetch_listings", 360))
            self.assertIn(str(tmp / "out" / "reports" / "fresh_a2179" / "r2" / "candidates.jsonl"), cmd)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
