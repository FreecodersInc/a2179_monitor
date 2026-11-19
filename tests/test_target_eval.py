"""T-002 / T-004: eval and sweep per target profile, CI loops the other targets."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "ci"))
sys.path.insert(0, str(ROOT / "eval"))
import run_experiments as rx  # noqa: E402
import write_results as wr  # noqa: E402
from a2179.target import eval_data_path, load_target  # noqa: E402

JOBS = [  # a tiny hand-made set, only to exercise the plumbing (not a measurement)
    ("Senior Data Engineer (Python, Spark, Airflow), Remote Canada", "Build batch and streaming pipelines", 1),
    ("Data Engineer, dbt + Snowflake, Toronto", "ELT pipelines, data modelling", 1),
    ("Data Engineer, Kafka, remote EU (English)", "Streaming platform team", 1),
    ("Frontend Engineer, React", "Build UI components", 0),
    ("Data Analyst, Tableau dashboards", "Reporting for sales", 0),
    ("Sales Manager, SaaS", "Quota carrying role", 0),
    ("Machine Learning Researcher, PhD", "Publish papers", 0),
    ("Customer Support Specialist", "Help our customers", 0),
]


def write_jobs(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": i, "title": t, "snippet": s, "label": y, "lang": "en",
                                        "kind": "job", "source": "test"}) + "\n"
                            for i, (t, s, y) in enumerate(JOBS, 1)), encoding="utf-8")


def py(*args, cwd=ROOT):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=120,
                          env={**os.environ, "HF_HUB_OFFLINE": "1"})


class TestProfilePaths(unittest.TestCase):
    def test_labeled_set_from_profile(self):
        self.assertEqual(eval_data_path("a2179_screen"), ROOT / "eval" / "labeled.jsonl")
        self.assertEqual(eval_data_path("de_jobs"), ROOT / "eval" / "de_jobs" / "labeled.jsonl")
        self.assertEqual(load_target("a2179_screen").eval.get("data"), "eval/labeled.jsonl")


class TestFeatureGroups(unittest.TestCase):
    def test_profile_groups(self):
        import sweep as S
        rx = S.feature_rx("de_jobs")
        self.assertEqual(S.features("Senior Data Engineer, Spark", "", rx), (1.0, ["display"]))
        self.assertEqual(S.features("Data Analyst, Tableau", "", rx), (-1.0, ["analyst"]))
        self.assertIs(S.feature_rx("a2179_screen")[0], S.RX_DISPLAY)   # A2179 keeps its fixed groups


class TestEvalSweepPerTarget(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_de_jobs_runs_on_its_own_rules_and_prototypes(self):
        data = self.tmp / "jobs.jsonl"
        write_jobs(data)
        p = py("eval/run_eval.py", "--target", "de_jobs", "--data", str(data), "--no-embed",
               "--out-dir", str(self.tmp / "ev"), "--config", str(self.tmp / "x.json"))
        self.assertEqual(p.returncode, 0, p.stderr[-1500:])
        rep = json.loads((self.tmp / "ev" / "eval_report.json").read_text())
        self.assertEqual((rep["target"], rep["dataset"]["n"], rep["dataset"]["positives"]), ("de_jobs", 8, 3))
        self.assertIn("DE jobs", (self.tmp / "ev" / "eval_report.md").read_text())
        p = py("eval/sweep.py", "--target", "de_jobs", "--data", str(data), "--out", str(self.tmp / "sw" / "s.json"),
               "--dense-scores", str(self.tmp / "sw" / "none.jsonl"))
        self.assertEqual(p.returncode, 0, p.stderr[-1500:])
        sw = json.loads((self.tmp / "sw" / "s.json").read_text())
        self.assertEqual((sw["target"], sw["n"]), ("de_jobs", 8))
        # the A2179 display / hinge feature groups must not fire on job titles
        self.assertTrue(all("hinge" not in g for g in sw["features"].values()))

    def test_missing_labeled_set_is_exit_4(self):
        for script in ("eval/run_eval.py", "eval/sweep.py"):
            p = py(script, "--target", "de_jobs", "--data", str(self.tmp / "none.jsonl"),
                   *(["--out-dir", str(self.tmp)] if "eval" in script.split("/")[1] else ["--out", str(self.tmp / "s.json")]))
            self.assertEqual(p.returncode, 4, (script, p.stderr))
            self.assertIn("no labeled set", p.stderr)


class TestCiTargets(unittest.TestCase):
    def test_other_targets(self):
        self.assertEqual(rx.other_targets("a2179_screen,de_jobs,de_jobs, x;y ,"), ["de_jobs"])

    def test_skipped_without_labels_and_metrics(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fake = tmp / "fake.py"   # stands in for the venv python: exits 4 like a target with no set
            fake.write_text("import sys; print('target de_jobs: no labeled set at x', file=sys.stderr); sys.exit(4)\n")
            steps = rx.target_steps("de_jobs", sys.executable, tmp, False, True, 30)
            # with the real scripts: de_jobs has no eval/de_jobs/labeled.jsonl in the repo yet
            if not eval_data_path("de_jobs").is_file():
                self.assertEqual(len(steps), 1)
                self.assertTrue(steps[0]["ok"] and steps[0]["advisory"])
                self.assertIn("no labeled set", steps[0]["skipped"])
                m = wr.target_metrics(tmp, steps)
                self.assertIn("no labeled set", m["de_jobs"]["skipped"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_metrics_and_reports_per_target(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "targets" / "de_jobs"
            (d / "eval").mkdir(parents=True)
            (d / "sweep").mkdir(parents=True)
            shutil.copy(ROOT / "tests" / "fixtures" / "eval_report_99.json", d / "eval" / "eval_report.json")
            (d / "eval" / "eval_report.md").write_text("# Eval report: DE jobs\n")
            shutil.copy(ROOT / "tests" / "fixtures" / "sweep_99.json", d / "sweep" / "sweep.json")
            m = wr.target_metrics(tmp, [])
            self.assertIn("modes", m["de_jobs"]["eval"])
            self.assertIn("variants", m["de_jobs"]["sweep"])
            res = tmp / "results" / "r.json"
            res.parent.mkdir()
            res.write_text("{}")
            out = wr.attach_reports(tmp, res)
            self.assertTrue((out / "targets" / "de_jobs" / "eval_report.md").is_file())
            self.assertFalse((out / "targets" / "de_jobs" / "sweep.json").exists())   # big file stays out
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
