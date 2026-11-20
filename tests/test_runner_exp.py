"""EXP-021/22/23 runner experiments and the once-per-day KP request. No model and no network."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "ci"))

import granularity as G  # noqa: E402
import rerank_jobs as RJ  # noqa: E402
import nested_cv_jobs as N  # noqa: E402
import run_experiments as R  # noqa: E402


class DailyKP(unittest.TestCase):
    def test_once_per_day(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"A2179_CI_FETCH_LISTINGS": "auto"}):
            self.assertIsNone(R.kp_daily_request(d, "2026-11-20"))
            p = Path(d) / "ci" / "requests"
            p.mkdir(parents=True)
            (p / "kp_daily.json").write_text(json.dumps({"id": "kp-v1", "queries_file": "q.txt"}))
            self.assertIsNotNone(R.kp_daily_request(d, "2026-11-20"))
            done = Path(d) / "results" / "x" / "reports" / "kp_daily" / "2026-11-20"
            done.mkdir(parents=True)
            (done / "candidates.jsonl").write_text("")
            self.assertIsNone(R.kp_daily_request(d, "2026-11-20"))
            self.assertIsNotNone(R.kp_daily_request(d, "2026-11-21"))

    def test_cmd_low_rate(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(R, "run", return_value={}) as run:
            R.kp_daily_step("python", Path(d), 600, {"date": "2026-11-20", "queries_file": "q.txt"})
            cmd = run.call_args[0][1]
            self.assertEqual(cmd[cmd.index("--max-pages") + 1], "1")
            self.assertEqual(cmd[cmd.index("--shops") + 1], "none")
            self.assertIn("--summary", cmd)


class RunnerRequests(unittest.TestCase):
    def test_once_per_step(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"A2179_CI_RUNNER_EXP": "auto"}):
            self.assertIsNone(R.runner_request(d))
            p = Path(d) / "ci" / "requests"
            p.mkdir(parents=True)
            (p / "runner_experiments.json").write_text(json.dumps({"id": "v1", "steps": ["granularity", "rerank_jobs"]}))
            self.assertEqual(R.runner_request(d)["steps"], ["granularity", "rerank_jobs"])
            done = Path(d) / "results" / "x" / "reports" / "runner_exp" / "v1"
            done.mkdir(parents=True)
            (done / "granularity.json").write_text("{}")
            self.assertEqual(R.runner_request(d)["steps"], ["rerank_jobs"])
            (done / "rerank_jobs.json").write_text("{}")
            self.assertIsNone(R.runner_request(d))


class Metrics(unittest.TestCase):
    def test_auc_matches_reference_on_real_rows(self):
        import cv_distance as CV
        rows = [json.loads(l) for l in (ROOT / "eval" / "de_jobs" / "labeled.jsonl").read_text().splitlines()]
        # actual stored per-row binary labels, score = title length, only to check the AUC implementation
        y = [r["esci_text"] in {"E", "S"} for r in rows]
        s = [len(r["title"]) for r in rows]
        self.assertAlmostEqual(N.fast_auc(s, y), CV.auc(s, y))

    def test_size_rejects_unknown_and_over_cap(self):
        sizes = {f: 10 for f in RJ.FILES}
        self.assertTrue(RJ.check_size(sizes, 100)[0])
        self.assertFalse(RJ.check_size(sizes, 50)[0])
        self.assertFalse(RJ.check_size({}, 100)[0])
        sizes[RJ.FILES[0]] = None
        self.assertFalse(RJ.check_size(sizes, 100)[0])

    def test_granularity_no_snippet_means_no_difference(self):
        rows = [json.loads(l) for l in (ROOT / "eval" / "labeled.jsonl").read_text().splitlines()]
        rows = [r for r in rows if not r.get("snippet") and not str(r.get("source", "")).startswith("synth")][:20]
        # test the composition only, not an experiment: score = actual title length
        V = G.variants(rows, lambda ts: [(float(len(t)), 0.0) for t in ts])
        for k in G.VARIANTS:
            self.assertEqual(V[k], V["title"])



class Exp028DryRun(unittest.TestCase):
    def test_dry_run_writes_plan_without_torch(self):
        import exp028_blocked_models as E
        with tempfile.TemporaryDirectory() as d:
            rc = E.main(["--out", d, "--dry-run"])
            self.assertEqual(rc, 0)
            doc = json.loads((Path(d) / "exp028.json").read_text())
            self.assertEqual(doc["device"], "cpu")
            self.assertTrue(doc["dry_run"])
            self.assertEqual(len(doc["models"]), 4)
            self.assertTrue(all(m["status"] == "dry-run" for m in doc["models"]))
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")
            # no listing texts in the artifact
            blob = (Path(d) / "exp028.json").read_text()
            self.assertNotIn("snippet", blob.lower())

    def test_runner_request_picks_exp028(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"A2179_CI_RUNNER_EXP": "auto"}):
            p = Path(d) / "ci" / "requests"
            p.mkdir(parents=True)
            (p / "runner_experiments.json").write_text(json.dumps({"id": "v2", "steps": ["exp028"]}))
            self.assertEqual(R.runner_request(d)["steps"], ["exp028"])
            done = Path(d) / "results" / "x" / "reports" / "runner_exp" / "v2"
            done.mkdir(parents=True)
            (done / "exp028.json").write_text("{}")
            self.assertIsNone(R.runner_request(d))


class Exp029DryRun(unittest.TestCase):
    def test_dry_run_or_no_cuda_is_clean(self):
        import exp029_h11 as H
        with tempfile.TemporaryDirectory() as d:
            rc = H.main(["--out", d, "--dry-run"])
            self.assertEqual(rc, 0)
            doc = json.loads((Path(d) / "exp029_h11.json").read_text())
            self.assertIn(doc["status"], ("blocked", "dry-run"))
            if doc["status"] == "blocked":
                self.assertEqual(doc.get("suggested_verdict"), "blocked")
                self.assertIn("CUDA", (doc.get("reason") or ""))
            blob = (Path(d) / "exp029_h11.json").read_text().lower()
            self.assertNotIn("snippet", blob)
            self.assertNotIn("ekran za macbook", blob)


if __name__ == "__main__":
    unittest.main()
