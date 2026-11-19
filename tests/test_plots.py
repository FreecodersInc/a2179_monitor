import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "ci"))
import plots as P  # noqa: E402
import write_results as W  # noqa: E402

try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

DATA = ROOT / "tests" / "fixtures" / "labeled_base.jsonl"     # frozen 99 rows
SWEEP = ROOT / "tests" / "fixtures" / "sweep_99.json"   # frozen sweep on those rows (reports/ gets rerun)


class TestRanking(unittest.TestCase):
    def test_hand_computed(self):
        y = [1, 0, 1, 0, 0, 1]
        m = P.ranking_metrics(y, [0, 1, 2, 3, 4, 5])   # relevant at ranks 1, 3, 6
        self.assertEqual(m["npos"], 3)
        self.assertEqual(m["R@npos"], round(2 / 3, 3))
        self.assertEqual(m["R@2npos"], 1.0)
        self.assertEqual(m["depth@R1"], 6)
        self.assertEqual(m["P@R1"], 0.5)
        self.assertEqual(m["P@R95"], 0.5)          # 0.95 * 3 rounds up to all 3
        self.assertEqual(m["MRR"], 1.0)
        self.assertEqual(m["AP"], round((1 + 2 / 3 + 3 / 6) / 3, 3))

    def test_perfect_order(self):
        m = P.ranking_metrics([0, 1, 1], [1, 2, 0])
        self.assertEqual((m["P@R1"], m["AP"], m["nDCG@npos"], m["depth@R1"]), (1.0, 1.0, 1.0, 2))

    def test_hybrid_order_puts_regex_first_and_gate_drops_last(self):
        sw = mock.Mock(regex=[False, True, False], gate=[True, True, False])
        self.assertEqual(P.order(sw, [0.9, 0.1, 1.0], True), [1, 0, 2])
        self.assertEqual(P.order(sw, [0.9, 0.1, 1.0], False), [2, 0, 1])


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_without_matplotlib_still_writes_metrics(self):
        with mock.patch.object(P, "_plt", side_effect=ImportError("no matplotlib")):
            meta = P.build(SWEEP, DATA, self.tmp)
        self.assertEqual(meta["charts"], [])
        self.assertIn("matplotlib", meta["note"])
        self.assertIn("tfidf hybrid", meta["ranking"])
        md = (self.tmp / "plots.md").read_text()
        self.assertIn("| tfidf | hybrid |", md)
        self.assertEqual(json.loads((self.tmp / "plots.json").read_text())["n"], 99)

    def test_scores_must_cover_the_data(self):
        res = json.loads(SWEEP.read_text())
        res["scores"].pop("1")
        p = self.tmp / "s.json"
        p.write_text(json.dumps(res))
        with self.assertRaises(SystemExit):
            P.build(p, DATA, self.tmp / "o")

    @unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
    def test_all_charts_and_the_report_embeds_them(self):
        meta = P.build(SWEEP, DATA, self.tmp)
        self.assertEqual(len(meta["charts"]), 9)
        for name in meta["charts"]:
            b = (self.tmp / name).read_bytes()
            self.assertTrue(b.startswith(b"\x89PNG"), name)
            self.assertIn(f"]({name})", (self.tmp / "plots.md").read_text())


class TestAttach(unittest.TestCase):
    def test_pngs_copied_and_embedded_and_not_gating(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            ci = tmp / "ci_out"
            (ci / "sweep").mkdir(parents=True)
            (ci / "plots").mkdir()
            (ci / "sweep" / "sweep.tables.md").write_text("### Variants\n")
            (ci / "plots" / "01_dataset.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
            (ci / "plots" / "plots.md").write_text("## Charts\n\n![01_dataset](01_dataset.png)\n")
            (ci / "plots" / "plots.json").write_text(json.dumps({"ranking": {"tfidf hybrid": {"AP": 0.9}}}))
            d = W.attach_reports(ci, tmp / "results" / "2026-11-19_abc.json")
            self.assertEqual((d / "01_dataset.png").read_bytes(), b"\x89PNG\r\n\x1a\nfake")
            t = (d / "sweep.tables.md").read_text()
            self.assertTrue(t.startswith("### Variants"))
            self.assertIn("![01_dataset](01_dataset.png)", t)
            (ci / "steps.json").write_text(json.dumps({"steps": [
                {"name": "sweep", "ok": True}, {"name": "plots", "ok": False, "advisory": True}]}))
            r = W.build(ci, env={"GITHUB_SHA": "0" * 40})
            self.assertTrue(r["ok"])
            self.assertEqual(r["metrics"]["ranking"]["tfidf hybrid"]["AP"], 0.9)
            self.assertTrue(r["steps"][-1]["advisory"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
