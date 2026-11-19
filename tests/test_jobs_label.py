"""eval/jobs_label.py: sheet from CI review reports, merge only reviewed ESCI rows (T-019)."""
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "eval"))
import jobs_label as J  # noqa: E402

ALL = [
    {"title": "Senior Data Engineer", "company": "A", "location": "Toronto", "score": "0.71", "decision": "match",
     "url": "https://x/1", "fetched_at": "t", "source": "remotive"},
    {"title": "Data Analyst", "company": "B", "location": "Remote", "score": "0.45", "decision": "no",
     "url": "https://x/2", "fetched_at": "t", "source": "remoteok"},
    {"title": "Senior Data Engineer", "company": "A", "location": "Toronto", "score": "0.71", "decision": "match",
     "url": "https://x/1", "fetched_at": "t", "source": "remotive"},   # duplicate
    {"title": "Barista", "company": "C", "location": "EU", "score": "0.05", "decision": "no",
     "url": "https://x/3", "fetched_at": "t", "source": "remoteok"},
]


class TestJobsLabel(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        d = self.tmp / "reports" / "2026-11-19" / "de_jobs"
        d.mkdir(parents=True)
        with (d / "all.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ALL[0]))
            w.writeheader()
            w.writerows(ALL)
        self.dest = self.tmp / "labeled.jsonl"
        self.p = mock.patch.object(J, "eval_data_path", lambda t: self.dest)
        self.p.start()

    def tearDown(self):
        self.p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sheet_then_merge(self):
        out = self.tmp / "to_label.csv"
        rows = J.sheet(self.tmp / "reports", out)
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["pre_esci"] for r in rows if r["title"] == "Barista"}, {"I"})
        self.assertEqual([r["title"] for r in rows if r["title"] == "Senior Data Engineer"][0], "Senior Data Engineer")
        sheet = J.read_csv(out)
        for r in sheet:   # the human: reviews two rows, leaves one
            if r["title"] == "Senior Data Engineer":
                r.update(reviewed="1", esci="e")
            if r["title"] == "Data Analyst":
                r.update(reviewed="1", esci="C", labeler="bot", unsure="1")
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=J.COLS)
            w.writeheader()
            w.writerows(sheet)
        new = J.merge(out)
        self.assertEqual(sorted((r["title"], r["label"], r["esci"]) for r in new),
                         [("Data Analyst", 0, "C"), ("Senior Data Engineer", 1, "E")])
        by = {r["title"]: r for r in new}
        self.assertEqual((by["Data Analyst"]["labeler"], by["Data Analyst"]["review_unsure"]), ("bot", True))
        self.assertNotIn("labeler", by["Senior Data Engineer"])
        self.assertEqual(J.merge(out), [])   # idempotent
        self.assertEqual(len(J.sheet(self.tmp / "reports", out)), 1)   # labeled rows leave the sheet

    def test_capped_batch_keeps_matcher_rows_and_writes_strata(self):
        out = self.tmp / "to_label.csv"
        rows = J.sheet(self.tmp / "reports", out, max_rows=2, seed=0)
        self.assertEqual({r["stratum"] for r in rows if r["title"] == "Senior Data Engineer"}, {"matcher"})
        self.assertEqual(len(rows), 2)
        strata = json.loads((self.tmp / "to_label_strata.json").read_text())["strata"]
        self.assertEqual(strata["matcher"], {"pool": 1, "kept": 1})
        self.assertEqual(sum(v["pool"] for v in strata.values()), 3)
        again = J.sheet(self.tmp / "reports", out, max_rows=2, seed=0)   # seeded: same batch
        self.assertEqual([r["title"] for r in rows], [r["title"] for r in again])
        sheet = J.read_csv(out)
        for r in sheet:
            r.update(reviewed="1", esci="E" if r["stratum"] == "matcher" else "I")
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=J.COLS)
            w.writeheader()
            w.writerows(sheet)
        self.assertEqual({r["stratum"] for r in J.merge(out)} - {"matcher", "random", "hard_neg"}, set())


class TestText(unittest.TestCase):
    def test_mojibake_repaired_and_lines_stay_whole(self):
        import json
        from a2179.jobs import fix_text
        bad = "Senior Software Engineer \u00e2\u0080\u0094 Kubernetes"
        self.assertEqual(fix_text(bad), "Senior Software Engineer \u2014 Kubernetes")
        self.assertEqual(fix_text("M\u00fcnchen"), "M\u00fcnchen")   # real Latin-1 text stays
        self.assertEqual(fix_text("M\u00c3\u00bcnchen"), "M\u00fcnchen")
        line = J.jsonl_line({"title": "a\x85b\u2028c"})
        self.assertEqual(len(line.splitlines()), 1)
        self.assertEqual(json.loads(line)["title"], "a\x85b\u2028c")


if __name__ == "__main__":
    unittest.main()
