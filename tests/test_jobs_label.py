"""eval/jobs_label.py: sheet from CI review reports, merge only reviewed ESCI rows (T-019)."""
import csv
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
                r.update(reviewed="1", esci="C")
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=J.COLS)
            w.writeheader()
            w.writerows(sheet)
        new = J.merge(out)
        self.assertEqual([(r["title"], r["label"], r["esci"]) for r in new],
                         [("Senior Data Engineer", 1, "E"), ("Data Analyst", 0, "C")])
        self.assertEqual(J.merge(out), [])   # idempotent
        self.assertEqual(len(J.sheet(self.tmp / "reports", out)), 1)   # labeled rows leave the sheet


if __name__ == "__main__":
    unittest.main()
