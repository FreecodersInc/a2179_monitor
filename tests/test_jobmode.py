"""T-036: job mode ranks by CV closeness, pings only the top new vacancies, seed and dedup."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from a2179 import jobmode as JM  # noqa: E402
from a2179.jobs import Job  # noqa: E402


class FakeAlerter:
    def __init__(self):
        self.sent = []

    def send(self, msgs):
        self.sent.extend(msgs)
        return {"console": len(msgs)}


def jobs():
    return [Job("remotive", "https://x/1", "Senior Data Engineer", "A", text="python spark airflow kafka aws terraform"),
            Job("remotive", "https://x/2", "Account Executive", "B", text="sales quota"),
            Job("remotive", "https://x/3", "Staff MLOps Platform Engineer", "C", text="kubernetes sagemaker python")]


class JobMode(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def run_(self, **kw):
        al = FakeAlerter()
        r = JM.run_jobs({}, self.d, al, collect=lambda: (jobs(), []), **kw)
        return r, al

    def test_real_scorer_ranks_data_engineer_over_sales(self):
        r, _ = self.run_(dry_run=True, top=3)
        titles = [j.title for j, _ in r["ranked"]]
        self.assertLess(titles.index("Senior Data Engineer"), titles.index("Account Executive"))

    def test_top_dedup_and_seed(self):
        r, al = self.run_(top=1)
        self.assertEqual(len(al.sent), 1)
        r2, al2 = self.run_(top=1)
        self.assertEqual(len(al2.sent), 1)
        self.assertNotEqual(al.sent[0], al2.sent[0])     # the first one is not pinged again
        r3, al3 = self.run_(seed=True)
        self.assertEqual(al3.sent, [])
        r4, al4 = self.run_(top=5)
        self.assertEqual(al4.sent, [])                   # everything seen after seed

    def test_dry_run_writes_nothing(self):
        self.run_(dry_run=True)
        self.assertFalse((self.d / "jobs_seen.json").exists())


if __name__ == "__main__":
    unittest.main()
