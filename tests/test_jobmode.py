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


class Rerank(unittest.TestCase):
    """ADR-0036 (H-43): rank mix of the base score and a title reranker, no model loaded here."""
    def test_rank_mix_and_fields(self):
        base = lambda j: {"cv": 0.5, "feat": 0.5, "score": {"https://x/1": 0.9, "https://x/2": 0.1, "https://x/3": 0.5}[j.url]}  # noqa: E731
        fake = lambda titles: [3.0 if "MLOps" in t else 1.0 if "Data" in t else 0.0 for t in titles]  # noqa: E731  fake reranker for the mix only
        ranked = JM.rank(jobs(), base, fake)
        s = {j.url: s for j, s in ranked}
        self.assertEqual(s["https://x/1"]["base"], 0.9)
        self.assertIn("ce", s["https://x/3"])
        # base ranks 1 > 3 > 2, reranker 3 first: mixed 1 and 3 tie at 0.75, 2 last
        self.assertEqual(ranked[-1][0].url, "https://x/2")
        self.assertEqual(s["https://x/1"]["score"], s["https://x/3"]["score"])
        self.assertIn("реранк", JM.message(ranked[0][0], ranked[0][1]))

    def test_off_and_unavailable_keep_old_ranking(self):
        self.assertIsNone(JM.load_reranker("off")[0])
        r, _ = self.__class__._run_off()
        self.assertTrue(all("ce" not in s for _, s in r["ranked"]))
        self.assertEqual(r["rerank"], "rerank off")

    @staticmethod
    def _run_off():
        d = Path(tempfile.mkdtemp())
        try:
            al = FakeAlerter()
            return JM.run_jobs({}, d, al, dry_run=True, collect=lambda: (jobs(), [])), al
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_same_model_and_query_as_exp022(self):
        sys.path.insert(0, str(ROOT / "eval"))
        import rerank_jobs as RJ
        self.assertEqual((JM.RERANK_MODEL, JM.RERANK_QUERY, JM.RERANK_FILES), (RJ.MODEL, RJ.QUERY, RJ.FILES))


if __name__ == "__main__":
    unittest.main()
