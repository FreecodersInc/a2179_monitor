"""eval/cv_distance.py: parts of the CV distance score (T-024) on fixed examples, no network."""
import sys
import unittest

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "eval"))
import cv_distance as C  # noqa: E402
from a2179.target import load_target  # noqa: E402


class TestCvDistance(unittest.TestCase):
    def setUp(self):
        self.tg = load_target("de_jobs")
        self.stack = C.stack_terms({"plus": ["Kafka", "dbt"], "groups": {"x": ["Apache Spark", "SQL", "AWS"]}})

    def test_parts(self):
        p = C.parts("Senior Data Engineer", "We use SQL, Python, Apache Spark, Kafka and AWS daily", self.tg, self.stack)
        self.assertEqual(p["skills"], 1.0)
        self.assertEqual(p["stack"], round(4 / C.STACK_SAT, 3))  # no dbt in the text
        self.assertEqual((p["domain"], p["seniority"]), (1.0, 1.0))
        q = C.parts("Account Executive", "Sell software", self.tg, self.stack)
        self.assertLess(q["cv_score"], p["cv_score"])
        self.assertEqual(C.parts("Data Engineering Intern", "", self.tg, self.stack)["seniority"], 0.2)
        self.assertEqual(C.parts("Director, Data Platform", "", self.tg, self.stack)["seniority"], 0.4)

    def test_term_rx_word_bounds(self):
        self.assertTrue(C.term_rx("SQL").search("strong sql skills"))
        self.assertFalse(C.term_rx("SQL").search("nosqlish"))

    def test_auc_and_spearman(self):
        self.assertEqual(C.auc([0.9, 0.8, 0.1], [1, 1, 0]), 1.0)
        self.assertEqual(C.auc([0.5, 0.5], [1, 0]), 0.5)
        self.assertAlmostEqual(C.spearman([1, 2, 3], [10, 20, 30]), 1.0)


class TestProfileV2AndRulesNow(unittest.TestCase):
    def test_v2_roles_match(self):
        from a2179.rules import classify
        tg = load_target("de_jobs")
        for t in ("Senior Site Reliability Engineer", "DevOps Engineer", "Machine Learning Engineer", "Senior Data Engineer"):
            self.assertTrue(classify(t, "", target=tg).match, t)
        self.assertFalse(classify("Account Executive", "", target=tg).match)

    def test_ht_rules_weights_strata(self):
        import strata_eval as SE
        rows = [{"title": "Senior Data Engineer", "esci": "E", "stratum": "matcher"},
                {"title": "Account Executive", "esci": "I", "stratum": "random"},
                {"title": "Data Platform Lead", "esci": "S", "stratum": "random"}]
        r = SE.ht_rules(rows, {"matcher": {"pool": 10}, "random": {"pool": 100}}, {"E", "S"}, reps=50)
        from a2179.rules import classify
        tg = load_target("de_jobs")
        self.assertEqual(r["matches_in_sample"], sum(1 for x in rows if classify(x["title"], "", target=tg).match))
        self.assertGreaterEqual(r["precision"], 0.0)
        self.assertLessEqual(r["recall"], 1.0)
