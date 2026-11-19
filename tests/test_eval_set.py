import json
import unittest
from collections import Counter

from tests.helpers import ROOT
from a2179.config import DEFAULTS
from a2179.matcher import Matcher
from a2179.similarity import TfidfSim
from a2179 import prototypes


class TestEvalSet(unittest.TestCase):
    def setUp(self):
        p = ROOT / "eval" / "labeled.jsonl"
        self.rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    def test_size_and_mix(self):
        self.assertGreaterEqual(len(self.rows), 60)
        langs = Counter(r["lang"] for r in self.rows)
        for lang in ("sr", "en", "ru"):
            self.assertGreater(langs[lang], 0)
        self.assertGreater(sum(r["label"] for r in self.rows), 20)
        self.assertEqual(len({r["id"] for r in self.rows}), len(self.rows))

    def test_prototypes_not_copied_from_eval(self):
        titles = {r["title"].strip().lower() for r in self.rows}
        for p in prototypes.POSITIVE + prototypes.NEGATIVE:
            self.assertNotIn(p.strip().lower(), titles)

    def test_default_hybrid_has_full_recall(self):
        m = Matcher(TfidfSim(), "tfidf", DEFAULTS["thresholds"])
        missed = [r["title"] for r in self.rows if r["label"] and not m.decide(r["title"], r["snippet"]).match]
        self.assertEqual(missed, [])


if __name__ == "__main__":
    unittest.main()
