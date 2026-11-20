"""ADR-0037: A2179 stays rules first. Similarity alone never matches without the model gate, and the
similarity text is one string (title + snippet), no separate title embedding (H-24, H-12)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from a2179.matcher import Matcher  # noqa: E402


class AlwaysHigh:
    """Stub similarity that records the texts it sees and scores every one as a perfect positive."""
    def __init__(self):
        self.seen = []

    def score(self, text):
        self.seen.append(text)
        return 1.0, 0.0


class RulesPrimary(unittest.TestCase):
    def test_similarity_needs_the_gate(self):
        m = Matcher(AlwaysHigh(), mode="tfidf")
        self.assertFalse(m.decide("Futrola za iPhone 13", "").match)     # no compatible model: gate closed

    def test_regex_match_without_similarity(self):
        self.assertTrue(Matcher(None).decide("LCD ekran za MacBook Air A2179 2020", "").match)

    def test_one_text_for_similarity(self):
        sim = AlwaysHigh()
        Matcher(sim, mode="tfidf").decide("Ekran MacBook Air 2020", "A2179 original")
        self.assertEqual(sim.seen, ["Ekran MacBook Air 2020 A2179 original"])


if __name__ == "__main__":
    unittest.main()
