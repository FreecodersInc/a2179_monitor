"""EXP-012: RRF channel, title weight, operating point from the profile."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import nested_cv as N  # noqa: E402
from a2179.target import load_target  # noqa: E402


class Fusion(unittest.TestCase):
    def test_rank_against_train(self):
        self.assertEqual(N._rank(5, [1, 6, 9]), 3)
        self.assertEqual(N._rank(10, [1, 6, 9]), 1)

    def test_rrf_orders_by_both(self):
        base, feat = [0.9, 0.1, 0.5], [0.8, 0.2, 0.1]
        s = N.rrf(base, feat, [0, 1, 2], [0, 1, 2])
        self.assertGreater(s[0], s[2])
        self.assertGreater(s[0], s[1])
        self.assertAlmostEqual(s[0], 2 / 61)

    def test_pick_rrf_full_recall(self):
        y = [True, False, True, False]
        base, feat = [0.9, 0.2, 0.7, 0.1], [0.9, 0.1, 0.8, 0.3]
        t = N.pick_rrf(y, [False] * 4, [True] * 4, base, feat, [0, 1, 2, 3])
        s = N.rrf(base, feat, [0, 1, 2, 3], [0, 1, 2, 3])
        self.assertEqual([s[i] >= t for i in range(4)], y)

    def test_title_weight(self):
        self.assertEqual(N.build_text("data engineer", "body", 3), "data engineer data engineer data engineer body")
        self.assertEqual(N.build_text("x", "", 1), "x")

    def test_operating_point_from_profile(self):
        self.assertEqual(N.operating_point(load_target("a2179_screen")), 1.0)
        self.assertEqual(N.operating_point(load_target("de_jobs")), 0.0)


if __name__ == "__main__":
    unittest.main()
