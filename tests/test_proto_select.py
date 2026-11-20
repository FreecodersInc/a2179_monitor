"""EXP-013: candidate prototype sets and the nested selection never look at test rows."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import proto_select as P  # noqa: E402
from a2179.target import load_target  # noqa: E402


class ProtoSelect(unittest.TestCase):
    def test_candidates(self):
        tg = load_target("a2179_screen")
        c = P.candidates(tg, 3)
        self.assertEqual(list(c), ["full", "half0", "half1", "half2"])
        self.assertEqual(len(c["full"][0]), len(tg.positive))
        self.assertLess(len(c["half0"][0]), len(tg.positive))

    def test_selection_ignores_test_labels(self):
        y = [True, False, True, False, True, False]
        reg, gate, feat = [False] * 6, [True] * 6, [0.5] * 6
        good = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
        scores = {"full": [0.5] * 6, "half0": good}
        folds = [[0, 1], [2, 3], [4, 5]]
        pred, chosen = P.select_run(y, reg, gate, feat, scores, False, folds, 0.0, True)
        self.assertEqual(chosen, ["half0"] * 3)
        y2 = y[:]
        y2[0], y2[1] = False, True   # flip labels of fold 0 only: what fold 0 picks must not change
        _, chosen2 = P.select_run(y2, reg, gate, feat, scores, False, folds, 0.0, True)
        self.assertEqual(chosen2[0], chosen[0])


if __name__ == "__main__":
    unittest.main()
