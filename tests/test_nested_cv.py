"""eval/nested_cv.py: grouped folds, choices made on train only (EXP-004)."""
import random
import sys
import unittest

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "eval"))
import nested_cv as N  # noqa: E402


class TestFolds(unittest.TestCase):
    def test_groups_never_split_and_positives_spread(self):
        y = [i % 7 == 0 for i in range(70)]
        groups = [f"g{i // 2}" for i in range(70)]
        F = N.folds_grouped(y, groups, 5, random.Random(0))
        self.assertEqual(sorted(i for f in F for i in f), list(range(70)))
        where = {i: k for k, f in enumerate(F) for i in f}
        for i in range(0, 70, 2):
            self.assertEqual(where[i], where[i + 1])
        self.assertTrue(all(any(y[i] for i in f) for f in F))


class TestNoLeak(unittest.TestCase):
    def test_pick_ignores_test_rows(self):
        rnd = random.Random(1)
        n = 40
        y = [i % 4 == 0 for i in range(n)]
        base = [rnd.random() + (0.5 if y[i] else 0) for i in range(n)]
        feat = [1.0 if y[i] else rnd.choice([0.0, 0.5]) for i in range(n)]
        regex, gate = [False] * n, [True] * n
        tr = list(range(30))
        a = N.pick(y, regex, gate, base, feat, tr, True)
        base2 = base[:30] + [99.0] * 10
        y2 = y[:30] + [not v for v in y[30:]]
        b = N.pick(y2, regex, gate, base2, feat, tr, True)
        self.assertEqual(a, b)

    def test_min_recall_below_one_allows_a_stricter_threshold(self):
        y = [1, 1, 1, 0, 0, 1]
        base = [0.9, 0.8, 0.7, 0.6, 0.5, 0.1]   # one positive scores below every negative
        regex, gate, feat = [False] * 6, [True] * 6, [0.0] * 6
        tr = list(range(6))
        full = N.pick(y, regex, gate, base, feat, tr, False)
        part = N.pick(y, regex, gate, base, feat, tr, False, min_recall=0.75)
        self.assertLess(full[1], part[1])   # recall 1.0 has to go down to the outlier

    def test_apply_mm_clips(self):
        self.assertEqual(N.apply_mm(5, 0, 1), 1.0)
        self.assertEqual(N.apply_mm(-1, 0, 1), 0.0)
        self.assertEqual(N.apply_mm(3, 3, 3), 0.0)


if __name__ == "__main__":
    unittest.main()
