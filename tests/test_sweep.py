"""eval/sweep.py: the transparent pieces (BM25, features, fusion bookkeeping) and an honest dense gap."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "eval"))
import sweep as S  # noqa: E402


class TestPieces(unittest.TestCase):
    def test_bm25_prefers_matching_prototypes(self):
        bm = S.BM25(["lcd ekran macbook air a2179"], ["tastatura macbook air a2179"])
        p, n = bm.score("LCD ekran za MacBook Air A2179")
        self.assertGreater(p, n)
        p, n = bm.score("Tastatura MacBook Air A2179")
        self.assertGreater(n, p)
        self.assertEqual(bm.score("nista zajednicko"), (0.0, 0.0))

    def test_features_are_visible_groups(self):
        self.assertEqual(S.features("LCD panel MacBook Air A2179"), (1.0, ["display"]))
        v, g = S.features("Šarke za ekran MacBook Air A2179")
        self.assertEqual((v, g), (0.0, ["display", "hinge"]))
        v, g = S.features("Tražim tastaturu za MacBook Air")
        self.assertEqual((v, set(g)), (-1.0, {"keyboard", "wanted"}))

    def test_minmax_and_ranks(self):
        self.assertEqual(S.minmax([1, 3, 2]), [0.0, 1.0, 0.5])
        self.assertEqual(S.ranks([0.1, 0.9, 0.5]), [3, 1, 2])

    def test_tune_is_recall_first(self):
        rows = [{"id": i, "title": "LCD ekran MacBook Air A2179", "snippet": "", "label": l, "source": "x"}
                for i, l in enumerate([1, 1, 0, 0])]
        sw = S.Sweep(rows)
        sw.regex = [False] * 4
        b = sw.tune([0.9, 0.4, 0.6, 0.1], hybrid=False)
        self.assertEqual((b[1], b[2]["R"], b[2]["fp"]), (0.4, 1.0, 1))


class TestRun(unittest.TestCase):
    def test_full_run_without_dense_says_so(self):
        out = Path(tempfile.mkdtemp()) / "s.json"
        old = sys.argv
        # frozen eval set: the fp pin below belongs to it, the live eval/labeled.jsonl keeps growing
        sys.argv = ["sweep.py", "--out", str(out), "--dense-scores", str(out.parent / "none.jsonl"),
                    "--data", str(ROOT / "tests" / "fixtures" / "labeled_base.jsonl")]
        try:
            S.main()
        finally:
            sys.argv = old
        r = json.loads(out.read_text())
        self.assertIsNone(r["dense"])
        names = {v["variant"] for v in r["variants"]}
        self.assertFalse([n for n in names if "dense" in n])
        self.assertIn("rrf(tfidf,bm25)", names)
        self.assertEqual(len(r["w_curves"]["tfidf+bm25 hybrid"]), 11)
        self.assertEqual(r["tfidf_2d"]["hybrid"]["fp"], 1)
        self.assertIn("not computed here", out.with_suffix(".tables.md").read_text())

    def test_dense_scores_file_is_used(self):
        rows = [json.loads(l) for l in (ROOT / "eval" / "labeled.jsonl").read_text().splitlines() if l]
        d = Path(tempfile.mkdtemp())
        ds = d / "dense.jsonl"
        ds.write_text("".join(json.dumps({"id": r["id"], "pos": 0.9 if r["label"] else 0.5, "neg": 0.5}) + "\n"
                              for r in rows))
        old = sys.argv
        sys.argv = ["sweep.py", "--out", str(d / "s.json"), "--dense-scores", str(ds)]
        try:
            S.main()
        finally:
            sys.argv = old
        r = json.loads((d / "s.json").read_text())
        self.assertIn("lin(dense,bm25)", " ".join(v["variant"] for v in r["variants"]))
        self.assertEqual(len(r["w_curves"]["dense+bm25 pure"]), 11)


if __name__ == "__main__":
    unittest.main()
