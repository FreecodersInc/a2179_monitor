"""eval/freeze.py (T-023): frozen settings are applied unchanged, bounds clip, overlap is refused."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "eval"))
import freeze as F  # noqa: E402


class TestFreezeScore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def frozen(self, t=0.5, lam=0.5, lo=-1.0, hi=1.0):
        tg = F.load_target("a2179_screen")
        fz = {"version": 1, "target": "a2179_screen", "profile_sha": F.profile_hash(tg),
              "data": {"path": "eval/does_not_exist.jsonl"},
              "variants": {"regex": {"kind": "regex"},
                           "tfidf+feat": {"kind": "hybrid", "base": "tfidf", "tie": "sweep", "lam": lam, "t": t, "lo": lo, "hi": hi},
                           "dense:mpnet+feat": {"kind": "hybrid", "base": "dense:mpnet", "tie": "sweep", "lam": lam, "t": t,
                                                "lo": lo, "hi": hi}}}
        p = self.tmp / "fz.json"
        p.write_text(json.dumps(fz))
        return p

    def data(self, rows):
        p = self.tmp / "new.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return p

    def test_scores_without_refit_and_skips_missing_dense(self):
        rows = [{"id": 9001, "title": "Ekran za MacBook Air A2179 LCD", "label": 1, "source": "kp"},
                {"id": 9002, "title": "Futrola za iPhone 13", "label": 0, "source": "kp"}]
        out = self.tmp / "o"
        rc = F.main(["score", "--frozen", str(self.frozen()), "--data", str(self.data(rows)), "--out", str(out)])
        self.assertEqual(rc, 0)
        res = json.loads((out / "frozen_score.json").read_text())
        self.assertEqual(res["variants"]["dense:mpnet+feat"]["kind"], "skipped")
        self.assertIn(9001, res["variants"]["regex"]["flagged_ids"])
        self.assertNotIn(9002, res["variants"]["tfidf+feat"]["flagged_ids"])
        self.assertEqual(res["variants"]["regex"]["real"]["fp"], 0)

    def test_threshold_one_flags_only_regex(self):
        rows = [{"id": 9003, "title": "MacBook Air 13 ekran", "label": 1, "source": "kp"}]
        fz = self.frozen(t=1.01, lam=0.5)
        rc = F.main(["score", "--frozen", str(fz), "--data", str(self.data(rows))])
        self.assertEqual(rc, 0)

    def test_rows_from_freeze_set_are_refused(self):
        old = [json.loads(l) for l in (ROOT / "eval" / "labeled.jsonl").read_text().splitlines() if l.strip()][:2]
        fz = json.loads(self.frozen().read_text())
        fz["data"]["path"] = "eval/labeled.jsonl"
        p = self.tmp / "fz2.json"
        p.write_text(json.dumps(fz))
        self.assertEqual(F.main(["score", "--frozen", str(p), "--data", str(self.data(old))]), 3)


class TestCommittedFreeze(unittest.TestCase):
    def test_file_matches_current_data_and_profile(self):
        fz = json.loads((ROOT / "eval" / "frozen" / "a2179_screen.json").read_text())
        import hashlib
        self.assertEqual(hashlib.sha256((ROOT / fz["data"]["path"]).read_bytes()).hexdigest(), fz["data"]["sha256"])
        self.assertEqual(F.profile_hash(F.load_target("a2179_screen")), fz["profile_sha"])
        self.assertEqual(set(fz["variants"]), set(F.VARIANTS))


if __name__ == "__main__":
    unittest.main()
