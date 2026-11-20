"""T-005: target price with a currency in the profile, target_usd as the old name, probe phrases from the profile."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from a2179.config import load_config  # noqa: E402
from a2179.embedder import probe_phrases  # noqa: E402
from a2179.prices import target_price_usd  # noqa: E402
from a2179.target import load_target  # noqa: E402

RATES = {"usd_per_eur": 1.16, "rsd_per_eur": 117.2}


class TargetPrice(unittest.TestCase):
    def test_old_name(self):
        self.assertEqual(target_price_usd({"target_usd": 20, "rates": RATES}), 20.0)

    def test_currency(self):
        self.assertAlmostEqual(target_price_usd({"target_price": {"amount": 20, "currency": "EUR"}, "rates": RATES}), 23.2)
        self.assertAlmostEqual(target_price_usd({"target_price": {"amount": 2344, "currency": "rsd"}, "target_usd": 1,
                                                 "rates": RATES}), 23.2)
        with self.assertRaises(ValueError):
            target_price_usd({"target_price": {"amount": 1, "currency": "GBP"}, "rates": RATES})

    def test_a2179_unchanged(self):
        cfg = load_config("/nonexistent/config.json")
        self.assertNotIn("target_price", cfg)
        self.assertEqual(target_price_usd(cfg), 20.0)

    def test_profile_price_and_legacy_override(self):
        prof = json.loads((ROOT / "targets" / "a2179_screen.json").read_text(encoding="utf-8"))
        prof["defaults"]["target_price"] = {"amount": 2000, "currency": "RSD"}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.json"
            p.write_text(json.dumps(prof))
            c = Path(d) / "config.json"
            c.write_text(json.dumps({"target": str(p)}))
            with mock.patch.dict(os.environ, {}, clear=False):
                for k in ("A2179_TARGET", "A2179_TARGET_USD", "A2179_TARGET_PRICE"):
                    os.environ.pop(k, None)
                self.assertAlmostEqual(target_price_usd(load_config(str(c))), 2000 / 117.2 * 1.16)
                c.write_text(json.dumps({"target": str(p), "target_usd": 12}))
                self.assertEqual(target_price_usd(load_config(str(c))), 12.0)   # explicit old name wins


class Probe(unittest.TestCase):
    def test_a2179_keeps_its_pair(self):
        self.assertEqual(probe_phrases({}), ("LCD ekran za MacBook Air A2179 2020", "Futrola za MacBook Air A2179"))

    def test_other_target_uses_its_prototypes(self):
        tg = load_target("de_jobs")
        self.assertEqual(probe_phrases({"target": "de_jobs"}), (tg.positive[0], tg.negative[0]))


if __name__ == "__main__":
    unittest.main()
