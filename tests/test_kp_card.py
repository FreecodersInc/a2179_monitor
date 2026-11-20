"""T-031: KP card parser on sanitized snapshots (2026-11-20), no seller data kept, contacts masked."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from a2179.parsers import mask_contacts, parse_kp_card  # noqa: E402

FIX = ROOT / "tests" / "fixtures"


class KpCard(unittest.TestCase):
    def test_all_snapshots_parse(self):
        files = sorted(FIX.glob("kp_card_*.html"))
        self.assertEqual(len(files), 8)
        for f in files:
            c = parse_kp_card(f.read_text(encoding="utf-8"), f"https://www.kupujemprodajem.com/x/oglas/{f.stem[8:]}")
            self.assertEqual(c["id"], f.stem[8:])
            self.assertTrue(c["title"])
            self.assertGreater(len(c["text"]), 50)

    def test_fields(self):
        f = FIX / "kp_card_191871490.html"
        c = parse_kp_card(f.read_text(encoding="utf-8"))
        self.assertEqual((c["price"], c["currency"]), (150.0, "€"))
        self.assertIn("A1932", c["text"])
        self.assertNotIn("<", c["text"])

    def test_no_seller_fields_in_fixtures(self):
        for f in FIX.glob("kp_card_*.html"):
            t = f.read_text(encoding="utf-8")
            for k in ("ownerName", "ownerPhone", "userId", '"user"'):
                self.assertNotIn(k, t, f.name)

    def test_mask(self):
        self.assertEqual(mask_contacts("064 123 4567 / +381 64/123-4567 a@b.rs A1932 (2018)"),
                         "[tel] / [tel] [email] A1932 (2018)")

    def test_not_an_ad(self):
        self.assertIsNone(parse_kp_card("<html>nothing</html>"))


if __name__ == "__main__":
    unittest.main()
