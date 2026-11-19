import unittest

from tests.helpers import ROOT  # noqa: F401
from a2179.normalize import normalize
from a2179.prices import parse_price, to_usd, parse_number


class TestNormalize(unittest.TestCase):
    def test_cyrillic_and_diacritics(self):
        self.assertEqual(normalize("Екран за MacBook Air А2179"), "ekran za macbook air a2179")
        self.assertEqual(normalize("Экран в сборе"), "ekran v sbore")
        self.assertEqual(normalize("Kućište, šarke, đak"), "kuciste sarke djak")

    def test_model_spacing(self):
        self.assertIn("a2179", normalize("A 2179"))
        self.assertIn("a2179", normalize("a-2179"))
        self.assertIn("emc3302", normalize("EMC 3302"))


class TestPrices(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_number("18.500"), 18500)
        self.assertEqual(parse_number("32,560"), 32560)
        self.assertEqual(parse_number("39.000,00"), 39000)
        self.assertEqual(parse_number("29000.00"), 29000)
        self.assertEqual(parse_number("1.234.567"), 1234567)
        self.assertAlmostEqual(parse_number("12,5"), 12.5)

    def test_parse_price(self):
        p = parse_price("18.500 din")
        self.assertEqual((p.amount, p.currency), (18500, "RSD"))
        p = parse_price("120 €")
        self.assertEqual((p.amount, p.currency), (120, "EUR"))
        p = parse_price("39.000,00 RSD 42000")
        self.assertEqual(p.amount, 39000)
        self.assertIsNone(parse_price("Dogovor"))
        self.assertIsNone(parse_price("1500"))  # no currency, no default

    def test_to_usd(self):
        r = {"usd_per_eur": 1.2, "rsd_per_eur": 120}
        self.assertAlmostEqual(to_usd(100, "EUR", r), 120)
        self.assertAlmostEqual(to_usd(1200, "RSD", r), 12)
        self.assertAlmostEqual(to_usd(1200, "SRDIN", r), 12)
        self.assertEqual(to_usd(5, "USD", r), 5)
        self.assertIsNone(to_usd(5, "GBP", r))


if __name__ == "__main__":
    unittest.main()
