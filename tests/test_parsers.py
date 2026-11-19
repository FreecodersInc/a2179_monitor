import unittest

from tests.helpers import fixture
from a2179.parsers import parse_woo_product, parse_woo_search, parse_kp_search, kp_is_blocked


class TestShopParsers(unittest.TestCase):
    def test_ifix_product_uses_jsonld_price(self):
        l = parse_woo_product(fixture("ifix_product.html.gz"), "https://ifix.rs/proizvodi/x/", "ifix.rs")
        self.assertIn("A2179", l.title)
        self.assertEqual((l.price, l.currency), (29000.0, "RSD"))  # not the 39.000 variation range
        self.assertTrue(l.in_stock)

    def test_ihelp_product(self):
        l = parse_woo_product(fixture("ihelp_product_komplet.html.gz"), "https://ihelp.rs/p/", "ihelp.rs")
        self.assertEqual(l.title, "LCD Panel Macbook A2179 komplet")
        self.assertEqual(l.price, 32560.0)

    def test_laptop_punjaci_product(self):
        l = parse_woo_product(fixture("lp_product_lcd.html.gz"), "https://laptop-punjaci.com/p/", "laptop-punjaci.com")
        self.assertEqual(l.title, "LCD Panel Macbook A2179 LCD")
        self.assertEqual(l.price, 25999.0)

    def test_searches(self):
        for name, host, n in (("ifix_search.html.gz", "ifix.rs", 8), ("ihelp_search.html.gz", "ihelp.rs", 3),
                              ("lp_search.html.gz", "laptop-punjaci.com", 3)):
            items = parse_woo_search(fixture(name), f"https://{host}/?s=a2179", host)
            self.assertGreaterEqual(len(items), n, name)
            self.assertTrue(any("A2179" in i.title and i.price for i in items), name)
            self.assertTrue(all(i.url.startswith("https://") for i in items))


class TestKP(unittest.TestCase):
    base = "https://www.kupujemprodajem.com/pretraga?keywords=a2179"

    def test_blocked_page(self):
        self.assertTrue(kp_is_blocked(403, fixture("kp_blocked_403.html.gz")))
        self.assertTrue(kp_is_blocked(200, fixture("kp_blocked_403.html.gz")))
        self.assertEqual(parse_kp_search(fixture("kp_blocked_403.html.gz"), self.base), [])

    def test_anchor_cards(self):
        items = parse_kp_search(fixture("kp_search_synthetic.html"), self.base)
        self.assertEqual(len(items), 4)
        by = {i.key: i for i in items}
        a = by["kp:180001111"]
        self.assertEqual((a.price, a.currency), (18500.0, "RSD"))
        self.assertIn("Predmet nov", a.snippet)
        self.assertEqual((by["kp:180002222"].price, by["kp:180002222"].currency), (120.0, "EUR"))
        self.assertEqual(by["kp:180004444"].price, 0.0)
        self.assertTrue(a.url.startswith("https://www.kupujemprodajem.com/"))

    def test_next_data(self):
        items = parse_kp_search(fixture("kp_next_data_synthetic.html"), self.base)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].location in ("Niš", "Bujanovac"), True)
        self.assertTrue(all(i.currency == "RSD" for i in items))


if __name__ == "__main__":
    unittest.main()


class TestKeys(unittest.TestCase):
    def test_same_product_two_paths(self):
        from a2179.parsers import Listing
        a = Listing("ihelp.rs", "https://ihelp.rs/prodavnica/a/b/lcd/lcd-panel-macbook-a2179-lcd/", "x")
        b = Listing("ihelp.rs", "https://ihelp.rs/prodavnica/c/d/e/lcd-panel-macbook-a2179-lcd/", "x")
        self.assertEqual(a.key, b.key)
        k = Listing("kupujemprodajem", "https://www.kupujemprodajem.com/x/y/oglas/123", "x").key
        self.assertEqual(k, "kp:123")
