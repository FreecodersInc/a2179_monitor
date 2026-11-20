"""T-033 / EXP-017: Limundo and KupindO category pages, parser on saved snapshots and the daily collector."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "ci"))

from a2179.parsers import parse_used_listing, sanitize_used_listing_html, used_site  # noqa: E402
import used_sources as U  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
KU = "https://www.kupindo.com/Racunari-i-oprema/artikli/322"
LI = "https://www.limundo.com/Racunari-i-oprema/aukcije/g12"
KEYS = {"id", "url", "title", "price", "currency", "condition", "category", "source"}


def page(name):
    return (FIX / name).read_text(encoding="utf-8")


class Parser(unittest.TestCase):
    def test_kupindo_snapshot(self):
        rows = parse_used_listing(page("used_kupindo_322.html"), KU)
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({r["id"] for r in rows}), 24)
        r = next(x for x in rows if x["id"] == "kupindo:86011139")
        self.assertEqual((r["title"], r["price"], r["currency"], r["category"]),
                         ("Acer Nitro 5 Gejmer sa GTX 1050", 45000.0, "RSD", "Laptopovi"))
        self.assertTrue(all(set(x) == KEYS and x["url"].startswith("https://www.kupindo.com/") for x in rows))

    def test_limundo_snapshot(self):
        rows = parse_used_listing(page("used_limundo_g12.html"), LI)
        self.assertEqual(len(rows), 25)
        r = next(x for x in rows if x["id"] == "limundo:161601407")
        self.assertEqual((r["title"], r["price"], r["condition"], r["category"]),
                         ("HP EliteBook x360 1030 G7 - i5 10210U - 256GB - TOP", 13833.0, "Polovno",
                          "Racunari-i-oprema/Laptop-i-tablet/Laptop"))
        self.assertTrue(all(x["price"] and x["currency"] == "RSD" for x in rows))

    def test_fixtures_have_no_seller_blocks(self):
        for n in ("used_kupindo_322.html", "used_limundo_g12.html"):
            h = page(n)
            self.assertNotIn("/Clan/", h)
            self.assertNotIn("media-body", h)
            self.assertNotIn("<img", h)

    def test_sanitize_keeps_the_parse(self):
        h = page("used_limundo_g12.html")
        self.assertEqual(parse_used_listing(sanitize_used_listing_html(h, LI), LI), parse_used_listing(h, LI))

    def test_other_sites(self):
        self.assertIsNone(used_site("https://www.kupujemprodajem.com/x"))
        self.assertEqual(parse_used_listing(page("used_limundo_g12.html"), "https://example.com/"), [])


class FakeFetcher:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, url, retry_403=None):
        self.calls.append(url)
        return self.pages.get(url, (404, ""))


SCREEN = ('<div class="col-12 list-view-listing-item"><h3><a href="/kupovina/Racunari-i-oprema/Apple/'
          'Ekran-MacBook-Air-A2179-komplet/161000001">Ekran MacBook Air A2179 komplet</a></h3>'
          '<div class="item-price"><strong>9.000 RSD</strong></div></div>')


class Collector(unittest.TestCase):
    def test_robots_first_and_refusal_stops_host(self):
        f = FakeFetcher({
            "https://www.limundo.com/robots.txt": (200, "User-agent: *\nDisallow: /pretraga\n"),
            LI: (200, page("used_limundo_g12.html").replace("</body>", SCREEN + "</body>")),
            LI + "?&page=1": (403, "no"),
            "https://www.kupindo.com/robots.txt": (200, "User-agent: *\nDisallow: /pretraga.php\n"),
        })
        pages = [LI, "https://www.limundo.com/pretraga?q=a2179", LI + "?&page=1", LI + "?&page=2",
                 "https://www.kupindo.com/pretraga.php?Pretraga=a2179", "https://example.com/x"]
        rows, log = U.collect(pages, f)
        self.assertEqual([p["verdict"] for p in log], ["ok", "disallowed", "blocked", "skipped", "disallowed", "skipped"])
        self.assertNotIn("https://www.limundo.com/pretraga?q=a2179", f.calls)
        self.assertNotIn(LI + "?&page=2", f.calls)
        self.assertEqual(len(rows), 26)
        hit = [r for r in rows if r["rule_match"]]
        self.assertEqual([r["id"] for r in hit], ["limundo:161000001"])
        self.assertTrue(all(set(r) == KEYS | {"page", "rule_match", "rule_gate"} for r in rows))

    def test_unreadable_robots_means_blocked(self):
        f = FakeFetcher({"https://www.kupindo.com/robots.txt": (200, "<html>Just a moment</html>")})
        rows, log = U.collect([KU], f)
        self.assertEqual((rows, log[0]["verdict"]), ([], "blocked"))
        self.assertEqual(f.calls, ["https://www.kupindo.com/robots.txt"])


class Links(unittest.TestCase):
    def test_no_member_search_or_paging_links(self):
        got = U.category_links(["https://www.limundo.com/Clan/someone", "https://www.limundo.com/pretraga/x?a=1&amp;b=2",
                                LI + "?&page=1", LI + "?type=list", KU + "/rss", KU + "_strana_2",
                                "https://www.limundo.com/kupovina/Racunari-i-oprema/Apple/x/161000001",
                                "https://www.limundo.com/Racunari-i-oprema/Apple/aukcije/g12_s99", LI, LI])
        self.assertEqual(got, ["https://www.limundo.com/Racunari-i-oprema/Apple/aukcije/g12_s99", LI])


class DailyRequest(unittest.TestCase):
    def test_once_per_day(self):
        import os
        from unittest import mock
        import run_experiments as R
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"A2179_CI_FETCH_LISTINGS": "auto"}):
            self.assertIsNone(R.used_request(d, "2026-11-20"))   # no request file
            (Path(d) / "ci" / "requests").mkdir(parents=True)
            self.assertIsNone(R.used_request(d, "2026-11-20"))
            (Path(d) / "ci" / "requests" / "used_sources_daily.json").write_text(json.dumps({"id": "x", "pages": [LI]}))
            self.assertEqual(R.used_request(d, "2026-11-20")["date"], "2026-11-20")
            done = Path(d) / "results" / "2026-11-20_abc" / "reports" / "used_sources" / "2026-11-20"
            done.mkdir(parents=True)
            (done / "summary.json").write_text("{}")
            self.assertIsNone(R.used_request(d, "2026-11-20"))
            self.assertIsNotNone(R.used_request(d, "2026-11-21"))


if __name__ == "__main__":
    unittest.main()
