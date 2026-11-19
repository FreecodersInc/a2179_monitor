"""Regression on real data: Leonid's dry run from a Serbian connection (19.11.2026,
`python monitor.py --dry-run --list`, 59 listings, 17 matched, KP ok, 0/16 fetch errors).

tests/fixtures/dryrun_2026-11-19.jsonl   the 17 matched listings as printed (title, price, $, source, url)
tests/fixtures/kp_next_data_dryrun.html  the 12 KP ads of that run as a __NEXT_DATA__ page

The run only printed matches, so every row is a positive. What it exposed: KP sends price 0 for a
"Kontakt" ad and that fired a 🎯 below target alert ("Macbook air A2179 delovi", 0 RSD ~$0).
"""
import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import FIX, fixture
from tests.test_run import FakeSession, make_cfg
from a2179.alerts import format_event
from a2179.config import DEFAULTS
from a2179.fetch import Fetcher
from a2179.matcher import Matcher
from a2179.obs import Metrics, setup_logging
from a2179.parsers import Listing, parse_kp_search
from a2179.prices import to_usd
from a2179.run import run_once
from a2179.similarity import TfidfSim

KPQ = "https://www.kupujemprodajem.com/pretraga?keywords=a2179+lcd"
ZERO_AD = "kp:193930826"  # "Macbook air A2179 delovi", shown as 0 RSD


def rows():
    return [json.loads(l) for l in (FIX / "dryrun_2026-11-19.jsonl").read_text(encoding="utf-8").splitlines() if l]


class TestRealTitles(unittest.TestCase):
    def test_fixture_shape(self):
        r = rows()
        self.assertEqual(len(r), 17)
        self.assertEqual(sum(x["source"] == "kupujemprodajem" for x in r), 12)
        self.assertEqual(len({Listing(x["source"], x["url"], x["title"]).key for x in r}), 17)

    def test_regex_alone_matches_every_real_listing(self):
        m = Matcher()
        missed = [(x["title"], m.decide(x["title"]).reasons) for x in rows() if not m.decide(x["title"]).match]
        self.assertEqual(missed, [])

    def test_hybrid_matches_every_real_listing(self):
        m = Matcher(TfidfSim(), "tfidf", DEFAULTS["thresholds"])
        self.assertEqual([x["title"] for x in rows() if not m.decide(x["title"]).match], [])

    def test_near_misses_of_real_titles_stay_out(self):
        """Same real titles turned into things that are not a panel: must not match."""
        m = Matcher()
        for t in ("Zaštitna folija za ekran MacBook Air 13 A2179",
                  "LCD kabl Macbook A2179",
                  "Zamena ekrana MacBook Air 13\" A1932 / A2179, servis Beograd",
                  "Futrola za MacBook Air 13 2020 A2179",
                  "Tastatura MacBook Air A2179 2020",
                  "Apple MacBook Air 13 2020 A2337 M1 LCD Panel"):
            self.assertFalse(m.decide(t).match, t)

    def test_usd_as_printed(self):
        for x in rows():
            if x["price"] > 0:
                self.assertEqual(round(to_usd(x["price"], x["currency"], DEFAULTS["rates"])), x["usd_shown"], x)


class TestRealKP(unittest.TestCase):
    base = "https://www.kupujemprodajem.com/pretraga?keywords=a2179"

    def test_parse_real_ads(self):
        items = {i.key: i for i in parse_kp_search(fixture("kp_next_data_dryrun.html"), self.base)}
        kp = [x for x in rows() if x["source"] == "kupujemprodajem"]
        self.assertEqual(len(items), len(kp))
        for x in kp:
            l = items[Listing("kupujemprodajem", x["url"], "").key]
            self.assertEqual((l.title, l.url, l.location), (x["title"], x["url"], x["location"]))
            if x["price"] > 0:
                self.assertEqual((l.price, l.currency), (x["price"], x["currency"]))

    def test_zero_price_is_unknown_not_free(self):
        l = {i.key: i for i in parse_kp_search(fixture("kp_next_data_dryrun.html"), self.base)}[ZERO_AD]
        self.assertEqual((l.price, l.currency), (None, None))
        self.assertIn("цена не указана", format_event("new", l, None))

    def test_zero_price_with_giveaway_marker_is_free(self):
        page = ('<script id="__NEXT_DATA__" type="application/json">{"ads":['
                '{"name":"Ekran MacBook Air A2179","adUrl":"/x/oglas/1","price":0,"currency":"rsd","priceText":"Besplatno"},'
                '{"name":"Poklanjam ekran MacBook Air A2179","adUrl":"/x/oglas/2","price":0,"currency":"rsd"},'
                '{"name":"Ekran MacBook Air A2179","adUrl":"/x/oglas/3","price":0,"currency":"rsd","isFree":true},'
                '{"name":"Ekran MacBook Air A2179","adUrl":"/x/oglas/4","price":0,"currency":"rsd","priceText":"Kontakt"}'
                ']}</script>')
        got = {i.key: (i.price, i.currency) for i in parse_kp_search(page, self.base)}
        self.assertEqual(got, {"kp:1": (0.0, "RSD"), "kp:2": (0.0, "RSD"), "kp:3": (0.0, "RSD"), "kp:4": (None, None)})

    def test_html_card_free_word_in_description_is_not_the_price(self):
        page = ('<section class="AdItem"><a href="/x/oglas/77"><div>LCD Panel Macbook A2179 LCD</div></a>'
                '<p>Poklanjam punjač uz ekran, besplatno slanje</p><div>17.890 din</div></section>')
        (l,) = parse_kp_search(page, self.base)
        self.assertEqual((l.price, l.currency), (17890.0, "RSD"))
        page = ('<section class="AdItem"><a href="/x/oglas/78"><div>Ekran MacBook Air A2179</div></a>'
                '<p>Ekran radi, poklanjam uz kupovinu kabla 2.000 din</p></section>')
        (l,) = parse_kp_search(page, self.base)
        self.assertEqual(l.price, 2000.0)  # a number in the text beats a free word before it

    def test_run_once_on_real_kp_page_no_false_below_target(self):
        tmp = Path(tempfile.mkdtemp())
        log = setup_logging(tmp / "logs")
        cfg = make_cfg(tmp)
        cfg["sources"]["shop_products"] = []
        f = Fetcher({"min_delay_s": 0, "jitter_s": 0, "retries": 0, "retry_403": 0}, Metrics(), log,
                    sleep=lambda s: None, session=FakeSession({KPQ: (200, fixture("kp_next_data_dryrun.html"))}))
        r = run_once(cfg, log, dry_run=True, fetcher=f)
        self.assertEqual(r["kp"], "ok")
        self.assertEqual(len(r["matched"]), 12)
        self.assertEqual(len(r["messages"]), 12)
        self.assertFalse([m for m in r["messages"] if "ниже цели" in m], "0 RSD Kontakt ad must not be below target")
        zero = [m for m in r["messages"] if "/oglas/193930826" in m]
        self.assertEqual(len(zero), 1)
        self.assertIn("цена не указана", zero[0])


if __name__ == "__main__":
    unittest.main()
