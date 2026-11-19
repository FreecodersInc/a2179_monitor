import importlib.util
import multiprocessing
import tempfile
import threading
import unittest
from pathlib import Path

from tests.helpers import fixture
from tests import fake_aiohttp as FA
from tests.test_run import FakeSession, make_cfg
from a2179 import parallel
from a2179.fetch import Fetcher
from a2179.obs import Metrics, setup_logging
from a2179.paginate import Pager, find_next
from a2179.parsers import Listing
from a2179.run import collect, run_once
from a2179.tracked import TrackedStore

HAS_FORK = "fork" in multiprocessing.get_all_start_methods()
SHOP = "https://shop.rs/?s=a2179&post_type=product"
KP = "https://www.kupujemprodajem.com/pretraga?keywords=a2179"


def woo_page(ids, next_href=None, rel_link=False):
    cards = "".join(
        f'<li class="product"><a class="woocommerce-LoopProduct-link" href="https://shop.rs/product/lcd-a2179-{i}/">'
        f'<h2 class="woocommerce-loop-product__title">LCD ekran za MacBook Air A2179 nr {i}</h2></a>'
        f'<span class="price">{20000 + i} RSD</span></li>' for i in ids)
    head = f'<link rel="next" href="{next_href}">' if next_href and rel_link else ""
    nav = (f'<nav class="woocommerce-pagination"><ul class="page-numbers"><li>'
           f'<a class="next page-numbers" href="{next_href}">→</a></li></ul></nav>') if next_href and not rel_link else ""
    return f"<html><head>{head}</head><body><ul class='products'>{cards}</ul>{nav}</body></html>"


def kp_page(ids, next_page=None):
    cards = "".join(
        f'<section class="AdItem"><a href="/delovi/lcd-a2179-{i}/oglas/{190000 + i}">'
        f'<div>LCD ekran MacBook Air A2179 broj {i}</div></a><div>{15000 + i}.000 din</div></section>' for i in ids)
    nav = f'<a href="/pretraga?keywords=a2179&page={next_page}">Sledeća</a>' if next_page else ""
    return f"<html><body>{cards}{nav}</body></html>"


def drive(pager, pages):
    """Feeds a Pager from {url: (status, html)}; returns the urls fetched."""
    fetched = []
    while pager.url:
        fetched.append(pager.url)
        st, html = pages.get(pager.url, (404, "not found"))
        pager.feed(st, html)
    return fetched


class TestPagerStops(unittest.TestCase):
    def test_follows_next_links_until_no_next(self):
        p2, p3 = "https://shop.rs/page/2/?s=a2179&post_type=product", "https://shop.rs/page/3/?s=a2179&post_type=product"
        pages = {SHOP: (200, woo_page([1, 2], p2)), p2: (200, woo_page([3, 4], p3, rel_link=True)),
                 p3: (200, woo_page([5]))}
        pg = Pager("shop_search", SHOP, {})
        self.assertEqual(drive(pg, pages), [SHOP, p2, p3])
        self.assertEqual((pg.stop, pg.pages, len(pg.items)), ("no_next", 3, 5))

    def test_stop_on_empty_page(self):
        p2 = "https://shop.rs/page/2/?s=x"
        pg = Pager("shop_search", SHOP, {})
        drive(pg, {SHOP: (200, woo_page([1], p2)), p2: (200, woo_page([], "https://shop.rs/page/3/"))})
        self.assertEqual((pg.stop, pg.pages, len(pg.items)), ("empty", 2, 1))

    def test_stop_on_no_new_items(self):
        p2 = "https://shop.rs/page/2/?s=x"
        pg = Pager("shop_search", SHOP, {})
        drive(pg, {SHOP: (200, woo_page([1, 2], p2)), p2: (200, woo_page([2, 1], "https://shop.rs/page/3/"))})
        self.assertEqual((pg.stop, pg.pages, len(pg.items)), ("no_new", 2, 2))

    def test_stop_on_404_past_the_end_and_on_error(self):
        p2 = "https://shop.rs/page/2/?s=x"
        pg = Pager("shop_search", SHOP, {})
        drive(pg, {SHOP: (200, woo_page([1], p2))})
        self.assertEqual((pg.stop, len(pg.items)), ("end", 1))
        pg = Pager("shop_search", SHOP, {})
        drive(pg, {SHOP: (200, woo_page([1], p2)), p2: ("error", "")})
        self.assertEqual(pg.stop, "error")
        pg = Pager("shop_search", SHOP, {})
        drive(pg, {SHOP: (500, "")})
        self.assertEqual((pg.stop, pg.status), ("first_page", 500))

    def test_high_ceiling_loop_and_off_host(self):
        pages = {}
        for n in range(1, 50):
            u = SHOP if n == 1 else f"https://shop.rs/page/{n}/"
            pages[u] = (200, woo_page([n], f"https://shop.rs/page/{n + 1}/"))
        pg = Pager("shop_search", SHOP, {"pagination": {"max_pages": 7}})
        drive(pg, pages)
        self.assertEqual((pg.stop, pg.pages), ("ceiling", 7))
        pg = Pager("shop_search", SHOP, {})
        drive(pg, pages)
        self.assertEqual((pg.stop, pg.pages), ("end", 50))  # default ceiling 500 is far away
        p2 = "https://shop.rs/page/2/"
        pg = Pager("shop_search", SHOP, {})
        drive(pg, {SHOP: (200, woo_page([1], p2)), p2: (200, woo_page([2], SHOP))})
        self.assertEqual(pg.stop, "loop")
        pg = Pager("shop_search", SHOP, {})
        drive(pg, {SHOP: (200, woo_page([1], "https://evil.example/page/2/"))})
        self.assertEqual(pg.stop, "off_host")

    def test_disabled_reads_only_page_one(self):
        pg = Pager("shop_search", SHOP, {"pagination": {"enabled": False}})
        drive(pg, {SHOP: (200, woo_page([1], "https://shop.rs/page/2/"))})
        self.assertEqual((pg.stop, pg.pages), ("disabled", 1))

    def test_kp_next_anchor_guess_and_block(self):
        k2, k3 = KP + "&page=2", KP + "&page=3"
        pg = Pager("kp", KP, {})
        self.assertEqual(drive(pg, {KP: (200, kp_page([1, 2], 2)), k2: (200, kp_page([3])),
                                    k3: (200, kp_page([]))}), [KP, k2, k3])  # page 3 by the page= guess
        self.assertEqual((pg.stop, len(pg.items)), ("empty", 3))
        pg = Pager("kp", KP, {"pagination": {"kp_page_param": ""}})
        drive(pg, {KP: (200, kp_page([1]))})
        self.assertEqual(pg.stop, "no_next")  # no link and guessing off
        pg = Pager("kp", KP, {})
        drive(pg, {KP: (200, kp_page([1], 2)), k2: (403, fixture("kp_blocked_403.html.gz"))})
        self.assertEqual((pg.stop, pg.blocked, len(pg.items)), ("blocked", True, 1))
        pg = Pager("kp", KP, {})
        drive(pg, {KP: (200, kp_page([1], 2))})  # page 2: 404 is the end, not a block
        self.assertEqual((pg.stop, pg.blocked), ("end", False))
        self.assertEqual(find_next(kp_page([1], 2), KP, "kp", 1), k2)


class TestSequentialPagination(unittest.TestCase):
    def test_collect_reads_every_page(self):
        tmp = Path(tempfile.mkdtemp())
        log = setup_logging(tmp / "logs")
        cfg = make_cfg(tmp)
        p2 = "https://shop.rs/page/2/?s=a2179"
        cfg["sources"] = {"shop_products": [], "shop_searches": [SHOP], "kp_keywords": ["a2179"],
                          "kp_search_url": "https://www.kupujemprodajem.com/pretraga?keywords={q}"}
        routes = {SHOP: (200, woo_page([1, 2], p2)), p2: (200, woo_page([3])),
                  KP: (200, kp_page([1], 2)), KP + "&page=2": (200, kp_page([2], 3)), KP + "&page=3": (200, kp_page([]))}
        m = Metrics()
        f = Fetcher({"min_delay_s": 0, "jitter_s": 0, "retries": 0}, m, log, sleep=lambda s: None,
                    session=FakeSession(routes))
        listings, kp = collect(cfg, f, m, log)
        self.assertEqual(kp, "ok")
        self.assertEqual(len(listings), 5)
        self.assertEqual(m.counters["pagination.shop_search.pages"], 2)
        self.assertEqual(m.counters["pagination.kp.pages"], 3)
        self.assertEqual((m.counters["pagination.stop.no_next"], m.counters["pagination.stop.empty"]), (1, 1))


def _hammer(path, lock, key, n, own):
    s = TrackedStore(path, lock)
    for i in range(n):
        s.record(key, 500)  # read-modify-write of the same card: lost updates would show
        s.add([Listing("shop.rs", f"https://shop.rs/product/{own}-{i}/", "x")])


class TestTrackedStore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_add_record_gone_and_persist(self):
        s = TrackedStore(self.tmp / "t.shelve", drop_after_misses=2)
        l = Listing("shop.rs", "https://shop.rs/product/lcd-a2179/", "LCD A2179", price=100.0, currency="RSD")
        self.assertEqual(s.add([l, l]), [l.key])
        self.assertEqual(s.add([l]), [])
        s.record(l.key, 200, Listing("shop.rs", l.url, "LCD A2179", price=90.0, currency="RSD", in_stock=False))
        s.record(l.key, 404)
        self.assertEqual(TrackedStore(self.tmp / "t.shelve").active(), [(l.key, l.url)])
        s.record(l.key, 404)
        card = TrackedStore(self.tmp / "t.shelve").all()[l.key]  # a fresh handle reads it from disk
        self.assertEqual((card["price"], card["in_stock"], card["misses"], card["gone"], card["checks"]),
                         (90.0, False, 2, True, 3))
        self.assertEqual(s.active(), [])

    def test_read_only_writes_nothing(self):
        s = TrackedStore(self.tmp / "t.shelve", read_only=True)
        self.assertEqual(s.add([Listing("shop.rs", "https://shop.rs/product/a/", "x")]), [])
        self.assertEqual(s.all(), {})
        self.assertFalse(any(self.tmp.iterdir()))

    @unittest.skipUnless(HAS_FORK, "needs fork")
    def test_lock_prevents_lost_writes_with_three_processes(self):
        ctx = multiprocessing.get_context("fork")
        lock, path = ctx.Lock(), self.tmp / "t.shelve"
        key = Listing("shop.rs", "https://shop.rs/product/shared/", "x").key
        TrackedStore(path, lock).add([Listing("shop.rs", "https://shop.rs/product/shared/", "x")])
        ps = [ctx.Process(target=_hammer, args=(path, lock, key, 40, f"p{k}")) for k in range(3)]
        for p in ps:
            p.start()
        for p in ps:
            p.join(60)
        self.assertEqual([p.exitcode for p in ps], [0, 0, 0])
        cards = TrackedStore(path, lock).all()
        self.assertEqual(cards[key]["checks"], 120)  # 3 x 40, not one increment lost
        self.assertEqual(len(cards), 1 + 3 * 40)


class TrackedRunMixin:
    parallel_mode = False

    def setUp(self):
        parallel._LAST_DELAY.clear()
        self.tmp = Path(tempfile.mkdtemp())
        self.log = setup_logging(self.tmp / "logs")
        self.cfg = make_cfg(self.tmp)
        self.cfg["sources"] = {"shop_products": [], "shop_searches": [SHOP], "kp_keywords": [],
                               "kp_search_url": "https://www.kupujemprodajem.com/pretraga?keywords={q}"}
        self.cfg["tracked"] = {"drop_after_misses": 2}
        if self.parallel_mode:
            self.cfg["parallel"] = {"workers": 3, "max_concurrency": 10, "start_method": "fork"}

    def run_pass(self, routes, **kw):
        if self.parallel_mode:
            with FA.fake_aiohttp(routes) as st:
                r = run_once(self.cfg, self.log, **kw)
                return r, st["calls"].value
        s = FakeSession(routes)
        f = Fetcher({"min_delay_s": 0, "jitter_s": 0, "retries": 0}, Metrics(), self.log, sleep=lambda x: None,
                    session=s)
        return run_once(self.cfg, self.log, fetcher=f, **kw), len(s.calls)

    def store(self):
        return TrackedStore(self.tmp / "tracked.shelve")

    def test_tracked_set_persists_and_is_monitored(self):
        page = (200, woo_page([1, 2]))
        r1, _ = self.run_pass({SHOP: page}, dry_run=True)
        self.assertEqual(self.store().all(), {})  # dry run: nothing written
        r1, _ = self.run_pass({SHOP: page})
        cards = self.store().all()
        self.assertEqual(sorted(c["url"] for c in cards.values()),
                         ["https://shop.rs/product/lcd-a2179-1/", "https://shop.rs/product/lcd-a2179-2/"])
        self.assertIn("tracked=2", r1["metrics"]["notes"])
        # next pass: the search no longer lists card 1, its product page is still checked
        prod = fixture("ifix_product.html.gz")
        routes = {SHOP: (200, woo_page([2])), "https://shop.rs/product/lcd-a2179-1/": (200, prod),
                  "https://shop.rs/product/lcd-a2179-2/": (200, prod)}
        r2, calls = self.run_pass(routes)
        self.assertEqual(calls, 3)
        c = self.store().all()[Listing("", "https://shop.rs/product/lcd-a2179-1/", "").key]
        self.assertEqual((c["last_status"], c["price"], c["in_stock"], c["checks"]), (200, 29000.0, True, 1))
        self.assertEqual(r2["metrics"]["counters"]["parse.tracked.ok"], 2)
        self.assertIn("lcd-a2179-1", " ".join(l.url for l in r2["listings"]))
        # two passes of 404: gone, not fetched any more, but kept in the file
        gone = {SHOP: (200, woo_page([2])), "https://shop.rs/product/lcd-a2179-2/": (200, prod)}
        self.run_pass(gone)
        self.run_pass(gone)
        _, calls = self.run_pass(gone)
        self.assertEqual(calls, 2)  # search + card 2
        self.assertTrue(self.store().all()[Listing("", "https://shop.rs/product/lcd-a2179-1/", "").key]["gone"])


class TestTrackedSequential(TrackedRunMixin, unittest.TestCase):
    parallel_mode = False


@unittest.skipUnless(HAS_FORK, "needs the fork start method to share the fake aiohttp")
class TestTrackedParallel(TrackedRunMixin, unittest.TestCase):
    parallel_mode = True

    def test_parallel_pagination_and_stop_counters(self):
        p2 = "https://shop.rs/page/2/?s=a2179"
        self.cfg["sources"]["kp_keywords"] = ["a2179"]
        routes = {SHOP: (200, woo_page([1, 2], p2)), p2: (200, woo_page([1, 2], "https://shop.rs/page/3/")),
                  KP: (200, kp_page([1], 2)), KP + "&page=2": (200, kp_page([2]))}
        r, calls = self.run_pass(routes, dry_run=True)
        c = r["metrics"]["counters"]
        self.assertEqual(calls, 5)  # shop 2 pages (page 2 has no new items), KP 2 pages + guessed page 3 (404)
        self.assertEqual((c["pagination.shop_search.pages"], c["pagination.kp.pages"]), (2, 3))
        self.assertEqual((c["pagination.stop.no_new"], c["pagination.stop.end"]), (1, 1))
        self.assertEqual(r["kp"], "ok")
        self.assertEqual(c["parse.kp.items"], 2)


@unittest.skipUnless(importlib.util.find_spec("aiohttp"), "real aiohttp not installed")
class TestRealAiohttpTrackedPaging(unittest.TestCase):
    """Real aiohttp, spawned workers, local server on 127.0.0.1: pages, tracked writes from workers."""

    def test_two_passes(self):
        from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
        prod = fixture("ifix_product.html.gz").encode()
        hits = []

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                base = f"http://127.0.0.1:{self.server.server_port}"
                if self.path.startswith("/?s="):
                    body = woo_page([1, 2], base + "/page/2/?s=a").replace("https://shop.rs", base).encode()
                elif self.path.startswith("/page/2/"):
                    body = woo_page([3]).replace("https://shop.rs", base).encode()
                elif self.path.startswith("/product/"):
                    body = prod
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        base = f"http://127.0.0.1:{srv.server_port}"
        parallel._LAST_DELAY.clear()
        tmp = Path(tempfile.mkdtemp())
        log = setup_logging(tmp / "logs")
        cfg = make_cfg(tmp)
        cfg["sources"] = {"shop_products": [], "shop_searches": [base + "/?s=a"], "kp_keywords": [],
                          "kp_search_url": base + "/k?q={q}"}
        cfg["parallel"] = {"workers": 3, "max_concurrency": 4, "start_method": "spawn"}
        r1 = run_once(cfg, log)
        self.assertEqual(r1["metrics"]["counters"]["pagination.shop_search.pages"], 2)
        self.assertEqual(len(TrackedStore(tmp / "tracked.shelve").active()), 3)
        r2 = run_once(cfg, log)
        cards = TrackedStore(tmp / "tracked.shelve").all()
        self.assertEqual(sorted(c["checks"] for c in cards.values()), [1, 1, 1])  # written by the workers
        self.assertTrue(all(c["last_status"] == 200 and c["price"] == 29000.0 for c in cards.values()))
        self.assertEqual(r2["metrics"]["counters"]["parse.tracked.ok"], 3)


if __name__ == "__main__":
    unittest.main()
