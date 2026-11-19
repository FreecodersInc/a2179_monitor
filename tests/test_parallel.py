import asyncio
import importlib.util
import json
import multiprocessing
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests.helpers import fixture
from tests import fake_aiohttp as FA
from tests.test_run import make_cfg, FakeSession, IFIX, KPQ
from a2179 import parallel
from a2179.afetch import AsyncFetcher, Limiter, Pacer
from a2179.fetch import Fetcher
from a2179.obs import Metrics, setup_logging
from a2179.run import run_once

HAS_FORK = "fork" in multiprocessing.get_all_start_methods()
KP2 = "https://www.kupujemprodajem.com/pretraga?keywords=a2179+ekran"


def par_cfg(tmp, workers=3, mc=10, **pace):
    cfg = make_cfg(tmp)
    cfg["parallel"] = dict({"workers": workers, "max_concurrency": mc, "start_method": "fork"}, **pace)
    return cfg


def shop_cfg(tmp, urls, **kw):
    cfg = par_cfg(tmp, **kw)
    cfg["sources"] = {"shop_products": [], "shop_searches": urls, "kp_keywords": [],
                      "kp_search_url": "https://www.kupujemprodajem.com/pretraga?keywords={q}"}
    return cfg


def pacer(hosts=("a.rs",), **settings):
    n = len(hosts)
    return Pacer(list(hosts), [0.0] * n, [0.0] * n, threading.Lock(), dict(settings))


class Sleeps:
    """Async sleep that records and only yields (tests stay fast)."""
    def __init__(self):
        self.calls = []

    async def __call__(self, s):
        self.calls.append(s)
        await asyncio.sleep(0)


class TestSettings(unittest.TestCase):
    def setUp(self):
        parallel._LAST_DELAY.clear()

    def test_settings_cli_over_config(self):
        cfg = {"parallel": {"workers": 5, "max_concurrency": 7, "delay_up": 3}}
        w, mc, pace, _ = parallel.settings(cfg)
        self.assertEqual((w, mc, pace["delay_up"], pace["delay_max_s"]), (5, 7, 3.0, 30.0))
        self.assertEqual(parallel.settings(cfg, 2, 4)[:2], (2, 4))
        self.assertEqual(parallel.settings({})[:2], (3, 10))

    def test_env_overrides(self):
        import os
        from a2179.config import load_config
        old = {k: os.environ.get(k) for k in ("A2179_WORKERS", "A2179_MAX_CONCURRENCY")}
        os.environ.update(A2179_WORKERS="2", A2179_MAX_CONCURRENCY="6")
        try:
            c = load_config("does-not-exist.json")
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self.assertEqual((c["parallel"]["workers"], c["parallel"]["max_concurrency"]), (2, 6))
        self.assertEqual(c["http"]["retries"], 3)

    def test_tasks_keep_source_order(self):
        cfg = make_cfg(Path(tempfile.mkdtemp()))
        t = parallel.build_tasks(cfg)
        self.assertEqual([k for _, k, _ in t], ["shop_product", "kp"])
        self.assertEqual(t[1][2], KPQ)


class TestPacer(unittest.TestCase):
    def test_429_grows_multiplicatively_and_success_decays(self):
        p = pacer(delay_on_429_s=0.5, delay_up=2.0, delay_max_s=3.0, delay_down_s=0.25, delay_min_s=0.0)
        self.assertEqual([p.on_rate_limited("a.rs") for _ in range(4)], [0.5, 1.0, 2.0, 3.0])  # capped
        p.on_success("a.rs")
        self.assertAlmostEqual(p.current("a.rs"), 2.75)
        for _ in range(20):
            p.on_success("a.rs")
        self.assertEqual(p.current("a.rs"), 0.0)  # floor delay_min_s

    def test_floor_and_unknown_host(self):
        p = pacer(delay_min_s=0.2, delay_down_s=1.0)
        p.on_success("a.rs")
        self.assertEqual(p.current("a.rs"), 0.2)
        self.assertIsNone(p.on_rate_limited("other.rs"))
        self.assertEqual(p.reserve("other.rs"), 0.0)

    def test_spacing_and_retry_after(self):
        p = pacer(delay_on_429_s=1.0)
        p.on_rate_limited("a.rs")
        self.assertEqual(p.reserve("a.rs"), 0.0)
        self.assertAlmostEqual(p.reserve("a.rs"), 1.0, places=2)  # next start one delay later
        self.assertAlmostEqual(p.reserve("a.rs"), 2.0, places=2)
        q = pacer(delay_on_429_s=0.1)
        q.on_rate_limited("a.rs", retry_after=5)
        self.assertGreater(q.reserve("a.rs"), 4.9)  # Retry-After holds every start to that host


class TestLimiter(unittest.TestCase):
    def test_caps_inflight_without_blocking_loop(self):
        sem = multiprocessing.BoundedSemaphore(2)
        lim = Limiter(sem)
        state = {"now": 0, "peak": 0, "ticks": 0}

        async def job():
            async with lim:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
                await asyncio.sleep(0.05)
                state["now"] -= 1

        async def ticker():  # proves the loop keeps running while jobs wait for a slot
            for _ in range(10):
                state["ticks"] += 1
                await asyncio.sleep(0.01)

        async def main():
            await asyncio.gather(ticker(), *(job() for _ in range(6)))
        asyncio.run(main())
        self.assertEqual(state["peak"], 2)
        self.assertEqual(state["ticks"], 10)
        self.assertTrue(sem.acquire(False) and sem.acquire(False))  # every slot was given back


class TestAsyncFetcherRetries(unittest.TestCase):
    def run_get(self, routes, url, http=None, p=None, **fake):
        http = dict({"retries": 3, "retry_403": 1, "backoff_s": 1, "backoff_max_s": 60}, **(http or {}))
        m, sl = Metrics(), Sleeps()
        with FA.fake_aiohttp(routes, **fake) as st:
            async def go():
                import aiohttp
                async with aiohttp.ClientSession() as s:
                    f = AsyncFetcher(http, m, parallel._BufLog(), s, Limiter(multiprocessing.BoundedSemaphore(2)),
                                     p, sleep=sl)
                    return await f.get(url)
            res = asyncio.run(go())
            calls = st["calls"].value
        return res, calls, m, sl

    def test_5xx_retried_three_times_with_exponential_jittered_backoff(self):
        res, calls, m, sl = self.run_get({"https://a.rs/x": (503, "busy")}, "https://a.rs/x")
        self.assertEqual(res[0], 503)
        self.assertEqual(calls, 4)  # 1 + 3 retries
        self.assertEqual(m.counters["fetch.retries"], 3)
        self.assertEqual(len(sl.calls), 3)
        for n, s in enumerate(sl.calls, 1):  # 1 s * 2^(n-1), jitter 0.5..1.5
            self.assertTrue(0.5 * 2 ** (n - 1) <= s <= 1.5 * 2 ** (n - 1), sl.calls)

    def test_network_error_retried(self):
        res, calls, m, _ = self.run_get({"https://a.rs/x": ConnectionError("boom")}, "https://a.rs/x")
        self.assertEqual((res, calls), (("error", ""), 4))

    def test_retry_then_ok_and_no_retry_on_404(self):
        seq = iter([(503, "busy"), (200, "ok")])
        res, calls, _, _ = self.run_get({"https://a.rs/x": lambda: next(seq)}, "https://a.rs/x")
        self.assertEqual((res, calls), ((200, "ok"), 2))
        res, calls, _, sl = self.run_get({}, "https://a.rs/missing")
        self.assertEqual((res[0], calls, sl.calls), (404, 1, []))

    def test_403_keeps_its_own_rule(self):
        res, calls, _, _ = self.run_get({"https://a.rs/y": (403, "no")}, "https://a.rs/y")
        self.assertEqual((res[0], calls), (403, 2))  # retry_403=1

    def test_429_grows_shared_delay_and_honours_retry_after(self):
        p = pacer(("a.rs",), delay_on_429_s=0.5, delay_up=2.0, delay_down_s=0.1)
        res, calls, m, sl = self.run_get({"https://a.rs/x": (200, "ok")}, "https://a.rs/x", p=p,
                                         http={"backoff_s": 0.01}, rate_limit_first=2, retry_after="7")
        self.assertEqual((res, calls), ((200, "ok"), 3))
        self.assertEqual(m.counters["fetch.rate_limited"], 2)
        self.assertAlmostEqual(p.current("a.rs"), 1.0 - 0.1)  # 0.5, x2 = 1.0, one success -0.1
        self.assertTrue(any(s >= 6.9 for s in sl.calls), sl.calls)  # waited for Retry-After

    def test_429_retries_are_capped_too(self):
        res, calls, m, _ = self.run_get({"https://a.rs/x": (200, "ok")}, "https://a.rs/x",
                                        http={"backoff_s": 0.01}, rate_limit_first=10)
        self.assertEqual((res[0], calls, m.counters["fetch.rate_limited"]), (429, 4, 4))


@unittest.skipUnless(HAS_FORK, "needs the fork start method to share the fake aiohttp")
class TestParallelRun(unittest.TestCase):
    def setUp(self):
        parallel._LAST_DELAY.clear()
        self.tmp = Path(tempfile.mkdtemp())
        self.log = setup_logging(self.tmp / "logs")

    def test_same_result_as_sequential(self):
        kp = fixture("kp_search_synthetic.html")
        routes = {IFIX: (200, fixture("ifix_product.html.gz")), KPQ: (200, kp)}
        seq_dir, par_dir = self.tmp / "seq", self.tmp / "par"
        f = Fetcher({"min_delay_s": 0, "jitter_s": 0, "retries": 0}, Metrics(), self.log, sleep=lambda s: None,
                    session=FakeSession(routes))
        r_seq = run_once(make_cfg(seq_dir), self.log, fetcher=f)
        with FA.fake_aiohttp(routes):
            r_par = run_once(par_cfg(par_dir), self.log)
        self.assertEqual(r_par["messages"], r_seq["messages"])
        self.assertEqual([l.key for l in r_par["listings"]], [l.key for l in r_seq["listings"]])
        self.assertEqual(r_par["kp"], "ok")
        m = json.loads((par_dir / "metrics.json").read_text())
        self.assertEqual(m["counters"]["fetch.requests"], 3)  # product, KP page 1, guessed KP page 2 (404 = end)
        self.assertIn("fetch=parallel processes=2 max_concurrency=10 (one shared limiter)", m["notes"])
        log = (self.tmp / "logs" / "monitor.jsonl").read_text()
        self.assertIn('"worker": 1', log)  # worker log lines land in the main log file

    def test_shared_limit_caps_total_over_processes(self):
        many = [f"https://ifix.rs/?s=q{i}&post_type=product" for i in range(30)]
        routes = {u: (200, fixture("ifix_search.html.gz")) for u in many}
        with FA.fake_aiohttp(routes, delay_s=0.15) as st:
            r = run_once(shop_cfg(self.tmp, many, workers=3, mc=5), self.log, dry_run=True)
        self.assertEqual(st["calls"].value, 30)
        self.assertEqual(st["peak"].value, 5)  # never more than the global limit, and it is used fully
        self.assertEqual(st["sessions"].value, 3)  # one ClientSession per process
        self.assertTrue(r["listings"])

    def test_idle_processes_hold_no_slots(self):
        # round robin over 3 processes: every slow source lands in process 2, the others finish at once.
        # A per-process split (2+2+1) would give process 2 one slot: 5 x 0.4 s = 2 s. Shared: about 0.4 s.
        urls = [f"https://ifix.rs/?s=q{i}&post_type=product" for i in range(15)]
        page = fixture("ifix_search.html.gz")
        routes = {u: (200, page, 0.4 if i % 3 == 2 else 0.0) for i, u in enumerate(urls)}
        with FA.fake_aiohttp(routes) as st:
            t0 = time.monotonic()
            parallel.fetch_all(shop_cfg(self.tmp, urls, workers=3, mc=5), Metrics(), self.log)
            dt = time.monotonic() - t0
        self.assertEqual(st["calls"].value, 15)
        self.assertLess(dt, 1.3, dt)

    def test_429_delay_is_shared_and_decays(self):
        urls = [f"https://ifix.rs/?s=q{i}&post_type=product" for i in range(6)]
        # successes answer after 0.2 s, the 429 at once: every 429 is counted before any success
        routes = {u: (200, fixture("ifix_search.html.gz"), 0.2) for u in urls}
        cfg = shop_cfg(self.tmp, urls, workers=3, mc=10, delay_on_429_s=0.1, delay_up=2.0, delay_max_s=0.4,
                       delay_down_s=0.05)
        cfg["http"].update(retries=3, backoff_s=0.01)
        with FA.fake_aiohttp(routes, rate_limit_first=3) as st:
            r = run_once(cfg, self.log, dry_run=True)
        c = r["metrics"]["counters"]
        self.assertEqual(st["calls"].value, 9)
        self.assertEqual((c["fetch.rate_limited"], c["fetch.retries"]), (3, 3))
        # 0.1 -> 0.2 -> 0.4 (cap) over three 429 from different processes, then 6 successes x -0.05
        self.assertAlmostEqual(parallel._LAST_DELAY["ifix.rs"], 0.1, places=6)
        self.assertIn("pace ifix.rs=0.10s", r["metrics"]["notes"])
        log = (self.tmp / "logs" / "monitor.jsonl").read_text()
        self.assertEqual(log.count('"rate_limited"'), 3)

    def test_learned_delay_carries_to_next_pass(self):
        urls = ["https://ifix.rs/?s=a&post_type=product"]
        parallel._LAST_DELAY["ifix.rs"] = 0.3
        with FA.fake_aiohttp({urls[0]: (200, fixture("ifix_search.html.gz"))}):
            r = run_once(shop_cfg(self.tmp, urls, delay_down_s=0.1), self.log, dry_run=True)
        self.assertIn("pace ifix.rs=0.20s", r["metrics"]["notes"])

    def test_kp_block_is_reported_once(self):
        cfg = par_cfg(self.tmp)
        cfg["kp_block_alert_after_runs"] = 1
        cfg["sources"]["kp_keywords"] = ["a2179 lcd", "a2179 ekran"]
        blocked = (403, fixture("kp_blocked_403.html.gz"))
        routes = {IFIX: (200, fixture("ifix_product.html.gz")), KPQ: blocked, KP2: blocked}
        with FA.fake_aiohttp(routes):
            r = run_once(cfg, self.log)
        self.assertTrue(r["kp"].startswith("blocked(403)"), r["kp"])
        self.assertEqual(r["metrics"]["counters"]["kp.blocked"], 1)
        self.assertTrue(any("A2179" in m for m in r["messages"]))  # shops still work
        self.assertTrue(any("KupujemProdajem" in m for m in r["messages"]))

    def test_workers_zero_is_sequential(self):
        cfg = par_cfg(self.tmp, workers=0)
        routes = {IFIX: (200, fixture("ifix_product.html.gz")), KPQ: (200, fixture("kp_search_synthetic.html"))}
        with FA.fake_aiohttp(routes) as st:
            import a2179.fetch as F
            orig = F.requests.Session
            F.requests.Session = lambda: FakeSession(routes)
            try:
                r = run_once(cfg, self.log, dry_run=True)
            finally:
                F.requests.Session = orig
        self.assertEqual(st["calls"].value, 0)
        self.assertEqual(r["kp"], "ok")


class TestNoAiohttp(unittest.TestCase):
    def test_falls_back_to_sequential(self):
        import io
        from contextlib import redirect_stderr
        import a2179.fetch as F
        tmp = Path(tempfile.mkdtemp())
        cfg = par_cfg(tmp)
        routes = {IFIX: (200, fixture("ifix_product.html.gz")), KPQ: (200, fixture("kp_search_synthetic.html"))}
        orig_find, orig_session = importlib.util.find_spec, F.requests.Session
        importlib.util.find_spec = lambda name, *a: None if name == "aiohttp" else orig_find(name, *a)
        F.requests.Session = lambda: FakeSession(routes)
        err = io.StringIO()
        try:
            with redirect_stderr(err):
                r = run_once(cfg, setup_logging(tmp / "logs"), dry_run=True)
        finally:
            importlib.util.find_spec, F.requests.Session = orig_find, orig_session
        self.assertIn("aiohttp", err.getvalue())
        self.assertEqual(r["kp"], "ok")


@unittest.skipUnless(importlib.util.find_spec("aiohttp"), "real aiohttp not installed")
class TestRealAiohttpLocal(unittest.TestCase):
    """End to end with the real aiohttp against a local server on 127.0.0.1 (no internet)."""

    def serve(self, latency_s=0.2, first_429=0, retry_after="1"):
        from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
        state = {"now": 0, "peak": 0, "n": 0, "starts": []}
        lock = threading.Lock()
        page = fixture("ifix_search.html.gz").encode()

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                with lock:
                    state["n"] += 1
                    state["now"] += 1
                    state["peak"] = max(state["peak"], state["now"])
                    state["starts"].append(time.monotonic())
                    limited = state["n"] <= first_429
                if not limited:
                    time.sleep(latency_s)
                with lock:
                    state["now"] -= 1
                body = b"slow down" if limited else page
                self.send_response(429 if limited else 200)
                if limited and retry_after is not None:
                    self.send_header("Retry-After", retry_after)
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
        return f"http://127.0.0.1:{srv.server_port}", state

    def cfg(self, base, n, **par):
        parallel._LAST_DELAY.clear()
        tmp = Path(tempfile.mkdtemp())
        cfg = make_cfg(tmp)
        cfg["sources"] = {"shop_products": [], "shop_searches": [f"{base}/?s={i}" for i in range(n)],
                          "kp_keywords": [], "kp_search_url": base + "/k?q={q}"}
        cfg["parallel"] = dict({"workers": 2, "max_concurrency": 4, "start_method": "spawn"}, **par)
        return cfg, setup_logging(tmp / "logs")

    def test_local_server(self):
        base, state = self.serve()
        cfg, log = self.cfg(base, 12)
        t0 = time.monotonic()
        r = run_once(cfg, log, dry_run=True)
        dt = time.monotonic() - t0
        self.assertEqual(r["metrics"]["counters"]["fetch.requests"], 12)
        self.assertEqual(r["metrics"]["counters"].get("fetch.errors", 0), 0)
        self.assertLessEqual(state["peak"], 4)
        self.assertGreaterEqual(state["peak"], 3)
        self.assertTrue(r["listings"])
        self.assertLess(dt, 12 * 0.2)

    def test_local_429_retry_after(self):
        base, state = self.serve(latency_s=0.05, first_429=2, retry_after="1")
        cfg, log = self.cfg(base, 4, delay_on_429_s=0.2, delay_down_s=0.01)
        cfg["http"].update(retries=3, backoff_s=0.01)
        r = run_once(cfg, log, dry_run=True)
        c = r["metrics"]["counters"]
        self.assertEqual((c["fetch.rate_limited"], c["fetch.retries"], state["n"]), (2, 2, 6))
        self.assertEqual(c.get("fetch.errors", 0), 2)  # the two 429 answers
        self.assertTrue(any(n.startswith("pace 127.0.0.1:") for n in r["metrics"]["notes"]), r["metrics"]["notes"])
        # Retry-After 1 s: nothing started for about a second after the 429s
        st = sorted(state["starts"])
        self.assertGreater(max(b - a for a, b in zip(st, st[1:])), 0.9)
        self.assertEqual(len(r["listings"]) > 0, True)


if __name__ == "__main__":
    unittest.main()
