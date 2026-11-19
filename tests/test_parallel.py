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
from a2179.fetch import Fetcher
from a2179.obs import Metrics, setup_logging
from a2179.run import run_once

HAS_FORK = "fork" in multiprocessing.get_all_start_methods()
KP2 = "https://www.kupujemprodajem.com/pretraga?keywords=a2179+ekran"


def par_cfg(tmp, workers=3, mc=10):
    cfg = make_cfg(tmp)
    cfg["parallel"] = {"workers": workers, "max_concurrency": mc, "delay_s": 0, "start_method": "fork"}
    return cfg


class TestSplit(unittest.TestCase):
    def test_split_adds_up(self):
        self.assertEqual(parallel.split_limit(10, 3), [4, 3, 3])
        self.assertEqual(parallel.split_limit(10, 1), [10])
        self.assertEqual(parallel.split_limit(10, 4), [3, 3, 2, 2])
        self.assertEqual(parallel.split_limit(2, 5), [1, 1])  # never more processes than slots
        for total in range(1, 25):
            for w in range(1, 8):
                self.assertEqual(sum(parallel.split_limit(total, w)), total)

    def test_settings_cli_over_config(self):
        cfg = {"parallel": {"workers": 5, "max_concurrency": 7}}
        self.assertEqual(parallel.settings(cfg)[:2], (5, 7))
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

    def test_tasks_keep_source_order(self):
        cfg = make_cfg(Path(tempfile.mkdtemp()))
        t = parallel.build_tasks(cfg)
        self.assertEqual([k for _, k, _ in t], ["shop_product", "kp"])
        self.assertEqual(t[1][2], KPQ)


class TestWorkerInProcess(unittest.TestCase):
    def test_semaphore_caps_inflight_and_overlaps(self):
        urls = {f"https://a.rs/p{i}": (200, "<h1>x</h1>") for i in range(12)}
        tasks = [(i, "shop_search", u) for i, u in enumerate(urls)]
        with FA.fake_aiohttp(urls, delay_s=0.1) as st:
            t0 = time.monotonic()
            out = parallel.worker(0, tasks, 4, {"retries": 0}, 0, None)
            dt = time.monotonic() - t0
        self.assertEqual(st["peak"].value, 4)
        self.assertEqual(st["sessions"].value, 1)  # one ClientSession per process
        self.assertLess(dt, 0.9)  # 12 x 0.1 s sequential would be 1.2 s; 3 waves of 4 = 0.3 s
        self.assertEqual(len(out["results"]), 12)
        self.assertEqual(out["counters"]["fetch.requests"], 12)

    def test_retry_and_403(self):
        seq = iter([(503, "busy"), (200, "ok")])
        routes = {"https://a.rs/x": lambda: next(seq), "https://b.rs/y": (403, "no")}
        with FA.fake_aiohttp(routes):
            m = Metrics()

            async def go():
                import aiohttp
                from a2179.afetch import AsyncFetcher
                async with aiohttp.ClientSession() as s:
                    f = AsyncFetcher({"retries": 2, "retry_403": 1, "backoff_s": 0}, m, parallel._BufLog(), s,
                                     asyncio.Semaphore(2), sleep=lambda x: asyncio.sleep(0))
                    return await f.get("https://a.rs/x"), await f.get("https://b.rs/y")
            a, b = asyncio.run(go())
        self.assertEqual(a, (200, "ok"))
        self.assertEqual(b[0], 403)
        self.assertEqual(m.counters["fetch.errors"], 3)


@unittest.skipUnless(HAS_FORK, "needs the fork start method to share the fake aiohttp")
class TestParallelRun(unittest.TestCase):
    def setUp(self):
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
        self.assertEqual(m["counters"]["fetch.requests"], 2)
        self.assertTrue(any(n.startswith("fetch=parallel processes=2") for n in m["notes"]), m["notes"])
        log = (self.tmp / "logs" / "monitor.jsonl").read_text()
        self.assertIn('"worker": 1', log)  # worker log lines land in the main log file

    def test_global_limit_over_processes(self):
        cfg = par_cfg(self.tmp, workers=3, mc=5)
        many = [f"https://ifix.rs/?s=q{i}&post_type=product" for i in range(30)]
        cfg["sources"] = {"shop_products": [], "shop_searches": many, "kp_keywords": [],
                          "kp_search_url": "https://www.kupujemprodajem.com/pretraga?keywords={q}"}
        routes = {u: (200, fixture("ifix_search.html.gz")) for u in many}
        with FA.fake_aiohttp(routes, delay_s=0.15) as st:
            t0 = time.monotonic()
            r = run_once(cfg, self.log, dry_run=True)
            dt = time.monotonic() - t0
        self.assertEqual(st["calls"].value, 30)
        self.assertLessEqual(st["peak"].value, 5)
        self.assertGreaterEqual(st["peak"].value, 4)
        self.assertEqual(st["sessions"].value, 3)
        self.assertLess(dt, 30 * 0.15)
        self.assertTrue(r["listings"])

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

    def test_local_server(self):
        from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
        state = {"now": 0, "peak": 0}
        lock = threading.Lock()
        page = fixture("ifix_search.html.gz").encode()

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                with lock:
                    state["now"] += 1
                    state["peak"] = max(state["peak"], state["now"])
                time.sleep(0.2)
                with lock:
                    state["now"] -= 1
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            tmp = Path(tempfile.mkdtemp())
            cfg = make_cfg(tmp)
            base = f"http://127.0.0.1:{srv.server_port}"
            cfg["sources"] = {"shop_products": [], "shop_searches": [f"{base}/?s={i}" for i in range(12)],
                              "kp_keywords": [], "kp_search_url": base + "/k?q={q}"}
            cfg["parallel"] = {"workers": 2, "max_concurrency": 4, "start_method": "spawn"}
            t0 = time.monotonic()
            r = run_once(cfg, setup_logging(tmp / "logs"), dry_run=True)
            dt = time.monotonic() - t0
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(r["metrics"]["counters"]["fetch.requests"], 12)
        self.assertEqual(r["metrics"]["counters"].get("fetch.errors", 0), 0)
        self.assertLessEqual(state["peak"], 4)
        self.assertGreaterEqual(state["peak"], 3)
        self.assertTrue(r["listings"])
        self.assertLess(dt, 12 * 0.2)


if __name__ == "__main__":
    unittest.main()
