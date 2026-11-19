import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import fixture
from a2179.config import load_config
from a2179.fetch import Fetcher
from a2179.obs import Metrics, setup_logging
from a2179.run import run_once
from a2179.state import State


class FakeResp:
    def __init__(self, status, text, headers=None):
        self.status_code, self._t, self.headers = status, text.encode(), headers or {}
        self.url, self.encoding = "", "utf-8"

    def iter_content(self, n):
        for i in range(0, len(self._t), n):
            yield self._t[i:i + n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls, self.headers = routes, [], {}

    def get(self, url, **kw):
        self.calls.append(url)
        r = self.routes.get(url)
        if callable(r):
            r = r()
        if isinstance(r, Exception):
            raise r
        status, text = r if r else (404, "not found")
        resp = FakeResp(status, text)
        resp.url = url
        return resp


IFIX = "https://ifix.rs/proizvodi/ekran-za-apple-macbook-air-13-a1932-a2179-2019-2020/"
KPQ = "https://www.kupujemprodajem.com/pretraga?keywords=a2179+lcd"


def make_cfg(tmp, kp_html=None, kp_status=200):
    cfg = load_config("does-not-exist.json")
    cfg["data_dir"] = str(tmp)
    cfg["sources"] = {"shop_products": [IFIX], "shop_searches": [],
                      "kp_search_url": "https://www.kupujemprodajem.com/pretraga?keywords={q}",
                      "kp_keywords": ["a2179 lcd"]}
    cfg["http"] = {"min_delay_s": 0, "jitter_s": 0, "retries": 1, "retry_403": 0, "backoff_s": 0}
    cfg["desktop_notify"] = False
    cfg["embedder"]["enabled"] = "false"
    return cfg


class TestFetcher(unittest.TestCase):
    def test_retry_then_ok_and_403_stop(self):
        seq = iter([(503, "busy"), (200, "ok")])
        s = FakeSession({"https://a.rs/x": lambda: next(seq), "https://b.rs/y": (403, "no")})
        m = Metrics()
        f = Fetcher({"min_delay_s": 0, "jitter_s": 0, "retries": 2, "retry_403": 1, "backoff_s": 0}, m,
                    setup_logging(Path(tempfile.mkdtemp())), sleep=lambda s: None, session=s)
        self.assertEqual(f.get("https://a.rs/x"), (200, "ok"))
        self.assertEqual(f.get("https://b.rs/y")[0], 403)
        self.assertEqual(s.calls.count("https://b.rs/y"), 2)  # one polite retry, then give up
        self.assertEqual(m.counters["fetch.errors"], 3)

    def test_rate_limit_sleeps_per_host(self):
        slept = []
        s = FakeSession({"https://a.rs/1": (200, "x"), "https://a.rs/2": (200, "x"), "https://c.rs/1": (200, "x")})
        f = Fetcher({"min_delay_s": 3, "jitter_s": 0}, Metrics(), setup_logging(Path(tempfile.mkdtemp())),
                    sleep=slept.append, session=s)
        f.get("https://a.rs/1"); f.get("https://c.rs/1"); f.get("https://a.rs/2")  # noqa: E702
        self.assertEqual(len(slept), 1)
        self.assertGreater(slept[0], 2.5)

    def test_connection_error(self):
        s = FakeSession({"https://a.rs/x": ConnectionError("boom")})
        f = Fetcher({"min_delay_s": 0, "jitter_s": 0, "retries": 1, "backoff_s": 0}, Metrics(),
                    setup_logging(Path(tempfile.mkdtemp())), sleep=lambda s: None, session=s)
        self.assertEqual(f.get("https://a.rs/x"), ("error", ""))


class TestRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = setup_logging(self.tmp / "logs")

    def fetcher(self, routes):
        return Fetcher({"min_delay_s": 0, "jitter_s": 0, "retries": 0, "retry_403": 0}, Metrics(), self.log,
                       sleep=lambda s: None, session=FakeSession(routes))

    def test_new_then_dedupe_then_drop_then_target(self):
        cfg = make_cfg(self.tmp)
        kp = fixture("kp_search_synthetic.html")
        routes = {IFIX: (200, fixture("ifix_product.html.gz")), KPQ: (200, kp)}
        r1 = run_once(cfg, self.log, fetcher=self.fetcher(routes))
        keys = {l.key for l, _, _ in r1["matched"]}
        self.assertIn("kp:180001111", keys)
        self.assertIn("kp:180004444", keys)        # giveaway
        self.assertNotIn("kp:180003333", keys)     # case
        self.assertEqual(r1["kp"], "ok")
        self.assertTrue(any("Новое" in m for m in r1["messages"]))
        self.assertTrue(any("ниже цели" in m.lower() for m in r1["messages"]))  # giveaway at 0 is under $20
        # second run, same data: nothing new
        r2 = run_once(cfg, self.log, fetcher=self.fetcher(routes))
        self.assertEqual(r2["messages"], [])
        # price drop on one KP ad: 18.500 -> 15.000 din
        routes[KPQ] = (200, kp.replace("18.500 din", "15.000 din"))
        r3 = run_once(cfg, self.log, fetcher=self.fetcher(routes))
        self.assertEqual(len(r3["messages"]), 1)
        self.assertIn("Цена упала", r3["messages"][0])
        # drop under target ($20 ~ 2020 RSD)
        routes[KPQ] = (200, kp.replace("18.500 din", "1.500 din"))
        r4 = run_once(cfg, self.log, fetcher=self.fetcher(routes))
        self.assertIn("ниже цели", r4["messages"][0].lower())
        metrics = json.loads((self.tmp / "metrics.json").read_text())
        self.assertIn("fetch.requests", metrics["counters"])
        self.assertTrue((self.tmp / "alerts.log").exists())
        self.assertTrue((self.tmp / "logs" / "monitor.jsonl").exists())

    def test_dry_run_writes_nothing_and_seed_is_silent(self):
        cfg = make_cfg(self.tmp)
        routes = {IFIX: (200, fixture("ifix_product.html.gz")), KPQ: (200, fixture("kp_search_synthetic.html"))}
        r = run_once(cfg, self.log, dry_run=True, fetcher=self.fetcher(routes))
        self.assertTrue(r["messages"])
        self.assertFalse((self.tmp / "state.sqlite").exists())
        self.assertFalse((self.tmp / "metrics.json").exists())
        self.assertTrue((self.tmp / "metrics_dryrun.json").exists())
        r = run_once(cfg, self.log, seed=True, fetcher=self.fetcher(routes))
        self.assertEqual(r["messages"], [])
        r = run_once(cfg, self.log, fetcher=self.fetcher(routes))
        self.assertEqual(r["messages"], [])

    def test_kp_blocked_degrades_and_streak_alert(self):
        cfg = make_cfg(self.tmp)
        cfg["kp_block_alert_after_runs"] = 2
        cfg["sources"]["kp_keywords"] = ["a2179 lcd", "a2179 ekran"]
        routes = {IFIX: (200, fixture("ifix_product.html.gz")), KPQ: (403, fixture("kp_blocked_403.html.gz"))}
        f = self.fetcher(routes)
        r1 = run_once(cfg, self.log, fetcher=f)
        self.assertTrue(r1["kp"].startswith("blocked"))
        self.assertEqual(sum(1 for c in f.s.calls if "kupujemprodajem" in c), 1)  # stops after first block
        self.assertTrue(any("A2179" in m for m in r1["messages"]))  # shops still work
        r2 = run_once(cfg, self.log, fetcher=self.fetcher(routes))
        self.assertTrue(any("KupujemProdajem" in m for m in r2["messages"]))
        st = State(self.tmp / "state.sqlite")
        self.assertEqual(st.meta_get("kp_block_streak"), "2")
        st.close()


if __name__ == "__main__":
    unittest.main()
