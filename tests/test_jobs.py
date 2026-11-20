"""Remotive / Remote OK collectors: parsing, 24h delay, cache, daily cap, no network, blocked hosts."""
import datetime as dt
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.helpers import FIX, ROOT
from a2179 import jobs as J

NOW = dt.datetime(2026, 11, 19, 21, 0, tzinfo=dt.timezone.utc)
REM = (FIX / "remotive_data.json").read_text(encoding="utf-8")
ROK = (FIX / "remoteok_api.json").read_text(encoding="utf-8")


class TestParse(unittest.TestCase):
    def test_remotive_drops_jobs_newer_than_24h_and_keeps_the_rest(self):
        got = J.parse_remotive(REM, fetched_at=J.iso(NOW), now=NOW)
        titles = [j.title for j in got]
        self.assertIn("Senior Data Engineer", titles)
        self.assertIn("Analytics Engineer dbt Snowflake EU remote", titles)
        self.assertIn("Marketing copywriter remote", titles)
        self.assertNotIn("Data Engineer", titles)  # the too-new posting (id 103)
        de = next(j for j in got if j.title == "Senior Data Engineer")
        self.assertEqual(de.company, "Northwind Labs")
        self.assertEqual(de.location, "Toronto, Canada")
        self.assertTrue(de.url.startswith("https://remotive.com/"))
        self.assertIn("Spark", de.snippet)
        self.assertNotIn("<p>", de.snippet)
        self.assertEqual(de.source, "remotive")
        self.assertEqual(de.fetched_at, "2026-11-19T21:00:00Z")
        self.assertIn("Spark", de.text)  # full plain description for the CV distance score (T-024)
        self.assertNotIn("<p>", de.text)
        self.assertGreaterEqual(len(de.text), len(de.snippet.rstrip("…")))

    def test_remoteok_skips_legal_notice(self):
        got = J.parse_remoteok(ROK, fetched_at=J.iso(NOW), now=NOW)
        self.assertEqual([j.title for j in got], [
            "Data Engineer Databricks Spark AWS",
            "Customer support agent remote Canada",
        ])
        self.assertEqual(got[0].company, "Cedar Analytics")
        self.assertIn("remoteok.com", got[0].url)
        self.assertEqual(got[0].source, "remoteok")

    def test_bad_body_is_empty(self):
        self.assertEqual(J.parse_remotive("not json", now=NOW), [])
        self.assertEqual(J.parse_remoteok("{}", now=NOW), [])


class TestCacheAndCap(unittest.TestCase):
    def test_second_call_uses_cache(self):
        calls = []

        def get(url):
            calls.append(url)
            return 200, REM if "remotive" in url else ROK

        with TemporaryDirectory() as d:
            cache = Path(d)
            b1, m1 = J.load_or_fetch("remotive", J.REMOTIVE_URL, cache, fetch="auto", get_fn=get, now=NOW)
            b2, m2 = J.load_or_fetch("remotive", J.REMOTIVE_URL, cache, fetch="auto", get_fn=get, now=NOW)
            later = NOW + dt.timedelta(hours=2)
            b3, m3 = J.load_or_fetch("remotive", J.REMOTIVE_URL, cache, fetch="auto", get_fn=get, now=later)
            self.assertEqual(len(calls), 1)
            self.assertFalse(m1.get("from_cache"))
            self.assertTrue(m2.get("from_cache") and m3.get("from_cache"))
            self.assertEqual(b1, b2)
            # after 24h the cache is stale
            stale = NOW + dt.timedelta(hours=25)
            J.load_or_fetch("remotive", J.REMOTIVE_URL, cache, fetch="auto", get_fn=get, now=stale)
            self.assertEqual(len(calls), 2)

    def test_fetch_never_does_not_call(self):
        def get(url):
            raise AssertionError("network")
        with TemporaryDirectory() as d:
            body, meta = J.load_or_fetch("remotive", J.REMOTIVE_URL, Path(d), fetch="never", get_fn=get, now=NOW)
            self.assertEqual(body, "")
            self.assertIn("never", meta.get("note", ""))

    def test_remotive_daily_cap(self):
        calls = []

        def get(url):
            calls.append(url)
            return 200, REM

        with TemporaryDirectory() as d:
            cache = Path(d)
            # four requests already today
            times = [(NOW - dt.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ") for h in (1, 2, 3, 4)]
            (cache / "remotive.requests.json").write_text(json.dumps(times))
            (cache / "remotive.json").write_text(REM)
            (cache / "remotive.meta.json").write_text(json.dumps({
                "url": J.REMOTIVE_URL, "fetched_at": times[0], "status": 200,
            }))
            body, meta = J.load_or_fetch("remotive", J.REMOTIVE_URL, cache, fetch="always", get_fn=get,
                                         now=NOW, max_per_day=4)
            self.assertEqual(calls, [])
            self.assertTrue(meta.get("cap_hit"))
            self.assertEqual(body, REM)

    def test_collect_jobs_offline(self):
        def get(url):
            return 200, REM if "remotive" in url else ROK
        with TemporaryDirectory() as d:
            jobs, notes = J.collect_jobs(Path(d), fetch="auto", get_fn=get, now=NOW)
            self.assertEqual({n["source"] for n in notes}, {"remotive", "remoteok"})
            # remotive: 4 jobs in the fixture, one dropped as too new
            self.assertEqual(sum(1 for j in jobs if j.source == "remotive"), 3)
            self.assertEqual(sum(1 for j in jobs if j.source == "remoteok"), 2)

    def test_blocked_and_unknown_hosts_refused(self):
        for url in ("https://www.linkedin.com/jobs", "https://indeed.com/api",
                    "https://www.glassdoor.com/api", "https://example.com/jobs"):
            with self.assertRaises(ValueError):
                J.assert_allowed(url)
        J.assert_allowed(J.REMOTIVE_URL)
        J.assert_allowed(J.REMOTEOK_URL)


# touch ROOT so the import is used (repo layout assumption)
assert ROOT.name == "a2179_monitor" or (ROOT / "a2179").is_dir()

if __name__ == "__main__":
    unittest.main()
