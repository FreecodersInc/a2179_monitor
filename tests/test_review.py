"""Manual-review reports: matches.csv / .md for both cases, with source links, no network."""
import csv
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

import sys
from tests.helpers import FIX, ROOT
sys.path.insert(0, str(ROOT / "eval"))
import review_run as rr  # noqa: E402
from a2179 import jobs as J
from a2179.review import COLUMNS, load_listings, score_rows, write_case_reports
from a2179.target import load_target

NOW = dt.datetime(2026, 11, 19, 21, 0, tzinfo=dt.timezone.utc)
REM = (FIX / "remotive_data.json").read_text(encoding="utf-8")
ROK = (FIX / "remoteok_api.json").read_text(encoding="utf-8")


def get(url):
    if "remotive" in url:
        return 200, REM
    if "remoteok" in url:
        return 200, ROK
    raise AssertionError(url)


class TestWriteMatches(unittest.TestCase):
    def test_csv_and_md_have_required_columns_and_links(self):
        target = load_target("de_jobs")
        jobs, _ = J.collect_jobs(Path(tempfile.mkdtemp()), fetch="auto", get_fn=get, now=NOW)
        rows = score_rows([
            {"title": j.title, "company": j.company, "location": j.location, "url": j.url,
             "fetched_at": j.fetched_at, "source": j.source, "snippet": j.snippet}
            for j in jobs
        ], target)
        matches = [r for r in rows if r["decision"] == "match"]
        self.assertGreaterEqual(len(matches), 3)  # DE Toronto, Analytics EU, DE Vancouver
        titles = {r["title"] for r in matches}
        self.assertIn("Senior Data Engineer", titles)
        self.assertIn("Data Engineer Databricks Spark AWS", titles)
        self.assertNotIn("Marketing copywriter remote", titles)
        self.assertNotIn("Customer support agent remote Canada", titles)
        with tempfile.TemporaryDirectory() as d:
            summary = write_case_reports(Path(d) / "de_jobs", "de_jobs", rows, date="2026-11-19",
                                         notes=["Remotive cache"])
            self.assertEqual(summary["n_matches"], len(matches))
            with (Path(d) / "de_jobs" / "matches.csv").open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(list(reader.fieldnames)[: len(COLUMNS)], list(COLUMNS))
                recs = list(reader)
            self.assertEqual(len(recs), len(matches))
            for rec in recs:
                self.assertTrue(rec["url"].startswith("http"))
                self.assertEqual(rec["decision"], "match")
                self.assertTrue(rec["fetched_at"])
                self.assertTrue(rec["title"])
            md = (Path(d) / "de_jobs" / "matches.md").read_text(encoding="utf-8")
            self.assertIn("[link](", md)
            self.assertIn("Remotive", md)
            self.assertIn("https://remotive.com", md)
            self.assertIn("Remote OK", md)
            self.assertIn("https://remoteok.com", md)
            with (Path(d) / "de_jobs" / "all.csv").open(encoding="utf-8") as f:
                all_n = sum(1 for _ in f) - 1
            self.assertEqual(all_n, len(rows))

    def test_screen_listings_fixture_scores_and_keeps_links(self):
        rows = load_listings(FIX / "screen_listings.jsonl")
        self.assertEqual(len(rows), 2)
        scored = score_rows(rows, load_target("a2179_screen"))
        matches = [r for r in scored if r["decision"] == "match"]
        self.assertEqual([r["title"] for r in matches], ["LCD panel MacBook Air A2179"])
        self.assertTrue(matches[0]["url"].startswith("https://ifix.rs/"))
        self.assertEqual(matches[0]["company"], "ifix.rs")


class TestReviewRunBothCases(unittest.TestCase):
    def test_writes_reports_for_both_cases_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "reports"
            copy = Path(d) / "ci_out" / "reports"
            cache = Path(d) / "cache"
            report = rr.run(
                ["a2179_screen", "de_jobs"], root, "2026-11-19", "auto", cache,
                listings=FIX / "screen_listings.jsonl", get_fn=get, copy_to=copy, now=NOW,
            )
            self.assertEqual(report["date"], "2026-11-19")
            for case in ("a2179_screen", "de_jobs"):
                for name in ("matches.csv", "matches.md", "all.csv", "all.md"):
                    self.assertTrue((root / "2026-11-19" / case / name).is_file(), name)
                    self.assertTrue((copy / "2026-11-19" / case / name).is_file(), name)
            de = json.loads((root / "2026-11-19" / "de_jobs" / "summary.json").read_text())
            self.assertGreater(de["n_matches"], 0)
            self.assertGreater(de["n"], de["n_matches"])
            md = (root / "2026-11-19" / "de_jobs" / "matches.md").read_text()
            self.assertIn("https://remotive.com", md)
            with (root / "2026-11-19" / "a2179_screen" / "matches.csv").open(encoding="utf-8") as f:
                scr = list(csv.DictReader(f))
            self.assertTrue(scr and scr[0]["url"].startswith("https://"))
            # a second run must not hit the injected get_fn if we drop it: cache is warm
            def boom(url):
                raise AssertionError("network")
            rr.run(["de_jobs"], root, "2026-11-19", "auto", cache, get_fn=boom, now=NOW)

    def test_cli_fetch_never_without_cache_still_writes_the_files(self):
        with tempfile.TemporaryDirectory() as d:
            code = rr.main(["--cases", "de_jobs", "--out-root", str(Path(d) / "reports"),
                            "--date", "2026-11-19", "--fetch", "never",
                            "--jobs-cache", str(Path(d) / "empty")])
            self.assertEqual(code, 0)
            p = Path(d) / "reports" / "2026-11-19" / "de_jobs" / "matches.csv"
            self.assertTrue(p.is_file())
            # header only
            self.assertEqual(p.read_text().splitlines()[0].split(",")[:7], list(
                __import__("a2179.review", fromlist=["COLUMNS"]).COLUMNS))


if __name__ == "__main__":
    unittest.main()
