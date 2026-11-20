"""T-037: Arbeitnow, We Work Remotely and HN Who is hiring, parsers on saved real responses."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from a2179 import jobs as J  # noqa: E402

FIX = ROOT / "tests" / "fixtures"


def fx(n):
    return (FIX / n).read_text(encoding="utf-8")


class Parsers(unittest.TestCase):
    def test_arbeitnow(self):
        rows = J.parse_arbeitnow(fx("jobs_arbeitnow.json"))
        self.assertEqual(len(rows), 6)
        r = rows[0]
        self.assertEqual((r.source, r.company, r.title), ("arbeitnow", "isaraerospace", "Working Student, POQ (m/f/d)"))
        self.assertTrue(r.url.startswith("https://www.arbeitnow.com/jobs/"))
        self.assertNotIn("&lt;", r.text)
        self.assertNotIn("<p>", r.snippet)
        self.assertTrue(r.published_at.startswith("2026-11-"))

    def test_wwr(self):
        rows = J.parse_wwr(fx("jobs_wwr.rss"))
        self.assertEqual(len(rows), 5)
        self.assertEqual((rows[1].company, rows[1].title), ("Oportun", "Senior ML Engineer (R14284)"))
        self.assertEqual(rows[0].published_at, "2026-11-01T07:30:53Z")
        self.assertTrue(all(r.url.startswith("https://weworkremotely.com/remote-jobs/") for r in rows))

    def test_hn_top_level_only(self):
        self.assertEqual(J.hn_story_id(fx("jobs_hn_story.json")), "50338728")
        rows = J.parse_hn(fx("jobs_hn_comments.json"))
        self.assertEqual(len(rows), 6)   # the fixture has 6 top level comments and 1 reply
        r = rows[0]
        self.assertEqual((r.company, r.location), ("outGoing", "Lisbon, Portugal"))
        self.assertEqual(r.url, "https://news.ycombinator.com/item?id=50518905")
        self.assertFalse(any(hasattr(x, "author") for x in rows))

    def test_bad_bodies(self):
        self.assertEqual(J.parse_arbeitnow("nope"), [])
        self.assertEqual(J.parse_wwr("<rss"), [])
        self.assertEqual(J.parse_hn("{}"), [])
        self.assertEqual(J.hn_story_id("[]"), "")

    def test_hosts_allowed(self):
        for u in (J.ARBEITNOW_URL, J.WWR_URL, J.HN_STORY_URL):
            J.assert_allowed(u)


class Collect(unittest.TestCase):
    def test_collect_three_sources_offline(self):
        pages = {J.ARBEITNOW_URL: fx("jobs_arbeitnow.json"), J.WWR_URL: fx("jobs_wwr.rss"),
                 J.HN_STORY_URL: fx("jobs_hn_story.json"),
                 J.HN_COMMENTS_URL.format(id="50338728"): fx("jobs_hn_comments.json")}
        calls = []

        def get(u):
            calls.append(u)
            return (200, pages[u]) if u in pages else (404, "")
        with tempfile.TemporaryDirectory() as d:
            jobs, notes = J.collect_jobs(Path(d), sources=("arbeitnow", "wwr", "hn"), fetch="auto", get_fn=get)
            self.assertEqual({n["source"]: n["n"] for n in notes}, {"arbeitnow": 6, "wwr": 5, "hn": 6})
            self.assertEqual(len(calls), 4)
            J.collect_jobs(Path(d), sources=("arbeitnow", "wwr", "hn"), fetch="auto", get_fn=get)
            self.assertEqual(len(calls), 4)   # cached for 24 h, no second call


if __name__ == "__main__":
    unittest.main()
