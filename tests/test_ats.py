"""a2179/ats.py: Greenhouse, Lever, Ashby boards with mocked HTTP only (no network in tests)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from a2179 import ats as A
from a2179 import jobs as J

NOW = J.parse_dt("2026-11-20T00:00:00Z")
GH = {"jobs": [
    {"id": 1, "title": "Senior Data Engineer", "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
     "location": {"name": "Toronto, ON"}, "updated_at": "2026-11-18T10:00:00-05:00",
     "content": "&lt;p&gt;Spark &amp;amp; Airflow&lt;/p&gt;"},
    {"id": 2, "title": "Account Executive", "absolute_url": "https://boards.greenhouse.io/acme/jobs/2",
     "location": {"name": "Remote"}}]}
LV = [{"id": "a", "text": "Analytics Engineer", "hostedUrl": "https://jobs.lever.co/beta/a",
       "categories": {"location": "Vancouver", "allLocations": ["Vancouver", "Remote Canada"]},
       "createdAt": 1763500000000, "descriptionPlain": "dbt Snowflake"},
      {"id": "b", "text": "Office Manager", "hostedUrl": "https://jobs.lever.co/beta/b", "categories": {}}]
AB = {"jobs": [{"id": "x", "title": "ML Platform Engineer", "jobUrl": "https://jobs.ashbyhq.com/gamma/x",
                "location": "Berlin", "secondaryLocations": [{"location": "London"}], "isRemote": True,
                "publishedAt": "2026-11-17T08:00:00Z", "descriptionPlain": "Kubernetes", "isListed": True},
               {"id": "y", "title": "Data Analyst", "jobUrl": "https://jobs.ashbyhq.com/gamma/y", "isListed": False}]}
COMPANIES = [{"ats": "greenhouse", "slug": "acme", "name": "Acme"}, {"ats": "lever", "slug": "beta", "name": "Beta"},
             {"ats": "ashby", "slug": "gamma", "name": "Gamma"}]


class FakeHTTP:
    def __init__(self, big_greenhouse=False):
        self.calls, self.big = [], big_greenhouse

    def __call__(self, url):
        self.calls.append(url)
        J.assert_allowed(url)
        if "greenhouse" in url:
            if self.big and "content=true" in url:
                return A.TOO_BIG, ""
            return 200, json.dumps(GH)
        if "lever" in url:
            return 200, json.dumps(LV)
        if "ashby" in url:
            return 200, json.dumps(AB)
        return 404, "{}"


class TestAts(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sleeps = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def collect(self, http, **kw):
        return A.collect_ats(self.tmp, companies=COMPANIES, get_fn=http, now=NOW, sleep_fn=self.sleeps.append, **kw)

    def test_parse_prefilter_fields_and_politeness(self):
        http = FakeHTTP()
        jobs, notes = self.collect(http)
        self.assertEqual(sorted(j.title for j in jobs), ["Analytics Engineer", "ML Platform Engineer", "Senior Data Engineer"])
        gh = next(j for j in jobs if j.source == "greenhouse")
        self.assertEqual((gh.company, gh.location, gh.url), ("Acme", "Toronto, ON", "https://boards.greenhouse.io/acme/jobs/1"))
        self.assertIn("Spark & Airflow", gh.snippet)
        lv = next(j for j in jobs if j.source == "lever")
        self.assertEqual(lv.location, "Vancouver, Remote Canada")
        ab = next(j for j in jobs if j.source == "ashby")
        self.assertEqual(ab.location, "Berlin, London, Remote")
        self.assertEqual(len(self.sleeps), 3)          # a pause after every real request
        self.assertEqual({n["source"] for n in notes}, {"greenhouse", "lever", "ashby"})
        self.assertEqual(next(n for n in notes if n["source"] == "greenhouse")["boards_detail"][0]["all"], 2)

    def test_cache_24h_and_compact(self):
        http = FakeHTTP()
        self.collect(http)
        n = len(http.calls)
        jobs, _ = self.collect(http)
        self.assertEqual(len(http.calls), n)           # second run inside 24 hours: cache only
        self.assertEqual(len(jobs), 3)
        cached = json.loads((self.tmp / "ats" / "greenhouse_acme.json").read_text())
        self.assertEqual(len(cached["jobs"]), 1)        # only prefiltered jobs are stored, not the raw body

    def test_big_greenhouse_board_refetched_without_content(self):
        http = FakeHTTP(big_greenhouse=True)
        jobs, _ = self.collect(http)
        self.assertTrue(any("content=true" in u for u in http.calls))
        self.assertTrue(any(u.endswith("/jobs") for u in http.calls))
        self.assertIn("Senior Data Engineer", [j.title for j in jobs])

    def test_budget_spent_leaves_boards_for_next_run(self):
        t = [0.0]

        def clock():
            t[0] += 10
            return t[0]
        http = FakeHTTP()
        jobs, notes = A.collect_ats(self.tmp, companies=COMPANIES, get_fn=http, now=NOW, sleep_fn=lambda s: None,
                                    clock=clock, budget_s=25)
        self.assertLess(len(http.calls), 3)
        self.assertTrue(any("next run" in n["note"] for n in notes))

    def test_dedup_across_sources(self):
        a = J.Job(source="remotive", url="https://x/1", title="Data Engineer", company="Acme", location="Remote")
        b = J.Job(source="greenhouse", url="https://x/1/", title="Data Engineer", company="Acme", location="Remote")
        c = J.Job(source="lever", url="https://y/2", title="Data  Engineer", company="ACME", location="remote")
        d = J.Job(source="ashby", url="https://z/3", title="Data Engineer", company="Acme", location="Toronto")
        self.assertEqual([j.source for j in A.dedup([a, b, c, d])], ["remotive", "ashby"])

    def test_probe_drops_404_and_tries_other_ats(self):
        http = FakeHTTP()
        st = A.probe([{"slug": "acme", "ats": "lever"}, {"slug": "nope", "ats": "greenhouse"}], self.tmp,
                     get_fn=lambda u: (404, "{}") if "nope" in u or "lever" in u else http(u),
                     sleep_fn=lambda s: None, state_path=self.tmp / "p.json")
        self.assertEqual(st["lever:acme"]["status"], 404)
        self.assertEqual(st["greenhouse:acme"]["status"], 200)
        self.assertNotIn("ashby:acme", st)              # stops at the first board that answers
        self.assertEqual({k for k in st if k.endswith(":nope")}, {"greenhouse:nope", "lever:nope", "ashby:nope", "workable:nope"})
        lst = A.build_list(st)
        self.assertEqual([(c["ats"], c["slug"]) for c in lst["companies"]], [("greenhouse", "acme")])

    def test_hosts_allowed_and_tests_see_no_company_list(self):
        for u in (A.board_url("greenhouse", "a"), A.board_url("lever", "a"), A.board_url("ashby", "a"), A.board_url("workable", "a")):
            J.assert_allowed(u)
        self.assertEqual(A.load_companies(), [])        # tests/__init__.py turns the real list off

    def test_committed_list_shape(self):
        p = A.COMPANIES
        if not p.is_file():
            self.skipTest("no company list yet")
        d = json.loads(p.read_text())
        self.assertTrue(100 <= len(d["companies"]) <= 250)
        self.assertEqual(len({(c["ats"], c["slug"]) for c in d["companies"]}), len(d["companies"]))
        self.assertTrue(all(c["ats"] in A.PARSE for c in d["companies"]))

    def test_parse_workable(self):
        body = json.dumps({"jobs": [
            {"title": "Data Engineer", "application_url": "https://apply.workable.com/delta/j/AB/",
             "location": {"city": "Belgrade", "country": "Serbia", "telecommuting": True},
             "created_at": "2026-11-01T00:00:00Z", "description": "<p>Spark</p>", "shortcode": "AB"},
            {"title": "Office Manager", "application_url": "https://apply.workable.com/delta/j/CD/",
             "location": "New York"}]})
        jobs = A.parse_workable(body, "Delta", "2026-11-21T00:00:00Z")
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0].location, "Belgrade, Serbia, Remote")
        self.assertIn("Spark", jobs[0].snippet)
        self.assertEqual(jobs[0].source, "workable")


if __name__ == "__main__":
    unittest.main()
