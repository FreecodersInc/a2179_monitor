"""T-038 company base: parsers, slug probe, CSV, geo+CV filter, ensemble ranks. No network."""
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from a2179 import ats as A
from a2179 import company_base as C
from a2179.jobs import Job
from ci import run_experiments as R
from ci import write_results as wr
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import t038_company_base as T038  # noqa: E402

RI = """# Remote companies

| Company | Website | Region |
| --- | --- | --- |
| [10up](https://10up.com/) | [10up.com](https://10up.com/) | USA |
| [Acme Inc](https://acme.io/) | https://acme.io | Europe |
| Acme Inc | https://www.acme.io | EMEA |

## Also

- [Gamma Labs](https://gamma.dev/) — Remote worldwide
"""

YC = [
    {"name": "ActiveRemote", "website": "https://activeremote.com", "status": "Active",
     "all_locations": "Remote", "isHiring": True},
    {"name": "QuietCo", "website": "https://quiet.co", "status": "Active", "tags": ["devtools"]},
    {"name": "DeadCo", "website": "https://dead.co", "status": "Inactive"},
    {"name": "Acme Inc", "website": "https://acme.io", "status": "Active", "isHiring": True},
]

WB = {"jobs": [{"title": "Data Engineer", "application_url": "https://apply.workable.com/delta/j/AB/",
                "location": {"city": "Belgrade", "country": "Serbia", "telecommuting": True},
                "description": "Spark and Airflow", "shortcode": "AB"}]}


class TestParsers(unittest.TestCase):
    def test_remoteintech_table_and_bullets_dedup_domain(self):
        rows = C.parse_remoteintech(RI)
        names = [r["name"] for r in rows]
        self.assertIn("10up", names)
        self.assertIn("Gamma Labs", names)
        acme = [r for r in rows if "acme" in r["domain"]]
        self.assertTrue(acme)
        self.assertEqual(acme[0]["domain"], "acme.io")
        self.assertEqual(acme[0]["geo"], "Europe")
        gamma = next(r for r in rows if r["name"] == "Gamma Labs")
        self.assertIn("Remote worldwide", gamma["geo"])

    def test_yc_preferred_and_merge_to_1000(self):
        yc = C.parse_yc(YC)
        self.assertTrue(C.yc_preferred(next(r for r in yc if r["name"] == "ActiveRemote")))
        self.assertFalse(C.yc_preferred(next(r for r in yc if r["name"] == "DeadCo")))
        ri = C.parse_remoteintech(RI)
        extra = [C._row(f"Co{i}", f"https://co{i}.example.com", "Remote", "yc") for i in range(1200)]
        merged = C.merge_rows(ri, yc, extra)
        self.assertGreaterEqual(len(merged), 1000)
        acme = [r for r in merged if r["domain"] == "acme.io"]
        self.assertEqual(len(acme), 1)
        self.assertIn("remoteintech", acme[0]["source"])
        self.assertIn("yc", acme[0]["source"])


    def test_frontmatter_and_tarball(self):
        md = """---
title: "10up"
website: https://10up.com/
region: europe
remote_policy: remote-friendly
---
A company blurb.
"""
        rec = C.parse_remoteintech_frontmatter(md)
        self.assertEqual(rec["name"], "10up")
        self.assertEqual(rec["domain"], "10up.com")
        self.assertIn("europe", rec["geo"])
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            data = md.encode()
            info = tarfile.TarInfo(name="repo/src/companies/10up.md")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
            skip = b"not a company"
            info2 = tarfile.TarInfo(name="repo/README.md")
            info2.size = len(skip)
            tf.addfile(info2, io.BytesIO(skip))
        rows = C.companies_from_tarball(buf.getvalue())
        self.assertEqual([r["name"] for r in rows], ["10up"])

    def test_known_ats_boards_do_not_collapse_and_merge_by_name(self):
        known = [
            C._row("canonical", "", "", "ats_companies", ats="greenhouse", slug="canonical",
                   board_url="https://boards.greenhouse.io/canonical"),
            C._row("stripe", "", "", "ats_companies", ats="greenhouse", slug="stripe",
                   board_url="https://boards.greenhouse.io/stripe"),
        ]
        # board hosts must not become the domain
        self.assertEqual(known[0]["domain"], "")
        site = C._row("canonical", "https://canonical.com/", "Worldwide", "remoteintech")
        merged = C.merge_rows(known, [site])
        self.assertEqual(len(merged), 2)
        can = next(r for r in merged if r["slug"] == "canonical" or r["domain"] == "canonical.com")
        self.assertEqual(can["domain"], "canonical.com")
        self.assertEqual(can["ats"], "greenhouse")
        self.assertEqual(len([r for r in merged if r.get("ats") == "greenhouse"]), 2)

    def test_slug_variants(self):
        v = C.slug_variants("Acme Inc.", "acme.io")
        self.assertEqual(v[0], "acme")
        self.assertIn("acmeinc", v)
        self.assertTrue(all(s.isalnum() or "-" in s for s in v))

    def test_robots(self):
        self.assertFalse(C.robots_allows("User-agent: *\nDisallow: /\n", "/api/companies.json"))
        self.assertTrue(C.robots_allows("User-agent: *\nDisallow:\n", "/api/companies.json"))
        self.assertTrue(C.robots_allows("User-agent: *\nDisallow: /secret\n", "/api/companies.json"))
        self.assertTrue(C.robots_allows("", "/x"))


class TestProbeAndCsv(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_probe_workable_and_csv_fields(self):
        def http(url):
            from a2179 import jobs as J
            J.assert_allowed(url)
            if "workable" in url and "delta" in url:
                return 200, json.dumps(WB)
            if "greenhouse" in url and "acme" in url:
                return 200, json.dumps({"jobs": [{"title": "X", "absolute_url": "https://boards.greenhouse.io/acme/jobs/1"}]})
            return 404, "{}"

        rows = [
            C._row("Delta", "https://delta.com", "Serbia", "remoteintech"),
            C._row("Acme Inc", "https://acme.io", "Europe", "yc"),
            C._row("Nope Ltd", "https://nope.test", "", "yc"),
        ]
        # fix slugs we want to hit first
        rows[0]["slug"] = "delta"
        rows[1]["slug"] = "acme"
        rows[1]["ats"] = "greenhouse"
        state = A.probe(C.candidates_for(rows[0]) + C.candidates_for(rows[1]) + C.candidates_for(rows[2]),
                        self.tmp, get_fn=http, budget_s=30, sleep_fn=lambda s: None,
                        state_path=self.tmp / "p.json")
        checked = "2026-11-21"
        out = [C.apply_probe(r, state, checked) for r in rows]
        self.assertEqual(out[0]["ats"], "workable")
        self.assertEqual(out[0]["status"], C.STATUS_LIVE)
        self.assertTrue(out[0]["board_url"].startswith("https://apply.workable.com/"))
        self.assertEqual(out[1]["ats"], "greenhouse")
        self.assertEqual(out[2]["status"], C.STATUS_NONE)
        path = self.tmp / "companies.csv"
        C.write_csv(path, out)
        back = C.read_csv(path)
        self.assertEqual(list(back[0]), C.CSV_FIELDS)
        self.assertEqual(back[0]["name"], "Delta")
        self.assertEqual(back[0]["domain"], "delta.com")

    def test_liveness_sleeps_after_three_fails(self):
        row = C._row("Acme", "https://acme.io", "", "yc", ats="greenhouse", slug="acme",
                     status=C.STATUS_LIVE, last_ok="2026-11-20", fail_count="0")
        miss = {"greenhouse:acme": {"status": 404}}
        r1 = C.liveness_update([row], miss, "2026-11-21")[0]
        self.assertEqual(r1["fail_count"], "1")
        self.assertEqual(r1["status"], C.STATUS_LIVE)
        r2 = C.liveness_update([r1], miss, "2026-11-22")[0]
        r3 = C.liveness_update([r2], miss, "2026-11-23")[0]
        self.assertEqual(r3["status"], C.STATUS_SLEEP)
        self.assertEqual(r3["fail_count"], "3")
        ok = {"greenhouse:acme": {"status": 200, "jobs": 4}}
        r4 = C.liveness_update([r3], ok, "2026-11-24")[0]
        self.assertEqual(r4["status"], C.STATUS_LIVE)
        self.assertEqual(r4["fail_count"], "0")
        self.assertEqual(r4["n_jobs"], "4")


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_filters_and_ranks(self):
        companies_src = [
            C._row("Acme Inc", "https://acme.io", "Europe", "yc", slug="acme", ats="greenhouse"),
            C._row("Delta", "https://delta.com", "Serbia", "remoteintech", slug="delta"),
        ]
        jobs = [
            Job(source="greenhouse", url="https://boards.greenhouse.io/acme/jobs/1",
                title="Senior Data Engineer", company="Acme Inc", location="Remote, Europe",
                snippet="SQL Python Spark Airflow", text="SQL Python Spark Airflow dbt Kafka"),
            Job(source="workable", url="https://apply.workable.com/delta/j/AB/",
                title="Data Engineer", company="Delta", location="Belgrade, Serbia, Remote",
                snippet="Spark Airflow", text="Spark Airflow Python SQL"),
            Job(source="greenhouse", url="https://boards.greenhouse.io/acme/jobs/2",
                title="Senior Data Engineer", company="Acme Inc", location="San Francisco, CA",
                snippet="SQL Python Spark", text="SQL Python Spark"),
        ]

        def get_fn(url):
            from a2179 import jobs as J
            J.assert_allowed(url)
            if "robots.txt" in url:
                return 200, "User-agent: *\nDisallow:\n"
            if url.endswith("README.md"):
                return 200, RI
            if "companies.json" in url:
                return 200, json.dumps(YC)
            if "greenhouse" in url and "acme" in url:
                return 200, json.dumps({"jobs": []})
            if "workable" in url and "delta" in url:
                return 200, json.dumps(WB)
            return 404, "{}"

        def collect(live):
            return jobs, [{"source": "greenhouse", "boards_ok": 1}]

        def score(job):
            # high for the two DE roles; still computed so the 0.25 filter can run
            return {"cv_score": 0.7 if "San Francisco" not in job.location else 0.7,
                    "skills": 1, "stack": 0.5, "domain": 1, "seniority": 1}

        def dense(titles, out):
            return {"tinybert": {u: 0.9 if "Belgrade" in next(j.location for j in jobs if j.url == u) else 0.2
                                 for u in titles}}, {}, {}

        # skip real source merge of 162 boards: patch collect_sources
        def sources(get_fn, include_known=True):
            return companies_src, {"remoteintech": {"rows": 3}, "yc": {"rows": 4}}

        out = self.tmp / "out"
        out.mkdir()
        with mock.patch.object(C, "collect_sources", sources):
            summary = T038.run_build(out, self.tmp / "cache",
                                     {"time_budget_s": 20, "delay_s": 0, "cv_min": 0.25},
                                     get_fn=get_fn, sleep_fn=lambda s: None, collect=collect,
                                     score_fn=score, dense_fn=dense)
        self.assertTrue((out / "companies.csv").is_file())
        self.assertEqual(summary["vacancies_fetched"], 3)
        self.assertEqual(summary["geo_counts"]["cut"], 1)
        self.assertEqual(summary["geo_keep_check"], 2)
        self.assertEqual(summary["cv_match"], 2)
        ranked = Path(out / "vacancies_ranked.csv").read_text(encoding="utf-8").splitlines()[1:]
        self.assertEqual(len(ranked), 2)
        top = ranked[0]
        self.assertIn("apply.workable.com/delta", top)
        self.assertIn("delta.com", top)
        self.assertTrue((out / "weights.csv").is_file())

    def test_ensemble_equal_families_and_ties(self):
        rows = [{"url": "a", "title": "A"}, {"url": "b", "title": "B"}, {"url": "c", "title": "C"}]
        raw = {"cv_score": {"a": 1, "b": 1, "c": 0}, "tinybert": {"a": 0, "b": 3, "c": 2}}
        w = T038.ensemble_weights(["cv_score", "tinybert"], {})
        self.assertAlmostEqual(w["cv_score"], 0.5)
        self.assertAlmostEqual(w["tinybert"], 0.5)
        ranked = T038.apply_ensemble(rows, raw, w)
        self.assertEqual([r["url"] for r in ranked][0], "b")

    def test_robots_block_skips_yc_catalog(self):
        def get_fn(url):
            from a2179 import jobs as J
            J.assert_allowed(url)
            if "robots.txt" in url:
                return 200, "User-agent: *\nDisallow: /\n"
            if "yc-oss.github.io" in url and "robots.txt" not in url:
                raise AssertionError("must not fetch YC when robots disallows it: " + url)
            return 404, b""
        rows, notes = C.collect_sources(get_fn, include_known=False)
        self.assertEqual(notes["robots"]["yc"], "disallow")
        self.assertEqual(rows, [])

    def test_collect_sources_tarball_and_yc(self):
        md = "---\ntitle: Gamma Labs\nwebsite: https://gamma.dev/\nregion: worldwide\n---\n"
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            data = md.encode()
            info = tarfile.TarInfo(name="repo/src/companies/gamma-labs.md")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        blob = buf.getvalue()
        def get_fn(url):
            from a2179 import jobs as J
            J.assert_allowed(url)
            if "robots.txt" in url:
                return 200, "User-agent: *\nDisallow:\n"
            if "codeload.github.com" in url:
                return 200, blob
            if "companies/all.json" in url:
                return 200, json.dumps(YC)
            raise AssertionError(url)
        rows, notes = C.collect_sources(get_fn, include_known=False)
        self.assertGreaterEqual(notes["remoteintech"]["rows"], 1)
        self.assertGreaterEqual(notes["yc"]["rows"], 3)
        self.assertGreaterEqual(notes["merged"], 4)
        self.assertTrue(any(r["name"] == "Gamma Labs" for r in rows))


class TestCiRequest(unittest.TestCase):
    def test_company_base_once_per_id(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"A2179_CI_COMPANY_BASE": "auto"}):
            self.assertIsNone(R.company_base_request(d))
            p = Path(d) / "ci" / "requests"
            p.mkdir(parents=True)
            (p / "company_base.json").write_text(json.dumps({"id": "t038-v1", "time_budget_s": 10}))
            self.assertEqual(R.company_base_request(d)["id"], "t038-v1")
            done = Path(d) / "results" / "x" / "reports" / "company_base" / "t038-v1"
            done.mkdir(parents=True)
            (done / "summary.json").write_text("{}")
            self.assertIsNone(R.company_base_request(d))

    def test_liveness_once_per_day_and_skipped_after_build_stamp(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"A2179_CI_COMPANY_BASE": "auto"}):
            p = Path(d) / "ci" / "requests"
            p.mkdir(parents=True)
            (p / "board_liveness_daily.json").write_text(json.dumps({"id": "live"}))
            self.assertIsNotNone(R.liveness_request(d, "2026-11-21"))
            stamp = Path(d) / "results" / "x" / "reports" / "company_base" / "liveness" / "2026-11-21"
            stamp.mkdir(parents=True)
            (stamp / "summary.json").write_text("{}")
            self.assertIsNone(R.liveness_request(d, "2026-11-21"))
            self.assertIsNotNone(R.liveness_request(d, "2026-11-22"))

    def test_attach_reports_keeps_companies_csv_without_matches(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            co = root / "co"
            dest = root / "results"
            dest.mkdir()
            case = co / "reports" / "company_base" / "t038-v1"
            case.mkdir(parents=True)
            (case / "companies.csv").write_text("name,domain\nAcme,acme.io\n")
            (case / "summary.json").write_text('{"companies":1}')
            p = dest / "2026-11-21_abc1234.json"
            p.write_text("{}")
            attached = wr.attach_reports(co, p)
            self.assertTrue((attached / "reports" / "company_base" / "t038-v1" / "companies.csv").is_file())
            self.assertTrue((attached / "reports" / "company_base" / "t038-v1" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
