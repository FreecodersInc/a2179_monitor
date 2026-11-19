"""eval/enrich.py offline: saved KP/shop pages -> pre-labelled candidates -> reviewed merge."""
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tests.helpers import FIX, ROOT
from tests.test_run import FakeSession
from a2179.config import load_config
from a2179.fetch import Fetcher
from a2179.obs import Metrics, setup_logging
from a2179.parsers import Listing

sys.path.insert(0, str(ROOT / "eval"))
import enrich as E  # noqa: E402

KP_PAGES = ("kp_next_data_dryrun.html", "kp_search_synthetic.html", "kp_blocked_403.html.gz")


NEW_KP = [  # not in labeled.jsonl (the saved KP fixtures are all labeled already, they must drop out)
    ("Šarke MacBook Air A2179 par, original", 2500),
    ("LCD ekran sa šarkama i kamerom MacBook Air A2179", 21000),
    ("Kupujem ekran MacBook Air A2179", 0),
    ("Prodajem MacBook Air 2020 i5 8GB 256GB, baterija 90%", 65000),
]


def offline_dir(*names, new_kp=True):
    d = Path(tempfile.mkdtemp())
    for n in names:
        shutil.copy(FIX / n, d / n)
    if new_kp:
        ads = [{"name": t, "adUrl": f"/kompjuteri/delovi/x/oglas/{900000 + i}", "price": p, "currency": "rsd",
                "locationName": "Beograd"} for i, (t, p) in enumerate(NEW_KP)]
        (d / "kp_new.html").write_text('<script id="__NEXT_DATA__" type="application/json">'
                                       + json.dumps({"props": {"pageProps": {"ads": ads}}}) + "</script>",
                                       encoding="utf-8")
    return d


def run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = E.main(argv)
    return rc, buf.getvalue()


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l]


class TestRules(unittest.TestCase):
    def test_intent(self):
        self.assertEqual(E.intent_of("Tražim ekran za MacBook Air A2179"), "seek")
        self.assertEqual(E.intent_of("Kupujem MacBook Air A2179 u bilo kom stanju"), "seek")
        self.assertEqual(E.intent_of("Servis za MacBook: zamena LCD ekrana A2179"), "service")
        self.assertEqual(E.intent_of("Prodajem LCD MacBook Air A2179"), "sell")
        self.assertEqual(E.intent_of("LCD Panel Macbook A2179 komplet", "Novo, garancija 6 meseci"), "sell")
        self.assertEqual(E.intent_of("LCD Panel Macbook A2179"), "unknown")

    def test_labels_and_flags(self):
        lab = E.Labeler(load_config("does-not-exist.json"))
        L = lambda t: lab.label(Listing("kupujemprodajem", "https://www.kupujemprodajem.com/x/oglas/1", t), "q")  # noqa: E731
        r = L("LCD Panel Macbook A2179 komplet")
        self.assertEqual((r["auto_label"], r["kind"], r["priority"], r["needs_review"], r["reviewed"]),
                         (1, "display", 2, True, False))
        r = L("Šarke za ekran MacBook Air A2179")
        self.assertEqual(r["auto_label"], 0)
        self.assertIn("hard_negative_candidate", r["flags"])
        self.assertEqual(r["priority"], 1)
        self.assertIn("hinge", r["feature_groups"])
        r = L("Tražim ekran za MacBook Air A2179")
        self.assertEqual((r["auto_label"], r["intent"], r["kind"]), (0, "seek", "wanted"))
        r = L("Futrola za MacBook Air A2179")
        self.assertIn("rules=" if False else "model=strong", r["reason"])
        self.assertIn("intent=", r["reason"])
        self.assertIn("matcher=", r["reason"])

    def test_disagreement_first(self):
        lab = E.Labeler(load_config("does-not-exist.json"))
        # regex says display, intent says the seller wants one: a disagreement, reviewed first
        # regex: a display (lcd ekran ...), features: hinge + camera words. A human decides (here it is
        # probably a full lid, a positive), so it goes to the top of the review list
        r = lab.label(Listing("kupujemprodajem", "https://www.kupujemprodajem.com/x/oglas/2",
                              "LCD ekran sa šarkama i kamerom MacBook Air A2179"), "q")
        self.assertEqual((r["auto_label"], r["priority"]), (1, 0), r)
        self.assertIn("rules_vs_features", r["flags"])
        # a buyer: regex already says no (hard exclusion), so it is a hard negative, not a disagreement
        r = lab.label(Listing("kupujemprodajem", "https://www.kupujemprodajem.com/x/oglas/3",
                              "Kupujem LCD ekran za MacBook Air A2179"), "q")
        self.assertEqual((r["auto_label"], r["priority"], r["intent"]), (0, 1, "seek"))

    def test_shop_url_and_plan(self):
        self.assertEqual(E.shop_url("https://ifix.rs/?s=a2179&post_type=product", "a2179 lcd"),
                         "https://ifix.rs/?s=a2179+lcd&post_type=product")
        cfg = load_config("does-not-exist.json")
        p = E.plan(cfg, ["a2179", "a2179 sarke"], "base")
        kinds = [k for k, _, _ in p]
        self.assertEqual(kinds.count("shop_search"), 3)   # only the base query goes to the 3 shops
        self.assertEqual(kinds.count("kp"), 2)
        self.assertIn("keywords=a2179+sarke", p[-1][1])
        self.assertEqual(len(E.plan(cfg, ["a2179 sarke"], "none")), 1)

    def test_default_queries_cover_parts_and_intents(self):
        q = " ".join(E.DEFAULT_QUERIES)
        for w in ("ekran", "displej", "lcd", "sarke", "srafovi", "kuler", "ventilator", "staklo", "punjac",
                  "tastatura", "baterija", "maska", "kabl", "maticna", "servis", "popravka", "kupujem", "trazim",
                  "prodajem", "delove", "macbook air 2020"):
            self.assertIn(w, q)


class TestCollectOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.labeled = self.tmp / "labeled.jsonl"
        shutil.copy(ROOT / "eval" / "labeled.jsonl", self.labeled)

    def collect(self, d, *extra):
        out = self.tmp / "cand.jsonl"
        rc, text = run(["collect", "--offline-html", str(d), "--out", str(out), "--labeled", str(self.labeled),
                        "--config", "does-not-exist.json", *extra])
        self.assertEqual(rc, 0, text)
        return read_jsonl(out), text, out

    def test_kp_and_shop_pages(self):
        rows, text, out = self.collect(offline_dir(*KP_PAGES, "ifix_search.html.gz"), "--csv")
        self.assertIn("kp:blocked_page_skipped", text)
        self.assertTrue(rows)
        self.assertTrue(all(r["needs_review"] and not r["reviewed"] for r in rows))
        self.assertEqual([r["priority"] for r in rows], sorted(r["priority"] for r in rows))
        self.assertEqual([r["cand_id"] for r in rows], list(range(1, len(rows) + 1)))
        labeled_titles = {E.normalize(json.loads(l)["title"]) for l in self.labeled.read_text().splitlines() if l}
        self.assertFalse([r for r in rows if E.normalize(r["title"]) in labeled_titles])  # dedup vs labeled
        self.assertIn("already_labeled", text)
        self.assertEqual(len({r["key"] for r in rows}), len(rows))
        sites = {r["site"] for r in rows}
        self.assertIn("kupujemprodajem", sites)
        self.assertIn("ifix.rs", sites)
        kp = {r["title"]: r for r in rows if r["site"] == "kupujemprodajem"}
        self.assertEqual(set(kp), {t for t, _ in NEW_KP})  # only the new ads, the labeled ones dropped
        self.assertEqual(rows[0]["title"], "LCD ekran sa šarkama i kamerom MacBook Air A2179")  # disagreement first
        self.assertEqual(kp["Kupujem ekran MacBook Air A2179"]["kind"], "wanted")
        self.assertIsNone(kp["Kupujem ekran MacBook Air A2179"]["price"])  # KP price 0 = no price
        for r in rows:
            for k in ("title", "url", "label", "kind", "reason", "flags", "intent", "sim_pos", "query"):
                self.assertIn(k, r)
        with out.with_suffix(".csv").open(encoding="utf-8-sig") as f:
            c = list(csv.DictReader(f))
        self.assertEqual(len(c), len(rows))
        self.assertEqual(c[0]["reviewed"], "")

    def test_limit_keeps_top_priority(self):
        rows, text, _ = self.collect(offline_dir(*KP_PAGES, "ifix_search.html.gz"), "--limit", "2")
        self.assertEqual(len(rows), 2)
        self.assertIn("over_limit", text)

    def test_dry_run_writes_nothing(self):
        out = self.tmp / "none.jsonl"
        rc, text = run(["collect", "--offline-html", str(offline_dir(*KP_PAGES)), "--out", str(out), "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertFalse(out.exists())
        rc, text = run(["collect", "--config", "does-not-exist.json", "--dry-run", "--out", str(out)])
        self.assertIn("nothing fetched", text)
        self.assertIn("kupujemprodajem.com/pretraga?keywords=a2179+sarke", text)
        self.assertFalse(out.exists())


class TestCollectOnline(unittest.TestCase):
    """The network path with a fake session (sequential) and a fake fetch_all (parallel): no live KP."""
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = setup_logging(self.tmp / "logs")
        self.cfg = load_config("does-not-exist.json")
        self.cfg["http"] = {"min_delay_s": 0, "jitter_s": 0, "retries": 0, "retry_403": 0}
        self.cfg["pagination"] = {"max_pages": 2}

    def test_sequential_stops_kp_after_block(self):
        tasks = E.plan(self.cfg, ["a2179", "a2179 sarke", "a2179 kuler"], "none")
        blocked = (403, (FIX / "kp_blocked_403.html.gz").read_bytes().decode("latin-1"))
        routes = {tasks[0][1]: (200, (FIX / "kp_next_data_dryrun.html").read_text()), tasks[1][1]: blocked}
        s = FakeSession(routes)
        f = Fetcher(self.cfg["http"], Metrics(), self.log, sleep=lambda x: None, session=s)
        found, summary = E.collect_online(self.cfg, tasks, Metrics(), self.log, workers=0, fetcher=f)
        self.assertEqual(len({l.key for l, _ in found}), 12)
        self.assertEqual({q for _, q in found}, {"a2179"})
        self.assertEqual(summary["kp:blocked"], 1)
        self.assertEqual(summary["kp:skipped_kp_blocked"], 1)
        self.assertFalse(any(tasks[2][1] in c for c in s.calls))  # third KP search never sent

    def test_parallel_results_are_mapped_back_to_queries(self):
        tasks = E.plan(self.cfg, ["a2179", "a2179 lcd"], "base")
        seen = {}

        def fake_fetch_all(cfg, metrics, log, tb, w, mc, store):
            seen["cfg"] = cfg
            res = []
            for i, (k, u, q) in enumerate(tasks):
                items = [Listing("kupujemprodajem", f"https://www.kupujemprodajem.com/x/oglas/{i}", f"LCD A2179 {q}")]
                res.append({"i": i, "kind": k, "url": u, "status": 200, "items": items, "skipped": None,
                            "blocked": False})
            return res
        orig = E.parallel.fetch_all
        E.parallel.fetch_all = fake_fetch_all
        try:
            found, summary = E.collect_online(self.cfg, tasks, Metrics(), self.log, workers=2)
        finally:
            E.parallel.fetch_all = orig
        self.assertEqual(seen["cfg"]["sources"]["kp_keywords"], ["a2179", "a2179 lcd"])
        self.assertEqual(seen["cfg"]["sources"]["shop_products"], [])
        self.assertEqual(len(seen["cfg"]["sources"]["shop_searches"]), 3)
        self.assertEqual(sorted(q for _, q in found), sorted(q for _, _, q in tasks))
        self.assertEqual(summary["kp:200"], 2)


class TestMerge(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.labeled = self.tmp / "labeled.jsonl"
        shutil.copy(ROOT / "eval" / "labeled.jsonl", self.labeled)
        self.before = read_jsonl(self.labeled)

    def cand(self):
        out = self.tmp / "cand.jsonl"
        run(["collect", "--offline-html", str(offline_dir(*KP_PAGES, "ifix_search.html.gz")), "--out", str(out),
             "--labeled", str(self.labeled), "--config", "does-not-exist.json", "--csv"])
        return out

    def test_only_reviewed_rows_are_appended(self):
        out = self.cand()
        rows = read_jsonl(out)
        self.assertGreaterEqual(len(rows), 3)
        rows[0]["reviewed"] = True
        rows[1]["reviewed"] = True
        rows[1]["label"] = 1 - rows[1]["label"]  # a human correction wins
        rows[2]["reviewed"] = False
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        rc, text = run(["merge", "--reviewed", str(out), "--labeled", str(self.labeled)])
        after = read_jsonl(self.labeled)
        self.assertEqual(after[:len(self.before)], self.before)  # nothing changed or removed
        new = after[len(self.before):]
        self.assertEqual(len(new), 2)
        self.assertEqual([n["id"] for n in new], [max(r["id"] for r in self.before) + 1, max(r["id"] for r in self.before) + 2])
        self.assertTrue(all(n["source"] == "real" for n in new))
        self.assertEqual(new[1]["label"], rows[1]["label"])
        self.assertEqual(new[0]["url"], rows[0]["url"])
        # merging the same file again adds nothing
        rc, text = run(["merge", "--reviewed", str(out), "--labeled", str(self.labeled)])
        self.assertIn("'duplicate': 2", text)
        self.assertEqual(len(read_jsonl(self.labeled)), len(after))

    def test_csv_review_and_bad_labels(self):
        c = self.cand().with_suffix(".csv")
        with c.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        rows[0]["reviewed"] = "1"
        rows[1]["reviewed"] = "да"
        rows[1]["label"] = "maybe"
        with c.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        rc, text = run(["merge", "--reviewed", str(c), "--labeled", str(self.labeled), "--dry-run"])
        self.assertIn("'added': 1", text)
        self.assertIn("'bad_label': 1", text)
        self.assertEqual(read_jsonl(self.labeled), self.before)  # dry run
        run(["merge", "--reviewed", str(c), "--labeled", str(self.labeled)])
        self.assertEqual(len(read_jsonl(self.labeled)), len(self.before) + 1)


if __name__ == "__main__":
    unittest.main()
