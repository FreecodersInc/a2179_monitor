"""Target profiles: the A2179 profile reproduces the old hardcoded filter exactly, and another
profile can be swapped in through config.json or A2179_TARGET."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from tests.helpers import FIX, ROOT
from a2179 import config, rules, target as T
from a2179.matcher import Matcher
from a2179.rules import classify
from a2179.similarity import TfidfSim

TOY = {
    "name": "iphone11_battery",
    "label": "iPhone 11 battery",
    "description": "toy profile: iPhone 11 battery or donor phone",
    "rules": {
        "models": [{"name": "strong", "regex": r"\biphone\s?11\b(?!\s?pro)|\ba2111\b"},
                   {"name": "a2221", "regex": r"\ba2221\b", "accept_option": "accept_a2221",
                    "alert_note": "A2221, check the region"}],
        "family": r"\biphone\b",
        "conflict": r"\bpro\b|\bmax\b",
        "weak": [{"all": [r"\b2019\b"], "unless_conflict": True}],
        "no_model_reason": "no iPhone 11",
        "part_terms": ["baterija", "battery", "батерија"],
        "strong_part": r"^\s*(baterija|battery)\b",
        "donor_terms": ["za delove", "ne radi"],
        "hard_excl": r"\b(maska|futrola|staklo)\b",
        "soft_excl_terms": ["kamera", "ekran"],
        "specs": r"\b\d{2,3}\s?gb\b",
    },
    "prototypes": {"positive": ["Baterija za iPhone 11 nova"], "negative": ["Maska za iPhone 11"]},
    "defaults": {"target_usd": 15.0, "accept_a2221": True,
                 "sources": {"shop_products": [], "shop_searches": ["https://example.test/?s=iphone+11+baterija"],
                             "kp_keywords": ["iphone 11 baterija"]}},
}


class TestA2179Identical(unittest.TestCase):
    def test_compiled_patterns_equal_the_old_hardcoded_ones(self):
        frozen = json.loads((FIX / "a2179_rules_frozen.json").read_text())
        R = T.default_target().rules
        now = {"STRONG_MODEL": R.models[0].rx, "A1932": R.models[1].rx, "A2337": R.conflict, "MACBOOK_AIR": R.family,
               "PART": R.part, "STRONG_PART": R.strong_part, "DONOR": R.donor, "HARD_EXCL": R.hard_excl,
               "SOFT_EXCL": R.soft_excl, "LAPTOP_SPECS": R.specs,
               "YEAR_2020": R.weak[0].all[0], "INTEL": R.weak[0].all[1], "RANGE_2018_2020": R.weak[1].all[0]}
        for k, pat in frozen.items():
            if k in ("PART", "DONOR", "SOFT_EXCL"):  # term lists: same alternatives, longest first
                alts = lambda p: p[len(r"\b(?:"):-1].split("|")
                self.assertEqual(sorted(alts(now[k].pattern)), sorted(alts(pat)), k)
                lens = [len(a) for a in alts(now[k].pattern)]
                self.assertEqual(lens, sorted(lens, reverse=True), k)
            else:
                self.assertEqual(now[k].pattern, pat, k)

    def test_decisions_equal_the_old_ones(self):
        rows = json.loads((FIX / "a2179_decisions_frozen.json").read_text())
        self.assertGreater(len(rows), 250)
        for title, snippet, acc, match, model, gate, reasons in rows:
            r = classify(title, snippet, acc)
            self.assertEqual((r.match, r.model, r.gate, r.reasons), (match, model, gate, reasons), title)
            r2 = classify(title, snippet, target=T.default_target(), options={"accept_a1932": acc})
            self.assertEqual((r2.match, r2.model, r2.gate, r2.reasons), (match, model, gate, reasons), title)

    def test_defaults_equal_the_old_ones(self):
        frozen = json.loads((FIX / "a2179_defaults_frozen.json").read_text())
        now = {k: v for k, v in config.DEFAULTS.items() if k != "target"}
        self.assertEqual(now, frozen)
        self.assertEqual(config.DEFAULTS["target"], "a2179_screen")

    def test_module_views(self):
        self.assertIs(rules.STRONG_MODEL, T.default_target().rules.models[0].rx)
        from a2179 import prototypes
        self.assertEqual(prototypes.POSITIVE, T.default_target().positive)

    def test_load_config_default_profile(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = config.load_config(str(Path(d) / "none.json"))
        self.assertEqual(cfg["target"], "a2179_screen")
        self.assertEqual(cfg["sources"], config.DEFAULTS["sources"])
        self.assertEqual(cfg["target_usd"], 20.0)


class TestSwap(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.prof = self.tmp / "iphone11_battery.json"
        self.prof.write_text(json.dumps(TOY))
        self._env = os.environ.pop("A2179_TARGET", None)

    def tearDown(self):
        if self._env is not None:
            os.environ["A2179_TARGET"] = self._env
        else:
            os.environ.pop("A2179_TARGET", None)

    def test_toy_rules(self):
        t = T.load_target(str(self.prof))
        self.assertTrue(classify("Baterija za iPhone 11 nova", target=t).match)
        self.assertFalse(classify("Maska za iPhone 11", target=t).match)
        self.assertFalse(classify("Baterija iPhone 11 Pro", target=t).match)       # conflict, no tier
        self.assertEqual(classify("Baterija iPhone 11 Pro", target=t).reasons[-1], "no iPhone 11")
        self.assertTrue(classify("iPhone 11 ne radi", target=t).match)              # donor
        self.assertFalse(classify("iPhone 11 64GB baterija 89%", target=t).match)    # specs
        self.assertEqual(classify("Baterija iPhone 2019", target=t).model, "weak")
        r = classify("Baterija A2221", target=t, options={"accept_a2221": False})
        self.assertEqual(r.model, "none")
        self.assertEqual(classify("Baterija A2221", target=t).model, "a2221")
        # the A2179 profile does not know this phone
        self.assertFalse(classify("Baterija za iPhone 11 nova").match)

    def test_config_json_picks_profile(self):
        cfgp = self.tmp / "config.json"
        cfgp.write_text(json.dumps({"target": str(self.prof), "target_usd": 12}))
        cfg = config.load_config(str(cfgp))
        self.assertEqual(cfg["target"], str(self.prof))
        self.assertEqual(cfg["sources"]["kp_keywords"], ["iphone 11 baterija"])
        self.assertEqual(cfg["target_usd"], 12)                 # config.json over profile defaults
        self.assertNotIn("accept_a1932", cfg)                   # A2179 option does not leak in
        self.assertEqual(cfg["thresholds"], {}) if "thresholds" in cfg else None

    def test_env_picks_profile(self):
        os.environ["A2179_TARGET"] = str(self.prof)
        cfg = config.load_config(str(self.tmp / "none.json"))
        self.assertEqual(T.target_of(cfg).name, "iphone11_battery")
        self.assertEqual(cfg["target_usd"], 15.0)

    def test_matcher_and_similarity_use_profile(self):
        t = T.load_target(str(self.prof))
        sim = TfidfSim(t.positive, t.negative)
        m = Matcher(sim, "tfidf", {"tfidf": {"pos": 0.3, "margin": 0.0}}, target=t)
        self.assertTrue(m.decide("Baterija za iPhone 11 nova").match)
        self.assertFalse(m.decide("Maska za iPhone 11").match)

    def test_named_profile_in_targets_dir(self):
        self.assertIn("a2179_screen", T.list_targets())
        self.assertEqual(T.load_target("a2179_screen").label, "A2179")
        self.assertIn("de_jobs", T.list_targets())
        self.assertNotIn("de_jobs.local", T.list_targets())
        de = T.load_target("de_jobs")
        self.assertTrue(de.positive and de.negative)
        blob = (ROOT / "targets" / "de_jobs.json").read_text()
        for needle in ("cv_base", "Novi Sad", "@", "Blokhin"):
            self.assertNotIn(needle, blob)
        self.assertTrue(classify("Senior Data Engineer Toronto Spark Airflow", target=de).match)
        self.assertFalse(classify("Excel data entry clerk no SQL", target=de).match)


class TestErrors(unittest.TestCase):
    def test_bad_profiles(self):
        bad = copy.deepcopy(TOY)
        bad["rules"]["family"] = "("
        with self.assertRaises(T.TargetError):
            T.load_target(bad)
        bad = copy.deepcopy(TOY)
        bad["prototypes"]["negative"] = []
        with self.assertRaises(T.TargetError):
            T.load_target(bad)
        bad = copy.deepcopy(TOY)
        bad["rules"]["models"].append({"name": "weak", "regex": "x"})
        with self.assertRaises(T.TargetError):
            T.load_target(bad)
        with self.assertRaises(T.TargetError):
            T.load_target("../../etc/passwd")
        with self.assertRaises(T.TargetError):
            T.load_target("no_such_target")

    def test_empty_term_lists_match_nothing(self):
        t = copy.deepcopy(TOY)
        t["rules"]["donor_terms"] = []
        self.assertFalse(classify("iPhone 11 ne radi", target=T.load_target(t)).match)


class TestRunUsesProfile(unittest.TestCase):
    def test_alert_note_comes_from_tier(self):
        t = T.default_target()
        self.assertEqual(t.tier("a1932").alert_note, "A1932, совместимость уточни у продавца")
        self.assertIsNone(t.tier("weak"))


if __name__ == "__main__":
    unittest.main()
