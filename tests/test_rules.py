import unittest

from tests.helpers import ROOT  # noqa: F401
from a2179.rules import classify
from a2179.matcher import Matcher
from a2179.similarity import TfidfSim
from a2179.embedder import load_similarity


class TestRules(unittest.TestCase):
    def yes(self, t, s=""):
        self.assertTrue(classify(t, s).match, (t, classify(t, s).reasons))

    def no(self, t, s=""):
        self.assertFalse(classify(t, s).match, (t, classify(t, s).reasons))

    def test_positives(self):
        self.yes("LCD Panel Macbook A2179 komplet")
        self.yes("13,3-inčni LCD ekran za Apple MacBook Air Retina A2179 2020")
        self.yes("Экран в сборе для MacBook Air 13 2020 A2179")
        self.yes("Дисплеј MacBook Air А2179")  # Cyrillic А
        self.yes("Ekran za MacBook Air 2020 Intel, polovan")
        self.yes("Macbook air A2179 delovi", "u delove")
        self.yes("MacBook Air A2179", "Prodajem ispravan ekran skinut sa laptopa")
        self.yes("Ekran - LCD za MacBook Air 13\" A1932 / A2179 - zamenski HQ")

    def test_negatives(self):
        self.no("LCD Panel Macbook A2337 LCD")
        self.no("Displej MacBook Air M1 2020 original")
        self.no("Futrola za Macbook Air 13\" M1 A1932/A2179/A2337")
        self.no("Zastita za ekran za MacBook Air 13 A2179")
        self.no("Flet kabl LVDS eDP LCDa za MacBook Air 13\" A1932/A2179")
        self.no("Apple MacBook Air A2179 (2020) Core i3 / 8GB RAM / 256GB")
        self.no("Zamena ekrana za MacBook Air A2179", "SiriX Servis")
        self.no("Baterija A1965 za MacBook Air 13\" A1932/A2179")
        self.no("LCD ekran za Lenovo ThinkPad T480")

    def test_exclusions_only_from_title(self):
        # a description mentioning installation/service must not kill a real panel listing
        self.yes("LCD ekran MacBook Air A2179", "Moguca ugradnja u nasem servisu")

    def test_a1932_switch(self):
        self.assertTrue(classify("LCD Panel Macbook A1932 LCD", accept_a1932=True).match)
        self.assertFalse(classify("LCD Panel Macbook A1932 LCD", accept_a1932=False).match)


class TestMatcher(unittest.TestCase):
    def test_regex_only(self):
        m = Matcher(None)
        self.assertEqual(m.mode, "regex")
        self.assertTrue(m.decide("LCD Panel Macbook A2179 LCD").match)

    def test_sim_rescue_needs_gate(self):
        sim = TfidfSim()
        m = Matcher(sim, "tfidf", {"tfidf": {"pos": 0.0, "margin": -1.0}})
        # everything passes thresholds, but no A2179-compatible model -> no match
        self.assertFalse(m.decide("Ekran za iPhone 11").match)
        self.assertFalse(m.decide("LCD ekran za MacBook Air M1 A2337").match)
        # gated + regex miss -> similarity rescues
        d = m.decide("Prodajem LCD od Macbook Air 2020 (Intel i5), ispravan")
        self.assertTrue(d.match)
        self.assertFalse(d.regex)
        self.assertTrue(d.sim)

    def test_tfidf_orders_prototypes(self):
        sim = TfidfSim()
        p1, n1 = sim.score("LCD ekran MacBook Air A2179 komplet")
        p2, n2 = sim.score("Futrola za MacBook Air A2179")
        self.assertGreater(p1 - n1, p2 - n2)

    def test_embedder_fallback(self):
        cfg = {"embedder": {"enabled": "auto", "model": "this/model-does-not-exist-xyz", "local_only": True}}
        sim, mode, err = load_similarity(cfg, console=None)
        self.assertEqual(mode, "tfidf")
        self.assertIsInstance(sim, TfidfSim)


if __name__ == "__main__":
    unittest.main()
