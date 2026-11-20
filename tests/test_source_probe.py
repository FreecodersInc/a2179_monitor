"""T-033: the source probe respects robots.txt and records blocks instead of working around them."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import source_probe as P  # noqa: E402


class FakeFetcher:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, url, retry_403=None):
        self.calls.append(url)
        return self.pages.get(url, (404, ""))


class Probe(unittest.TestCase):
    def test_disallowed_path_is_never_fetched(self):
        f = FakeFetcher({"https://a.rs/robots.txt": (200, "User-agent: *\nDisallow: /search\n"),
                         "https://a.rs/": (200, "<title>A</title>ok")})
        rows = P.probe(["https://a.rs/", "https://a.rs/search?q=x"], f)
        self.assertEqual([r["verdict"] for r in rows], ["ok", "disallowed"])
        self.assertNotIn("https://a.rs/search?q=x", f.calls)
        self.assertEqual(rows[0]["title"], "A")

    def test_challenge_or_403_is_blocked(self):
        f = FakeFetcher({"https://b.rs/robots.txt": (403, "<html>Just a moment...</html>")})
        rows = P.probe(["https://b.rs/"], f)
        self.assertEqual(rows[0]["verdict"], "blocked")
        self.assertEqual(f.calls, ["https://b.rs/robots.txt"])
        self.assertEqual(P.verdict(200, "<title>Just a moment...</title>"), "blocked")
        self.assertEqual(P.verdict(200, "<title>ok</title>"), "ok")

    def test_snapshot_only_when_ok(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        f = FakeFetcher({"https://a.rs/robots.txt": (200, "User-agent: *\nAllow: /\n"),
                         "https://a.rs/x": (200, "<title>A</title>"), "https://a.rs/y": (403, "no")})
        rows = P.probe(["https://a.rs/x", "https://a.rs/y"], f, save_dir=d)
        self.assertEqual(rows[0]["snapshot"], "snap_01_a.html")
        self.assertNotIn("snapshot", rows[1])
        self.assertEqual([x.name for x in d.iterdir()], ["snap_01_a.html"])

    def test_robots_rules_and_links(self):
        self.assertEqual(P.robots_rules("User-agent: x\nDisallow: /a\nUser-agent: *\nDisallow: /pretraga # s\nAllow: /k"),
                         ["Disallow: /pretraga", "Allow: /k"])
        html = '<a href="/Racunari-i-oprema/Laptopovi/1">x</a><a href="https://other.rs/laptop">y</a><a href="/Auto">z</a>'
        self.assertEqual(P.links_of(html, "https://a.rs/"), ["https://a.rs/Racunari-i-oprema/Laptopovi/1"])

    def test_forms(self):
        html = '<form action="/Pretraga.php" method="get"><input name="Pretraga"><input type="submit"></form>'
        self.assertEqual(P.forms_of(html), [{"action": "/Pretraga.php", "inputs": ["Pretraga"]}])


if __name__ == "__main__":
    unittest.main()
