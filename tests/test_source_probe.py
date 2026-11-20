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


if __name__ == "__main__":
    unittest.main()
