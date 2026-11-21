import importlib.util
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location("geo_filter", Path(__file__).resolve().parents[1] / "scripts" / "geo_filter.py")
geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geo)


class TestGeoFilter(unittest.TestCase):
    def test_buckets(self):
        cases = {
            "Remote, Global": "keep", "Home based - EMEA": "keep", "Remote": "keep",
            "Belgrade, London, Dublin, Remote": "keep", "Europe or US, Remote": "keep",
            "Germany (Remote)": "check", "London, Remote": "check",
            "Remote - United States": "cut", "Remote (U.S.)": "cut", "Home Based - Americas": "cut",
            "Remote - India": "cut", "San Francisco, CA": "cut", "Paris, France": "cut",
        }
        for loc, want in cases.items():
            self.assertEqual(geo.verdict(loc)[0], want, loc)

    def test_title_country_goes_to_check(self):
        self.assertEqual(geo.verdict("Remote", "SRE — UK (Senior)")[0], "check")

    def test_jetbrains_belgrade_has_no_special_case(self):
        """T-038 control: a JetBrains Belgrade remote role uses the same rules as any other company."""
        self.assertEqual(geo.verdict("Belgrade, Remote")[0], "keep")
        self.assertEqual(geo.verdict("Belgrade, Serbia, Remote", "Software Engineer, JetBrains")[0], "keep")
        src = Path(geo.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("jetbrains", src)
        self.assertNotIn("jet brains", src)


if __name__ == "__main__":
    unittest.main()
