"""CI guard: HYPOTHESES, EXP verdict lines, work/T-* headers, work/STATUS.md and the PROJECT_REPORT status block agree."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import status_check  # noqa: E402


class StatusSync(unittest.TestCase):
    def test_repo_is_in_sync(self):
        errs = status_check.problems() + status_check.stale()
        self.assertEqual(errs, [], "\n".join(errs))

    def _copy(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d)
        for sub in ("experiments", "work", "reports"):
            (d / sub).mkdir()
        for f in (ROOT / "experiments").glob("*.md"):
            shutil.copy(f, d / "experiments" / f.name)
        for f in (ROOT / "work").glob("*.md"):
            shutil.copy(f, d / "work" / f.name)
        shutil.copy(ROOT / "reports" / "PROJECT_REPORT.md", d / "reports" / "PROJECT_REPORT.md")
        return d

    def test_verdict_divergence_is_caught(self):
        d = self._copy()
        f = next((d / "experiments").glob("EXP-026_*.md"))
        f.write_text(f.read_text().replace("| вердикты | H-45 refuted |", "| вердикты | H-45 accepted |"))
        self.assertTrue(any("H-45" in e for e in status_check.problems(d)))

    def test_stale_status_file_is_caught(self):
        d = self._copy()
        (d / "work" / "STATUS.md").write_text("old\n")
        self.assertTrue(status_check.stale(d))

    def test_checked_hypothesis_needs_a_decision(self):
        d = self._copy()
        h = d / "experiments" / "HYPOTHESES.md"
        lines = [l if not l.startswith("| H-43 | accepted |") else "|".join(l.split("|")[:5]) + "|  |" for l in h.read_text().split("\n")]
        h.write_text("\n".join(lines))
        self.assertTrue(any(e.startswith("H-43") and "решения" in e for e in status_check.problems(d)))


if __name__ == "__main__":
    unittest.main()
