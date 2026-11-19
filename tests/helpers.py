import gzip
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIX = ROOT / "tests" / "fixtures"


def fixture(name: str) -> str:
    p = FIX / name
    if name.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8").read()
    return p.read_text(encoding="utf-8")
