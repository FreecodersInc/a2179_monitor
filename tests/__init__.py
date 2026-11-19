"""Tests stay hermetic: CI exports A2179_DENSE_CACHE (ci/run_experiments.py) for the sweep step, and a
test that writes fake scores there poisons the next run (634fa55: TestOomFallback read its own fake
scores back from the runner cache and never called the embedder). Tests that need the cache set a
temp dir themselves."""
import os

os.environ.pop("A2179_DENSE_CACHE", None)
# no real ATS company list in tests: a mocked get_fn must not be walked through 150 boards with pauses
os.environ["A2179_ATS_COMPANIES"] = "off"
