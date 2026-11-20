"""Tests stay hermetic: CI exports A2179_DENSE_CACHE (ci/run_experiments.py) for the sweep step, and a
test that writes fake scores there poisons the next run (634fa55: TestOomFallback read its own fake
scores back from the runner cache and never called the embedder). Tests that need the cache set a
temp dir themselves."""
import os

os.environ.pop("A2179_DENSE_CACHE", None)
# no real ATS company list in tests: a mocked get_fn must not be walked through 150 boards with pauses
os.environ["A2179_ATS_COMPANIES"] = "off"
# run.sh tests must not depend on what the repo diff looks like right now (ADR-0026 select would skip
# a docs-only HEAD and the run.sh tests would see no result file): force the full pipeline.
os.environ["A2179_CI_MODE"] = "full"
# no network from a test: the fetch request step (ADR-0022) stays off unless a test turns it on
os.environ["A2179_CI_FETCH_LISTINGS"] = "off"
