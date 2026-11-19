import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from tests.helpers import ROOT, fixture
from tests import fake_st as F
from a2179.config import load_config
from a2179.embedder import EmbedSim, EmbedderUnavailable, load_similarity, check_embedder, embedder_wanted
from a2179.similarity import TfidfSim
from a2179.obs import setup_logging


def cfg_with(**emb):
    """In-process paths (the fake sentence_transformers lives in this process): no probe child.
    The probe child itself is tested in tests/test_embed_probe.py."""
    c = load_config("does-not-exist.json")
    c["embedder"]["probe"] = False
    c["embedder"].update(emb)
    return c


class TestEmbedderPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = setup_logging(self.tmp / "logs")

    def fails(self, ctx, stage, **emb):
        err = io.StringIO()
        with ctx:
            sim, mode, e = load_similarity(cfg_with(**emb), self.log, console=err)
        self.assertEqual(mode, "tfidf")
        self.assertIsInstance(sim, TfidfSim)
        self.assertEqual(e.stage, stage, e)
        text = err.getvalue()
        self.assertIn(f"({stage})", text)
        self.assertIn("как починить", text)
        self.assertEqual(text.count("\n"), 3)  # one short block, no traceback unless -v
        log = (self.tmp / "logs" / "monitor.jsonl").read_text(encoding="utf-8")
        self.assertIn('"embedder_unavailable"', log)
        self.assertIn(f'"stage": "{stage}"', log)
        self.assertIn("Traceback", log)  # full traceback kept in the log file
        return e

    def test_verbose_prints_traceback(self):
        err = io.StringIO()
        with F.fake_module(F.raising(OSError("We couldn't connect to 'https://huggingface.co'"))):
            load_similarity(cfg_with(), self.log, console=err, verbose=True)
        self.assertIn("Traceback (most recent call last)", err.getvalue())

    def test_good_module_end_to_end(self):
        with F.fake_module(F.GoodST):
            sim, mode, e = load_similarity(cfg_with(), self.log)
        self.assertEqual((mode, e), ("embed", None))
        p, n = sim.score("LCD ekran za MacBook Air A2179")
        self.assertTrue(-1.01 <= n <= 1.01 and -1.01 <= p <= 1.01)
        self.assertEqual(sim.P.dtype.name, "float32")

    def test_not_installed(self):
        e = self.fails(F.fake_module(None), "not_installed")
        self.assertIn("requirements-embed.txt", e.fix)

    def test_import_oserror_windows_dll(self):
        e = self.fails(F.import_raises(OSError("[WinError 126] fbgemm.dll not found")), "import_failed")
        self.assertIn("Visual C++", e.fix)

    def test_import_numpy_abi(self):
        e = self.fails(F.import_raises(ImportError(
            "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.1")), "import_failed")
        self.assertIn("numpy<2", e.fix)

    def test_download_failed(self):
        self.fails(F.fake_module(F.raising(OSError(
            "We couldn't connect to 'https://huggingface.co' to load this file"))), "download_failed")

    def test_rate_limited(self):
        self.fails(F.fake_module(F.raising(F.HfHubHTTPError("429 Client Error: Too Many Requests"))),
                   "download_failed")

    def test_wrong_model(self):
        e = self.fails(F.fake_module(F.raising(F.RepositoryNotFoundError("404 Repository Not Found"))),
                       "wrong_model", model="sentence-transformers/typo")
        self.assertIn("embedder.model", e.fix)

    def test_offline_not_cached(self):
        self.fails(F.fake_module(F.raising(F.LocalEntryNotFoundError(
            "Cannot find the requested files in the disk cache"))), "offline", local_only=True)

    def test_cache_permission(self):
        e = self.fails(F.fake_module(F.raising(PermissionError("[Errno 13] Permission denied: '/root/.cache'"))),
                       "cache_permission")
        self.assertIn("HF_HOME", e.fix)

    def test_generic_load_failure(self):
        self.fails(F.fake_module(F.raising(ValueError("unrecognized configuration class"))), "load_failed")

    def test_encode_numpy_abi(self):
        e = self.fails(F.fake_module(F.NumpyAbiST), "encode_failed")
        self.assertIn("numpy<2", e.fix)

    def test_encode_bad_shape(self):
        self.fails(F.fake_module(F.BadShapeST), "encode_failed")

    def test_encode_bad_signature(self):
        self.fails(F.fake_module(F.BadSignatureST), "encode_failed")

    def test_old_api_and_list_output_still_work(self):
        with F.fake_module(F.OldApiST):
            sim = EmbedSim("x", local_only=True)  # local_files_only not accepted -> env fallback
        self.assertAlmostEqual(float((sim.P ** 2).sum(axis=1).max()), 1.0, places=4)  # we normalize ourselves

    def test_tensor_output(self):
        with F.fake_module(F.TensorST):
            sim = EmbedSim("x")
        self.assertEqual(sim.P.shape[0], len(sim.pos))

    def test_required_raises_instead_of_fallback(self):
        with F.fake_module(None):
            with self.assertRaises(EmbedderUnavailable):
                load_similarity(cfg_with(), self.log, require=True)
            with self.assertRaises(EmbedderUnavailable):
                load_similarity(cfg_with(enabled="required"), self.log)

    def test_off_never_imports(self):
        with F.import_raises(AssertionError("must not import")):
            sim, mode, e = load_similarity(cfg_with(enabled=False), self.log)
        self.assertEqual(mode, "tfidf")

    def test_wanted_values(self):
        self.assertEqual(embedder_wanted(cfg_with(enabled="auto")), "auto")
        self.assertEqual(embedder_wanted(cfg_with(enabled=True)), "auto")
        self.assertEqual(embedder_wanted(cfg_with(enabled="false")), "off")
        self.assertEqual(embedder_wanted(cfg_with(enabled="required")), "required")
        self.assertEqual(embedder_wanted(cfg_with(required=True)), "required")


class TestCheckEmbedder(unittest.TestCase):
    def test_ok(self):
        out = io.StringIO()
        with F.fake_module(F.GoodST):
            rc = check_embedder(cfg_with(), out, isolate=False)
        s = out.getvalue()
        self.assertEqual(rc, 0, s)
        for k in ("python:", "os:", "torch:", "sentence-transformers:", "numpy:", "model:", "cache dir:",
                  "load:", "encode:", "mode:       embed", "NOT calibrated"):
            self.assertIn(k, s)

    def test_missing(self):
        out = io.StringIO()
        with F.fake_module(None):
            rc = check_embedder(cfg_with(), out, isolate=False)
        self.assertEqual(rc, 3)
        self.assertIn("not_installed", out.getvalue())
        self.assertIn("mode:       tfidf (fallback)", out.getvalue())


class TestRunWithEmbedder(unittest.TestCase):
    def test_monitor_cli_exit_codes(self):
        """Real CLI in a subprocess: real environment has no sentence_transformers."""
        try:
            import sentence_transformers  # noqa: F401
            self.skipTest("sentence_transformers really installed here")
        except ImportError:
            pass
        tmp = Path(tempfile.mkdtemp())
        conf = tmp / "c.json"
        conf.write_text(json.dumps({"data_dir": str(tmp), "sources": {"shop_products": [], "shop_searches": [],
                                                                    "kp_keywords": []}}))
        base = [sys.executable, str(ROOT / "monitor.py"), "--config", str(conf), "--dry-run"]
        r = subprocess.run(base + ["--require-embedder"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("not_installed", r.stderr)
        r = subprocess.run(base, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("not_installed", r.stderr)          # says why
        self.assertIn("regex+tfidf", r.stdout)             # and falls back
        for flag in ("--check-embedder", "check"):
            cmd = [base[0], base[1], flag, "--config", str(conf)] if flag == "check" else base[:4] + [flag]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 3, r.stderr)
            self.assertIn("python:", r.stdout)

    def test_run_once_embed_mode_and_uncalibrated_notice(self):
        from tests.test_run import make_cfg, FakeSession, IFIX, KPQ
        from a2179.fetch import Fetcher
        from a2179.obs import Metrics
        from a2179.run import run_once
        tmp = Path(tempfile.mkdtemp())
        log = setup_logging(tmp / "logs")
        cfg = make_cfg(tmp)
        cfg["embedder"]["enabled"] = "auto"
        cfg["embedder"]["probe"] = False  # the fake module exists only in this process
        routes = {IFIX: (200, fixture("ifix_product.html.gz")), KPQ: (200, fixture("kp_search_synthetic.html"))}
        f = Fetcher({"min_delay_s": 0, "jitter_s": 0, "retries": 0}, Metrics(), log, sleep=lambda s: None,
                    session=FakeSession(routes))
        out = io.StringIO()
        with F.fake_module(F.GoodST), redirect_stdout(out), redirect_stderr(io.StringIO()):
            r = run_once(cfg, log, dry_run=True, fetcher=f)
        self.assertEqual(r["mode"], "embed")
        self.assertIn("не откалиброваны", out.getvalue())
        keys = {l.key for l, _, _ in r["matched"]}
        self.assertIn("kp:180001111", keys)  # regex recall is never reduced by the embedder


class TestWriteThresholds(unittest.TestCase):
    def test_eval_writes_calibrated_thresholds(self):
        sys.path.insert(0, str(ROOT / "eval"))
        import run_eval
        tmp = Path(tempfile.mkdtemp())
        conf = tmp / "config.json"
        conf.write_text(json.dumps({"target_usd": 25}))
        argv = sys.argv
        sys.argv = ["run_eval.py", "--write-thresholds", "--config", str(conf), "--model", "fake/model",
                    "--out-dir", str(tmp)]
        try:
            with F.fake_module(F.GoodST), redirect_stdout(io.StringIO()):
                run_eval.main()
        finally:
            sys.argv = argv
        c = json.loads(conf.read_text())
        self.assertEqual(c["target_usd"], 25)
        self.assertTrue(c["thresholds"]["embed"]["calibrated"])
        self.assertEqual(c["thresholds"]["embed"]["model"], "fake/model")
        rep = json.loads((tmp / "eval_report.json").read_text())
        self.assertEqual(rep["modes"]["hybrid_embed"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
