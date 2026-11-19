"""Crash-safe embedder: the probe child, device choice, import order."""
import importlib.abc
import importlib.machinery
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from tests.helpers import ROOT  # noqa: F401
from tests import fake_st as F
from a2179 import embedder as E
from a2179.config import load_config
from a2179.obs import setup_logging
from a2179.similarity import TfidfSim

POSIX = os.name == "posix"
PY = sys.executable


def child(code):
    return [PY, "-c", code]


SEGV = child("import os, signal; os.kill(os.getpid(), signal.SIGSEGV)")
ABORT = child("import os; os.abort()")
HANG = child("import time; time.sleep(30)")
OK_CUDA = child(f"print({E.PROBE_MARK!r} + '{{\"ok\": true, \"device\": \"cuda\", \"dim\": 64, \"load_s\": 0.1}}')")


def cfg_probe(**emb):
    c = load_config("does-not-exist.json")
    c["embedder"].update(emb)
    return c


class TestRunProbe(unittest.TestCase):
    @unittest.skipUnless(POSIX, "signals")
    def test_segfault_child(self):
        r = E.run_probe(cfg_probe(), cmd=SEGV)
        self.assertFalse(r["ok"])
        self.assertEqual(r["stage"], "crashed")
        self.assertIn("SIGSEGV", r["cause"])
        self.assertIn("Segmentation fault", r["stderr_tail"])  # faulthandler is on in the child

    @unittest.skipUnless(POSIX, "signals")
    def test_abort_child(self):
        r = E.run_probe(cfg_probe(), cmd=ABORT)
        self.assertEqual((r["ok"], r["stage"]), (False, "crashed"))
        self.assertIn("SIGABRT", r["cause"])

    def test_timeout_child_is_killed(self):
        import time
        t0 = time.monotonic()
        r = E.run_probe(cfg_probe(), timeout_s=1, cmd=HANG)
        self.assertLess(time.monotonic() - t0, 10)
        self.assertEqual((r["ok"], r["stage"]), (False, "probe_timeout"))
        self.assertIn("probe_timeout_s", r["cause"])

    def test_nonzero_exit_without_result(self):
        r = E.run_probe(cfg_probe(), cmd=child("import sys; sys.stderr.write('boom here\\n'); sys.exit(2)"))
        self.assertEqual(r["stage"], "probe_failed")
        self.assertIn("boom here", r["cause"])

    @unittest.skipUnless(POSIX, "signals")
    def test_crash_at_exit_after_ok_still_counts_as_crash(self):
        code = (f"import os, signal; print({E.PROBE_MARK!r} + '{{\"ok\": true, \"device\": \"cuda\"}}', flush=True); "
                "os.kill(os.getpid(), signal.SIGSEGV)")
        r = E.run_probe(cfg_probe(), cmd=child(code))
        self.assertEqual(r["stage"], "crashed")
        self.assertIn("после успешной", r["cause"])

    def test_ok_result_and_device(self):
        r = E.run_probe(cfg_probe(), cmd=OK_CUDA)
        self.assertTrue(r["ok"])
        self.assertEqual((r["device"], r["dim"]), ("cuda", 64))

    def test_real_child_protocol(self):
        """The real child module, here without sentence_transformers installed: a clean stage, no crash."""
        try:
            import sentence_transformers  # noqa: F401
            self.skipTest("sentence_transformers really installed here")
        except ImportError:
            pass
        r = E.run_probe(cfg_probe(local_only=True))
        self.assertEqual((r["ok"], r["stage"], r["returncode"]), (False, "not_installed", 0))
        out = io.StringIO()
        self.assertEqual(E.check_embedder(cfg_probe(local_only=True), out), 3)
        self.assertIn("python:", out.getvalue())
        self.assertIn("device:     embedder.device=auto", out.getvalue())
        self.assertIn("not_installed", out.getvalue())


class CountingST(F.GoodST):
    made = []

    def __init__(self, name, device=None, **kw):
        CountingST.made.append(device)
        super().__init__(name, device=device, **kw)


class TestFallbackOnCrash(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = setup_logging(self.tmp / "logs")
        self.orig = E.run_probe
        CountingST.made = []

    def tearDown(self):
        E.run_probe = self.orig

    def probe_with(self, cmd):
        orig = self.orig
        E.run_probe = lambda cfg, *a, **k: orig(cfg, *a, **dict(k, cmd=cmd, timeout_s=k.get("timeout_s", 2)))

    @unittest.skipUnless(POSIX, "signals")
    def test_segfault_falls_back_to_regex_and_never_loads_in_process(self):
        self.probe_with(SEGV)
        err = io.StringIO()
        with F.fake_module(CountingST):
            sim, mode, e = E.load_similarity(cfg_probe(), self.log, console=err)
        self.assertEqual(mode, "tfidf")
        self.assertIsInstance(sim, TfidfSim)
        self.assertEqual(e.stage, "crashed")
        self.assertEqual(CountingST.made, [])  # torch/model never touched in the main process
        self.assertIn("(crashed)", err.getvalue())
        self.assertIn("regex + TF-IDF", err.getvalue())
        log = (self.tmp / "logs" / "monitor.jsonl").read_text(encoding="utf-8")
        self.assertIn('"embedder_probe"', log)
        self.assertIn('"returncode": -11', log)
        self.assertIn('"stage": "crashed"', log)

    def test_timeout_falls_back(self):
        self.probe_with(HANG)
        with F.fake_module(CountingST):
            sim, mode, e = E.load_similarity(cfg_probe(probe_timeout_s=1), self.log, console=io.StringIO())
        self.assertEqual((mode, e.stage, CountingST.made), ("tfidf", "probe_timeout", []))

    @unittest.skipUnless(POSIX, "signals")
    def test_required_raises_with_crash_stage(self):
        self.probe_with(ABORT)
        with self.assertRaises(E.EmbedderUnavailable) as cm:
            E.load_similarity(cfg_probe(), self.log, require=True, console=io.StringIO())
        self.assertEqual(cm.exception.stage, "crashed")

    def test_probe_ok_loads_in_process_on_the_probed_device(self):
        self.probe_with(OK_CUDA)
        with F.fake_module(CountingST):
            sim, mode, e = E.load_similarity(cfg_probe(), self.log, console=io.StringIO())
        self.assertEqual((mode, sim.device, CountingST.made), ("embed", "cuda", ["cuda"]))

    @unittest.skipUnless(POSIX, "signals")
    def test_check_embedder_reports_crash(self):
        out = io.StringIO()
        rc = E.check_embedder(cfg_probe(), out, cmd=SEGV)
        s = out.getvalue()
        self.assertEqual(rc, 3)
        for k in ("probe:      FAILED", "SIGSEGV", "(crashed)", "mode:       tfidf (fallback)"):
            self.assertIn(k, s)

    @unittest.skipUnless(POSIX, "signals")
    def test_monitor_cli_survives_crashing_probe(self):
        """Whole CLI: the probe child crashes, the monitor finishes its pass on regex + TF-IDF."""
        import subprocess
        tmp = Path(tempfile.mkdtemp())
        conf = tmp / "c.json"
        conf.write_text(json.dumps({"data_dir": str(tmp), "sources": {"shop_products": [], "shop_searches": [],
                                                                    "kp_keywords": []}}))
        boot = ("import sys, runpy; sys.path.insert(0, %r); import a2179.embedder as E; "
                "orig = E.run_probe; "
                "E.run_probe = lambda cfg, *a, **k: orig(cfg, *a, **dict(k, cmd=%r)); "
                "sys.argv = ['monitor.py', '--config', %r, '--dry-run']; "
                "runpy.run_path(%r, run_name='__main__')") % (str(ROOT), SEGV, str(conf), str(ROOT / "monitor.py"))
        r = subprocess.run([PY, "-c", boot], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("crashed", r.stderr)
        self.assertIn("regex+tfidf", r.stdout)


class FakeTorch(types.ModuleType):
    def __init__(self, cuda_ok):
        super().__init__("torch")
        self.cuda = types.SimpleNamespace(is_available=lambda: cuda_ok)


class TestDevice(unittest.TestCase):
    def with_torch(self, mod):
        saved = sys.modules.get("torch", "absent")
        sys.modules["torch"] = mod
        self.addCleanup(lambda: sys.modules.pop("torch", None) if saved == "absent"
                        else sys.modules.__setitem__("torch", saved))

    def test_auto_picks_cuda_when_available(self):
        self.with_torch(FakeTorch(True))
        self.assertEqual(E.resolve_device("auto"), "cuda")
        self.assertEqual(E.resolve_device(None), "cuda")

    def test_auto_falls_back_to_cpu(self):
        self.with_torch(FakeTorch(False))
        self.assertEqual(E.resolve_device("auto"), "cpu")

    def test_explicit_device_passes_through(self):
        self.with_torch(FakeTorch(True))
        self.assertEqual(E.resolve_device("cpu"), "cpu")
        self.assertEqual(E.resolve_device("CUDA:1"), "cuda:1")

    def test_env_override(self):
        old = os.environ.get("A2179_DEVICE")
        os.environ["A2179_DEVICE"] = "cpu"
        try:
            self.assertEqual(load_config("does-not-exist.json")["embedder"]["device"], "cpu")
        finally:
            os.environ.pop("A2179_DEVICE") if old is None else os.environ.__setitem__("A2179_DEVICE", old)

    def test_device_and_backend_reach_sentence_transformers(self):
        seen = {}

        class ST(F.GoodST):
            def __init__(self, name, device=None, **kw):
                seen.update(device=device, **kw)
                super().__init__(name, device=device)
        self.with_torch(FakeTorch(False))
        with F.fake_module(ST):
            sim = E.EmbedSim("x", device="auto", backend="onnx")
        self.assertEqual((sim.device, seen["device"], seen["backend"]), ("cpu", "cpu", "onnx"))
        seen.clear()
        with F.fake_module(ST):
            E.EmbedSim("x", device="cuda")
        self.assertNotIn("backend", seen)  # default torch backend: nothing extra passed

    def test_old_sentence_transformers_without_backend(self):
        class Old(F.GoodST):
            def __init__(self, name, device=None):
                super().__init__(name, device=device)
        with F.fake_module(Old):
            with self.assertRaises(E.EmbedderUnavailable) as cm:
                E.EmbedSim("x", device="cpu", backend="onnx")
        self.assertIn("3.2", cm.exception.fix)


class _OrderFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves fake torch and sentence_transformers and records the import order."""
    def __init__(self):
        self.order = []

    def find_spec(self, name, path=None, target=None):
        if name in ("torch", "sentence_transformers"):
            return importlib.machinery.ModuleSpec(name, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.order.append(module.__name__)
        if module.__name__ == "torch":
            module.cuda = types.SimpleNamespace(is_available=lambda: False)
        else:
            module.SentenceTransformer = F.GoodST


class TestImportOrder(unittest.TestCase):
    def test_torch_is_imported_before_sentence_transformers(self):
        saved = {k: sys.modules.pop(k) for k in ("torch", "sentence_transformers") if k in sys.modules}
        f = _OrderFinder()
        sys.meta_path.insert(0, f)
        try:
            sim = E.EmbedSim("x")
        finally:
            sys.meta_path.remove(f)
            for k in ("torch", "sentence_transformers"):
                sys.modules.pop(k, None)
            sys.modules.update(saved)
        self.assertEqual(f.order, ["torch", "sentence_transformers"])
        self.assertEqual(sim.device, "cpu")


if __name__ == "__main__":
    unittest.main()
