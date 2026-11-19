import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import ROOT

sys.path.insert(0, str(ROOT / "eval"))
import plots as P  # noqa: E402
import sweep as S  # noqa: E402

DATA = ROOT / "tests" / "fixtures" / "labeled_base.jsonl"
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class TestRegistry(unittest.TestCase):
    def test_entries_complete_and_unique(self):
        reg = S.registry()
        shorts = [m["short"] for m in reg]
        self.assertEqual(len(shorts), len(set(shorts)))
        for m in reg:
            for k in ("id", "proto_prefix", "text_prefix", "params_m", "weights_mb", "license", "mx130"):
                self.assertIn(k, m, m["short"])
        e5 = {m["short"]: m for m in reg}
        self.assertEqual(e5["e5-base"]["text_prefix"], "query: ")
        self.assertEqual(e5["e5-base-qp"]["proto_prefix"], "passage: ")
        self.assertTrue(e5["e5-large-instruct"]["text_prefix"].startswith("Instruct: "))
        self.assertNotIn("jinaai/jina-embeddings-v3", [m["id"] for m in reg])   # cc-by-nc
        en = {m["short"]: m for m in reg}
        self.assertEqual(en["e5-base-en"]["id"], "intfloat/e5-base-v2")
        self.assertEqual(en["e5-base-en"]["text_prefix"], "query: ")
        self.assertEqual(en["bge-base-en"]["id"], "BAAI/bge-base-en-v1.5")
        self.assertTrue(en["bge-base-en"]["text_prefix"].startswith("Represent this sentence"))
        self.assertEqual(en["bge-base-en"]["license"], "mit")
        self.assertFalse(en["bge-base-en"]["sr"])
        self.assertEqual(en["bge-base-en"]["weights_mb"], 438)
        import json
        groups = json.loads((ROOT / "eval" / "embedders.json").read_text())["groups"]
        self.assertEqual(set(groups), {"multilingual", "english"})

    def test_select(self):
        reg = S.registry()
        self.assertEqual(S.select_embedders("none", reg), [])
        got = S.select_embedders("e5-small,BAAI/bge-m3", reg, default_model=MINILM)
        self.assertEqual([m["short"] for m in got], ["e5-small", "bge-m3"])
        allm = S.select_embedders("all", reg, default_model=MINILM)
        self.assertNotIn("minilm", [m["short"] for m in allm])   # the default is the plain dense channel
        with self.assertRaises(SystemExit):
            S.select_embedders("nope", reg)

    def test_cached_looks_at_the_cache_folder_only(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            snap = tmp / "hub" / "models--intfloat--multilingual-e5-small" / "snapshots" / "abc"
            snap.mkdir(parents=True)
            with mock.patch.dict(os.environ, {"HF_HOME": str(tmp)}):
                os.environ.pop("HF_HUB_CACHE", None)
                self.assertTrue(S.hf_cached("intfloat/multilingual-e5-small"))
                self.assertFalse(S.hf_cached("BAAI/bge-m3"))
                self.assertEqual([m["short"] for m in S.select_embedders("cached", default_model=MINILM)],
                                 ["e5-small"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestPrefixes(unittest.TestCase):
    def test_embedsim_puts_prefixes_on_each_side(self):
        import numpy as np
        from a2179 import embedder as E
        seen = []

        class FakeST:
            def __init__(self, *a, **k):
                pass

            def encode(self, texts, **kw):
                seen.extend(texts)
                return np.ones((len(texts), 3)) / np.sqrt(3)
        fake_mod = mock.Mock(SentenceTransformer=FakeST)
        with mock.patch.object(E, "import_torch_first"), mock.patch.object(E.importlib, "import_module",
                                                                            return_value=fake_mod), \
                mock.patch.object(E, "resolve_device", return_value="cpu"), \
                mock.patch.object(E, "patch_cuda_capture_checks"), mock.patch.object(E, "cpu_guard", return_value={}):
            sim = E.EmbedSim("m", pos=["lcd"], neg=["futrola"], proto_prefix="passage: ", text_prefix="query: ")
            sim.score_many(["Ekran A2179"])
        self.assertEqual(seen, ["passage: lcd", "passage: futrola", "query: Ekran A2179"])


class TestOomFallback(unittest.TestCase):
    def test_runner_dense_cache_is_not_seen_by_tests(self):
        self.assertNotIn("A2179_DENSE_CACHE", os.environ)

    def test_big_model_on_small_gpu_reruns_on_cpu(self):
        from a2179 import embedder as E
        devices = []

        class Sim:
            def __init__(self, model, local_only=False, device="auto", **kw):
                devices.append(device)
                if device != "cpu":
                    raise E.EmbedderUnavailable("encode_failed", "CUDA out of memory. Tried to allocate 20 MiB", "x")
                self.device = device

            def score_many(self, texts):
                return [(0.5, 0.4)] * len(texts)
        tmp = Path(tempfile.mkdtemp())
        try:
            rows = [{"id": 1, "title": "a", "snippet": ""}, {"id": 2, "title": "b", "snippet": ""}]
            with mock.patch.object(E, "EmbedSim", Sim):
                S.compute_dense(rows, tmp / "d.jsonl", "BAAI/bge-m3")
            self.assertEqual(devices, ["auto", "cpu"])
            self.assertEqual(len((tmp / "d.jsonl").read_text().splitlines()), 2)
            devices.clear()

            class Bad(Sim):
                def __init__(self, *a, **k):
                    raise E.EmbedderUnavailable("wrong_model", "Repository Not Found", "x")
            with mock.patch.object(E, "EmbedSim", Bad), self.assertRaises(E.EmbedderUnavailable):
                S.compute_dense(rows, tmp / "e.jsonl", "nope/model")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestSweepSeveralEmbedders(unittest.TestCase):
    def test_one_run_compares_models(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
            # test only stand-in scores (not eval data): positives a bit higher, deterministic
            for short, gap in (("e5-small", 0.2), ("bge-m3", 0.05)):
                with (tmp / f"dense_scores_{short}.jsonl").open("w") as f:
                    for r in rows:
                        f.write(json.dumps({"id": r["id"], "pos": 0.5 + gap * r["label"] + (r["id"] % 7) / 100,
                                            "neg": 0.5}) + "\n")
            out = tmp / "s.json"
            with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
                S.main(["--data", str(DATA), "--dense-scores", str(tmp / "dense_scores.jsonl"), "--out", str(out),
                        "--embedders", "e5-small,bge-m3,e5-base"])
            res = json.loads(out.read_text())
            self.assertEqual(sorted(res["embedders"]), ["bge-m3", "e5-small"])
            self.assertIn("e5-base", res["embedders_skipped"])
            names = {v["variant"] for v in res["variants"]}
            self.assertIn("dense:e5-small (d=pos-neg)", names)
            self.assertTrue(any(n.startswith("dense:bge-m3+feat lam=") for n in names))
            md = out.with_suffix(".tables.md").read_text()
            self.assertIn("Dense models side by side", md)
            with mock.patch.object(P, "_plt", side_effect=ImportError("no matplotlib")):
                meta = P.build(out, DATA, tmp / "plots")
            self.assertIn("dense:e5-small hybrid", meta["dense_compare"])
            ci = meta["dense_compare"]["dense:e5-small pure"]["ci95"]["P@R1"]
            self.assertLessEqual(ci[0], meta["dense_compare"]["dense:e5-small pure"]["P@R1"] + 1e-9)
            self.assertLessEqual(ci[0], ci[1])
            self.assertIn("95% CI", (tmp / "plots" / "plots.md").read_text())
            # reproducible: same seed, same interval
            meta2 = P.build(out, DATA, tmp / "plots2")
            self.assertEqual(meta2["dense_compare"], meta["dense_compare"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
