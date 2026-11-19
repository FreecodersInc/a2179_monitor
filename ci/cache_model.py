"""Put embedder models into the Hugging Face cache (HF_HOME), downloading only what is missing.
Run by ci/setup_venv.sh, the only CI step that may download; everything else runs with
HF_HUB_OFFLINE=1. Uses huggingface_hub.snapshot_download, so torch is not imported.

Which models: command line ids, else A2179_CI_MODELS (comma separated ids or short names of
eval/embedders.json, or "all"), else A2179_CI_MODEL, else the model of the config.
Only what sentence-transformers needs is fetched (configs, tokenizer, one weight file:
model.safetensors, or pytorch_model.bin where a repo has no safetensors), not the ONNX,
OpenVINO, TF and Flax copies some repos also carry (multilingual-e5-large: ~2.2 GB instead of ~9.6).

By hand, e.g. on Leonid's machine:
  python ci/cache_model.py intfloat/multilingual-e5-base BAAI/bge-m3
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATTERNS = ["*.json", "*.txt", "*.model", "tokenizer*", "sentencepiece*", "1_Pooling/*", "2_Dense/*",
                 "modules.json"]
NO_SAFETENSORS = {"BAAI/bge-m3"}   # checked 2026-11-19: only pytorch_model.bin in the repo root


def registry():
    try:
        with open(os.path.join(ROOT, "eval", "embedders.json"), encoding="utf-8") as f:
            return json.load(f)["models"]
    except (OSError, ValueError, KeyError):
        return []


def model_names(argv=None):
    argv = list(argv or [])
    if argv:
        return argv
    spec = os.environ.get("A2179_CI_MODELS", "").strip()
    if spec:
        reg = registry()
        if spec == "all":
            return list(dict.fromkeys(m["id"] for m in reg))
        short = {m["short"]: m["id"] for m in reg}
        return list(dict.fromkeys(short.get(x.strip(), x.strip()) for x in spec.split(",") if x.strip()))
    if os.environ.get("A2179_CI_MODEL"):
        return [os.environ["A2179_CI_MODEL"]]
    return [model_name()]


def model_name():
    if os.environ.get("A2179_CI_MODEL"):
        return os.environ["A2179_CI_MODEL"]
    sys.path.insert(0, ROOT)
    from a2179.config import DEFAULTS
    return DEFAULTS["embedder"]["model"]


def patterns(name):
    return BASE_PATTERNS + (["pytorch_model.bin"] if name in NO_SAFETENSORS else ["model.safetensors"])


def cache_one(snapshot_download, name):
    try:
        p = snapshot_download(name, local_files_only=True, allow_patterns=patterns(name))
        print(f"cache_model: {name} already cached in {p}")
        return 0
    except Exception:
        pass
    if os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "yes", "on"):
        print(f"cache_model: {name} missing and HF_HUB_OFFLINE is set, cannot download")
        return 4
    t0 = time.time()
    p = snapshot_download(name, allow_patterns=patterns(name))
    print(f"cache_model: downloaded {name} into {p} in {time.time() - t0:.0f} s")
    return 0


def main(argv=None):
    names = model_names(argv)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("cache_model: huggingface_hub not installed (requirements-embed.txt), skip")
        return 0
    rc = 0
    for name in names:
        try:
            rc = max(rc, cache_one(snapshot_download, name))
        except Exception as e:  # one model failing (gated, typo, network) does not stop the others
            print(f"cache_model: {name} failed: {type(e).__name__}: {str(e)[:300]}")
            rc = max(rc, 5)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
