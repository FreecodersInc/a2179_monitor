"""Put the embedder model into the Hugging Face cache (HF_HOME), downloading only if it is
missing. Run by ci/setup_venv.sh, the only CI step that may download; everything else runs
with HF_HUB_OFFLINE=1. Uses huggingface_hub.snapshot_download, so torch is not imported."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_name():
    if os.environ.get("A2179_CI_MODEL"):
        return os.environ["A2179_CI_MODEL"]
    sys.path.insert(0, ROOT)
    from a2179.config import DEFAULTS
    return DEFAULTS["embedder"]["model"]


def main():
    name = model_name()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("cache_model: huggingface_hub not installed (requirements-embed.txt), skip")
        return 0
    try:
        p = snapshot_download(name, local_files_only=True)
        print(f"cache_model: {name} already cached in {p}")
        return 0
    except Exception:
        pass
    if os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "yes", "on"):
        print("cache_model: model missing and HF_HUB_OFFLINE is set, cannot download")
        return 4
    t0 = time.time()
    p = snapshot_download(name)
    print(f"cache_model: downloaded {name} into {p} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
