"""Put embedder models into the Hugging Face cache (HF_HOME), downloading only what is missing.
Run by ci/setup_venv.sh, the only CI step that may download; everything else runs with
HF_HUB_OFFLINE=1. Uses huggingface_hub.snapshot_download, so torch is not imported.

Which models: command line ids, short names or registry groups from eval/embedders.json
(multilingual, english, all), else A2179_CI_MODELS (comma separated, same tokens, or "all"),
else A2179_CI_MODEL, else the model of the config.
Only what sentence-transformers needs is fetched (configs, tokenizer, one weight file:
model.safetensors, or pytorch_model.bin where a repo has no safetensors), not the ONNX,
OpenVINO, TF and Flax copies some repos also carry (multilingual-e5-large: ~2.2 GB instead of ~9.6).
Already cached models are skipped. --dry-run prints the plan and never downloads.

By hand, e.g. on Leonid's machine:
  python scripts/download_embedders.py multilingual english
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
HELP = """cache embedders into HF_HOME, only the files sentence-transformers needs.

Usage:
  python ci/cache_model.py [options] [id_or_short_or_group ...]
  python scripts/download_embedders.py multilingual english

Options:
  --dry-run   plan only: resolve names, report weights_mb, skip downloads
  -h, --help  this text

Tokens are Hugging Face ids, short names from eval/embedders.json, or groups
(multilingual, english, all). Already cached models are skipped.
"""


def registry_doc():
    try:
        with open(os.path.join(ROOT, "eval", "embedders.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, KeyError):
        return {}


def registry():
    return registry_doc().get("models") or []


def group_ids():
    doc = registry_doc()
    models = doc.get("models") or []
    short = {m["short"]: m["id"] for m in models}
    groups = {"all": list(dict.fromkeys(m["id"] for m in models))}
    for name, members in (doc.get("groups") or {}).items():
        groups[name] = list(dict.fromkeys(short.get(x, x) for x in members))
    return groups, short


def resolve_specs(specs):
    groups, short = group_ids()
    ids = []
    for raw in specs:
        x = (raw or "").strip()
        if not x:
            continue
        if x in groups:
            ids.extend(groups[x])
        else:
            ids.append(short.get(x, x))
    return list(dict.fromkeys(ids))


def parse_args(argv=None):
    argv = list(argv or [])
    dry, names = False, []
    for a in argv:
        if a in ("-h", "--help"):
            return "help", False, []
        if a in ("--dry-run", "--dry_run"):
            dry = True
        elif a.startswith("-"):
            print(f"cache_model: unknown option {a}")
            return "error", False, []
        else:
            names.append(a)
    return "ok", dry, names


def model_names(argv=None):
    argv = list(argv or [])
    if argv:
        return resolve_specs(argv)
    spec = os.environ.get("A2179_CI_MODELS", "").strip()
    if spec:
        return resolve_specs([x.strip() for x in spec.split(",") if x.strip()] if spec != "all" else ["all"])
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


def weights_mb(name):
    for m in registry():
        if m.get("id") == name and "weights_mb" in m:
            return int(m["weights_mb"])
    return None


def cache_one(snapshot_download, name, dry=False):
    try:
        p = snapshot_download(name, local_files_only=True, allow_patterns=patterns(name))
        print(f"cache_model: {name} already cached in {p}")
        return 0
    except Exception:
        pass
    mb = weights_mb(name)
    size = f" (~{mb} MB weights)" if mb is not None else ""
    if dry:
        print(f"cache_model: {name} missing{size}, dry-run so not downloading")
        return 0
    if os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "yes", "on"):
        print(f"cache_model: {name} missing and HF_HUB_OFFLINE is set, cannot download")
        return 4
    t0 = time.time()
    p = snapshot_download(name, allow_patterns=patterns(name))
    print(f"cache_model: downloaded {name} into {p} in {time.time() - t0:.0f} s")
    return 0


def main(argv=None):
    status, dry, names = parse_args(argv)
    if status == "help":
        print(HELP, end="")
        return 0
    if status == "error":
        return 2
    names = model_names(names)
    planned = sum(weights_mb(n) or 0 for n in names)
    print(f"cache_model: {len(names)} model(s), ~{planned} MB of weight files (tokenizers on top, already cached skipped)")
    if dry:
        print("cache_model: dry-run, will not download")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("cache_model: huggingface_hub not installed (requirements-embed.txt), skip")
        return 0
    rc = 0
    for name in names:
        try:
            rc = max(rc, cache_one(snapshot_download, name, dry=dry))
        except Exception as e:  # one model failing (gated, typo, network) does not stop the others
            print(f"cache_model: {name} failed: {type(e).__name__}: {str(e)[:300]}")
            rc = max(rc, 5)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
