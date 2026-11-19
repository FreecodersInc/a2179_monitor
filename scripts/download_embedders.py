#!/usr/bin/env python3
"""Prefetch embedders into the Hugging Face cache (HF_HOME).

Thin entrypoint around ci/cache_model.py: only the files sentence-transformers
needs, already cached models are skipped. Pass Hugging Face ids, short names
from eval/embedders.json, or registry groups (multilingual, english, all).

  python scripts/download_embedders.py multilingual english
  python scripts/download_embedders.py intfloat/e5-base-v2 BAAI/bge-base-en-v1.5
  python scripts/download_embedders.py --dry-run all

On WSL2, from the repo, with the project venv:

  cd /home/vessyl-runner/a2179_monitor && .venv/bin/python scripts/download_embedders.py multilingual english
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ci"))
import cache_model  # noqa: E402


if __name__ == "__main__":
    sys.exit(cache_model.main(sys.argv[1:]))
