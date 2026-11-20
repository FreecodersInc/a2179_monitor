#!/usr/bin/env python3
"""EXP-027: Hub metadata size probe (no weight download) and a GPU check.

Needed files = primary weight in the repo root (model.safetensors, else
pytorch_model.bin) plus tokenizer/config files, same idea as ci/cache_model.py.
Cap is 50 MiB. Network: curl with a 2 MiB body cap and a 30 s timeout.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAP_MIB = 50
CAP = CAP_MIB * 1024 * 1024
API_BODY_CAP = 2 * 1024 * 1024
TIMEOUT = 30

# Primary weight plus the tokenizer/config files sentence-transformers typically needs.
NEEDED_EXTRA = (
    "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.txt", "vocab.json", "merges.txt", "sentencepiece.bpe.model", "spiece.model",
    "modules.json", "1_Pooling/config.json", "2_Dense/config.json", "2_Dense/model.safetensors",
    "sentence_bert_config.json", "config_sentence_transformers.json",
)

MODELS = [
    # cross-encoders / rerankers / NLI (H-04, H-13)
    "cross-encoder/ms-marco-TinyBERT-L2-v2",
    "cross-encoder/ms-marco-TinyBERT-L4-v2",
    "cross-encoder/ms-marco-MiniLM-L-2-v2",
    "cross-encoder/ms-marco-MiniLM-L-4-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross-encoder/stsb-TinyBERT-L4",
    "cross-encoder/nli-MiniLM2-L6-H768",
    "cross-encoder/nli-distilroberta-base",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "cross-encoder/mmarco-mMiniLMv2-L6-H384-v1",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-v2-m3",
    "jinaai/jina-reranker-v1-tiny-en",
    "jinaai/jina-reranker-v1-turbo-en",
    "jinaai/jina-reranker-v2-base-multilingual",
    "mixedbread-ai/mxbai-rerank-xsmall-v1",
    "mixedbread-ai/mxbai-rerank-base-v1",
    "Alibaba-NLP/gte-multilingual-reranker-base",
    "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "typeform/distilbert-base-uncased-mnli",
    # ColBERT (H-23)
    "colbert-ir/colbertv2.0",
    "answerdotai/answerai-colbert-small-v1",
    "lightonai/GTE-ModernColBERT-v1",
    # learned sparse (H-19); efficient SPLADE is two towers
    "naver/splade-cocondenser-ensembledistil",
    "naver/splade-v3",
    "naver/splade-v3-distilbert",
    "naver/splade-v3-lexical",
    "naver/splade_v2_distil",
    "naver/efficient-splade-VI-BT-large-query",
    "naver/efficient-splade-VI-BT-large-doc",
    # dense: monitor zoo + published tiny variants
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-large-instruct",
    "BAAI/bge-m3",
    "sentence-transformers/LaBSE",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-albert-small-v2",
    "TaylorAI/bge-micro-v2",
    "TaylorAI/gte-tiny",
    "thenlper/gte-small",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
    "cointegrated/rubert-tiny2",
    "nreimers/mMiniLMv2-L6-H384",
    "microsoft/Multilingual-MiniLM-L12-H384",
]


def mib(n):
    if n is None:
        return None
    return round(n / (1024 * 1024), 1)


def probe_gpu():
    info = {
        "nvidia_smi": None,
        "nvidia_smi_ok": False,
        "torch_version": None,
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "note": "",
    }
    smi = shutil.which("nvidia-smi")
    if smi:
        r = subprocess.run([smi], capture_output=True, text=True, timeout=20)
        info["nvidia_smi_ok"] = r.returncode == 0
        info["nvidia_smi"] = (r.stdout or r.stderr)[:2000]
    else:
        info["note"] = "nvidia-smi not found"
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_count"] = int(torch.cuda.device_count())
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001  torch may be missing
        if info["note"]:
            info["note"] += f"; torch not importable: {type(e).__name__}: {e}"
        else:
            info["note"] = f"torch not importable: {type(e).__name__}: {e}"
    return info


def disk_ram():
    du = shutil.disk_usage(str(ROOT))
    ram = {}
    try:
        txt = Path("/proc/meminfo").read_text()
        for line in txt.splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:", "MemFree:")):
                k, v, *_ = line.replace(":", "").split()
                ram[k] = int(v) * 1024
    except OSError:
        pass
    return {
        "disk_free_mib": round(du.free / (1024 * 1024), 1),
        "disk_total_mib": round(du.total / (1024 * 1024), 1),
        "mem_total_mib": mib(ram.get("MemTotal")),
        "mem_available_mib": mib(ram.get("MemAvailable")),
    }


def hub_info(model_id: str):
    url = f"https://huggingface.co/api/models/{model_id}?blobs=true"
    out = Path(f"/tmp/a2179_hf_{model_id.replace('/', '__')}.json")
    cmd = [
        "curl", "-sS", "-L", "--max-time", str(TIMEOUT),
        "--max-filesize", str(API_BODY_CAP),
        "-A", "a2179-monitor-size-probe/1.0",
        "-o", str(out), "-w", "%{http_code} %{size_download}",
        url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT + 5)
    code = (r.stdout or "").strip()
    http = code.split()[0] if code else "?"
    if r.returncode != 0:
        return None, {"error": f"curl {r.returncode} {((r.stderr or '')[-200:])} {code}"}
    if http != "200":
        return None, {"error": f"http {http}"}
    try:
        data = json.loads(out.read_text())
    except Exception as e:  # noqa: BLE001
        return None, {"error": f"json: {e}"}
    finally:
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
    return data, None


def summarize(model_id: str, data: dict):
    files = []
    for s in data.get("siblings") or []:
        name = s.get("rfilename") or s.get("filename") or ""
        files.append((name, s.get("size")))
    primary = None
    for pref in ("model.safetensors", "pytorch_model.bin"):
        hit = [x for x in files if x[0] == pref]
        if hit:
            primary = hit[0]
            break
    if primary is None:
        root_w = [(n, sz) for n, sz in files if n.endswith((".safetensors", ".bin")) and "/" not in n]
        if root_w:
            primary = min(root_w, key=lambda x: x[1] if x[1] is not None else 10**18)
    wanted = set(NEEDED_EXTRA)
    if primary:
        wanted.add(primary[0])
    selected = [(n, sz) for n, sz in files if n in wanted]
    unknown = any(sz is None for _, sz in selected)
    total = None if unknown or not selected else sum(sz or 0 for _, sz in selected)
    fits = (total is not None) and (total <= CAP)
    return {
        "id": model_id,
        "gated": data.get("gated"),
        "library": data.get("library_name"),
        "primary": primary[0] if primary else None,
        "primary_bytes": primary[1] if primary else None,
        "primary_mib": mib(primary[1]) if primary else None,
        "needed_bytes": total,
        "needed_mib": mib(total),
        "fits_50_mib": fits,
        "files": [{"name": n, "bytes": sz, "mib": mib(sz)} for n, sz in selected],
    }


def probe_one(model_id: str):
    data, err = hub_info(model_id)
    if err:
        return {"id": model_id, "fits_50_mib": False, **err}
    return summarize(model_id, data)


def markdown_table(rows, gpu, res):
    lines = [
        "# EXP-027 inventory",
        "",
        f"Cap: {CAP_MIB} MiB. Sizes are Hub metadata (needed files), not a download.",
        "",
        "## Machine",
        "",
        f"- nvidia-smi: {'ok' if gpu.get('nvidia_smi_ok') else 'missing or failed'} ({gpu.get('note') or '—'})",
        f"- torch: {gpu.get('torch_version') or 'not importable'}, cuda_available={gpu.get('cuda_available')}, "
        f"device={gpu.get('device_name') or '—'}",
        f"- disk free: {res.get('disk_free_mib')} MiB, RAM available: {res.get('mem_available_mib')} MiB",
        "",
        "## Models",
        "",
        "| id | needed MiB | primary | fits 50 MiB | note |",
        "|---|---:|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| `{r['id']}` | – | – | no | {r['error']} |")
        else:
            flag = "yes" if r.get("fits_50_mib") else "no"
            lines.append(
                f"| `{r['id']}` | {r.get('needed_mib') if r.get('needed_mib') is not None else '–'} | "
                f"{r.get('primary') or '–'} ({r.get('primary_mib') if r.get('primary_mib') is not None else '–'} MiB) | "
                f"{flag} | gated={r.get('gated')} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    global CAP_MIB, CAP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "reports" / "exp027"))
    ap.add_argument("--cap-mib", type=float, default=CAP_MIB)
    args = ap.parse_args(argv)
    CAP_MIB = args.cap_mib
    CAP = int(CAP_MIB * 1024 * 1024)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu = probe_gpu()
    res = disk_ram()
    rows = [probe_one(m) for m in MODELS]
    payload = {"cap_mib": CAP_MIB, "gpu": gpu, "resources": res, "models": rows}
    (out / "inventory.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (out / "gpu.json").write_text(json.dumps(gpu, indent=2, ensure_ascii=False) + "\n")
    (out / "inventory.md").write_text(markdown_table(rows, gpu, res))
    fits = [r["id"] for r in rows if r.get("fits_50_mib")]
    errs = [r["id"] for r in rows if r.get("error")]
    print(f"model_size_probe: {len(rows)} models, {len(fits)} fit {CAP_MIB} MiB, {len(errs)} errors")
    for i in fits:
        rec = next(r for r in rows if r["id"] == i)
        print(f"  FIT  {rec.get('needed_mib')} MiB  {i}")
    print(f"wrote {out / 'inventory.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
