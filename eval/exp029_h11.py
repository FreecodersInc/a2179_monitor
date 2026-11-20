#!/usr/bin/env python3
"""EXP-029 / H-11 on the runner: compare MiniLM cosine scores on CPU vs CUDA.

--dry-run only reports whether CUDA is visible and does not load weights.
Texts of listings are never written to the artifact (ADR-0032).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

REQUEST_ID = "runner-exp-2026-11-20-v2"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ACCEPT = 1e-4
REFUTE = 1e-3


def real_rows():
    rows = [json.loads(l) for l in (ROOT / "eval" / "labeled.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if not str(r.get("source") or "").startswith("synth")]


def texts(rows):
    return [f"{r.get('title') or ''} {r.get('snippet') or ''}".strip() for r in rows]


def cuda_info():
    info = {"torch": None, "cuda_available": False, "device_count": 0, "device_name": None, "cuda_version": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_count"] = int(torch.cuda.device_count())
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def encode_scores(model, texts_, pos, neg, device, batch=1):
    """Return list of dicts {sim_pos, sim_neg, d} using cosine to prototype embeddings."""
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    mdl = SentenceTransformer(model, device=device)
    with torch.no_grad():
        pe = F.normalize(mdl.encode(pos, batch_size=batch, convert_to_tensor=True, show_progress_bar=False).float(), dim=-1)
        ne = F.normalize(mdl.encode(neg, batch_size=batch, convert_to_tensor=True, show_progress_bar=False).float(), dim=-1)
        te = F.normalize(mdl.encode(texts_, batch_size=batch, convert_to_tensor=True, show_progress_bar=False).float(), dim=-1)
        sp = (te @ pe.T).max(dim=1).values.cpu()
        sn = (te @ ne.T).max(dim=1).values.cpu()
    del mdl, pe, ne, te
    gc.collect()
    try:
        import torch as T
        if T.cuda.is_available():
            T.cuda.empty_cache()
    except Exception:
        pass
    return [{"sim_pos": float(sp[i]), "sim_neg": float(sn[i]), "d": float(sp[i] - sn[i])} for i in range(len(texts_))]


def verdict(max_abs):
    if max_abs is None:
        return "blocked"
    if max_abs < ACCEPT:
        return "accepted"
    if max_abs >= REFUTE:
        return "refuted"
    return "inconclusive"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = cuda_info()
    payload = {
        "id": REQUEST_ID, "exp": "EXP-029", "hyp": "H-11", "model": a.model,
        "dry_run": bool(a.dry_run), "cuda": info, "accept_lt": ACCEPT, "refute_ge": REFUTE,
        "status": "planned", "verdict_rule": "accepted if max|dcos|<1e-4, refuted if >=1e-3",
    }
    if not info.get("cuda_available"):
        payload["status"] = "blocked"
        payload["reason"] = "no CUDA on this runner"
        payload["suggested_verdict"] = "blocked"
        (out / "exp029_h11.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "exp029_h11.md").write_text(
            f"# EXP-029 H-11\n\nblocked: no CUDA. torch={info.get('torch')}, "
            f"error={info.get('error') or 'cuda_available=false'}\n", encoding="utf-8")
        print("exp029: blocked, no CUDA", flush=True)
        return 0
    if a.dry_run:
        payload["status"] = "dry-run"
        payload["device_name"] = info.get("device_name")
        (out / "exp029_h11.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("exp029: dry-run, CUDA visible:", info.get("device_name"), flush=True)
        return 0

    from a2179.target import load_target
    tg = load_target("a2179_screen")
    pos, neg = list(tg.positive), list(tg.negative)
    rows = real_rows()
    ts = texts(rows)
    payload["n"] = len(rows)
    t0 = time.time()
    try:
        cpu = encode_scores(a.model, ts, pos, neg, "cpu", batch=1)
        gpu = encode_scores(a.model, ts, pos, neg, "cuda", batch=1)
        diffs = []
        max_abs = 0.0
        worst = None
        for i, (c, g) in enumerate(zip(cpu, gpu)):
            rec = {k: abs(c[k] - g[k]) for k in ("sim_pos", "sim_neg", "d")}
            rec["i"] = i
            m = max(rec["sim_pos"], rec["sim_neg"], rec["d"])
            if m >= max_abs:
                max_abs, worst = m, {"row_index": i, **rec}
            diffs.append(m)
        payload["max_abs_dcos"] = max_abs
        payload["mean_abs_dcos"] = sum(diffs) / len(diffs) if diffs else None
        payload["n_ge_1e-4"] = sum(1 for x in diffs if x >= ACCEPT)
        payload["n_ge_1e-3"] = sum(1 for x in diffs if x >= REFUTE)
        payload["worst"] = worst  # index and deltas only, no text
        payload["suggested_verdict"] = verdict(max_abs)
        payload["status"] = "done"
        payload["cuda_device"] = info.get("device_name")
    except Exception as e:  # noqa: BLE001  OOM or load failure: clean record, no traceback texts of listings
        payload["status"] = "error"
        payload["reason"] = f"{type(e).__name__}: {e}"[:500]
        payload["suggested_verdict"] = "blocked"
    finally:
        payload["duration_s"] = round(time.time() - t0, 1)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    (out / "exp029_h11.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# EXP-029 H-11", "",
             f"status {payload['status']}, suggested {payload.get('suggested_verdict')}, "
             f"max |dcos| {payload.get('max_abs_dcos')}, device {payload.get('cuda_device') or info.get('device_name')}",
             ""]
    if payload.get("reason"):
        lines.append(f"reason: {payload['reason']}")
    (out / "exp029_h11.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"exp029: {payload['status']} verdict={payload.get('suggested_verdict')} max={payload.get('max_abs_dcos')}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
