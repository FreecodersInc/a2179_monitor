#!/usr/bin/env python3
"""EXP-028 on the runner: H-04, H-13, H-19, H-23, CPU only, one model at a time.

Downloads and inference happen only on the runner. --dry-run writes the plan and
never loads weights (safe to run anywhere). Results are metrics only (ADR-0032).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

REQUEST_ID = "runner-exp-2026-11-20-v2"
NLI_HYPOTHESIS = ("This listing sells a MacBook Air A2179 LCD screen. "
                  "Ovaj oglas prodaje LCD ekran za MacBook Air A2179.")
SPECS = [
    {"hyp": "H-04", "id": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli", "kind": "nli", "needed_mib": 429.3},
    {"hyp": "H-13", "id": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "kind": "ce", "needed_mib": 469.9},
    {"hyp": "H-19", "id": "naver/splade-v3-distilbert", "kind": "splade", "needed_mib": 256.5},
    {"hyp": "H-23", "id": "answerdotai/answerai-colbert-small-v1", "kind": "colbert", "needed_mib": 128.3},
]


def resources():
    du = shutil.disk_usage(str(ROOT))
    ram = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:", "MemFree:")):
                k, v, *_ = line.replace(":", "").split()
                ram[k] = int(v) // 1024  # MiB
    except OSError:
        pass
    return {"disk_free_mib": round(du.free / (1024 * 1024), 1),
            "mem_available_mib": ram.get("MemAvailable"),
            "mem_total_mib": ram.get("MemTotal")}


def can_load(needed_mib, res):
    disk = res.get("disk_free_mib") or 0
    ram = res.get("mem_available_mib")
    if disk < needed_mib + 2048:
        return False, f"disk free {disk} MiB < {needed_mib} + 2048"
    if ram is not None and ram < 2 * needed_mib + 1024:
        return False, f"RAM available {ram} MiB < 2×{needed_mib} + 1024"
    return True, ""


def real_rows():
    rows = [json.loads(l) for l in (ROOT / "eval" / "labeled.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if not str(r.get("source") or "").startswith("synth")]


def texts(rows):
    return [f"{r.get('title') or ''} {r.get('snippet') or ''}".strip() for r in rows]


def labels(rows):
    return [int(r.get("label") == 1) for r in rows]


def ranking(scores, y):
    import plots as P
    order = sorted(range(len(y)), key=lambda i: (-scores[i], i))
    return P.ranking_metrics(y, order)


def prototypes():
    from a2179.target import load_target
    tg = load_target("a2179_screen")
    return list(tg.positive), list(tg.negative)


def baselines(rows):
    from a2179.similarity import TfidfSim
    from a2179.prototypes import POSITIVE, NEGATIVE
    import sweep as S
    pos, neg = list(POSITIVE), list(NEGATIVE)
    try:
        p, n = prototypes()
        pos, neg = p, n
    except Exception:
        pass
    ts, y = texts(rows), labels(rows)
    bm = S.BM25(pos, neg)
    tf = TfidfSim(pos, neg)
    out = {}
    for name, fn in (("bm25", lambda t: bm.score(t)), ("tfidf", lambda t: tf.score(t))):
        d = [float(fn(t)[0] - fn(t)[1]) for t in ts]
        out[name] = {"scores_kind": "sim_pos-sim_neg", **ranking(d, y)}
    return out


def force_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["A2179_DEVICE"] = "cpu"


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def drop_hf_repo(model_id):
    """Delete only this repo from the HF cache. Other cached models stay."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        hashes = [rev.commit_hash for repo in info.repos if repo.repo_id == model_id for rev in repo.revisions]
        if hashes:
            info.delete_revisions(*hashes).execute()
            return f"deleted {len(hashes)} revision(s)"
        return "not in cache"
    except Exception as e:  # noqa: BLE001
        return f"cache cleanup failed: {type(e).__name__}: {e}"[:240]


def score_nli(model_id, ts):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    mdl.to("cpu")
    mdl.eval()
    id2 = {v: k for k, v in mdl.config.id2label.items()} if hasattr(mdl.config, "id2label") else {}
    # find entailment index
    ent = None
    for i, name in (id2.items() if id2 else []):
        if "entail" in str(name).lower():
            ent = int(i)
            break
    if ent is None:
        ent = 2 if mdl.config.num_labels > 2 else 0
    out = []
    with torch.no_grad():
        for t in ts:
            enc = tok(t, NLI_HYPOTHESIS, return_tensors="pt", truncation=True, max_length=256)
            logits = mdl(**enc).logits[0]
            prob = torch.softmax(logits.float(), dim=-1)[ent].item()
            out.append(float(prob))
    del mdl, tok
    return out


def score_ce(model_id, ts, pos, neg):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_id, max_length=256, device="cpu")
    out = []
    for t in ts:
        sp = max(float(x) for x in ce.predict([(p, t) for p in pos], batch_size=8, show_progress_bar=False))
        sn = max(float(x) for x in ce.predict([(n, t) for n in neg], batch_size=8, show_progress_bar=False))
        out.append(sp - sn)
    del ce
    return out


def score_splade(model_id, ts, pos, neg):
    from sentence_transformers import SparseEncoder
    enc = SparseEncoder(model_id, device="cpu")
    q = enc.encode(ts, batch_size=8, show_progress_bar=False)
    p = enc.encode(pos, batch_size=8, show_progress_bar=False)
    n = enc.encode(neg, batch_size=8, show_progress_bar=False)
    # similarity matrix
    try:
        sp = enc.similarity(q, p)
        sn = enc.similarity(q, n)
        import torch
        if hasattr(sp, "detach"):
            sp, sn = sp.detach().cpu(), sn.detach().cpu()
        out = [(float(sp[i].max()) - float(sn[i].max())) for i in range(len(ts))]
    except Exception:
        # fallback: dense-like if similarity missing
        raise
    del enc
    return out


def score_colbert(model_id, ts, pos, neg):
    # Prefer PyLate if present, else SentenceTransformer token MaxSim.
    try:
        from pylate import models, scores as SC
        mdl = models.ColBERT(model_id, device="cpu")
        q = mdl.encode(ts, is_query=True, batch_size=4, show_progress_bar=False)
        p = mdl.encode(pos, is_query=False, batch_size=4, show_progress_bar=False)
        n = mdl.encode(neg, is_query=False, batch_size=4, show_progress_bar=False)
        sp, sn = SC.colbert_scores(q, p), SC.colbert_scores(q, n)
        out = [(float(sp[i].max()) - float(sn[i].max())) for i in range(len(ts))]
        del mdl
        return out
    except Exception:
        pass
    from sentence_transformers import SentenceTransformer
    import torch
    import torch.nn.functional as F
    mdl = SentenceTransformer(model_id, device="cpu")

    def tok_emb(texts):
        # token embeddings if the model supports them, else sentence embeddings as 1-token
        if hasattr(mdl, "encode") and "output_value" in mdl.encode.__code__.co_varnames:
            e = mdl.encode(texts, batch_size=4, show_progress_bar=False, output_value="token_embeddings",
                           convert_to_tensor=True)
            if isinstance(e, list):
                return [F.normalize(x.float(), dim=-1) for x in e]
            return F.normalize(e.float(), dim=-1)
        e = mdl.encode(texts, batch_size=4, show_progress_bar=False, convert_to_tensor=True)
        return F.normalize(e.float(), dim=-1).unsqueeze(1)

    def maxsim(a, b):
        # a: list/tensor of (tokens, dim), b: list of (tokens, dim)
        scores = []
        bstack = b if isinstance(b, list) else [b[i] for i in range(b.shape[0])]
        ast = a if isinstance(a, list) else [a[i] for i in range(a.shape[0])]
        for qa in ast:
            if qa.dim() == 1:
                qa = qa.unsqueeze(0)
            best = None
            for pb in bstack:
                if pb.dim() == 1:
                    pb = pb.unsqueeze(0)
                sim = qa @ pb.T
                s = float(sim.max(dim=1).values.sum())
                best = s if best is None else max(best, s)
            scores.append(best)
        return scores

    qt, pt, nt = tok_emb(ts), tok_emb(pos), tok_emb(neg)
    sp, sn = maxsim(qt, pt), maxsim(qt, nt)
    del mdl
    return [a - b for a, b in zip(sp, sn)]


SCORERS = {"nli": lambda spec, ts, pos, neg: score_nli(spec["id"], ts),
           "ce": lambda spec, ts, pos, neg: score_ce(spec["id"], ts, pos, neg),
           "splade": lambda spec, ts, pos, neg: score_splade(spec["id"], ts, pos, neg),
           "colbert": lambda spec, ts, pos, neg: score_colbert(spec["id"], ts, pos, neg)}


def run_one(spec, ts, y, pos, neg, dry=False):
    rec = {"hyp": spec["hyp"], "id": spec["id"], "kind": spec["kind"], "needed_mib": spec["needed_mib"],
           "status": "planned"}
    res = resources()
    rec["resources_before"] = res
    ok, why = can_load(spec["needed_mib"], res)
    if not ok:
        rec["status"] = "skipped"
        rec["reason"] = why
        return rec
    if dry:
        rec["status"] = "dry-run"
        return rec
    t0 = time.time()
    try:
        scores = SCORERS[spec["kind"]](spec, ts, pos, neg)
        rec["metrics"] = ranking(scores, y)
        rec["status"] = "done"
    except Exception as e:  # noqa: BLE001  missing lib, OOM, bad checkpoint: record, continue
        rec["status"] = "error"
        rec["reason"] = f"{type(e).__name__}: {e}"[:500]
    finally:
        rec["duration_s"] = round(time.time() - t0, 1)
        free_memory()
        rec["cache_cleanup"] = drop_hf_repo(spec["id"])
        rec["resources_after"] = resources()
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.device != "cpu":
        print("exp028: forcing CPU, ignoring --device", a.device, file=sys.stderr)
    force_cpu()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = real_rows()
    ts, y = texts(rows), labels(rows)
    pos, neg = prototypes()
    payload = {
        "id": REQUEST_ID,
        "exp": "EXP-028",
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dry_run": bool(a.dry_run),
        "n": len(rows),
        "pos": sum(y),
        "nli_hypothesis": NLI_HYPOTHESIS,
        "resources": resources(),
        "baselines": None,
        "models": [],
    }
    if not a.dry_run:
        payload["baselines"] = baselines(rows)
    else:
        payload["baselines"] = "skipped in --dry-run"
    for spec in SPECS:
        rec = run_one(spec, ts, y, pos, neg, dry=a.dry_run)
        payload["models"].append(rec)
        print(f"exp028: {rec['hyp']} {rec['id']} -> {rec['status']}", flush=True)
    (out / "exp028.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # short markdown, metrics only
    lines = [f"# EXP-028 {REQUEST_ID}", "", f"rows {payload['n']} (pos {payload['pos']}), device=cpu, dry_run={a.dry_run}", ""]
    if isinstance(payload["baselines"], dict):
        lines += ["## Baselines", "", "| channel | P@R1 | P@R95 | nDCG@npos |", "|---|---:|---:|---:|"]
        for k, m in payload["baselines"].items():
            lines.append(f"| {k} | {m.get('P@R1', '–')} | {m.get('P@R95', '–')} | {m.get('nDCG@npos', '–')} |")
        lines.append("")
    lines += ["## Models", "", "| hyp | model | status | P@R1 | P@R95 | note |", "|---|---|---|---:|---:|---|"]
    for rec in payload["models"]:
        m = rec.get("metrics") or {}
        lines.append(f"| {rec['hyp']} | `{rec['id']}` | {rec['status']} | {m.get('P@R1', '–')} | {m.get('P@R95', '–')} | "
                     f"{(rec.get('reason') or '')[:80]} |")
    (out / "exp028.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'exp028.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
