"""Job mode of the monitor (T-036): fresh vacancies from the M-003 sources, ranked by closeness to the
CV (eval/cv_distance.py, variant B: stack terms from cv/master_cv.md) plus the profile features, and
only the top new ones are pinged. A ranking aid, not a judge: on fresh full-text labels the CV score has
AUC 0.656 (EXP-011).

score = 0.5 * cv_score + 0.5 * feat, feat = (features(title, snippet) + 1) / 2 of the de_jobs profile.
ADR-0036 (H-43, EXP-022): when the TinyBERT-L2 MS MARCO cross-encoder is available, the final score is the
mean of the normalised ranks of that base score and the cross-encoder score of the title (title only: the
full text scored worse). job_rerank: auto (use the model if it is already in the local HF cache, never
download), download (fetch it once, only if the files are under 50 MiB), off. Env A2179_JOB_RERANK wins.
Seen vacancies live in <data_dir>/jobs_seen.json (url -> first seen), --seed fills it silently.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

DEFAULT_SOURCES = ("remotive", "remoteok", "ats", "arbeitnow", "wwr", "hn")   # T-037: the last three
# ADR-0036: exactly the model, files and query tested in EXP-022 (eval/rerank_jobs.py)
RERANK_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"
RERANK_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
RERANK_QUERY = ("Data Engineer job: build data pipelines and a data platform with SQL, Python, Spark, Airflow, dbt, "
                "Kafka, Snowflake, Databricks on AWS")
RERANK_CAP_BYTES = 50 * 1024 * 1024


def load_reranker(mode: str = "auto"):
    """(callable titles -> scores, note) or (None, why). Never raises: no reranker = the old ranking."""
    import os
    mode = (os.environ.get("A2179_JOB_RERANK") or mode or "auto").strip().lower()
    if mode in ("off", "0", "no", "false"):
        return None, "rerank off"
    try:
        from huggingface_hub import snapshot_download
        if mode == "download":
            from huggingface_hub import HfApi
            sizes = {s.rfilename: s.size for s in HfApi().model_info(RERANK_MODEL, files_metadata=True).siblings}
            total = sum(sizes.get(f) or RERANK_CAP_BYTES for f in RERANK_FILES)
            if total > RERANK_CAP_BYTES:
                return None, f"rerank blocked: {total} bytes > {RERANK_CAP_BYTES}"
            path = snapshot_download(RERANK_MODEL, allow_patterns=RERANK_FILES)
        else:
            path = snapshot_download(RERANK_MODEL, allow_patterns=RERANK_FILES, local_files_only=True)
        from a2179.embedder import import_torch_first
        import_torch_first()
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(path, max_length=512, device="cpu")
    except Exception as e:  # noqa: BLE001  not installed, not cached, offline: keep the old ranking
        return None, f"rerank unavailable ({type(e).__name__}), base score only"

    def rerank(titles):
        return [float(x) for x in ce.predict([(RERANK_QUERY, t) for t in titles], batch_size=32, show_progress_bar=False)]
    return rerank, "rerank TinyBERT-L2 (title)"


def _norm_ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(o):
        r[i] = k / max(1, len(v) - 1)
    return r


def scorer(target="de_jobs", cv_path: Path = ROOT / "cv" / "master_cv.md"):
    import cv_distance as CV
    import sweep as S
    from a2179.target import load_target
    tg = load_target(target)
    terms = CV.cv_stack_terms(cv_path.read_text(encoding="utf-8")) if cv_path.is_file() else []
    if terms:
        stack = CV.stack_terms({"plus": terms})
    else:   # no CV in the checkout: variant A, the profile skills
        stack = CV.stack_terms(json.loads((ROOT / "targets" / f"{target}.json").read_text(encoding="utf-8")).get("skills", {}))
    frx = S.feature_rx(tg)

    def score(job) -> dict:
        p = CV.parts(job.title, job.text or job.snippet, tg, stack)
        feat = (S.features(job.title, job.snippet, frx)[0] + 1) / 2
        return {"cv": p["cv_score"], "feat": round(feat, 3), "score": round(0.5 * p["cv_score"] + 0.5 * feat, 4)}
    return score


def load_seen(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def rank(jobs, score, rerank=None) -> list[tuple]:
    out = [(j, score(j)) for j in jobs]
    if rerank is not None and len(out) > 1:
        ce = rerank([j.title for j, _ in out])
        rb, rc = _norm_ranks([s["score"] for _, s in out]), _norm_ranks(ce)
        for (_, s), c, a, b in zip(out, ce, rb, rc):
            s["base"], s["ce"], s["score"] = s["score"], round(c, 3), round(0.5 * a + 0.5 * b, 4)
    return sorted(out, key=lambda x: -x[1]["score"])


def message(job, s) -> str:
    where = f" · {job.location}" if job.location else ""
    return (f"💼 {job.title} · {job.company}{where}\n"
            f"близость к CV {s['cv']:.2f}, признаки {s['feat']:.2f}"
            + (f", реранк {s['ce']:.2f}" if "ce" in s else "") + f", итог {s['score']:.2f}\n{job.url}")


def run_jobs(cfg, data: Path, alerter, dry_run=False, seed=False, top=10, min_score=0.0,
             sources=DEFAULT_SOURCES, fetch="auto", collect=None, score=None, now=None, rerank="cfg") -> dict:
    from a2179.jobs import collect_jobs
    collect = collect or (lambda: collect_jobs(data / "cache" / "jobs", sources=tuple(sources), fetch=fetch))
    jobs, notes = collect()
    score = score or scorer(cfg.get("job_target", "de_jobs"))
    note = "rerank given"
    if rerank == "cfg":
        rerank, note = load_reranker(cfg.get("job_rerank", "auto"))
    ranked = rank(jobs, score, rerank)
    seen_path = data / "jobs_seen.json"
    seen = load_seen(seen_path)
    new = [(j, s) for j, s in ranked if j.url not in seen]
    picked = [] if seed else [(j, s) for j, s in new if s["score"] >= min_score][:top]
    msgs = [message(j, s) for j, s in picked]
    sent = alerter.send(msgs) if msgs else {}
    if not dry_run:
        stamp = (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for j, _ in (new if seed else picked):
            seen[j.url] = stamp
        seen_path.parent.mkdir(parents=True, exist_ok=True)
        seen_path.write_text(json.dumps(seen, indent=0, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"jobs": len(jobs), "new": len(new), "ranked": ranked, "picked": picked, "sent": sent, "notes": notes,
            "rerank": note if rerank is not None else (note if note != "rerank given" else "rerank off")}
