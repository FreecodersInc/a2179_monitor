#!/usr/bin/env python3
"""EXP-022 / H-43 on the runner: zero-shot MS MARCO cross-encoder (TinyBERT-L2) on the labeled vacancies.

The model is fetched once into the HF cache, only after the summed size of the needed files (Hub
metadata) is checked against a byte cap (default 50 MiB, PROJECTS.md); over the cap: blocked, nothing
downloaded. Vacancy texts come from the runner's own jobs cache (never committed, ADR-0032).
Scores: AUC (E or S vs C or I, esci_text), Spearman by grade, paired bootstrap against cv_score A on
the same rows, and the mean of normalised ranks of both."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
QUERY = ("Data Engineer job: build data pipelines and a data platform with SQL, Python, Spark, Airflow, dbt, "
         "Kafka, Snowflake, Databricks on AWS")
POS = {"E", "S"}
GRADE = {"E": 3, "S": 2, "C": 1, "I": 0}
MIN_TEXT_ROWS = 100


def check_size(sizes: dict, cap_bytes: int):
    """sizes: file -> bytes (None = unknown). Returns (ok, total, why)."""
    missing = [f for f in FILES if f not in sizes]
    if missing:
        return False, None, f"files missing on the Hub: {missing}"
    if any(sizes[f] is None for f in FILES):
        return False, None, "size unknown for some files, not downloading"
    total = sum(sizes[f] for f in FILES)
    if total > cap_bytes:
        return False, total, f"{total} bytes > cap {cap_bytes}"
    return True, total, ""


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(o):
        r[i] = k / max(1, len(v) - 1)
    return r


def evaluate(scores: dict, labels, reps=1000, seed=0):
    import cv_distance as CV
    y = [l in POS for l in labels]
    g = [GRADE[l] for l in labels]
    out = {"n": len(y), "pos": sum(y)}
    for k, s in scores.items():
        out[k] = {"auc": round(CV.auc(s, y), 3), "spearman": round(CV.spearman(s, g), 3)}
    if "cv_score" in scores:
        rnd, n = random.Random(seed), len(y)
        for k in [k for k in scores if k != "cv_score"]:
            d = []
            for _ in range(reps):
                ix = [rnd.randrange(n) for _ in range(n)]
                yb = [y[i] for i in ix]
                if 0 < sum(yb) < n:
                    d.append(CV.auc([scores[k][i] for i in ix], yb) - CV.auc([scores["cv_score"][i] for i in ix], yb))
            d.sort()
            out[k]["minus_cv_score"] = round(out[k]["auc"] - out["cv_score"]["auc"], 3)
            out[k]["minus_cv_score_ci95"] = [round(d[int(0.025 * len(d))], 3), round(d[int(0.975 * len(d)) - 1], 3)]
    return out


def fetch_model(cap_mib: float):
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(MODEL, files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings}
    ok, total, why = check_size(sizes, int(cap_mib * 1024 * 1024))
    if not ok:
        return None, {"blocked": why, "total_bytes": total}
    p = snapshot_download(MODEL, allow_patterns=FILES)
    return p, {"total_bytes": total}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs-cache", default=os.environ.get("A2179_JOBS_CACHE") or str(Path.home() / ".cache" / "a2179-ci" / "jobs"))
    ap.add_argument("--cap-mib", type=float, default=50.0)
    a = ap.parse_args(argv)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    res = {"model": MODEL, "query": QUERY}
    path, meta = fetch_model(a.cap_mib)
    res.update(meta)
    if path is None:
        (o / "rerank_jobs.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(res))
        return 0
    import cv_distance as CV
    from a2179.target import load_target
    from sentence_transformers import CrossEncoder
    tg = load_target("de_jobs")
    stack = CV.stack_terms(json.loads((ROOT / "targets" / "de_jobs.json").read_text(encoding="utf-8")).get("skills", {}))
    texts = CV.job_texts(Path(a.jobs_cache))
    ce = CrossEncoder(path, max_length=512, device="cpu")
    rd = lambda p: [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]  # noqa: E731
    for name, file in (("labeled462", ROOT / "eval" / "de_jobs" / "labeled.jsonl"), ("batch3", ROOT / "eval" / "de_jobs" / "batch3.jsonl")):
        rows = [r for r in rd(file) if r.get("esci_text") in GRADE]
        part = {"rows": len(rows)}
        ce_t = [float(x) for x in ce.predict([(QUERY, r["title"]) for r in rows], batch_size=32, show_progress_bar=False)]
        cv_t = [CV.parts(r["title"], "", tg, stack)["cv_score"] for r in rows]
        part["title_only"] = evaluate({"cv_score": cv_t, "ce": ce_t,
                                       "fusion": [(x + z) / 2 for x, z in zip(ranks(ce_t), ranks(cv_t))]},
                                      [r["esci_text"] for r in rows])
        tr = [r for r in rows if texts.get(r["url"])]
        part["with_text"] = len(tr)
        if len(tr) >= MIN_TEXT_ROWS:
            ce_x = [float(x) for x in ce.predict([(QUERY, f"{r['title']}. {texts[r['url']]}") for r in tr],
                                                 batch_size=16, show_progress_bar=False)]
            cv_x = [CV.parts(r["title"], texts[r["url"]], tg, stack)["cv_score"] for r in tr]
            part["title_text"] = evaluate({"cv_score": cv_x, "ce": ce_x,
                                           "fusion": [(x + z) / 2 for x, z in zip(ranks(ce_x), ranks(cv_x))]},
                                          [r["esci_text"] for r in tr])
        else:
            part["title_text"] = {"not_run": f"{len(tr)} rows with text in the runner cache < {MIN_TEXT_ROWS}"}
        res[name] = part
    (o / "rerank_jobs.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
