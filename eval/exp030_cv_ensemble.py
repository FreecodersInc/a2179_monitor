"""EXP-030. Compare CV and vacancy titles on the runner and export reweightable rankings."""
import argparse
import csv
import gc
import hashlib
import os
import json
import sys
from pathlib import Path

from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv_distance as cv
import exp028_blocked_models as screens
import rerank_jobs
import sweep
from a2179.target import load_target


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cosine(model, query, titles):
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model["id"], device="cpu", local_files_only=True)
    prefix = model["text_prefix"]
    if model["short"] == "e5-large-instruct":
        encoder.float()
        prefix = "Instruct: Given a candidate resume, retrieve relevant job titles.\nQuery: "
    q = encoder.encode([prefix + query], normalize_embeddings=True)[0]
    documents = encoder.encode([model["proto_prefix"] + t for t in titles], batch_size=16,
                               normalize_embeddings=True, show_progress_bar=False)
    return documents @ q


def rank_table(rows, raw, weights, out):
    idx = [r["title"] for r in rows]
    matrix = {}
    for channel, scores in raw.items():
        vals = [scores[t] for t in idx]
        ranks = (rankdata(vals) - 1) / max(1, len(vals) - 1)
        for r, v, rank in zip(rows, vals, ranks):
            r["raw_" + channel] = v
            r["rank_" + channel] = float(rank)
        matrix[channel] = ranks
    ensemble = sum(weights[k] * v for k, v in matrix.items())
    for r, s in zip(rows, ensemble):
        r["ensemble"] = float(s)
    write_csv(out, sorted(rows, key=lambda r: -r["ensemble"]))
    if "esci_text" in rows[0]:
        y = [r["esci_text"] in rerank_jobs.POS for r in rows]
        return {k: cv.auc(list(map(float, v)), y) for k, v in {**matrix, "ensemble": ensemble}.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--review", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    live = csv_rows(max(args.review.glob("????-??-??/de_jobs/all.csv")))
    previous = csv_rows(ROOT / "results/2026-11-21_beb1925/reports/2026-11-21/de_jobs/all.csv")
    seen = {r["url"] for r in previous}
    for r in live:
        r["new_since_previous_report"] = int(r["url"] not in seen)
    sets = {"vacancies": live}
    for name in ("labeled", "batch3"):
        sets[name] = [{k: r.get(k, "") for k in ("title", "company", "url", "esci_text")}
                      for r in jsonl(ROOT / "eval/de_jobs" / (name + ".jsonl")) if r.get("esci_text") in rerank_jobs.GRADE]
    titles = sorted({r["title"] for rows in sets.values() for r in rows})
    cv_text = (ROOT / "cv/master_cv.md").read_text()
    query = cv_text.split("## Professional Summary", 1)[1].split("\n## ", 1)[0].strip()
    tg = load_target("de_jobs")
    skills = json.loads((ROOT / "targets/de_jobs.json").read_text())["skills"]
    stack = cv.stack_terms(skills)
    raw = {"cv_score": {t: cv.parts(t, "", tg, stack)["cv_score"] for t in titles}}
    models = sweep.registry()
    errors = {}
    holdout = [r for p in sorted((ROOT / "eval/holdout").glob("*.jsonl")) for r in jsonl(p)]
    pos, neg = screens.prototypes()
    for model in models:
        name = model["short"]
        try:
            # Resume title scores on retries. Include the encoder protocol and CV in the key.
            cache = Path(os.environ["A2179_DENSE_CACHE"]) / "exp030"
            cache.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(json.dumps([2, model, query]).encode()).hexdigest()
            file = cache / (digest + ".json")
            scores = json.loads(file.read_text()) if file.exists() else {}
            missing = [t for t in titles if t not in scores]
            if missing:
                scores.update(zip(missing, map(float, cosine(model, query, missing)), strict=True))
                tmp = file.with_suffix(".tmp")
                tmp.write_text(json.dumps(scores))
                tmp.replace(file)
            raw[name] = scores
            sweep.compute_dense(holdout, out / f"holdout_{name}.jsonl", model["id"], model["proto_prefix"],
                                model["text_prefix"], local_only=True, device="cpu", pos=pos, neg=neg)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"[:500]
        gc.collect()
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(rerank_jobs.MODEL, device="cpu", local_files_only=True)
        scores = ce.predict([(rerank_jobs.QUERY, t) for t in titles], batch_size=32, show_progress_bar=False)
        raw["tinybert"] = dict(zip(titles, map(float, scores), strict=True))
        del ce
    except Exception as exc:
        errors["tinybert"] = str(exc)[:500]
    # Equal family weights. Prefix variants share their checkpoint's vote.
    active = [m for m in models if m["short"] in raw]
    ids = {m["id"] for m in active}
    groups = 1 + int(bool(active)) + int("tinybert" in raw)
    weights = {"cv_score": 1 / groups}
    if "tinybert" in raw:
        weights["tinybert"] = 1 / groups
    for m in active:
        variants = sum(x["id"] == m["id"] for x in active)
        weights[m["short"]] = 1 / (groups * len(ids) * variants)
    write_csv(out / "weights.csv", [dict(channel=k, weight=v) for k, v in weights.items()])
    metrics = {name: rank_table(rows, raw, weights, out / (name + ".csv")) for name, rows in sets.items()}
    try:
        rows = screens.real_rows() + holdout
        scores = screens.score_ce("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", screens.texts(rows), pos, neg)
        write_csv(out / "screen_rerank.csv", [dict(id=r["id"], label=r["label"], score=s)
                  for r, s in zip(rows, scores, strict=True)])
    except Exception as exc:
        errors["screen_rerank"] = str(exc)[:500]
    result = {"coverage": {k: len(v) for k, v in sets.items()}, "new_urls": sum(r["new_since_previous_report"] for r in live),
              "weights": weights, "auc": metrics, "errors": errors}
    (out / "exp030.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
