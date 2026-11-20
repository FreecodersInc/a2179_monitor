"""Job mode of the monitor (T-036): fresh vacancies from the M-003 sources, ranked by closeness to the
CV (eval/cv_distance.py, variant B: stack terms from cv/master_cv.md) plus the profile features, and
only the top new ones are pinged. A ranking aid, not a judge: on fresh full-text labels the CV score has
AUC 0.656 (EXP-011).

score = 0.5 * cv_score + 0.5 * feat, feat = (features(title, snippet) + 1) / 2 of the de_jobs profile.
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

DEFAULT_SOURCES = ("remotive", "remoteok", "ats")


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


def rank(jobs, score) -> list[tuple]:
    out = [(j, score(j)) for j in jobs]
    return sorted(out, key=lambda x: -x[1]["score"])


def message(job, s) -> str:
    where = f" · {job.location}" if job.location else ""
    return (f"💼 {job.title} · {job.company}{where}\n"
            f"близость к CV {s['cv']:.2f}, признаки {s['feat']:.2f}, итог {s['score']:.2f}\n{job.url}")


def run_jobs(cfg, data: Path, alerter, dry_run=False, seed=False, top=10, min_score=0.0,
             sources=DEFAULT_SOURCES, fetch="auto", collect=None, score=None, now=None) -> dict:
    from a2179.jobs import collect_jobs
    collect = collect or (lambda: collect_jobs(data / "cache" / "jobs", sources=tuple(sources), fetch=fetch))
    jobs, notes = collect()
    score = score or scorer(cfg.get("job_target", "de_jobs"))
    ranked = rank(jobs, score)
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
    return {"jobs": len(jobs), "new": len(new), "ranked": ranked, "picked": picked, "sent": sent, "notes": notes}
