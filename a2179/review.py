"""Score collected listings or vacancies and write a manual-review report.

Canonical output: reports/<date>/<case>/matches.csv and matches.md, plus all.csv
and all.md with every scored item (so vacancies can be checked by hand later,
with the source link). Columns: title, company, location, score, decision, url,
fetched_at.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

from .jobs import Job, REMOTIVE_HOME, REMOTEOK_HOME, iso, utcnow
from .matcher import DEFAULT_THRESHOLDS, Matcher
from .parsers import Listing
from .similarity import TfidfSim

CASES = ("a2179_screen", "de_jobs")
COLUMNS = ("title", "company", "location", "score", "decision", "url", "fetched_at")
EXTRA = ("source", "reasons")


def case_dir(out_root: Path, date: str, case: str) -> Path:
    d = Path(out_root) / date / case
    d.mkdir(parents=True, exist_ok=True)
    return d


def today(now=None) -> str:
    return utcnow(now).strftime("%Y-%m-%d")


def _md_cell(s) -> str:
    return str(s or "").replace("|", "/").replace("\n", " ").strip()


def row_from_listing(l: Listing, fetched_at: str = "", company: str = "") -> dict:
    return {
        "title": l.title or "",
        "company": company or (l.source or ""),
        "location": l.location or "",
        "url": l.url or "",
        "fetched_at": fetched_at,
        "source": l.source or "",
        "snippet": l.snippet or "",
    }


def row_from_job(j: Job) -> dict:
    d = j.to_dict()
    d.setdefault("snippet", j.snippet)
    return d


def row_from_dict(d: dict, fetched_at: str = "") -> dict:
    return {
        "title": d.get("title") or "",
        "company": d.get("company") or d.get("site") or d.get("source") or "",
        "location": d.get("location") or "",
        "url": d.get("url") or "",
        "fetched_at": d.get("fetched_at") or d.get("collected_at") or fetched_at,
        "source": d.get("source") or d.get("site") or "",
        "snippet": d.get("snippet") or "",
    }


def load_listings(path: Path, fetched_at: str = "") -> list[dict]:
    """JSONL (one object per line) or a JSON list. Missing file -> empty list."""
    p = Path(path)
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl" or p.suffixes[-1:] == [".jsonl"]:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(row_from_dict(json.loads(line), fetched_at=fetched_at))
        return rows
    data = json.loads(text)
    if isinstance(data, list):
        return [row_from_dict(x, fetched_at=fetched_at) for x in data if isinstance(x, dict)]
    return []


def make_matcher(target, no_embed=True, thresholds=None):
    sim = TfidfSim(target.positive, target.negative)
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    return Matcher(sim, "tfidf", th, target=target), "tfidf"


def score_rows(rows: list[dict], target, matcher=None) -> list[dict]:
    matcher = matcher or make_matcher(target)[0]
    out = []
    for r in rows:
        d = matcher.decide(r.get("title") or "", r.get("snippet") or "")
        item = {k: r.get(k) or "" for k in ("title", "company", "location", "url", "fetched_at", "source")}
        item["score"] = round(float(d.sim_pos), 4)
        item["decision"] = "match" if d.match else "no"
        item["reasons"] = "; ".join(d.reasons)
        item["regex"] = d.regex
        item["model"] = d.model
        out.append(item)
    out.sort(key=lambda x: (x["decision"] != "match", -float(x["score"] or 0), x["title"]))
    return out


def write_table(path: Path, rows: list[dict], columns=COLUMNS + EXTRA):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def write_md(path: Path, rows: list[dict], title: str, notes=None, sources=None, columns=COLUMNS):
    lines = [f"# {title}", "", f"{len(rows)} rows.", ""]
    if notes:
        for n in notes:
            if n:
                lines.append(f"- {n}")
        lines.append("")
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines += [head, sep]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            if c == "url" and v:
                cells.append(f"[link]({v})")
            elif c == "score" and v != "":
                cells.append(f"{float(v):.3f}")
            else:
                cells.append(_md_cell(v))
        lines.append("| " + " | ".join(cells) + " |")
    used = {s for s in (sources or []) if s}
    attr = []
    if "remotive" in used:
        attr.append(f"[Remotive]({REMOTIVE_HOME})")
    if "remoteok" in used:
        attr.append(f"[Remote OK]({REMOTEOK_HOME})")
    if attr:
        lines += ["", "Sources: " + " and ".join(attr)
                  + ". Title, company, location and a link only, not a republication of the full posting."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case_reports(out_dir: Path, case: str, rows: list[dict], notes=None, date: str = "") -> dict:
    """Write all.csv / all.md and matches.csv / matches.md for one case."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = [r for r in rows if r.get("decision") == "match"]
    sources = [r.get("source") for r in rows]
    title_all = f"{case}: all scored items ({date or today()})"
    title_m = f"{case}: matches for manual review ({date or today()})"
    write_table(out_dir / "all.csv", rows)
    write_table(out_dir / "matches.csv", matches)
    write_md(out_dir / "all.md", rows, title_all, notes=notes, sources=sources)
    write_md(out_dir / "matches.md", matches, title_m, notes=notes, sources=sources)
    summary = {
        "case": case, "date": date or today(), "n": len(rows), "n_matches": len(matches),
        "notes": [n for n in (notes or []) if n],
        "files": ["matches.csv", "matches.md", "all.csv", "all.md"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
