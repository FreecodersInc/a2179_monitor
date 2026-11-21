"""Public ATS job boards for the DE vacancies case (T-019, T-038, ADR-0025): Greenhouse, Lever, Ashby, Workable.

Each company publishes its own openings through a documented public JSON endpoint, no key, no login:
  * Greenhouse Job Board API  https://boards-api.greenhouse.io/v1/boards/<slug>/jobs?content=true
  * Lever Postings API         https://api.lever.co/v0/postings/<slug>?mode=json
  * Ashby Job Postings API     https://api.ashbyhq.com/posting-api/job-board/<slug>
  * Workable widget API        https://apply.workable.com/api/v1/widget/accounts/<slug>
The company list lives in targets/ats_companies.json, every slug checked with one request.

Polite by construction: one request per board per 24 hours (the cache of a2179.jobs), a pause after
every real request, a timeout, the project User-Agent, a byte cap per response (a Greenhouse board
too big with descriptions is fetched once more without them), a wall clock budget per run (boards
left over are read from cache only, the next run continues). Bodies are decoded as UTF-8.
Only titles that look like data, ML, platform or analyst work are kept (the prefilter), the rest of a
company's openings never leave this module.
"""
from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path

import requests

from a2179 import jobs as J

ROOT = Path(__file__).resolve().parent.parent
COMPANIES = ROOT / "targets" / "ats_companies.json"
MAX_BYTES = 20 * 1024 * 1024   # big boards with descriptions (OpenAI, Palantir) run 5 to 15 MB
DELAY_S = 1.0
BUDGET_S = 300.0
HOSTS = {"greenhouse": "boards-api.greenhouse.io", "lever": "api.lever.co", "ashby": "api.ashbyhq.com",
         "workable": "apply.workable.com"}
TOO_BIG = 413

TITLE_RX = re.compile(
    r"\b(data|analytics?|etl|elt|big\s*data|warehouse|lakehouse|bi|ml|mlops|llmops|machine\s*learning|ai|llm|"
    r"platform|devops|sre|site\s*reliability|cloud|infrastructure|scientists?|analysts?|spark|kafka)\b", re.I)
TORONTO_VANCOUVER = re.compile(r"\b(toronto|vancouver)\b", re.I)


def board_url(ats: str, slug: str, content: bool = True) -> str:
    if ats == "greenhouse":
        return f"https://{HOSTS[ats]}/v1/boards/{slug}/jobs" + ("?content=true" if content else "")
    if ats == "lever":
        return f"https://{HOSTS[ats]}/v0/postings/{slug}?mode=json"
    if ats == "ashby":
        return f"https://{HOSTS[ats]}/posting-api/job-board/{slug}"
    if ats == "workable":
        return f"https://{HOSTS[ats]}/api/v1/widget/accounts/{slug}"
    raise ValueError(f"unknown ATS {ats}")


def capped_get(url: str, timeout: float = 20.0, max_bytes: int = MAX_BYTES) -> tuple:
    """GET with the project User-Agent, streamed, aborted past max_bytes (status 413, empty body)."""
    J.assert_allowed(url)
    with requests.get(url, timeout=timeout, stream=True, headers={
            "User-Agent": J.USER_AGENT, "Accept": "application/json"}) as r:
        n = r.headers.get("Content-Length")
        if n and n.isdigit() and int(n) > max_bytes:
            return TOO_BIG, ""
        buf = bytearray()
        for chunk in r.iter_content(65536):
            buf += chunk
            if len(buf) > max_bytes:
                return TOO_BIG, ""
        return r.status_code, bytes(buf).decode("utf-8", "replace")


def keep_title(title: str) -> bool:
    return bool(TITLE_RX.search(title or ""))


def _plain(text: str, n: int = 0) -> str:
    """Greenhouse sends escaped HTML: unescape, drop tags, unescape the text entities, then trim
    (n = 0: snippet length, else up to n chars, the full text for T-024)."""
    t = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return J.snippet_of(html.unescape(t), n or J.SNIPPET_CHARS)


def parse_greenhouse(body: str, company: str, fetched_at: str) -> list[J.Job]:
    try:
        data = json.loads(body or "{}")
    except ValueError:
        return []
    out = []
    for j in (data.get("jobs") or []) if isinstance(data, dict) else []:
        title, url = (j.get("title") or "").strip(), (j.get("absolute_url") or "").strip()
        if not title or not url:
            continue
        loc = (j.get("location") or {}).get("name", "") if isinstance(j.get("location"), dict) else ""
        pub = J.parse_dt(j.get("first_published") or j.get("updated_at"))
        out.append(J.Job(source="greenhouse", url=url, title=title, company=company, location=(loc or "").strip(),
                         snippet=_plain(j.get("content") or ""), text=_plain(j.get("content") or "", J.TEXT_CHARS),
                         fetched_at=fetched_at,
                         published_at=pub.strftime("%Y-%m-%dT%H:%M:%SZ") if pub else "", source_id=str(j.get("id", ""))))
    return out


def parse_lever(body: str, company: str, fetched_at: str) -> list[J.Job]:
    try:
        data = json.loads(body or "[]")
    except ValueError:
        return []
    out = []
    for j in data if isinstance(data, list) else []:
        if not isinstance(j, dict):
            continue
        title, url = (j.get("text") or "").strip(), (j.get("hostedUrl") or "").strip()
        if not title or not url:
            continue
        cat = j.get("categories") or {}
        locs = cat.get("allLocations") or ([cat.get("location")] if cat.get("location") else [])
        pub = J.parse_dt(int(j["createdAt"]) // 1000) if str(j.get("createdAt", "")).isdigit() else None
        out.append(J.Job(source="lever", url=url, title=title, company=company,
                         location=", ".join(str(x) for x in locs if x), snippet=_plain(j.get("descriptionPlain") or ""),
                         text=_plain(" ".join([j.get("descriptionPlain") or ""] + [f"{x.get('text', '')} {x.get('content', '')}" for x in (j.get("lists") or []) if isinstance(x, dict)] + [j.get("additionalPlain") or ""]), J.TEXT_CHARS),
                         fetched_at=fetched_at, published_at=pub.strftime("%Y-%m-%dT%H:%M:%SZ") if pub else "",
                         source_id=str(j.get("id", ""))))
    return out


def parse_ashby(body: str, company: str, fetched_at: str) -> list[J.Job]:
    try:
        data = json.loads(body or "{}")
    except ValueError:
        return []
    out = []
    for j in (data.get("jobs") or []) if isinstance(data, dict) else []:
        if j.get("isListed") is False:
            continue
        title, url = (j.get("title") or "").strip(), (j.get("jobUrl") or "").strip()
        if not title or not url:
            continue
        locs = [j.get("location") or ""] + [(s or {}).get("location", "") for s in j.get("secondaryLocations") or []]
        loc = ", ".join(x for x in locs if x)
        if j.get("isRemote") and "remote" not in loc.lower():
            loc = f"{loc}, Remote" if loc else "Remote"
        pub = J.parse_dt(j.get("publishedAt"))
        out.append(J.Job(source="ashby", url=url, title=title, company=company, location=loc,
                         snippet=_plain(j.get("descriptionPlain") or ""), text=_plain(j.get("descriptionPlain") or "", J.TEXT_CHARS), fetched_at=fetched_at,
                         published_at=pub.strftime("%Y-%m-%dT%H:%M:%SZ") if pub else "", source_id=str(j.get("id", ""))))
    return out



def parse_workable(body: str, company: str, fetched_at: str) -> list[J.Job]:
    try:
        data = json.loads(body or "{}")
    except ValueError:
        return []
    jobs = data.get("jobs") if isinstance(data, dict) else (data if isinstance(data, list) else [])
    out = []
    for j in jobs or []:
        if not isinstance(j, dict):
            continue
        title = (j.get("title") or "").strip()
        url = (j.get("application_url") or j.get("url") or j.get("shortlink") or "").strip()
        if not title or not url:
            short = (j.get("shortcode") or "").strip()
            if title and short:
                url = f"https://apply.workable.com/{company}/j/{short}/"
            else:
                continue
        loc = j.get("location")
        if isinstance(loc, dict):
            parts = [loc.get("city") or "", loc.get("region") or "", loc.get("country") or ""]
            loc_s = ", ".join(x for x in parts if x)
            if loc.get("telecommuting") or loc.get("telecommute"):
                if "remote" not in loc_s.lower():
                    loc_s = f"{loc_s}, Remote" if loc_s else "Remote"
        else:
            loc_s = (loc or "").strip()
        pub = J.parse_dt(j.get("created_at") or j.get("published_on") or j.get("updated_at"))
        desc = j.get("description") or j.get("descriptionPlain") or ""
        out.append(J.Job(source="workable", url=url, title=title, company=company, location=loc_s,
                         snippet=_plain(desc), text=_plain(desc, J.TEXT_CHARS), fetched_at=fetched_at,
                         published_at=pub.strftime("%Y-%m-%dT%H:%M:%SZ") if pub else "",
                         source_id=str(j.get("shortcode") or j.get("id") or "")))
    return out


PARSE = {"greenhouse": parse_greenhouse, "lever": parse_lever, "ashby": parse_ashby, "workable": parse_workable}


def load_companies(path: Path | None = None) -> list[dict]:
    """targets/ats_companies.json, or A2179_ATS_COMPANIES (a path, or "off" for none: the tests)."""
    import os
    env = os.environ.get("A2179_ATS_COMPANIES")
    if path is None:
        if env is not None and env.strip().lower() in ("", "off", "0", "none"):
            return []
        path = Path(env) if env else COMPANIES
    if not Path(path).is_file():
        return []
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return [c for c in d.get("companies", []) if c.get("ats") in PARSE and c.get("slug") and c.get("ok", True)]


class Budget:
    def __init__(self, seconds: float, clock=time.monotonic):
        self.clock, self.end = clock, clock() + seconds

    def left(self) -> bool:
        return self.clock() < self.end


def _cache_file(cache_dir: Path, ats: str, slug: str) -> Path:
    return Path(cache_dir) / "ats" / f"{ats}_{slug}.json"


def fetch_board(c: dict, cache_dir: Path, fetch: str, get_fn, now, budget: Budget, delay_s: float, sleep_fn,
                prefilter: bool = True) -> dict:
    """One board: {"status", "fetched_at", "all", "jobs": [kept Job dicts], "from_cache", "note"}.

    The cache keeps only the jobs that pass the prefilter, trimmed to the Job fields, never the raw
    body (boards with descriptions run to megabytes). Fresh for 24 hours like the other sources.
    Network only while the run budget lasts, after that cache only and the next run continues."""
    ats, slug = c["ats"], c["slug"]
    cf = _cache_file(cache_dir, ats, slug)
    old = json.loads(cf.read_text(encoding="utf-8")) if cf.is_file() else {}
    fresh = J.cache_fresh(old, now)
    if fetch == "never" or (fetch == "auto" and fresh) or not budget.left():
        rec = dict(old or {"status": None, "all": 0, "jobs": []})
        rec["from_cache"] = True
        if not old and fetch != "never":
            rec["note"] = f"{ats}:{slug}: run budget spent, next run fetches it"
        return rec
    fn = get_fn or capped_get
    url = board_url(ats, slug)
    try:
        status, body = fn(url)
    except requests.RequestException as e:
        status, body = None, ""
        err = type(e).__name__
    else:
        err = ""
    sleep_fn(delay_s)
    if ats == "greenhouse" and status == TOO_BIG:
        url = board_url(ats, slug, content=False)
        status, body = fn(url)
        sleep_fn(delay_s)
    fetched = J.iso(now)
    parsed = PARSE[ats](body, c.get("name") or slug, fetched) if status == 200 and body else []
    kept = [j for j in parsed if keep_title(j.title)] if prefilter else parsed
    rec = {"url": url, "status": status, "fetched_at": fetched, "all": len(parsed),
           "jobs": [j.to_dict() for j in kept], "note": err}
    if status == 200:
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    elif old:
        rec = {**old, "note": f"{ats}:{slug}: HTTP {status}, kept the last good cache"}
    rec["from_cache"] = False
    return rec


def collect_ats(cache_dir: Path, companies=None, fetch: str = "auto", get_fn=None, now=None,
                budget_s: float = BUDGET_S, delay_s: float = DELAY_S, sleep_fn=time.sleep, clock=time.monotonic,
                prefilter: bool = True) -> tuple[list[J.Job], list[dict]]:
    now = J.utcnow(now)
    companies = load_companies() if companies is None else companies
    budget = Budget(budget_s, clock)
    jobs, per = [], []
    for c in companies:
        rec = fetch_board(c, Path(cache_dir), fetch, get_fn, now, budget, delay_s, sleep_fn, prefilter)
        kept = [J.Job(**j) for j in rec.get("jobs") or []]
        jobs.extend(kept)
        per.append({"ats": c["ats"], "slug": c["slug"], "status": rec.get("status"), "all": rec.get("all", 0),
                    "kept": len(kept), "from_cache": bool(rec.get("from_cache")), "note": rec.get("note", "")})
    notes = []
    for ats in PARSE:
        rows = [p for p in per if p["ats"] == ats]
        if not rows:
            continue
        left = sum(1 for p in rows if "budget" in p["note"])
        notes.append({"source": ats, "ok": any(p["status"] == 200 for p in rows),
                      "n": sum(p["kept"] for p in rows), "from_cache": all(p["from_cache"] for p in rows),
                      "status": 200 if any(p["status"] == 200 for p in rows) else None,
                      "fetched_at": J.iso(now), "boards": len(rows),
                      "boards_ok": sum(1 for p in rows if p["status"] == 200),
                      "note": f"{len(rows)} boards, {sum(p['all'] for p in rows)} openings, "
                              f"{sum(p['kept'] for p in rows)} kept by the title prefilter"
                              + (f", {left} boards left for the next run" if left else ""),
                      "home": f"https://{HOSTS[ats]}", "label": ats.capitalize(), "boards_detail": rows})
    return jobs, notes


def dedup(jobs: list[J.Job]) -> list[J.Job]:
    """First occurrence wins: same URL, or same normalised title + company + location."""
    from a2179.normalize import normalize
    seen, out = set(), []
    for j in jobs:
        keys = {("u", j.url.strip().lower().rstrip("/"))} if j.url else set()
        keys.add(("t", normalize(j.title), normalize(j.company), normalize(j.location)))
        if keys & seen:
            continue
        seen |= keys
        out.append(j)
    return out


def probe(candidates: list[dict], cache_dir: Path, get_fn=None, budget_s: float = 100.0, delay_s: float = DELAY_S,
          sleep_fn=time.sleep, clock=time.monotonic, state_path: Path = None) -> dict:
    """Check slugs with one request each: the guessed ATS first, then the others. Resumable via state_path."""
    fn = get_fn or (lambda u: capped_get(u, max_bytes=1 << 20))   # a check, not a download
    state_path = Path(state_path or Path(cache_dir) / "ats_probe.json")
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    budget = Budget(budget_s, clock)
    for c in candidates:
        order = [c["ats"]] + [a for a in PARSE if a != c["ats"]] if c.get("ats") in PARSE else list(PARSE)
        for ats in order:
            k = f"{ats}:{c['slug']}"
            if any(state.get(f"{a}:{c['slug']}", {}).get("status") == 200 for a in PARSE):
                break
            if k in state:
                continue
            if not budget.left():
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(json.dumps(state, indent=1))
                return state
            url = board_url(ats, c["slug"], content=False) + ("&limit=1" if ats == "lever" else "")
            try:
                status, body = fn(url)
            except requests.RequestException as e:
                status, body = None, ""
                state[k] = {"status": None, "error": type(e).__name__}
            else:
                n = None
                if status == 200:
                    try:
                        d = json.loads(body)
                        n = len(d) if isinstance(d, list) else len(d.get("jobs") or [])
                    except ValueError:
                        status = "bad_json"
                if status == TOO_BIG:   # a 404 body is tiny, an oversized answer is a real board
                    status = 200
                state[k] = {"status": status, "jobs": n, "name": c.get("name", c["slug"])}
            sleep_fn(delay_s)
            if state[k]["status"] == 200:
                break
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1))
    return state


def build_list(state: dict, names: dict | None = None, checked: str = "") -> dict:
    """targets/ats_companies.json from probe results: boards that answered 200, 404s dropped."""
    out = []
    for k, v in sorted(state.items()):
        ats, slug = k.split(":", 1)
        if v.get("status") == 200:
            out.append({"slug": slug, "ats": ats, "name": (names or {}).get(slug) or v.get("name") or slug,
                        "jobs_at_check": v.get("jobs")})
    by = {a: sum(1 for c in out if c["ats"] == a) for a in PARSE}
    return {"_doc": "Public ATS boards polled by a2179/ats.py (ADR-0025). Every slug answered 200 to one request "
                    "on the checked date, candidates that answered 404 on all three ATS were dropped. Names and "
                    "slugs from public knowledge, no personal data.",
            "checked": checked, "counts": by, "companies": out}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--candidates", required=True)
    p.add_argument("--cache", default=str(ROOT / "data" / "cache" / "jobs"))
    p.add_argument("--budget", type=float, default=100.0)
    b = sub.add_parser("build")
    b.add_argument("--cache", default=str(ROOT / "data" / "cache" / "jobs"))
    b.add_argument("--out", default=str(COMPANIES))
    a = ap.parse_args(argv)
    st = Path(a.cache) / "ats_probe.json"
    if a.cmd == "probe":
        cands = json.loads(Path(a.candidates).read_text())["candidates"]
        state = probe(cands, Path(a.cache), budget_s=a.budget, state_path=st)
        done = {c["slug"] for c in cands
                if any(state.get(f"{x}:{c['slug']}", {}).get("status") == 200 for x in PARSE)
                or all(f"{x}:{c['slug']}" in state for x in PARSE)}
        ok = sum(1 for v in state.values() if v.get("status") == 200)
        print(json.dumps({"candidates": len(cands), "touched": len(done), "ok_boards": ok, "requests": len(state)}))
        return 0
    state = json.loads(st.read_text())
    d = build_list(state, checked=J.iso()[:10])
    Path(a.out).write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(d["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
