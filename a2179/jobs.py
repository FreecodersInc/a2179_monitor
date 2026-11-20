"""Official job-board APIs for the DE vacancies case (T-018, ADR-0017).

Remotive and Remote OK, both official JSON APIs, plus public ATS company boards (Greenhouse, Lever,
Ashby) in a2179/ats.py. Terms, checked 2026-11-19:

* Remotive (`https://remotive.com/api/remote-jobs`): mention Remotive and link back,
  do not republish the jobs on other boards, at most 4 requests per day, listings
  may be delayed 24 hours. We cache 24 hours (one call a day) and drop jobs newer
  than 24 hours.
* Remote OK (`https://remoteok.com/api`): mention Remote OK and link back. We cache
  24 hours so CI does not hammer the API.

Responses are cached on disk. LinkedIn, Indeed and Glassdoor are never called.
Reports keep title, company, location, a short snippet and the source URL, not
the full description.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import requests

REMOTIVE_URL = "https://remotive.com/api/remote-jobs?category=data"
REMOTEOK_URL = "https://remoteok.com/api"
REMOTIVE_HOME = "https://remotive.com"
REMOTEOK_HOME = "https://remoteok.com"
CACHE_TTL_S = 24 * 3600
REMOTIVE_MAX_PER_DAY = 4
PUBLISH_DELAY_S = 24 * 3600
SNIPPET_CHARS = 280
USER_AGENT = ("a2179_monitor/0.1 (+https://github.com/FreecodersInc/a2179_monitor; "
              "personal research, not a job board)")
BLOCKED_HOSTS = ("linkedin.com", "indeed.com", "glassdoor.com")
GetFn = Callable[[str], tuple]  # (status, text)

# Hosts we are allowed to call. Anything else, including the blocked boards, is refused.
ALLOWED_HOSTS = frozenset({"remotive.com", "www.remotive.com", "remoteok.com", "www.remoteok.com",
                           # public ATS job board APIs, one board per company (a2179/ats.py, ADR-0025)
                           "boards-api.greenhouse.io", "api.lever.co", "api.ashbyhq.com",
                           # T-037: Arbeitnow free API, We Work Remotely RSS, HN search API (robots checked 2026-11-20)
                           "www.arbeitnow.com", "weworkremotely.com", "hn.algolia.com"})


@dataclass
class Job:
    source: str
    url: str
    title: str
    company: str = ""
    location: str = ""
    snippet: str = ""
    fetched_at: str = ""
    published_at: str = ""
    source_id: str = ""
    text: str = ""   # plain description, capped at TEXT_CHARS (CV distance score, T-024); not shown in reports

    def to_dict(self):
        return asdict(self)


def utcnow(now=None) -> dt.datetime:
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    return now.astimezone(dt.timezone.utc)


def iso(now=None) -> str:
    return utcnow(now).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_dt(value) -> Optional[dt.datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(int(value), dt.timezone.utc)
    s = str(value).strip().replace("Z", "+00:00")
    try:
        x = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.timezone.utc)
    return x.astimezone(dt.timezone.utc)


TEXT_CHARS = 20000


def text_of(text: str) -> str:
    """Whole description as plain text, capped (T-024 needs skills and stack, not only the teaser)."""
    return snippet_of(text, TEXT_CHARS)


def snippet_of(text: str, n: int = SNIPPET_CHARS) -> str:
    if not text:
        return ""
    t = re.sub(r"<[^>]+>", " ", str(text))
    t = re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&quot;|&#39;", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= n:
        return t
    return t[: n - 1].rstrip() + "…"



def host_of(url: str) -> str:
    return urlparse(url).netloc.lower()


def assert_allowed(url: str):
    host = host_of(url)
    bare = host.removeprefix("www.")
    if bare in {h.removeprefix("www.") for h in BLOCKED_HOSTS} or any(bare.endswith("." + h) for h in BLOCKED_HOSTS):
        raise ValueError(f"refusing blocked job board: {host}")
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"refusing unexpected job API host: {host}")


def default_get(url: str, timeout: float = 20.0) -> tuple:
    """One GET with a project User-Agent. Tests inject their own get_fn and never call this."""
    assert_allowed(url)
    r = requests.get(url, timeout=timeout, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain;q=0.8, */*;q=0.5",
    })
    # JSON is UTF-8 (RFC 8259). r.text guesses ISO-8859-1 when the header has no charset, and Remote OK
    # then came out as mojibake ("â\x80\x94" for an em dash), whose \x85 broke splitlines() readers.
    return r.status_code, r.content.decode("utf-8", "replace")


def fix_text(s: str) -> str:
    """Undo UTF-8 read as Latin-1 (bodies cached before the fix). Unchanged when it does not round trip."""
    if not s or not any("\x80" <= c <= "\xff" for c in s):
        return s
    try:
        fixed = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s
    return fixed


def cache_paths(cache_dir: Path, name: str) -> tuple[Path, Path, Path]:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.json", d / f"{name}.meta.json", d / f"{name}.requests.json"


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def request_times(path: Path) -> list[dt.datetime]:
    out = []
    for s in _load_json(path, []):
        t = parse_dt(s)
        if t:
            out.append(t)
    return out


def recent_count(times: list[dt.datetime], now: dt.datetime, window_s: float = CACHE_TTL_S) -> int:
    return sum(1 for t in times if (now - t).total_seconds() < window_s)


def cache_fresh(meta: dict, now: dt.datetime, ttl_s: int = CACHE_TTL_S) -> bool:
    t = parse_dt((meta or {}).get("fetched_at"))
    return bool(t and (now - t).total_seconds() < ttl_s and (meta or {}).get("status") == 200)


def load_or_fetch(name: str, url: str, cache_dir: Path, fetch: str = "auto", get_fn: GetFn = None,
                  now=None, ttl_s: int = CACHE_TTL_S, max_per_day: int = 0) -> tuple[str, dict]:
    """Return (body, meta). fetch: auto | never | always.

    `always` still respects Remotive's 4 requests / 24h cap: if the cap is hit, the
    cache is used when present, otherwise a cap error is recorded and the body is empty.
    """
    now = utcnow(now)
    assert_allowed(url)
    body_p, meta_p, req_p = cache_paths(cache_dir, name)
    meta = _load_json(meta_p, {})
    body = body_p.read_text(encoding="utf-8") if body_p.is_file() else ""
    times = request_times(req_p)
    used_cache = bool(body and cache_fresh(meta, now, ttl_s))
    want = fetch == "always" or (fetch == "auto" and not used_cache)
    if fetch == "never":
        want = False
    cap_hit = False
    if want and max_per_day and recent_count(times, now) >= max_per_day:
        cap_hit = True
        want = False
    if want:
        fn = get_fn or default_get
        status, text = fn(url)
        rec = iso(now)
        times.append(now)
        req_p.write_text(json.dumps([t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in times], indent=1), encoding="utf-8")
        meta = {"url": url, "fetched_at": rec, "status": status, "bytes": len(text or ""), "from_cache": False}
        if status == 200 and text:
            body_p.write_text(text, encoding="utf-8")
            body = text
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return body, meta
    meta = dict(meta or {})
    meta["from_cache"] = True
    if cap_hit:
        meta["cap_hit"] = True
        meta["note"] = f"{name}: daily request cap reached, using cache"
    elif fetch == "never" and not body:
        meta["note"] = f"{name}: no cache and fetch=never"
        meta.setdefault("status", None)
        meta.setdefault("fetched_at", "")
        meta.setdefault("url", url)
    elif not body:
        meta["note"] = f"{name}: no cache and no fetch"
        meta.setdefault("url", url)
    return body, meta


def _too_new(published_at: Optional[dt.datetime], now: dt.datetime, delay_s: int) -> bool:
    return bool(published_at and (now - published_at).total_seconds() < delay_s)


def parse_remotive(body: str, fetched_at: str = "", now=None, delay_s: int = PUBLISH_DELAY_S) -> list[Job]:
    now = utcnow(now)
    fetched_at = fetched_at or iso(now)
    try:
        data = json.loads(body or "{}")
    except ValueError:
        return []
    jobs = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(jobs, list):
        return []
    out = []
    for j in jobs:
        if not isinstance(j, dict):
            continue
        url = (j.get("url") or j.get("url_short") or "").strip()
        title = fix_text((j.get("title") or "").strip())
        if not url or not title:
            continue
        pub = parse_dt(j.get("publication_date") or j.get("publication_date_utc"))
        if _too_new(pub, now, delay_s):
            continue
        out.append(Job(
            source="remotive",
            url=url,
            title=title,
            company=fix_text((j.get("company_name") or j.get("company") or "").strip()),
            location=fix_text((j.get("candidate_required_location") or j.get("location") or "").strip()),
            snippet=snippet_of(j.get("description") or ""), text=text_of(j.get("description") or ""),
            fetched_at=fetched_at,
            published_at=pub.strftime("%Y-%m-%dT%H:%M:%SZ") if pub else "",
            source_id=str(j.get("id") or ""),
        ))
    return out


def parse_remoteok(body: str, fetched_at: str = "", now=None) -> list[Job]:
    now = utcnow(now)
    fetched_at = fetched_at or iso(now)
    try:
        data = json.loads(body or "[]")
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for j in data:
        if not isinstance(j, dict):
            continue
        # First element is the legal notice (keys like "legal"), not a job.
        if "legal" in j and not (j.get("position") or j.get("title")):
            continue
        title = fix_text((j.get("position") or j.get("title") or "").strip())
        # Prefer the board URL so the report links back to Remote OK, as their terms ask.
        jid = j.get("id") or j.get("slug") or ""
        url = (j.get("url") or "").strip()
        if jid and not url.startswith("https://remoteok.com/") and not url.startswith("https://remoteOK.com/"):
            url = f"https://remoteok.com/remote-jobs/{jid}"
        if not url or not title:
            continue
        pub = parse_dt(j.get("epoch") or j.get("date"))
        loc = j.get("location")
        if isinstance(loc, list):
            loc = ", ".join(str(x) for x in loc if x)
        out.append(Job(
            source="remoteok",
            url=url,
            title=title,
            company=fix_text((j.get("company") or j.get("company_name") or "").strip()),
            location=fix_text((loc or "").strip()),
            snippet=fix_text(snippet_of(j.get("description") or "")), text=fix_text(text_of(j.get("description") or "")),
            fetched_at=fetched_at,
            published_at=pub.strftime("%Y-%m-%dT%H:%M:%SZ") if pub else "",
            source_id=str(jid),
        ))
    return out


# T-037. Terms and robots, checked 2026-11-20:
# * Arbeitnow (`/api/job-board-api`): free public API, no key, "please do not abuse, link back", robots
#   allows everything. One page (300 newest jobs) a day, cached 24 h.
# * We Work Remotely: RSS feeds, robots allows / (only account and admin paths are closed). One feed a day.
# * HN «Ask HN: Who is hiring?»: the search API behind the HN footer (hn.algolia.com, no robots.txt, 404).
#   Two calls a day: the latest thread of the whoishiring account, then its comments. Only top level
#   comments are jobs. The HN user name of the poster is not kept.
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"
WWR_URL = "https://weworkremotely.com/categories/remote-programming-jobs.rss"
HN_STORY_URL = "https://hn.algolia.com/api/v1/search_by_date?tags=story,author_whoishiring&hitsPerPage=5"
HN_COMMENTS_URL = "https://hn.algolia.com/api/v1/search?tags=comment,story_{id}&hitsPerPage=1000"


def _plain_html(s: str) -> str:
    import html as H
    return H.unescape(H.unescape(s or ""))


def parse_arbeitnow(body: str, fetched_at: str = "", now=None) -> list[Job]:
    fetched_at = fetched_at or iso(now)
    try:
        data = json.loads(body or "{}")
    except ValueError:
        return []
    rows = data.get("data") if isinstance(data, dict) else None
    out = []
    for j in rows if isinstance(rows, list) else []:
        if not isinstance(j, dict) or not j.get("url") or not j.get("title"):
            continue
        pub = parse_dt(j.get("created_at"))
        loc = (j.get("location") or "").strip()
        if j.get("remote") in (True, "true", "True") and "remote" not in loc.lower():
            loc = (loc + ", Remote").strip(", ")
        d = _plain_html(j.get("description"))
        out.append(Job(source="arbeitnow", url=j["url"].strip(), title=j["title"].strip(),
                       company=(j.get("company_name") or "").strip(), location=loc,
                       snippet=snippet_of(d), text=text_of(d), fetched_at=fetched_at,
                       published_at=pub.strftime("%Y-%m-%dT%H:%M:%SZ") if pub else "",
                       source_id=str(j.get("slug") or "")))
    return out


def parse_wwr(body: str, fetched_at: str = "", now=None) -> list[Job]:
    import email.utils
    import xml.etree.ElementTree as ET
    fetched_at = fetched_at or iso(now)
    try:
        root = ET.fromstring((body or "").encode("utf-8"))
    except ET.ParseError:
        return []
    out = []
    for it in root.iter("item"):
        g = lambda k: (it.findtext(k) or "").strip()   # noqa: E731
        raw, url = g("title"), g("link") or g("guid")
        if not raw or not url.startswith("https://weworkremotely.com/"):
            continue
        company, _, title = raw.partition(": ")
        if not title:
            company, title = "", raw
        try:
            pub = email.utils.parsedate_to_datetime(g("pubDate")).astimezone(dt.timezone.utc)
        except (TypeError, ValueError):
            pub = None
        d = g("description")
        out.append(Job(source="wwr", url=url, title=title.strip(), company=company.strip(), location=g("region"),
                       snippet=snippet_of(d), text=text_of(d), fetched_at=fetched_at,
                       published_at=pub.strftime("%Y-%m-%dT%H:%M:%SZ") if pub else "",
                       source_id=url.rsplit("/", 1)[-1]))
    return out


def hn_story_id(body: str) -> str:
    try:
        hits = json.loads(body or "{}").get("hits") or []
    except (ValueError, AttributeError):
        return ""
    for h in hits:
        if "who is hiring" in (h.get("title") or "").lower():
            return str(h.get("objectID") or "")
    return ""


def parse_hn(body: str, fetched_at: str = "", now=None) -> list[Job]:
    """Top level comments of a Who is hiring thread. First line «Company | Role | Location | ...»."""
    fetched_at = fetched_at or iso(now)
    try:
        hits = json.loads(body or "{}").get("hits") or []
    except (ValueError, AttributeError):
        return []
    out = []
    for h in hits:
        if str(h.get("parent_id")) != str(h.get("story_id")) or not h.get("comment_text"):
            continue
        html_ = _plain_html(h["comment_text"])
        first = re.split(r"<p>|\n", html_, maxsplit=1)[0]
        first = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", first)).strip()
        if len(first) < 8:
            continue
        parts = [x.strip() for x in first.split("|")]
        loc = next((x for x in parts[1:] if re.search(r"remote|onsite|on-site|hybrid|\b[A-Z][a-z]+, [A-Z]", x, re.I)), "")
        pub = parse_dt(h.get("created_at"))
        oid = str(h.get("objectID"))
        out.append(Job(source="hn", url=f"https://news.ycombinator.com/item?id={oid}", title=first[:160],
                       company=parts[0][:80] if len(parts) > 1 else "", location=loc[:120],
                       snippet=snippet_of(html_), text=text_of(html_), fetched_at=fetched_at,
                       published_at=pub.strftime("%Y-%m-%dT%H:%M:%SZ") if pub else "", source_id=oid))
    return out


SOURCES = {
    "arbeitnow": {"url": ARBEITNOW_URL, "parse": parse_arbeitnow, "max_per_day": 2,
                  "home": "https://www.arbeitnow.com", "label": "Arbeitnow"},
    "wwr": {"url": WWR_URL, "parse": parse_wwr, "max_per_day": 2,
            "home": "https://weworkremotely.com", "label": "We Work Remotely"},
    "remotive": {"url": REMOTIVE_URL, "parse": parse_remotive, "max_per_day": REMOTIVE_MAX_PER_DAY,
                 "home": REMOTIVE_HOME, "label": "Remotive"},
    "remoteok": {"url": REMOTEOK_URL, "parse": parse_remoteok, "max_per_day": 0,
                 "home": REMOTEOK_HOME, "label": "Remote OK"},
}


def collect_jobs(cache_dir: Path, sources=("remotive", "remoteok"), fetch: str = "auto",
                 get_fn: GetFn = None, now=None, ats_kw: dict | None = None) -> tuple[list[Job], list[dict]]:
    """Fetch (or load cache) and parse the configured official APIs. "ats" = every company board in
    targets/ats_companies.json (a2179/ats.py). Jobs are deduplicated across sources, first one wins."""
    now = utcnow(now)
    jobs, notes = [], []
    for name in sources:
        if name == "ats":
            from a2179 import ats
            kw = {"budget_s": float(os.environ.get("A2179_ATS_BUDGET_S", ats.BUDGET_S))}
            kw.update(ats_kw or {})
            aj, an = ats.collect_ats(cache_dir, fetch=fetch, get_fn=get_fn, now=now, **kw)
            jobs.extend(aj)
            notes.extend(an)
            continue
        if name not in SOURCES and name != "hn":
            notes.append({"source": name, "ok": False, "note": f"unknown job source {name}"})
            continue
        if name == "hn":   # two calls: latest thread id, then its comments
            sb, sm = load_or_fetch("hn_story", HN_STORY_URL, cache_dir, fetch=fetch, get_fn=get_fn, now=now, max_per_day=2)
            sid = hn_story_id(sb)
            body, meta = load_or_fetch("hn", HN_COMMENTS_URL.format(id=sid), cache_dir, fetch=fetch, get_fn=get_fn,
                                       now=now, max_per_day=2) if sid else ("", {"note": "hn: no Who is hiring thread", **sm})
            parsed = parse_hn(body, fetched_at=meta.get("fetched_at") or iso(now)) if body else []
            jobs.extend(parsed)
            notes.append({"source": "hn", "ok": bool(parsed), "n": len(parsed), "from_cache": bool(meta.get("from_cache")),
                          "status": meta.get("status"), "fetched_at": meta.get("fetched_at") or "", "note": meta.get("note") or "",
                          "home": "https://news.ycombinator.com", "label": "HN Who is hiring", "thread": sid})
            continue
        spec = SOURCES[name]
        body, meta = load_or_fetch(name, spec["url"], cache_dir, fetch=fetch, get_fn=get_fn, now=now,
                                   max_per_day=spec["max_per_day"])
        parsed = spec["parse"](body, fetched_at=meta.get("fetched_at") or iso(now), now=now) if body else []
        jobs.extend(parsed)
        notes.append({
            "source": name, "ok": bool(parsed) or (bool(body) and meta.get("status") in (200, None)),
            "n": len(parsed), "from_cache": bool(meta.get("from_cache")),
            "status": meta.get("status"), "fetched_at": meta.get("fetched_at") or "",
            "note": meta.get("note") or meta.get("cap_hit") and f"{name} daily cap, used cache" or "",
            "home": spec["home"], "label": spec["label"],
        })
    from a2179.ats import dedup
    return dedup(jobs), notes
