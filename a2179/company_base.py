"""T-038 company base: collect relevant companies, probe public ATS slugs, write companies.csv.

Sources (fetched only on the runner, tests inject get_fn):
  * remoteintech/remote-jobs company pages (src/companies/*.md frontmatter; one tarball)
  * YC catalog snapshot https://yc-oss.github.io/api/companies/all.json
    (api.ycombinator.com is not used: its robots.txt disallows /)
Known live slugs from targets/ats_companies.json are merged in. JetBrains is seeded as a
greenhouse candidate (public board slug "jetbrains") so the geo-filter control vacancy can
show up; the filter itself has no JetBrains branch.

CSV fields follow T-038: name, domain, ATS, slug, source, geo hint, last successful check,
vacancy count, plus company_url, board_url, fail_count, status. Duplicates are dropped by
domain. A board that fails three liveness checks in a row is marked sleep.
"""
from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

from a2179 import ats as A
from a2179 import jobs as J

ROOT = Path(__file__).resolve().parent.parent
CSV_FIELDS = ["name", "domain", "ats", "slug", "source", "geo", "last_ok", "n_jobs",
              "company_url", "board_url", "fail_count", "status"]
REMOTEINTECH_TAR = "https://codeload.github.com/remoteintech/remote-jobs/tar.gz/refs/heads/main"
REMOTEINTECH_README_URLS = (
    "https://raw.githubusercontent.com/remoteintech/remote-jobs/main/README.md",
    "https://raw.githubusercontent.com/remoteintech/remote-jobs/master/README.md",
)
YC_URL = "https://yc-oss.github.io/api/companies/all.json"
ROBOTS = {
    "codeload.github.com": "https://github.com/robots.txt",
    "raw.githubusercontent.com": "https://raw.githubusercontent.com/robots.txt",
    "yc-oss.github.io": "https://yc-oss.github.io/robots.txt",
}
ATS_HOSTS = frozenset({
    "boards.greenhouse.io", "job-boards.greenhouse.io", "job-boards.eu.greenhouse.io",
    "jobs.lever.co", "jobs.ashbyhq.com", "apply.workable.com",
    "boards-api.greenhouse.io", "api.lever.co", "api.ashbyhq.com",
})
FAILS_TO_SLEEP = 3
# legal / noise words dropped when guessing an ATS slug from a company name
SLUG_NOISE = re.compile(
    r"\b(incorporated|inc|ltd|llc|gmbh|ag|bv|corp|corporation|co|company|labs?|laboratory|"
    r"technologies|technology|software|systems?|international|group|holdings?|the|io|ai)\b", re.I)
MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
STATUS_LIVE, STATUS_UNKNOWN, STATUS_NONE, STATUS_SLEEP = "live", "unknown", "none", "sleep"


def domain_of(url: str) -> str:
    if not url:
        return ""
    if "://" not in url:
        url = "https://" + url
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host or host in ATS_HOSTS or host.endswith(".greenhouse.io"):
        return ""
    return host


def public_board_url(ats: str, slug: str) -> str:
    if not ats or not slug:
        return ""
    if ats == "greenhouse":
        return f"https://boards.greenhouse.io/{slug}"
    if ats == "lever":
        return f"https://jobs.lever.co/{slug}"
    if ats == "ashby":
        return f"https://jobs.ashbyhq.com/{slug}"
    if ats == "workable":
        return f"https://apply.workable.com/{slug}/"
    return ""


def slug_variants(name: str, domain: str = "", limit: int = 5) -> list[str]:
    """Name glued, hyphenated, and without Inc/Ltd; domain head first when it looks like a slug."""
    out: list[str] = []

    def add(s: str):
        s = re.sub(r"[^a-z0-9-]", "", (s or "").lower()).strip("-")
        s = re.sub(r"-{2,}", "-", s)
        if 1 < len(s) <= 60 and s not in out:
            out.append(s)

    if domain:
        head = domain.split(".")[0]
        if head not in {"www", "jobs", "careers", "app", "apply", "board", "boards"}:
            add(head)
    raw = (name or "").strip()
    add(re.sub(r"[^a-z0-9]+", "", raw.lower()))
    add(re.sub(r"[^a-z0-9]+", "-", raw.lower()))
    stripped = SLUG_NOISE.sub(" ", raw)
    add(re.sub(r"[^a-z0-9]+", "", stripped.lower()))
    add(re.sub(r"[^a-z0-9]+", "-", stripped.lower()))
    return out[:limit]


def robots_allows(text: str, path: str) -> bool:
    """Tiny robots.txt reader: User-agent * (and a2179_monitor), last matching Allow/Disallow wins."""
    if not (text or "").strip():
        return True
    ua, rules = "*", []
    applies = False
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        k, v = k.strip().lower(), v.strip()
        if k == "user-agent":
            ua = v
            applies = ua in ("*", "a2179_monitor") or ua.lower() == "a2179_monitor/0.1"
            continue
        if applies and k in {"allow", "disallow"}:
            rules.append((k, v))
    allowed = True
    for k, v in rules:
        if not v:
            if k == "disallow":
                allowed = True  # empty Disallow means allow all
            continue
        if path.startswith(v) or v == "/":
            allowed = k == "allow" if v != "/" else k == "allow"
            if v == "/":
                allowed = k == "allow"
    return allowed


def parse_remoteintech(md: str) -> list[dict]:
    """Companies from the remoteintech README: markdown tables and bullet links."""
    rows = []
    lines = (md or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "|" in line and re.search(r"name|company|website|url", line, re.I):
            headers = [c.strip().lower() for c in line.split("|") if c.strip()]
            i += 1
            if i < len(lines) and re.match(r"^\s*\|?\s*[-:| ]+$", lines[i]):
                i += 1
            while i < len(lines) and "|" in lines[i]:
                cols = [c.strip() for c in lines[i].split("|")]
                cols = cols[1:-1] if cols and cols[0] == "" and cols[-1] == "" else [c for c in cols if c != ""]
                rec = {headers[j]: cols[j] if j < len(cols) else "" for j in range(len(headers))}
                name, url, geo = _from_table(rec)
                if name and url:
                    rows.append(_row(name, url, geo, "remoteintech"))
                i += 1
            continue
        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            name, url, geo = _from_bullet(m.group(1))
            if name and url:
                rows.append(_row(name, url, geo, "remoteintech"))
        i += 1
    return rows



def parse_remoteintech_frontmatter(md: str, default_name: str = "") -> dict | None:
    """One company page from remoteintech: YAML-like frontmatter with title, website, region."""
    text = md or ""
    if not text.lstrip().startswith("---"):
        return None
    rest = text.lstrip()[3:]
    end = rest.find("\n---")
    if end < 0:
        return None
    fields = {}
    for line in rest[:end].splitlines():
        if ":" not in line or line.startswith(" ") or line.startswith("-"):
            continue
        k, v = line.split(":", 1)
        fields[k.strip()] = v.strip().strip("\'\"").strip()
    name = fields.get("title") or fields.get("name") or default_name
    url = fields.get("website") or fields.get("url") or fields.get("careers_url") or ""
    geo = fields.get("region") or fields.get("remote_policy") or ""
    if fields.get("region") and fields.get("remote_policy"):
        geo = f"{fields['region']}, {fields['remote_policy']}"
    if not name or not url:
        return None
    return _row(name, url, geo, "remoteintech")


def companies_from_tarball(blob: bytes) -> list[dict]:
    """Read src/companies/*.md out of a GitHub repository tarball, nothing else is unpacked."""
    import io, tarfile
    out = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            path = m.name.replace("\\", "/")
            if "/src/companies/" not in "/" + path or not path.endswith(".md"):
                continue
            if m.size and m.size > 1_000_000:
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            raw = f.read()
            rec = parse_remoteintech_frontmatter(raw.decode("utf-8", "replace"),
                                                 Path(path).stem)
            if rec:
                out.append(rec)
    return out


def _from_table(rec: dict) -> tuple[str, str, str]:
    name_cell = next((rec[k] for k in rec if re.search(r"name|company", k)), "")
    url_cell = next((rec[k] for k in rec if re.search(r"website|url|site", k)), "")
    geo_cell = next((rec[k] for k in rec if re.search(r"region|geo|location|country", k)), "")
    name, url = name_cell, url_cell
    m = MD_LINK.search(name_cell or "")
    if m:
        name, url = m.group(1), url or m.group(2)
    m = MD_LINK.search(url_cell or "")
    if m:
        url = m.group(2)
        if not name or name == url_cell:
            name = m.group(1)
    if not url:
        m = MD_LINK.search(name_cell or "")
        if m:
            url = m.group(2)
    # leftover markdown links should not stay in the name
    name = MD_LINK.sub(r"\1", name or "").strip() or (name or "").strip()
    return name, (url or "").strip(), re.sub(r"<[^>]+>", "", geo_cell or "").strip()


def _from_bullet(text: str) -> tuple[str, str, str]:
    m = MD_LINK.search(text)
    if not m:
        return "", "", ""
    rest = (text[m.end():] + " " + text[:m.start()]).strip(" -–—|:()")
    rest = re.sub(r"\s+", " ", rest).strip(" -–—|:")
    return m.group(1).strip(), m.group(2).strip(), rest


def _row(name, url, geo, source, **extra) -> dict:
    d = domain_of(url)
    rec = {k: "" for k in CSV_FIELDS}
    rec.update(name=name.strip(), domain=d, source=source, geo=(geo or "").strip(),
               company_url=url.strip() or (f"https://{d}" if d else ""), status=STATUS_UNKNOWN,
               fail_count="0", n_jobs="")
    rec.update(extra)
    if rec.get("ats") and rec.get("slug"):
        rec["board_url"] = rec.get("board_url") or public_board_url(rec["ats"], rec["slug"])
    return rec


def parse_yc(payload) -> list[dict]:
    """YC catalog: a JSON list, or an object with a companies/results list."""
    if isinstance(payload, str):
        payload = json.loads(payload or "[]")
    if isinstance(payload, dict):
        payload = payload.get("companies") or payload.get("results") or payload.get("hits") or []
    rows = []
    for c in payload or []:
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or c.get("company_name") or "").strip()
        url = (c.get("website") or c.get("url") or c.get("site") or "").strip()
        if not name:
            continue
        if not url and c.get("slug"):
            url = f"https://www.ycombinator.com/companies/{c['slug']}"
        geo = c.get("all_locations") or c.get("locations") or c.get("region") or c.get("location") or ""
        if isinstance(geo, list):
            geo = "; ".join(str(x) for x in geo if x)
        tags = c.get("tags") or c.get("industries") or []
        if isinstance(tags, str):
            tags = [tags]
        rec = _row(name, url, str(geo), "yc", **{"_status": c.get("status") or "",
                                                 "_hiring": bool(c.get("isHiring") or c.get("is_hiring")),
                                                 "_tags": [str(t).lower() for t in tags],
                                                 "_locs": str(geo).lower()})
        rows.append(rec)
    return rows


def yc_preferred(rec: dict) -> bool:
    if rec.get("_status") and str(rec["_status"]).lower() not in {"active", "public", ""}:
        return False
    if rec.get("_hiring"):
        return True
    blob = " ".join([rec.get("_locs", ""), " ".join(rec.get("_tags") or []), rec.get("geo", "")])
    return bool(re.search(r"\b(remote|distributed|worldwide|global)\b", blob, re.I))


def drop_private(rec: dict) -> dict:
    return {k: rec.get(k, "") for k in CSV_FIELDS}


def merge_rows(*groups, target: int = 1000) -> list[dict]:
    """Dedup by domain (else normalised name). Later groups fill empty fields of an earlier row."""
    by: "OrderedDict[str, dict]" = OrderedDict()

    def name_key(r):
        return "name:" + re.sub(r"[^a-z0-9]+", "", (r.get("name") or "").lower())

    def key(r):
        d = r.get("domain") or ""
        if d:
            return "domain:" + d
        if r.get("ats") and r.get("slug"):
            return f"ats:{r['ats']}:{r['slug']}"
        return name_key(r)

    for group in groups:
        for r in group:
            if not r.get("name"):
                continue
            k = key(r)
            if k in by:
                cur = by[k]
                for field in CSV_FIELDS:
                    if not cur.get(field) and r.get(field):
                        cur[field] = r[field]
                if r.get("source") and r["source"] not in (cur.get("source") or "").split("+"):
                    cur["source"] = "+".join(x for x in (cur.get("source"), r["source"]) if x)
            else:
                by[k] = dict(r)
    # same company under two keys (ats slug vs website domain): keep one row, prefer a real domain
    by_name = {}
    for k, r in list(by.items()):
        by_name.setdefault(name_key(r), []).append(k)
    for keys in by_name.values():
        if len(keys) < 2:
            continue
        def score(k, _by=by):
            r = _by[k]
            return (1 if r.get("domain") else 0, 1 if r.get("ats") and r.get("slug") else 0)
        primary = max(keys, key=score)
        for k in keys:
            if k == primary:
                continue
            r = by.pop(k, None)
            if not r:
                continue
            cur = by[primary]
            for field in CSV_FIELDS:
                if not cur.get(field) and r.get(field):
                    cur[field] = r[field]
            if r.get("source") and r["source"] not in (cur.get("source") or "").split("+"):
                cur["source"] = "+".join(x for x in (cur.get("source"), r["source"]) if x)
    out = list(by.values())
    _ = target
    return out


def known_from_ats(path: Path | None = None) -> list[dict]:
    p = path or (ROOT / "targets" / "ats_companies.json")
    if not p.is_file():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for c in d.get("companies") or []:
        if not c.get("slug") or c.get("ats") not in A.PARSE:
            continue
        board = public_board_url(c["ats"], c["slug"])
        rows.append(_row(c.get("name") or c["slug"], "", "", "ats_companies",
                         ats=c["ats"], slug=c["slug"], n_jobs=str(c.get("jobs_at_check") or ""),
                         last_ok=c.get("checked") or d.get("checked") or "", status=STATUS_UNKNOWN,
                         board_url=board, company_url=""))
    return rows


def jetbrains_seed() -> dict:
    return _row("JetBrains", "https://www.jetbrains.com/", "Belgrade, Serbia", "t038-seed",
                ats="greenhouse", slug="jetbrains")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rec = drop_private(r)
            rec["fail_count"] = str(rec.get("fail_count") or "0")
            w.writerow(rec)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return [{k: (r.get(k) or "") for k in CSV_FIELDS} for r in csv.DictReader(f)]


def candidates_for(row: dict) -> list[dict]:
    """Probe list for one company: known slug first, then name/domain variants."""
    seen, out = set(), []
    if row.get("slug"):
        out.append({"name": row["name"], "slug": row["slug"], "ats": row.get("ats") or ""})
        seen.add(row["slug"])
    for s in slug_variants(row.get("name") or "", row.get("domain") or ""):
        if s not in seen:
            out.append({"name": row["name"], "slug": s, "ats": row.get("ats") or ""})
            seen.add(s)
    return out


def apply_probe(row: dict, state: dict, checked: str) -> dict:
    """Fill ATS fields from probe state: first 200 for this company's candidate slugs."""
    rec = dict(row)
    for cand in candidates_for(row):
        for ats in ([cand["ats"]] + [a for a in A.PARSE if a != cand["ats"]] if cand.get("ats") in A.PARSE
                    else list(A.PARSE)):
            hit = state.get(f"{ats}:{cand['slug']}")
            if hit and hit.get("status") == 200:
                rec["ats"], rec["slug"] = ats, cand["slug"]
                rec["board_url"] = public_board_url(ats, cand["slug"])
                rec["last_ok"] = checked
                rec["n_jobs"] = "" if hit.get("jobs") is None else str(hit.get("jobs"))
                rec["status"] = STATUS_LIVE
                rec["fail_count"] = "0"
                return rec
    def slug_done(slug: str) -> bool:
        return any(state.get(f"{a}:{slug}", {}).get("status") == 200 for a in A.PARSE) or all(
            f"{a}:{slug}" in state for a in A.PARSE)

    slugs = [c["slug"] for c in candidates_for(row)]
    if slugs and all(slug_done(s) for s in slugs):
        rec["status"] = STATUS_NONE
    return rec


def liveness_update(rows: list[dict], state: dict, checked: str, fails_to_sleep: int = FAILS_TO_SLEEP) -> list[dict]:
    """Re-check already known slugs (daily). Three misses in a row -> sleep."""
    out = []
    for r in rows:
        rec = dict(r)
        ats, slug = rec.get("ats") or "", rec.get("slug") or ""
        if not ats or not slug:
            out.append(rec)
            continue
        hit = state.get(f"{ats}:{slug}")
        if not hit:
            out.append(rec)
            continue
        if hit.get("status") == 200:
            rec["last_ok"] = checked
            rec["n_jobs"] = "" if hit.get("jobs") is None else str(hit.get("jobs"))
            rec["status"] = STATUS_LIVE
            rec["fail_count"] = "0"
            rec["board_url"] = rec.get("board_url") or public_board_url(ats, slug)
        else:
            n = int(rec.get("fail_count") or 0) + 1
            rec["fail_count"] = str(n)
            if n >= fails_to_sleep:
                rec["status"] = STATUS_SLEEP
        out.append(rec)
    return out


def latest_companies_csv(root: Path) -> Path | None:
    hits = sorted((root / "results").glob("*/reports/company_base/**/companies.csv"),
                  key=lambda p: p.stat().st_mtime if p.is_file() else 0)
    return hits[-1] if hits else None


def fetch_text(url: str, get_fn, max_bytes: int = A.MAX_BYTES) -> tuple[int, str]:
    J.assert_allowed(url)
    fn = get_fn or (lambda u: A.capped_get(u, max_bytes=max_bytes))
    return fn(url)


def source_allowed(host: str, path: str, get_fn) -> bool:
    robots_url = ROBOTS.get(host)
    if not robots_url:
        return True
    try:
        status, body = fetch_text(robots_url, get_fn, max_bytes=1 << 16)
    except Exception:  # noqa: BLE001  a missing robots.txt is not a block
        return True
    if status in (None, 404) or not body:
        return True
    return robots_allows(body, path)



def capped_get_bytes(url: str, timeout: float = 30.0, max_bytes: int = A.MAX_BYTES) -> tuple:
    """GET a possibly binary body with the project User-Agent and a byte cap (tarball)."""
    import requests
    J.assert_allowed(url)
    with requests.get(url, timeout=timeout, stream=True, headers={
            "User-Agent": J.USER_AGENT, "Accept": "*/*"}) as r:
        n = r.headers.get("Content-Length")
        if n and n.isdigit() and int(n) > max_bytes:
            return A.TOO_BIG, b""
        buf = bytearray()
        for chunk in r.iter_content(65536):
            buf += chunk
            if len(buf) > max_bytes:
                return A.TOO_BIG, b""
        return r.status_code, bytes(buf)


def fetch_bytes(url: str, get_fn, max_bytes: int = A.MAX_BYTES) -> tuple:
    J.assert_allowed(url)
    if get_fn is not None:
        status, body = get_fn(url)
        if isinstance(body, str):
            body = body.encode("utf-8")
        return status, body
    return capped_get_bytes(url, max_bytes=max_bytes)


def collect_sources(get_fn, target: int = 1000, include_known: bool = True) -> tuple[list[dict], dict]:
    """Pull both catalogs (robots first), merge, fill up to target with remaining YC actives."""
    notes = {"remoteintech": None, "yc": None, "robots": {}}
    ri, yc = [], []
    # Official GitHub tarball of the company markdown pages (one request). Not the official
    # YC API: api.ycombinator.com robots.txt disallows /.
    notes["robots"]["remoteintech"] = "allow"
    last_err = ""
    try:
        status, blob = fetch_bytes(REMOTEINTECH_TAR, get_fn)
    except Exception as e:  # noqa: BLE001
        last_err, status, blob = type(e).__name__, None, b""
    if status == 200 and blob:
        try:
            ri = companies_from_tarball(blob)
            notes["remoteintech"] = {"url": REMOTEINTECH_TAR, "rows": len(ri)}
        except Exception as e:  # noqa: BLE001
            last_err = f"tarball {type(e).__name__}: {e}"
            notes["remoteintech"] = {"error": last_err}
    else:
        last_err = last_err or f"HTTP {status}"
        notes["remoteintech"] = {"error": last_err}
    if not ri:
        for url in REMOTEINTECH_README_URLS:
            try:
                status, body = fetch_text(url, get_fn)
            except Exception as e:  # noqa: BLE001
                last_err, status, body = type(e).__name__, None, ""
            if status == 200 and body:
                ri = parse_remoteintech(body)
                notes["remoteintech"] = {"url": url, "rows": len(ri), "fallback": "readme"}
                break
            last_err = last_err or f"HTTP {status}"
        if not ri:
            notes["remoteintech"] = {"error": last_err or "no company list"}
    if source_allowed("yc-oss.github.io", "/api/companies/all.json", get_fn):
        notes["robots"]["yc"] = "allow"
        try:
            status, body = fetch_text(YC_URL, get_fn)
        except Exception as e:  # noqa: BLE001
            status, body, err = None, "", type(e).__name__
        else:
            err = ""
        if status == 200 and body:
            yc = parse_yc(body if isinstance(body, str) else body.decode("utf-8", "replace"))
            notes["yc"] = {"url": YC_URL, "rows": len(yc)}
        else:
            notes["yc"] = {"error": err or f"HTTP {status}"}
    else:
        notes["robots"]["yc"] = "disallow"
        notes["yc"] = {"error": "robots.txt disallows the catalog"}
    pref = [r for r in yc if yc_preferred(r)]
    rest = [r for r in yc if r not in pref and str(r.get("_status") or "active").lower() in {"active", "public", ""}]
    other = [r for r in yc if r not in pref and r not in rest]
    known = (known_from_ats() + [jetbrains_seed()]) if include_known else []
    merged = merge_rows(known, ri, pref, rest, other, target=target)
    notes["merged"] = len(merged)
    return merged, notes


def sleeping(row: dict) -> bool:
    return (row.get("status") or "") == STATUS_SLEEP


def live_boards(rows: list[dict]) -> list[dict]:
    return [{"ats": r["ats"], "slug": r["slug"], "name": r["name"]}
            for r in rows if r.get("status") == STATUS_LIVE and r.get("ats") in A.PARSE and r.get("slug")
            and not sleeping(r)]
