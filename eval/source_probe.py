#!/usr/bin/env python3
"""T-033..T-035: polite probe of a new listing source before any collector is written.

For every URL of the request: read the host's robots.txt (5 MiB cap), check that our user agent may
fetch the path, and only then GET it once with the monitor's own Fetcher (per host delay, no retry
on 403). A challenge page (Cloudflare «Just a moment», JS challenge) or 401/403/429 is recorded as
blocked and never worked around. Only status, size, <title>, search form actions and verdicts are written. Page HTML only with
"save_html": true in the request, as parser fixtures (like the shop snapshots in tests/fixtures).

  python eval/source_probe.py --request ci/requests/probe_sources.json --out reports/source_probe/<id>/probe.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from a2179.fetch import DEFAULT_UA, Fetcher  # noqa: E402
from a2179.obs import Metrics  # noqa: E402

CHALLENGE = re.compile(r"just a moment|cf-chl|challenge-platform|captcha|access denied|_0x[0-9a-f]{4}", re.I)


def verdict(status, text: str) -> str:
    if status == "error":
        return "error"
    if status in (401, 403, 429) or (isinstance(status, int) and status >= 500):
        return "blocked"
    if CHALLENGE.search(text[:20000]):
        return "blocked"
    return "ok" if isinstance(status, int) and status < 400 else "error"


def title_of(text: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip()[:120] if m else ""


def forms_of(text: str) -> list[dict]:
    """Search forms only (action + input names), so the next request can use the site's own search URL."""
    out = []
    for m in re.finditer(r"<form\b([^>]*)>(.*?)</form>", text, re.I | re.S):
        act = re.search(r'action\s*=\s*["\']([^"\']*)', m.group(1), re.I)
        names = re.findall(r'<input[^>]*\bname\s*=\s*["\']([^"\']+)', m.group(2), re.I)
        if act and names:
            out.append({"action": act.group(1)[:200], "inputs": names[:40]})
    return out[:6]


LINK_RX = re.compile(r"lapto|racunar|kompjuter|apple|macbook|delovi", re.I)


def links_of(text: str, base: str) -> list[str]:
    """Category links on a home page (laptops, computers, Apple, parts), to find listing pages robots allow."""
    from urllib.parse import urljoin
    out = []
    for h in re.findall(r'<a\b[^>]*href\s*=\s*["\']([^"\'#]+)', text, re.I):
        if LINK_RX.search(h):
            v = urljoin(base, h)
            if urlparse(v).netloc == urlparse(base).netloc and v not in out:
                out.append(v)
    return out[:80]


def robots_rules(txt: str) -> list[str]:
    """The Disallow / Allow lines of the * group, for the record (public file, no page text)."""
    out, on = [], False
    for line in txt.splitlines():
        l = line.split("#", 1)[0].strip()
        if l.lower().startswith("user-agent:"):
            on = l.split(":", 1)[1].strip() == "*"
        elif on and l.lower().startswith(("disallow:", "allow:", "crawl-delay:")):
            out.append(l)
    return out[:80]


def probe(urls, fetcher, ua=DEFAULT_UA, save_dir: Path | None = None) -> list[dict]:
    robots, out = {}, []
    for u in urls:
        p = urlparse(u)
        base = f"{p.scheme}://{p.netloc}"
        if base not in robots:
            st, txt = fetcher.get(base + "/robots.txt", retry_403=0)
            rp = None
            if verdict(st, txt) == "ok" and "<html" not in txt[:500].lower():
                rp = urllib.robotparser.RobotFileParser()
                rp.parse(txt.splitlines())
            robots[base] = {"status": st, "parsed": rp is not None, "rp": rp, "rules": robots_rules(txt) if rp else []}
        r = robots[base]
        row = {"url": u, "robots_status": r["status"], "robots_parsed": r["parsed"]}
        if r["rules"] and not any(o.get("robots_rules") for o in out if o["url"].startswith(base)):
            row["robots_rules"] = r["rules"]
        if not r["parsed"]:
            row.update(allowed=None, status=None, verdict="blocked", why="robots.txt not readable (blocked or challenge)")
        elif not r["rp"].can_fetch(ua, u):
            row.update(allowed=False, status=None, verdict="disallowed", why="robots.txt disallows this path")
        else:
            st, txt = fetcher.get(u, retry_403=0)
            row.update(allowed=True, status=st, bytes=len(txt), title=title_of(txt), verdict=verdict(st, txt), forms=forms_of(txt))
            row["links"] = links_of(txt, u)
            if save_dir is not None and row["verdict"] == "ok":   # snapshot for parser fixtures (T-033, T-031)
                save_dir.mkdir(parents=True, exist_ok=True)
                name = f"snap_{len(out) + 1:02d}_{p.netloc.removeprefix('www.').split('.')[0]}.html"
                if "kupujemprodajem" in p.netloc and "/oglas/" in u:   # never keep seller data (T-031)
                    from a2179.parsers import sanitize_kp_card_html
                    txt = sanitize_kp_card_html(txt, u)
                (save_dir / name).write_text(txt, encoding="utf-8")
                row["snapshot"] = name
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", default=str(ROOT / "ci" / "requests" / "probe_sources.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    req = json.loads(Path(a.request).read_text(encoding="utf-8"))
    f = Fetcher({"min_delay_s": 6, "jitter_s": 3, "retries": 0, "retry_403": 0}, Metrics(), logging.getLogger("source_probe"))
    rows = probe(req["urls"], f, save_dir=Path(a.out).parent if req.get("save_html") else None)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"id": req.get("id"), "rows": rows}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in rows:
        print(r["verdict"], r["url"], r.get("status"), r.get("title", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
