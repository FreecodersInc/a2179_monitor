#!/usr/bin/env python3
"""T-033 / EXP-017 (H-40): once a day collect the newest lots of the Limundo and KupindO category pages.

Their own search is disallowed by robots.txt, so only category pages listed in the request are read,
each one only after robots.txt allows it, with the monitor's Fetcher (6..9 s per host, no retry, a 403
or challenge stops that host for the day). Kept per lot: id, url, title, price, currency, condition,
category and the A2179 rule match. Seller data and page HTML are never written.

  python eval/used_sources.py --request ci/requests/used_sources_daily.json --out reports/used_sources/<date>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from a2179.fetch import DEFAULT_UA, Fetcher  # noqa: E402
from a2179.obs import Metrics  # noqa: E402
from a2179.parsers import parse_used_listing, used_site  # noqa: E402
from a2179.rules import classify  # noqa: E402
from a2179.target import load_target  # noqa: E402
from source_probe import links_of, verdict  # noqa: E402


def collect(pages, fetcher, ua=DEFAULT_UA, target="a2179_screen", budget_s=240.0, clock=time.monotonic):
    tg = load_target(target)
    t0, robots, dead = clock(), {}, set()
    log, rows, seen = [], [], set()
    for u in pages:
        p = urlparse(u)
        base = f"{p.scheme}://{p.netloc}"
        rec = {"url": u}
        if used_site(u) is None:
            rec.update(verdict="skipped", why="not a Limundo or KupindO page")
        elif base in dead:
            rec.update(verdict="skipped", why="host refused earlier today")
        elif clock() - t0 > budget_s:
            rec.update(verdict="skipped", why="time budget")
        else:
            if base not in robots:
                st, txt = fetcher.get(base + "/robots.txt", retry_403=0)
                rp = None
                if verdict(st, txt) == "ok" and "<html" not in txt[:500].lower():
                    rp = urllib.robotparser.RobotFileParser()
                    rp.parse(txt.splitlines())
                robots[base] = rp
            rp = robots[base]
            if rp is None:
                rec.update(verdict="blocked", why="robots.txt not readable")
                dead.add(base)
            elif not rp.can_fetch(ua, u):
                rec.update(verdict="disallowed", why="robots.txt disallows this path")
            else:
                st, txt = fetcher.get(u, retry_403=0)
                rec.update(status=st, verdict=verdict(st, txt))
                if rec["verdict"] != "ok":
                    dead.add(base)
                else:
                    got = parse_used_listing(txt, u)
                    rec["lots"] = len(got)
                    rec["category_links"] = [x for x in links_of(txt, u) if used_site(x) and "/kupovina/" not in x
                                             and not x.rsplit("/", 1)[-1][:1].isdigit()][:40]
                    for r in got:
                        if r["id"] in seen:
                            continue
                        seen.add(r["id"])
                        rr = classify(r["title"], "", target=tg)
                        rows.append({**r, "page": u, "rule_match": bool(rr.match), "rule_gate": bool(rr.gate)})
        log.append(rec)
    return rows, log


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", default=str(ROOT / "ci" / "requests" / "used_sources_daily.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    req = json.loads(Path(a.request).read_text(encoding="utf-8"))
    d = float(req.get("min_delay_s", 6))
    f = Fetcher({"min_delay_s": d, "jitter_s": 3, "retries": 0, "retry_403": 0}, Metrics(), logging.getLogger("used_sources"))
    rows, log = collect(req["pages"], f, budget_s=float(req.get("time_budget_s", 240)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lots.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    by = {}
    for r in rows:
        s = by.setdefault(r["source"], {"lots": 0, "rule_match": 0, "rule_gate": 0})
        s["lots"] += 1
        s["rule_match"] += r["rule_match"]
        s["rule_gate"] += r["rule_gate"]
    summary = {"date": time.strftime("%Y-%m-%d", time.gmtime()), "request": req.get("id"), "pages": log, "by_source": by,
               "matches": [{"id": r["id"], "title": r["title"], "price": r["price"], "url": r["url"]} for r in rows if r["rule_match"]]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for p in log:
        print(p["verdict"], p["url"], p.get("status"), p.get("lots", ""))
    print(json.dumps(by))
    return 0


if __name__ == "__main__":
    sys.exit(main())
