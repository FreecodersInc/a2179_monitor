"""Rule-based geo admission filter for de_jobs matches (no ML).

Goal: remote roles that can be taken from Serbia. Only the ATS location string is
available in matches.csv, so verdicts are about location text, not the job body.

Buckets:
  keep  - remote worldwide / Europe / EMEA / Serbia, or plain "Remote"
  check - remote but tied to a European country, or "<city>, Remote" with no
          country limit; may work via B2B/EOR, needs a look at the posting
  cut   - onsite only, or remote locked to US/Canada/India/LatAm/APAC etc.

  python scripts/geo_filter.py reports/2026-11-21/de_jobs/matches.csv
"""
import csv, json, re, sys
from collections import Counter
from pathlib import Path

REMOTE = re.compile(r"remote|anywhere|worldwide|home based|distributed", re.I)
OPEN = re.compile(r"\b(global|worldwide|emea|europe|serbia|belgrade|cet)\b", re.I)
EUROPE = re.compile(r"\b(united kingdom|uk|ireland|poland|portugal|france|spain|germany|italy|"
                    r"netherlands|sweden|denmark|hungary|bulgaria|romania|croatia|london|dublin|"
                    r"paris|berlin|amsterdam|warsaw|lisbon|madrid)\b", re.I)
LOCKED = [
    ("us_canada", re.compile(r"\b(united states|usa|u\.s|us|canada|ontario|north america|americas|"
                             r"toronto|montreal|mississauga|san francisco|new york|seattle|chicago|"
                             r"houston|boston|austin|palo alto|mountain view|santa clara|san jose|"
                             r"california|ca|ny|wa|washington)\b", re.I)),
    ("india", re.compile(r"\b(india|bangalore|bengaluru|mumbai|delhi)\b", re.I)),
    ("latam", re.compile(r"\b(brazil|mexico|latam)\b", re.I)),
    ("apac", re.compile(r"\b(australia|singapore|apac|japan|hong kong)\b", re.I)),
    ("other_country", re.compile(r"\b(ukraine)\b", re.I)),
]


TITLE_LOCK = re.compile(r"\b(uk|us|usa|united states|united kingdom|canada|germany|ireland|spain|sweden|poland)\b", re.I)


def verdict(loc: str, title: str = ""):
    l = loc.strip()
    if not REMOTE.search(l):
        return "cut", "onsite"
    if not l or re.fullmatch(r"\s*remote\s*", l, re.I) or OPEN.search(l):
        if TITLE_LOCK.search(title):
            return "check", "remote_open_title_country"
        return "keep", "remote_open"
    if EUROPE.search(l):
        return "check", "remote_eu_country"
    for name, rx in LOCKED:
        if rx.search(l):
            return "cut", f"remote_{name}"
    return "check", "remote_unclear"


def main(path):
    src = Path(path)
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    out = src.parent / "geo"
    out.mkdir(exist_ok=True)
    buckets = {"keep": [], "check": [], "cut": []}
    reasons = Counter()
    for r in rows:
        b, why = verdict(r.get("location", ""), r.get("title", ""))
        r["geo"], r["geo_reason"] = b, why
        buckets[b].append(r)
        reasons[why] += 1
    cols = list(rows[0].keys()) if rows else []
    for b, rs in buckets.items():
        rs.sort(key=lambda r: -float(r.get("score") or 0))
        with (out / f"{b}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rs)
    with (out / "keep_check.md").open("w", encoding="utf-8") as f:
        for b in ("keep", "check"):
            f.write(f"## {b} ({len(buckets[b])})\n\n| # | score | title | company | location |\n|---|---|---|---|---|\n")
            for i, r in enumerate(buckets[b], 1):
                f.write(f"| {i} | {r['score']} | [{r['title']}]({r['url']}) | {r['company']} | {r['location']} |\n")
            f.write("\n")
    summary = {"source": str(src), "n": len(rows), "keep": len(buckets["keep"]),
               "check": len(buckets["check"]), "cut": len(buckets["cut"]),
               "reasons": dict(reasons.most_common()),
               "note": "location field only; job body (work authorization, timezone) not checked"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/2026-11-21/de_jobs/matches.csv")
