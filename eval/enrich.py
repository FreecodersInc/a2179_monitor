#!/usr/bin/env python3
"""Grow eval/labeled.jsonl with REAL listings. Transparent pre-labels, a human decides.

  python eval/enrich.py collect                 # KP + shop searches for the built-in query set
  python eval/enrich.py collect --dry-run       # only print what would be fetched
  python eval/enrich.py collect --offline-html DIR   # parse saved pages instead of the network
  python eval/enrich.py merge --reviewed eval/candidates_<date>.jsonl   # or the .csv

collect: every query goes to KupujemProdajem (sources.kp_search_url) and, for the base
queries, to every shop search of config (the `s=` parameter is replaced). Pages are fetched
by the monitor's own machinery: parallel.fetch_all (shared limiter, per host pacer, retries,
pagination) or, with --workers 0 / without aiohttp, Fetcher + Pager in sequence. KP stops for
the whole run at the first refusal, as in the monitor.
Rows already in labeled.jsonl (same url key, or same normalised title) are dropped.

Every row gets a pre-label from rules only, no LLM:
  rules     a2179/rules.classify (the monitor's regex: model gate, part words, exclusions)
  features  the word lists of eval/sweep.py (display words +1, each other part / accessory /
            service / wanted group -1), matched as stems here so Serbian case endings count too
            (sarke -> sark*, matches "šarkama"; kamera -> kamer*, matches "kamerom")
  intent    sell | seek | service | unknown, from fixed word lists below
  matcher   the monitor's hybrid score (regex OR tfidf with config thresholds)
and `reason`, `flags`, `needs_review: true`, `reviewed: false`. Sorted for review:
  0 disagree   rules and matcher (or rules and features / intent) say different things
  1 hard_neg   on topic (A2179 compatible model, display word or high similarity) but auto label 0
  2 positive   auto label 1, nothing odd
  3 rest

merge: appends only rows with reviewed true (and label 0 or 1) to labeled.jsonl with
source "real", new ids after the last one, and skips duplicates. Nothing is ever removed.
"""
import argparse
import csv
import gzip
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlunparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from a2179 import parallel, prototypes  # noqa: E402
from a2179.config import data_dir, load_config  # noqa: E402
from a2179.fetch import Fetcher  # noqa: E402
from a2179.matcher import Matcher  # noqa: E402
from a2179.normalize import normalize  # noqa: E402
from a2179.obs import Metrics, setup_logging  # noqa: E402
from a2179.paginate import Pager  # noqa: E402
from a2179.parsers import Listing, kp_is_blocked, parse_kp_search, parse_woo_search  # noqa: E402
from a2179.rules import classify  # noqa: E402
from a2179.similarity import TfidfSim  # noqa: E402
from sweep import F_DISPLAY, F_OTHER  # noqa: E402

DATE = time.strftime("%Y-%m-%d")
LABELED = ROOT / "eval" / "labeled.jsonl"

# Base queries also go to the shops (small catalogues, a broad query already lists everything).
BASE_QUERIES = ["macbook air a2179", "a2179", "macbook air 2020", "macbook air 13 2020 intel"]
PART_QUERIES = [f"a2179 {p}" for p in (
    "ekran", "displej", "lcd", "matrica", "sarke", "srafovi", "kuler", "ventilator", "staklo", "folija",
    "punjac", "tastatura", "baterija", "maska", "futrola", "kabl", "flet kabl", "maticna ploca", "kuciste",
    "donji poklopac", "kamera", "zvucnik", "touchpad")] + [
    "macbook air 2020 ekran", "macbook air 2020 lcd", "macbook air 2020 sarke", "macbook air 2020 tastatura"]
INTENT_QUERIES = ["kupujem macbook air", "trazim ekran macbook air", "trazim macbook air a2179",
                  "servis macbook air", "popravka macbook air", "zamena ekrana macbook air",
                  "macbook air delovi", "macbook air za delove", "macbook air neispravan",
                  "prodajem macbook air 2020", "екран macbook air", "дисплеј macbook air"]
DEFAULT_QUERIES = BASE_QUERIES + PART_QUERIES + INTENT_QUERIES


def _rx(words):
    return re.compile(r"\b(?:" + "|".join(sorted({re.escape(normalize(w)) for w in words}, key=len, reverse=True))
                      + r")\b")


INTENT = {  # first hit in this order wins: a "kupujem" ad that mentions "servis" is a buyer
    "seek": _rx(["kupujem", "trazim", "potrebno", "potreban", "potrebna", "treba mi", "otkup", "otkupljujem",
                 "куплю", "ищу", "купујем", "тражим"]),
    "service": _rx(["servis", "zamena", "zamenu", "popravka", "popravak", "ugradnja", "ugradnju", "repair",
                    "service", "ремонт", "замена", "сервис", "поправка"]),
    "sell": _rx(["prodajem", "prodaja", "na prodaju", "prodaje se", "nov", "novo", "nova", "polovan", "polovno",
                 "original", "garancija", "продајем", "продаю", "продам", "нов", "ново"]),
}


def _stem_rx(words):
    stems = set()
    for w in words:
        n = normalize(w)
        stems.add(n[:-1] if len(n) > 4 and n[-1] in "aeiouy" and " " not in n else n)
    return re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, stems), key=len, reverse=True)) + r")\w*")


RX_DISPLAY = _stem_rx(F_DISPLAY)
RX_OTHER = {k: _stem_rx(v) for k, v in F_OTHER.items()}


def features(title, snippet=""):
    """(value in [-1, 1], groups). Display words from title + snippet, other groups from the title."""
    tt, t = normalize(title), normalize(f"{title} {snippet}")
    disp = 1 if RX_DISPLAY.search(t) else 0
    other = [k for k, rx in RX_OTHER.items() if rx.search(tt)]
    return max(-1.0, min(1.0, disp - len(other))), (["display"] if disp else []) + other


def intent_of(title, snippet=""):
    tt, t = normalize(title), normalize(f"{title} {snippet}")
    for k in ("seek", "service"):
        if INTENT[k].search(tt):
            return k
    for k in ("seek", "service", "sell"):
        if INTENT[k].search(t):
            return k
    return "unknown"


def lang_of(text):
    if re.search(r"[ђћџљњјЂЋЏЉЊЈ]", text):
        return "sr_cyr"
    if re.search(r"[а-яА-ЯёЁ]", text):
        return "ru"
    return "sr"


def kind_guess(rr, groups, intent):
    reasons = " ".join(rr.reasons)
    if rr.model == "none":
        return "other_model"
    if intent == "seek":
        return "wanted"
    if intent == "service" and not rr.match:
        return "service"
    if rr.match:
        if "donor=" in reasons:
            return "donor"
        return "display"
    other = [g for g in groups if g != "display"]
    if any(g in ("case_accessory", "protector") for g in other):
        return "accessory"
    if other:
        return "other_part"
    if "laptop_specs" in reasons:
        return "whole_laptop"
    return "other"


# ------------------------------------------------------------------ pre-label
class Labeler:
    def __init__(self, cfg):
        self.matcher = Matcher(TfidfSim(), "tfidf", cfg.get("thresholds"), cfg.get("accept_a1932", True))
        self.accept_a1932 = cfg.get("accept_a1932", True)

    def label(self, l: Listing, query: str) -> dict:
        rr = classify(l.title, l.snippet, self.accept_a1932)
        fval, groups = features(l.title, l.snippet)
        intent = intent_of(l.title, l.snippet)
        d = self.matcher.decide(l.title, l.snippet)
        flags = []
        if d.match != rr.match:
            flags.append("rules_vs_matcher")  # similarity alone says yes
        if rr.match and fval < 0:
            flags.append("rules_vs_features")  # regex yes, but other part / accessory words in the title
        if rr.match and intent in ("seek", "service"):
            flags.append(f"rules_vs_intent:{intent}")
        if not rr.match and rr.model in ("strong", "a1932") and fval > 0 and intent == "sell":
            flags.append("display_for_sale_but_no_rule")
        on_topic = rr.model in ("strong", "a1932", "weak") and ("display" in groups or d.sim_pos >= 0.30)
        if not rr.match and on_topic:
            flags.append("hard_negative_candidate")
        disagree = any(f.startswith(("rules_vs", "display_for")) for f in flags)
        priority = 0 if disagree else 1 if "hard_negative_candidate" in flags else 2 if rr.match else 3
        reason = "; ".join(rr.reasons + [f"features={fval:+.0f} {','.join(groups) or '-'}", f"intent={intent}",
                                         f"matcher={'yes' if d.match else 'no'} pos={d.sim_pos:.2f} neg={d.sim_neg:.2f}"])
        return {"title": l.title, "snippet": l.snippet, "label": int(rr.match), "auto_label": int(rr.match),
                "kind": kind_guess(rr, groups, intent), "lang": lang_of(l.title), "needs_review": True,
                "reviewed": False, "priority": priority, "flags": flags, "reason": reason,
                "rules_match": rr.match, "model": rr.model, "features": fval, "feature_groups": groups,
                "intent": intent, "matcher_match": d.match, "sim_pos": round(d.sim_pos, 3),
                "sim_neg": round(d.sim_neg, 3), "url": l.url, "key": l.key, "site": l.source, "query": query,
                "price": l.price, "currency": l.currency, "location": l.location}


# ------------------------------------------------------------------ collect
def shop_url(template: str, q: str) -> str:
    u = urlparse(template)
    qs = [(k, q if k == "s" else v) for k, v in parse_qsl(u.query, keep_blank_values=True)]
    if not any(k == "s" for k, _ in qs):
        qs.append(("s", q))
    return urlunparse(u._replace(query=urlencode(qs)))


def plan(cfg, queries, shops="base"):
    """[(kind, url, query)] in fetch order: shops first (fast), then KP."""
    src = cfg["sources"]
    shop_q = [] if shops == "none" else [q for q in queries if q in BASE_QUERIES] if shops == "base" else list(queries)
    out = [("shop_search", shop_url(t, q), q) for q in shop_q for t in src.get("shop_searches", [])]
    out += [("kp", src["kp_search_url"].format(q=quote_plus(q)), q) for q in queries]
    seen, uniq = set(), []
    for t in out:
        if t[1] not in seen:
            seen.add(t[1])
            uniq.append(t)
    return uniq


def collect_online(cfg, tasks, metrics, log, workers=None, max_concurrency=None, time_budget_s=None, fetcher=None):
    """[(Listing, query)] plus a summary. Uses the monitor's fetch machinery unchanged."""
    by_url = {u: (k, q) for k, u, q in tasks}
    cfg2 = dict(cfg)
    cfg2["sources"] = {"shop_products": [], "kp_search_url": cfg["sources"]["kp_search_url"],
                       "shop_searches": [u for k, u, _ in tasks if k == "shop_search"], "kp_keywords": []}
    kp_urls = [u for k, u, _ in tasks if k == "kp"]
    summary = Counter()
    out = []
    w = parallel.settings(cfg, workers, max_concurrency)[0]
    if fetcher is None and w >= 1:
        # fetch_all builds KP URLs from kp_keywords: hand it the queries in plan order
        cfg2["sources"]["kp_keywords"] = [by_url[u][1] for u in kp_urls]
        try:
            res = parallel.fetch_all(cfg2, metrics, log, time_budget_s, workers, max_concurrency, None)
            for r in res:
                kind, q = by_url.get(r["url"], (r["kind"], ""))
                summary[f"{kind}:{'skipped_' + r['skipped'] if r['skipped'] else 'blocked' if r['blocked'] else r['status']}"] += 1
                out += [(l, q) for l in r["items"]]
            return out, summary
        except parallel.ParallelUnavailable:
            print("[enrich] aiohttp не установлен, качаю последовательно", file=sys.stderr)
    f = fetcher or Fetcher(cfg.get("http"), metrics, log)
    kp_blocked = False
    deadline = time.monotonic() + time_budget_s if time_budget_s else None
    for kind, url, q in tasks:
        if kind == "kp" and kp_blocked:
            summary["kp:skipped_kp_blocked"] += 1
            continue
        if deadline and time.monotonic() > deadline:
            summary[f"{kind}:skipped_time"] += 1
            continue
        pg = Pager(kind, url, cfg)
        while pg.url:
            st, html = f.get(pg.url)
            pg.feed(st, html)
        pg.note_metrics(metrics)
        summary[f"{kind}:{'blocked' if pg.blocked else pg.status}"] += 1
        out += [(l, q) for l in pg.items]
        if pg.blocked:
            kp_blocked = True
    return out, summary


def _read(p: Path) -> str:
    if p.name.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace").read()
    return p.read_text(encoding="utf-8", errors="replace")


def collect_offline(d: Path):
    """Saved pages: *.html / *.htm / *.html.gz. Optional d/pages.json {"file": {"url", "query"}}.
    Without it: a name starting with kp (or a page with /oglas/ links) is KP, else a WooCommerce
    search of the host named by the file prefix (ifix_search.html -> ifix.rs, ihelp -> ihelp.rs,
    lp or laptop-punjaci -> laptop-punjaci.com)."""
    meta = json.loads((d / "pages.json").read_text(encoding="utf-8")) if (d / "pages.json").exists() else {}
    hosts = {"ifix": "ifix.rs", "ihelp": "ihelp.rs", "lp": "laptop-punjaci.com", "laptop-punjaci": "laptop-punjaci.com"}
    out, summary = [], Counter()
    for p in sorted(d.iterdir()):
        if not re.search(r"\.(html?|html\.gz)$", p.name):
            continue
        html = _read(p)
        m = meta.get(p.name, {})
        url = m.get("url")
        is_kp = (url and "kupujemprodajem" in url) or (not url and (p.name.lower().startswith("kp") or "/oglas/" in html))
        if is_kp:
            url = url or "https://www.kupujemprodajem.com/pretraga?keywords=offline"
            if kp_is_blocked(200, html):
                summary["kp:blocked_page_skipped"] += 1
                continue
            items = parse_kp_search(html, url)
            summary["kp:file"] += 1
        else:
            pre = p.name.split("_")[0].split(".")[0].lower()
            host = hosts.get(pre, pre if "." in p.name.split("_")[0] else f"{pre}.offline")
            url = url or f"https://{host}/?s=offline&post_type=product"
            items = parse_woo_search(html, url, urlparse(url).netloc.lower().removeprefix("www."))
            summary["shop_search:file"] += 1
        out += [(l, m.get("query", f"offline:{p.name}")) for l in items]
    return out, summary


# ------------------------------------------------------------------ dedupe, write
def load_labeled(path: Path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []
    keys = {r["key"] for r in rows if r.get("key")} | {Listing("", r["url"], "").key for r in rows if r.get("url")}
    titles = {normalize(r["title"]) for r in rows}
    return rows, keys, titles


def build_candidates(found, cfg, labeled_path: Path, limit=None):
    _, keys, titles = load_labeled(labeled_path)
    proto = {normalize(t) for t in prototypes.POSITIVE + prototypes.NEGATIVE}
    lab = Labeler(cfg)
    seen, rows, stats = set(), [], Counter()
    for l, q in found:
        nt = normalize(l.title)
        if l.key in seen:
            stats["dup_in_run"] += 1
            continue
        seen.add(l.key)
        if l.key in keys or nt in titles:
            stats["already_labeled"] += 1
            continue
        if nt in proto:
            stats["is_prototype"] += 1  # would leak into the eval set
            continue
        titles.add(nt)  # same title from two shops: keep the first only
        rows.append(lab.label(l, q))
    rows.sort(key=lambda r: (r["priority"], -r["sim_pos"], r["title"]))
    if limit:
        stats["over_limit"] = max(0, len(rows) - limit)
        rows = rows[:limit]
    for i, r in enumerate(rows, 1):
        r["cand_id"] = i
    stats["candidates"] = len(rows)
    stats.update({f"priority_{p}": n for p, n in Counter(r["priority"] for r in rows).items()})
    return rows, stats


CSV_COLS = ["cand_id", "reviewed", "label", "kind", "title", "snippet", "priority", "flags", "reason", "intent",
            "auto_label", "sim_pos", "sim_neg", "site", "price", "currency", "location", "url", "query", "lang"]


def write_out(rows, out: Path, want_csv=False):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    paths = [out]
    if want_csv:
        c = out.with_suffix(".csv")
        with c.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, CSV_COLS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({**r, "flags": " ".join(r["flags"]), "reviewed": ""})
        paths.append(c)
    return paths


# ------------------------------------------------------------------ merge
TRUE = {"1", "true", "yes", "y", "x", "+", "da", "да", "ok"}


def read_reviewed(path: Path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["reviewed"] = str(r.get("reviewed", "")).strip().lower() in TRUE
        return rows
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        v = r.get("reviewed")
        r["reviewed"] = v is True or str(v).strip().lower() in TRUE
    return rows


def merge(reviewed: Path, labeled: Path, dry_run=False):
    rows, keys, titles = load_labeled(labeled)
    nid = max([r["id"] for r in rows] or [0]) + 1
    proto = {normalize(t) for t in prototypes.POSITIVE + prototypes.NEGATIVE}
    add, stats = [], Counter()
    for r in read_reviewed(reviewed):
        if not r["reviewed"]:
            stats["not_reviewed"] += 1
            continue
        try:
            label = int(str(r.get("label")).strip())
        except ValueError:
            label = None
        if label not in (0, 1):
            stats["bad_label"] += 1
            continue
        url = r.get("url") or ""
        key = r.get("key") or (Listing("", url, "").key if url else "")
        nt = normalize(r.get("title", ""))
        if not nt:
            stats["no_title"] += 1
            continue
        if (key and key in keys) or nt in titles:
            stats["duplicate"] += 1
            continue
        if nt in proto:
            stats["is_prototype"] += 1
            continue
        keys.add(key)
        titles.add(nt)
        add.append({"id": nid, "title": r["title"], "snippet": r.get("snippet") or "", "label": label,
                    "lang": r.get("lang") or lang_of(r["title"]), "kind": r.get("kind") or "other",
                    "source": "real", "url": url, "key": key, "query": r.get("query") or "",
                    "added": DATE, "from": reviewed.name})
        nid += 1
    stats["added"] = len(add)
    if add and not dry_run:
        text = labeled.read_text(encoding="utf-8") if labeled.exists() else ""
        if text and not text.endswith("\n"):
            text += "\n"
        tmp = labeled.with_suffix(".jsonl.tmp")
        tmp.write_text(text + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in add), encoding="utf-8")
        tmp.replace(labeled)
    return add, stats


# ------------------------------------------------------------------ CLI
def read_queries(path):
    if not path:
        return list(DEFAULT_QUERIES)
    qs = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()]
    return [q for q in qs if q and not q.startswith("#")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("collect", help="fetch real listings and pre-label them")
    c.add_argument("--config", default=None)
    c.add_argument("--queries", help="file, one query per line (# comments). Default: built-in set")
    c.add_argument("--max-pages", type=int, default=3, help="pages per search (default 3)")
    c.add_argument("--limit", type=int, default=0, help="keep at most N candidates (highest priority first)")
    c.add_argument("--shops", choices=["base", "all", "none"], default="base",
                   help="which queries also go to the shop searches (default: the base queries)")
    c.add_argument("--dry-run", action="store_true", help="print the fetch plan, fetch and write nothing")
    c.add_argument("--offline-html", help="parse saved pages from this folder instead of the network")
    c.add_argument("--out", default=None, help="default eval/candidates_<date>.jsonl")
    c.add_argument("--csv", action="store_true", help="also write the .csv next to it (Excel/LibreOffice)")
    c.add_argument("--labeled", default=str(LABELED))
    c.add_argument("--workers", type=int, default=None)
    c.add_argument("--max-concurrency", type=int, default=None)
    c.add_argument("--time-budget", type=float, default=None, help="seconds")
    m = sub.add_parser("merge", help="append reviewed rows to labeled.jsonl")
    m.add_argument("--reviewed", required=True)
    m.add_argument("--labeled", default=str(LABELED))
    m.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        add, st = merge(Path(a.reviewed), Path(a.labeled), a.dry_run)
        print(f"merge: {dict(st)}{' (dry run, nothing written)' if a.dry_run else ''}")
        for r in add[:20]:
            print(f"  + id {r['id']} label={r['label']} {r['kind']}: {r['title'][:80]}")
        return 0
    if a.cmd != "collect":
        ap.print_help()
        return 2
    cfg = load_config(a.config)
    cfg["pagination"] = {**(cfg.get("pagination") or {}), "max_pages": max(1, a.max_pages)}
    queries = read_queries(a.queries)
    out = Path(a.out) if a.out else ROOT / "eval" / f"candidates_{DATE}.jsonl"
    if a.offline_html:
        if a.dry_run:
            print(f"dry run: would parse {sorted(p.name for p in Path(a.offline_html).iterdir())}")
            return 0
        found, summary = collect_offline(Path(a.offline_html))
    else:
        tasks = plan(cfg, queries, a.shops)
        if a.dry_run:
            print(f"dry run: {len(tasks)} searches x up to {a.max_pages} pages, {len(queries)} queries, nothing fetched")
            for k, u, q in tasks:
                print(f"  {k:12} {q:32} {u}")
            return 0
        log = setup_logging(data_dir(cfg) / "logs")
        metrics = Metrics()
        found, summary = collect_online(cfg, tasks, metrics, log, a.workers, a.max_concurrency, a.time_budget)
        print(f"fetch: requests {metrics.counters.get('fetch.requests', 0)}, errors {metrics.counters.get('fetch.errors', 0)}")
    rows, stats = build_candidates(found, cfg, Path(a.labeled), a.limit or None)
    paths = write_out(rows, out, a.csv)
    print(f"pages: {dict(summary)}")
    print(f"listings: {len(found)} | {dict(stats)}")
    if any(k.startswith("kp:blocked") for k in summary):
        print("KP отказал (403/капча): KP пропущен до следующего запуска, магазины собраны")
    print("written: " + ", ".join(str(p) for p in paths))
    print("next: открой файл, поставь reviewed true (в CSV: 1) и поправь label/kind где надо, потом\n"
          f"  python eval/enrich.py merge --reviewed {paths[-1] if a.csv else paths[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
