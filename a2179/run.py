"""One monitor pass: fetch -> parse -> filter -> diff against state -> alerts -> metrics."""
import logging
import sys
import time
from urllib.parse import quote_plus, urlparse

from .alerts import Alerter, format_event
from .config import data_dir
from .target import target_of
from .fetch import Fetcher
from .matcher import Matcher
from .obs import Metrics, ev
from . import parallel
from .paginate import Pager
from .parsers import Listing, parse_woo_product
from .tracked import TrackedStore, settings as tracked_settings
from .prices import target_price_usd, to_usd
from .embedder import load_similarity
from .similarity import TfidfSim
from .state import State


def _host(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


def _product(metrics, log, url, st, l, kind="shop_product"):
    """Bookkeeping for one product page (configured shop product or tracked card)."""
    out = []
    if st == 200:
        if l:
            out.append(l)
            metrics.inc(f"parse.{kind}.ok")
        else:
            metrics.inc(f"parse.{kind}.empty")
    elif st == 404:
        metrics.inc(f"parse.{kind}.gone")
        ev(log, "product_gone", logging.WARNING, url=url, tracked=kind == "tracked")
    return out


def _kp_result(metrics, log, kp_status, items, pages_url, nbytes):
    metrics.inc("parse.kp.items", len(items))
    if items:
        return "ok"
    if kp_status != "ok":
        ev(log, "kp_no_items", logging.WARNING, url=pages_url, bytes=nbytes)
        return "ok_but_empty"
    return kp_status


def collect(cfg, fetcher, metrics, log, deadline=None, store=None):
    """Fetch all sources (sequential pass). Returns (listings, kp_status).
    Searches are followed page by page (paginate.Pager). Tracked cards (store) are fetched
    like shop products and their price/stock is written back to the store."""
    src = cfg["sources"]
    out = []

    def time_left():
        return deadline is None or time.monotonic() < deadline

    products = [("shop_product", u) for u in src.get("shop_products", [])]
    have = {Listing("", u, "").key for _, u in products}
    products += [("tracked", u) for k, u in (store.active() if store is not None else []) if k not in have]
    for kind, url in products:
        if not time_left():
            metrics.notes.append("time budget reached, skipped remaining sources")
            break
        st, html = fetcher.get(url)
        l = parse_woo_product(html, url, _host(url)) if st == 200 else None
        out.extend(_product(metrics, log, url, st, l, kind))
        if kind == "tracked":
            store.record(Listing("", url, "").key, st, l)

    def follow(kind, url):
        pager = Pager(kind, url, cfg)
        while pager.url and time_left():
            st, html = fetcher.get(pager.url)
            pager.feed(st, html)
        pager.note_metrics(metrics)
        return pager

    for url in src.get("shop_searches", []):
        if not time_left():
            break
        pg = follow("shop_search", url)
        if pg.status == 200:
            metrics.inc("parse.shop_search.items", len(pg.items))
            out.extend(pg.items)
    kp_status = "skipped"
    for q in src.get("kp_keywords", []):
        if not time_left():
            break
        url = src["kp_search_url"].format(q=quote_plus(q))
        pg = follow("kp", url)
        out.extend(pg.items)
        if pg.blocked:
            if pg.items:
                metrics.inc("parse.kp.items", len(pg.items))
            kp_status = f"blocked({pg.last_status})"
            metrics.inc("kp.blocked")
            ev(log, "kp_blocked", logging.WARNING, url=url, status=pg.last_status, page=pg.pages)
            break  # be polite: do not hammer a site that is refusing us
        kp_status = _kp_result(metrics, log, kp_status, pg.items, url, pg.bytes)
    return _dedupe(out), kp_status


def _dedupe(out):
    # same listing from search + product page: keep the product page (has stock info)
    best = {}
    for l in out:
        if l.key not in best or (best[l.key].in_stock is None and l.in_stock is not None):
            best[l.key] = l
    return list(best.values())


def open_tracked(cfg, ddir, dry_run=False):
    """TrackedStore for this pass, or None when tracked.enabled is false."""
    t = tracked_settings(cfg)
    if not t.get("enabled", True):
        return None
    return TrackedStore(ddir / t.get("file", "tracked.shelve"), read_only=dry_run,
                        drop_after_misses=t.get("drop_after_misses", 3))


def track_matches(store, matched, cfg, metrics, log):
    """Shop cards from search results that matched (regex or by meaning) join the tracked set.
    KP ads are not tracked as pages (no verified KP ad page parser), the paginated KP search
    covers them."""
    if store is None:
        return []
    shop = {Listing("", u, "").key for u in cfg["sources"].get("shop_products", [])}
    cand = [l for l, _, _ in matched if l.source != "kupujemprodajem" and l.key not in shop]
    added = store.add(cand)
    if added:
        metrics.inc("tracked.added", len(added))
        ev(log, "tracked_added", keys=added)
    return added


def collect_parallel(cfg, metrics, log, time_budget_s=None, workers=None, max_concurrency=None, store=None):
    """Same result as collect(), but fetched and parsed by parallel.fetch_all (processes + asyncio).
    Raises parallel.ParallelUnavailable when aiohttp is missing."""
    results = parallel.fetch_all(cfg, metrics, log, time_budget_s, workers, max_concurrency, store)
    out = []
    if any(r["skipped"] == "time" for r in results):
        metrics.notes.append("time budget reached, skipped remaining sources")
    for r in results:
        if r["kind"] == "kp" or r["skipped"]:
            continue
        if r["kind"] in ("shop_product", "tracked"):
            out.extend(_product(metrics, log, r["url"], r["status"], (r["items"] or [None])[0], r["kind"]))
        else:
            if r["status"] == 200:
                metrics.inc("parse.shop_search.items", len(r["items"]))
                out.extend(r["items"])
    kp = [r for r in results if r["kind"] == "kp" and r["skipped"] != "time"]
    kp_status = "skipped"
    first_block = next((r for r in kp if r["blocked"]), None)
    for r in kp:
        if r["blocked"] or r["skipped"] == "kp_blocked":
            # as in collect(): KP refused us, nothing after the refusal counts in this pass
            out.extend(r["items"])  # pages fetched before the refusal
            if r["items"]:
                metrics.inc("parse.kp.items", len(r["items"]))
            b = first_block or r
            kp_status = f"blocked({b['last_status']})"
            metrics.inc("kp.blocked")
            ev(log, "kp_blocked", logging.WARNING, url=b["url"], status=b["status"], page=b["pages"])
            break
        kp_status = _kp_result(metrics, log, kp_status, r["items"], r["url"], r["bytes"])
        out.extend(r["items"])
    return _dedupe(out), kp_status


def prepare_similarity(cfg, log, no_embed=False, require_embedder=False, verbose=False):
    """(sim, mode). Raises EmbedderUnavailable only when the embedder is required."""
    if no_embed and not require_embedder:
        t = target_of(cfg)
        return TfidfSim(t.positive, t.negative), "tfidf"
    sim, mode, _err = load_similarity(cfg, log, require=require_embedder, verbose=verbose)
    if mode == "embed":
        th = (cfg.get("thresholds") or {}).get("embed") or {}
        if not th.get("calibrated") or (th.get("model") and th.get("model") != sim.model_name):
            ev(log, "embed_thresholds_uncalibrated", logging.WARNING, thresholds=th, model=sim.model_name)
            print("[embedder] пороги для embed не откалиброваны под эту модель: запусти "
                  "python eval/run_eval.py --write-thresholds (пропусков не будет, regex работает "
                  "всегда, но лишних или недостающих подсказок от эмбеддера может быть больше)")
    return sim, mode


def run_once(cfg, log, dry_run=False, seed=False, no_embed=False, fetcher=None, time_budget_s=None,
             require_embedder=False, similarity=None, workers=None, max_concurrency=None):
    """workers=0 (or a fetcher passed in, as the tests do): old sequential pass with requests.
    Otherwise parallel: `workers` processes (default 3), at most `max_concurrency` GET in flight
    in total (default 10). Both default to cfg['parallel']."""
    metrics = Metrics()
    ddir = data_dir(cfg)
    sim, mode = similarity or prepare_similarity(cfg, log, no_embed, require_embedder)
    metrics.notes.append(f"filter=regex+{mode}")
    target = target_of(cfg)
    options = {m.accept_option: cfg.get(m.accept_option, True) for m in target.rules.models if m.accept_option}
    matcher = Matcher(sim, mode, cfg.get("thresholds"), cfg.get("accept_a1932", True), target, options)
    store = open_tracked(cfg, ddir, dry_run)
    n_workers = parallel.settings(cfg, workers, max_concurrency)[0]
    listings = None
    if fetcher is None and n_workers >= 1:
        try:
            listings, kp_status = collect_parallel(cfg, metrics, log, time_budget_s, workers, max_concurrency,
                                                   store)
        except parallel.ParallelUnavailable as e:
            ev(log, "parallel_unavailable", logging.WARNING, error=str(e))
            print("[parallel] aiohttp не установлен (pip install -r requirements.txt), "
                  "этот проход идёт последовательно", file=sys.stderr)
    if listings is None:
        fetcher = fetcher or Fetcher(cfg.get("http"), metrics, log)
        if fetcher.metrics is not metrics:
            fetcher.metrics = metrics
        deadline = time.monotonic() + time_budget_s if time_budget_s else None
        listings, kp_status = collect(cfg, fetcher, metrics, log, deadline, store)
    metrics.notes.append(f"kp={kp_status}")
    metrics.inc("listings.seen", len(listings))

    state = State(ddir / "state.sqlite", read_only=dry_run)
    rates, target_usd = cfg["rates"], target_price_usd(cfg)
    drop_pct = float(cfg.get("price_drop_min_pct", 3))
    on = cfg.get("alert_on", {})
    messages, matched = [], []
    for l in listings:
        d = matcher.decide(l.title, l.snippet)
        usd = to_usd(l.price, l.currency, rates) if l.price is not None and l.currency else None
        ev(log, "decision", key=l.key, title=l.title[:120], match=d.match, regex=d.regex, sim=d.sim,
           sim_pos=round(d.sim_pos, 3), price_usd=usd and round(usd, 2), reasons=d.reasons)
        if not d.match:
            state.upsert(l, usd, False)
            continue
        metrics.inc("listings.matched")
        matched.append((l, usd, d))
        prev = state.get(l.key)
        kinds = []
        if prev is None or not prev["matched"]:
            if on.get("new", True):
                kinds.append("new")
        elif usd is not None and prev["price_usd"] and usd < prev["price_usd"] * (1 - drop_pct / 100):
            if on.get("price_drop", True):
                kinds.append("price_drop")
        if (usd is not None and usd <= target_usd and on.get("below_target", True)
                and (prev is None or prev["last_alert_price_usd"] is None or usd < prev["last_alert_price_usd"])):
            kinds.append("below_target")
        alerted = usd if "below_target" in kinds else None
        if kinds and not seed:
            extra = "" if d.regex else " · (похоже по смыслу, проверь)"
            tier = target.tier(d.model)
            if tier is not None and tier.alert_note:
                extra += f" · {tier.alert_note}"
            msg = format_event(kinds[0], l, usd, extra)
            if len(kinds) > 1:
                names = {"price_drop": "цена упала", "below_target": "ниже цели"}
                msg = msg.replace("\n", f" + {', '.join(names.get(k, k) for k in kinds[1:])}\n", 1)
            messages.append(msg)
            for k in kinds:
                metrics.inc(f"events.{k}")
                state.log_alert(l.key, k, msg)
        state.upsert(l, usd, True, alerted)
    track_matches(store, matched, cfg, metrics, log)
    if store is not None:
        n = len(store.active())
        metrics.inc("tracked.active", n)
        metrics.notes.append(f"tracked={n}")

    # KP blocked for many runs in a row: tell the user once
    if not dry_run:
        n = int(state.meta_get("kp_block_streak", 0))
        n = n + 1 if kp_status.startswith("blocked") else 0
        state.meta_set("kp_block_streak", n)
        if n == int(cfg.get("kp_block_alert_after_runs", 6)) and not seed:
            messages.append(f"⚠️ KupujemProdajem не отдаёт страницы {n} запусков подряд (статус {kp_status}). "
                            "Магазины проверяются как обычно.")
    alerter = Alerter(cfg, ddir, log, dry_run=dry_run)
    res = alerter.send(messages, metrics) if not seed else {}
    if not alerter.tg and messages and not dry_run:
        metrics.notes.append("telegram not configured: console + alerts.log only")
    state.close()
    snap = metrics.write(ddir, dry_run)
    ev(log, "run_done", **{"matched": len(matched), "alerts": len(messages), "kp": kp_status,
                           "duration_s": snap["duration_s"], "sent": res})
    return {"listings": listings, "matched": matched, "messages": messages, "kp": kp_status,
            "metrics": snap, "mode": mode}
