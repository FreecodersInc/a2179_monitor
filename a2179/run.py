"""One monitor pass: fetch -> parse -> filter -> diff against state -> alerts -> metrics."""
import logging
import time
from urllib.parse import quote_plus, urlparse

from .alerts import Alerter, format_event
from .config import data_dir
from .fetch import Fetcher
from .matcher import Matcher
from .obs import Metrics, ev
from .parsers import parse_kp_search, parse_woo_product, parse_woo_search, kp_is_blocked
from .prices import to_usd
from .embedder import load_similarity
from .similarity import TfidfSim
from .state import State


def _host(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


def collect(cfg, fetcher, metrics, log, deadline=None):
    """Fetch all sources. Returns (listings, kp_status)."""
    src = cfg["sources"]
    out = []

    def time_left():
        return deadline is None or time.monotonic() < deadline

    for url in src.get("shop_products", []):
        if not time_left():
            metrics.notes.append("time budget reached, skipped remaining sources")
            break
        st, html = fetcher.get(url)
        if st == 200:
            l = parse_woo_product(html, url, _host(url))
            if l:
                out.append(l)
                metrics.inc("parse.shop_product.ok")
            else:
                metrics.inc("parse.shop_product.empty")
        elif st == 404:
            metrics.inc("parse.shop_product.gone")
            ev(log, "product_gone", logging.WARNING, url=url)
    for url in src.get("shop_searches", []):
        if not time_left():
            break
        st, html = fetcher.get(url)
        if st == 200:
            items = parse_woo_search(html, url, _host(url))
            metrics.inc("parse.shop_search.items", len(items))
            out.extend(items)
    kp_status = "skipped"
    for q in src.get("kp_keywords", []):
        if not time_left():
            break
        url = src["kp_search_url"].format(q=quote_plus(q))
        st, html = fetcher.get(url)
        if kp_is_blocked(st if isinstance(st, int) else 0, html) or st != 200:
            kp_status = f"blocked({st})"
            metrics.inc("kp.blocked")
            ev(log, "kp_blocked", logging.WARNING, url=url, status=st)
            break  # be polite: do not hammer a site that is refusing us
        items = parse_kp_search(html, url)
        metrics.inc("parse.kp.items", len(items))
        if items:
            kp_status = "ok"
        elif kp_status != "ok":
            kp_status = "ok_but_empty"
            ev(log, "kp_no_items", logging.WARNING, url=url, bytes=len(html))
        out.extend(items)
    # same listing from search + product page: keep the product page (has stock info)
    best = {}
    for l in out:
        if l.key not in best or (best[l.key].in_stock is None and l.in_stock is not None):
            best[l.key] = l
    return list(best.values()), kp_status


def prepare_similarity(cfg, log, no_embed=False, require_embedder=False, verbose=False):
    """(sim, mode). Raises EmbedderUnavailable only when the embedder is required."""
    if no_embed and not require_embedder:
        return TfidfSim(), "tfidf"
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
             require_embedder=False, similarity=None):
    metrics = Metrics()
    ddir = data_dir(cfg)
    sim, mode = similarity or prepare_similarity(cfg, log, no_embed, require_embedder)
    metrics.notes.append(f"filter=regex+{mode}")
    matcher = Matcher(sim, mode, cfg.get("thresholds"), cfg.get("accept_a1932", True))
    fetcher = fetcher or Fetcher(cfg.get("http"), metrics, log)
    if fetcher.metrics is not metrics:
        fetcher.metrics = metrics
    deadline = time.monotonic() + time_budget_s if time_budget_s else None
    listings, kp_status = collect(cfg, fetcher, metrics, log, deadline)
    metrics.notes.append(f"kp={kp_status}")
    metrics.inc("listings.seen", len(listings))

    state = State(ddir / "state.sqlite", read_only=dry_run)
    rates, target = cfg["rates"], float(cfg["target_usd"])
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
        if (usd is not None and usd <= target and on.get("below_target", True)
                and (prev is None or prev["last_alert_price_usd"] is None or usd < prev["last_alert_price_usd"])):
            kinds.append("below_target")
        alerted = usd if "below_target" in kinds else None
        if kinds and not seed:
            extra = "" if d.regex else " · (похоже по смыслу, проверь)"
            if d.model == "a1932":
                extra += " · A1932, совместимость уточни у продавца"
            msg = format_event(kinds[0], l, usd, extra)
            if len(kinds) > 1:
                names = {"price_drop": "цена упала", "below_target": "ниже цели"}
                msg = msg.replace("\n", f" + {', '.join(names.get(k, k) for k in kinds[1:])}\n", 1)
            messages.append(msg)
            for k in kinds:
                metrics.inc(f"events.{k}")
                state.log_alert(l.key, k, msg)
        state.upsert(l, usd, True, alerted)

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
