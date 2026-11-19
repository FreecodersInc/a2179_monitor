"""Parallel fetch: N worker processes, each one asyncio loop with one aiohttp ClientSession.

Why: a pass is mostly waiting on the network. Workers fetch and parse at the same time,
so parsing in one coroutine overlaps with other requests waiting on their responses.

Global limit, chosen design: the total `max_concurrency` (default 10) is SPLIT between the
processes, each process gets its own asyncio.Semaphore and its share adds up to exactly the
total (10 over 3 processes: 4 + 3 + 3). No cross-process lock is needed, so a slow or stuck
process can never block the others, and at no moment more than `max_concurrency` GET
requests are in flight in total (retries included). The cost: a process that has run out of
work does not lend its free slots to the others. Workers are capped at `max_concurrency`
so every process gets at least one slot.

Results go back to the main process in the original source order, so dedupe, the KP block
rule, matching, state and alerts behave exactly as in the sequential pass.
"""
import asyncio
import dataclasses
import importlib.util
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import quote_plus, urlparse

from .afetch import AsyncFetcher, make_session
from .obs import Metrics
from .parsers import Listing, kp_is_blocked, parse_kp_search, parse_woo_product, parse_woo_search

DEFAULT_WORKERS = 3
DEFAULT_MAX_CONCURRENCY = 10

_ONE_THREAD = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_KP_BLOCKED = None  # multiprocessing.Event, set in every worker by _init_worker


class ParallelUnavailable(RuntimeError):
    """aiohttp is not installed: the caller falls back to the sequential pass."""


def settings(cfg: dict, workers=None, max_concurrency=None):
    """(workers, max_concurrency, delay_s, start_method) from CLI args over cfg['parallel']."""
    p = cfg.get("parallel") or {}
    w = int(p.get("workers", DEFAULT_WORKERS) if workers is None else workers)
    mc = int(p.get("max_concurrency", DEFAULT_MAX_CONCURRENCY) if max_concurrency is None else max_concurrency)
    return max(0, w), max(1, mc), float(p.get("delay_s", 0) or 0), str(p.get("start_method") or "spawn")


def split_limit(total: int, workers: int) -> list:
    """Per-process semaphore sizes that add up to `total`; never more processes than slots."""
    total = max(1, int(total))
    workers = max(1, min(int(workers), total))
    base, extra = divmod(total, workers)
    return [base + (1 if i < extra else 0) for i in range(workers)]


def build_tasks(cfg: dict) -> list:
    """[(index, kind, url)] in the same order as the sequential collect()."""
    src = cfg["sources"]
    urls = [("shop_product", u) for u in src.get("shop_products", [])]
    urls += [("shop_search", u) for u in src.get("shop_searches", [])]
    urls += [("kp", src["kp_search_url"].format(q=quote_plus(q))) for q in src.get("kp_keywords", [])]
    return [(i, kind, url) for i, (kind, url) in enumerate(urls)]


def _src_host(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


class _BufLog:
    """Logger stand-in for workers: keeps records, the main process writes them to its log."""
    def __init__(self):
        self.records = []

    def log(self, level, event, extra=None):
        self.records.append((time.time(), level, event, dict((extra or {}).get("fields") or {})))


def _init_worker(kp_event):
    global _KP_BLOCKED
    _KP_BLOCKED = kp_event


async def _run_task(fetcher, task, deadline):
    i, kind, url = task
    res = {"i": i, "kind": kind, "url": url, "status": None, "items": [], "skipped": None, "bytes": 0,
           "blocked": False}

    def gate():  # checked when a slot is free, right before the request goes out
        if deadline is not None and time.time() >= deadline:
            return "time"
        if kind == "kp" and _KP_BLOCKED is not None and _KP_BLOCKED.is_set():
            return "kp_blocked"  # be polite: KP already refused us in this pass
        return None

    st, html = await fetcher.get(url, gate=gate)
    if st == "skipped":
        res["skipped"] = html
        return res
    res["status"], res["bytes"] = st, len(html)
    if kind == "shop_product":
        if st == 200:
            l = parse_woo_product(html, url, _src_host(url))
            res["items"] = [l] if l else []
    elif kind == "shop_search":
        if st == 200:
            res["items"] = parse_woo_search(html, url, _src_host(url))
    else:
        if kp_is_blocked(st if isinstance(st, int) else 0, html) or st != 200:
            res["blocked"] = True
            if _KP_BLOCKED is not None:
                _KP_BLOCKED.set()
        else:
            res["items"] = parse_kp_search(html, url)
    res["items"] = [dataclasses.asdict(l) for l in res["items"]]
    return res


async def _worker_async(tasks, limit, http_cfg, delay_s, deadline, metrics, log):
    import aiohttp
    sem = asyncio.Semaphore(limit)
    async with make_session(aiohttp, http_cfg, limit) as session:
        f = AsyncFetcher(http_cfg, metrics, log, session, sem, delay_s=delay_s)
        return await asyncio.gather(*(_run_task(f, t, deadline) for t in tasks))


def worker(idx, tasks, limit, http_cfg, delay_s, deadline):
    """Runs in a worker process. Returns plain data only (picklable)."""
    metrics, log = Metrics(), _BufLog()
    results = asyncio.run(_worker_async(tasks, limit, http_cfg, delay_s, deadline, metrics, log))
    return {
        "worker": idx,
        "results": results,
        "counters": dict(metrics.counters),
        "latency_ms": {h: list(v) for h, v in metrics.latency_ms.items()},
        "status": {h: dict(v) for h, v in metrics.status.items()},
        "log": log.records,
    }


def _merge_metrics(metrics, out):
    for k, v in out["counters"].items():
        metrics.counters[k] += v
    for h, xs in out["latency_ms"].items():
        metrics.latency_ms[h].extend(xs)
    for h, d in out["status"].items():
        for s, n in d.items():
            metrics.status[h][s] += n


def _replay_log(log, out):
    for created, level, event, fields in out["log"]:
        rec = log.makeRecord(log.name, level, "a2179.parallel", 0, event, None, None,
                             extra={"fields": dict(fields, worker=out["worker"])})
        rec.created = created
        log.handle(rec)


def fetch_all(cfg, metrics, log, time_budget_s=None, workers=None, max_concurrency=None):
    """Fetch and parse every source in parallel. Returns task results sorted by source order."""
    tasks = build_tasks(cfg)
    w, mc, delay_s, start_method = settings(cfg, workers, max_concurrency)
    if not tasks:
        return []
    if importlib.util.find_spec("aiohttp") is None:
        raise ParallelUnavailable("aiohttp is not installed")
    limits = split_limit(mc, min(max(1, w), len(tasks)))
    n = len(limits)
    shards = [tasks[k::n] for k in range(n)]
    deadline = time.time() + time_budget_s if time_budget_s else None
    metrics.notes.append(f"fetch=parallel processes={n} max_concurrency={sum(limits)} split={'+'.join(map(str, limits))}")
    ctx = multiprocessing.get_context(start_method)
    kp_event = ctx.Event()
    outs = []
    # workers only fetch and parse: no BLAS thread pools (numpy gets imported with the CLI module)
    saved = {k: os.environ.get(k) for k in _ONE_THREAD}
    for k in _ONE_THREAD:
        os.environ.setdefault(k, "1")
    try:
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx, initializer=_init_worker,
                                 initargs=(kp_event,)) as pool:
            futs = [pool.submit(worker, k, shards[k], limits[k], cfg.get("http"), delay_s, deadline)
                    for k in range(n)]
            for fu in futs:
                outs.append(fu.result())
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    results = []
    for out in outs:
        _merge_metrics(metrics, out)
        _replay_log(log, out)
        results.extend(out["results"])
    results.sort(key=lambda r: r["i"])
    for r in results:
        r["items"] = [Listing(**d) for d in r["items"]]
    return results
