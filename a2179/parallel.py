"""Parallel fetch: N worker processes, each one asyncio loop with one aiohttp ClientSession.

Why: a pass is mostly waiting on the network. Workers fetch and parse at the same time,
so parsing in one coroutine overlaps with other requests waiting on their responses.

Global limit: ONE multiprocessing.BoundedSemaphore(max_concurrency) (default 10) above all
workers. Every request (retries included) takes a slot right before it goes out and gives it
back as soon as the response is read, so at no moment more than max_concurrency GET are in
flight in total, and a process that has run out of work holds no slots: the busy ones get
them all. It is an OS semaphore handed to the workers at start, no Manager process. Workers
take it with a non-blocking poll (afetch.Limiter), so the event loop never blocks.

Rate limit: the delay between request starts to one host lives in shared memory too
(afetch.Pacer, AIMD): times `delay_up` on every 429 (Retry-After honoured), minus
`delay_down_s` on every success, kept within `delay_min_s`..`delay_max_s`. The learned
delays carry over to the next pass of the same run (--every).

Results go back to the main process in the original source order, so dedupe, the KP block
rule, matching, state and alerts behave exactly as in the sequential pass.
"""
import asyncio
import dataclasses
import importlib.util
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from urllib.parse import quote_plus, urlparse

from .afetch import AsyncFetcher, Limiter, Pacer, make_session
from .obs import Metrics
from . import paginate
from .paginate import Pager
from .parsers import Listing, parse_woo_product
from .tracked import TrackedStore

DEFAULT_WORKERS = 3
DEFAULT_MAX_CONCURRENCY = 10
PACE_DEFAULTS = {"delay_s": 0.0, "delay_min_s": 0.0, "delay_max_s": 30.0, "delay_up": 2.0,
                 "delay_down_s": 0.05, "delay_on_429_s": 0.5}

_ONE_THREAD = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_LAST_DELAY = {}    # host -> delay learned in the previous pass (main process)
_KP_BLOCKED = None  # multiprocessing.Event, set in every worker by _init_worker
_SLOTS = None       # multiprocessing.BoundedSemaphore(max_concurrency), shared by all workers
_PACE = None        # (hosts, delay Array, next-start Array, Lock), shared by all workers
_STATE_LOCK = None  # multiprocessing.Lock around every tracked.shelve access, shared by all workers


class ParallelUnavailable(RuntimeError):
    """aiohttp is not installed: the caller falls back to the sequential pass."""


def settings(cfg: dict, workers=None, max_concurrency=None):
    """(workers, max_concurrency, pace settings, start_method) from CLI args over cfg['parallel']."""
    p = cfg.get("parallel") or {}
    w = int(p.get("workers", DEFAULT_WORKERS) if workers is None else workers)
    mc = int(p.get("max_concurrency", DEFAULT_MAX_CONCURRENCY) if max_concurrency is None else max_concurrency)
    pace = {k: float(v if p.get(k) is None else p[k]) for k, v in PACE_DEFAULTS.items()}
    return max(0, w), max(1, mc), pace, str(p.get("start_method") or "spawn")


def build_tasks(cfg: dict, tracked=None) -> list:
    """[(index, kind, url)] in the same order as the sequential collect():
    shop products, tracked cards (not already a shop product), shop searches, KP searches."""
    src = cfg["sources"]
    urls = [("shop_product", u) for u in src.get("shop_products", [])]
    have = {Listing("", u, "").key for _, u in urls}
    urls += [("tracked", u) for k, u in (tracked or []) if k not in have]
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


def _init_worker(kp_event, slots, pace, state_lock=None):
    global _KP_BLOCKED, _SLOTS, _PACE, _STATE_LOCK
    _KP_BLOCKED, _SLOTS, _PACE, _STATE_LOCK = kp_event, slots, pace, state_lock


def _store(opts):
    t = (opts or {}).get("tracked")
    if not t:
        return None
    return TrackedStore(t["path"], _STATE_LOCK, read_only=t["read_only"], drop_after_misses=t["drop_after"])


async def _run_task(fetcher, task, deadline, opts=None, store=None, io=None):
    i, kind, url = task
    res = {"i": i, "kind": kind, "url": url, "status": None, "items": [], "skipped": None, "bytes": 0,
           "blocked": False, "pages": 0, "stop": None}

    def gate():  # checked when a slot is free, right before the request goes out
        if deadline is not None and time.time() >= deadline:
            return "time"
        if kind == "kp" and _KP_BLOCKED is not None and _KP_BLOCKED.is_set():
            return "kp_blocked"  # be polite: KP already refused us in this pass
        return None

    if kind in ("shop_search", "kp"):
        pager = Pager(kind, url, {"pagination": (opts or {}).get("pagination")})
        while pager.url:
            st, html = await fetcher.get(pager.url, gate=gate)
            if st == "skipped":
                if pager.pages == 0:
                    res["skipped"] = html
                    return res
                pager.stop, pager.url = html, None
                break
            pager.feed(st, html)
        pager.note_metrics(fetcher.metrics)
        res.update(status=pager.status, last_status=pager.last_status, bytes=pager.bytes, pages=pager.pages,
                   stop=pager.stop,
                   blocked=pager.blocked, items=[dataclasses.asdict(l) for l in pager.items])
        if pager.blocked and _KP_BLOCKED is not None:
            _KP_BLOCKED.set()
        return res
    st, html = await fetcher.get(url, gate=gate)
    if st == "skipped":
        res["skipped"] = html
        return res
    res["status"], res["bytes"] = st, len(html)
    l = parse_woo_product(html, url, _src_host(url)) if st == 200 else None
    res["items"] = [dataclasses.asdict(l)] if l else []
    if kind == "tracked" and store is not None:
        key = Listing("", url, "").key
        # shelve write under the cross-process lock, off the event loop (one IO thread per process)
        await asyncio.get_running_loop().run_in_executor(io, store.record, key, st, l)
    return res


async def _worker_async(tasks, max_concurrency, http_cfg, pace_cfg, deadline, metrics, log, opts=None):
    import aiohttp
    hosts, delay, nxt, lock = _PACE
    pacer = Pacer(hosts, delay, nxt, lock, pace_cfg)
    store = _store(opts)
    with ThreadPoolExecutor(max_workers=1) as io:
        async with make_session(aiohttp, http_cfg, max_concurrency) as session:
            f = AsyncFetcher(http_cfg, metrics, log, session, Limiter(_SLOTS), pacer)
            return await asyncio.gather(*(_run_task(f, t, deadline, opts, store, io) for t in tasks))


def worker(idx, tasks, max_concurrency, http_cfg, pace_cfg, deadline, opts=None):
    """Runs in a worker process. Returns plain data only (picklable)."""
    metrics, log = Metrics(), _BufLog()
    results = asyncio.run(_worker_async(tasks, max_concurrency, http_cfg, pace_cfg, deadline, metrics, log, opts))
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


def fetch_all(cfg, metrics, log, time_budget_s=None, workers=None, max_concurrency=None, store=None):
    """Fetch and parse every source in parallel. Returns task results sorted by source order.
    store: TrackedStore (its file path is used; the workers get their own lock-guarded handle)."""
    tracked = store.active() if store is not None else []
    tasks = build_tasks(cfg, tracked)
    w, mc, pace_cfg, start_method = settings(cfg, workers, max_concurrency)
    if not tasks:
        return []
    if importlib.util.find_spec("aiohttp") is None:
        raise ParallelUnavailable("aiohttp is not installed")
    n = max(1, min(w, len(tasks)))
    shards = [tasks[k::n] for k in range(n)]
    deadline = time.time() + time_budget_s if time_budget_s else None
    metrics.notes.append(f"fetch=parallel processes={n} max_concurrency={mc} (one shared limiter)")
    ctx = multiprocessing.get_context(start_method)
    hosts = sorted({_src_host(u) for _, _, u in tasks})
    lo, hi = pace_cfg["delay_min_s"], pace_cfg["delay_max_s"]
    delay = ctx.Array("d", [max(lo, min(hi, _LAST_DELAY.get(h, pace_cfg["delay_s"]))) for h in hosts], lock=False)
    nxt = ctx.Array("d", [0.0] * len(hosts), lock=False)
    pace = (hosts, delay, nxt, ctx.Lock())
    kp_event, slots, state_lock = ctx.Event(), ctx.BoundedSemaphore(mc), ctx.Lock()
    opts = {"pagination": paginate.settings(cfg),
            "tracked": None if store is None else {"path": str(store.path), "read_only": store.read_only,
                                                   "drop_after": store.drop_after}}
    outs = []
    # workers only fetch and parse: no BLAS thread pools (numpy gets imported with the CLI module)
    saved = {k: os.environ.get(k) for k in _ONE_THREAD}
    for k in _ONE_THREAD:
        os.environ.setdefault(k, "1")
    try:
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx, initializer=_init_worker,
                                 initargs=(kp_event, slots, pace, state_lock)) as pool:
            futs = [pool.submit(worker, k, shards[k], mc, cfg.get("http"), pace_cfg, deadline, opts)
                    for k in range(n)]
            for fu in futs:
                outs.append(fu.result())
    finally:
        if store is not None:
            store.lock = state_lock  # later writes of this pass (main process) use the same mutex
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    for h, d in zip(hosts, list(delay)):
        _LAST_DELAY[h] = d
        if d > 0:
            metrics.notes.append(f"pace {h}={d:.2f}s")
    results = []
    for out in outs:
        _merge_metrics(metrics, out)
        _replay_log(log, out)
        results.extend(out["results"])
    results.sort(key=lambda r: r["i"])
    for r in results:
        r["items"] = [Listing(**d) for d in r["items"]]
    return results
