"""Async twin of fetch.Fetcher for the parallel mode: aiohttp, one ClientSession per process.

Same 403, byte cap and metrics rules as fetch.Fetcher. What is different:
- concurrency: every request takes a slot of ONE limiter shared by all worker processes
  (Limiter below, a multiprocessing.BoundedSemaphore), so at most `max_concurrency` GET are
  in flight in total and a process without work holds no slots;
- retries: network errors, 5xx and 429 are retried up to `http.retries` times (default 3)
  with exponential backoff and jitter;
- 429: an adaptive delay per host, shared by all processes (Pacer below, AIMD): it grows
  multiplicatively on every 429, honours Retry-After, and shrinks a little on every success,
  so the request rate settles just under the site's limit.
A slot is held only while a request runs, never while sleeping.
"""
import asyncio
import logging
import random
import time
from urllib.parse import urlparse

from .fetch import DEFAULT_UA, MAX_BYTES
from .obs import ev

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "sr-RS,sr;q=0.9,en;q=0.7,ru;q=0.5",
}


def _host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def make_session(aiohttp, http_cfg: dict, limit: int):
    """One ClientSession for the whole process. Its pool may use every global slot."""
    cfg = http_cfg or {}
    headers = dict(HEADERS, **{"User-Agent": cfg.get("user_agent") or DEFAULT_UA})
    return aiohttp.ClientSession(
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=float(cfg.get("timeout_s", 20))),
        connector=aiohttp.TCPConnector(limit=max(1, int(limit))),
    )


def retry_after_s(value):
    """Retry-After in seconds (only the delta-seconds form), else None."""
    try:
        v = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return v if v >= 0 else None


class Limiter:
    """Cross-process slot: acquire on a multiprocessing semaphore without blocking the event loop.

    Polls sem.acquire(block=False) with a short asyncio sleep. One coroutine per process
    polls at a time (asyncio.Lock), the others wait on that lock for free.
    """
    def __init__(self, sem, poll_s: float = 0.005, poll_max_s: float = 0.05):
        self.sem, self.poll_s, self.poll_max_s = sem, poll_s, poll_max_s
        self._turn = None

    async def acquire(self):
        if self._turn is None:
            self._turn = asyncio.Lock()
        async with self._turn:
            wait = self.poll_s
            while not self.sem.acquire(False):
                await asyncio.sleep(wait)
                wait = min(self.poll_max_s, wait * 1.5)

    def release(self):
        self.sem.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *a):
        self.release()
        return False


class Pacer:
    """Adaptive delay between request starts, per host, shared by all processes (AIMD).

    delay[h]: seconds between two starts to host h. next[h]: earliest start (time.time()).
    429: delay = clamp(max(delay * up, on_429)), and next >= now + Retry-After if given.
    success: delay = max(min, delay - down). Everything lives in shared memory under one lock.
    """
    def __init__(self, hosts, delay, nxt, lock, settings: dict):
        self.idx = {h: i for i, h in enumerate(hosts)}
        self.delay, self.next, self.lock = delay, nxt, lock
        s = settings or {}
        self.dmin = float(s.get("delay_min_s", 0))
        self.dmax = float(s.get("delay_max_s", 30))
        self.up = float(s.get("delay_up", 2.0))
        self.down = float(s.get("delay_down_s", 0.05))
        self.on_429 = float(s.get("delay_on_429_s", 0.5))

    def current(self, host):
        i = self.idx.get(host)
        return None if i is None else self.delay[i]

    def reserve(self, host):
        """Claim the next start time for host. Returns seconds to wait (0: go now)."""
        i = self.idx.get(host)
        if i is None:
            return 0.0
        with self.lock:
            now = time.time()
            start = max(now, self.next[i])
            self.next[i] = start + self.delay[i]
            return start - now

    async def wait(self, host, sleep=asyncio.sleep):
        left = self.reserve(host)
        if left > 0:
            await sleep(left)

    def on_rate_limited(self, host, retry_after=None):
        i = self.idx.get(host)
        if i is None:
            return None
        with self.lock:
            d = min(self.dmax, max(self.dmin, self.on_429, self.delay[i] * self.up))
            self.delay[i] = d
            if retry_after is not None:
                self.next[i] = max(self.next[i], time.time() + min(retry_after, self.dmax))
            return d

    def on_success(self, host):
        i = self.idx.get(host)
        if i is None:
            return
        with self.lock:
            self.delay[i] = max(self.dmin, self.delay[i] - self.down)


class AsyncFetcher:
    def __init__(self, http_cfg: dict, metrics, log, session, limiter, pacer=None, sleep=asyncio.sleep):
        self.cfg = http_cfg or {}
        self.metrics = metrics
        self.log = log
        self.s = session
        self.limiter = limiter
        self.pacer = pacer
        self.sleep = sleep

    async def _once(self, url: str, host: str):
        status, text, retry_after = "error", "", None
        try:
            async with self.s.get(url, allow_redirects=True) as r:
                status = r.status
                final_host = (getattr(r.url, "host", None) or urlparse(str(r.url)).hostname or "")
                if final_host.lower().removeprefix("www.") != host.split(":")[0]:
                    ev(self.log, "redirect_off_host", logging.WARNING, url=url, final=str(r.url))
                buf, size = [], 0
                async for chunk in r.content.iter_chunked(65536):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise IOError(f"response over {MAX_BYTES} bytes")
                    buf.append(chunk)
                text = b"".join(buf).decode(r.charset or "utf-8", errors="replace")
                retry_after = r.headers.get("Retry-After")
        except Exception as e:  # aiohttp.ClientError, asyncio.TimeoutError, IOError, bad charset
            retry_after, status, text = None, "error", ""
            ev(self.log, "fetch_error", logging.WARNING, url=url, error=f"{type(e).__name__}: {e}"[:300])
        return status, text, retry_after

    def backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter: base * 2^(attempt-1), times a random 0.5..1.5, capped."""
        base = float(self.cfg.get("backoff_s", 5))
        cap = float(self.cfg.get("backoff_max_s", 60))
        return min(cap, base * (2 ** (attempt - 1)) * random.uniform(0.5, 1.5))

    async def get(self, url: str, retry_403: int = None, gate=None):
        """Returns (status, text). status is an int, or 'error' after all retries failed.
        gate: optional callable checked right before the first request, once a slot is free;
        if it returns a reason (str), nothing is sent and ('skipped', reason) is returned."""
        host = _host(url)
        retries = int(self.cfg.get("retries", 3))
        retry_403 = int(self.cfg.get("retry_403", 1)) if retry_403 is None else retry_403
        attempt, n403 = 0, 0
        while True:
            if self.pacer is not None:
                await self.pacer.wait(host, self.sleep)
            async with self.limiter:
                if gate is not None and attempt == 0 and n403 == 0:
                    why = gate()
                    if why:
                        return "skipped", why
                t0 = time.monotonic()
                status, text, retry_after = await self._once(url, host)
                ms = (time.monotonic() - t0) * 1000
            self.metrics.fetch(host, status, ms, len(text))
            ev(self.log, "fetch", url=url, status=status, ms=round(ms), bytes=len(text), attempt=attempt)
            if isinstance(status, int) and status < 400:
                if self.pacer is not None:
                    self.pacer.on_success(host)
                return status, text
            ra = None
            if status == 429:
                self.metrics.inc("fetch.rate_limited")
                ra = retry_after_s(retry_after)
                if self.pacer is not None:
                    d = self.pacer.on_rate_limited(host, ra)
                    ev(self.log, "rate_limited", logging.WARNING, url=url, host=host, delay_s=round(d or 0, 3),
                       retry_after=ra)
            elif status == 403:
                n403 += 1
                if n403 > retry_403:
                    return status, text
            elif isinstance(status, int) and 400 <= status < 500:
                return status, text  # 404 and other client errors: retrying will not help
            attempt += 1
            if attempt > retries:
                return status, text
            self.metrics.inc("fetch.retries")
            back = self.backoff(attempt)
            if ra is not None:
                back = max(back, min(ra, float(self.cfg.get("backoff_max_s", 60))))
            await self.sleep(back)
