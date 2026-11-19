"""Async twin of fetch.Fetcher for the parallel mode: aiohttp, one ClientSession per process.

Same retry, 403, backoff, byte cap and metrics rules as fetch.Fetcher. The difference is the
rate limit: instead of one request at a time with a pause per host, at most `limit` requests
of this process are in flight at once (asyncio.Semaphore). The pause per host is optional
(`parallel.delay_s`, default 0). A slot is held only while a request runs, never while
sleeping in a backoff.
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
    """One ClientSession for the whole process; the connector pool matches the semaphore."""
    cfg = http_cfg or {}
    headers = dict(HEADERS, **{"User-Agent": cfg.get("user_agent") or DEFAULT_UA})
    return aiohttp.ClientSession(
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=float(cfg.get("timeout_s", 20))),
        connector=aiohttp.TCPConnector(limit=max(1, int(limit))),
    )


class AsyncFetcher:
    def __init__(self, http_cfg: dict, metrics, log, session, sem: asyncio.Semaphore,
                 delay_s: float = 0.0, sleep=asyncio.sleep):
        self.cfg = http_cfg or {}
        self.metrics = metrics
        self.log = log
        self.s = session
        self.sem = sem
        self.delay = float(delay_s or 0)
        self.sleep = sleep
        self.last = {}
        self.locks = {}

    async def _wait(self, host: str):
        if self.delay <= 0:
            return
        lock = self.locks.setdefault(host, asyncio.Lock())
        async with lock:
            prev = self.last.get(host)
            if prev is not None:
                left = prev + self.delay - time.monotonic()
                if left > 0:
                    await self.sleep(left)
            self.last[host] = time.monotonic()

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

    async def get(self, url: str, retry_403: int = None, gate=None):
        """Returns (status, text). status is an int, or 'error' after all retries failed.
        gate: optional callable checked right before the first request, once a slot is free;
        if it returns a reason (str), nothing is sent and ('skipped', reason) is returned."""
        host = _host(url)
        retries = int(self.cfg.get("retries", 2))
        retry_403 = int(self.cfg.get("retry_403", 1)) if retry_403 is None else retry_403
        attempt, n403 = 0, 0
        while True:
            await self._wait(host)
            async with self.sem:
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
                return status, text
            if status == 403:
                n403 += 1
                if n403 > retry_403:
                    return status, text
            elif status == 404 or (isinstance(status, int) and 400 <= status < 500 and status != 429):
                return status, text
            attempt += 1
            if attempt > retries:
                return status, text
            back = min(60.0, float(self.cfg.get("backoff_s", 5)) * (2 ** (attempt - 1)) + random.uniform(0, 1))
            if retry_after and str(retry_after).isdigit():
                back = min(60.0, max(back, float(retry_after)))
            await self.sleep(back)
