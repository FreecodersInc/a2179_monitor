"""Polite HTTP: one session, per-host delay with jitter, retries with backoff, byte cap."""
import logging
import random
import time
from urllib.parse import urlparse

import requests

from .obs import ev

DEFAULT_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0 Safari/537.36 a2179-monitor/0.1 (personal, low rate)")
MAX_BYTES = 5 * 1024 * 1024


class Fetcher:
    def __init__(self, http_cfg: dict, metrics, log, sleep=time.sleep, session=None):
        self.cfg = http_cfg or {}
        self.metrics = metrics
        self.log = log
        self.sleep = sleep
        self.s = session or requests.Session()
        self.s.headers.update({
            "User-Agent": self.cfg.get("user_agent") or DEFAULT_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "sr-RS,sr;q=0.9,en;q=0.7,ru;q=0.5",
        })
        self.last = {}

    def _wait(self, host: str):
        delay = float(self.cfg.get("min_delay_s", 4)) + random.uniform(0, float(self.cfg.get("jitter_s", 2)))
        prev = self.last.get(host)
        if prev is not None:
            left = prev + delay - time.monotonic()
            if left > 0:
                self.sleep(left)
        self.last[host] = time.monotonic()

    def get(self, url: str, retry_403: int = None):
        """Returns (status, text). status is an int, or 'error' after all retries failed."""
        host = urlparse(url).netloc.lower().removeprefix("www.")
        retries = int(self.cfg.get("retries", 2))
        retry_403 = int(self.cfg.get("retry_403", 1)) if retry_403 is None else retry_403
        timeout = float(self.cfg.get("timeout_s", 20))
        attempt, n403 = 0, 0
        while True:
            self._wait(host)
            t0 = time.monotonic()
            status, text = "error", ""
            try:
                with self.s.get(url, timeout=timeout, stream=True, allow_redirects=True) as r:
                    status = r.status_code
                    if urlparse(r.url).netloc.lower().removeprefix("www.") != host:
                        ev(self.log, "redirect_off_host", logging.WARNING, url=url, final=r.url)
                    buf, size = [], 0
                    for chunk in r.iter_content(65536):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise IOError(f"response over {MAX_BYTES} bytes")
                        buf.append(chunk)
                    text = b"".join(buf).decode(r.encoding or "utf-8", errors="replace")
                    retry_after = r.headers.get("Retry-After")
            except Exception as e:
                retry_after, status, text = None, "error", ""
                ev(self.log, "fetch_error", logging.WARNING, url=url, error=f"{type(e).__name__}: {e}"[:300])
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
            self.sleep(back)
