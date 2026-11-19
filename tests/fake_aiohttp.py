"""Fake aiohttp module to exercise the parallel fetch without the network (and without aiohttp).

install(routes, delay_s) puts it into sys.modules. Worker processes started with the 'fork'
method inherit it. In-flight requests are counted in shared memory, so the test can check
the global limit over all processes.
"""
import asyncio
import contextlib
import importlib.machinery
import multiprocessing
import sys
import types
from urllib.parse import urlparse

STATE = {}


class ClientError(Exception):
    pass


class ClientTimeout:
    def __init__(self, total=None, **kw):
        self.total = total


class TCPConnector:
    def __init__(self, limit=100, **kw):
        self.limit = limit


class _URL:
    def __init__(self, url):
        self._u = url
        self.host = urlparse(url).hostname

    def __str__(self):
        return self._u


class _Content:
    def __init__(self, data):
        self.data = data

    async def iter_chunked(self, n):
        for i in range(0, len(self.data), n):
            yield self.data[i:i + n]


class _Resp:
    def __init__(self, url, status, text, headers=None):
        self.status, self.url, self.headers = status, _URL(url), headers or {}
        self.charset = "utf-8"
        self.content = _Content(text.encode("utf-8"))


class _Ctx:
    def __init__(self, session, url):
        self.session, self.url = session, url

    async def __aenter__(self):
        st = STATE
        with st["lock"]:
            st["inflight"].value += 1
            st["peak"].value = max(st["peak"].value, st["inflight"].value)
            st["calls"].value += 1
        try:
            await asyncio.sleep(st["delay"])
            r = st["routes"].get(self.url)
            if callable(r):
                r = r()
            if isinstance(r, Exception):
                raise r
            status, text = r if r else (404, "not found")
            return _Resp(self.url, status, text)
        finally:
            with st["lock"]:
                st["inflight"].value -= 1

    async def __aexit__(self, *a):
        return False


class ClientSession:
    def __init__(self, headers=None, timeout=None, connector=None, **kw):
        self.headers, self.timeout, self.connector = headers or {}, timeout, connector
        with STATE["lock"]:
            STATE["sessions"].value += 1

    def get(self, url, **kw):
        return _Ctx(self, url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def install(routes, delay_s=0.0):
    ctx = multiprocessing.get_context("fork")
    STATE.clear()
    STATE.update(routes=routes, delay=delay_s, lock=ctx.Lock(), inflight=ctx.Value("i", 0),
                 peak=ctx.Value("i", 0), calls=ctx.Value("i", 0), sessions=ctx.Value("i", 0))
    m = types.ModuleType("aiohttp")
    for name in ("ClientError", "ClientTimeout", "TCPConnector", "ClientSession"):
        setattr(m, name, globals()[name])
    m.__spec__ = importlib.machinery.ModuleSpec("aiohttp", None, origin="fake")
    return m


@contextlib.contextmanager
def fake_aiohttp(routes, delay_s=0.0):
    saved = sys.modules.get("aiohttp", "absent")
    sys.modules["aiohttp"] = install(routes, delay_s)
    try:
        yield STATE
    finally:
        if saved == "absent":
            sys.modules.pop("aiohttp", None)
        else:
            sys.modules["aiohttp"] = saved
