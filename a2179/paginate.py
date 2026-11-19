"""Search pagination: follow a search to its last page.

Pager is a small state machine shared by the sequential and the parallel pass:

    p = Pager(kind, first_url, cfg)
    while p.url:                      # next page to fetch, None when done
        st, html = get(p.url)         # through the normal fetcher: limiter, pacer, retries
        p.feed(st, html)
    p.items, p.pages, p.stop, p.blocked

Stops (p.stop): "empty" (page without listings), "no_next" (no next page link), "no_new"
(every listing on the page was already seen on earlier pages of this search), "end"
(404 or 410 after page 1, as WooCommerce answers past the last page), "error" (any other
failure after page 1), "blocked" (KP refused us), "loop" (next link points to a page we
already fetched), "off_host" (next link to another site), "ceiling" (`pagination.max_pages`,
a high safety limit, default 500), "first_page" (page 1 failed, handled by the caller as
before), "disabled" (pagination.enabled=false: only page 1).

Next page: <link rel="next">, <a rel="next">, WooCommerce `a.next.page-numbers`. KP markup is
not verified live, so for KP also an anchor to the same path with `page=<n+1>`, and if there
is none and `pagination.kp_page_param` is set (default "page"), the URL with that query
parameter set to n+1. The guess still stops on an empty page or a page without new listings.
"""
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from .parsers import _soup, kp_is_blocked, parse_kp_search, parse_woo_search

DEFAULTS = {"enabled": True, "max_pages": 500, "kp_page_param": "page"}


def settings(cfg: dict) -> dict:
    p = dict(DEFAULTS)
    p.update(cfg.get("pagination") or {})
    p["max_pages"] = max(1, int(p.get("max_pages") or DEFAULTS["max_pages"]))
    return p


def _host(url):
    return urlparse(url).netloc.lower().removeprefix("www.")


def _with_param(url, name, value):
    u = urlparse(url)
    q = [(k, v) for k, v in parse_qsl(u.query, keep_blank_values=True) if k != name] + [(name, str(value))]
    return urlunparse(u._replace(query=urlencode(q)))


def find_next(html: str, url: str, kind: str, page: int, kp_param: str = "page"):
    """Absolute URL of the next page, or None."""
    soup = _soup(html)
    el = soup.select_one('link[rel~="next"][href], a[rel~="next"][href], a.next.page-numbers[href], '
                         '.woocommerce-pagination a.next[href]')
    if el:
        return urljoin(url, el["href"])
    if kind != "kp":
        return None
    path = urlparse(url).path
    want = str(page + 1)
    for a in soup.find_all("a", href=True):
        h = urljoin(url, a["href"])
        u = urlparse(h)
        if u.path == path and dict(parse_qsl(u.query)).get(kp_param or "page") == want:
            return h
    if kp_param:
        return _with_param(url, kp_param, page + 1)
    return None


class Pager:
    def __init__(self, kind: str, first_url: str, cfg: dict):
        self.kind = kind  # "shop_search" or "kp"
        self.cfg = settings(cfg)
        self.url = first_url
        self.first_url = first_url
        self.host = _host(first_url)
        self.items, self.seen, self.visited = [], set(), set()
        self.pages = 0
        self.stop = None
        self.blocked = False
        self.status = None      # status of page 1 (what the caller reported before pagination)
        self.last_status = None
        self.bytes = 0          # size of page 1

    def _end(self, why):
        self.stop, self.url = why, None

    def feed(self, status, html: str):
        url = self.url
        self.visited.add(url)
        self.pages += 1
        self.last_status = status
        if self.pages == 1:
            self.status, self.bytes = status, len(html or "")
        if self.kind == "kp":
            refused = kp_is_blocked(status if isinstance(status, int) else 0, html)
            # page 1: anything but 200 counts as a refusal (as before pagination). Later pages:
            # only a real refusal (403/429/503 or the block page), a 404 there is just the end.
            if refused or (self.pages == 1 and status != 200):
                self.blocked = True
                return self._end("blocked")
        if status != 200:
            if self.pages == 1:
                return self._end("first_page")
            return self._end("end" if status in (404, 410) else "error")
        page_items = (parse_kp_search(html, url) if self.kind == "kp"
                      else parse_woo_search(html, url, _host(url)))
        if not page_items:
            return self._end("empty")
        new = [l for l in page_items if l.key not in self.seen]
        if not new:
            return self._end("no_new")
        self.seen.update(l.key for l in new)
        self.items.extend(new)
        if not self.cfg.get("enabled", True):
            return self._end("disabled")
        if self.pages >= self.cfg["max_pages"]:
            return self._end("ceiling")
        nxt = find_next(html, url, self.kind, self.pages, self.cfg.get("kp_page_param") or "")
        if not nxt:
            return self._end("no_next")
        if _host(nxt) != self.host:
            return self._end("off_host")
        if nxt in self.visited:
            return self._end("loop")
        self.url = nxt

    def note_metrics(self, metrics):
        metrics.inc(f"pagination.{self.kind}.pages", self.pages)
        if self.stop:
            metrics.inc(f"pagination.stop.{self.stop}")
