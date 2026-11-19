"""HTML parsers: WooCommerce product/search pages (ifix.rs, ihelp.rs, laptop-punjaci.com)
and KupujemProdajem search pages."""
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .prices import parse_price, parse_number, Price


@dataclass
class Listing:
    source: str
    url: str
    title: str
    snippet: str = ""
    price: Optional[float] = None
    currency: Optional[str] = None
    location: str = ""
    in_stock: Optional[bool] = None
    condition: str = ""

    @property
    def key(self) -> str:
        m = re.search(r"/oglas/(\d+)", self.url)
        if m:
            return f"kp:{m.group(1)}"
        u = urlparse(self.url)
        # shops expose one product under several category paths: key on host + product slug
        slug = u.path.rstrip("/").rsplit("/", 1)[-1] or u.path
        return f"{u.netloc.lower().removeprefix('www.')}:{slug}{('?' + u.query) if u.query else ''}"

    def to_dict(self):
        d = asdict(self)
        d["key"] = self.key
        return d


def _soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def _ld_products(soup):
    for sc in soup.find_all("script", type="application/ld+json"):
        try:
            d = json.loads(sc.string or sc.get_text() or "")
        except Exception:
            continue
        stack = [d]
        while stack:
            n = stack.pop()
            if isinstance(n, list):
                stack.extend(n)
            elif isinstance(n, dict):
                if "Product" in str(n.get("@type", "")):
                    yield n
                stack.extend(v for k, v in n.items() if k in ("@graph", "mainEntity", "itemListElement"))


def _offer_price(offers):
    """Lowest price over Offer / AggregateOffer / priceSpecification."""
    best = None
    stack = [offers]
    while stack:
        o = stack.pop()
        if isinstance(o, list):
            stack.extend(o)
            continue
        if not isinstance(o, dict):
            continue
        cur = (o.get("priceCurrency") or "").upper().replace("SRDIN", "RSD") or None
        for k in ("price", "lowPrice"):
            if o.get(k) not in (None, ""):
                v = parse_number(str(o[k]))
                if v and (best is None or v < best[0]):
                    best = (v, cur or "RSD")
        stack.extend(x for x in (o.get("priceSpecification"), o.get("offers")) if x)
    return best


def _stock(offers, soup) -> Optional[bool]:
    s = json.dumps(offers or "")
    if "OutOfStock" in s or "SoldOut" in s:
        return False
    if "InStock" in s:
        return True
    st = soup.select_one(".stock")
    if st:
        t = st.get_text(" ", strip=True).lower()
        if "nema" in t or "out of stock" in t or "rasprodat" in t:
            return False
        return True
    return None


def parse_woo_product(html: str, url: str, source: str) -> Optional[Listing]:
    soup = _soup(html)
    title, price, cur, offers = None, None, None, None
    for p in _ld_products(soup):
        title = title or p.get("name")
        offers = p.get("offers")
        bp = _offer_price(offers)
        if bp:
            price, cur = bp
            break
    h1 = soup.select_one("h1.product_title, h1.product-title, h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(" ", strip=True)
    if price is None:
        form = soup.select_one("form.variations_form[data-product_variations]")
        if form:
            try:
                vs = json.loads(form["data-product_variations"])
                ps = [float(v["display_price"]) for v in vs if v.get("display_price")]
                if ps:
                    price, cur = min(ps), "RSD"
            except Exception:
                pass
    if price is None:
        m = soup.select_one('meta[property="product:price:amount"]')
        if m and m.get("content"):
            price, cur = parse_number(m["content"]), "RSD"
    if price is None:
        el = soup.select_one(".summary .price ins, .summary .price, p.price")
        if el:
            pp = parse_price(el.get_text(" ", strip=True), "RSD")
            if pp:
                price, cur = pp.amount, pp.currency
    if not title:
        return None
    return Listing(source, url, title, price=price, currency=cur, in_stock=_stock(offers, soup), condition="new")


def parse_woo_search(html: str, base_url: str, source: str) -> list:
    soup = _soup(html)
    out = []
    for it in soup.select("li.product, div.product-small, .products .product"):
        a = it.select_one("a.woocommerce-LoopProduct-link, a[href*='/proizvodi/'], a[href*='/product/'], "
                          "a[href*='/prodavnica/'], a[href]")
        t = it.select_one(".woocommerce-loop-product__title, .product-title, .name, h2, h3")
        if not a or not t:
            continue
        pr = it.select_one(".price ins") or it.select_one(".price")
        pp = parse_price(pr.get_text(" ", strip=True), "RSD") if pr else None
        oos = "outofstock" in " ".join(it.get("class", [])).lower()
        out.append(Listing(source, urljoin(base_url, a["href"]), t.get_text(" ", strip=True),
                           price=pp.amount if pp else None, currency=pp.currency if pp else None,
                           in_stock=False if oos else None, condition="new"))
    # dedupe by url
    seen, uniq = set(), []
    for l in out:
        if l.url not in seen:
            seen.add(l.url)
            uniq.append(l)
    return uniq


KP_BLOCK_MARKERS = ("Uskoro ćemo ponovo biti dostupni", "Access denied", "cf-chl", "captcha")


def kp_is_blocked(status: int, html: str) -> bool:
    return status in (403, 429, 503) or any(m in (html or "")[:50000] for m in KP_BLOCK_MARKERS)


_KP_AD_HREF = re.compile(r"/oglas/\d+")
_KP_PRICE = re.compile(r"(?<![\w.,])(\d{1,3}(?:[.\s\u00a0]\d{3})+|\d+)(?:,\d+)?\s*(?:din\.?|€|eur|rsd)(?!\w)"
                       r"|\b(dogovor|kontakt|besplatno|poklanjam)\b", re.I)


def _kp_from_next_data(soup, base_url):
    sc = soup.find("script", id="__NEXT_DATA__")
    if not sc:
        return []
    try:
        d = json.loads(sc.string or "")
    except Exception:
        return []
    out, stack = [], [d]
    while stack:
        n = stack.pop()
        if isinstance(n, list):
            stack.extend(n)
            continue
        if not isinstance(n, dict):
            continue
        name = n.get("name") or n.get("title")
        link = n.get("adUrl") or n.get("url") or n.get("link") or n.get("href")
        if isinstance(name, str) and isinstance(link, str) and _KP_AD_HREF.search(link):
            price = n.get("price")
            cur = n.get("currency") or n.get("priceCurrency") or ""
            pv = parse_number(str(price)) if price not in (None, "") else None
            cur = {"rsd": "RSD", "din": "RSD", "eur": "EUR", "€": "EUR"}.get(str(cur).lower(), "RSD" if pv else None)
            out.append(Listing("kupujemprodajem", urljoin(base_url, link), name.strip(),
                               snippet=str(n.get("description") or n.get("text") or "")[:300],
                               price=pv, currency=cur,
                               location=str(n.get("locationName") or n.get("location") or "")))
        else:
            stack.extend(n.values())
    return out


def parse_kp_search(html: str, base_url: str) -> list:
    """Best effort. KP markup changes and the site often blocks bots, so this tries
    __NEXT_DATA__ JSON first and then any anchor that points at /oglas/<id>."""
    soup = _soup(html)
    out = _kp_from_next_data(soup, base_url)
    if not out:
        for a in soup.find_all("a", href=_KP_AD_HREF):
            box = a
            for _ in range(6):  # climb to the ad card
                if box.parent is None:
                    break
                box = box.parent
                if len(box.find_all("a", href=_KP_AD_HREF)) > 3:
                    box = a.parent
                    break
                if box.name in ("article", "section") or any("ad" in c.lower() for c in box.get("class", [])):
                    break
            title = a.get("title") or a.get_text(" ", strip=True)
            if not title or len(title) < 4:
                continue
            text = box.get_text(" ", strip=True)
            # prefer a text node that is only a price ("18.500 din"), else search the card text
            pm = None
            for node in box.find_all(string=True):
                m = _KP_PRICE.fullmatch(node.strip())
                if m:
                    pm = m
                    break
            pm = pm or _KP_PRICE.search(text.replace(title, " ", 1))
            pp = parse_price(pm.group(0)) if pm else None
            free = bool(pm and re.search(r"besplatno|poklanjam", pm.group(0), re.I))
            snippet = text.replace(title, "", 1)[:300]
            out.append(Listing("kupujemprodajem", urljoin(base_url, a["href"]), title, snippet=snippet,
                               price=0.0 if free else (pp.amount if pp else None),
                               currency="RSD" if free else (pp.currency if pp else None)))
    seen, uniq = set(), []
    for l in out:
        if l.key not in seen:
            seen.add(l.key)
            uniq.append(l)
    return uniq
