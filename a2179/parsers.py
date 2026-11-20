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


_KP_FREE = re.compile(r"\b(besplatno|poklanjam|gratis|бесплатно|поклањам)\b", re.I)


def _kp_json_free(n: dict, name: str) -> bool:
    """A zero price in KP JSON is free only with an explicit giveaway marker."""
    for k, v in n.items():
        kl = str(k).lower()
        if kl in ("isfree", "free", "giveaway", "isgiveaway") and v is True:
            return True
        if "price" in kl and isinstance(v, str) and _KP_FREE.search(v):
            return True
    return bool(_KP_FREE.search(name or ""))


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
            if pv is not None and pv <= 0:
                # KP sends price 0 for "Kontakt" / "Dogovor" ads (seen live 19.11.2026: a parts laptop
                # at "0 RSD" fired a below target alert). Only an explicit giveaway counts as free.
                pv = 0.0 if _kp_json_free(n, name) else None
            cur = {"rsd": "RSD", "din": "RSD", "eur": "EUR", "€": "EUR"}.get(str(cur).lower(), "RSD" if pv is not None else None)
            if pv is None:
                cur = None
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
            if pm is None:  # no standalone price node: first number with a currency in the card text
                pm = next((m for m in _KP_PRICE.finditer(text.replace(title, " ", 1)) if m.group(1)), None)
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


# T-031: full KP listing card. Only the fields the matcher needs, seller identity is never kept and
# phone numbers and e-mails inside the description are masked.
_PHONE = re.compile(r"(?<![\w+])(?:\+381|00381|0)[\s/.-]?\(?\d{2,3}\)?[\s/.-]?\d{3}[\s/.-]?\d{3,4}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
CARD_FIELDS = ("id", "name", "description", "price", "currency", "condition", "categoryName", "groupName")


def mask_contacts(text: str) -> str:
    return _PHONE.sub("[tel]", _EMAIL.sub("[email]", text or ""))


def html_text(fragment: str) -> str:
    t = BeautifulSoup(fragment or "", "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", t).strip()


def parse_kp_card(html: str, url: str = "") -> Optional[dict]:
    """{"id", "title", "text", "price", "currency", "condition", "category"} from a KP ad page
    (__NEXT_DATA__ initialReduxState.ad.byId), None if it is not an ad page."""
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html or "", re.S)
    if not m:
        return None
    try:
        by = json.loads(m.group(1))["props"]["initialReduxState"]["ad"]["byId"]
    except (ValueError, KeyError, TypeError):
        return None
    want = re.search(r"/oglas/(\d+)", url or "")
    ads = [a for k, a in by.items() if isinstance(a, dict) and (not want or str(k) == want.group(1))] or \
        [a for a in by.values() if isinstance(a, dict)]
    if not ads:
        return None
    a = ads[0]
    return {"id": str(a.get("id", "")), "title": (a.get("name") or "").strip(),
            "text": mask_contacts(html_text(a.get("description") or "")),
            "price": parse_number(str(a.get("price"))) if a.get("price") not in (None, "") else None,
            "currency": a.get("currency") or "", "condition": a.get("condition") or "",
            "category": " / ".join(x for x in (a.get("categoryName"), a.get("groupName")) if x)}


def sanitize_kp_card_html(html: str, url: str = "") -> str:
    """A minimal fixture page: only CARD_FIELDS of the ad, contacts masked, no seller data."""
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    by = json.loads(m.group(1))["props"]["initialReduxState"]["ad"]["byId"]
    want = re.search(r"/oglas/(\d+)", url or "")
    k = want.group(1) if want and want.group(1) in by else next(iter(by))
    ad = {f: by[k].get(f) for f in CARD_FIELDS}
    ad["description"] = mask_contacts(ad.get("description") or "")
    d = {"props": {"initialReduxState": {"ad": {"byId": {k: ad}}}}}
    return ('<html><head><title>' + (ad.get("name") or "") + '</title></head><body><script id="__NEXT_DATA__" '
            'type="application/json">' + json.dumps(d, ensure_ascii=False) + "</script></body></html>\n")
