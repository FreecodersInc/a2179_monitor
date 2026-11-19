"""Tracked product cards: shop listings that matched once, re-checked every pass like shop_products.

Storage: a shelve file (`data/tracked.shelve`, the dbm backend picks the real file name).
Concurrency, chosen design: a shelve must never be open twice for writing, so every access
takes ONE multiprocessing.Lock (created by the main process and handed to the workers at
start, like the request limiter) and opens the shelve, reads or writes, and closes it again
before the lock is released. Nothing keeps the shelve open between operations, so a write
is on disk before the next process can look. Workers only update the cards they fetched
(price, stock, status), the main process adds new cards after matching. dry_run never writes.

The lock covers the processes of one run. Two separate monitor runs on the same data folder
at the same time are not protected (do not start cron while --every is running).

A card that answers 404/410 `tracked.drop_after_misses` passes in a row (default 3) is marked
gone and is not fetched any more. It stays in the file (history), it is never deleted.
"""
import contextlib
import shelve
import threading
import time
from pathlib import Path

DEFAULTS = {"enabled": True, "file": "tracked.shelve", "drop_after_misses": 3}


def settings(cfg: dict) -> dict:
    t = dict(DEFAULTS)
    t.update(cfg.get("tracked") or {})
    return t


def _exists(path: Path) -> bool:
    return any(Path(str(path) + ext).exists() for ext in ("", ".db", ".dat", ".dir", ".pag"))


class TrackedStore:
    def __init__(self, path, lock=None, read_only: bool = False, drop_after_misses: int = 3):
        self.path = Path(path)
        self.lock = lock if lock is not None else threading.Lock()
        self.read_only = read_only
        self.drop_after = max(1, int(drop_after_misses))

    @contextlib.contextmanager
    def _open(self, write: bool):
        with self.lock:
            if not write and not _exists(self.path):
                yield {}
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = shelve.open(str(self.path), flag="c" if write else "r")
            try:
                yield db
            finally:
                db.close()

    def all(self) -> dict:
        with self._open(False) as db:
            return {k: dict(db[k]) for k in db.keys()}

    def active(self) -> list:
        """[(key, url)] of cards still fetched, oldest first."""
        cards = self.all()
        return sorted(((k, v["url"]) for k, v in cards.items() if not v.get("gone")),
                      key=lambda kv: (cards[kv[0]].get("added", 0), kv[0]))

    def add(self, listings) -> list:
        """Adds new cards (by listing key). Returns the keys that were new."""
        if self.read_only:
            return []
        added, now = [], time.time()
        with self._open(True) as db:
            for l in listings:
                if l.key in db:
                    continue
                db[l.key] = {"url": l.url, "title": l.title, "source": l.source, "added": now,
                             "last_seen": None, "last_status": None, "price": l.price, "currency": l.currency,
                             "in_stock": l.in_stock, "misses": 0, "gone": False, "checks": 0}
                added.append(l.key)
        return added

    def record(self, key: str, status, listing=None):
        """Result of one fetch of a tracked card. Read-modify-write under the lock."""
        if self.read_only:
            return None
        with self._open(True) as db:
            card = dict(db.get(key) or {})
            if not card:
                return None
            card["checks"] = card.get("checks", 0) + 1
            card["last_status"] = status
            if status == 200 and listing is not None:
                card.update(last_seen=time.time(), title=listing.title, price=listing.price,
                            currency=listing.currency, in_stock=listing.in_stock, misses=0)
            elif status in (404, 410):
                card["misses"] = card.get("misses", 0) + 1
                if card["misses"] >= self.drop_after:
                    card["gone"] = True
            db[key] = card
            return card
