"""Local sqlite state: dedupe, price history, alert history."""
import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
  key TEXT PRIMARY KEY, source TEXT, url TEXT, title TEXT,
  first_seen REAL, last_seen REAL, price_usd REAL, price REAL, currency TEXT,
  min_price_usd REAL, last_alert_price_usd REAL, matched INTEGER, data TEXT
);
CREATE TABLE IF NOT EXISTS price_history (key TEXT, ts REAL, price REAL, currency TEXT, price_usd REAL);
CREATE TABLE IF NOT EXISTS alerts (ts REAL, key TEXT, kind TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


class State:
    def __init__(self, path: Path, read_only: bool = False):
        self.read_only = read_only
        if read_only and not Path(path).exists():
            self.db = sqlite3.connect(":memory:")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    def get(self, key):
        cur = self.db.execute("SELECT key, price_usd, min_price_usd, last_alert_price_usd, matched FROM listings "
                              "WHERE key=?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        return dict(zip(("key", "price_usd", "min_price_usd", "last_alert_price_usd", "matched"), row))

    def upsert(self, listing, price_usd, matched: bool, alerted_price_usd=None):
        if self.read_only:
            return
        now = time.time()
        prev = self.get(listing.key)
        if prev is None:
            self.db.execute(
                "INSERT INTO listings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (listing.key, listing.source, listing.url, listing.title, now, now, price_usd, listing.price,
                 listing.currency, price_usd, alerted_price_usd, int(matched),
                 json.dumps(listing.to_dict(), ensure_ascii=False)))
        else:
            mn = prev["min_price_usd"]
            if price_usd is not None:
                mn = price_usd if mn is None else min(mn, price_usd)
            self.db.execute(
                "UPDATE listings SET last_seen=?, title=?, url=?, price_usd=?, price=?, currency=?, min_price_usd=?, "
                "matched=?, data=?, last_alert_price_usd=COALESCE(?, last_alert_price_usd) WHERE key=?",
                (now, listing.title, listing.url, price_usd, listing.price, listing.currency, mn, int(matched),
                 json.dumps(listing.to_dict(), ensure_ascii=False), alerted_price_usd, listing.key))
        if price_usd is not None and (prev is None or prev["price_usd"] != price_usd):
            self.db.execute("INSERT INTO price_history VALUES (?,?,?,?,?)",
                            (listing.key, now, listing.price, listing.currency, price_usd))

    def log_alert(self, key, kind, text):
        if not self.read_only:
            self.db.execute("INSERT INTO alerts VALUES (?,?,?,?)", (time.time(), key, kind, text))

    def meta_get(self, k, default=None):
        row = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return row[0] if row else default

    def meta_set(self, k, v):
        if not self.read_only:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, str(v)))

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.commit()
        self.db.close()
