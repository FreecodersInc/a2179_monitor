"""Config: built-in defaults <- config.json (or config.example.json) <- environment variables."""
import copy
import json
import os
from pathlib import Path

from . import PROJECT_ROOT

DEFAULTS = {
    "target_usd": 20.0,
    "rates": {"usd_per_eur": 1.16, "rsd_per_eur": 117.2},
    "price_drop_min_pct": 3.0,
    "alert_on": {"new": True, "price_drop": True, "below_target": True},
    "accept_a1932": True,
    "embedder": {"enabled": "auto",
                 "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 "local_only": False},
    "thresholds": {"tfidf": {"pos": 0.38, "margin": 0.18, "calibrated": True}, "embed": {"pos": 0.60, "margin": 0.03, "calibrated": False}},
    "sources": {
        "shop_products": [
            "https://ifix.rs/proizvodi/ekran-za-apple-macbook-air-13-a1932-a2179-2019-2020/",
            "https://ihelp.rs/prodavnica/rezervni-delovi-elektronika/rezervni-delovi-za-racunare-i-laptopove/lcd/lcd-panel-macbook-a2179-lcd/",
            "https://ihelp.rs/prodavnica/rezervni-delovi-elektronika/rezervni-delovi-za-racunare-i-laptopove/lcd/lcd-panel-macbook-a2179-komplet/",
            "https://laptop-punjaci.com/product/lcd-panel-macbook-a2179-lcd/",
            "https://laptop-punjaci.com/product/lcd-panel-macbook-a2179-komplet/",
        ],
        "shop_searches": [
            "https://ifix.rs/?s=a2179&post_type=product",
            "https://ihelp.rs/?s=a2179&post_type=product",
            "https://laptop-punjaci.com/?s=a2179&post_type=product",
        ],
        "kp_search_url": "https://www.kupujemprodajem.com/pretraga?keywords={q}",
        "kp_keywords": ["macbook air a2179", "a2179 ekran", "a2179 lcd", "macbook air 2020 ekran"],
    },
    "http": {"min_delay_s": 4, "jitter_s": 2, "timeout_s": 20, "retries": 2, "retry_403": 1, "backoff_s": 5,
             "user_agent": ""},
    # parallel fetch: worker processes, total GET requests in flight (split between processes),
    # optional pause per host inside each process; workers=0 is the old sequential pass
    "parallel": {"workers": 3, "max_concurrency": 10, "delay_s": 0},
    "kp_block_alert_after_runs": 6,
    "telegram": {"enabled": True},
    "desktop_notify": True,
    "data_dir": "data",
}

ENV_MAP = {
    "A2179_TARGET_USD": ("target_usd", float),
    "A2179_USD_PER_EUR": ("rates.usd_per_eur", float),
    "A2179_RSD_PER_EUR": ("rates.rsd_per_eur", float),
    "A2179_EMBEDDER": ("embedder.enabled", str),
    "A2179_MODEL": ("embedder.model", str),
    "A2179_WORKERS": ("parallel.workers", int),
    "A2179_MAX_CONCURRENCY": ("parallel.max_concurrency", int),
}


def _merge(a, b):
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def _set(d, dotted, v):
    ks = dotted.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = v


def load_config(path: str = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else PROJECT_ROOT / "config.json"
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if p.exists():
        _merge(cfg, json.loads(p.read_text(encoding="utf-8")))
    for env, (key, typ) in ENV_MAP.items():
        if os.environ.get(env):
            _set(cfg, key, typ(os.environ[env]))
    cfg["_config_file"] = p.name if p.exists() else "(defaults)"
    return cfg


def data_dir(cfg) -> Path:
    d = Path(cfg.get("data_dir", "data"))
    return d if d.is_absolute() else PROJECT_ROOT / d
