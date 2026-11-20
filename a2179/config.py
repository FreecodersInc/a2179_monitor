"""Config: built-in defaults <- target profile defaults (targets/<name>.json) <- config.json <- environment."""
import copy
import json
import os
from pathlib import Path

from . import PROJECT_ROOT
from .target import load_target

# Generic defaults. Everything specific to one search case (sources, target price, thresholds,
# accept_* options, regex rules, prototypes) comes from the target profile, see a2179/target.py.
BASE = {
    "target": "a2179_screen",   # targets/<name>.json: rules, prototypes, sources, price, thresholds
    "rates": {"usd_per_eur": 1.16, "rsd_per_eur": 117.2},
    "price_drop_min_pct": 3.0,
    "alert_on": {"new": True, "price_drop": True, "below_target": True},
    "embedder": {"enabled": "auto",
                 "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 "local_only": False,
                 "device": "auto",          # auto (cuda if available, else cpu) | cuda | cpu | cuda:1 | mps
                 "backend": "torch",        # torch | onnx | openvino (sentence-transformers >= 3.2)
                 "probe": True,             # load + test encode in a child process first (crash-safe)
                 "probe_timeout_s": 600},
    "http": {"min_delay_s": 4, "jitter_s": 2, "timeout_s": 20, "retries": 3, "retry_403": 1, "backoff_s": 5,
             "backoff_max_s": 60, "user_agent": ""},
    # parallel fetch: worker processes, GET requests in flight in total (one limiter shared by all
    # processes), adaptive delay per host on 429 (AIMD, see README); workers=0 is the old sequential pass
    "parallel": {"workers": 3, "max_concurrency": 10, "delay_s": 0, "delay_min_s": 0, "delay_max_s": 30,
                 "delay_up": 2.0, "delay_down_s": 0.05, "delay_on_429_s": 0.5},
    # searches are followed to the last page; max_pages is only a high safety ceiling
    "pagination": {"enabled": True, "max_pages": 500, "kp_page_param": "page"},
    # shop cards that matched once are re-checked every pass (data/tracked.shelve)
    "tracked": {"enabled": True, "file": "tracked.shelve", "drop_after_misses": 3},
    "kp_block_alert_after_runs": 6,
    "telegram": {"enabled": True},
    "desktop_notify": True,
    "data_dir": "data",
}



def with_target(base: dict, target_spec) -> dict:
    """base <- defaults of the target profile."""
    cfg = copy.deepcopy(base)
    t = load_target(target_spec)
    _merge(cfg, copy.deepcopy(t.defaults))
    cfg["target"] = target_spec if isinstance(target_spec, str) else t.name
    return cfg


ENV_MAP = {
    "A2179_TARGET": ("target", str),
    "A2179_TARGET_USD": ("target_usd", float),
    "A2179_TARGET_PRICE": ("target_price.amount", float),
    "A2179_TARGET_CURRENCY": ("target_price.currency", str),
    "A2179_USD_PER_EUR": ("rates.usd_per_eur", float),
    "A2179_RSD_PER_EUR": ("rates.rsd_per_eur", float),
    "A2179_EMBEDDER": ("embedder.enabled", str),
    "A2179_MODEL": ("embedder.model", str),
    "A2179_DEVICE": ("embedder.device", str),
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
    """defaults <- target profile defaults <- config.json <- environment variables.
    The target is picked by A2179_TARGET, else "target" in config.json, else a2179_screen."""
    p = Path(path) if path else PROJECT_ROOT / "config.json"
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    user = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    spec = os.environ.get("A2179_TARGET") or user.get("target") or BASE["target"]
    cfg = with_target(BASE, spec)
    if "target_usd" in user and "target_price" not in user:
        cfg.pop("target_price", None)   # T-005: an explicit old-style target_usd still wins over the profile
    _merge(cfg, user)
    cfg["target"] = spec
    if os.environ.get("A2179_TARGET_USD") and not os.environ.get("A2179_TARGET_PRICE"):
        cfg.pop("target_price", None)
    for env, (key, typ) in ENV_MAP.items():
        if os.environ.get(env):
            _set(cfg, key, typ(os.environ[env]))
    cfg["_config_file"] = p.name if p.exists() else "(defaults)"
    return cfg


def data_dir(cfg) -> Path:
    d = Path(cfg.get("data_dir", "data"))
    return d if d.is_absolute() else PROJECT_ROOT / d


# Full defaults for the default target (kept for code and tests that read DEFAULTS directly).
DEFAULTS = with_target(BASE, BASE["target"])
