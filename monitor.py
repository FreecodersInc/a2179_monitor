#!/usr/bin/env python3
"""A2179 monitor CLI.

  python monitor.py                 one pass, send alerts (Telegram if TG_BOT_TOKEN/TG_CHAT_ID set)
  python monitor.py --dry-run       fetch and filter, print what would be alerted, write nothing
  python monitor.py --seed          remember current listings silently (no alert flood on first run)
  python monitor.py --list          print every matched listing with price
  python monitor.py --every 60      loop every 60 min (run on YOUR machine only; cron is nicer)
  python monitor.py --test-alert    send one test message through the configured channels
  python monitor.py check             same as --check-embedder
  python monitor.py --check-embedder   diagnose the optional embedder (versions, cache, load, test encode)
  python monitor.py --require-embedder fail with exit code 3 instead of falling back to TF-IDF
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from a2179.alerts import Alerter  # noqa: E402
from a2179.config import load_config, data_dir  # noqa: E402
from a2179.obs import setup_logging  # noqa: E402
from a2179.embedder import EmbedderUnavailable, check_embedder  # noqa: E402
from a2179.run import run_once, prepare_similarity  # noqa: E402


def load_dotenv(path: Path):
    """Minimal .env reader: KEY=VALUE lines, does not override variables already set."""
    import os
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="MacBook Air A2179 display/lid market monitor (Serbia)")
    ap.add_argument("--config", help="path to config.json (default: ./config.json, else built-in defaults)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-embed", action="store_true", help="skip the embedder, use the TF-IDF fallback")
    ap.add_argument("--require-embedder", action="store_true",
                    help="exit with code 3 instead of falling back when the embedder cannot be used")
    ap.add_argument("--check-embedder", action="store_true",
                    help="print versions, cache dir, load time and a test encode, then exit (0 ok, 3 failed)")
    ap.add_argument("--every", type=float, metavar="MIN")
    ap.add_argument("--time-budget", type=float, metavar="SEC", help="stop fetching after SEC seconds")
    ap.add_argument("--test-alert", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["check"]:  # short alias: python monitor.py check
        argv = ["--check-embedder"] + argv[1:]
    a = ap.parse_args(argv)
    load_dotenv(ROOT / ".env")

    cfg = load_config(a.config)
    log = setup_logging(data_dir(cfg) / "logs", a.verbose)
    if a.test_alert:
        al = Alerter(cfg, data_dir(cfg), log)
        print(al.send(["✅ A2179 monitor: тестовое сообщение"]))
        print("telegram configured:", al.tg)
        return 0
    if a.check_embedder:
        return check_embedder(cfg)
    if a.require_embedder and a.no_embed:
        ap.error("--require-embedder and --no-embed contradict each other")
    try:
        sim = prepare_similarity(cfg, log, a.no_embed, a.require_embedder, a.verbose)
    except EmbedderUnavailable as err:
        print(err.console().rsplit("\n", 1)[0], file=sys.stderr)
        print("[embedder] эмбеддер обязателен (--require-embedder или embedder.enabled=required): выход с кодом 3", file=sys.stderr)
        return 3
    while True:
        r = run_once(cfg, log, dry_run=a.dry_run, seed=a.seed, time_budget_s=a.time_budget, similarity=sim)
        m = r["metrics"]["counters"]
        print(f"filter: regex+{r['mode']} | listings: {len(r['listings'])} | matched: {len(r['matched'])} | "
              f"alerts: {len(r['messages'])} | KP: {r['kp']} | fetch errors: {m.get('fetch.errors', 0)}"
              f"/{m.get('fetch.requests', 0)} | {r['metrics']['duration_s']} s")
        if a.list:
            for l, usd, d in sorted(r["matched"], key=lambda x: (x[1] is None, x[1] or 0)):
                p = f"{l.price:,.0f} {l.currency}".replace(",", " ") if l.price is not None else "?"
                u = f"~${usd:.0f}" if usd is not None else ""
                st = " (нет в наличии)" if l.in_stock is False else ""
                print(f"  {p:>12} {u:>6}  {l.title[:70]}{st}\n               {l.url}")
        if not a.every:
            return 0
        time.sleep(max(1.0, a.every * 60))


if __name__ == "__main__":
    sys.exit(main())
