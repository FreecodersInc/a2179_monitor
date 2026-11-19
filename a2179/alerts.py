"""Alert channels: Telegram bot (token/chat id from env), console, alerts.log, desktop notification."""
import logging
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import requests

from .obs import ev

TG_LIMIT = 3900


def format_event(kind: str, listing, price_usd, extra: str = "") -> str:
    icon = {"new": "🆕 Новое", "price_drop": "📉 Цена упала", "below_target": "🎯 Ниже цели",
            "kp_blocked": "⚠️ KP"}.get(kind, kind)
    price = "цена не указана"
    if listing.price is not None:
        price = f"{listing.price:,.0f} {listing.currency or ''}".replace(",", " ").strip()
        if price_usd is not None:
            price += f" (~${price_usd:.0f})"
    stock = "" if listing.in_stock is not False else " · нет в наличии"
    loc = f" · {listing.location}" if listing.location else ""
    return f"{icon}: {listing.title}\n{price}{stock}{loc} · {listing.source}{extra}\n{listing.url}"


class Alerter:
    def __init__(self, cfg: dict, data_dir: Path, log, dry_run: bool = False):
        self.cfg, self.log, self.dry = cfg, log, dry_run
        self.log_path = data_dir / "alerts.log"
        self.token = os.environ.get("TG_BOT_TOKEN", "").strip()
        self.chat = os.environ.get("TG_CHAT_ID", "").strip()
        self.tg = bool(cfg.get("telegram", {}).get("enabled", True) and self.token and self.chat)

    def send(self, messages: list, metrics=None) -> dict:
        res = {"console": 0, "telegram": 0, "telegram_errors": 0, "desktop": 0}
        if not messages:
            return res
        text = "\n\n".join(messages)
        print(("[DRY RUN, nothing sent]\n" if self.dry else "") + text + "\n")
        res["console"] = len(messages)
        if self.dry:
            return res
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text.replace("\n", " | ") + "\n")
        if self.tg:
            for chunk in _chunks(messages):
                if self._telegram(chunk):
                    res["telegram"] += 1
                else:
                    res["telegram_errors"] += 1
        if self.cfg.get("desktop_notify", True):
            if self._desktop(f"A2179: {len(messages)} событ.", messages[0].split("\n")[0]):
                res["desktop"] = 1
        if metrics:
            for k, v in res.items():
                metrics.inc(f"alerts.{k}", v)
        return res

    def _telegram(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        for attempt in range(3):
            try:
                r = requests.post(url, data={"chat_id": self.chat, "text": text,
                                             "disable_web_page_preview": "true"}, timeout=20)
                if r.status_code == 200:
                    return True
                ev(self.log, "telegram_error", logging.WARNING, status=r.status_code, body=r.text[:200])
                if r.status_code != 429 and r.status_code < 500:
                    return False
            except Exception as e:
                ev(self.log, "telegram_error", logging.WARNING, error=f"{type(e).__name__}"[:100])
            time.sleep(2 * (attempt + 1))
        return False

    def _desktop(self, title: str, body: str) -> bool:
        try:
            sysname = platform.system()
            if sysname == "Linux" and shutil.which("notify-send"):
                subprocess.run(["notify-send", title, body], timeout=5, check=False)
                return True
            if sysname == "Darwin" and shutil.which("osascript"):
                script = f'display notification {_as(body)} with title {_as(title)}'
                subprocess.run(["osascript", "-e", script], timeout=5, check=False)
                return True
        except Exception:
            pass
        return False


def _as(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"')[:200] + '"'


def _chunks(messages):
    cur = ""
    for m in messages:
        if cur and len(cur) + len(m) + 2 > TG_LIMIT:
            yield cur
            cur = ""
        cur = (cur + "\n\n" + m) if cur else m[:TG_LIMIT]
    if cur:
        yield cur
