"""Telegram integration test (CI step tg_integration, ci/run_experiments.py).

Runs only with TG_BOT_TOKEN and TG_CHAT_ID in the environment (GitHub environment `experiment`),
skips cleanly otherwise. Read only by default: getMe (token valid) and getChat (the bot can see
the chat). It sends one message only when A2179_CI_TG_SEND=1, through the app's own Alerter.
The token is part of the API URL, so no URL, response body or exception text is printed: only
status codes and Telegram's `ok` / `description` fields, scrubbed again by ci/redact.py.
Not under tests/, so normal unit test discovery never touches the network.
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ci"))
import redact  # noqa: E402

TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()
CHAT = os.environ.get("TG_CHAT_ID", "").strip()
API = "https://api.telegram.org"


def call(method, **params):
    """(http status, ok, description) without ever exposing the URL."""
    import requests
    try:
        r = requests.get(f"{API}/bot{TOKEN}/{method}", params=params, timeout=20)
    except requests.RequestException as e:
        return None, False, f"network error {type(e).__name__}"
    try:
        j = r.json()
    except ValueError:
        j = {}
    return r.status_code, bool(j.get("ok")), redact.scrub(str(j.get("description") or ""))[:200]


@unittest.skipUnless(TOKEN and CHAT, "TG_BOT_TOKEN / TG_CHAT_ID absent")
class TestTelegram(unittest.TestCase):
    def test_token_valid(self):
        status, ok, desc = call("getMe")
        self.assertTrue(ok, f"getMe: status {status} {desc}")

    def test_bot_sees_chat(self):
        status, ok, desc = call("getChat", chat_id=CHAT)
        self.assertTrue(ok, f"getChat: status {status} {desc}")

    @unittest.skipUnless(os.environ.get("A2179_CI_TG_SEND") == "1", "sending is off (A2179_CI_TG_SEND=1 to send)")
    def test_send_through_alerter(self):
        import tempfile
        from a2179.alerts import Alerter
        from a2179.config import load_config
        from a2179.obs import setup_logging
        with tempfile.TemporaryDirectory() as d:
            cfg = load_config(str(Path(d) / "none.json"))
            cfg["desktop_notify"] = False
            al = Alerter(cfg, Path(d), setup_logging(Path(d) / "logs"))
            sha = os.environ.get("GITHUB_SHA", "")[:7] or "local"
            res = al.send([f"✅ CI integration test, commit {sha}"])
        self.assertEqual(res["telegram"], 1, res)


if __name__ == "__main__":
    unittest.main()
