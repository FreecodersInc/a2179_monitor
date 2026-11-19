"""Secret handling for CI: which secrets are present, and scrubbing their values from any text.

Secret names: the fixed ones the workflow passes (GITHUB_TOKEN, TG_BOT_TOKEN, TG_CHAT_ID) plus
the keys of the optional A2179_CI_SECRETS JSON blob (see ci/secrets_env.py), listed by
ci/run.sh in A2179_CI_SECRET_NAMES. Values are never printed, written or returned: callers get
'present' / 'absent' and text with every value replaced by ***.
"""
import os
import re

FIXED = ("GITHUB_TOKEN", "TG_BOT_TOKEN", "TG_CHAT_ID")
MIN_LEN = 4  # shorter values are not scrubbed (would wipe digits everywhere), reported as short
MASK = "***"


def names(env=os.environ):
    extra = [n for n in re.split(r"[\s,]+", env.get("A2179_CI_SECRET_NAMES", "")) if n]
    return list(dict.fromkeys([*FIXED, *extra]))


def presence(env=os.environ):
    out = {}
    for n in names(env):
        v = env.get(n, "")
        out[n] = "absent" if not v else ("present (short, not scrubbed)" if len(v) < MIN_LEN else "present")
    return out


def _values(env):
    vals = set()
    for n in names(env):
        v = env.get(n, "")
        if len(v) >= MIN_LEN:
            vals.add(v)
            vals.add(v.strip())
    return sorted((v for v in vals if len(v) >= MIN_LEN), key=len, reverse=True)


def scrub(text, env=os.environ):
    if not text:
        return text
    for v in _values(env):
        text = text.replace(v, MASK)
    return text


def scrub_obj(obj, env=os.environ):
    if isinstance(obj, str):
        return scrub(obj, env)
    if isinstance(obj, list):
        return [scrub_obj(x, env) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_obj(v, env) for k, v in obj.items()}
    return obj
