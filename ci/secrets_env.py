"""Expand the optional secret A2179_CI_SECRETS (one JSON object {"NAME": "value", ...}) into
environment variables for ci/run.sh, so a new secret needs no workflow change: add the key to
that one secret in the GitHub environment `experiment`.

Prints shell lines: `echo "::add-mask::<value>"` for every value (GitHub masks only whole
secrets, not parts of a JSON blob), then `export NAME=<value>`, then A2179_CI_SECRET_NAMES.
Only names TG_*, SECRET_*, A2179_SECRET_* are taken, so the blob cannot override PATH, HOME,
GIT_*, GITHUB_* or the A2179_CI_* settings. Nothing here is written to disk.
"""
import json
import os
import re
import shlex
import sys

OK_NAME = re.compile(r"^(TG|SECRET|A2179_SECRET)_[A-Z0-9_]{1,60}$")


def lines(raw: str, env=os.environ):
    out, names = [], []
    if raw:
        try:
            d = json.loads(raw)
        except ValueError:
            d = None
        if not isinstance(d, dict):
            out.append("echo '::warning::A2179_CI_SECRETS is not a JSON object, ignored'")
            d = {}
        for k, v in sorted(d.items()):
            if not OK_NAME.match(str(k)) or not isinstance(v, (str, int, float)):
                out.append(f"echo {shlex.quote(f'::warning::A2179_CI_SECRETS key {k!s:.40} ignored (allowed TG_*, SECRET_*, A2179_SECRET_*)')}")
                continue
            if env.get(k):  # an explicitly passed secret wins over the blob
                names.append(k)
                continue
            v = str(v)
            out.append(f"echo {shlex.quote('::add-mask::' + v)}")
            out.append(f"export {k}={shlex.quote(v)}")
            names.append(k)
    out.append(f"export A2179_CI_SECRET_NAMES={shlex.quote(' '.join(names))}")
    return out


if __name__ == "__main__":
    print("\n".join(lines(os.environ.get("A2179_CI_SECRETS", ""))))
    sys.exit(0)
