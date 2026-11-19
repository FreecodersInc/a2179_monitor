"""Print `export K=V` lines for the A2179_CI_* / HF_* keys of $A2179_CI_VARS (JSON of all repo
variables, passed by the workflow as toJSON(vars)). Other keys are ignored, values are shell-quoted."""
import json
import os
import re
import shlex
import sys

KEY = re.compile(r"^(A2179_CI_[A-Z0-9_]+|HF_[A-Z0-9_]+)$")


def lines(raw: str):
    try:
        d = json.loads(raw or "{}") or {}
    except ValueError:
        return ["echo '::warning::A2179_CI_VARS is not JSON, ignored'"]
    return [f"export {k}={shlex.quote(str(v))}" for k, v in sorted(d.items())
            if KEY.match(k) and k != "A2179_CI_VARS"]


if __name__ == "__main__":
    print("\n".join(lines(os.environ.get("A2179_CI_VARS", ""))))
    sys.exit(0)
