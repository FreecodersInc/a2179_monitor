#!/usr/bin/env bash
# Persistent venv for the self-hosted runner, outside the checkout so torch (GBs) is installed once.
# Installs requirements.txt only when it changed (sha256 stamp). requirements-embed.txt is installed
# only when torch is already in the venv: CI never picks a torch build for you (CUDA build by hand,
# see docs/ci.md). Prints the venv bin dir to $GITHUB_PATH. No sudo, nothing outside $HOME.
set -euo pipefail
VENV="${A2179_CI_VENV:-$HOME/.cache/a2179-ci/venv}"
PY="${A2179_CI_PYTHON:-python3}"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "creating venv $VENV with $($PY --version)"
  mkdir -p "$(dirname "$VENV")"
  "$PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
fi
stamp_install() {  # $1 requirements file
  local req="$1" stamp="$VENV/.stamp_$(basename "$1")" sum
  sum="$(sha256sum "$req" | cut -d' ' -f1)"
  if [[ "$(cat "$stamp" 2>/dev/null || true)" != "$sum" ]]; then
    echo "pip install -r $req"
    "$VENV/bin/python" -m pip install --quiet -r "$req"
    echo "$sum" > "$stamp"
  else
    echo "$req unchanged, skip pip"
  fi
}
stamp_install requirements.txt
if "$VENV/bin/python" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"; then
  stamp_install requirements-embed.txt
else
  echo "no torch in $VENV: dense steps will be skipped (see docs/ci.md to install the CUDA build)"
fi
if [[ -n "${GITHUB_PATH:-}" ]]; then echo "$VENV/bin" >> "$GITHUB_PATH"; fi
"$VENV/bin/python" --version
