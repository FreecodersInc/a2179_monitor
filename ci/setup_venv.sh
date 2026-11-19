#!/usr/bin/env bash
# Persistent venv for the self-hosted runner, outside the checkout so torch (GBs) is installed once.
# Installs requirements.txt only when it changed (sha256 stamp). torch only on opt in
# (A2179_CI_INSTALL_TORCH=cu126), requirements-embed.txt and the model cache only when torch is
# there. See docs/ci.md. No sudo, nothing outside $HOME.
set -euo pipefail
source "$(dirname "$0")/home_path.sh"
VENV="$(expand_home_path "${A2179_CI_VENV:-$HOME/.cache/a2179-ci/venv}")"
PY="$(expand_home_path "${A2179_CI_PYTHON:-python3}")"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "creating venv $VENV with $("$PY" --version)"
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
has() { "$VENV/bin/python" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)"; }
stamp_install requirements.txt

# Optional, opt in: A2179_CI_INSTALL_TORCH=cu126 (or cu128, cpu, rocm6.2) installs that torch
# build into this venv once, when torch is missing. Never touches a torch that is already there.
TORCH="${A2179_CI_INSTALL_TORCH:-}"
if [[ -n "$TORCH" ]]; then
  if [[ ! "$TORCH" =~ ^(cu[0-9]{3}|cpu|rocm[0-9]+(\.[0-9]+)?)$ ]]; then
    echo "::error::A2179_CI_INSTALL_TORCH=$TORCH: expected cu126, cu128, cpu or rocm6.2"
    exit 3
  fi
  if has torch; then
    echo "torch already in $VENV, A2179_CI_INSTALL_TORCH ignored"
  else
    echo "installing torch ($TORCH build) into $VENV, a few GB, once"
    "$VENV/bin/python" -m pip install --quiet torch --index-url "https://download.pytorch.org/whl/$TORCH"
  fi
fi
if has torch; then
  stamp_install requirements-embed.txt
  # model into the HF cache (HF_HOME or ~/.cache/huggingface), the only download of the run
  # on by default: a no-op when the model is already in the cache, downloads it once otherwise
  case "${A2179_CI_CACHE_MODEL:-1}" in
    # a failed download never fails setup: dense then falls back like any missing model
    1|true|yes) HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$VENV/bin/python" ci/cache_model.py \
                  || echo "::warning::model cache step failed (exit $?), dense steps will fall back" ;;
    *) echo "model cache step off (A2179_CI_CACHE_MODEL=$A2179_CI_CACHE_MODEL)" ;;
  esac
else
  echo "no torch in $VENV: dense steps will be skipped (set repo variable A2179_CI_INSTALL_TORCH=cu126, see docs/ci.md)"
fi
"$VENV/bin/python" --version
