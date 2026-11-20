#!/usr/bin/env bash
# EXP-029 / H-11 only: CUDA vs CPU on the self-hosted runner. Does not hide the GPU.
# Does not rerun the full experiment suite. Writes results and pushes with [skip ci].
set -uo pipefail
cd "$(dirname "$0")/.."
source ci/home_path.sh
OUT="${A2179_CI_OUT:-ci_out_h11}"
mkdir -p "$OUT"
SYS_PY="$(command -v python3 || command -v python || true)"
if [[ -z "$SYS_PY" ]]; then
  echo "::error::no python3 on the runner PATH"
  exit 2
fi

if [[ -n "${A2179_CI_VARS:-}" && "${A2179_CI_VARS}" != "null" ]]; then
  eval "$("$SYS_PY" ci/vars_env.py)"
fi
eval "$("$SYS_PY" ci/secrets_env.py)"
unset A2179_CI_SECRETS

# H-11 needs CUDA visible. Do not set CUDA_VISIBLE_DEVICES= and do not force A2179_CI_DEVICE=cpu.
unset CUDA_VISIBLE_DEVICES || true
export A2179_CI_DEVICE=cuda
# The monitor MiniLM should already be in the runner cache. If it is not, allow one fetch.
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

VENV="$(expand_home_path "${A2179_CI_VENV:-$HOME/.cache/a2179-ci/venv}")"
export A2179_CI_VENV="$VENV"
if [[ ! -x "$VENV/bin/python" ]]; then
  t0=$(date +%s)
  bash ci/setup_venv.sh > >(tee "$OUT/setup.log") 2>&1
  setup_rc=$?
  "$SYS_PY" - "$OUT" "$setup_rc" "$(( $(date +%s) - t0 ))" <<'PY'
import json, sys
sys.path.insert(0, "ci")
import redact
out, rc, dur = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
log = redact.scrub(open(f"{out}/setup.log", encoding="utf-8", errors="replace").read())
open(f"{out}/setup.log", "w", encoding="utf-8").write(log)
json.dump({"name": "setup_venv", "cmd": ["bash", "ci/setup_venv.sh"], "exit_code": rc, "ok": rc == 0,
           "duration_s": dur, "log_tail": log[-3000:]}, open(f"{out}/setup.json", "w"))
PY
  if [[ $setup_rc -ne 0 ]]; then
    echo "::error::setup_venv failed for H-11"
    "$SYS_PY" ci/write_results.py --ci-out "$OUT" --results-dir "${A2179_CI_RESULTS_DIR:-results}" --attach-reports --commit --push --branch main || true
    exit "$setup_rc"
  fi
fi

d="$OUT/reports/runner_exp/runner-exp-2026-11-20-h11"
mkdir -p "$d"
t0=$(date +%s)
set +e
"$VENV/bin/python" eval/exp029_h11.py --out "$d"
exp_rc=$?
set -e
dur=$(( $(date +%s) - t0 ))
"$SYS_PY" - "$OUT" "$exp_rc" "$dur" <<'PY'
import json, sys
out, rc, dur = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
step = {"name": "h11", "cmd": ["eval/exp029_h11.py"], "exit_code": rc, "ok": rc == 0,
        "duration_s": dur, "advisory": True, "stderr_tail": ""}
json.dump({"env": {}, "dense": {}, "secrets": {}, "steps": [step]}, open(f"{out}/steps.json", "w"))
print(f"h11 step exit {rc} in {dur}s")
PY

res_rc=0
push_args=(--commit --push --branch main)
[[ -n "${A2179_CI_NO_PUSH:-}" ]] && push_args=()
"$SYS_PY" ci/write_results.py --ci-out "$OUT" --results-dir "${A2179_CI_RESULTS_DIR:-results}" --attach-reports "${push_args[@]}" || res_rc=$?
# The H-11 job itself stays green when the script exits 0 (including a clean 'no CUDA' block).
# A non-zero from the python script is advisory for results, but we still surface it.
if [[ $exp_rc -ne 0 ]]; then
  exit "$exp_rc"
fi
exit "$res_rc"
