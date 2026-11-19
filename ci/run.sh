#!/usr/bin/env bash
# Everything the experiments workflow does, after the thin YAML (ci/experiments.yml) has
# checked out the pushed commit. Changing CI = changing files in ci/, never the YAML.
#
# 1. repo variables A2179_CI_* from $A2179_CI_VARS (JSON of all repo variables) -> env
# 2. Hugging Face offline unless A2179_CI_ALLOW_DOWNLOAD=true
# 3. ci/setup_venv.sh (persistent venv in $HOME), its output in ci_out/setup.log
# 4. ci/run_experiments.py in that venv (tests, embedder check, eval, sweep)
# 5. ci/write_results.py with the system python (stdlib only, runs even if the venv broke):
#    results/<UTC date>_<sha>.json + results/<UTC date>_<sha>/ reports, commit [skip ci], push
# Exit code: the first failure among setup, tests, results push (results are written anyway).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${A2179_CI_OUT:-ci_out}"
mkdir -p "$OUT"
SYS_PY="$(command -v python3 || command -v python || true)"
if [[ -z "$SYS_PY" ]]; then
  echo "::error::no python3 on the runner PATH (install python3 for the runner user)"
  exit 2
fi

# 1. repo variables -> env (only A2179_CI_* and HF_* keys, values as given)
if [[ -n "${A2179_CI_VARS:-}" && "${A2179_CI_VARS}" != "null" ]]; then
  eval "$("$SYS_PY" ci/vars_env.py)"
fi
# 2. model downloads only when allowed
if [[ "${A2179_CI_ALLOW_DOWNLOAD:-}" != "true" ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
export A2179_CI_DENSE="${A2179_CI_DENSE:-auto}" A2179_CI_STEP_TIMEOUT="${A2179_CI_STEP_TIMEOUT:-1800}"

# 3. venv
VENV="${A2179_CI_VENV:-$HOME/.cache/a2179-ci/venv}"
export A2179_CI_VENV="$VENV"
t0=$(date +%s)
bash ci/setup_venv.sh > >(tee "$OUT/setup.log") 2>&1
setup_rc=$?
"$SYS_PY" - "$OUT" "$setup_rc" "$(( $(date +%s) - t0 ))" <<'PY'
import json, sys
out, rc, dur = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
log = open(f"{out}/setup.log", encoding="utf-8", errors="replace").read()
json.dump({"name": "setup_venv", "cmd": ["bash", "ci/setup_venv.sh"], "exit_code": rc, "ok": rc == 0,
           "duration_s": dur, "log_tail": log[-3000:]}, open(f"{out}/setup.json", "w"))
PY

# 4. experiments
exp_rc=0
if [[ $setup_rc -eq 0 ]]; then
  "$VENV/bin/python" ci/run_experiments.py --out "$OUT" || exp_rc=$?
else
  echo "::error::setup_venv failed (exit $setup_rc), see $OUT/setup.log; writing results without experiments"
fi

# 5. results (system python, the git auth header comes from the workflow env, never stored)
res_rc=0
push_args=(--commit --push --branch main)
[[ -n "${A2179_CI_NO_PUSH:-}" ]] && push_args=()   # local dry run: write files only
"$SYS_PY" ci/write_results.py --ci-out "$OUT" --results-dir "${A2179_CI_RESULTS_DIR:-results}" --attach-reports "${push_args[@]}" || res_rc=$?

for rc in $setup_rc $exp_rc $res_rc; do
  [[ $rc -ne 0 ]] && exit "$rc"
done
exit 0
