#!/usr/bin/env bash
# Everything the experiments workflow does, after the thin YAML (ci/experiments.yml) has
# checked out the pushed commit. Changing CI = changing files in ci/, never the YAML.
#
# 1. repo variables A2179_CI_* from $A2179_CI_VARS (JSON of all repo variables) -> env,
#    secrets: names recorded present/absent, values masked and scrubbed (ci/redact.py)
# 2. Hugging Face offline unless A2179_CI_ALLOW_DOWNLOAD=true
# 3. ci/setup_venv.sh (persistent venv in $HOME, optional torch + model cache), log ci_out/setup.log
# 4. ci/run_experiments.py in that venv (tests, embedder check, eval, sweep)
# 5. ci/write_results.py with the system python (stdlib only, runs even if the venv broke):
#    results/<UTC date>_<sha>.json + results/<UTC date>_<sha>/ reports, commit [skip ci], push
# Exit code: the first failure among setup, tests, results push (results are written anyway).
set -uo pipefail
cd "$(dirname "$0")/.."
source ci/home_path.sh
# This is the runner's configured work directory, not the venv. Moving a live
# checkout would break Actions bookkeeping; leave it intact and diagnose it.
if [[ "$PWD" == *'/~/'* ]]; then
  echo '::warning::runner work directory contains a literal ~; use an absolute work directory when reconfiguring the runner (see docs/ci.md)'
fi
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

# 1b. which steps this push needs (ci/select_steps.py, ADR-0026): docs only -> done here, shared code
#     -> full regression, anything else -> lint + tests for the touched files + the touched experiment
eval "$("$SYS_PY" ci/select_steps.py)"
if [[ "${A2179_CI_SELECT_MODE:-full}" == "skip" ]]; then
  echo "docs only since ${A2179_CI_SELECT_BASE:-?}: nothing to run, no results commit"
  exit 0
fi
# secrets: TG_BOT_TOKEN / TG_CHAT_ID come by name from the workflow (environment experiment),
# more can ride in the one secret A2179_CI_SECRETS (JSON), masked and exported by ci/secrets_env.py.
# Only names and present/absent are ever recorded (ci_out/secrets.json), never values.
eval "$("$SYS_PY" ci/secrets_env.py)"
unset A2179_CI_SECRETS
"$SYS_PY" -c "import json,sys; sys.path.insert(0,'ci'); import redact; json.dump(redact.presence(), open(sys.argv[1],'w'))" "$OUT/secrets.json"

# 2. model downloads only when allowed (the model cache step in setup_venv.sh opens it for itself)
if [[ "${A2179_CI_ALLOW_DOWNLOAD:-}" != "true" ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
export A2179_CI_DENSE="${A2179_CI_DENSE:-auto}" A2179_CI_STEP_TIMEOUT="${A2179_CI_STEP_TIMEOUT:-1800}"

# 3. venv (A2179_CI_VENV, e.g. your own .venv with torch, or the default one), python for a new
#    venv from A2179_CI_PYTHON (exported above from the repo variables, read by setup_venv.sh).
#    Hugging Face cache: HF_HOME if set, else the standard ~/.cache/huggingface of the runner
#    user, the same cache your own runs with that venv use.
VENV="$(expand_home_path "${A2179_CI_VENV:-$HOME/.cache/a2179-ci/venv}")"
export A2179_CI_VENV="$VENV"
if [[ -n "${A2179_CI_PYTHON:-}" ]]; then
  export A2179_CI_PYTHON="$(expand_home_path "$A2179_CI_PYTHON")"
fi
# A pending EXP-030 request opts this run into the full registry, including heavy models.
if "$SYS_PY" -c 'from ci.run_experiments import runner_request; import sys; sys.exit("exp030" not in (runner_request() or {}).get("steps", []))'; then
  export A2179_CI_MODELS="all,cross-encoder/ms-marco-TinyBERT-L2-v2,cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
  export A2179_CI_EMBEDDERS=all A2179_CI_STEP_TIMEOUT=4500 A2179_CPU_FP32=1
fi
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
