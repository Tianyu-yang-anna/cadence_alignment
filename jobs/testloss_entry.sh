#!/bin/bash
# Test-split teacher-forced per-scale loss for a set of base checkpoints
# (data-scaling curve). One single-GPU node, loops over RUNS, runs
# tools/per_sample_scale_bits.py against test.bin, writes one json per run to
# results/scaling/testloss_<run>.json on the Volume. Env: RUNS (space list of
# FULL_NAMEs), TOK_FULL, DATA_NAME, N (windows).
: "${RUNS:?RUNS env var is required}"
TOK_FULL="${TOK_FULL:-vqvae_owt2_1024_pqsh}"
export DATA_NAME="${DATA_NAME:-owt2_gpt2}"
N="${N:-3000}"
export JOB_TAG="testloss"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "testloss RUNS='$RUNS' n=$N"
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

restore_ckpt() {  # $1 = run dir under checkpoints
  local dir="$LOCAL_ROOT/runs/$1" vck="$VOL/checkpoints/$1"
  mkdir -p "$dir"
  local latest=""
  [ -f "$vck/latest.txt" ] && latest=$(tr -d '[:space:]' < "$vck/latest.txt")
  if [ -z "$latest" ] || [ ! -f "$vck/$latest" ]; then
    latest=$(cd "$vck" 2>/dev/null && ls ckpt_step*.pt 2>/dev/null | sort -V | tail -1)
  fi
  [ -n "$latest" ] && [ -f "$vck/$latest" ] || return 1
  cp -f "$vck/$latest" "$dir/$latest"
  printf '%s\n' "$latest" > "$dir/latest.txt"
  cp -f "$vck/config.yaml" "$dir/config.yaml" 2>/dev/null || true
}
restore_ckpt "$TOK_FULL" || { log "no tokenizer ckpt"; push_log; exit 1; }
mkdir -p "$VOL/results/scaling"

for R in $RUNS; do
  if ! restore_ckpt "$R"; then log "skip $R (no ckpt)"; continue; fi
  OUT="$LOCAL_ROOT/testloss_$R.json"
  log "testloss $R"
  (cd "$CODE" && "$PY" tools/per_sample_scale_bits.py \
      --config configs/planner_prefix_owt2_pqsh.yaml \
      --set "run_name=$R" \
      --set "planner.tokenizer_run_dir=$LOCAL_ROOT/runs/$TOK_FULL" \
      --bin "$LOCAL_ROOT/data/$DATA_NAME/test.bin" --n "$N" \
      --out "$OUT") >> "$LOG_LOCAL" 2>&1 \
    && cp -f "$OUT" "$VOL/results/scaling/testloss_$R.json" \
    || { log "testloss $R FAILED"; }
done
log "testloss DONE"
push_log || true
exit 0
