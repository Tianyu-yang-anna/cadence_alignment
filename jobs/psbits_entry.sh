#!/bin/bash
# Per-sample per-scale bits dump (tools/per_sample_scale_bits.py) — the
# content-adaptivity evidence. Env: PLANNER_FULL (required), TOK_FULL,
# CONFIG, DATA_NAME, N (windows, default 2048), TAG.
: "${PLANNER_FULL:?PLANNER_FULL env var is required}"
TOK_FULL="${TOK_FULL:-vqvae_owt2_1024_pqsh}"
CONFIG="${CONFIG:-configs/planner_prefix_owt2_pqsh.yaml}"
export DATA_NAME="${DATA_NAME:-owt2_gpt2}"
N="${N:-2048}"
TAG="${TAG:-}"
export JOB_TAG="psbits-$PLANNER_FULL$TAG"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "psbits planner=$PLANNER_FULL tok=$TOK_FULL n=$N"
trap 'kill "$HB_PID" 2>/dev/null' EXIT
ensure_env || { log "ABORT: env"; exit 1; }
ensure_data || { log "ABORT: data"; exit 1; }

restore_ckpt() {
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
}
restore_ckpt "$TOK_FULL" || { log "no tokenizer ckpt"; push_log; exit 1; }
restore_ckpt "$PLANNER_FULL" || { log "no planner ckpt"; push_log; exit 1; }

OUT="$LOCAL_ROOT/psbits_${PLANNER_FULL}${TAG}.json"
(cd "$CODE" && "$PY" tools/per_sample_scale_bits.py --config "$CONFIG" \
    --set "run_name=$PLANNER_FULL" \
    --set "planner.tokenizer_run_dir=$LOCAL_ROOT/runs/$TOK_FULL" \
    --bin "$LOCAL_ROOT/data/$DATA_NAME/val.bin" --n "$N" \
    --out "$OUT") >> "$LOG_LOCAL" 2>&1 \
  || { log "psbits FAILED"; push_log; exit 1; }
mkdir -p "$VOL/results/psbits"
cp -f "$OUT" "$VOL/results/psbits/" || { log "upload FAILED"; push_log; exit 1; }
log "psbits DONE"
push_log || true
exit 0
