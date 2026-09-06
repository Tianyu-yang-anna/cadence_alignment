#!/bin/bash
# Pretrain the fair-resource T5 encoder on our T5-tokenized corpus.
# Env: RUN_NAME (required), DATA_NAME (owt2_t5), STEPS (150000),
#      GLOBAL_BATCH (256), EXTRA_ARGS.
: "${RUN_NAME:?RUN_NAME env var is required}"
DATA_NAME="${DATA_NAME:-owt2_t5}"
STEPS="${STEPS:-150000}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export JOB_TAG="elfenc-$RUN_NAME${NODE_RANK:+-n$NODE_RANK}"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "t5enc pretrain run=$RUN_NAME data=$DATA_NAME steps=$STEPS batch=$GLOBAL_BATCH"
ensure_env || { log "ABORT: env"; push_log; exit 1; }
ensure_data || { log "ABORT: data"; push_log; exit 1; }
DATA_DIR="$LOCAL_ROOT/data/$DATA_NAME"

FULL_RUN_NAME="t5enc_${DATA_NAME}_$RUN_NAME"
RUN_DIR="$LOCAL_ROOT/runs/$FULL_RUN_NAME"
VCK="$VOL/checkpoints/$FULL_RUN_NAME"
mkdir -p "$RUN_DIR" "$VCK"
latest=$(tr -d '[:space:]' < "$VCK/latest.txt" 2>/dev/null || echo "")
if [ -n "$latest" ] && [ -f "$VCK/$latest" ]; then
  log "resume: restoring ckpts"
  for c in "$VCK"/ckpt_*.pt; do [ -f "$c" ] && cp -f "$c" "$RUN_DIR/$(basename "$c")"; done
  printf '%s\n' "$latest" > "$RUN_DIR/latest.txt"
  [ -f "$VCK/metrics.jsonl" ] && cp -f "$VCK/metrics.jsonl" "$RUN_DIR/metrics.jsonl"
fi

sync_once() {
  for f in metrics.jsonl config.json latest.txt; do
    [ -f "$RUN_DIR/$f" ] && cp -f "$RUN_DIR/$f" "$VCK/$f" 2>/dev/null
  done
  for c in "$RUN_DIR"/ckpt_*.pt; do
    [ -f "$c" ] || continue
    b=$(basename "$c")
    ss=$(stat -c%s "$c" 2>/dev/null || echo 0)
    ds=$(stat -c%s "$VCK/$b" 2>/dev/null || echo -1)
    [ "$ss" != "$ds" ] && cp -f "$c" "$VCK/$b" 2>/dev/null
  done
  tail -1 "$RUN_DIR/metrics.jsonl" 2>/dev/null > "$VOL/status/elfenc-$RUN_NAME-progress.txt" || true
  push_log
}
if [ "${NODE_RANK:-0}" = "0" ]; then
  ( while true; do sleep 300; sync_once; done ) & SIDECAR_PID=$!
else SIDECAR_PID=""; fi
trap 'kill $SIDECAR_PID "$HB_PID" 2>/dev/null' EXIT

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
else NPROC=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' '); fi
MN_NODES="${NUM_NODES:-1}"; MN_RANK="${NODE_RANK:-0}"; MN_MASTER="${MASTER_ADDR:-}"; MN_PORT="${MASTER_PORT:-29511}"
if [ "$MN_NODES" -gt 1 ] && [ -n "$MN_MASTER" ]; then
  LAUNCH=("$VENVS/main/bin/torchrun" --nnodes="$MN_NODES" --node_rank="$MN_RANK" \
          --nproc_per_node="$NPROC" --master_addr="$MN_MASTER" --master_port="$MN_PORT" --max-restarts=0)
elif [ "${NPROC:-1}" -gt 1 ]; then
  LAUNCH=("$VENVS/main/bin/torchrun" --standalone --nproc_per_node="$NPROC")
else LAUNCH=("$PY"); fi
export HF_HOME="$LOCAL_ROOT/hf_home"; mkdir -p "$HF_HOME"
# shellcheck disable=SC2086
(cd "$CODE" && env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u POD_RANK -u NUM_NODES \
    "${LAUNCH[@]}" pretrain_t5enc.py --data_dir "$DATA_DIR" --run_dir "$RUN_DIR" \
    --steps "$STEPS" --global_batch "$GLOBAL_BATCH" $EXTRA_ARGS) >> "$LOG_LOCAL" 2>&1
rc=$?
if [ "${NODE_RANK:-0}" = "0" ]; then kill $SIDECAR_PID 2>/dev/null; sync_once; fi
[ $rc -ne 0 ] && { log "t5enc FAILED rc=$rc (resubmit to resume)"; push_log; exit $rc; }
[ "${NODE_RANK:-0}" = "0" ] && touch "$LOCAL_ROOT/te.done" && cp -f "$LOCAL_ROOT/te.done" "$VOL/status/elfenc-$RUN_NAME.done"
log "t5enc DONE"
