#!/bin/bash
# Post-hoc extra-metrics pass (tools/eval_extra.py): MAUVE@1024 + Gen-PPL +
# unigram entropy over EXISTING gens files (no regeneration). Reads
# tools/evalx_manifest.txt (results-relative paths), restores each gens file
# from the Volume (git snapshots exclude gens_*.jsonl), shards files across
# the node's GPUs, writes gens_*.extra.json back next to the source.
# Env: MANIFEST (default tools/evalx_manifest.txt), JUDGE (gpt2-large).
MANIFEST="${MANIFEST:-tools/evalx_manifest.txt}"
JUDGE="${JUDGE:-gpt2-large}"
export JOB_TAG="evalx"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
log "evalx judge=$JUDGE manifest=$MANIFEST"
ensure_env || { log "ABORT: env"; exit 1; }

export HF_HOME="$LOCAL_ROOT/hf_home"
mkdir -p "$HF_HOME"
# pre-warm the judge + mauve featurizer once BEFORE the parallel loop
# (concurrent first-downloads race on the HF cache)
"$PY" - <<PYEOF || { log "ABORT: judge prefetch"; push_log; exit 1; }
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
GPT2TokenizerFast.from_pretrained("$JUDGE")
GPT2LMHeadModel.from_pretrained("$JUDGE")
PYEOF
log "judge prefetched"

NPROC=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
[ "${NPROC:-0}" -ge 1 ] || NPROC=1
mkdir -p "$LOCAL_ROOT/evalx"
fail=0
i=0
while IFS= read -r rel; do
  case "$rel" in ''|'#'*) continue ;; esac
  src="$VOL/results/$rel"
  [ -f "$src" ] || { log "MISSING $rel"; fail=1; continue; }
  loc="$LOCAL_ROOT/evalx/$(echo "$rel" | tr '/' '_')"
  cp -f "$src" "$loc"
  out="${loc%.jsonl}.extra.json"
  dev=$((i % NPROC))
  (
    cd "$CODE" && "$PY" tools/eval_extra.py --gen "$loc" --out "$out" \
        --device "$dev" --judge "$JUDGE" \
        > "$LOCAL_ROOT/evalx/log_$(basename "$loc").txt" 2>&1 \
      && cp -f "$out" "$VOL/results/${rel%.jsonl}.extra.json" \
      || { echo "FAIL $rel" >> "$LOCAL_ROOT/evalx/failures.txt"; }
  ) &
  i=$((i + 1))
  # keep at most one job per GPU in flight
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do wait -n; done
done < "$CODE/$MANIFEST"
wait
if [ -f "$LOCAL_ROOT/evalx/failures.txt" ]; then
  log "evalx FAILURES: $(tr '\n' ' ' < "$LOCAL_ROOT/evalx/failures.txt")"
  cat "$LOCAL_ROOT"/evalx/log_*.txt | tail -80 >> "$LOG_LOCAL"
  push_log
  exit 1
fi
log "evalx DONE ($i files)"
push_log
