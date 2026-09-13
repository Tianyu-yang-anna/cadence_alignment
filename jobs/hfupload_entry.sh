#!/bin/bash
# One-off: mirror the ENTIRE CADENCE Volume to a private HF dataset repo.
# Reads HF_TOKEN + HF_REPO from env; uploads $VOL (all subdirs: checkpoints/
# data/results/envs/hf/logs) via upload_large_folder (resumable, xet-enabled).
# Source data stays on the Volume (nothing is deleted). Resubmit to resume.
: "${HF_TOKEN:?HF_TOKEN required}"
export HF_REPO="${HF_REPO:-tyang4/cadence_alignment}"
export JOB_TAG="hfupload"
source "$(dirname "${BASH_SOURCE[0]}")/bootstrap.sh"

start_heartbeat
trap 'kill "$HB_PID" 2>/dev/null' EXIT
log "hfupload -> $HF_REPO  (folder=$VOL)"
ensure_env || { log "ABORT: env"; push_log; exit 1; }

export HF_HOME="$LOCAL_ROOT/hf_home"
mkdir -p "$HF_HOME"
"$PY" -m pip install -q -U "huggingface_hub>=0.28" hf_xet >> "$LOG_LOCAL" 2>&1 \
  || { log "pip install huggingface_hub failed"; push_log; exit 1; }

"$PY" - <<'PYEOF' >> "$LOG_LOCAL" 2>&1
import os
from huggingface_hub import HfApi
repo = os.environ["HF_REPO"]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
print(f"[hfupload] repo ready (private dataset): {repo}", flush=True)
api.upload_large_folder(
    repo_id=repo,
    repo_type="dataset",
    folder_path=os.environ["VOL"],
    print_report=True,
)
print("[hfupload] UPLOAD DONE", flush=True)
PYEOF
rc=$?
log "hfupload finished rc=$rc"
push_log || true
exit $rc
