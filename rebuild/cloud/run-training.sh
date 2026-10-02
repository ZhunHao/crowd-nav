#!/usr/bin/env bash
# One trainer process per chunk (resume between chunks) so every chunk is a publish boundary.
# Chunk indices are logical: `--chunks T` trains chunks 0..T-1, and the loop's first T comes from
# the latest local checkpoint (cloud/resume_target.py), so an instance stop/start never restarts at 1.
set -euo pipefail
: "${SHIPNAV_PYTHON:?SHIPNAV_PYTHON required}"
: "${SHIPNAV_SCENARIOS:?SHIPNAV_SCENARIOS required}"
: "${SHIPNAV_OUTPUT:?SHIPNAV_OUTPUT required}"
: "${SHIPNAV_S3_URI:?SHIPNAV_S3_URI required}"
: "${SHIPNAV_SEED:?SHIPNAV_SEED required}"
: "${SHIPNAV_VARIANT:?SHIPNAV_VARIANT required}"
: "${SHIPNAV_CHUNKS:?SHIPNAV_CHUNKS required}"
: "${SHIPNAV_ENVS:?SHIPNAV_ENVS required}"
: "${SHIPNAV_DEVICE:?SHIPNAV_DEVICE required}"
: "${SHIPNAV_SECONDS:?SHIPNAV_SECONDS required}"
export MPLBACKEND=Agg
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$SHIPNAV_OUTPUT"
state=failed
publish() {  # payload first, manifest last: a manifest in S3 means a complete checkpoint
    local dir=$1 name; name=$(basename "$dir")
    aws s3 cp "$dir/policy.zip" "$SHIPNAV_S3_URI/checkpoints/$name/policy.zip" --only-show-errors
    aws s3 cp "$dir/state.pt" "$SHIPNAV_S3_URI/checkpoints/$name/state.pt" --only-show-errors
    aws s3 cp "$dir/rng.json" "$SHIPNAV_S3_URI/checkpoints/$name/rng.json" --only-show-errors
    aws s3 cp "$dir/manifest.json" "$SHIPNAV_S3_URI/checkpoints/$name/manifest.json" --only-show-errors
}
published=  # last checkpoint folder uploaded successfully, so the EXIT trap does not re-upload it
publish_latest() {  # chunk-* and partial-* alike: the checkpoint the trainer would resume from
    local newest
    newest=$("$SHIPNAV_PYTHON" -c 'import sys
from shipnav.training.checkpoints import latest_checkpoint
latest = latest_checkpoint(sys.argv[1])
print(latest if latest else "")' "$SHIPNAV_OUTPUT")
    if [[ -n "$newest" && "$newest" != "$published" ]]; then publish "$newest"; published=$newest; fi
}
upload_status() {  # S3 status.json is the source of truth: bounded retry, then fail
    local attempt
    for attempt in 1 2 3; do
        if aws s3 cp "$SHIPNAV_OUTPUT/run-status.json" "$SHIPNAV_S3_URI/status.json" --only-show-errors; then return 0; fi
        if ((attempt<3)); then sleep "${SHIPNAV_RETRY_SLEEP:-5}"; fi
    done
    return 1
}
finish() {
    local code=$?
    trap '' TERM  # a second TERM (shutdown -h after systemctl stop) must not interrupt the uploads
    if [[ "$state" != complete ]]; then publish_latest || true; fi  # best effort, before the status says so
    printf '{"state":"%s","exit_code":%d}\n' "$state" "$code" > "$SHIPNAV_OUTPUT/run-status.json"
    aws s3 cp "$SHIPNAV_OUTPUT/curve.jsonl" "$SHIPNAV_S3_URI/curve.jsonl" --only-show-errors || true
    if ! upload_status; then
        echo "status.json upload failed after 3 attempts" >&2
        exit $((code==0 ? 1 : code))
    fi
}
trainer_pid=
on_term() {  # systemd stop / RuntimeMaxSec / shutdown: let the trainer save its partial checkpoint, then exit
    trap '' TERM
    state=terminated
    if [[ -n "$trainer_pid" ]]; then
        kill -TERM "$trainer_pid" 2>/dev/null || true  # harmless: the cgroup TERM usually reached it already
        wait "$trainer_pid" 2>/dev/null || true
    fi
    exit 143
}
trap finish EXIT
trap on_term TERM
start=$SECONDS
budget=()
if [[ -n "${SHIPNAV_BUDGET:-}" ]]; then budget=(--budget "$SHIPNAV_BUDGET"); fi
# First target from the latest local checkpoint: none -> 1, partial chunk c -> c+1, complete chunk c -> c+2.
first=$("$SHIPNAV_PYTHON" "$here/resume_target.py" "$SHIPNAV_OUTPUT")
publish_latest  # idempotent: covers a checkpoint saved just before a stop that never reached S3
if ((first>SHIPNAV_CHUNKS)); then state=complete; exit 0; fi  # already complete
for ((target=first; target<=SHIPNAV_CHUNKS; target++)); do
    remaining=$((SHIPNAV_SECONDS-(SECONDS-start)))
    if ((remaining<=300)); then state=deadline; exit 0; fi
    code=0  # --resume is a no-op without checkpoints and continues after a stop/start
    # Background + wait so the TERM trap can act while the trainer runs; wait returns its real exit code.
    timeout --signal=TERM --kill-after=120s "$remaining" "$SHIPNAV_PYTHON" -m shipnav.training.train \
        --scenarios "$SHIPNAV_SCENARIOS" --output "$SHIPNAV_OUTPUT" --seed "$SHIPNAV_SEED" \
        --variant "$SHIPNAV_VARIANT" --chunks "$target" --envs "$SHIPNAV_ENVS" --device "$SHIPNAV_DEVICE" \
        --max-seconds "$remaining" --resume ${budget[@]+"${budget[@]}"} &
    trainer_pid=$!
    wait "$trainer_pid" || code=$?
    trainer_pid=
    publish_latest
    if ((code==3)); then state=stopped; exit 0; fi
    if ((code!=0)); then exit "$code"; fi
done
state=complete
