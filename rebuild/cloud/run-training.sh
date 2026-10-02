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
publish_latest() {  # chunk-* and partial-* alike: the checkpoint the trainer would resume from
    local newest
    newest=$("$SHIPNAV_PYTHON" -c 'import sys
from shipnav.training.checkpoints import latest_checkpoint
latest = latest_checkpoint(sys.argv[1])
print(latest if latest else "")' "$SHIPNAV_OUTPUT")
    if [[ -n "$newest" ]]; then publish "$newest"; fi
}
finish() {
    local code=$?
    printf '{"state":"%s","exit_code":%d}\n' "$state" "$code" > "$SHIPNAV_OUTPUT/run-status.json"
    aws s3 cp "$SHIPNAV_OUTPUT/curve.jsonl" "$SHIPNAV_S3_URI/curve.jsonl" --only-show-errors || true
    aws s3 cp "$SHIPNAV_OUTPUT/run-status.json" "$SHIPNAV_S3_URI/status.json" --only-show-errors
}
trap finish EXIT
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
    timeout --signal=TERM --kill-after=120s "$remaining" "$SHIPNAV_PYTHON" -m shipnav.training.train \
        --scenarios "$SHIPNAV_SCENARIOS" --output "$SHIPNAV_OUTPUT" --seed "$SHIPNAV_SEED" \
        --variant "$SHIPNAV_VARIANT" --chunks "$target" --envs "$SHIPNAV_ENVS" --device "$SHIPNAV_DEVICE" \
        --max-seconds "$remaining" --resume ${budget[@]+"${budget[@]}"} || code=$?
    publish_latest
    if ((code==3)); then state=stopped; exit 0; fi
    if ((code!=0)); then exit "$code"; fi
done
state=complete
