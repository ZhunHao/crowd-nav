"""First --chunks target for run-training.sh, from the latest local checkpoint (Ruling B).

The trainer's chunk indices are logical and 0-based; `--chunks T` trains chunks 0..T-1. So after
an instance stop/start the runner must not restart its target loop at 1:
  no checkpoint                   -> 1
  partial checkpoint, chunk c     -> c+1   (the trainer re-enters chunk c and trains the remainder)
  complete checkpoint, chunk c    -> c+2   (the next logical chunk is c+1, i.e. target c+2)
A first target above the configured chunk count means the run is already complete.
"""
from pathlib import Path
import sys

from shipnav.training.checkpoints import latest_checkpoint, verify_checkpoint


def first_target(output):
    latest = latest_checkpoint(Path(output))
    if latest is None:
        return 1
    manifest = verify_checkpoint(latest)
    try:
        chunk, complete = manifest['chunk'], manifest['complete_chunk']
    except KeyError as error:
        raise ValueError(f'Checkpoint {latest} lacks {error} in its manifest') from error
    if not isinstance(chunk, int) or chunk < 0 or not isinstance(complete, bool):
        raise ValueError(f'Checkpoint {latest} has an invalid chunk or complete_chunk field')
    return chunk + (2 if complete else 1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: resume_target.py OUTPUT_DIR')
    print(first_target(sys.argv[1]))
