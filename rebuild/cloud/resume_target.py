"""First --chunks target for run-training.sh, from the latest local checkpoint (Ruling B).

The trainer's chunk indices are logical and 0-based; `--chunks T` trains chunks 0..T-1. So after
an instance stop/start the runner must not restart its target loop at 1:
  no checkpoint                   -> 1
  partial checkpoint, chunk c     -> c+1   (the trainer re-enters chunk c and trains the remainder)
  complete checkpoint, chunk c    -> c+2   (the next logical chunk is c+1, i.e. target c+2)
A first target above the configured chunk count means the run is already complete, but only when a
complete chunk-(chunks-1) checkpoint exists; a partial beyond --chunks is an error, not "complete".

The latest checkpoint must also belong to this run: with `expected` (variant, seed, envs, chunk_steps,
budget) a foreign checkpoint in the output directory is refused before anything is published or reported.
"""
from pathlib import Path
import argparse
import sys

from shipnav.training.checkpoints import DEFAULT_CHUNK_STEPS, latest_checkpoint, verify_checkpoint

RUN_FIELDS = ('variant', 'seed', 'envs', 'chunk_steps', 'budget')


def _check_identity(latest, manifest, expected):
    for field in RUN_FIELDS:
        if field in expected and manifest.get(field) != expected[field]:
            raise ValueError(f'Checkpoint {latest} belongs to another run: {field} is {manifest.get(field)!r}, '
                             f'this run uses {expected[field]!r}')


def _final_chunk_is_complete(output, chunks):
    final = Path(output)/f'chunk-{chunks-1:04d}'
    try:
        return bool(verify_checkpoint(final)['complete_chunk'])
    except (OSError, ValueError, KeyError):
        return False


def first_target(output, expected=None, chunks=None):
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
    if expected is not None:
        _check_identity(latest, manifest, expected)
    target = chunk + (2 if complete else 1)
    if chunks is not None and target > chunks and not _final_chunk_is_complete(output, chunks):
        raise ValueError(f'Checkpoint {latest} is beyond --chunks {chunks} and no complete chunk-{chunks-1:04d} '
                         'exists; raise the chunk count or use a new output directory')
    return target


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output')
    p.add_argument('--variant')
    p.add_argument('--seed', type=int)
    p.add_argument('--envs', type=int)
    p.add_argument('--chunk-steps', type=int, default=DEFAULT_CHUNK_STEPS)
    p.add_argument('--budget', type=float)
    p.add_argument('--chunks', type=int)
    return p.parse_args(argv)


if __name__ == '__main__':
    a = parse(sys.argv[1:])
    settings = None
    if a.variant is not None:
        if a.seed is None or a.envs is None:
            raise SystemExit('--variant requires --seed and --envs')
        settings = {'variant': a.variant, 'seed': a.seed, 'envs': a.envs, 'chunk_steps': a.chunk_steps,
                    'budget': a.budget}
    try:
        print(first_target(a.output, settings, a.chunks))
    except ValueError as error:
        raise SystemExit(f'resume_target: {error}') from error
