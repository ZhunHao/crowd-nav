"""Checks that tie a trained checkpoint to the frozen training configuration (shared by the selector and freezer)."""
from pathlib import Path
import os
import subprocess

from shipnav.training.checkpoints import IDENTITY, verify_checkpoint


def portable(path, root):
    """Rebuild-relative POSIX path when `path` lies under `root`, else the absolute path (no symlink resolution).

    A relative `path` is taken as relative to `root`; an absolute one replaces it."""
    path = Path(os.path.normpath(Path(root)/path))
    try:
        return path.relative_to(os.path.normpath(root)).as_posix()
    except ValueError:
        return path.as_posix()


def mixed_identity(manifests):
    """Names of the IDENTITY fields on which the manifests disagree (empty when they are one run)."""
    first = next(iter(manifests.values()), None)
    if first is None:
        return []
    return [key for key in IDENTITY if any(m.get(key) != first.get(key) for m in manifests.values())]


def manifest_problems(manifest, training, variant, seed):
    """Everything that ties `manifest` to `training` (the frozen training_config.json) for run variant/seed."""
    problems = []

    def expect(name, found, wanted):
        if found != wanted:
            problems.append(f'{name} is {found!r}, the training configuration says {wanted!r}')
    env = manifest.get('env_config')
    env = env if isinstance(env, dict) else {}
    expect('variant', manifest.get('variant'), variant)
    expect('seed', manifest.get('seed'), seed)
    expect('env_config.filtered', env.get('filtered'), training['filtered'])
    expect('env_config.observation', env.get('observation'), training['observation'])
    expect('chunk_steps', manifest.get('chunk_steps'), training['chunk_steps'])
    expect('envs', manifest.get('envs'), training['envs'])
    expect('scenario_manifest_sha256', manifest.get('scenario_manifest_sha256'), training['corpus_sha256']['train'])
    expect('budget', manifest.get('budget'), training['budget_domain_s'] if variant == 'ppo_lagrangian' else None)
    expect('complete_chunk', manifest.get('complete_chunk'), True)
    expect('source_commit', manifest.get('source_commit'), training['source_commit'])
    return problems


def final_chunk_problem(run_dir, training, verify=verify_checkpoint):
    """None when `run_dir` holds a complete chunk-(chunks-1): every run must train the same number of steps."""
    final = Path(run_dir)/f'chunk-{training["chunks"]-1:04d}'
    try:
        manifest = verify(final)
    except (OSError, ValueError) as error:
        return f'the run did not reach its final chunk {final.name}: {error}'
    if manifest.get('complete_chunk') is not True or manifest.get('chunk') != training['chunks']-1:
        return f'{final.name} is not the complete final chunk'
    return None


def check_source_unchanged(commit, root, run=subprocess.run):
    """Refuse a tree whose src/ differs from the commit the checkpoints were trained at."""
    done = run(['git', 'diff', '--quiet', commit, 'HEAD', '--', 'src'], cwd=root, capture_output=True)
    if done.returncode != 0:
        raise ValueError(f'src/ differs between the training source commit {commit} and HEAD '
                         '(or the commit is unknown): the checkpoints were trained on different code')


def dirty_paths(root, paths):
    """`git status --porcelain` lines (tracked changes and untracked files) under `paths`, relative to `root`."""
    status = subprocess.check_output(['git', 'status', '--porcelain', '--', *paths], cwd=root, text=True)
    return sorted(line for line in status.splitlines() if line)
