from pathlib import Path
import gzip
import json

import pytest

from shipnav.scenarios import scenario_hash, validate_scenario
from shipnav.training.scenarios import (SELECT_SEEDS, TRAIN_SEEDS, frozen_identities, generate,
                                        load_split, write_corpus)

ROOT = Path(__file__).resolve().parents[1]


def test_generation_is_deterministic_valid_and_labelled():
    a, b = generate(100_000, 'train'), generate(100_000, 'train')
    assert a == b and a['split'] == 'train' and a['seed'] == 100_000
    validate_scenario(a)
    assert 24 <= a['map']['bounds'][2] <= 48


def test_seed_ranges_are_disjoint_from_each_other_and_every_frozen_manifest():
    hashes, seeds = frozen_identities(ROOT)
    assert not set(TRAIN_SEEDS) & set(SELECT_SEEDS)
    assert not (set(TRAIN_SEEDS) | set(SELECT_SEEDS)) & seeds
    assert scenario_hash(generate(100_001, 'train')) not in hashes


def test_corpus_round_trip_verifies_hashes(tmp_path):
    scenes = [generate(100_002, 'train')]
    write_corpus(tmp_path, {'train': scenes, 'select': [generate(200_000, 'select')]}, [])
    assert load_split(tmp_path, 'train') == scenes
    raw = gzip.decompress((tmp_path/'train.jsonl.gz').read_bytes())
    (tmp_path/'train.jsonl.gz').write_bytes(gzip.compress(raw+b'\n', mtime=0))
    with pytest.raises(ValueError, match='hash'):
        load_split(tmp_path, 'train')


def test_committed_corpus_matches_its_manifest():
    manifest = json.loads((ROOT/'scenarios/training/manifest.json').read_text())
    assert manifest['splits']['train']['count'] >= 4_900
    assert len(load_split(ROOT/'scenarios/training', 'select')) == manifest['splits']['select']['count']


def test_committed_train_split_verifies_and_stays_inside_its_seed_range():
    manifest = json.loads((ROOT/'scenarios/training/manifest.json').read_text())
    scenarios = load_split(ROOT/'scenarios/training', 'train')  # verifies the file hash and the count
    seeds = [s['seed'] for s in scenarios]
    assert len(scenarios) == manifest['splits']['train']['count']
    assert all(seed in TRAIN_SEEDS for seed in seeds) and len(set(seeds)) == len(seeds)
    assert all(s['split'] == 'train' for s in scenarios)
    hashes, frozen_seeds = frozen_identities(ROOT)
    assert not set(seeds) & frozen_seeds and not {scenario_hash(s) for s in scenarios} & hashes
