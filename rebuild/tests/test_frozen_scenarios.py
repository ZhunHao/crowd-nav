import importlib.util
import json
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_freeze_scenarios():
    spec = importlib.util.spec_from_file_location('freeze_scenarios', ROOT / 'tools/freeze_scenarios.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _splits():
    return json.loads((ROOT / 'scenarios/splits.json').read_text())


def test_splits_hashes_match_files_on_disk():
    data = _splits()
    assert data['schema'] == 1
    for split, entries in data['splits'].items():
        for entry in entries:
            body = (ROOT / 'scenarios' / entry['file']).read_bytes()
            assert sha256(body).hexdigest() == entry['sha256']
            scenario = json.loads(body)
            assert scenario['split'] == split


def test_splits_are_disjoint_by_file():
    data = _splits()
    files = [entry['file'] for entries in data['splits'].values() for entry in entries]
    assert len(files) == len(set(files))


def _land_geometry(scenario):
    """Bounds + land features only, excluding metadata, so identical land
    geometry with different metadata (e.g. a different source filename) is
    still recognized as the same map."""
    map_dict = scenario['map']
    return json.dumps({'bounds': map_dict['bounds'], 'features': map_dict['features']}, sort_keys=True)


def test_seeded_splits_use_distinct_maps_and_disjoint_seed_ranges():
    data = _splits()
    seeded = ('train', 'dev', 'calibration')
    seeds_by_split, maps_by_split = {}, {}
    for split in seeded:
        seeds, maps = set(), set()
        for entry in data['splits'][split]:
            scenario = json.loads((ROOT / 'scenarios' / entry['file']).read_text())
            seeds.add(scenario['seed'])
            maps.add(_land_geometry(scenario))
        seeds_by_split[split], maps_by_split[split] = seeds, maps
    for a in seeded:
        for b in seeded:
            if a != b:
                assert not (seeds_by_split[a] & seeds_by_split[b])
                assert not (maps_by_split[a] & maps_by_split[b])


def test_every_canonical_family_is_present_once_in_test_and_absent_elsewhere():
    module = _load_freeze_scenarios()
    data = _splits()
    test_families = set()
    for entry in data['splits']['test']:
        scenario = json.loads((ROOT / 'scenarios' / entry['file']).read_text())
        assert scenario['split'] == 'test'
        test_families.add(scenario['family'])
    assert test_families == set(module.CANONICAL_FAMILIES)
    for split, entries in data['splits'].items():
        if split == 'test':
            continue
        for entry in entries:
            scenario = json.loads((ROOT / 'scenarios' / entry['file']).read_text())
            assert scenario['family'] not in module.CANONICAL_FAMILIES
    assert set(data['withheld_families']) == set(module.CANONICAL_FAMILIES)


def test_generalization_report_families_match_withheld_families():
    module = _load_freeze_scenarios()
    data = _splits()
    text = (ROOT / 'scenarios/generalization.md').read_text()
    for family in module.CANONICAL_FAMILIES:
        assert family in text
    assert set(data['withheld_families']) == set(module.CANONICAL_FAMILIES)


def test_readme_families_match_withheld_families():
    # scenarios/README.md documents the same canonical families by hand
    # (unlike splits.json, which is generated); it drifted out of sync once
    # (still said "nine" after course_change/reactive were added), so this
    # test pins README.md to the same source of truth as splits.json and
    # generalization.md so that drift can't recur silently.
    module = _load_freeze_scenarios()
    data = _splits()
    text = (ROOT / 'scenarios/README.md').read_text()
    for family in module.CANONICAL_FAMILIES:
        assert family in text
    assert set(data['withheld_families']) == set(module.CANONICAL_FAMILIES)


def test_regenerating_reproduces_identical_bytes(tmp_path):
    module = _load_freeze_scenarios()
    module.generate(tmp_path)
    committed = ROOT / 'scenarios'
    for path in sorted(committed.glob('*.json')):
        assert (tmp_path / path.name).read_bytes() == path.read_bytes()
    committed_names = {p.name for p in committed.glob('*.json')}
    regenerated_names = {p.name for p in tmp_path.glob('*.json')}
    assert committed_names == regenerated_names
