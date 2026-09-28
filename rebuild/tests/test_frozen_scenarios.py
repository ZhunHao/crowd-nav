import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import pytest

from shipnav.maps import SeaMap
from shipnav.planning import NoPath, astar, smooth
from shipnav.scenarios import load_traffic
from shipnav.simulation import Traffic

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


ENCOUNTER_FAMILIES = ('head_on', 'crossing', 'crossing_mirrored', 'overtake', 'narrow_passage',
                      'detour_harbour', 'multi_conflict', 'course_change', 'reactive')


def _canonical_files():
    data = _splits()
    return {json.loads((ROOT / 'scenarios' / e['file']).read_text())['family']: e['file']
            for e in data['splits']['test']}


def _load(name):
    return json.loads((ROOT / 'scenarios' / name).read_text())


def test_committed_splits_record_no_freeze_failures():
    assert _splits()['failures'] == []


def test_every_encounter_family_has_a_nominal_conflict_against_a_direct_ego():
    module = _load_freeze_scenarios()
    files = _canonical_files()
    for family in ENCOUNTER_FAMILIES:
        scenario = _load(files[family])
        assert module.encounter_errors(scenario) == [], family
        route = module.planned_route(scenario)
        cpas = [module.nominal_cpa(route, ship) for ship in load_traffic(scenario)]
        threshold = module.EGO_RADIUS + .6 + module.CPA_MARGIN
        assert min(c['distance'] for c in cpas) < threshold, family


def test_narrow_passage_conflict_is_inside_the_gap_and_course_change_after_the_turn():
    module = _load_freeze_scenarios()
    files = _canonical_files()
    narrow = _load(files['narrow_passage'])
    [ship] = load_traffic(narrow)
    cpa = module.nominal_cpa(module.planned_route(narrow), ship)
    assert 8 <= cpa['ego'][0] <= 12
    course = _load(files['course_change'])
    [turning] = load_traffic(course)
    cpa = module.nominal_cpa(module.planned_route(course), turning)
    assert cpa['t'] >= turning.waypoints[1][0]


def test_reactive_fixture_reaction_fires_against_a_direct_ego():
    module = _load_freeze_scenarios()
    assert module.reaction_fires(_load(_canonical_files()['reactive']))


def test_freeze_checks_flag_encounters_that_never_happen():
    # The pre-fix crossing (.3 m/s target 9 m from the crossing point while the
    # ego passes in 9 s) never comes within the CPA threshold; the freeze-time
    # check must report it instead of writing it silently.
    module = _load_freeze_scenarios()
    stale = module._scenario(SeaMap((0., 0., 24., 24.)), module.EGO_START, module.EGO_GOAL,
                             [Traffic((12., 3.), (12., 21.))], 'crossing', 'test')
    assert module.encounter_errors(stale)


def test_every_family_except_unreachable_plans_with_astar_smooth():
    module = _load_freeze_scenarios()
    for family, name in _canonical_files().items():
        scenario = _load(name)
        sea = SeaMap.from_dict(scenario['map'])
        start, goal = tuple(scenario['start']), tuple(scenario['goal'])
        if family == 'unreachable':
            with pytest.raises(NoPath):
                smooth(sea, astar(sea, start, goal))
        else:
            assert smooth(sea, astar(sea, start, goal))[-1] == goal
        assert module.planning_errors(scenario) == []
