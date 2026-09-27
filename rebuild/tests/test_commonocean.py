"""Tests for the CommonOcean adapter (task D4).

`import_scenario` only needs `commonocean-io`, which installs cleanly (pure
Python) in the disposable external environment; those tests run for real
there and skip in the modern env where `commonocean` is not installed.

`check_collision`/`check_trajectory` need `commonocean_dc`, whose native
build failed in every environment tried here (missing system Boost; see
migration/external/commonocean/install*.log and decision.md). Those tests
are marked xfail(raises=ImportError) rather than silently skipped, so they
would fail loudly (not silently pass) if commonocean_dc ever became
importable without the code being revisited.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip('commonocean')

from shipnav.adapters.commonocean import check_collision, check_trajectory, import_scenario  # noqa: E402

pytestmark = pytest.mark.external

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = (ROOT / 'migration/external/commonocean/upstream/commonocean-scenarios/scenarios'
            '/HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-786.xml')
EXPORTED = ROOT / 'scenarios/external/commonocean_ZAM_AAA-1_20240121_T-786.json'


def _skip_if_no_upstream_checkout():
    if not UPSTREAM.exists():
        pytest.skip('upstream commonocean-scenarios checkout not present '
                     '(git-ignored external spike checkout); see decision.md')


def test_import_scenario_matches_committed_export():
    _skip_if_no_upstream_checkout()
    imported = import_scenario(UPSTREAM)
    exported = json.loads(EXPORTED.read_text())
    assert imported['start'] == exported['start']
    assert imported['goal'] == exported['goal']
    assert imported['traffic'] == exported['traffic']
    assert imported['external_metadata']['benchmark_id'] == exported['external_metadata']['benchmark_id']
    assert imported['external_metadata']['dt_seconds'] == exported['external_metadata']['dt_seconds']


def test_import_scenario_canonical_schema_shape():
    _skip_if_no_upstream_checkout()
    d = import_scenario(UPSTREAM)
    assert d['schema'] == 1
    assert d['family'] == 'external_commonocean'
    assert d['split'] == 'external'
    assert len(d['start']) == len(d['goal']) == 2
    assert len(d['traffic']) == 1
    ship = d['traffic'][0]
    assert ship['radius'] > 0
    meta = d['external_metadata']
    assert 'coordinate_transform' in meta and 'north_east_to_xy' in meta['coordinate_transform']
    assert meta['control_limits']['note']
    assert meta['original_waters_polygons'] == []  # this scenario declares none
    assert meta['original_static_obstacles'] == []  # this scenario declares none


def test_import_scenario_missing_file_raises():
    with pytest.raises(Exception):
        import_scenario(ROOT / 'scenarios/external/does-not-exist.xml')


def test_committed_export_has_provenance():
    exported = json.loads(EXPORTED.read_text())
    meta = exported['external_metadata']
    assert meta['upstream_repository'] == 'https://gitlab.lrz.de/tum-cps/commonocean-scenarios.git'
    assert meta['upstream_revision'] == '2ab3a4b3964c9df70bc4ce93f7b6d59c80f9de39'
    assert 'BSD-3-Clause' in meta['upstream_license']
    assert meta['benchmark_id'] == 'ZAM_AAA-1_20240121_T-786'


@pytest.mark.xfail(raises=ImportError, strict=True,
                    reason='commonocean_dc native build failed (missing Boost); see decision.md')
def test_check_collision_requires_commonocean_dc():
    # The ImportError is raised before the upstream file is even opened, so
    # this does not depend on the git-ignored upstream checkout being present.
    result = {'frames': [{'t': 0.0, 'position': [0.0, 0.0]}], 'diagnostics': []}
    check_collision(UPSTREAM, result, length=4.0, width=3.0)


def test_check_trajectory_reports_collision_not_checked_when_unavailable():
    result = {'frames': [{'t': 0.0, 'position': [0.0, 0.0]}], 'diagnostics': []}
    report = check_trajectory(UPSTREAM, result, length=4.0, width=3.0)
    assert report['collision']['checked'] is False
    assert 'commonocean_dc unavailable' in report['collision']['reason']
    assert report['water_boundary'] == {
        'checked': False, 'result': None,
        'reason': 'water-boundary feasibility is not implemented by this adapter'}
    assert report['model_feasibility'] == {
        'checked': False, 'result': None,
        'reason': 'kinematic/model feasibility is not implemented by this adapter'}
