from shipnav.service import execute
from shipnav.maps import SeaMap
import pytest


def test_service_is_deterministic_and_records_its_settings():
    sea = SeaMap((0, 0, 24, 24)).to_dict()
    a = execute(sea, (2, 2), (22, 22), policy_name='direct', count=0)
    b = execute(sea, (2, 2), (22, 22), policy_name='direct', count=0)
    assert a['frames'] == b['frames']
    assert a['status'] == 'success'
    assert a['settings']['query_env'] is False
    assert a['model_hashes'] == {}


def test_paired_variants_keep_identical_scenarios():
    sea = SeaMap((0, 0, 24, 24), ((9, 5, 14, 18),)).to_dict()
    a = execute(sea, (2, 2), (22, 22), policy_name='direct', seed=3, count=3)
    b = execute(sea, (2, 2), (22, 22), policy_name='direct', global_goals=False, seed=3, count=3)
    assert a['traffic_definitions'] == b['traffic_definitions']
    assert a['goals'] != b['goals']
    assert b['status'] == 'collision'


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError, match='Policy must'):
        execute(SeaMap((0, 0, 24, 24)).to_dict(), (2, 2), (22, 22), policy_name='missing', count=0)
