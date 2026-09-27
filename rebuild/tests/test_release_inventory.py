import importlib.util
from pathlib import Path


def test_inventory_preserves_release_and_wheel_metadata():
    path = Path(__file__).parents[1] / 'tools/release_inventory.py'
    spec = importlib.util.spec_from_file_location('release_inventory', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = {'info': {'version': '1.2.3', 'requires_python': '>=3.14'}, 'urls': [{'filename': 'example.whl', 'digests': {'sha256': 'abc'}, 'yanked': False}]}
    result = module.inventory(lambda name: data)
    assert result['torch'] == {'version': '1.2.3', 'requires_python': '>=3.14', 'files': [{'name': 'example.whl', 'sha256': 'abc', 'yanked': False}]}
