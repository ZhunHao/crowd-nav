"""Local evidence tools reject foreign hosts before writing accepted artifacts."""
import importlib.util
from pathlib import Path
import runpy
import platform
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'

@pytest.mark.parametrize('name', ['build_rvo2', 'install_rvo2', 'accept_modern'])
def test_foreign_host_rejected(name, monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    with pytest.raises(RuntimeError, match='only macOS 26 arm64'):
        if name == 'build_rvo2': module.build(tmp_path / 'out')
        elif name == 'install_rvo2': module.install(tmp_path, Path('python'))
        else: module.record()
    assert list(tmp_path.iterdir()) == []

@pytest.mark.parametrize('system,machine,version', [('Darwin', 'x86_64', '26.0'), ('Darwin', 'arm64', '15.0'), ('Linux', 'aarch64', '')])
def test_smoke_rejects_before_import_or_output(monkeypatch, system, machine, version):
    monkeypatch.setattr(platform, 'system', lambda: system)
    monkeypatch.setattr(platform, 'machine', lambda: machine)
    monkeypatch.setattr(platform, 'mac_ver', lambda: (version, (), ''))
    with pytest.raises(RuntimeError, match='only macOS 26 arm64'):
        runpy.run_path(str(TOOLS / 'smoke_modern.py'))


def test_installer_rejects_mixed_manifest(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location('installer', TOOLS / 'install_rvo2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'require_local_platform', lambda: None)
    (tmp_path / 'wheels.json').write_text('{"mac.whl":"a","linux.whl":"b"}')
    with pytest.raises(ValueError, match='exactly one platform wheel'):
        module.install(tmp_path, Path('python'))
