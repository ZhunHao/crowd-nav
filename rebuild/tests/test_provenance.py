from hashlib import sha256
import subprocess

from shipnav.provenance import provenance


def _init_repo(root):
    subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
    subprocess.run(['git', 'config', 'user.email', 'a@b.c'], cwd=root, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=root, check=True)
    (root/'file.txt').write_text('hello\n')
    subprocess.run(['git', 'add', 'file.txt'], cwd=root, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', 'initial'], cwd=root, check=True)


def test_clean_repo_reports_revision_and_empty_diff_hash(tmp_path):
    _init_repo(tmp_path)
    (tmp_path/'rebuild').mkdir()
    (tmp_path/'rebuild'/'uv.lock').write_text('lock-contents\n')
    result = provenance({'a': 1}, tmp_path)
    assert result['git_error'] is None
    assert len(result['git_revision']) == 40
    assert result['dirty_diff_sha256'] == sha256(b'').hexdigest()
    assert result['uv_lock_sha256'] == sha256(b'lock-contents\n').hexdigest()
    assert result['python_implementation']
    assert result['python_version']
    assert result['platform']


def test_dirty_repo_hashes_the_actual_diff_not_the_empty_hash(tmp_path):
    _init_repo(tmp_path)
    (tmp_path/'file.txt').write_text('changed\n')
    result = provenance({}, tmp_path)
    diff = subprocess.run(['git', 'diff', 'HEAD'], cwd=tmp_path, capture_output=True, check=True).stdout
    assert result['dirty_diff_sha256'] == sha256(diff).hexdigest()
    assert result['dirty_diff_sha256'] != sha256(b'').hexdigest()


def test_non_repo_reports_git_error_without_fabricating_fields(tmp_path):
    result = provenance({}, tmp_path)
    assert result['git_revision'] is None
    assert result['dirty_diff_sha256'] is None
    assert result['git_error']


def test_missing_uv_lock_is_none_not_fabricated(tmp_path):
    _init_repo(tmp_path)
    result = provenance({}, tmp_path)
    assert result['uv_lock_sha256'] is None


def test_settings_hash_is_order_independent_and_content_sensitive(tmp_path):
    _init_repo(tmp_path)
    a = provenance({'x': 1, 'y': 2}, tmp_path)
    b = provenance({'y': 2, 'x': 1}, tmp_path)
    c = provenance({'x': 1, 'y': 3}, tmp_path)
    assert a['settings_sha256'] == b['settings_sha256']
    assert a['settings_sha256'] != c['settings_sha256']


def test_partial_git_failure_reports_no_git_fields(tmp_path, monkeypatch):
    # rev-parse succeeds but diff fails: never report a revision without its diff.
    import shipnav.provenance as module
    real = module._git

    def flaky(args, root):
        if args[0] == 'diff':
            raise subprocess.CalledProcessError(1, ['git', *args])
        return real(args, root)
    _init_repo(tmp_path)
    monkeypatch.setattr(module, '_git', flaky)
    result = provenance({}, tmp_path)
    assert result['git_revision'] is None
    assert result['dirty_diff_sha256'] is None
    assert result['git_error']
