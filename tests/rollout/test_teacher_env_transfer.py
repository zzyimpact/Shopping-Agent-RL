"""No network: fake SCP verifies exit, timeout and process cleanup."""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('transfer', ROOT / 'scripts/teacher_env_transfer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('exit_code', [0, 7])
def test_transfer_preserves_exit_status(tmp_path, monkeypatch, exit_code):
    scp = tmp_path / 'scp'
    scp.write_text(f'#!/bin/sh\nexit {exit_code}\n')
    scp.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    assert module.transfer('source', 'dest', timeout=2) == exit_code


def test_stalled_transfer_is_bounded_and_child_killed(tmp_path, monkeypatch, capsys):
    scp = tmp_path / 'scp'
    pid = tmp_path / 'pid'
    scp.write_text(f'#!/bin/sh\necho $$ > "{pid}"\nexec sleep 60\n')
    scp.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    assert module.transfer('source', 'dest', timeout=2) == 124
    assert 'timed out' in capsys.readouterr().err
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


@pytest.mark.parametrize('corruption', [None, 'ids', 'truncated'])
def test_manifest_cache_requires_frozen_identity(tmp_path, corruption):
    import hashlib
    import json
    digest = hashlib.sha256(b'1\n2').hexdigest()
    path = tmp_path / 'manifest.json'
    value = {'metadata': {'task_ids_sha256': digest},
             'tasks': [{'task_id': '1'}, {'task_id': '2'}]}
    if corruption == 'ids':
        value['tasks'][1]['task_id'] = '3'
    path.write_text('{' if corruption == 'truncated' else json.dumps(value))
    assert module.manifest_valid(path, expected_count=2, expected_hash=digest) == (corruption is None)


def test_frozen_policy_manifest_lookup(monkeypatch, tmp_path):
    calls = []
    def validate(path, **kwargs):
        calls.append(kwargs)
        return True
    monkeypatch.setattr(module, 'manifest_valid', validate)
    for name in ('train_single.json', 'sft_task_manifest_single.json',
                 'train_single_persona.json', 'sft_task_manifest_single_persona.json'):
        assert module.check_manifest(tmp_path / name, name)
    assert [c['expected_count'] for c in calls] == [21962, 3000, 3323, 3000]
    assert not module.check_manifest(tmp_path / 'unknown', 'unknown')
