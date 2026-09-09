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
