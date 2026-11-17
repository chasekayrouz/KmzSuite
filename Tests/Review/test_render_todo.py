"""The backlog check passes without a backlog but still catches a stale or orphaned view."""
import sys

import pytest

from scripts import render_todo


def run(monkeypatch, root, *arguments):
    monkeypatch.setattr(render_todo, 'ROOT', root)
    monkeypatch.setattr(sys, 'argv', ['render_todo.py', *arguments])
    render_todo.main()


def test_check_without_backlog_passes(tmp_path, monkeypatch, capsys):
    run(monkeypatch, tmp_path, '--check')
    assert 'nothing to check' in capsys.readouterr().out


def test_check_detects_stale_and_orphaned_views(tmp_path, monkeypatch):
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'TODO.md').write_text('# Backlog\n\n- **Item:** work\n')
    with pytest.raises(SystemExit, match='stale'):
        run(monkeypatch, tmp_path, '--check')
    run(monkeypatch, tmp_path)
    run(monkeypatch, tmp_path, '--check')
    (tmp_path / 'TODO.md').unlink()
    with pytest.raises(FileNotFoundError):
        run(monkeypatch, tmp_path, '--check')
