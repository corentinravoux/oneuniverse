import json
import re

import pytest

from oneuniverse._core.errors import FormatError
from oneuniverse._core.fs import (
    atomic_dir, atomic_write_bytes, atomic_write_json, hash_bytes, hash_file,
)


def test_atomic_write_json_replaces_and_is_readable(tmp_path):
    p = tmp_path / "m.json"; atomic_write_json(p, {"a": 1}); atomic_write_json(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 2} and not list(tmp_path.glob("*.tmp*"))


def test_atomic_dir_success_replaces_previous(tmp_path):
    target = tmp_path / "ds"; target.mkdir(); (target / "old").write_text("x")
    with atomic_dir(target) as d:
        (d / "new").write_text("y")
    assert (target / "new").exists() and not (target / "old").exists()
    assert not (tmp_path / "ds.tmp").exists() and not (tmp_path / "ds.old").exists()


def test_atomic_dir_failure_leaves_previous(tmp_path):
    target = tmp_path / "ds"; target.mkdir(); (target / "old").write_text("x")
    with pytest.raises(RuntimeError):
        with atomic_dir(target) as d:
            (d / "new").write_text("y"); raise RuntimeError
    assert (target / "old").exists() and not (target / "new").exists() and not (tmp_path / "ds.tmp").exists()


def test_atomic_dir_fresh_target(tmp_path):
    target = tmp_path / "ds"
    with atomic_dir(target) as d:
        (d / "f").write_text("y")
    assert (target / "f").exists()


def test_atomic_dir_refuses_stale_tmp(tmp_path):
    (tmp_path / "ds.tmp").mkdir()
    with pytest.raises(FormatError):
        with atomic_dir(tmp_path / "ds"):
            pass


def test_atomic_write_bytes_roundtrip(tmp_path):
    p = tmp_path / "b.bin"; atomic_write_bytes(p, b"abc")
    assert p.read_bytes() == b"abc"


def test_hash_bytes_is_16_hex():
    assert re.fullmatch(r"[0-9a-f]{16}", hash_bytes(b"abc"))


def test_hash_file_matches_hash_bytes(tmp_path):
    p = tmp_path / "h"; data = b"x" * 3_000_000; p.write_bytes(data)
    assert hash_file(p) == hash_bytes(data)


def test_atomic_dir_refuses_stale_old(tmp_path):
    target = tmp_path / "ds"; target.mkdir()
    (tmp_path / "ds.old").mkdir()
    with pytest.raises(FormatError):
        with atomic_dir(target):
            pass
    assert target.exists() and not (tmp_path / "ds.tmp").exists()


def test_atomic_dir_publish_failure_removes_tmp_and_restores(tmp_path, monkeypatch):
    import os as _os
    target = tmp_path / "ds"; target.mkdir(); (target / "old").write_text("x")
    real = _os.rename

    def flaky(src, dst):
        if str(src).endswith(".tmp"):
            raise OSError("boom")
        return real(src, dst)

    monkeypatch.setattr("oneuniverse._core.fs.os.rename", flaky)
    with pytest.raises(OSError):
        with atomic_dir(target) as d:
            (d / "new").write_text("y")
    assert (target / "old").exists()
    assert not (tmp_path / "ds.tmp").exists() and not (tmp_path / "ds.old").exists()
