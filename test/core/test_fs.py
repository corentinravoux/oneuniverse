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


def test_atomic_write_json_honours_umask(tmp_path, monkeypatch):
    import os, stat
    from oneuniverse._core import fs
    monkeypatch.setattr(fs, "_UMASK", 0o022)
    p = tmp_path / "m.json"; atomic_write_json(p, {"a": 1})
    assert stat.S_IMODE(p.stat().st_mode) == 0o644
    monkeypatch.setattr(fs, "_UMASK", 0o077)
    atomic_write_json(p, {"a": 1})
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_atomic_dir_published_mode_follows_umask(tmp_path):
    import os, stat
    from oneuniverse._core import fs
    old = os.umask(0o022)
    try:
        with atomic_dir(tmp_path / "ds") as d:
            (d / "f").write_text("x")
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "ds").stat().st_mode) == 0o755


def test_staging_helpers():
    from oneuniverse._core.fs import STAGING_SUFFIXES, is_staging
    assert STAGING_SUFFIXES == (".tmp", ".old")
    assert is_staging("a/ds.tmp") and is_staging("ds.old")
    assert not is_staging("a/ds") and not is_staging("ds.tmpl")


def test_atomic_write_bytes_cleans_up_on_failure(tmp_path):
    with pytest.raises(TypeError):
        atomic_write_bytes(tmp_path / "x", "not bytes")
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_json_numpy_and_nan(tmp_path):
    import numpy as np
    p = tmp_path / "m.json"
    atomic_write_json(p, {"i": np.int64(3), "f": np.float32(1.5), "b": np.bool_(True),
                          "a": np.arange(3)})
    assert json.loads(p.read_text()) == {"i": 3, "f": 1.5, "b": True, "a": [0, 1, 2]}
    with pytest.raises(ValueError):
        atomic_write_json(p, {"x": float("nan")})
    with pytest.raises(TypeError):
        atomic_write_json(p, {"x": object()})
    assert json.loads(p.read_text())["i"] == 3 and len(list(tmp_path.iterdir())) == 1


def test_atomic_dir_state_a_stale_tmp(tmp_path):
    target = tmp_path / "ds"; target.mkdir(); (tmp_path / "ds.tmp").mkdir()
    with pytest.raises(FormatError, match="stale temporary"):
        with atomic_dir(target):
            pass
    assert target.exists()


def test_atomic_dir_state_b_interrupted_swap(tmp_path):
    (tmp_path / "ds.tmp").mkdir(); (tmp_path / "ds.old").mkdir()
    with pytest.raises(FormatError, match=r"rename .*ds\.tmp to .*ds to finish"):
        with atomic_dir(tmp_path / "ds"):
            pass


def test_atomic_dir_state_c_leftover_old(tmp_path):
    (tmp_path / "ds").mkdir(); (tmp_path / "ds.old").mkdir()
    with pytest.raises(FormatError, match=r"delete .*ds\.old"):
        with atomic_dir(tmp_path / "ds"):
            pass


def test_atomic_dir_concurrent_writer_race(tmp_path, monkeypatch):
    from pathlib import Path
    real = Path.mkdir

    def racing(self, *a, **k):
        if self.name.endswith(".tmp"):
            real(self, *a, **k)  # the other writer wins
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", racing)
    with pytest.raises(FormatError, match="another writer"):
        with atomic_dir(tmp_path / "ds"):
            pass


def test_atomic_dir_rmtree_old_failure_reports_published(tmp_path, monkeypatch):
    import shutil
    target = tmp_path / "ds"; target.mkdir()
    real = shutil.rmtree

    def failing(p, *a, **k):
        if str(p).endswith(".old"):
            raise OSError("boom")
        return real(p, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", failing)
    with pytest.raises(FormatError, match=r"new version IS published.*ds\.old"):
        with atomic_dir(target) as d:
            (d / "new").write_text("y")
    assert (target / "new").exists()
