import json
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa
import pytest

import oneuniverse.data.sidecar as mod
from oneuniverse._core.errors import FormatError, NotFoundError, ValidationError
from oneuniverse.data.sidecar import VersionedSidecar


def _t(n):
    return pa.table({"a": np.arange(n)})


def test_write_then_load_current(tmp_path):
    sc = VersionedSidecar(tmp_path, "oneuid", "qso")
    v = sc.write(_t(3), {"rules": "x"})
    t, meta = sc.load()
    assert t.num_rows == 3 and meta["version"] == v and meta["valid_to"] is None and meta["rules"] == "x"
    assert meta["n_rows"] == 3 and meta["format_version"] == 1 and meta["kind"] == "oneuid"
    assert v.endswith("Z") and "." in v
    assert VersionedSidecar.names(tmp_path, "oneuid") == ["qso"]
    assert (tmp_path / "_oneuid/qso/current.json").exists()


def test_rewrite_archives_and_as_of(tmp_path):
    sc = VersionedSidecar(tmp_path, "oneuid", "qso")
    v1 = sc.write(_t(1), {})
    t_between = datetime.now(timezone.utc)
    time.sleep(0.01)
    v2 = sc.write(_t(2), {})
    assert sc.current() == v2 and sc.load()[0].num_rows == 2
    old_t, old_meta = sc.load(as_of=t_between)
    assert old_t.num_rows == 1 and old_meta["valid_to"] == v2
    assert [m["version"] for m in sc.versions()] == [v1, v2]
    with pytest.raises(NotFoundError):
        sc.load(as_of=datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert sc.load(as_of=datetime.now(timezone.utc) + timedelta(days=1))[0].num_rows == 2


def test_sidecar_crash_between_steps(tmp_path, monkeypatch):
    sc = VersionedSidecar(tmp_path, "oneuid", "qso")
    v1 = sc.write(_t(1), {})
    real = mod.atomic_write_json

    def flaky(path, obj):
        if path.name == "current.json":
            raise OSError("disk full")
        return real(path, obj)

    monkeypatch.setattr(mod, "atomic_write_json", flaky)
    with pytest.raises(OSError):
        sc.write(_t(2), {})
    monkeypatch.undo()
    assert sc.current() == v1 and sc.load()[0].num_rows == 1 and sc.load()[1]["valid_to"] is None
    assert len(sc.versions()) == 1


def test_crash_after_pointer_swap_derives_valid_to(tmp_path, monkeypatch):
    sc = VersionedSidecar(tmp_path, "oneuid", "qso")
    v1 = sc.write(_t(1), {})
    t_between = datetime.now(timezone.utc)
    time.sleep(0.01)
    real = mod.atomic_write_json
    n = {"c": 0}

    def flaky(path, obj):
        if path.name == "meta.json" and n["c"] >= 1:  # step 3 (patching v1)
            raise OSError("boom")
        if path.name == "meta.json":
            n["c"] += 1
        return real(path, obj)

    monkeypatch.setattr(mod, "atomic_write_json", flaky)
    with pytest.raises(OSError):
        sc.write(_t(2), {})
    monkeypatch.undo()
    v2 = sc.current()
    assert v2 != v1
    t, m = sc.load(as_of=t_between)
    assert t.num_rows == 1 and m["valid_to"] == v2
    assert sc.load()[0].num_rows == 2


def test_no_version_raises(tmp_path):
    with pytest.raises(NotFoundError):
        VersionedSidecar(tmp_path, "links", "none").load()
    assert VersionedSidecar.names(tmp_path, "links") == []
    assert VersionedSidecar(tmp_path, "links", "none").current() is None


def test_microsecond_collision_bumps(tmp_path, monkeypatch):
    frozen = datetime(2030, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
    monkeypatch.setattr(mod, "_now", lambda: frozen)
    sc = VersionedSidecar(tmp_path, "oneuid", "qso")
    v1 = sc.write(_t(1), {})
    v2 = sc.write(_t(2), {})
    assert v1 == "2030-01-01T12:00:00.123456Z" and v2 == "2030-01-01T12:00:00.123457Z"
    assert v2 > v1 and sc.load()[0].num_rows == 2 and sc.load(as_of=frozen)[0].num_rows == 1


def test_naive_as_of_rejected(tmp_path):
    sc = VersionedSidecar(tmp_path, "oneuid", "qso")
    sc.write(_t(1), {})
    with pytest.raises(ValidationError):
        sc.load(as_of=datetime(2030, 1, 1))


@pytest.mark.parametrize("key", ["format_version", "kind", "name", "version", "valid_from", "valid_to", "n_rows"])
def test_reserved_meta_keys(tmp_path, key):
    with pytest.raises(ValidationError):
        VersionedSidecar(tmp_path, "oneuid", "qso").write(_t(1), {key: 1})


def test_meta_must_be_json(tmp_path):
    with pytest.raises(ValidationError):
        VersionedSidecar(tmp_path, "oneuid", "qso").write(_t(1), {"x": object()})


@pytest.mark.parametrize("bad", ["", ".x", "_x", "a/b", "a b", "..", "a\\b"])
def test_unsafe_segments(tmp_path, bad):
    with pytest.raises(ValidationError):
        VersionedSidecar(tmp_path, bad, "qso")
    with pytest.raises(ValidationError):
        VersionedSidecar(tmp_path, "oneuid", bad)


def test_orphan_removed_on_next_write_and_staging_ignored(tmp_path, monkeypatch):
    sc = VersionedSidecar(tmp_path, "oneuid", "qso")
    v1 = sc.write(_t(1), {})
    real = mod.atomic_write_json

    def flaky(path, obj):
        if path.name == "current.json":
            raise OSError("x")
        return real(path, obj)

    monkeypatch.setattr(mod, "atomic_write_json", flaky)
    with pytest.raises(OSError):
        sc.write(_t(2), {})
    monkeypatch.undo()
    base = tmp_path / "_oneuid/qso"
    orphans = [p for p in base.iterdir() if p.is_dir() and p.name != v1]
    assert len(orphans) == 1
    (base / "2999-01-01T00:00:00.000000Z.tmp").mkdir()
    assert len(sc.versions()) == 1
    (base / "2999-01-01T00:00:00.000000Z.tmp").rmdir()
    v3 = sc.write(_t(3), {})
    assert sorted(p.name for p in base.iterdir() if p.is_dir()) == sorted([v1, v3])
    assert [m["version"] for m in sc.versions()] == [v1, v3]


def test_meta_roundtrip_is_json(tmp_path):
    sc = VersionedSidecar(tmp_path, "links", "x")
    v = sc.write(_t(1), {"k": [1, 2]})
    assert json.loads((tmp_path / "_links/x" / v / "meta.json").read_text())["k"] == [1, 2]


@pytest.mark.parametrize("bad", ["x.tmp", "x.old"])
def test_staging_suffix_segments_rejected(tmp_path, bad):
    with pytest.raises(ValidationError):
        VersionedSidecar(tmp_path, "oneuid", bad)
    with pytest.raises(ValidationError):
        VersionedSidecar(tmp_path, bad, "qso")


def test_unknown_format_version_and_corruption(tmp_path):
    sc = VersionedSidecar(tmp_path, "oneuid", "qso")
    v = sc.write(_t(2), {})
    meta_p = sc.dir / v / "meta.json"
    good = meta_p.read_text()
    meta = json.loads(good)
    meta["format_version"] = 99
    meta_p.write_text(json.dumps(meta))
    with pytest.raises(FormatError, match="format_version"):
        sc.load()
    meta_p.write_text("{not json")
    with pytest.raises(FormatError) as ei:
        sc.meta()
    assert ei.value.__cause__ is not None
    meta_p.write_text(json.dumps({"format_version": 1}))
    with pytest.raises(FormatError):
        sc.load()
    meta_p.write_text(good)
    (sc.dir / "current.json").write_text("garbage")
    with pytest.raises(FormatError):
        sc.current()
    (sc.dir / "current.json").write_text("{}")
    with pytest.raises(FormatError):
        sc.load()
