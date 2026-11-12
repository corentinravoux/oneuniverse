import json
import time
from datetime import UTC, datetime

import pytest
from fixtures.synthetic import write_synthetic_dataset

from oneuniverse._core.errors import FormatError, NotFoundError, ValidationError
from oneuniverse.data.database import Database
from oneuniverse.data.view import DatasetView


def test_discovery_and_views(tmp_path):
    write_synthetic_dataset(tmp_path / "eboss_qso", 1_000, 0, survey="eboss_qso")
    write_synthetic_dataset(tmp_path / "desi/qso", 1_000, 1, survey="desi_qso")
    (tmp_path / "_oneuid").mkdir()
    db = Database(tmp_path)
    assert db.names() == ["desi__qso", "eboss_qso"]
    assert db["eboss_qso"].n_rows == 1_000 and "desi__qso" in db and len(db) == 2
    assert list(db) == ["desi__qso", "eboss_qso"]
    assert db["eboss_qso"] is not db["eboss_qso"]
    assert isinstance(db["desi__qso"], DatasetView)
    assert db.path("desi__qso") == tmp_path / "desi" / "qso"
    assert db.manifest("eboss_qso").n_rows == 1_000


def test_unknown_name_lists_available(tmp_path):
    write_synthetic_dataset(tmp_path / "ok", 10, 0)
    with pytest.raises(NotFoundError, match="ok"):
        Database(tmp_path)["nope"]
    with pytest.raises(NotFoundError):
        Database(tmp_path).history("nope")


def test_staging_and_deep_dirs_skipped(tmp_path):
    write_synthetic_dataset(tmp_path / "ok", 10, 0)
    for n in ("ds.tmp", "ds.old"):
        (tmp_path / n).mkdir()
        (tmp_path / n / "manifest.json").write_text("{")
    write_synthetic_dataset(tmp_path / "a/b/c", 10, 0)
    assert Database(tmp_path).names() == ["ok"]


def test_strict_vs_lenient_on_corrupt_manifest(tmp_path):
    write_synthetic_dataset(tmp_path / "ok", 10, 0)
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad/manifest.json").write_text("{")
    with pytest.raises(FormatError, match="bad"):
        Database(tmp_path)
    db = Database(tmp_path, strict=False)
    assert db.names() == ["ok"] and list(db.errors) == ["bad"] and db.errors["bad"]


def test_history_and_as_of(tmp_path):
    write_synthetic_dataset(tmp_path / "ds", 10, 0)
    time.sleep(0.01)
    write_synthetic_dataset(tmp_path / "ds", 20, 1, overwrite=True)
    db = Database(tmp_path)
    assert [m.validity.version for m in db.history("ds")] == [1, 2]
    now = datetime.now(UTC)
    sub = db.as_of(now)
    assert "ds" in sub and sub.as_of_time == now and db.as_of_time is None
    assert "ds" not in db.as_of(datetime(2000, 1, 1, tzinfo=UTC))
    with pytest.raises(ValidationError):
        db.as_of(datetime(2030, 1, 1))


def test_as_of_excludes_when_live_postdates(tmp_path):
    write_synthetic_dataset(tmp_path / "ds", 10, 0)
    time.sleep(0.01)
    t = datetime.now(UTC)
    time.sleep(0.01)
    write_synthetic_dataset(tmp_path / "ds", 20, 1, overwrite=True)
    assert "ds" not in Database(tmp_path).as_of(t)


def test_summary_is_json_safe(tmp_path):
    write_synthetic_dataset(tmp_path / "ds", 10, 0)
    s = Database(tmp_path).summary()
    json.dumps(s)
    assert s["datasets"]["ds"]["n_rows"] == 10 and s["datasets"]["ds"]["version"] == 1
    assert s["oneuids"] == [] and s["links"] == []


def test_name_collision(tmp_path):
    write_synthetic_dataset(tmp_path / "a/b", 10, 0)
    write_synthetic_dataset(tmp_path / "a__b", 10, 1)
    write_synthetic_dataset(tmp_path / "ok", 10, 2)
    with pytest.raises(FormatError, match="a__b") as ei:
        Database(tmp_path)
    assert str(tmp_path / "a" / "b") in str(ei.value)
    db = Database(tmp_path, strict=False)
    assert db.names() == ["ok"]
    assert str(tmp_path / "a" / "b") in db.errors["a__b"] and str(tmp_path / "a__b") in db.errors["a__b"]


def test_symlinked_dirs_skipped(tmp_path):
    write_synthetic_dataset(tmp_path / "a/b", 10, 0)
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    assert Database(tmp_path).names() == ["a__b"]


def test_as_of_boundary_and_timezone(tmp_path):
    from datetime import timedelta
    from datetime import timezone as tz
    write_synthetic_dataset(tmp_path / "ds", 10, 0)
    vf = datetime.fromisoformat(Database(tmp_path).manifest("ds").validity.valid_from)
    assert "ds" in Database(tmp_path).as_of(vf)
    assert "ds" not in Database(tmp_path).as_of(vf - timedelta(microseconds=1))
    plus2 = tz(timedelta(hours=2))
    assert "ds" in Database(tmp_path).as_of(vf.astimezone(plus2))
    assert "ds" not in Database(tmp_path).as_of((vf - timedelta(hours=1)).astimezone(plus2))
