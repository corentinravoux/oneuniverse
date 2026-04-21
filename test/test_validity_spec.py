import datetime as dt

import pytest

from oneuniverse.data.validity import DatasetValidity


_T1 = "2026-01-01T00:00:00+00:00"
_T2 = "2026-06-01T00:00:00+00:00"


def test_defaults_make_current_entry():
    v = DatasetValidity(valid_from_utc=_T1)
    assert v.valid_to_utc is None
    assert v.version == "1.0"
    assert v.supersedes == ()
    assert v.is_current()


def test_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="timezone"):
        DatasetValidity(valid_from_utc="2026-01-01T00:00:00")


def test_rejects_inverted_interval():
    with pytest.raises(ValueError, match="valid_from"):
        DatasetValidity(valid_from_utc=_T2, valid_to_utc=_T1)


def test_contains_happy_path():
    v = DatasetValidity(valid_from_utc=_T1, valid_to_utc=_T2)
    assert v.contains(dt.datetime.fromisoformat(_T1))
    middle = dt.datetime.fromisoformat("2026-03-01T00:00:00+00:00")
    assert v.contains(middle)
    assert not v.contains(dt.datetime.fromisoformat(_T2))


def test_is_current_after_close():
    v = DatasetValidity(valid_from_utc=_T1, valid_to_utc=_T2)
    after = dt.datetime.fromisoformat("2027-01-01T00:00:00+00:00")
    assert not v.is_current(now=after)


def test_closed_at_returns_new_instance():
    v = DatasetValidity(valid_from_utc=_T1)
    closed = v.closed_at(_T2)
    assert closed.valid_to_utc == _T2
    assert v.valid_to_utc is None


def test_dict_roundtrip():
    v = DatasetValidity(
        valid_from_utc=_T1, valid_to_utc=_T2,
        version="dr17", supersedes=("eboss_qso_dr16",),
    )
    assert DatasetValidity.from_dict(v.to_dict()) == v
