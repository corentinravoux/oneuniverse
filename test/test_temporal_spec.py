import pytest
from oneuniverse.data.temporal import TemporalSpec


def test_defaults():
    s = TemporalSpec(t_min=58000.0, t_max=60000.0)
    assert s.time_column == "t_obs"
    assert s.time_unit == "MJD"
    assert s.time_reference == "TDB"
    assert s.cadence is None


def test_frozen():
    s = TemporalSpec(t_min=0.0, t_max=1.0)
    with pytest.raises((AttributeError, TypeError)):
        s.t_min = 2.0  # type: ignore[misc]


def test_rejects_inverted_range():
    with pytest.raises(ValueError, match="t_min"):
        TemporalSpec(t_min=10.0, t_max=5.0)


def test_rejects_unknown_time_reference():
    with pytest.raises(ValueError, match="time_reference"):
        TemporalSpec(t_min=0.0, t_max=1.0, time_reference="INVALID")


def test_dict_roundtrip():
    s = TemporalSpec(t_min=58000.0, t_max=60000.0, cadence=7.0,
                     time_column="mjd")
    assert TemporalSpec.from_dict(s.to_dict()) == s
