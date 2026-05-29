"""Phase 22 T2 — GwSkymapSpec sub-spec."""
import pytest

from oneuniverse.data.gwskymap_spec import GwSkymapSpec


def test_defaults():
    spec = GwSkymapSpec(map_nside=32)
    assert spec.map_nside == 32
    assert spec.map_nest is True
    assert spec.has_distance_extras is False


def test_rejects_non_power_of_two_nside():
    with pytest.raises(ValueError, match="power of two"):
        GwSkymapSpec(map_nside=30)


def test_to_from_dict_roundtrip():
    spec = GwSkymapSpec(
        map_nside=64, map_nest=False, has_distance_extras=True,
    )
    d = spec.to_dict()
    assert GwSkymapSpec.from_dict(d) == spec
