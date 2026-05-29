"""Phase 18 T4 — TomographicNzSpec sub-spec."""
import pytest

from oneuniverse.data.tomographic_nz import TomographicNzSpec


def test_defaults_and_required_fields():
    spec = TomographicNzSpec(
        bin_edges=[(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)],
        grid=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        values=[
            [0.0] * 11, [0.0] * 11, [0.0] * 11,
        ],
    )
    assert len(spec.bin_edges) == 3
    assert spec.bin_assignment_column == "tomo_bin"


def test_values_shape_must_match_bins_x_grid():
    with pytest.raises(ValueError, match="values"):
        TomographicNzSpec(
            bin_edges=[(0.0, 0.3), (0.3, 0.6)],
            grid=[0.0, 0.5, 1.0],
            values=[[0.0, 1.0, 0.0]],  # only 1 bin
        )


def test_to_dict_from_dict_roundtrip():
    spec = TomographicNzSpec(
        bin_edges=[(0.0, 0.3), (0.3, 0.6)],
        grid=[0.0, 0.5, 1.0],
        values=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        bin_assignment_column="tbin",
    )
    d = spec.to_dict()
    restored = TomographicNzSpec.from_dict(d)
    assert restored == spec
