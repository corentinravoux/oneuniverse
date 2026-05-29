"""Phase 21 T1/T2 — CrossMatchRules.attribute_filters."""
import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.oneuid_crossmatch import cross_match_surveys
from oneuniverse.data.oneuid_rules import CrossMatchRules


def _color_filter(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    """Keep pair iff |Δ(g-r)| < 0.1."""
    dg = (left["psfmag_g"] - left["psfmag_r"]).to_numpy()
    dr = (right["psfmag_g"] - right["psfmag_r"]).to_numpy()
    return np.abs(dg - dr) < 0.1


def test_default_attribute_filters_is_empty():
    r = CrossMatchRules()
    assert r.attribute_filters == ()


def test_attribute_filters_tuple_stored():
    r = CrossMatchRules(attribute_filters=(_color_filter,))
    assert r.attribute_filters == (_color_filter,)


def test_hash_includes_attribute_filters():
    a = CrossMatchRules()
    b = CrossMatchRules(attribute_filters=(_color_filter,))
    assert a.hash() != b.hash()


def test_attribute_filters_must_be_tuple_of_callables():
    with pytest.raises(TypeError, match="callable"):
        CrossMatchRules(attribute_filters=("not_callable",))


# ── T2 matcher integration ──────────────────────────────────────────────


def _build_catalogs():
    """Two surveys, two objects each at (10°, 0°) and (20°, 0°)."""
    a = pd.DataFrame({
        "ra":  [10.0, 20.0],
        "dec": [0.0, 0.0],
        "z":   [0.5, 0.5],
        "z_type": ["spec", "spec"],
        "z_err": [0.001, 0.001],
        "galaxy_id": [0, 1],
        "_original_row_index": [0, 1],
        "psfmag_g": [22.0, 22.0],
        "psfmag_r": [22.0, 21.0],
    })
    b = pd.DataFrame({
        "ra":  [10.000001, 20.000001],
        "dec": [0.0, 0.0],
        "z":   [0.5, 0.5],
        "z_type": ["spec", "spec"],
        "z_err": [0.001, 0.001],
        "galaxy_id": [2, 3],
        "_original_row_index": [0, 1],
        "psfmag_g": [22.0, 22.0],
        "psfmag_r": [21.95, 21.1],
    })
    return {"a": a, "b": b}


def test_attribute_filter_blocks_color_mismatch():
    rules = CrossMatchRules(
        sky_tol_arcsec=2.0,
        attribute_filters=(_color_filter,),
    )
    result = cross_match_surveys(_build_catalogs(), rules)
    # Result is a CrossMatchResult with .table, .n_groups, .n_multi.
    assert result.n_multi == 1


def test_no_filter_keeps_both_matches():
    rules = CrossMatchRules(sky_tol_arcsec=2.0)
    result = cross_match_surveys(_build_catalogs(), rules)
    assert result.n_multi == 2
