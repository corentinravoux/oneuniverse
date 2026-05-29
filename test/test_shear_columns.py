"""Phase 19 T1 — SHEAR_COLUMNS group + schema integration."""
import numpy as np
import pandas as pd

from oneuniverse.data.schema import (
    COLUMN_GROUPS,
    SHEAR_COLUMNS,
    get_all_columns,
    validate_dataframe,
)


def test_shear_columns_registered():
    assert "shear" in COLUMN_GROUPS
    assert COLUMN_GROUPS["shear"] is SHEAR_COLUMNS


def test_shear_columns_contents():
    names = {c.name for c in SHEAR_COLUMNS}
    expected = {
        "e1", "e2", "e1_err", "e2_err",
        "R11", "R22", "R12", "R21", "R_S",
        "m_bias", "c1_bias", "c2_bias",
        "shear_weight",
    }
    assert expected <= names


def test_no_shear_column_required_by_default():
    for c in SHEAR_COLUMNS:
        assert c.required is False


def test_validate_dataframe_accepts_shear_subset():
    df = pd.DataFrame({
        "e1": np.array([0.1, 0.0], dtype="f4"),
        "e2": np.array([0.0, -0.1], dtype="f4"),
        "shear_weight": np.array([1.0, 1.0], dtype="f4"),
    })
    warnings = validate_dataframe(df, ["shear"])
    assert warnings == []


def test_get_all_columns_includes_shear():
    cols = get_all_columns(["shear"])
    assert "e1" in cols and "R11" in cols
