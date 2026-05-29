"""Phase 21 T3 — composite_id optional CORE column."""
import numpy as np
import pandas as pd

from oneuniverse.data.schema import (
    CORE_COLUMNS,
    get_all_columns,
    validate_dataframe,
)


def test_composite_id_in_core_columns():
    names = {c.name for c in CORE_COLUMNS}
    assert "composite_id" in names


def test_composite_id_is_optional_string():
    by_name = {c.name: c for c in CORE_COLUMNS}
    col = by_name["composite_id"]
    assert col.required is False
    assert col.dtype.startswith("U")


def test_dataframe_without_composite_id_still_validates():
    df = pd.DataFrame({
        "ra": np.array([0.0], dtype="f8"),
        "dec": np.array([0.0], dtype="f8"),
        "z": np.array([0.5], dtype="f4"),
        "z_type": np.array(["spec"], dtype=object),
        "z_err": np.array([0.001], dtype="f4"),
        "galaxy_id": np.array([0], dtype="i8"),
        "survey_id": np.array(["x"], dtype=object),
        "_original_row_index": np.array([0], dtype="i8"),
        "_healpix32": np.array([0], dtype="i4"),
    })
    assert validate_dataframe(df, ["core"]) == []


def test_dataframe_with_composite_id_string_validates():
    df = pd.DataFrame({
        "ra": np.array([0.0], dtype="f8"),
        "dec": np.array([0.0], dtype="f8"),
        "z": np.array([0.5], dtype="f4"),
        "z_type": np.array(["spec"], dtype=object),
        "z_err": np.array([0.001], dtype="f4"),
        "galaxy_id": np.array([0], dtype="i8"),
        "survey_id": np.array(["x"], dtype=object),
        "_original_row_index": np.array([0], dtype="i8"),
        "_healpix32": np.array([0], dtype="i4"),
        "composite_id": np.array(["3551-55065-0010"], dtype=object),
    })
    assert validate_dataframe(df, ["core"]) == []
