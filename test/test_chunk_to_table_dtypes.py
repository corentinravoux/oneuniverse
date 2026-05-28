"""Phase 17 T2 — _chunk_to_table accepts per-column dtype overrides."""
import numpy as np
import pandas as pd
import pyarrow as pa

from oneuniverse.data.converter import _chunk_to_table


def test_fixed_size_list_column():
    df = pd.DataFrame({
        "id": np.arange(3, dtype="i8"),
        "vals": [np.arange(4, dtype="f4"),
                 np.arange(4, dtype="f4") * 2,
                 np.arange(4, dtype="f4") * 3],
    })
    table = _chunk_to_table(df, pdf_spec=None, column_dtypes={"vals": "f4[4]"})
    assert isinstance(table.schema.field("vals").type, pa.FixedSizeListType)
    assert table.schema.field("vals").type.list_size == 4


def test_int_bitweight_column():
    df = pd.DataFrame({
        "id": np.arange(2, dtype="i8"),
        "BITWEIGHTS": [np.zeros(64, dtype="i8"), np.ones(64, dtype="i8")],
    })
    table = _chunk_to_table(
        df, pdf_spec=None, column_dtypes={"BITWEIGHTS": "i8[64]"},
    )
    t = table.schema.field("BITWEIGHTS").type
    assert isinstance(t, pa.FixedSizeListType)
    assert t.list_size == 64


def test_variable_length_list_column():
    df = pd.DataFrame({
        "id": np.arange(3, dtype="i8"),
        "delta": [np.arange(3, dtype="f4"),
                  np.arange(5, dtype="f4"),
                  np.arange(7, dtype="f4")],
    })
    table = _chunk_to_table(
        df, pdf_spec=None, column_dtypes={"delta": "list<f4>"},
    )
    assert isinstance(table.schema.field("delta").type, pa.ListType)
    py = table.column("delta").to_pylist()
    assert [len(x) for x in py] == [3, 5, 7]


def test_large_list_column():
    df = pd.DataFrame({
        "id": np.arange(2, dtype="i8"),
        "lc": [np.arange(10, dtype="f4"), np.arange(20, dtype="f4")],
    })
    table = _chunk_to_table(
        df, pdf_spec=None, column_dtypes={"lc": "large_list<f4>"},
    )
    assert isinstance(table.schema.field("lc").type, pa.LargeListType)
