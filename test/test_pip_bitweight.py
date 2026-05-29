"""Phase 19 T3 — PipBitweightWeight expands BITWEIGHTS: i8[64]."""
import numpy as np
import pandas as pd
import pytest

from oneuniverse.combine.weights.pip import PipBitweightWeight


def test_fraction_mode_counts_set_bits():
    rows = [
        np.zeros(1, dtype="i8"),
        np.array([-1], dtype="i8"),    # 64 set bits
    ]
    df = pd.DataFrame({"BITWEIGHTS": rows})
    w = PipBitweightWeight(bitweights_col="BITWEIGHTS").compute(df)
    np.testing.assert_allclose(w, np.array([0.0, 1.0]), rtol=1e-6)


def test_fraction_intermediate_value():
    # 32 set bits → 0.5
    val = np.int64(0x00000000FFFFFFFF)
    df = pd.DataFrame({"BITWEIGHTS": [np.array([val], dtype="i8")]})
    w = PipBitweightWeight().compute(df)
    np.testing.assert_allclose(w, np.array([32.0 / 64.0]), rtol=1e-6)


def test_realisations_mode_returns_per_row_array():
    rows = [np.array([0], dtype="i8"), np.array([-1], dtype="i8")]
    df = pd.DataFrame({"BITWEIGHTS": rows})
    w = PipBitweightWeight(mode="realisations").compute(df)
    assert w.shape == (2, 64)
    assert (w[0] == 0).all()
    assert (w[1] == 1).all()


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        PipBitweightWeight(mode="bogus")


def test_missing_column_raises():
    with pytest.raises(KeyError):
        PipBitweightWeight().compute(pd.DataFrame({"x": [1, 2]}))
