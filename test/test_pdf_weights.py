"""PDF-aware weights + ``("photometric", "phot_pdf")`` default."""
from __future__ import annotations

import numpy as np
import pandas as pd

from oneuniverse.combine.weights import (
    PdfMeanRedshiftWeight,
    PdfWidthIVarWeight,
    default_weight_for,
)


def _df():
    return pd.DataFrame({
        "z_pdf_mean": np.array([0.3, 0.5, 0.9], dtype=np.float32),
        "z_pdf_std": np.array([0.04, 0.08, 0.02], dtype=np.float32),
    })


def test_pdf_width_ivar():
    w = PdfWidthIVarWeight(std_column="z_pdf_std")
    got = w(_df())
    expected = 1.0 / _df()["z_pdf_std"].to_numpy(dtype=np.float64) ** 2
    np.testing.assert_allclose(got, expected)


def test_pdf_mean_redshift_weight_reads_mean():
    w = PdfMeanRedshiftWeight(mean_column="z_pdf_mean")
    got = w(_df())
    np.testing.assert_allclose(got, _df()["z_pdf_mean"].to_numpy(dtype=np.float64))


def test_default_weight_for_phot_pdf_registered():
    w = default_weight_for("photometric", "phot_pdf")
    got = w(_df())
    assert got.shape == (3,)
