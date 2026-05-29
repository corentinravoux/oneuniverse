"""Phase 18 T3 — ProbabilisticRedshift handles sample + hist."""
import numpy as np
import pandas as pd

from oneuniverse.data.pdf import PdfSpec, ProbabilisticRedshift


def test_sample_mean_matches_empirical():
    spec = PdfSpec(
        parameterisation="sample", n_components=4,
        grid=None, grid_kind="z",
    )
    df = pd.DataFrame({
        "z_pdf_values": [
            np.array([0.10, 0.30], dtype="f4"),
            np.array([0.50, 0.50, 0.50], dtype="f4"),
            np.array([0.70, 0.80, 0.90, 1.00], dtype="f4"),
        ],
    })
    pz = ProbabilisticRedshift.from_dataframe(df, spec)
    means = pz.mean()
    np.testing.assert_allclose(means[0], 0.20, atol=1e-5)
    np.testing.assert_allclose(means[1], 0.50, atol=1e-5)
    np.testing.assert_allclose(means[2], 0.85, atol=1e-5)


def test_hist_mean_uses_edges():
    spec = PdfSpec(
        parameterisation="hist", n_components=4,
        grid=None, grid_kind="z",
        hist_edges=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    df = pd.DataFrame({
        "z_pdf_values": [
            np.array([1.0, 0.0, 0.0, 0.0], dtype="f4"),
            np.array([0.0, 0.0, 0.0, 1.0], dtype="f4"),
            np.array([0.25, 0.25, 0.25, 0.25], dtype="f4"),
        ],
    })
    pz = ProbabilisticRedshift.from_dataframe(df, spec)
    centres = np.array([0.125, 0.375, 0.625, 0.875])
    means = pz.mean()
    np.testing.assert_allclose(means[0], centres[0], atol=1e-5)
    np.testing.assert_allclose(means[1], centres[3], atol=1e-5)
    np.testing.assert_allclose(means[2], centres.mean(), atol=1e-5)
