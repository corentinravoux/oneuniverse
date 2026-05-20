"""Smoke tests for the Gaussian-PDF photo-z catalog fixture."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.pdf_catalog import make_gaussian_pdf_catalog  # noqa: E402


def test_gaussian_catalog_pdf_integrates_to_one():
    df, grid = make_gaussian_pdf_catalog(n_rows=100, n_grid=201, seed=0)
    assert len(df) == 100
    assert len(grid) == 201
    pdfs = np.stack(df["z_pdf_values"].values)
    dz = grid[1] - grid[0]
    integrals = pdfs.sum(axis=1) * dz
    assert np.allclose(integrals, 1.0, atol=1e-2)


def test_gaussian_catalog_mean_matches_column():
    df, grid = make_gaussian_pdf_catalog(n_rows=50, n_grid=301, seed=1)
    pdfs = np.stack(df["z_pdf_values"].values)
    dz = grid[1] - grid[0]
    means = (pdfs * grid[None, :]).sum(axis=1) * dz
    assert np.allclose(means, df["z_pdf_mean"].to_numpy(), atol=1e-2)
