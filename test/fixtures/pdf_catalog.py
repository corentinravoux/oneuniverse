"""Synthetic photo-z catalog: one Gaussian PDF per object on a uniform grid."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def make_gaussian_pdf_catalog(
    n_rows: int = 100,
    n_grid: int = 201,
    z_min: float = 0.0,
    z_max: float = 2.0,
    sigma_range: Tuple[float, float] = (0.02, 0.08),
    seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return ``(df, grid)`` with one normalised Gaussian per row.

    Columns: ``ra``, ``dec``, ``z_true``, ``z_pdf_kind``, ``z_pdf_values``,
    ``z_pdf_mean``, ``z_pdf_std``.
    """
    rng = np.random.default_rng(seed)
    grid = np.linspace(z_min, z_max, n_grid, dtype=np.float32)
    dz = grid[1] - grid[0]

    mu = rng.uniform(z_min + 0.1, z_max - 0.1, size=n_rows).astype(np.float32)
    sigma = rng.uniform(*sigma_range, size=n_rows).astype(np.float32)
    diff = grid[None, :] - mu[:, None]
    pdfs = np.exp(-0.5 * (diff / sigma[:, None]) ** 2)
    pdfs /= pdfs.sum(axis=1, keepdims=True) * dz
    pdfs = pdfs.astype(np.float32)

    ra = rng.uniform(150.0, 160.0, size=n_rows).astype(np.float64)
    dec = rng.uniform(0.0, 10.0, size=n_rows).astype(np.float64)

    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "z_true": mu.astype(np.float32),
        "z_pdf_kind": np.array(["interp"] * n_rows, dtype="U8"),
        "z_pdf_values": [row for row in pdfs],
        "z_pdf_mean": mu,
        "z_pdf_std": sigma,
    })
    return df, grid
