"""Per-tomographic-bin n(z): stack the photo-z kernel within each bin."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from oneuniverse.measure.nz import Nz


def tomographic_nz(catalog: pd.DataFrame, kernel, *, bin_column: str,
                   z_grid: np.ndarray, n_per: int = 10, seed: int = 0
                   ) -> Dict[int, Nz]:
    """Stack kernel samples per bin into an Nz (method='photo_stack')."""
    draws = kernel.sample(n_per, seed=seed)         # (N, n_per)
    out: Dict[int, Nz] = {}
    bins = catalog[bin_column].to_numpy()
    for b in np.unique(bins):
        z = draws[bins == b].ravel()
        counts, _ = np.histogram(z, bins=z_grid)
        out[int(b)] = Nz(edges=np.asarray(z_grid, float),
                         counts=counts.astype(float), method="photo_stack")
    return out
