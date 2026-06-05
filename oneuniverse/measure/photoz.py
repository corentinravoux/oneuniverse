"""Photo-z kernel: P1's per-object p(z) attached as the measure atom."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.pdf import ProbabilisticRedshift


def attach_photoz(view: DatasetView) -> ProbabilisticRedshift:
    """Return the per-object photo-z kernel (qp) from an OUF PDF dataset."""
    return view.load_pdf()


@dataclass
class PhotozArrays:
    """A light grid+values photo-z kernel view — the round-trippable form a
    saved MeasurementSet restores (a full qp ``ProbabilisticRedshift`` needs the
    source dataset; the grid + per-object p(z) are what estimators consume)."""
    grid: np.ndarray            # (n_grid,)
    values: np.ndarray          # (n_obj, n_grid) normalised p(z)

    def mean(self) -> np.ndarray:
        dz = np.gradient(self.grid)
        return (self.values * self.grid[None, :] * dz[None, :]).sum(axis=1)

    def std(self) -> np.ndarray:
        dz = np.gradient(self.grid)
        m = self.mean()
        var = (self.values * (self.grid[None, :] - m[:, None]) ** 2
               * dz[None, :]).sum(axis=1)
        return np.sqrt(np.clip(var, 0, None))

    def sample(self, n_per: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        cdf = np.cumsum(self.values, axis=1)
        cdf = cdf / cdf[:, -1:]
        u = rng.uniform(size=(self.values.shape[0], n_per))
        idx = np.array([np.searchsorted(cdf[i], u[i]) for i in range(len(cdf))])
        return self.grid[np.clip(idx, 0, len(self.grid) - 1)]
