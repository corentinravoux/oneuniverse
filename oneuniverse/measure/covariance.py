"""Row-correlated covariance handle (e.g. Pantheon+ 1701x1701). Lazy load."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CovarianceHandle:
    cov_id: str
    path: str
    n: int
    _cache: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def is_loaded(self) -> bool:
        return self._cache is not None

    def matrix(self) -> np.ndarray:
        if self._cache is None:
            mat = np.load(self.path)
            if mat.shape != (self.n, self.n):
                raise ValueError(
                    f"CovarianceHandle({self.cov_id}): matrix shape "
                    f"{mat.shape} != ({self.n},{self.n})")
            self._cache = mat
        return self._cache


@dataclass
class CovariancePlan:
    """How a MeasurementSet's covariance will be built (no cosmology stored).

    ``jackknife`` uses the shared region map; ``mocks`` references a mock suite;
    ``analytic`` carries the ingredients (n̄, shot noise, window multipoles) the
    Pillar-2 estimator needs. ``handle`` attaches an external row-correlated
    matrix (e.g. Pantheon+).
    """
    kind: str                                  # jackknife | mocks | analytic | external
    region_nside: Optional[int] = None
    mocks_handle: Optional[str] = None
    handle: Optional["CovarianceHandle"] = None
    ingredients: Optional[dict] = None

    def __post_init__(self) -> None:
        known = {"jackknife", "mocks", "analytic", "external"}
        if self.kind not in known:
            raise ValueError(
                f"CovariancePlan.kind must be one of {sorted(known)}, "
                f"got {self.kind!r}")
