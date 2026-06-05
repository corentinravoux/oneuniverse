"""Step 6: radial selection n(z). Records the estimation method (provenance)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Nz:
    edges: np.ndarray
    counts: np.ndarray               # weighted counts per bin
    method: str                      # "spec_hist" | "photo_stack" | "clustering_z"

    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def pdf(self) -> np.ndarray:
        width = np.diff(self.edges)
        area = float((self.counts * width).sum())
        return self.counts / area if area > 0 else self.counts


def nz_from_spec_z(z, *, edges, weights: Optional[np.ndarray] = None) -> Nz:
    counts, _ = np.histogram(np.asarray(z), bins=edges, weights=weights)
    return Nz(edges=np.asarray(edges, float), counts=counts.astype(float),
              method="spec_hist")
