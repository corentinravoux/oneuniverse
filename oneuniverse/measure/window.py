"""Step 5: angular footprint as a HEALPix completeness mask."""
from __future__ import annotations

from dataclasses import dataclass

import healpy as hp
import numpy as np


def _ang2pix(ra, dec, nside):
    theta = np.radians(90.0 - np.asarray(dec))
    phi = np.radians(np.asarray(ra))
    return hp.ang2pix(nside, theta, phi, nest=True)


@dataclass(frozen=True)
class Window:
    nside: int
    mask: np.ndarray                 # float completeness per NEST pixel [0,1]

    def contains(self, ra, dec) -> np.ndarray:
        return self.mask[_ang2pix(ra, dec, self.nside)] > 0.0

    def covered_fraction(self) -> float:
        return float((self.mask > 0).sum()) / self.mask.size


def footprint_from_positions(ra, dec, *, nside: int = 256) -> Window:
    """Binary completeness: pixels containing >=1 object are covered."""
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=np.float64)
    pix = _ang2pix(ra, dec, nside)
    mask[np.unique(pix)] = 1.0
    return Window(nside=int(nside), mask=mask)
