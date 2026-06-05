"""Step 5: angular footprint as a HEALPix completeness mask."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    systematics: Optional[dict] = None   # name -> HEALPix map (depth/seeing/PSF)
    polygon_path: Optional[str] = None   # mangle/MOC escape hatch (exact masks)

    def contains(self, ra, dec) -> np.ndarray:
        return self.mask[_ang2pix(ra, dec, self.nside)] > 0.0

    def covered_fraction(self) -> float:
        return float((self.mask > 0).sum()) / self.mask.size

    def with_systematics(self, **maps) -> "Window":
        """Return a copy carrying named depth/systematics HEALPix maps."""
        return Window(nside=self.nside, mask=self.mask, systematics=dict(maps),
                      polygon_path=self.polygon_path)


def footprint_from_positions(ra, dec, *, nside: int = 256) -> Window:
    """Binary completeness inferred from object positions (a **stop-gap**).

    Pixels containing >=1 object are marked covered. This is circular — the true
    selection mask is an *input*, not the object distribution — so prefer
    :func:`window_from_mask` whenever the survey's angular mask is available.
    Used when no mask is supplied (e.g. a quasar superset without an LSS mask).
    """
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=np.float64)
    pix = _ang2pix(ra, dec, nside)
    mask[np.unique(pix)] = 1.0
    return Window(nside=int(nside), mask=mask)


def window_from_mask(completeness, *, nside: int, nest: bool = True,
                     systematics: Optional[dict] = None,
                     polygon_path: Optional[str] = None) -> Window:
    """Build a Window from the survey's **own** angular mask (the correct path).

    ``completeness`` is a HEALPix map (fractional completeness in [0, 1], or a
    boolean coverage mask) at ``nside``. Optionally carries named depth/
    systematics maps and a mangle/MOC polygon reference for exact edges.
    """
    completeness = np.asarray(completeness, float)
    expected = hp.nside2npix(nside)
    if completeness.shape[0] != expected:
        raise ValueError(
            f"window_from_mask: completeness has {completeness.shape[0]} pixels "
            f"but nside={nside} needs {expected}")
    if not nest:                                  # store NEST-ordered internally
        completeness = hp.reorder(completeness, r2n=True)
    return Window(nside=int(nside), mask=completeness, systematics=systematics,
                  polygon_path=polygon_path)
