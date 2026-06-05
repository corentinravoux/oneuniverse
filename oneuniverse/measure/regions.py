"""Step 8: shared HEALPix region_id (jackknife/bootstrap basis)."""
from __future__ import annotations

import healpy as hp
import numpy as np


def assign_regions(ra, dec, *, nside: int = 8) -> np.ndarray:
    """NEST HEALPix pixel id at ``nside`` — the shared resampling scheme."""
    theta = np.radians(90.0 - np.asarray(dec))
    phi = np.radians(np.asarray(ra))
    return hp.ang2pix(nside, theta, phi, nest=True).astype(np.int64)
