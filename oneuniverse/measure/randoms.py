"""Step 4: randoms. Ingest survey-published, or generate from window x n(z).

Owner decision (2026-06-05): both first-class; provenance records which.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.nz import Nz
from oneuniverse.measure.window import Window


def randoms_from_view(view: DatasetView, *,
                      columns: Optional[Sequence[str]] = None
                      ) -> Tuple[pd.DataFrame, str]:
    """Ingest an official random catalog stored as an OUF POINT dataset."""
    rnd = view.read(columns=columns)
    return rnd.reset_index(drop=True), "ingested"


def generate_randoms(window: Window, nz: Nz, *, n_randoms: int,
                     seed: int = 0) -> Tuple[pd.DataFrame, str]:
    """Uniform-in-window angular positions × n(z)-sampled redshifts."""
    rng = np.random.default_rng(seed)
    total = float(np.sum(nz.counts))
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "generate_randoms: n(z) has zero/invalid total weight — cannot "
            "sample redshifts (B3: silent NaN-cdf garbage otherwise)")
    covered = np.nonzero(window.mask > 0)[0]
    if covered.size == 0:
        raise ValueError("generate_randoms: window has no covered pixels")
    probs = window.mask[covered] / window.mask[covered].sum()
    pix = rng.choice(covered, size=n_randoms, p=probs)
    ra, dec = _uniform_in_pixels(pix, window.nside, rng)
    # inverse-CDF sample z from the n(z) histogram
    cdf = np.cumsum(nz.counts); cdf = cdf / cdf[-1]
    u = rng.uniform(size=n_randoms)
    bins = np.clip(np.searchsorted(cdf, u), 0, len(nz.edges) - 2)
    z = nz.edges[bins] + rng.uniform(size=n_randoms) * np.diff(nz.edges)[bins]
    rnd = pd.DataFrame({"ra": ra, "dec": dec, "z": z,
                        "weight": np.ones(n_randoms)})
    return rnd, "generated"


#: sub-pixel refinement factor: each window pixel is sampled via a random
#: NEST child at nside*2**_SUBPIX_ORDER (review B7).
_SUBPIX_ORDER = 5


def _uniform_in_pixels(pix, nside, rng):
    """Area-uniform sky point inside each NEST pixel (review B7).

    HEALPix children are equal-area and strictly nested, so drawing a random
    child at a finer NSIDE and taking its centre is (a) uniform on the sphere
    within the parent — the previous θ/φ jitter compressed φ near the poles —
    and (b) guaranteed to stay inside the parent, removing the old
    snap-escapers-back hack.
    """
    f = 4 ** _SUBPIX_ORDER                       # children per parent
    child = np.asarray(pix, dtype=np.int64) * f + rng.integers(
        0, f, size=len(pix))
    theta, phi = hp.pix2ang(nside * 2 ** _SUBPIX_ORDER, child, nest=True)
    return np.degrees(phi), 90.0 - np.degrees(theta)
