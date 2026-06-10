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
    # jitter can cross a footprint-edge pixel; snap escapers back to the
    # (covered) source-pixel centre so randoms stay strictly in-window.
    outside = ~window.contains(ra, dec)
    if outside.any():
        th, ph = hp.pix2ang(window.nside, pix[outside], nest=True)
        ra[outside] = np.degrees(ph)
        dec[outside] = 90.0 - np.degrees(th)
    # inverse-CDF sample z from the n(z) histogram
    cdf = np.cumsum(nz.counts); cdf = cdf / cdf[-1]
    u = rng.uniform(size=n_randoms)
    bins = np.clip(np.searchsorted(cdf, u), 0, len(nz.edges) - 2)
    z = nz.edges[bins] + rng.uniform(size=n_randoms) * np.diff(nz.edges)[bins]
    rnd = pd.DataFrame({"ra": ra, "dec": dec, "z": z,
                        "weight": np.ones(n_randoms)})
    return rnd, "generated"


def _uniform_in_pixels(pix, nside, rng):
    """Jittered sky point inside each NEST pixel (stays at the window NSIDE)."""
    theta, phi = hp.pix2ang(nside, pix, nest=True)
    res = hp.nside2resol(nside)                 # rad
    theta = np.clip(theta + (rng.uniform(size=len(pix)) - 0.5) * res, 1e-6,
                    np.pi - 1e-6)
    phi = (phi + (rng.uniform(size=len(pix)) - 0.5) * res) % (2 * np.pi)
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)
    return ra, dec
