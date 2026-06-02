"""Toy lightcone catalogue (galaxy-lightcone product, HEALPix sky).

This is the OUF-Sim "lightcone" product — the sky-partitioned case that
mirrors how OUF stores survey catalogues (HEALPix NSIDE32 NEST). A dummy
observer sits at the box centre; for every redshift snapshot the halos
are projected onto the sphere (lon, lat, comoving radius) and tagged with
that shell redshift. Each object gets a ``_healpix32`` NEST pixel exactly
as OUF does, so the store can partition by sky super-pixel.

Pure numpy + healpy. No cosmology engine — the shell redshift is carried
verbatim from the snapshot label (Pillar 3 owns cosmology, but this dummy
needs none for the projection itself).
"""
from __future__ import annotations

from typing import Dict, Sequence

import healpy as hp
import numpy as np

from oneuniverse.simulation.linear.halos import find_peaks

_HEALPIX_NSIDE = 32  # mirror OUF's _healpix32 column


def _project_to_sky(pos: np.ndarray, centre: np.ndarray):
    """Return (lon_deg, lat_deg, radius) for ``pos`` about ``centre``."""
    d = pos - centre
    r = np.sqrt((d ** 2).sum(axis=1))
    keep = r > 0.0
    d, r = d[keep], r[keep]
    lon = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0
    lat = np.degrees(np.arcsin(np.clip(d[:, 2] / r, -1.0, 1.0)))
    return lon, lat, r, keep


def build_lightcone_catalog(
    fields_by_z: Dict[float, np.ndarray],
    *,
    box_size: float,
    halo_threshold: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Build a toy lightcone halo catalogue from per-z density fields.

    Parameters
    ----------
    fields_by_z
        Mapping ``{redshift: density_field}`` (each an N^3 array).
    box_size
        Box side (Mpc/h); observer is placed at the box centre.
    halo_threshold
        Peak threshold passed to ``find_peaks``.

    Returns
    -------
    dict of equal-length arrays: ``lon``, ``lat``, ``redshift``,
    ``comoving_radius``, ``mass``, ``_healpix32``.
    """
    centre = np.full(3, box_size / 2.0)
    lon_all, lat_all, z_all, r_all, m_all = [], [], [], [], []
    for z in sorted(fields_by_z):
        halos = find_peaks(fields_by_z[z], box_size=box_size,
                           threshold=halo_threshold)
        if len(halos["halo_id"]) == 0:
            continue
        pos = np.stack([halos["x"], halos["y"], halos["z"]], axis=1)
        lon, lat, r, keep = _project_to_sky(pos, centre)
        lon_all.append(lon)
        lat_all.append(lat)
        z_all.append(np.full(lon.shape, float(z)))
        r_all.append(r)
        m_all.append(np.asarray(halos["mass"])[keep])

    if not lon_all:
        empty = np.empty(0, dtype=np.float64)
        return {
            "lon": empty, "lat": empty, "redshift": empty,
            "comoving_radius": empty, "mass": empty,
            "_healpix32": np.empty(0, dtype=np.int32),
        }

    lon = np.concatenate(lon_all)
    lat = np.concatenate(lat_all)
    theta = np.radians(90.0 - lat)
    phi = np.radians(lon)
    healpix32 = hp.ang2pix(_HEALPIX_NSIDE, theta, phi, nest=True).astype(np.int32)
    return {
        "lon": lon,
        "lat": lat,
        "redshift": np.concatenate(z_all),
        "comoving_radius": np.concatenate(r_all),
        "mass": np.concatenate(m_all),
        "_healpix32": healpix32,
    }
