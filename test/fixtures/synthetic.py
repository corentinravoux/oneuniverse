"""Synthetic OUF 3.0 datasets shared by the data-layer tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from oneuniverse.data.manifest import LoaderInfo, Manifest, SurveyMeta
from oneuniverse.data.schema import Geometry, stable_object_id
from oneuniverse.data.writer import write_dataset


def synthetic_point_table(n: int, seed: int, *, survey: str = "synthetic",
                          ra_range=(0.0, 360.0), dec_range=(-90.0, 90.0),
                          z_range=(0.01, 2.0), with_pdf=None) -> pa.Table:
    """Core POINT columns (no ``hpix32``) plus ``mag``; rows shuffled.

    ``source_row`` is the index before shuffling, so file order differs from
    source order. ``dec`` is uniform in sin(dec) inside *dec_range*.
    """
    if with_pdf is not None:
        raise NotImplementedError("with_pdf is wired up with the pdf module")
    rng = np.random.default_rng(seed)
    ra = rng.uniform(*ra_range, n)
    s0, s1 = np.sin(np.radians(dec_range[0])), np.sin(np.radians(dec_range[1]))
    dec = np.degrees(np.arcsin(rng.uniform(s0, s1, n)))
    z = rng.uniform(*z_range, n)
    z_err = rng.uniform(1e-4, 1e-3, n).astype(np.float32)
    mag = rng.uniform(16.0, 22.0, n).astype(np.float32)
    src = np.arange(n, dtype=np.int64)
    perm = rng.permutation(n)
    ra, dec, z, z_err, mag, src = (a[perm] for a in (ra, dec, z, z_err, mag, src))
    return pa.table({
        "ra": ra, "dec": dec, "z": z,
        "z_type": pa.array(["spec"] * n, pa.string()),
        "z_err": z_err,
        "object_id": stable_object_id(survey, src),
        "survey": pa.array([survey] * n, pa.string()),
        "source_row": src,
        "mag": mag,
    })


def write_synthetic_dataset(out: Path, n: int, seed: int, *, partition_nside=None,
                            overwrite: bool = False, **kw) -> Manifest:
    return write_dataset(
        synthetic_point_table(n, seed, **kw), out, geometry=Geometry.POINT,
        survey=SurveyMeta("synthetic", "test"), loader=LoaderInfo("test", "1", {}),
        partition_nside=partition_nside, extra_stats_columns=["mag"],
        overwrite=overwrite)


def synthetic_sightline_table(n: int, seed: int, *, n_pixels: int = 50) -> pa.Table:
    rng = np.random.default_rng(seed)
    lam = np.linspace(3600, 5000, n_pixels).astype(np.float32)
    f4 = pa.list_(pa.float32())
    return pa.table({
        "sightline_id": np.arange(n, dtype=np.int64),
        "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-30, 30, n),
        "z": rng.uniform(2, 3, n), "survey": ["lya"] * n,
        "source_row": np.arange(n, dtype=np.int64),
        "wavelength": pa.array([lam] * n, f4),
        "delta": pa.array([rng.normal(0, 0.1, n_pixels).astype(np.float32)
                           for _ in range(n)], f4),
        "weight": pa.array([np.ones(n_pixels, np.float32)] * n, f4),
    })


def write_synthetic_sightlines(out: Path, n: int, seed: int, *, n_pixels: int = 50,
                               overwrite: bool = False) -> Manifest:
    return write_dataset(
        synthetic_sightline_table(n, seed, n_pixels=n_pixels), out,
        geometry=Geometry.SIGHTLINE, survey=SurveyMeta("lya", "lyman_alpha"),
        loader=LoaderInfo("test", "1", {}), overwrite=overwrite)
