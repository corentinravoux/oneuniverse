"""Minimal DESI DR1 QSO-shaped FITS writer for tests."""
from __future__ import annotations

from pathlib import Path

import fitsio
import numpy as np


def write_fake_desi_dr1_fits(
    out_dir: Path, n_rows: int = 5000, seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    ra = rng.uniform(120.0, 260.0, n_rows)
    dec = rng.uniform(-10.0, 60.0, n_rows)
    z = rng.uniform(0.8, 3.5, n_rows)
    zerr = rng.uniform(1e-4, 1e-3, n_rows).astype("f8")

    spectype = np.where(rng.uniform(size=n_rows) < 0.9, "QSO", "GALAXY")
    spectype = np.asarray(spectype, dtype="<U6")
    subtype = np.where(z > 2.1, "LYA", "HIZ").astype("<U8")
    zwarn = np.where(rng.uniform(size=n_rows) < 0.98, 0, 4).astype("i4")
    targetid = (39_627_000_000 + np.arange(n_rows)).astype("i8")

    arr = {
        "RA": ra, "DEC": dec, "Z_RR": z, "ZERR": zerr, "ZWARN": zwarn,
        "SPECTYPE": spectype, "SUBTYPE": subtype,
        "DELTACHI2": rng.uniform(10.0, 300.0, n_rows).astype("f8"),
        "TARGETID": targetid,
        "Z_QN": z + rng.normal(0, 1e-3, n_rows).astype("f8"),
        "Z_QN_CONF": rng.uniform(0.5, 1.0, n_rows).astype("f8"),
        "IS_QSO_QN": np.ones(n_rows, dtype="i2"),
        "NTILE": rng.integers(1, 4, n_rows).astype("i4"),
        "FIBER": rng.integers(0, 5000, n_rows).astype("i4"),
        "FLUX_G": rng.uniform(0.1, 20.0, n_rows).astype("f4"),
        "FLUX_R": rng.uniform(0.1, 20.0, n_rows).astype("f4"),
        "FLUX_Z": rng.uniform(0.1, 20.0, n_rows).astype("f4"),
        "FLUX_IVAR_G": rng.uniform(1.0, 500.0, n_rows).astype("f4"),
        "FLUX_IVAR_R": rng.uniform(1.0, 500.0, n_rows).astype("f4"),
        "FLUX_IVAR_Z": rng.uniform(1.0, 500.0, n_rows).astype("f4"),
        "FLUX_W1": rng.uniform(0.1, 20.0, n_rows).astype("f4"),
        "FLUX_W2": rng.uniform(0.1, 20.0, n_rows).astype("f4"),
        "FLUX_IVAR_W1": rng.uniform(1.0, 500.0, n_rows).astype("f4"),
        "FLUX_IVAR_W2": rng.uniform(1.0, 500.0, n_rows).astype("f4"),
        "MW_TRANSMISSION_G": rng.uniform(0.7, 1.0, n_rows).astype("f4"),
        "MW_TRANSMISSION_R": rng.uniform(0.7, 1.0, n_rows).astype("f4"),
        "MW_TRANSMISSION_Z": rng.uniform(0.7, 1.0, n_rows).astype("f4"),
        "MW_TRANSMISSION_W1": rng.uniform(0.9, 1.0, n_rows).astype("f4"),
        "MW_TRANSMISSION_W2": rng.uniform(0.9, 1.0, n_rows).astype("f4"),
        "EBV": rng.uniform(0.0, 0.3, n_rows).astype("f4"),
        "TSNR2_QSO": rng.uniform(1.0, 100.0, n_rows).astype("f4"),
        "TSNR2_LYA": rng.uniform(0.0, 50.0, n_rows).astype("f4"),
        "COADD_NUMEXP": rng.integers(1, 8, n_rows).astype("i4"),
        "COADD_EXPTIME": rng.uniform(1000.0, 4000.0, n_rows).astype("f4"),
        "COADD_NUMNIGHT": rng.integers(1, 3, n_rows).astype("i4"),
        "WEIGHT_ZFAIL": rng.uniform(0.95, 1.05, n_rows).astype("f8"),
        "COMP_TILE": rng.uniform(0.8, 1.0, n_rows).astype("f8"),
        "FRACZ_TILELOCID": rng.uniform(0.5, 1.0, n_rows).astype("f8"),
        "FRAC_TLOBS_TILES": rng.uniform(0.5, 1.0, n_rows).astype("f8"),
        "PROB_OBS": rng.uniform(0.5, 1.0, n_rows).astype("f8"),
        "DESI_TARGET": rng.integers(1, 1 << 20, n_rows).astype("i8"),
        "MORPHTYPE": np.full(n_rows, "PSF", dtype="<U4"),
        "PHOTSYS": np.where(dec > 32.0, "N", "S").astype("<U1"),
    }
    out = out_dir / "QSO_full.dat.fits"
    fitsio.write(str(out), arr, clobber=True)
    return out
