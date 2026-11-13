"""Fake survey FITS writers (astropy, so columns are big-endian like real files)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits


def _write(out: Path, arrays: dict[str, np.ndarray]) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = []
    for name, a in arrays.items():
        if a.dtype.kind in "US":
            width = max(int(a.dtype.itemsize // (4 if a.dtype.kind == "U" else 1)), 1)
            cols.append(fits.Column(name=name, format=f"{width}A", array=a.astype(f"S{width}")))
        elif a.ndim == 2:
            code = {"f4": "E", "f8": "D", "i4": "J"}[a.dtype.str[1:]]
            cols.append(fits.Column(name=name, format=f"{a.shape[1]}{code}", array=a))
        else:
            code = {"f4": "E", "f8": "D", "i2": "I", "i4": "J", "i8": "K"}[a.dtype.str[1:]]
            cols.append(fits.Column(name=name, format=code, array=a))
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(out, overwrite=True)
    return out


def write_fake_desi_dr1_fits(out_dir: Path, n_rows: int = 5000, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n = n_rows
    z = rng.uniform(0.8, 3.5, n)
    dec = rng.uniform(-10.0, 60.0, n)
    f4 = lambda lo, hi: rng.uniform(lo, hi, n).astype("f4")
    f8 = lambda lo, hi: rng.uniform(lo, hi, n).astype("f8")
    i4 = lambda lo, hi: rng.integers(lo, hi, n).astype("i4")
    flux = {b: f4(0.1, 20.0) for b in "GRZ"}
    flux["G"][::97] = -1.0          # non-positive fluxes exercise the NaN-magnitude path
    arr = {
        "RA": rng.uniform(120.0, 260.0, n), "DEC": dec, "Z_RR": z, "ZERR": f8(1e-4, 1e-3),
        "ZWARN": np.where(rng.uniform(size=n) < 0.98, 0, 4).astype("i4"),
        "SPECTYPE": np.where(rng.uniform(size=n) < 0.9, "QSO", "GALAXY").astype("<U6"),
        "SUBTYPE": np.where(z > 2.1, "LYA", "HIZ").astype("<U8"),
        "DELTACHI2": f8(10.0, 300.0),
        "TARGETID": (39_627_000_000 + np.arange(n)).astype("i8"),
        "Z_QN": z + rng.normal(0, 1e-3, n), "Z_QN_CONF": f8(0.5, 1.0),
        "IS_QSO_QN": np.ones(n, dtype="i2"), "NTILE": i4(1, 4), "FIBER": i4(0, 5000),
        "FLUX_G": flux["G"], "FLUX_R": flux["R"], "FLUX_Z": flux["Z"],
        "FLUX_IVAR_G": f4(1, 500), "FLUX_IVAR_R": f4(1, 500), "FLUX_IVAR_Z": f4(1, 500),
        "FLUX_W1": f4(0.1, 20), "FLUX_W2": f4(0.1, 20),
        "FLUX_IVAR_W1": f4(1, 500), "FLUX_IVAR_W2": f4(1, 500),
        "MW_TRANSMISSION_G": f4(0.7, 1), "MW_TRANSMISSION_R": f4(0.7, 1),
        "MW_TRANSMISSION_Z": f4(0.7, 1), "MW_TRANSMISSION_W1": f4(0.9, 1),
        "MW_TRANSMISSION_W2": f4(0.9, 1), "EBV": f4(0, 0.3),
        "TSNR2_QSO": f4(1, 100), "TSNR2_LYA": f4(0, 50),
        "COADD_NUMEXP": i4(1, 8), "COADD_EXPTIME": f4(1000, 4000), "COADD_NUMNIGHT": i4(1, 3),
        "WEIGHT_ZFAIL": f8(0.95, 1.05), "COMP_TILE": f8(0.8, 1), "FRACZ_TILELOCID": f8(0.5, 1),
        "FRAC_TLOBS_TILES": f8(0.5, 1), "PROB_OBS": f8(0.5, 1),
        "DESI_TARGET": rng.integers(1, 1 << 20, n).astype("i8"),
        "MORPHTYPE": np.full(n, "PSF", dtype="<U4"),
        "PHOTSYS": np.where(dec > 32.0, "N", "S").astype("<U1"),
    }
    return _write(Path(out_dir) / "QSO_full.dat.fits", arr)


def write_fake_eboss_dr16q_fits(out_dir: Path, n_rows: int = 5000, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n = n_rows
    f4 = lambda lo, hi: rng.uniform(lo, hi, n).astype("f4")
    f8 = lambda lo, hi: rng.uniform(lo, hi, n).astype("f8")
    i4 = lambda lo, hi: rng.integers(lo, hi, n).astype("i4")
    z_dla = np.full((n, 5), -1.0, dtype="f4")
    n_found = rng.integers(0, 4, n)
    for k in range(3):
        has = n_found > k
        z_dla[has, k] = rng.uniform(2.0, 4.0, has.sum())
    nhi = np.where(z_dla > 0, rng.uniform(20.0, 22.0, (n, 5)), -1.0).astype("f4")
    conf = np.where(z_dla > 0, rng.uniform(0.2, 1.0, (n, 5)), -1.0).astype("f4")
    arr = {
        "RA": rng.uniform(0, 360, n), "DEC": rng.uniform(-10, 70, n), "Z": f8(0.1, 5.0),
        "IS_QSO_FINAL": rng.choice([1, 0, -2, 2], n, p=[0.7, 0.2, 0.05, 0.05]).astype("i2"),
        "SOURCE_Z": rng.choice(["VI", "PIPE", "DR12QV"], n).astype("<U8"),
        "Z_PIPE": f8(0, 5), "Z_PCA": f8(0, 5), "Z_VI": f8(0, 5), "Z_QN": f8(0, 5),
        "Z_CONF": rng.integers(0, 4, n).astype("i2"),
        "ZWARNING": np.where(rng.uniform(size=n) < 0.95, 0, 4).astype("i4"),
        "CLASS_PERSON": rng.choice([3, 30, 1, 4], n).astype("i2"),
        "BAL_PROB": f4(0, 1), "BI_CIV": f8(0, 5000),
        "Z_MGII": f8(0, 5), "Z_CIII": f8(0, 5), "Z_CIV": f8(0, 5), "Z_LYA": f8(0, 5),
        "Z_HALPHA": f8(0, 5), "Z_HBETA": f8(0, 5),
        "PLATE": i4(266, 12000), "MJD": i4(51000, 59000), "FIBERID": i4(1, 1000),
        "THING_ID": rng.integers(1, 10**9, n).astype("i8"),
        "SDSS_NAME": np.array([f"J{i:06d}.00+000000.0" for i in range(n)], dtype="<U18"),
        "BOSS_TARGET1": rng.integers(0, 1 << 30, n).astype("i8"),
        "EBOSS_TARGET0": rng.integers(0, 1 << 20, n).astype("i8"),
        "EBOSS_TARGET1": rng.integers(0, 1 << 20, n).astype("i8"),
        "EBOSS_TARGET2": rng.integers(0, 1 << 20, n).astype("i8"),
        "SN_MEDIAN_ALL": f4(0, 30), "LAMBDA_EFF": f4(4000, 5500), "NSPEC": i4(1, 5),
        "PSFMAG": rng.uniform(17, 23, (n, 5)).astype("f4"),
        "EXTINCTION": rng.uniform(0, 0.5, (n, 5)).astype("f4"),
        "Z_DLA": z_dla, "NHI_DLA": nhi, "CONF_DLA": conf,
    }
    return _write(Path(out_dir) / "DR16Q_Superset_v3.fits", arr)
