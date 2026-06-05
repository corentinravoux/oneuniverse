"""Synthetic OUF POINT dataset → DatasetView, for measure/ tests."""
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec


def synthetic_point_view(tmp: Path, *, n: int = 3000, seed: int = 0,
                         name: str = "synth") -> DatasetView:
    """Write a synthetic galaxy OUF POINT dataset; return its DatasetView."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(150.0, 170.0, n)
    dec = rng.uniform(0.0, 15.0, n)
    z = np.clip(rng.normal(0.5, 0.12, n), 0.05, 1.2)
    df = pd.DataFrame({
        "ra": ra, "dec": dec, "z": z,
        "z_type": np.full(n, "spec"), "z_err": np.full(n, 1e-4),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.zeros(n, dtype=np.int64),
        "nbar": np.full(n, 1e-3),                  # constant n̄ for FKP
        "weight_comp": rng.uniform(0.9, 1.0, n),   # completeness
        "weight_sys": rng.uniform(0.95, 1.05, n),  # imaging systematics
        "quality": (rng.uniform(size=n) > 0.02).astype(np.int64),  # 2% bad
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })
    out = tmp / name / "oneuniverse"
    write_ouf_dataset(df=df, out_dir=out, survey_name=name,
                      survey_type="spectroscopic", geometry=DataGeometry.POINT,
                      loader=LoaderSpec(name=name, version="0"))
    return DatasetView.from_path(out.parent)


def synthetic_lightcurve_view(tmp: Path, *, n_obj: int = 20, n_epoch: int = 8,
                              seed: int = 0, name: str = "lc") -> DatasetView:
    """Synthetic OUF LIGHTCURVE dataset (one row per source + per-epoch flux)."""
    from oneuniverse.data._converter_lightcurve import (
        write_ouf_lightcurve_dataset)
    from oneuniverse.data.format_spec import ONEUNIVERSE_SUBDIR
    rng = np.random.default_rng(seed)
    objects = pd.DataFrame({
        "object_id": np.arange(n_obj, dtype=np.int64),
        "ra": rng.uniform(0.0, 360.0, n_obj),
        "dec": rng.uniform(-60.0, 60.0, n_obj),
        "z": rng.uniform(0.01, 0.5, n_obj),
        "z_type": ["spec"] * n_obj,
        "z_err": rng.uniform(1e-4, 1e-3, n_obj),
    })
    rows = []
    for oid in objects["object_id"]:
        for t in np.sort(rng.uniform(58000.0, 60000.0, n_epoch)):
            rows.append({"object_id": int(oid), "mjd": float(t),
                         "filter": rng.choice(["g", "r", "i"]),
                         "flux": float(rng.normal(100.0, 5.0)),
                         "flux_err": 1.0, "flag": 0})
    survey_dir = tmp / name
    write_ouf_lightcurve_dataset(
        objects=objects, epochs=pd.DataFrame(rows), survey_path=survey_dir,
        survey_name=name, survey_type="transient", loader_name="syn",
        loader_version="0")
    return DatasetView.from_ou_dir(survey_dir / ONEUNIVERSE_SUBDIR)


def synthetic_pv_view(tmp: Path, *, n: int = 2000, seed: int = 0,
                      name: str = "pv") -> DatasetView:
    """Synthetic peculiar-velocity OUF POINT dataset (distance indicators)."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, n)
    dec = rng.uniform(-60.0, 60.0, n)
    z = np.clip(rng.uniform(0.0, 0.08, n), 1e-3, 0.1)
    df = pd.DataFrame({
        "ra": ra, "dec": dec, "z": z.astype(np.float32),
        "z_type": np.full(n, "pv"), "z_err": np.full(n, 1e-4, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.zeros(n, dtype=np.int64),
        "mu": (35.0 + 5.0 * np.log10(z / 0.02)).astype(np.float32),
        "mu_err": rng.uniform(0.1, 0.3, n).astype(np.float32),
        "eta": rng.normal(0.0, 0.05, n).astype(np.float32),
        "v_pec": rng.normal(0.0, 300.0, n).astype(np.float32),
        "sigma_v": np.full(n, 250.0, dtype=np.float32),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })
    out = tmp / name / "oneuniverse"
    write_ouf_dataset(df=df, out_dir=out, survey_name=name,
                      survey_type="peculiar_velocity", geometry=DataGeometry.POINT,
                      loader=LoaderSpec(name=name, version="0"))
    return DatasetView.from_path(out.parent)


def synthetic_sn_view(tmp: Path, *, n: int = 200, seed: int = 0,
                      name: str = "sn"):
    """Synthetic SN Ia OUF POINT dataset (z + mu + mu_err). Returns (view, n)."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, n)
    dec = rng.uniform(-60.0, 60.0, n)
    z = np.clip(rng.uniform(0.01, 1.0, n), 0.01, 1.2)
    df = pd.DataFrame({
        "ra": ra, "dec": dec, "z": z.astype(np.float32),
        "z_type": np.full(n, "spec"), "z_err": np.full(n, 1e-3, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.zeros(n, dtype=np.int64),
        "mu": (5.0 * np.log10(z) + 43.0).astype(np.float32),
        "mu_err": rng.uniform(0.1, 0.2, n).astype(np.float32),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })
    out = tmp / name / "oneuniverse"
    write_ouf_dataset(df=df, out_dir=out, survey_name=name,
                      survey_type="supernova", geometry=DataGeometry.POINT,
                      loader=LoaderSpec(name=name, version="0"))
    return DatasetView.from_path(out.parent), n


def synthetic_shear_view(tmp: Path, *, n: int = 3000, seed: int = 0,
                         kind: str = "metacal", with_pdf: bool = False,
                         n_tomo: int = 2, name: str = "src") -> DatasetView:
    """Synthetic weak-lensing source OUF POINT dataset (shapes [+ photo-z])."""
    from oneuniverse.data.pdf import PdfSpec
    rng = np.random.default_rng(seed)
    ra = rng.uniform(150.0, 170.0, n)
    dec = rng.uniform(0.0, 15.0, n)
    z = np.clip(rng.normal(0.7, 0.25, n), 0.05, 2.0)
    df = pd.DataFrame({
        "ra": ra, "dec": dec, "z": z.astype(np.float32),
        "z_type": np.full(n, "phot" if with_pdf else "spec"),
        "z_err": np.full(n, 0.03, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.zeros(n, dtype=np.int64),
        "e1": rng.normal(0.0, 0.28, n).astype(np.float32),
        "e2": rng.normal(0.0, 0.28, n).astype(np.float32),
        "e1_err": np.full(n, 0.25, dtype=np.float32),
        "e2_err": np.full(n, 0.25, dtype=np.float32),
        "R11": rng.uniform(0.6, 0.8, n).astype(np.float32),
        "R22": rng.uniform(0.6, 0.8, n).astype(np.float32),
        "R_S": np.zeros(n, dtype=np.float32),
        "m_bias": rng.uniform(-0.02, 0.02, n).astype(np.float32),
        "c1_bias": np.zeros(n, dtype=np.float32),
        "c2_bias": np.zeros(n, dtype=np.float32),
        "shear_weight": rng.uniform(0.5, 1.0, n).astype(np.float32),
        "tomo_bin": (np.floor(rng.uniform(0, n_tomo, n))).astype(np.int64),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })
    pdf_spec = None
    if with_pdf:
        grid = np.linspace(0.0, 2.0, 81, dtype=np.float32)
        dz = grid[1] - grid[0]
        mu = z.astype(np.float32)
        sig = df["z_err"].to_numpy()
        pdfs = np.exp(-0.5 * ((grid[None, :] - mu[:, None]) / sig[:, None]) ** 2)
        pdfs = (pdfs / (pdfs.sum(axis=1, keepdims=True) * dz)).astype(np.float32)
        df["z_pdf_values"] = [row for row in pdfs]
        pdf_spec = PdfSpec(parameterisation="interp", n_components=len(grid),
                           grid=list(map(float, grid)), grid_kind="z")
    out = tmp / name / "oneuniverse"
    write_ouf_dataset(df=df, out_dir=out, survey_name=name,
                      survey_type="photometric", geometry=DataGeometry.POINT,
                      loader=LoaderSpec(name=name, version="0"),
                      pdf_spec=pdf_spec)
    return DatasetView.from_path(out.parent)
