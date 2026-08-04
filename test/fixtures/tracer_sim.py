"""Dummy tracer catalog: Poisson-sample galaxies from a known linear truth field
and write them as an OUF POINT dataset with box positions (x, y, z_box). Returns
the (DatasetView, truth_delta) so tests can check recovery against ground truth.

Uses only the package's dummy simulator (simulation.linear) + the OUF writer —
no real data. Positions are box coordinates; trivial ra/dec/z are filled to
satisfy the CORE schema (they are unused by observe_from_view here).
"""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field


def _cosmo() -> CosmologySpec:
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def synthetic_tracer_view(tmp: Path, *, box_size: float, n_grid: int,
                          nbar: float, bias: float, seed: int = 0,
                          name: str = "tracers"):
    """Return (DatasetView, truth_delta). Galaxies Poisson-sampled from the
    linear field with intensity lambda = nbar_cell * max(0, 1 + b*delta)."""
    truth = generate_density_field(_cosmo(), box_size=box_size, n_grid=n_grid,
                                   z=0.0, seed=seed)
    rng = np.random.default_rng(seed + 1)
    cell = box_size / n_grid
    lam = nbar * cell ** 3 * np.clip(1.0 + bias * truth, 0.0, None)
    counts = rng.poisson(lam)  # (n,n,n)
    cells = np.argwhere(counts > 0)
    reps = counts[counts > 0]
    base = np.repeat(cells, reps, axis=0).astype(float)      # integer cell idx
    jitter = rng.random(base.shape)                          # uniform in-cell
    xyz = (base + jitter) * cell                             # box positions
    ngal = len(xyz)

    # trivial sky coords (unused downstream) just to satisfy the CORE schema
    ra = rng.uniform(0, 360, ngal)
    dec = rng.uniform(-60, 60, ngal)
    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "z": np.full(ngal, 0.1, np.float32),
        "z_type": np.full(ngal, "spec"),
        "z_err": np.full(ngal, 1e-4, np.float32),
        "galaxy_id": np.arange(ngal, dtype=np.int64),
        "survey_id": np.zeros(ngal, dtype=np.int64),
        "x": xyz[:, 0].astype(np.float32),
        "y": xyz[:, 1].astype(np.float32),
        "z_box": xyz[:, 2].astype(np.float32),  # 'z' is redshift; box-z is z_box
        "_original_row_index": np.arange(ngal, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })
    out = tmp / name / "oneuniverse"
    write_ouf_dataset(df=df, out_dir=out, survey_name=name,
                      survey_type="spectroscopic", geometry=DataGeometry.POINT,
                      loader=LoaderSpec(name=name, version="0"))
    return DatasetView.from_path(out.parent), truth
