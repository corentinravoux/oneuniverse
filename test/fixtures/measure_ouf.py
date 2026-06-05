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
