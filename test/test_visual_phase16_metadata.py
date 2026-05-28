"""Phase 16 T9 — diagnostic figure showing observational metadata in a
written manifest. Per the visual-testing convention: produce a PNG that
makes the new metadata visible at a glance.
"""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.data.converter import write_ouf_dataset  # noqa: E402
from oneuniverse.data.coordinate_spec import CoordinateSpec  # noqa: E402
from oneuniverse.data.format_spec import DataGeometry  # noqa: E402
from oneuniverse.data.manifest import LoaderSpec, read_manifest  # noqa: E402
from oneuniverse.data.spectrum_spec import SpectrumSpec  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_phase16_visual(tmp_path):
    n = 200
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 360, n).astype("f8")
    dec = rng.uniform(-30, 30, n).astype("f8")
    df = pd.DataFrame({
        "ra": ra,
        "dec": dec,
        "z": rng.uniform(0.1, 1.0, n).astype("f4"),
        "z_type": rng.choice(["spec", "phot"], size=n).astype(object),
        "z_err": np.full(n, 0.01, dtype="f4"),
        "galaxy_id": np.arange(n, dtype="i8"),
        "survey_id": np.array(["phase16_viz"] * n, dtype=object),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })

    out = tmp_path / "phase16_viz" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="phase16_viz", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="phase16_viz", version="0"),
        coordinate=CoordinateSpec(
            frame="icrs", epoch=2016.0, proper_motion_available=True,
        ),
        spectrum=SpectrumSpec(
            wavelength_convention="vacuum", log_binned=True,
        ),
    )
    m = read_manifest(out / "manifest.json")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sc = ax[0].scatter(df["ra"], df["dec"], c=df["z"], s=4, alpha=0.7)
    plt.colorbar(sc, ax=ax[0], label="z")
    ax[0].set_xlabel("RA [deg]")
    ax[0].set_ylabel("Dec [deg]")
    ax[0].set_title(
        f"frame={m.coordinate.frame}  epoch={m.coordinate.epoch}\n"
        f"PM available: {m.coordinate.proper_motion_available}"
    )

    labels = sorted(set(df["z_type"]))
    counts = [int((df["z_type"] == lbl).sum()) for lbl in labels]
    ax[1].bar(labels, counts)
    ax[1].set_ylabel("rows")
    ax[1].set_title(
        f"observed_z_types = {tuple(sorted(m.observed_z_types))}\n"
        f"spectrum: {m.spectrum.wavelength_convention} / "
        f"log_binned={m.spectrum.log_binned}"
    )

    fig.tight_layout()
    out_png = OUT / "phase16_observational_metadata.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 30_000
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(out_png) as im:
        assert im.width >= 800 and im.height >= 200
