"""Phase 17 visual diagnostic — variable-length payload + extra-range pushdown."""
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
from oneuniverse.data.dataset_view import DatasetView  # noqa: E402
from oneuniverse.data.format_spec import DataGeometry  # noqa: E402
from oneuniverse.data.manifest import LoaderSpec  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_phase17_visual(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    ra = rng.uniform(0, 360, n).astype("f8")
    dec = rng.uniform(-30, 30, n).astype("f8")
    snr = rng.uniform(1.0, 200.0, n).astype("f4")
    deltas = [
        rng.normal(0.0, 0.1, size=int(rng.integers(20, 60))).astype("f4")
        for _ in range(n)
    ]
    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "z": rng.uniform(0.1, 1.0, n).astype("f4"),
        "z_type": np.array(["spec"] * n, dtype=object),
        "z_err": np.full(n, 0.01, dtype="f4"),
        "galaxy_id": np.arange(n, dtype="i8"),
        "survey_id": np.array(["phase17"] * n, dtype=object),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
        "snr": snr,
        "delta": deltas,
    })
    out = tmp_path / "phase17_viz" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="phase17", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="phase17_viz", version="0"),
        column_dtypes={"delta": "list<f4>"},
        extra_stats_columns=["snr"],
    )

    view = DatasetView.from_path(out.parent)
    hi_snr = view.read(extra_filters={"snr": (100.0, None)})

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    lengths = [len(x) for x in df["delta"]]
    ax[0].hist(lengths, bins=20, color="tab:blue", alpha=0.8)
    ax[0].set_xlabel("delta length per row")
    ax[0].set_ylabel("count")
    ax[0].set_title("variable-length `delta` payload")

    ax[1].hist(df["snr"], bins=40, color="tab:gray", alpha=0.6, label="all")
    ax[1].hist(hi_snr["snr"], bins=40, color="tab:red", alpha=0.8,
               label="extra_filters snr >= 100")
    ax[1].set_xlabel("snr")
    ax[1].legend()
    ax[1].set_title("extra-range pushdown")

    ax[2].plot(df["delta"].iloc[0], lw=0.8)
    ax[2].plot(df["delta"].iloc[1], lw=0.8)
    ax[2].plot(df["delta"].iloc[2], lw=0.8)
    ax[2].set_xlabel("pixel")
    ax[2].set_ylabel("delta")
    ax[2].set_title("3 example `delta` series (different lengths)")

    fig.tight_layout()
    out_png = OUT / "phase17_variable_length_and_extra_stats.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 30_000
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(out_png) as im:
        assert im.width >= 800 and im.height >= 200
