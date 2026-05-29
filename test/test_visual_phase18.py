"""Phase 18 visual diagnostic — hist PDFs + mean-vs-truth + tomographic n(z)."""
from __future__ import annotations

from dataclasses import replace
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
from oneuniverse.data.manifest import LoaderSpec, read_manifest  # noqa: E402
from oneuniverse.data.pdf import PdfSpec, ProbabilisticRedshift  # noqa: E402
from oneuniverse.data.tomographic_nz import TomographicNzSpec  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_phase18_visual(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    ra = rng.uniform(0, 360, n).astype("f8")
    dec = rng.uniform(-30, 30, n).astype("f8")
    edges = np.linspace(0.0, 1.0, 6)
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    z_true = rng.uniform(0.05, 0.95, n).astype("f4")
    hist_rows = []
    for z in z_true:
        h = np.exp(-0.5 * ((bin_centres - z) / 0.08) ** 2)
        h = h / h.sum()
        hist_rows.append(h.astype("f4"))
    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "z": z_true,
        "z_type": np.array(["phot_pdf"] * n, dtype=object),
        "z_err": np.full(n, 0.05, dtype="f4"),
        "galaxy_id": np.arange(n, dtype="i8"),
        "survey_id": np.array(["phase18"] * n, dtype=object),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
        "z_pdf_values": hist_rows,
        "tomo_bin": np.digitize(z_true, edges) - 1,
    })
    spec = PdfSpec(
        parameterisation="hist", n_components=5,
        grid=None, grid_kind="z",
        hist_edges=list(map(float, edges)),
    )
    nbins = len(edges) - 1
    z_grid = np.linspace(0.0, 1.0, 51)
    tnz_values = np.zeros((nbins, z_grid.size))
    for b in range(nbins):
        sel = df["tomo_bin"] == b
        if sel.any():
            mean = float(z_true[sel].mean())
            tnz_values[b] = np.exp(-0.5 * ((z_grid - mean) / 0.1) ** 2)
            tnz_values[b] /= tnz_values[b].sum()
    tomo_spec = TomographicNzSpec(
        bin_edges=[(float(edges[b]), float(edges[b + 1])) for b in range(nbins)],
        grid=list(map(float, z_grid)),
        values=[list(map(float, tnz_values[b])) for b in range(nbins)],
    )

    out = tmp_path / "phase18_viz" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="phase18", survey_type="photometric",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="phase18_viz", version="0"),
        pdf_spec=spec,
    )
    m = read_manifest(out / "manifest.json")
    # Attach the tomo spec post-hoc for the visual (real loaders pass
    # it via convert_survey; for this diagnostic we patch it here).
    m = replace(m, tomographic_nz=tomo_spec)

    view = DatasetView.from_path(out.parent)
    df_read = view.read()
    pz = ProbabilisticRedshift.from_dataframe(df_read, spec)
    pdf_mean = pz.mean()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    for i in range(5):
        ax[0].step(0.5 * (edges[:-1] + edges[1:]),
                   df_read["z_pdf_values"].iloc[i],
                   where="mid", lw=0.8)
    ax[0].set_xlabel("z")
    ax[0].set_ylabel("bin height")
    ax[0].set_title("5 example hist PDFs")

    ax[1].scatter(z_true, pdf_mean, s=8, alpha=0.6)
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[1].set_xlabel("z_true")
    ax[1].set_ylabel("<z> from hist PDF")
    ax[1].set_title("hist PDF mean vs truth")

    for b in range(nbins):
        ax[2].plot(z_grid, tnz_values[b], lw=1.0,
                   label=f"bin {b} ({edges[b]:.2f}-{edges[b + 1]:.2f})")
    ax[2].set_xlabel("z")
    ax[2].set_ylabel("n(z)")
    ax[2].legend(fontsize=7)
    ax[2].set_title("tomographic n(z) per bin")

    fig.tight_layout()
    out_png = OUT / "phase18_pdf_polymorphism_and_tomographic_nz.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 30_000
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(out_png) as im:
        assert im.width >= 800 and im.height >= 200
