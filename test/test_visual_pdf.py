"""Diagnostic figure for the photo-z PDF onboarding pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.pdf_catalog import (  # noqa: E402
    make_gaussian_pdf_catalog, materialise_core_cols,
)

from oneuniverse.data.converter import write_ouf_dataset  # noqa: E402
from oneuniverse.data.dataset_view import DatasetView  # noqa: E402
from oneuniverse.data.format_spec import DataGeometry  # noqa: E402
from oneuniverse.data.manifest import LoaderSpec  # noqa: E402
from oneuniverse.data.pdf import PdfSpec  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_phase10_visual_end_to_end(tmp_path):
    df, grid = make_gaussian_pdf_catalog(n_rows=300, n_grid=201, seed=5)
    spec = PdfSpec(
        parameterisation="interp", n_components=len(grid),
        grid=list(grid), grid_kind="z",
    )
    df = materialise_core_cols(df)

    out_dir = tmp_path / "pdf_viz" / "oneuniverse"
    out_dir.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out_dir,
        survey_name="pdf_viz", survey_type="photometric",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="pdf_viz", version="0"),
        pdf_spec=spec,
    )

    view = DatasetView.from_path(out_dir.parent)
    pz = view.load_pdf()
    df_read = view.read()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(5):
        ax[0].plot(pz.grid, pz.values[i], alpha=0.7, lw=1)
    ax[0].set_xlabel("z"); ax[0].set_ylabel("p(z)")
    ax[0].set_title("5 example photo-z PDFs")

    cdf = pz.cdf()
    ax[1].plot(pz.grid, cdf[:20].T, color="tab:blue", alpha=0.3, lw=0.6)
    ax[1].set_xlabel("z"); ax[1].set_ylabel("CDF(z)")
    ax[1].set_title("CDFs (first 20)")

    ax[2].scatter(df_read["z_true"], pz.mean(), s=8, alpha=0.6)
    zmin = min(df_read["z_true"].min(), pz.mean().min())
    zmax = max(df_read["z_true"].max(), pz.mean().max())
    ax[2].plot([zmin, zmax], [zmin, zmax], "k--", lw=0.8)
    ax[2].set_xlabel("z_true"); ax[2].set_ylabel("<z> from PDF")
    ax[2].set_title("PDF mean vs input truth")
    fig.tight_layout()
    out_png = OUT / "phase10_pdf_overview.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 10_000
