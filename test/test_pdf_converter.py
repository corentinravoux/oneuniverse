"""End-to-end: write_ouf_dataset with pdf_spec → FixedSizeList parquet → reader."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.pdf_catalog import (  # noqa: E402
    make_gaussian_pdf_catalog, materialise_core_cols,
)

from oneuniverse.data.converter import write_ouf_dataset  # noqa: E402
from oneuniverse.data.dataset_view import DatasetView  # noqa: E402
from oneuniverse.data.format_spec import DataGeometry  # noqa: E402
from oneuniverse.data.manifest import LoaderSpec  # noqa: E402
from oneuniverse.data.pdf import PdfSpec, ProbabilisticRedshift  # noqa: E402


def test_converter_writes_fixed_size_list_for_pdf(tmp_path):
    df, grid = make_gaussian_pdf_catalog(n_rows=200, n_grid=101, seed=2)
    spec = PdfSpec(
        parameterisation="interp", n_components=len(grid),
        grid=list(grid), grid_kind="z",
    )
    df = materialise_core_cols(df)

    out_dir = tmp_path / "pdf_fake" / "oneuniverse"
    out_dir.mkdir(parents=True)
    manifest = write_ouf_dataset(
        df=df,
        out_dir=out_dir,
        survey_name="pdf_fake",
        survey_type="photometric",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="pdf_fake", version="0"),
        pdf_spec=spec,
    )

    assert manifest.pdf_spec == spec

    parquet_files = list(out_dir.rglob("*.parquet"))
    assert parquet_files
    pa_schema = pq.read_schema(parquet_files[0])
    field = pa_schema.field("z_pdf_values")
    assert "fixed_size_list" in str(field.type), str(field.type)

    view = DatasetView.from_path(out_dir.parent)
    df_read = view.read()
    assert len(df_read) == 200
    pz = ProbabilisticRedshift.from_dataframe(df_read, manifest.pdf_spec)
    np.testing.assert_allclose(
        pz.mean(), df_read["z_pdf_mean"].to_numpy(), atol=1e-2,
    )
