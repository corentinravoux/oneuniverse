"""ProbabilisticRedshift reader correctness on Gaussian-PDF fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.pdf_catalog import (  # noqa: E402
    make_gaussian_pdf_catalog, materialise_core_cols,
)

from oneuniverse.data.pdf import PdfSpec, ProbabilisticRedshift  # noqa: E402


def _reader():
    df, grid = make_gaussian_pdf_catalog(n_rows=64, n_grid=401, seed=42)
    spec = PdfSpec(
        parameterisation="interp",
        n_components=len(grid),
        grid=list(grid),
        grid_kind="z",
    )
    pz = ProbabilisticRedshift.from_dataframe(df, spec)
    return df, grid, pz


def test_reader_reports_length_and_grid():
    df, grid, pz = _reader()
    assert len(pz) == len(df)
    np.testing.assert_allclose(pz.grid, grid)


def test_reader_mean_matches_input_mu():
    df, _, pz = _reader()
    np.testing.assert_allclose(
        pz.mean(), df["z_pdf_mean"].to_numpy(), atol=1e-2,
    )


def test_reader_std_matches_input_sigma():
    df, _, pz = _reader()
    np.testing.assert_allclose(
        pz.std(), df["z_pdf_std"].to_numpy(), rtol=2e-1,
    )


def test_reader_cdf_monotone():
    _, _, pz = _reader()
    cdf = pz.cdf()
    diffs = np.diff(cdf, axis=1)
    assert (diffs >= -1e-6).all()


def test_reader_ppf_inverts_cdf_at_median():
    _, _, pz = _reader()
    z05 = pz.ppf(0.5)
    np.testing.assert_allclose(z05, pz.mean(), atol=2e-2)


def test_reader_sample_covers_pdf():
    _, _, pz = _reader()
    samples = pz.sample(n_per=500, seed=0)
    assert samples.shape == (len(pz), 500)
    emp = samples.mean(axis=1)
    # 5-sigma per-row band: with 64 rows, fluke rate ~ 6e-7 each, ~4e-5 overall.
    tol = 5.0 * pz.std() / np.sqrt(500)
    assert (np.abs(emp - pz.mean()) < tol).all()


def test_datasetview_load_pdf(tmp_path):
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.dataset_view import DatasetView
    from oneuniverse.data.format_spec import DataGeometry
    from oneuniverse.data.manifest import LoaderSpec

    df, grid = make_gaussian_pdf_catalog(n_rows=32, n_grid=51, seed=3)
    spec = PdfSpec(
        parameterisation="interp", n_components=len(grid),
        grid=list(grid), grid_kind="z",
    )
    df = materialise_core_cols(df)
    out_dir = tmp_path / "pdf2" / "oneuniverse"
    out_dir.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out_dir,
        survey_name="pdf2", survey_type="photometric",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="pdf2", version="0"),
        pdf_spec=spec,
    )

    view = DatasetView.from_path(out_dir.parent)
    pz = view.load_pdf()
    assert len(pz) == 32
    assert pz.spec == spec
