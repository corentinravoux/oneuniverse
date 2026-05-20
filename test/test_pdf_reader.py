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


def test_quant_reader_recovers_moments():
    pytest = __import__("pytest")
    sp = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    n = 10
    mu = rng.uniform(0.2, 1.8, size=n)
    sigma = rng.uniform(0.02, 0.08, size=n)
    levels = np.linspace(0.005, 0.995, 81)
    qvals = np.stack([sp.norm.ppf(levels, loc=mu[i], scale=sigma[i]) for i in range(n)])
    spec = PdfSpec(
        parameterisation="quant", n_components=len(levels),
        grid=None, grid_kind="quantile",
        quant_levels=list(levels.astype(float)),
    )
    pz = ProbabilisticRedshift(spec, values=qvals, grid=None)
    np.testing.assert_allclose(pz.mean(), mu, atol=5e-2)
    np.testing.assert_allclose(pz.std(), sigma, rtol=2e-1)


def test_mixmod_reader_recovers_moments():
    rng = np.random.default_rng(1)
    n, K = 8, 3
    mu = rng.uniform(0.1, 1.9, size=(n, K)).astype(np.float64)
    sigma = rng.uniform(0.03, 0.1, size=(n, K)).astype(np.float64)
    w = rng.dirichlet(np.ones(K), size=n)
    spec = PdfSpec(
        parameterisation="mixmod", n_components=K, grid=None,
        grid_kind="component",
    )
    pz = ProbabilisticRedshift.from_mixmod(spec, mu, sigma, w)
    expected_mean = (w * mu).sum(axis=1)
    np.testing.assert_allclose(pz.mean(), expected_mean, rtol=1e-6)
    samples = pz.sample(n_per=2000, seed=0)
    assert samples.shape == (n, 2000)
