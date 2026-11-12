import numpy as np
import pytest
from fixtures.synthetic import write_synthetic_dataset

from oneuniverse._core.errors import FormatError, ValidationError
from oneuniverse.data import DatasetView
from oneuniverse.data.pdf import PdfSpec
from oneuniverse.measure.photoz import attach_photoz, tomographic_nz
from oneuniverse.measure.pipeline import build_pointset


@pytest.fixture(scope="module")
def pz_view(tmp_path_factory):
    p = tmp_path_factory.mktemp("pz") / "ds"
    spec = PdfSpec("interp", grid=tuple(np.linspace(0, 3, 31)))
    write_synthetic_dataset(p, 5_000, 0, with_pdf=spec)
    return DatasetView.from_path(p)


@pytest.fixture(scope="module")
def prod(pz_view):
    return build_pointset(pz_view, z=(0.5, 1.5), nz_edges=np.linspace(0.5, 1.5, 6),
                          quality=("quality", ">=", 1), randoms="none")


def test_attach_photoz_aligns_rows_after_cuts(pz_view, prod):
    p2 = attach_photoz(prod, pz_view, n_draws=3, seed=0)
    full = pz_view.table(columns=["source_row", "z_pdf"])
    mean_by_row = dict(zip(full["source_row"].to_pylist(), pz_view.load_pdf(full).mean()))
    assert prod.n < 5_000
    assert p2.extras["z_draws"].shape == (prod.n, 3)
    assert p2.extras["z_draws"].dtype == np.float64
    assert np.allclose(p2.extras["z_pdf_mean"], [mean_by_row[r] for r in prod.catalog["source_row"].to_pylist()])
    assert p2.extras["z_pdf_std"].shape == (prod.n,)


def test_attach_photoz_deterministic(pz_view, prod):
    a = attach_photoz(prod, pz_view, n_draws=2, seed=4)
    b = attach_photoz(prod, pz_view, n_draws=2, seed=4)
    c = attach_photoz(prod, pz_view, n_draws=2, seed=5)
    assert np.array_equal(a.extras["z_draws"], b.extras["z_draws"])
    assert not np.array_equal(a.extras["z_draws"], c.extras["z_draws"])


def test_attach_photoz_version_mismatch(pz_view, prod, tmp_path):
    write_synthetic_dataset(tmp_path / "other", 5_000, 0,
                            with_pdf=PdfSpec("interp", grid=tuple(np.linspace(0, 3, 31))))
    other = DatasetView.from_path(tmp_path / "other")
    with pytest.raises(FormatError, match="another version"):
        attach_photoz(prod, other)


def test_tomographic_nz_shapes(pz_view, prod):
    p2 = attach_photoz(prod, pz_view, n_draws=3, seed=0)
    p3, nz = tomographic_nz(p2, bin_edges=[0.5, 1.0, 1.5], nz_edges=np.linspace(0, 3, 31))
    assert nz.counts.shape == (2, 30) and nz.method == "photoz_stack" and nz.bins == (0, 1)
    assert set(np.unique(p3.extras["tomo_bin"])) <= {-1, 0, 1}
    assert p3.extras["tomo_bin"].dtype == np.int32 and p3.nz is nz
    mean = p2.extras["z_pdf_mean"]
    n0 = int(((mean >= 0.5) & (mean < 1.0)).sum())
    assert nz.counts[0].sum() <= n0 + 1e-9 and nz.counts[0].sum() > 0.5 * n0


def test_tomographic_nz_requires_draws(prod):
    with pytest.raises(ValidationError, match="z_draws"):
        tomographic_nz(prod, bin_edges=[0.5, 1.5], nz_edges=np.linspace(0, 3, 31))
