import numpy as np
import pyarrow as pa
import pytest
from fixtures.synthetic import synthetic_point_table, write_synthetic_dataset

from oneuniverse._core.errors import FormatError, SchemaError, ValidationError
from oneuniverse.data import DatasetView, PdfSpec, ProbabilisticRedshift
from oneuniverse.data.manifest import LoaderInfo, SurveyMeta
from oneuniverse.data.schema import Geometry
from oneuniverse.data.writer import write_dataset


def _gauss_interp(n=200, ng=61):  # rows with known mu, sigma on grid 0..3
    grid = np.linspace(0, 3, ng); rng = np.random.default_rng(0)
    mu = rng.uniform(0.5, 2.5, n); sig = rng.uniform(0.05, 0.2, n)
    p = np.exp(-0.5 * ((grid[None] - mu[:, None]) / sig[:, None]) ** 2)
    p /= np.trapezoid(p, grid, axis=1)[:, None]
    return PdfSpec("interp", grid=tuple(grid)), p.astype(np.float32), mu, sig


def test_interp_moments_cdf_ppf():
    spec, p, mu, sig = _gauss_interp(); pz = ProbabilisticRedshift(spec, p)
    assert np.allclose(pz.mean(), mu, atol=2e-3) and np.allclose(pz.std(), sig, atol=5e-3)
    assert np.allclose(pz.cdf(mu), 0.5, atol=0.02) and np.allclose(pz.ppf(0.5), mu, atol=0.02)


def test_sample_kind_mean_std_and_unsupported():
    draws = [np.random.default_rng(i).normal(1.0, 0.1, 50).astype(np.float32) for i in range(10)]
    pz = ProbabilisticRedshift(PdfSpec("sample"), draws)
    assert np.allclose(pz.mean(), 1.0, atol=0.06) and np.allclose(pz.std(), 0.1, atol=0.05)
    with pytest.raises(NotImplementedError):
        pz.cdf(1.0)
    with pytest.raises(NotImplementedError):
        pz.ppf(0.5)


def test_quant_ppf_and_sample_seeded():
    levels = (0.16, 0.5, 0.84); vals = np.array([[0.9, 1.0, 1.1], [1.9, 2.0, 2.1]], np.float32)
    pz = ProbabilisticRedshift(PdfSpec("quant", quant_levels=levels), vals)
    assert np.allclose(pz.ppf(0.5), [1.0, 2.0]) and pz.sample(5, seed=1).shape == (2, 5) \
        and np.array_equal(pz.sample(5, seed=1), pz.sample(5, seed=1))
    for op in (lambda: pz.cdf(1.0), pz.mean, pz.std):
        with pytest.raises(NotImplementedError):
            op()


def test_pdf_round_trip_through_dataset(tmp_path):
    spec = PdfSpec("interp", grid=tuple(np.linspace(0, 3, 31)))
    write_synthetic_dataset(tmp_path / "ds", 2_000, 0, with_pdf=spec)
    v = DatasetView.from_path(tmp_path / "ds")
    assert v.manifest.extension("pdf")["kind"] == "interp" \
        and v.schema.field("z_pdf").type == pa.list_(pa.float32(), 31)
    pz = v.load_pdf(); assert len(pz) == 2_000 and pz.mean().shape == (2_000,)
    rows = v.table(z=(0.5, 1.0), columns=["z_pdf"])
    assert len(v.load_pdf(rows)) == rows.num_rows


def test_spec_validation_and_dict_round_trip():
    with pytest.raises(ValidationError):
        PdfSpec("bogus")
    with pytest.raises(ValidationError):
        PdfSpec("interp", grid=(0.0, 1.0, 1.0))
    with pytest.raises(ValidationError):
        PdfSpec("interp")
    with pytest.raises(ValidationError):
        PdfSpec("quant", quant_levels=(0.0, 0.5))
    with pytest.raises(ValidationError):
        PdfSpec("quant", quant_levels=(0.5, 0.4))
    s = PdfSpec("quant", quant_levels=(0.1, 0.9))
    assert PdfSpec.from_dict(s.to_dict()) == s and s.n_components == 2
    assert PdfSpec("sample").n_components is None


def test_interp_sample_follows_pdf_and_is_float64():
    spec, p, mu, sig = _gauss_interp(50)
    pz = ProbabilisticRedshift(spec, p)
    s = pz.sample(2000, seed=3)
    assert s.dtype == np.float64 and s.shape == (50, 2000)
    assert np.allclose(s.mean(1), mu, atol=0.02) and np.allclose(s.std(1), sig, atol=0.02)


def test_sample_kind_ragged_sample_and_from_table():
    rows = [np.array([1.0, 2.0, 3.0]), np.array([5.0]), np.array([], np.float32)]
    tab = pa.table({"z_pdf": pa.array(rows, pa.list_(pa.float32()))})
    pz = ProbabilisticRedshift.from_table(tab.slice(0), PdfSpec("sample"))
    s = pz.sample(4, seed=0)
    assert np.all(np.isin(s[0], [1, 2, 3])) and np.all(s[1] == 5.0) and np.all(np.isnan(s[2]))
    sub = ProbabilisticRedshift.from_table(tab.slice(1), PdfSpec("sample"))
    assert len(sub) == 2 and sub.mean()[0] == 5.0


def test_quant_round_trip_and_from_table_errors(tmp_path):
    spec = PdfSpec("quant", quant_levels=(0.16, 0.5, 0.84))
    write_synthetic_dataset(tmp_path / "q", 1_000, 1, with_pdf=spec)
    v = DatasetView.from_path(tmp_path / "q")
    assert v.schema.field("z_pdf").type == pa.list_(pa.float32(), 3)
    pz = v.load_pdf(); z = v.table(columns=["z"])["z"].to_numpy()
    assert np.allclose(pz.ppf(0.5), z, atol=1e-5)
    with pytest.raises(SchemaError):
        ProbabilisticRedshift.from_table(pa.table({"x": [1]}), spec)
    bad = pa.table({"z_pdf": pa.array([[1.0, 2.0], [1.0, 2.0, 3.0]], pa.list_(pa.float32()))})
    with pytest.raises(SchemaError):
        ProbabilisticRedshift.from_table(bad, spec)


def test_sample_dataset_round_trip(tmp_path):
    write_synthetic_dataset(tmp_path / "s", 500, 2, with_pdf=PdfSpec("sample"))
    v = DatasetView.from_path(tmp_path / "s")
    z = v.table(columns=["z"])["z"].to_numpy()
    assert np.allclose(v.load_pdf().mean(), z, atol=0.15)


def test_load_pdf_without_extension(tmp_path):
    write_synthetic_dataset(tmp_path / "n", 100, 0)
    with pytest.raises(FormatError):
        DatasetView.from_path(tmp_path / "n").load_pdf()


def test_writer_rejects_bad_pdf_column(tmp_path):
    spec = PdfSpec("interp", grid=(0.0, 1.0, 2.0))
    ext = {"pdf": {"version": 1, **spec.to_dict()}}
    kw = dict(geometry=Geometry.POINT, survey=SurveyMeta("synthetic", "test"),
              loader=LoaderInfo("t", "1", {}), extensions=ext)
    t = synthetic_point_table(10, 0)
    with pytest.raises(SchemaError, match="z_pdf"):
        write_dataset(t, tmp_path / "a", **kw)
    wrong = t.append_column("z_pdf", pa.array([[0.1, 0.2]] * 10, pa.list_(pa.float32())))
    with pytest.raises(SchemaError, match="components"):
        write_dataset(wrong, tmp_path / "b", **kw)
    ok = t.append_column("z_pdf", pa.array([[0.1, 0.2, 0.3]] * 10, pa.list_(pa.float64())))
    m = write_dataset(ok, tmp_path / "c", **kw)
    assert m.schema.field("z_pdf").type == pa.list_(pa.float32(), 3)
