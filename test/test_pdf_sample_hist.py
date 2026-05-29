"""Phase 18 T1/T2 — PdfSpec covers sample + hist parameterisations."""
import healpy as hp
import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec
from oneuniverse.data.pdf import PdfParameterisation, PdfSpec


def test_enum_has_sample_and_hist():
    assert PdfParameterisation.SAMPLE.value == "sample"
    assert PdfParameterisation.HIST.value == "hist"


def test_sample_spec_does_not_require_grid():
    spec = PdfSpec(
        parameterisation="sample", n_components=100,
        grid=None, grid_kind="z",
    )
    assert spec.parameterisation == "sample"
    assert spec.n_components == 100


def test_hist_spec_requires_edges():
    with pytest.raises(ValueError, match="hist"):
        PdfSpec(
            parameterisation="hist", n_components=4,
            grid=None, grid_kind="z",
        )


def test_hist_spec_with_edges_roundtrips():
    spec = PdfSpec(
        parameterisation="hist", n_components=4,
        grid=None, grid_kind="z",
        hist_edges=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    d = spec.to_dict()
    restored = PdfSpec.from_dict(d)
    assert restored.hist_edges == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_default_column_aliases_match_phase10():
    spec = PdfSpec(
        parameterisation="interp", n_components=5,
        grid=[0.0, 0.25, 0.5, 0.75, 1.0], grid_kind="z",
    )
    assert spec.value_column == "z_pdf_values"
    assert spec.sigma_column == "z_pdf_sigma"
    assert spec.weights_column == "z_pdf_weights"


def test_custom_column_aliases_roundtrip():
    spec = PdfSpec(
        parameterisation="interp", n_components=5,
        grid=[0.0, 0.25, 0.5, 0.75, 1.0], grid_kind="z",
        value_column="z_post",
    )
    d = spec.to_dict()
    restored = PdfSpec.from_dict(d)
    assert restored.value_column == "z_post"


# ── T2 round-trip via writer ────────────────────────────────────────────


def _base_core(n: int) -> pd.DataFrame:
    ra = np.linspace(0.0, 10.0, n).astype("f8")
    dec = np.linspace(-5.0, 5.0, n).astype("f8")
    return pd.DataFrame({
        "ra": ra, "dec": dec,
        "z": np.full(n, 0.5, dtype="f4"),
        "z_type": np.array(["phot_pdf"] * n, dtype=object),
        "z_err": np.full(n, 0.01, dtype="f4"),
        "galaxy_id": np.arange(n, dtype="i8"),
        "survey_id": np.array(["fix"] * n, dtype=object),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })


def test_sample_pdf_roundtrip(tmp_path):
    n = 3
    df = _base_core(n)
    df["z_pdf_values"] = [
        np.array([0.30, 0.32, 0.28, 0.35], dtype="f4"),
        np.array([0.50, 0.49], dtype="f4"),
        np.array([0.70, 0.71, 0.73], dtype="f4"),
    ]
    spec = PdfSpec(
        parameterisation="sample", n_components=4,
        grid=None, grid_kind="z",
    )
    out = tmp_path / "x" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="x", survey_type="photometric",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="x", version="0"),
        pdf_spec=spec,
    )
    view = DatasetView.from_path(out.parent)
    out_df = view.read()
    lengths = [len(v) for v in out_df["z_pdf_values"]]
    assert sorted(lengths) == [2, 3, 4]


def test_hist_pdf_roundtrip(tmp_path):
    n = 3
    df = _base_core(n)
    df["z_pdf_values"] = [
        np.array([0.1, 0.4, 0.3, 0.2], dtype="f4"),
        np.array([0.25, 0.25, 0.25, 0.25], dtype="f4"),
        np.array([0.5, 0.3, 0.15, 0.05], dtype="f4"),
    ]
    spec = PdfSpec(
        parameterisation="hist", n_components=4,
        grid=None, grid_kind="z",
        hist_edges=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    out = tmp_path / "x" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="x", survey_type="photometric",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="x", version="0"),
        pdf_spec=spec,
    )
    view = DatasetView.from_path(out.parent)
    out_df = view.read()
    arr = np.stack([np.asarray(r) for r in out_df["z_pdf_values"]])
    assert arr.shape == (n, 4)
