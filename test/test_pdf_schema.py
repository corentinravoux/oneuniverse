"""Schema-level tests for the probabilistic-redshift column group."""
from __future__ import annotations

import pytest

from oneuniverse.data.pdf import PdfParameterisation, PdfSpec


def test_parameterisation_values():
    assert {p.value for p in PdfParameterisation} == {"interp", "quant", "mixmod"}


def test_pdfspec_interp_basic():
    spec = PdfSpec(
        parameterisation="interp",
        n_components=41,
        grid=[0.0, 0.05, 0.10],
        grid_kind="z",
    )
    assert spec.parameterisation == "interp"
    assert spec.n_components == 41
    assert spec.grid == [0.0, 0.05, 0.10]


def test_pdfspec_rejects_unknown_parameterisation():
    with pytest.raises(ValueError, match="unknown PDF parameterisation"):
        PdfSpec(parameterisation="zzz", n_components=10, grid=None, grid_kind="z")


def test_pdfspec_interp_requires_nonempty_grid():
    with pytest.raises(ValueError, match="grid"):
        PdfSpec(parameterisation="interp", n_components=10, grid=None, grid_kind="z")


def test_pdfspec_quant_requires_levels():
    spec = PdfSpec(
        parameterisation="quant",
        n_components=21,
        grid=None,
        grid_kind="quantile",
        quant_levels=[0.0, 0.05, 0.10, 0.95, 1.0],
    )
    assert spec.quant_levels[0] == 0.0


def test_pdfspec_roundtrip_dict():
    spec = PdfSpec(
        parameterisation="interp", n_components=3, grid=[0.0, 0.5, 1.0],
        grid_kind="z",
    )
    d = spec.to_dict()
    assert PdfSpec.from_dict(d) == spec


def test_probabilistic_redshift_group_registered():
    from oneuniverse.data import schema
    assert "probabilistic_redshift" in schema.COLUMN_GROUPS


def test_probabilistic_redshift_required_columns():
    from oneuniverse.data import schema
    req = set(schema.get_required_columns(["probabilistic_redshift"]))
    assert {"z_pdf_kind", "z_pdf_values"} <= req


def test_probabilistic_redshift_scalar_summary_columns_optional():
    from oneuniverse.data import schema
    cols = schema.get_all_columns(["probabilistic_redshift"])
    assert cols["z_pdf_mean"].required is False
    assert cols["z_pdf_std"].required is False
