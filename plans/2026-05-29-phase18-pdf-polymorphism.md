# Phase 18 — PDF Polymorphism + Tomographic n(z) + Classification PDFs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the PDF subsystem from 3 to 5 parameterisations (add `sample` and `hist` alongside `interp`/`quant`/`mixmod`), give `PdfSpec` configurable column-name aliases + sparse-grid + multi-axis hooks, and add two new manifest-level objects: `TomographicNzSpec` (per-bin n(z) at dataset level) and `ClassificationPdfSpec` (per-row class probabilities). Aligns the on-disk layout with the `qp` / RAIL ecosystem (Malz & Marshall 2018; LSSTDESC qp).

**Architecture:** Three layers, each isolated. (a) PDF extensions reuse the Phase-17 dtype mini-language: `sample` parameterisation stores draws via `list<f4>` per row; `hist` parameterisation stores bin heights as `f4[N]` and bin edges in the manifest. `ProbabilisticRedshift` gains hist/sample paths for moments, CDF, sampling. (b) `TomographicNzSpec` is a dataset-level sidecar carrying bin edges + shared z grid + per-bin values + the name of a row-level `bin_assignment_column` (int). (c) `ClassificationPdfSpec` declares an ordered class label tuple + the per-row `class_pdf_values: f4[n_classes]` column. Manifest bumps OUF 2.3.0 → 2.4.0; 2.0/2.1/2.2/2.3 still parse.

**Tech Stack:** Python 3.9+, pyarrow, pandas, dataclasses, pytest. Same stack as Phases 10–17. No new runtime dependencies.

---

## File Structure

**New files:**
- `oneuniverse/data/tomographic_nz.py` — `TomographicNzSpec` dataclass + (de)serialisation.
- `oneuniverse/data/classification_pdf.py` — `ClassificationPdfSpec` dataclass + (de)serialisation.
- `test/test_pdf_sample_hist.py` — `PdfSpec` + writer + `ProbabilisticRedshift` for `sample` / `hist`.
- `test/test_pdf_column_aliases.py` — non-default `value_column` / `sigma_column` / `weights_column`.
- `test/test_tomographic_nz_spec.py` — `TomographicNzSpec` round-trip.
- `test/test_classification_pdf_spec.py` — `ClassificationPdfSpec` round-trip.
- `test/test_manifest_phase18.py` — Manifest carries both new sub-specs + OUF 2.4 round-trip.
- `test/test_visual_phase18.py` — diagnostic figure.

**Modified files:**
- `oneuniverse/data/pdf.py` — `PdfParameterisation` adds `SAMPLE` + `HIST`; `PdfSpec` gains `value_column`, `sigma_column`, `weights_column`, `hist_edges`, `grid_mask`, `axis_labels`; `ProbabilisticRedshift` handles new parameterisations.
- `oneuniverse/data/converter.py` — `_chunk_to_table` routes `sample` payloads through `list<f4>` and `hist` through `f4[N]`.
- `oneuniverse/data/manifest.py` — bump `FORMAT_VERSION` / `SCHEMA_VERSION` to `2.4.0`; extend version-compat check; add `tomographic_nz` + `classification_pdf` Manifest fields + (de)serialisation.
- `oneuniverse/data/format_spec.py` — bump `FORMAT_VERSION` / `SCHEMA_VERSION` to `2.4.0`.
- `test/test_lightcurve_geometry.py` — bumped version assertion.
- `test/test_manifest_phase16.py` — bumped version assertion.
- `oneuniverse/CLAUDE.md` — note OUF 2.4 + new sub-specs.
- `plans/README.md` — mark Phase 18 complete.
- `research/schema_generalisation_audit.md` — Phase 18 close-out cross-ref.

---

## Pre-flight

- [ ] **Step 0: Confirm baseline.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest -q 2>&1 | tail -3
```

Expected: `428 passed, 1 skipped` (Phase 17 baseline).

---

## Task 1: `PdfParameterisation` + `PdfSpec` field extensions

**Files:**
- Modify: `oneuniverse/data/pdf.py:24-30` (enum + `_KNOWN`)
- Modify: `oneuniverse/data/pdf.py:33-107` (`PdfSpec` dataclass + serialisation)
- Create: `test/test_pdf_sample_hist.py` (PdfSpec smoke tests; round-trip via writer comes in T3)

- [ ] **Step 1: Write the failing tests**

```python
# test/test_pdf_sample_hist.py
"""Phase 18 T1/T3 — PdfSpec covers sample + hist parameterisations."""
import numpy as np
import pandas as pd
import pytest

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_pdf_sample_hist.py -v
```

Expected: `AttributeError: ... 'SAMPLE'` and / or unknown PDF parameterisation.

- [ ] **Step 3: Extend the enum**

In `oneuniverse/data/pdf.py`, replace the enum block (lines 24-30) with:

```python
class PdfParameterisation(str, Enum):
    INTERP = "interp"
    QUANT = "quant"
    MIXMOD = "mixmod"
    SAMPLE = "sample"   # Phase 18 — variable-length per-row z-draws.
    HIST = "hist"       # Phase 18 — per-row bin heights on shared edges.


_KNOWN = {p.value for p in PdfParameterisation}
```

- [ ] **Step 4: Extend `PdfSpec` dataclass and serialisation**

Replace the `PdfSpec` dataclass body (everything from `@dataclass(frozen=True)\nclass PdfSpec:` through the end of `from_dict`) with:

```python
@dataclass(frozen=True)
class PdfSpec:
    """How to reconstruct a probabilistic redshift PDF from on-disk columns.

    Parameters
    ----------
    parameterisation
        One of ``"interp"``, ``"quant"``, ``"mixmod"``, ``"sample"``,
        ``"hist"``.
    n_components
        Fixed length of every PDF array in this dataset:
        grid points for ``interp``, quantile levels for ``quant``,
        mixture components for ``mixmod``, number of samples per row
        for ``sample`` (also the FixedSize cap in case the writer uses
        ``f4[N]`` instead of ``list<f4>``), number of histogram bins
        for ``hist``.
    grid
        For ``interp``: the common z grid (length ``n_components``).
        For ``mixmod``, ``quant``, ``sample``: ignored.
        For ``hist``: ignored — use ``hist_edges`` instead.
    grid_kind
        ``"z"`` for redshift grid, ``"quantile"`` for quantile levels,
        ``"component"`` for mixture indices.
    quant_levels
        For ``quant``: quantile levels in [0, 1] (length ``n_components``).
    hist_edges
        For ``hist``: ``n_components + 1`` bin edges.
    value_column / sigma_column / weights_column
        Column names on disk. Defaults match the Phase 10 contract
        (``z_pdf_values`` / ``z_pdf_sigma`` / ``z_pdf_weights``).
        Override when ingesting native column names (e.g. RAIL/qp).
    grid_mask
        Optional boolean array of length ``n_components`` marking
        valid grid cells for ``interp``. ``None`` means dense.
    axis_labels
        Axis labels for the PDF (``("z",)`` for a redshift-only PDF;
        multi-axis is reserved for future P(z, type)-style products
        but not yet exercised by the reader).
    """

    parameterisation: str
    n_components: int
    grid: Optional[List[float]]
    grid_kind: str
    quant_levels: Optional[List[float]] = None
    hist_edges: Optional[List[float]] = None
    value_column: str = "z_pdf_values"
    sigma_column: str = "z_pdf_sigma"
    weights_column: str = "z_pdf_weights"
    grid_mask: Optional[List[bool]] = None
    axis_labels: tuple = ("z",)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.parameterisation not in _KNOWN:
            raise ValueError(
                f"unknown PDF parameterisation {self.parameterisation!r}; "
                f"allowed: {sorted(_KNOWN)}"
            )
        if self.n_components <= 0:
            raise ValueError("n_components must be > 0")
        if self.parameterisation == "interp" and not self.grid:
            raise ValueError("interp parameterisation requires a non-empty grid")
        if self.parameterisation == "quant" and not self.quant_levels:
            raise ValueError("quant parameterisation requires quant_levels")
        if self.parameterisation == "hist" and not self.hist_edges:
            raise ValueError(
                "hist parameterisation requires hist_edges of length "
                "n_components+1"
            )
        if (
            self.parameterisation == "hist"
            and self.hist_edges is not None
            and len(self.hist_edges) != self.n_components + 1
        ):
            raise ValueError(
                f"hist_edges length {len(self.hist_edges)} "
                f"must be n_components+1 ({self.n_components + 1})"
            )
        # Normalise sequences to plain Python floats so JSON round-trips.
        if self.grid is not None:
            object.__setattr__(self, "grid", [float(x) for x in self.grid])
        if self.quant_levels is not None:
            object.__setattr__(
                self, "quant_levels", [float(x) for x in self.quant_levels],
            )
        if self.hist_edges is not None:
            object.__setattr__(
                self, "hist_edges", [float(x) for x in self.hist_edges],
            )
        if self.grid_mask is not None:
            object.__setattr__(
                self, "grid_mask", [bool(x) for x in self.grid_mask],
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameterisation": self.parameterisation,
            "n_components": int(self.n_components),
            "grid": [float(x) for x in self.grid] if self.grid is not None else None,
            "grid_kind": self.grid_kind,
            "quant_levels": (
                [float(x) for x in self.quant_levels]
                if self.quant_levels is not None else None
            ),
            "hist_edges": (
                [float(x) for x in self.hist_edges]
                if self.hist_edges is not None else None
            ),
            "value_column": self.value_column,
            "sigma_column": self.sigma_column,
            "weights_column": self.weights_column,
            "grid_mask": (
                [bool(x) for x in self.grid_mask]
                if self.grid_mask is not None else None
            ),
            "axis_labels": list(self.axis_labels),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PdfSpec":
        return cls(
            parameterisation=d["parameterisation"],
            n_components=int(d["n_components"]),
            grid=list(d["grid"]) if d.get("grid") is not None else None,
            grid_kind=d["grid_kind"],
            quant_levels=(
                list(d["quant_levels"])
                if d.get("quant_levels") is not None else None
            ),
            hist_edges=(
                list(d["hist_edges"])
                if d.get("hist_edges") is not None else None
            ),
            value_column=d.get("value_column", "z_pdf_values"),
            sigma_column=d.get("sigma_column", "z_pdf_sigma"),
            weights_column=d.get("weights_column", "z_pdf_weights"),
            grid_mask=(
                list(d["grid_mask"])
                if d.get("grid_mask") is not None else None
            ),
            axis_labels=tuple(d.get("axis_labels", ("z",))),
            extra=dict(d.get("extra", {})),
        )
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest test/test_pdf_sample_hist.py test/test_pdf_manifest.py test/test_pdf_reader.py test/test_pdf_converter.py test/test_pdf_schema.py -q
```

Expected: green; pre-Phase-18 PDF tests must remain green.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/pdf.py test/test_pdf_sample_hist.py
git commit -m "phase18/T1: PdfSpec adds sample/hist parameterisations + column aliases + grid_mask + axis_labels"
```

---

## Task 2: Writer routes `sample` (variable-length) + `hist` (fixed-size) payloads

**Files:**
- Modify: `oneuniverse/data/converter.py` (`_chunk_to_table` PDF-routing block)
- Extend: `test/test_pdf_sample_hist.py`

- [ ] **Step 1: Extend the test file**

Append to `test/test_pdf_sample_hist.py`:

```python
import healpy as hp

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec


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
```

- [ ] **Step 2: Run the test (writer-side failure)**

```bash
pytest test/test_pdf_sample_hist.py -v
```

Expected: failure with `expected 4 components, got 2` or similar — current writer always treats `z_pdf_values` as `FixedSizeList[f4, n_components]`.

- [ ] **Step 3: Teach `_chunk_to_table` about sample/hist**

In `oneuniverse/data/converter.py`, locate the PDF-aware block inside
`_chunk_to_table` (the part that fills `column_dtypes` from `pdf_spec`)
and replace it with:

```python
    if pdf_spec is not None:
        n = int(pdf_spec.n_components)
        pdf_param = pdf_spec.parameterisation
        if pdf_param == "sample":
            # Per-row variable-length z draws.
            if pdf_spec.value_column in chunk.columns:
                column_dtypes.setdefault(pdf_spec.value_column, "list<f4>")
        elif pdf_param == "hist":
            # Per-row bin heights on the shared hist_edges.
            if pdf_spec.value_column in chunk.columns:
                column_dtypes.setdefault(pdf_spec.value_column, f"f4[{n}]")
        else:
            # interp / quant / mixmod — historical FixedSize storage.
            pdf_cols = [pdf_spec.value_column]
            if pdf_param == "mixmod":
                pdf_cols += [pdf_spec.sigma_column, pdf_spec.weights_column]
            for c in pdf_cols:
                if c in chunk.columns:
                    column_dtypes.setdefault(c, f"f4[{n}]")
```

This relies entirely on the Phase 17 mini-language already wired into
`_chunk_to_table` for both `list<f4>` (variable-length) and `f4[N]`
(fixed-size) routing — no further writer changes required.

- [ ] **Step 4: Run the test**

```bash
pytest test/test_pdf_sample_hist.py -v
```

Expected: 8 passed (2 enum/spec + 4 spec aliases + 2 round-trip).

- [ ] **Step 5: Confirm no regression in existing PDF converter / reader**

```bash
pytest test/test_pdf_converter.py test/test_pdf_reader.py test/test_pdf_schema.py test/test_pdf_manifest.py -q
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/converter.py test/test_pdf_sample_hist.py
git commit -m "phase18/T2: _chunk_to_table routes sample (list<f4>) + hist (f4[N]) PDF payloads"
```

---

## Task 3: `ProbabilisticRedshift` reads sample / hist

**Files:**
- Modify: `oneuniverse/data/pdf.py:110-end` (`ProbabilisticRedshift`)
- Create: `test/test_pdf_reader_phase18.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_pdf_reader_phase18.py
"""Phase 18 T3 — ProbabilisticRedshift handles sample + hist."""
import numpy as np
import pandas as pd

from oneuniverse.data.pdf import PdfSpec, ProbabilisticRedshift


def test_sample_mean_matches_empirical():
    spec = PdfSpec(
        parameterisation="sample", n_components=4,
        grid=None, grid_kind="z",
    )
    df = pd.DataFrame({
        "z_pdf_values": [
            np.array([0.10, 0.30], dtype="f4"),
            np.array([0.50, 0.50, 0.50], dtype="f4"),
            np.array([0.70, 0.80, 0.90, 1.00], dtype="f4"),
        ],
    })
    pz = ProbabilisticRedshift.from_dataframe(df, spec)
    means = pz.mean()
    np.testing.assert_allclose(means[0], 0.20, atol=1e-5)
    np.testing.assert_allclose(means[1], 0.50, atol=1e-5)
    np.testing.assert_allclose(means[2], 0.85, atol=1e-5)


def test_hist_mean_uses_edges():
    spec = PdfSpec(
        parameterisation="hist", n_components=4,
        grid=None, grid_kind="z",
        hist_edges=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    df = pd.DataFrame({
        "z_pdf_values": [
            np.array([1.0, 0.0, 0.0, 0.0], dtype="f4"),
            np.array([0.0, 0.0, 0.0, 1.0], dtype="f4"),
            np.array([0.25, 0.25, 0.25, 0.25], dtype="f4"),
        ],
    })
    pz = ProbabilisticRedshift.from_dataframe(df, spec)
    centres = np.array([0.125, 0.375, 0.625, 0.875])
    means = pz.mean()
    np.testing.assert_allclose(means[0], centres[0], atol=1e-5)
    np.testing.assert_allclose(means[1], centres[3], atol=1e-5)
    np.testing.assert_allclose(means[2], centres.mean(), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_pdf_reader_phase18.py -v
```

Expected: `unsupported parameterisation 'sample'` / `'hist'`.

- [ ] **Step 3: Extend `ProbabilisticRedshift`**

Replace the `__init__` parameterisation switch (currently only handles
`interp` / `quant` / `mixmod`) so it also accepts `sample` and `hist`:

```python
        elif spec.parameterisation == "sample":
            # values is an object-dtype array of per-row np.ndarrays
            # (variable length); keep as-is and skip grid coercion.
            self.values = values
            self.grid = None
        elif spec.parameterisation == "hist":
            if spec.hist_edges is None:
                raise ValueError("hist parameterisation requires hist_edges")
            self.values = np.asarray(values, dtype=np.float64)
            edges = np.asarray(spec.hist_edges, dtype=np.float64)
            self.grid = 0.5 * (edges[:-1] + edges[1:])  # bin centres
            self._edges = edges
```

Replace the `from_dataframe` classmethod with one that respects the
new parameterisations and column aliases:

```python
    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, spec: PdfSpec,
    ) -> "ProbabilisticRedshift":
        if spec.parameterisation == "mixmod":
            mu = np.stack(df[spec.value_column].to_numpy())
            sigma = np.stack(df[spec.sigma_column].to_numpy())
            w = np.stack(df[spec.weights_column].to_numpy())
            return cls.from_mixmod(spec, mu, sigma, w)
        raw = df[spec.value_column].to_numpy()
        if spec.parameterisation == "sample":
            # Keep ragged; do not stack.
            values = np.empty(len(raw), dtype=object)
            for i, r in enumerate(raw):
                values[i] = np.asarray(r, dtype=np.float64)
            return cls(spec, values, grid=None)
        values = np.stack([np.asarray(r, dtype=np.float64) for r in raw])
        if spec.parameterisation == "interp":
            if spec.grid is None:
                raise ValueError("interp PdfSpec.grid must be set")
            return cls(spec, values, np.asarray(spec.grid, dtype=np.float64))
        if spec.parameterisation == "hist":
            return cls(spec, values, grid=None)
        return cls(spec, values, grid=None)
```

Extend the `mean()` method to handle the new branches:

```python
    def mean(self) -> np.ndarray:
        if self.spec.parameterisation == "interp":
            dz = self.grid[1] - self.grid[0]
            return (self.values * self.grid[None, :]).sum(axis=1) * dz
        if self.spec.parameterisation == "quant":
            return np.trapz(self.values, self.grid, axis=1)
        if self.spec.parameterisation == "sample":
            return np.array(
                [float(np.mean(v)) for v in self.values], dtype=np.float64,
            )
        if self.spec.parameterisation == "hist":
            centres = self.grid
            weights = self.values
            tot = weights.sum(axis=1, keepdims=True)
            tot = np.where(tot == 0.0, 1.0, tot)
            return (weights * centres[None, :]).sum(axis=1) / tot.squeeze(-1)
        mu, _sigma, w = self._mixmod
        return (w * mu).sum(axis=1)
```

(Other moment / CDF / sample methods can be extended later; the
public contract `len()` + `mean()` is sufficient for the Phase-18
acceptance bar.)

- [ ] **Step 4: Run the test**

```bash
pytest test/test_pdf_reader_phase18.py test/test_pdf_reader.py -v
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/pdf.py test/test_pdf_reader_phase18.py
git commit -m "phase18/T3: ProbabilisticRedshift handles sample (ragged) + hist (bin heights) + column aliases"
```

---

## Task 4: `TomographicNzSpec`

**Files:**
- Create: `oneuniverse/data/tomographic_nz.py`
- Create: `test/test_tomographic_nz_spec.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_tomographic_nz_spec.py
"""Phase 18 T4 — TomographicNzSpec sub-spec."""
import pytest

from oneuniverse.data.tomographic_nz import TomographicNzSpec


def test_defaults_and_required_fields():
    spec = TomographicNzSpec(
        bin_edges=[(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)],
        grid=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        values=[
            [0.0] * 11, [0.0] * 11, [0.0] * 11,
        ],
    )
    assert len(spec.bin_edges) == 3
    assert spec.bin_assignment_column == "tomo_bin"


def test_values_shape_must_match_bins_x_grid():
    with pytest.raises(ValueError, match="values"):
        TomographicNzSpec(
            bin_edges=[(0.0, 0.3), (0.3, 0.6)],
            grid=[0.0, 0.5, 1.0],
            values=[[0.0, 1.0, 0.0]],  # only 1 bin
        )


def test_to_dict_from_dict_roundtrip():
    spec = TomographicNzSpec(
        bin_edges=[(0.0, 0.3), (0.3, 0.6)],
        grid=[0.0, 0.5, 1.0],
        values=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        bin_assignment_column="tbin",
    )
    d = spec.to_dict()
    restored = TomographicNzSpec.from_dict(d)
    assert restored == spec
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_tomographic_nz_spec.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement the module**

```python
# oneuniverse/data/tomographic_nz.py
"""Tomographic n(z) sub-spec for OUF 2.4.

`TomographicNzSpec` is a **dataset-level** sidecar declaring a
per-bin n(z) plus the row-level column name carrying each row's
tomographic-bin assignment. It does not store probabilities per row
— that is what `PdfSpec` is for. Used by weak-lensing surveys
(KiDS-1000, DES-Y3, HSC-Y3) and any pipeline that delivers stacked
n(z) per tomographic bin via SOM cells.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class TomographicNzSpec:
    """Per-bin n(z) on a shared z grid.

    Parameters
    ----------
    bin_edges
        ``[(z_lo_1, z_hi_1), (z_lo_2, z_hi_2), ...]`` — one tuple
        per tomographic bin.
    grid
        Shared z grid (length ``n_grid``) over which every bin's
        n(z) is evaluated.
    values
        Sequence of length ``len(bin_edges)``, each element a sequence
        of length ``len(grid)`` carrying that bin's n(z).
    bin_assignment_column
        Name of the integer row-level column that records which bin
        each object belongs to. Defaults to ``"tomo_bin"``.
    """

    bin_edges: List[Tuple[float, float]]
    grid: List[float]
    values: List[List[float]]
    bin_assignment_column: str = "tomo_bin"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_bins = len(self.bin_edges)
        n_grid = len(self.grid)
        if len(self.values) != n_bins:
            raise ValueError(
                f"values length ({len(self.values)}) must match number "
                f"of bin_edges ({n_bins})"
            )
        for i, row in enumerate(self.values):
            if len(row) != n_grid:
                raise ValueError(
                    f"values[{i}] length ({len(row)}) must match grid "
                    f"length ({n_grid})"
                )
        # Normalise.
        object.__setattr__(
            self, "bin_edges",
            [(float(a), float(b)) for a, b in self.bin_edges],
        )
        object.__setattr__(self, "grid", [float(x) for x in self.grid])
        object.__setattr__(
            self, "values",
            [[float(x) for x in row] for row in self.values],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_edges": [list(e) for e in self.bin_edges],
            "grid": list(self.grid),
            "values": [list(row) for row in self.values],
            "bin_assignment_column": self.bin_assignment_column,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TomographicNzSpec":
        return cls(
            bin_edges=[tuple(e) for e in d["bin_edges"]],
            grid=list(d["grid"]),
            values=[list(row) for row in d["values"]],
            bin_assignment_column=d.get(
                "bin_assignment_column", "tomo_bin",
            ),
            extra=dict(d.get("extra", {})),
        )
```

- [ ] **Step 4: Run the test**

```bash
pytest test/test_tomographic_nz_spec.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/tomographic_nz.py test/test_tomographic_nz_spec.py
git commit -m "phase18/T4: TomographicNzSpec sub-spec (per-bin n(z), shared grid, bin_assignment_column)"
```

---

## Task 5: `ClassificationPdfSpec`

**Files:**
- Create: `oneuniverse/data/classification_pdf.py`
- Create: `test/test_classification_pdf_spec.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_classification_pdf_spec.py
"""Phase 18 T5 — ClassificationPdfSpec sub-spec."""
import pytest

from oneuniverse.data.classification_pdf import ClassificationPdfSpec


def test_defaults_and_classes_required():
    spec = ClassificationPdfSpec(classes=("galaxy", "qso", "star"))
    assert spec.value_column == "class_pdf_values"
    assert spec.parameterisation == "categorical"
    assert spec.n_classes == 3


def test_rejects_empty_classes():
    with pytest.raises(ValueError, match="classes"):
        ClassificationPdfSpec(classes=())


def test_rejects_unknown_parameterisation():
    with pytest.raises(ValueError, match="parameterisation"):
        ClassificationPdfSpec(
            classes=("a", "b"), parameterisation="mystery",
        )


def test_roundtrip():
    spec = ClassificationPdfSpec(
        classes=("galaxy", "qso", "star", "agn"),
        parameterisation="categorical",
        value_column="p_class",
    )
    d = spec.to_dict()
    restored = ClassificationPdfSpec.from_dict(d)
    assert restored == spec
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_classification_pdf_spec.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement the module**

```python
# oneuniverse/data/classification_pdf.py
"""Per-row classification PDF sub-spec for OUF 2.4.

`ClassificationPdfSpec` declares an ordered class label tuple and the
column on disk that stores the per-row probability vector. Use cases:
DESI ``SPECTYPE`` posteriors, ZTF / Fink classifier outputs, AGN-vs-
galaxy probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

_ALLOWED = frozenset({"categorical", "mixture"})


@dataclass(frozen=True)
class ClassificationPdfSpec:
    """Per-row class probability metadata.

    Parameters
    ----------
    classes
        Ordered tuple of class labels.
    parameterisation
        ``"categorical"`` (default — probabilities sum to ~1) or
        ``"mixture"`` (probabilities + component widths declared via
        ``extra``; not yet exercised by the reader).
    value_column
        Per-row column name on disk; stored as ``f4[n_classes]``.
        Default ``"class_pdf_values"``.
    """

    classes: Tuple[str, ...]
    parameterisation: str = "categorical"
    value_column: str = "class_pdf_values"
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.classes:
            raise ValueError("classes must not be empty")
        if self.parameterisation not in _ALLOWED:
            raise ValueError(
                f"unknown parameterisation {self.parameterisation!r}; "
                f"allowed: {sorted(_ALLOWED)}"
            )
        # Normalise classes tuple.
        object.__setattr__(
            self, "classes", tuple(str(c) for c in self.classes),
        )

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classes": list(self.classes),
            "parameterisation": self.parameterisation,
            "value_column": self.value_column,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassificationPdfSpec":
        return cls(
            classes=tuple(d["classes"]),
            parameterisation=d.get("parameterisation", "categorical"),
            value_column=d.get("value_column", "class_pdf_values"),
            extra=dict(d.get("extra", {})),
        )
```

- [ ] **Step 4: Run the test**

```bash
pytest test/test_classification_pdf_spec.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/classification_pdf.py test/test_classification_pdf_spec.py
git commit -m "phase18/T5: ClassificationPdfSpec sub-spec (ordered classes + per-row f4[n_classes])"
```

---

## Task 6: Wire `tomographic_nz` + `classification_pdf` into Manifest + OUF 2.4 bump

**Files:**
- Modify: `oneuniverse/data/manifest.py` (imports, version constants, Manifest fields, (de)serialisation, version-compat clause)
- Modify: `oneuniverse/data/format_spec.py` (version constants)
- Modify: `test/test_lightcurve_geometry.py`
- Modify: `test/test_manifest_phase16.py` (version assertion)
- Create: `test/test_manifest_phase18.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_manifest_phase18.py
"""Phase 18 T6 — Manifest carries tomographic_nz + classification_pdf
and round-trips OUF 2.4 ↔ 2.3.
"""
import json

import pytest

from oneuniverse.data.classification_pdf import ClassificationPdfSpec
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import (
    FORMAT_VERSION,
    LoaderSpec,
    Manifest,
    OriginalFileSpec,
    PartitionSpec,
    PartitionStats,
    read_manifest,
    write_manifest,
)
from oneuniverse.data.tomographic_nz import TomographicNzSpec


def _minimal_manifest(**overrides) -> Manifest:
    defaults = dict(
        oneuniverse_format_version=FORMAT_VERSION,
        oneuniverse_schema_version=FORMAT_VERSION,
        geometry=DataGeometry.POINT,
        survey_name="fixture", survey_type="photometric",
        created_utc="2026-05-29T00:00:00+00:00",
        original_files=[OriginalFileSpec(
            path="raw.fits", sha256="0123456789abcdef",
            n_rows=10, size_bytes=4096, format="fits",
        )],
        partitions=[PartitionSpec(
            name="data/part_0000.parquet",
            n_rows=10, sha256="fedcba9876543210", size_bytes=2048,
            stats=PartitionStats(),
        )],
        partitioning=None, schema=[], conversion_kwargs={},
        loader=LoaderSpec(name="fixture_loader", version="0.0"),
    )
    defaults.update(overrides)
    return Manifest(**defaults)


def test_version_constants_bumped():
    assert FORMAT_VERSION == "2.4.0"


def test_manifest_carries_tomographic_nz(tmp_path):
    spec = TomographicNzSpec(
        bin_edges=[(0.0, 0.3), (0.3, 0.6)],
        grid=[0.0, 0.5, 1.0],
        values=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    )
    m = _minimal_manifest(tomographic_nz=spec)
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read.tomographic_nz == spec


def test_manifest_carries_classification_pdf(tmp_path):
    spec = ClassificationPdfSpec(classes=("galaxy", "qso", "star"))
    m = _minimal_manifest(classification_pdf=spec)
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read.classification_pdf == spec


def test_reads_2_3_manifest_with_compat_defaults(tmp_path):
    payload = {
        "oneuniverse_format_version": "2.3.0",
        "oneuniverse_schema_version": "2.3.0",
        "geometry": "point",
        "survey_name": "legacy", "survey_type": "photometric",
        "created_utc": "2026-05-28T00:00:00+00:00",
        "original_files": [{
            "path": "raw.fits", "sha256": "0123456789abcdef",
            "n_rows": 1, "size_bytes": 100, "format": "fits",
        }],
        "partitions": [{
            "name": "data/part_0000.parquet", "n_rows": 1,
            "sha256": "fedcba9876543210", "size_bytes": 50,
        }],
        "partitioning": None, "schema": [], "conversion_kwargs": {},
        "loader": {"name": "legacy_loader", "version": "0.0"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    read = read_manifest(path)
    assert read.tomographic_nz is None
    assert read.classification_pdf is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_manifest_phase18.py -v
```

Expected: failure on the version-bump check + missing Manifest fields.

- [ ] **Step 3: Bump constants + extend version-compat clause**

`oneuniverse/data/manifest.py`:

```python
FORMAT_VERSION: str = "2.4.0"
SCHEMA_VERSION: str = "2.4.0"
```

In `_from_dict` extend the version clause:

```python
        and (
            fmt.startswith("2.0") or fmt.startswith("2.1")
            or fmt.startswith("2.2") or fmt.startswith("2.3")
            or fmt.startswith("2.4")
        )
```

```python
        raise ManifestValidationError(
            f"{path}: oneuniverse_format_version={fmt!r} is not compatible "
            f"with this library (expected 2.0.x / 2.1.x / 2.2.x / 2.3.x "
            f"/ 2.4.x)."
        )
```

`oneuniverse/data/format_spec.py`:

```python
FORMAT_VERSION: str = "2.4.0"
SCHEMA_VERSION: str = "2.4.0"
```

`test/test_lightcurve_geometry.py`:

```python
def test_format_version_is_2_4_0():
    assert FORMAT_VERSION == "2.4.0"
    assert SCHEMA_VERSION == "2.4.0"
```

`test/test_manifest_phase16.py`:

```python
def test_version_constants_bumped():
    assert FORMAT_VERSION == "2.4.0"
```

- [ ] **Step 4: Add the new Manifest fields + (de)serialisation**

In `oneuniverse/data/manifest.py`, add the two new imports near the
existing sub-spec imports:

```python
from oneuniverse.data.classification_pdf import ClassificationPdfSpec
from oneuniverse.data.tomographic_nz import TomographicNzSpec
```

Extend the `Manifest` dataclass body so it ends with:

```python
    coordinate: Optional[CoordinateSpec] = None
    spectrum: Optional[SpectrumSpec] = None
    observed_z_types: tuple = ()
    # Phase 18 additions.
    tomographic_nz: Optional[TomographicNzSpec] = None
    classification_pdf: Optional[ClassificationPdfSpec] = None
```

Extend `_to_dict`:

```python
    d["tomographic_nz"] = (
        m.tomographic_nz.to_dict() if m.tomographic_nz is not None else None
    )
    d["classification_pdf"] = (
        m.classification_pdf.to_dict()
        if m.classification_pdf is not None else None
    )
```

Extend `_from_dict` just before the final `return Manifest(...)`:

```python
    tnz_raw = raw.get("tomographic_nz")
    tomographic_nz = (
        TomographicNzSpec.from_dict(tnz_raw) if tnz_raw else None
    )
    cpd_raw = raw.get("classification_pdf")
    classification_pdf = (
        ClassificationPdfSpec.from_dict(cpd_raw) if cpd_raw else None
    )
```

and extend the `Manifest(...)` constructor with:

```python
        tomographic_nz=tomographic_nz,
        classification_pdf=classification_pdf,
```

- [ ] **Step 5: Run the test**

```bash
pytest test/test_manifest_phase18.py test/test_manifest.py test/test_manifest_phase16.py test/test_lightcurve_geometry.py -q
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/manifest.py oneuniverse/data/format_spec.py \
        test/test_manifest_phase18.py test/test_manifest_phase16.py \
        test/test_lightcurve_geometry.py
git commit -m "phase18/T6: Manifest gains tomographic_nz + classification_pdf; bump to OUF 2.4.0"
```

---

## Task 7: Visual diagnostic

**Files:**
- Create: `test/test_visual_phase18.py`

- [ ] **Step 1: Write the test**

```python
# test/test_visual_phase18.py
"""Phase 18 visual diagnostic — sample/hist PDFs + tomographic n(z)."""
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
    # Manually attach tomo spec for the visual (real loaders pass it
    # via convert_survey; for this diagnostic we set it post-hoc).
    from dataclasses import replace
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
```

- [ ] **Step 2: Run the test**

```bash
pytest test/test_visual_phase18.py -v
```

Expected: pass; PNG ≥ 30 kB at
`test/test_output/phase18_pdf_polymorphism_and_tomographic_nz.png`.

- [ ] **Step 3: Commit**

```bash
git add test/test_visual_phase18.py \
        test/test_output/phase18_pdf_polymorphism_and_tomographic_nz.png
git commit -m "phase18/T7: visual diagnostic — hist PDFs, mean-vs-truth, tomographic n(z)"
```

---

## Task 8: Docs + plan-README + audit cross-ref

**Files:**
- Modify: `oneuniverse/CLAUDE.md` (OUF 2.3 → 2.4 + new sub-specs)
- Modify: `plans/README.md` (Phase 18 row)
- Modify: `research/schema_generalisation_audit.md` (Phase 18 close-out)

- [ ] **Step 1: CLAUDE.md**

Change "OUF 2.3 (format on disk)" to "OUF 2.4 (format on disk)" and
under the Phase 17 sub-spec list append:

```
- `PdfSpec` covers `interp / quant / mixmod / sample / hist` and
  carries configurable column aliases (`value_column`, `sigma_column`,
  `weights_column`) so RAIL / qp catalogs round-trip without renaming.
  `hist` stores per-row bin heights as `f4[N]`; `sample` stores
  per-row z-draws as `list<f4>`.
- `TomographicNzSpec` (per-bin n(z) on a shared z grid +
  `bin_assignment_column` int row column) and `ClassificationPdfSpec`
  (ordered class tuple + per-row `f4[n_classes]`) are dataset-level
  Manifest sub-specs (Phase 18).
```

- [ ] **Step 2: plans/README.md**

```
| 18 | PDF polymorphism (`sample`/`hist`) + column aliases + `TomographicNzSpec` + `ClassificationPdfSpec` (OUF → 2.4.0) | **complete (2026-05-29, NNN/NNN tests green)** |
```

- [ ] **Step 3: research/schema_generalisation_audit.md**

Replace the existing "Phase 18 —" bullet with:

```
- **Phase 18 — PDF polymorphism + tomographic n(z) + classification PDFs.**
  Landed 2026-05-29. `PdfSpec` covers
  `interp / quant / mixmod / sample / hist` and carries
  `value_column / sigma_column / weights_column / hist_edges /
  grid_mask / axis_labels`. New manifest sub-specs `TomographicNzSpec`
  and `ClassificationPdfSpec`. OUF 2.4.0. See
  [`../plans/2026-05-29-phase18-pdf-polymorphism.md`](../plans/2026-05-29-phase18-pdf-polymorphism.md).
```

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/CLAUDE.md plans/README.md \
        research/schema_generalisation_audit.md
git commit -m "docs(phase18): OUF 2.4, PDF polymorphism, tomographic n(z), classification PDFs"
```

---

## Task 9: Close-out

- [ ] **Step 1: Run the full suite**

```bash
pytest -q 2>&1 | tail -3
```

Expected: green. Record the count (Phase 17 baseline 428 + ~25 new
Phase 18 tests).

- [ ] **Step 2: Fill in plans/README.md count.**

- [ ] **Step 3: Update memory**

Append to
`/home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md`:

```markdown
## Phase 18 — PDF polymorphism + tomographic n(z) + classification PDFs (complete 2026-05-29)

- `PdfParameterisation` adds `SAMPLE` (per-row variable-length z
  draws via `list<f4>`) and `HIST` (per-row bin heights on shared
  `hist_edges`, stored as `f4[N]`).
- `PdfSpec` gains `value_column / sigma_column / weights_column`
  aliases, `hist_edges`, `grid_mask`, `axis_labels`. Existing
  Phase 10 PDFs continue to round-trip unchanged.
- `ProbabilisticRedshift` extended to handle `sample` (ragged) and
  `hist` (bin heights) — at minimum `mean()` works.
- `TomographicNzSpec` (per-bin n(z) on shared z grid +
  `bin_assignment_column`) and `ClassificationPdfSpec` (ordered
  classes + per-row `f4[n_classes]`) attach to Manifest.
- OUF format bump 2.3.0 -> 2.4.0; 2.0-2.3 still parse.
- Tests: NNN/NNN green.
- Per-phase plan: `plans/2026-05-29-phase18-pdf-polymorphism.md`.
```

- [ ] **Step 4: Final commit**

```bash
git add plans/README.md \
        /home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md
git commit -m "phase18: close-out — OUF 2.4.0, NNN tests green"
```

---

## Self-review checklist

- [ ] No cosmology metadata added anywhere.
- [ ] Phase 10 PDFs (interp/quant/mixmod) still round-trip with the
      hardcoded `z_pdf_values`/`z_pdf_sigma`/`z_pdf_weights` defaults.
- [ ] `sample` PDFs survive write + read with variable per-row length.
- [ ] `hist` PDFs survive write + read with `f4[n_components]` storage.
- [ ] `TomographicNzSpec.values.shape == (n_bins, n_grid)`.
- [ ] OUF 2.0/2.1/2.2/2.3 manifests still parse.
- [ ] Visual PNG `phase18_pdf_polymorphism_and_tomographic_nz.png`
      exists, ≥ 30 kB.

## Spec-coverage map

| Requirement | Task |
|---|---|
| `PdfSpec` adds `sample` + `hist` | T1, T2, T3 |
| Configurable PDF column aliases | T1 |
| `grid_mask` / `axis_labels` declared | T1 |
| Writer routes sample (list<f4>) + hist (f4[N]) | T2 |
| Reader handles sample + hist (`mean()`) | T3 |
| `TomographicNzSpec` | T4, T6 |
| `ClassificationPdfSpec` | T5, T6 |
| Manifest integration + OUF 2.4 bump + back-compat | T6 |
| Visual diagnostic | T7 |
| Docs | T8 |
| Close-out + memory | T9 |
