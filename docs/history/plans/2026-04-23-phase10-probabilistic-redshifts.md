# Phase 10 — Probabilistic Redshifts (photo-z PDFs)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let oneuniverse store *full redshift PDFs* per object — not only a scalar `z` + `z_err` — so photometric/probabilistic surveys (LSST/Rubin DP, Euclid photo-z, legacy BPZ/EAZY outputs) can be onboarded losslessly and consumed by downstream inference code that wants the PDF.

**Architecture:** Add a **`probabilistic_redshift` column group** to the OUF CORE+extension schema. Each PDF is stored as one or more **fixed-size list parquet columns** (`FixedSizeList[f4, N]`) alongside the scalar columns, so DatasetView can read them back without special-casing and without blowing row size up (N is the same for every object in a dataset). The manifest's `extra` dict records the parameterisation (`"interp" | "quant" | "mixmod"`) + grid / components. A lightweight reader (`oneuniverse.data.pdf.ProbabilisticRedshift`) returns `(grid, pdf_values)` tuples; no hard `qp` dependency — `qp` stays an optional extra.

**Why fixed-size list columns:** PDFs have identical length per object within a dataset. pyarrow's `FixedSizeListType` stores them contiguously (no offset array), gives random access without decoding a variable-size list, and round-trips through pandas→pyarrow→parquet without schema drift. Variable-size list columns are only needed if PDF length varies per-object, which is never the case for standard photo-z pipelines.

**Tech Stack:** pyarrow (FixedSizeList), numpy, pandas, existing oneuniverse schema + manifest + converter, fitsio (fixture only), matplotlib (diagnostic). `qp` is optional — guarded import.

---

## File Structure

- Create: `oneuniverse/data/pdf.py` — `PdfParameterisation` enum + `ProbabilisticRedshift` reader + helpers.
- Modify: `oneuniverse/data/schema.py` — add `PROBABILISTIC_REDSHIFT_COLUMNS` group + register in `COLUMN_GROUPS`.
- Modify: `oneuniverse/data/manifest.py` — add optional `pdf_spec: Optional[PdfSpec]` on `Manifest`; expose `PdfSpec` dataclass + round-trip in `_to_dict` / `_from_dict`.
- Modify: `oneuniverse/data/converter.py` — when schema declares `probabilistic_redshift`, cast list columns to `FixedSizeList[f4, N]` on the pyarrow side before writing.
- Modify: `oneuniverse/data/dataset_view.py` — expose `.load_pdf(column_group="probabilistic_redshift")` that returns `ProbabilisticRedshift` bound to the manifest's `pdf_spec`.
- Modify: `oneuniverse/data/_base_loader.py` — loaders may now declare `"probabilistic_redshift"` in `column_groups`; validate PDF-column shapes.
- Create: `oneuniverse/combine/weights/pdf.py` — `PdfMeanRedshiftWeight` (plugs PDF mean into FKP/IVar) and `PdfWidthIVarWeight(pdf_std_column)` (inverse-variance of PDF width).
- Modify: `oneuniverse/combine/weights/__init__.py` — export new classes.
- Modify: `oneuniverse/combine/weights/registry.py` — register `("photometric", "phot_pdf")` default.
- Create: `test/test_pdf_schema.py` — schema/group registration + required cols + dtypes.
- Create: `test/test_pdf_manifest.py` — `PdfSpec` round-trip.
- Create: `test/test_pdf_reader.py` — `ProbabilisticRedshift` CDF/PPF/mean/sample correctness on interp/quant/mixmod fixtures.
- Create: `test/test_pdf_converter.py` — end-to-end: fake loader with interp PDFs → converter → DatasetView → reader → plots.
- Create: `test/fixtures/pdf_catalog.py` — synthetic photo-z catalog with known-Gaussian PDFs for golden tests.
- Create: `test/test_output/phase10_pdf_*.png` — diagnostic figures (PDF samples, mean vs. z_true, CDF round-trip).

---

### Task 1: `PdfParameterisation` enum + `PdfSpec` dataclass

**Files:**
- Create: `oneuniverse/data/pdf.py`
- Test: `test/test_pdf_schema.py`

**Why:** The manifest needs a strongly-typed record of *which* PDF representation a dataset uses so readers can reconstruct without parsing heuristics. Mirrors how `TemporalSpec` and `DatasetValidity` are handled.

- [ ] **Step 1: Write failing test**

```python
# test/test_pdf_schema.py
import pytest
from oneuniverse.data.pdf import PdfParameterisation, PdfSpec


def test_parameterisation_values():
    assert {p.value for p in PdfParameterisation} == {"interp", "quant", "mixmod"}


def test_pdfspec_interp_requires_grid():
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


def test_pdfspec_quant_requires_levels_not_grid():
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
```

Run: `pytest test/test_pdf_schema.py -v` → FAIL (module missing).

- [ ] **Step 2: Implement `pdf.py` (first cut: enum + PdfSpec only)**

```python
# oneuniverse/data/pdf.py
"""Probabilistic-redshift support for OUF 2.1.

Defines:

* :class:`PdfParameterisation` — enum of supported PDF representations
  (``interp`` / ``quant`` / ``mixmod``).  Mirrors the qp package (Malz
  & Marshall 2018, arXiv:1806.00014) but stays pure-numpy so ``qp`` is
  an optional extra.
* :class:`PdfSpec` — dataclass stored in ``Manifest.pdf_spec``; single
  source of truth for how to reconstruct a PDF from on-disk columns.
* :class:`ProbabilisticRedshift` — reader that wraps a ``DatasetView``
  and returns ``(grid, pdf_values)`` tuples plus convenience methods
  (``mean``, ``std``, ``cdf``, ``ppf``, ``sample``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PdfParameterisation(str, Enum):
    INTERP = "interp"
    QUANT = "quant"
    MIXMOD = "mixmod"


_KNOWN = {p.value for p in PdfParameterisation}


@dataclass(frozen=True)
class PdfSpec:
    """Describes a probabilistic-redshift representation.

    Parameters
    ----------
    parameterisation
        One of ``"interp"``, ``"quant"``, ``"mixmod"``.
    n_components
        Fixed length of every PDF array in this dataset (grid points for
        interp, quantiles for quant, mixture components for mixmod).
    grid
        For ``interp``: the common z grid (length ``n_components``).
        For ``mixmod``: ignored.  For ``quant``: ignored; use
        ``quant_levels`` instead.
    grid_kind
        ``"z"`` for redshift grid, ``"quantile"`` for quantile levels,
        ``"component"`` for mixture indices.
    quant_levels
        For ``quant``: quantile levels in [0, 1] (length ``n_components``).
    """

    parameterisation: str
    n_components: int
    grid: Optional[List[float]]
    grid_kind: str
    quant_levels: Optional[List[float]] = None
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameterisation": self.parameterisation,
            "n_components": int(self.n_components),
            "grid": list(self.grid) if self.grid is not None else None,
            "grid_kind": self.grid_kind,
            "quant_levels": (
                list(self.quant_levels) if self.quant_levels is not None else None
            ),
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
                list(d["quant_levels"]) if d.get("quant_levels") is not None else None
            ),
            extra=dict(d.get("extra", {})),
        )
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_pdf_schema.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/data/pdf.py test/test_pdf_schema.py
git commit -m "phase10: add PdfParameterisation + PdfSpec dataclass"
```

---

### Task 2: `probabilistic_redshift` column group in schema

**Files:**
- Modify: `oneuniverse/data/schema.py`
- Test: `test/test_pdf_schema.py` (extend)

**Why:** Downstream validation must know which scalar + list columns a probabilistic-redshift dataset *must* expose. Keeps the schema the single source of truth; no loader-private column lists.

- [ ] **Step 1: Extend failing test**

```python
# test/test_pdf_schema.py (append)
from oneuniverse.data import schema


def test_probabilistic_redshift_group_registered():
    assert "probabilistic_redshift" in schema.COLUMN_GROUPS


def test_probabilistic_redshift_required_columns():
    req = set(schema.get_required_columns(["probabilistic_redshift"]))
    assert {"z_pdf_kind", "z_pdf_values"} <= req


def test_probabilistic_redshift_scalar_summary_columns_optional():
    cols = schema.get_all_columns(["probabilistic_redshift"])
    assert cols["z_pdf_mean"].required is False
    assert cols["z_pdf_std"].required is False
```

Run: `pytest test/test_pdf_schema.py -v` → FAIL (group missing).

- [ ] **Step 2: Add group to `schema.py`**

```python
# oneuniverse/data/schema.py — insert after SNIA_COLUMNS
PROBABILISTIC_REDSHIFT_COLUMNS: Tuple[ColumnDef, ...] = (
    ColumnDef(
        "z_pdf_kind", "U8", "",
        "PDF parameterisation tag: interp | quant | mixmod",
    ),
    # Fixed-size list columns — dtype hint is the element dtype. The
    # converter/manifest records the component count in PdfSpec; the
    # reader rebuilds the full signature.
    ColumnDef(
        "z_pdf_values", "f4", "",
        "PDF values: interp p(z), quant quantile z values, or mixmod means",
    ),
    ColumnDef(
        "z_pdf_sigma", "f4", "",
        "mixmod only: component std devs", required=False,
    ),
    ColumnDef(
        "z_pdf_weights", "f4", "",
        "mixmod only: component weights", required=False,
    ),
    # Scalar summaries — optional but strongly recommended so downstream
    # code that only wants a point redshift does not have to parse PDFs.
    ColumnDef("z_pdf_mean", "f4", "", "PDF first moment", required=False),
    ColumnDef("z_pdf_std", "f4", "", "PDF standard deviation", required=False),
    ColumnDef("z_pdf_median", "f4", "", "PDF median", required=False),
    ColumnDef("z_pdf_mode", "f4", "", "PDF mode", required=False),
    ColumnDef("z_pdf_l68", "f4", "", "PDF 16th percentile", required=False),
    ColumnDef("z_pdf_u68", "f4", "", "PDF 84th percentile", required=False),
)

COLUMN_GROUPS: Dict[str, Tuple[ColumnDef, ...]] = {
    "core": CORE_COLUMNS,
    "spectroscopic": SPECTROSCOPIC_COLUMNS,
    "photometric": PHOTOMETRIC_COLUMNS,
    "peculiar_velocity": PV_COLUMNS,
    "qso": QSO_COLUMNS,
    "snia": SNIA_COLUMNS,
    "probabilistic_redshift": PROBABILISTIC_REDSHIFT_COLUMNS,
}
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_pdf_schema.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/data/schema.py test/test_pdf_schema.py
git commit -m "phase10: register probabilistic_redshift column group"
```

---

### Task 3: `PdfSpec` round-trip through the manifest

**Files:**
- Modify: `oneuniverse/data/manifest.py`
- Test: `test/test_pdf_manifest.py`

**Why:** The manifest must record parameterisation + grid so the reader reconstructs PDFs without guessing. Mirrors how `TemporalSpec`/`DatasetValidity` were threaded through.

- [ ] **Step 1: Failing test**

```python
# test/test_pdf_manifest.py
from pathlib import Path

from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import (
    ColumnSpec, LoaderSpec, Manifest, OriginalFileSpec, read_manifest,
    write_manifest,
)
from oneuniverse.data.pdf import PdfSpec


def _make_minimal_manifest(pdf_spec):
    return Manifest(
        oneuniverse_format_version="2.1.0",
        oneuniverse_schema_version="2.1.0",
        geometry=DataGeometry.POINT,
        survey_name="fake",
        survey_type="photometric",
        created_utc="2026-04-23T00:00:00+00:00",
        original_files=[OriginalFileSpec(
            path="x.fits", sha256="abc", n_rows=1, size_bytes=1, format="fits"
        )],
        partitions=[],
        partitioning=None,
        schema=[ColumnSpec(name="z", dtype="f4")],
        conversion_kwargs={},
        loader=LoaderSpec(name="fake", version="0"),
        pdf_spec=pdf_spec,
    )


def test_manifest_roundtrip_with_pdf_spec(tmp_path):
    spec = PdfSpec(
        parameterisation="interp", n_components=3, grid=[0.0, 0.5, 1.0],
        grid_kind="z",
    )
    m = _make_minimal_manifest(spec)
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    back = read_manifest(path)
    assert back.pdf_spec == spec


def test_manifest_pdf_spec_is_none_by_default(tmp_path):
    m = _make_minimal_manifest(None)
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    back = read_manifest(path)
    assert back.pdf_spec is None
```

Run: `pytest test/test_pdf_manifest.py -v` → FAIL.

- [ ] **Step 2: Thread `pdf_spec` through `Manifest` + serialisers**

```python
# oneuniverse/data/manifest.py — add import
from oneuniverse.data.pdf import PdfSpec

# inside @dataclass(frozen=True) class Manifest — add field:
    pdf_spec: Optional["PdfSpec"] = None
```

In `_to_dict`:

```python
    d["pdf_spec"] = m.pdf_spec.to_dict() if m.pdf_spec is not None else None
```

In `_from_dict` (before the return, alongside `temporal`/`validity`):

```python
    pdf_raw = raw.get("pdf_spec")
    pdf_spec = PdfSpec.from_dict(pdf_raw) if pdf_raw is not None else None
```

Add `pdf_spec=pdf_spec` to the `Manifest(...)` return.

- [ ] **Step 3: Run test**

Run: `pytest test/test_pdf_manifest.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/data/manifest.py test/test_pdf_manifest.py
git commit -m "phase10: thread PdfSpec through Manifest read/write"
```

---

### Task 4: Synthetic photo-z catalog fixture (Gaussian PDFs on a grid)

**Files:**
- Create: `test/fixtures/pdf_catalog.py`
- Test: `test/test_pdf_fixture.py`

**Why:** Every downstream test (reader, converter, weight) needs a reproducible catalog with *known* PDF moments so CDF/PPF/mean assertions have a golden reference. Gaussian PDFs on a uniform grid are the easiest case with analytic moments.

- [ ] **Step 1: Failing test**

```python
# test/test_pdf_fixture.py
import numpy as np
from test.fixtures.pdf_catalog import make_gaussian_pdf_catalog


def test_gaussian_catalog_pdf_integrates_to_one():
    df, grid = make_gaussian_pdf_catalog(n_rows=100, n_grid=201, seed=0)
    assert len(df) == 100
    assert len(grid) == 201
    pdfs = np.stack(df["z_pdf_values"].values)  # (100, 201)
    dz = grid[1] - grid[0]
    integrals = pdfs.sum(axis=1) * dz
    assert np.allclose(integrals, 1.0, atol=1e-2)


def test_gaussian_catalog_mean_matches_column():
    df, grid = make_gaussian_pdf_catalog(n_rows=50, n_grid=301, seed=1)
    pdfs = np.stack(df["z_pdf_values"].values)
    dz = grid[1] - grid[0]
    means = (pdfs * grid[None, :]).sum(axis=1) * dz
    assert np.allclose(means, df["z_pdf_mean"].to_numpy(), atol=1e-2)
```

Run: `pytest test/test_pdf_fixture.py -v` → FAIL (module missing).

- [ ] **Step 2: Implement fixture**

```python
# test/fixtures/pdf_catalog.py
"""Synthetic photo-z catalog: one Gaussian per object on a uniform grid."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_gaussian_pdf_catalog(
    n_rows: int = 100,
    n_grid: int = 201,
    z_min: float = 0.0,
    z_max: float = 2.0,
    sigma_range: tuple = (0.02, 0.08),
    seed: int = 0,
) -> tuple:
    """Return ``(df, grid)`` — DataFrame with one Gaussian PDF per row.

    Columns:
      ra, dec              — uniform on a small patch
      z_pdf_kind           — "interp"
      z_pdf_values         — list[float], length n_grid
      z_pdf_mean, z_pdf_std, z_true
    """
    rng = np.random.default_rng(seed)
    grid = np.linspace(z_min, z_max, n_grid, dtype=np.float32)
    dz = grid[1] - grid[0]

    mu = rng.uniform(z_min + 0.1, z_max - 0.1, size=n_rows).astype(np.float32)
    sigma = rng.uniform(*sigma_range, size=n_rows).astype(np.float32)
    # Normalised Gaussian sampled on the grid then renormalised discretely.
    diff = grid[None, :] - mu[:, None]
    pdfs = np.exp(-0.5 * (diff / sigma[:, None]) ** 2)
    pdfs /= pdfs.sum(axis=1, keepdims=True) * dz
    pdfs = pdfs.astype(np.float32)

    ra = rng.uniform(150.0, 160.0, size=n_rows).astype(np.float64)
    dec = rng.uniform(0.0, 10.0, size=n_rows).astype(np.float64)

    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "z_true": mu.astype(np.float32),
        "z_pdf_kind": np.array(["interp"] * n_rows, dtype="U8"),
        "z_pdf_values": [row for row in pdfs],  # each a 1D ndarray
        "z_pdf_mean": mu,
        "z_pdf_std": sigma,
    })
    return df, grid
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_pdf_fixture.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add test/fixtures/pdf_catalog.py test/test_pdf_fixture.py
git commit -m "phase10: Gaussian-PDF photo-z catalog fixture"
```

---

### Task 5: `ProbabilisticRedshift` reader (interp parameterisation first)

**Files:**
- Modify: `oneuniverse/data/pdf.py`
- Test: `test/test_pdf_reader.py`

**Why:** Downstream code needs a stable API for PDF consumption — `mean`, `std`, `cdf(z)`, `ppf(q)`, `sample(n)` — without hard-coupling to `qp`. Start with `interp` (the most common case); `quant`/`mixmod` follow.

- [ ] **Step 1: Failing test**

```python
# test/test_pdf_reader.py
import numpy as np

from oneuniverse.data.pdf import PdfSpec, ProbabilisticRedshift
from test.fixtures.pdf_catalog import make_gaussian_pdf_catalog


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
    mean = pz.mean()
    np.testing.assert_allclose(mean, df["z_pdf_mean"].to_numpy(), atol=1e-2)


def test_reader_std_matches_input_sigma():
    df, _, pz = _reader()
    std = pz.std()
    np.testing.assert_allclose(std, df["z_pdf_std"].to_numpy(), rtol=5e-2)


def test_reader_cdf_monotone():
    _, _, pz = _reader()
    cdf = pz.cdf()  # (n_rows, n_grid) cumulative sum of p(z) dz
    diffs = np.diff(cdf, axis=1)
    assert (diffs >= -1e-6).all()


def test_reader_ppf_inverts_cdf_at_median():
    _, _, pz = _reader()
    z05 = pz.ppf(0.5)  # per-row
    # A Gaussian's median equals its mean.
    np.testing.assert_allclose(z05, pz.mean(), atol=2e-2)


def test_reader_sample_covers_pdf():
    _, _, pz = _reader()
    samples = pz.sample(n_per=200, seed=0)  # (n_rows, 200)
    assert samples.shape == (len(pz), 200)
    # Row-wise empirical mean ~ pdf mean within 3 sigma / sqrt(200).
    emp = samples.mean(axis=1)
    tol = 3.0 * pz.std() / np.sqrt(200)
    assert (np.abs(emp - pz.mean()) < tol).all()
```

Run: `pytest test/test_pdf_reader.py -v` → FAIL.

- [ ] **Step 2: Implement the reader**

```python
# oneuniverse/data/pdf.py — append below PdfSpec

import numpy as np
import pandas as pd


class ProbabilisticRedshift:
    """Vectorised PDF accessor bound to a :class:`PdfSpec`.

    Today only supports ``parameterisation == "interp"``; ``quant`` and
    ``mixmod`` arrive in later tasks.  All methods operate on *all* rows
    at once and return ``(n_rows,)`` or ``(n_rows, ...)`` arrays.
    """

    def __init__(self, spec: PdfSpec, values: np.ndarray, grid: np.ndarray):
        if spec.parameterisation != "interp":
            raise NotImplementedError(
                f"parameterisation {spec.parameterisation!r} not yet supported "
                f"by ProbabilisticRedshift; see Task 8."
            )
        if values.ndim != 2:
            raise ValueError(f"values must be 2D, got shape {values.shape}")
        if values.shape[1] != len(grid):
            raise ValueError(
                f"values second axis ({values.shape[1]}) must match "
                f"grid length ({len(grid)})"
            )
        self.spec = spec
        self.values = np.asarray(values, dtype=np.float64)
        self.grid = np.asarray(grid, dtype=np.float64)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, spec: PdfSpec) -> "ProbabilisticRedshift":
        raw = df["z_pdf_values"].to_numpy()
        values = np.stack([np.asarray(r, dtype=np.float64) for r in raw])
        if spec.grid is None:
            raise ValueError("spec.grid must be set for interp parameterisation")
        return cls(spec, values, np.asarray(spec.grid, dtype=np.float64))

    def __len__(self) -> int:
        return self.values.shape[0]

    # ── Moments ──
    def _dz(self) -> np.ndarray:
        return np.diff(self.grid, prepend=self.grid[0] - (self.grid[1] - self.grid[0]))

    def mean(self) -> np.ndarray:
        dz = self.grid[1] - self.grid[0]
        return (self.values * self.grid[None, :]).sum(axis=1) * dz

    def std(self) -> np.ndarray:
        dz = self.grid[1] - self.grid[0]
        m = self.mean()
        var = (self.values * (self.grid[None, :] - m[:, None]) ** 2).sum(axis=1) * dz
        return np.sqrt(np.maximum(var, 0.0))

    # ── CDF / PPF / sampling ──
    def cdf(self) -> np.ndarray:
        dz = self.grid[1] - self.grid[0]
        c = np.cumsum(self.values, axis=1) * dz
        # renormalise so each row ends exactly at 1
        c /= np.maximum(c[:, -1:], 1e-300)
        return c

    def ppf(self, q) -> np.ndarray:
        q = np.atleast_1d(np.asarray(q, dtype=np.float64))
        c = self.cdf()  # (n_rows, n_grid)
        # per-row linear interp of c -> grid at target q
        out = np.empty((c.shape[0], q.size), dtype=np.float64)
        for i in range(c.shape[0]):
            out[i] = np.interp(q, c[i], self.grid)
        return out[:, 0] if q.size == 1 else out

    def sample(self, n_per: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        q = rng.uniform(0.0, 1.0, size=(len(self), n_per))
        c = self.cdf()
        out = np.empty_like(q)
        for i in range(len(self)):
            out[i] = np.interp(q[i], c[i], self.grid)
        return out
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_pdf_reader.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/data/pdf.py test/test_pdf_reader.py
git commit -m "phase10: ProbabilisticRedshift reader (interp parameterisation)"
```

---

### Task 6: Converter support for fixed-size list PDF columns

**Files:**
- Modify: `oneuniverse/data/converter.py`
- Test: `test/test_pdf_converter.py`

**Why:** pyarrow infers variable-size list columns from pandas object-dtype columns, which blows up parquet metadata and disallows random slicing. We cast to `FixedSizeList[f4, n_components]` ahead of write so DatasetView reads them back cheaply.

- [ ] **Step 1: Failing test**

```python
# test/test_pdf_converter.py
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from oneuniverse.data import schema
from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data.converter import convert_survey
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import get_manifest
from oneuniverse.data.pdf import PdfSpec, ProbabilisticRedshift
from test.fixtures.pdf_catalog import make_gaussian_pdf_catalog


class _PdfLoader(BaseSurveyLoader):
    config = SurveyConfig(
        name="pdf_fake",
        survey_type="photometric",
        description="test photo-z PDFs",
        column_groups=["core", "probabilistic_redshift"],
    )

    def __init__(self, df, grid):
        self._df = df
        self._grid = grid

    def _load_raw(self, data_path=None, **kwargs):
        df = self._df.copy()
        # Core cols the converter requires
        df["z"] = df["z_pdf_mean"]
        df["z_type"] = np.array(["phot"] * len(df), dtype="U8")
        df["z_err"] = df["z_pdf_std"]
        df["galaxy_id"] = np.arange(len(df), dtype=np.int64)
        df["survey_id"] = np.array(["pdf_fake"] * len(df), dtype="U32")
        df["_original_row_index"] = np.arange(len(df), dtype=np.int64)
        return df


def test_converter_writes_fixed_size_list_for_pdf(tmp_path):
    df, grid = make_gaussian_pdf_catalog(n_rows=200, n_grid=101, seed=2)
    spec = PdfSpec(
        parameterisation="interp", n_components=len(grid),
        grid=list(grid), grid_kind="z",
    )
    loader = _PdfLoader(df, grid)

    out = tmp_path / "pdf_fake"
    convert_survey(
        loader=loader, out_dir=out, overwrite=True,
        pdf_spec=spec,               # new kwarg
    )

    # Manifest has pdf_spec
    manifest = get_manifest(out)
    assert manifest.pdf_spec == spec

    # Parquet uses fixed-size list
    parquet_files = list(out.rglob("*.parquet"))
    assert parquet_files
    pa_schema = pq.read_schema(parquet_files[0])
    field = pa_schema.field("z_pdf_values")
    assert str(field.type).startswith("fixed_size_list"), str(field.type)

    # DatasetView round-trips PDFs and the reader recovers the means.
    from oneuniverse.data.dataset_view import DatasetView
    dv = DatasetView(out)
    df_read = dv.to_pandas()
    pz = ProbabilisticRedshift.from_dataframe(df_read, manifest.pdf_spec)
    recovered = pz.mean()
    np.testing.assert_allclose(
        recovered, df_read["z_pdf_mean"].to_numpy(), atol=1e-2,
    )
```

Run: `pytest test/test_pdf_converter.py -v` → FAIL.

- [ ] **Step 2: Add `pdf_spec` kwarg to `convert_survey`**

Find `def convert_survey(` in `converter.py`. Add `pdf_spec: Optional[PdfSpec] = None` (import at top). Thread it through `write_ouf_dataset` to the `Manifest(...)` construction. Inside `_write_partitions` (and `_write_partitions_by_healpix` if used for POINT), **before** `pa.Table.from_pandas(chunk, ...)`, call:

```python
    chunk = _coerce_pdf_columns(chunk, pdf_spec)
```

Add helper next to `_default_stats_builder`:

```python
def _coerce_pdf_columns(df, pdf_spec):
    """Return a pyarrow Table with fixed-size list columns for PDF data.

    Called on every converter chunk.  When ``pdf_spec`` is None the
    function is a no-op (returns the DataFrame as-is).  Otherwise the
    PDF columns (``z_pdf_values``, plus ``z_pdf_sigma``/``z_pdf_weights``
    for mixmod) are converted to ``FixedSizeList[f4, n_components]``.

    All other columns are passed through via ``pa.Table.from_pandas``.
    """
    if pdf_spec is None:
        return df
    import pyarrow as pa
    import numpy as np

    n = int(pdf_spec.n_components)
    list_cols = ["z_pdf_values"]
    if pdf_spec.parameterisation == "mixmod":
        list_cols += ["z_pdf_sigma", "z_pdf_weights"]

    scalar = df.drop(columns=[c for c in list_cols if c in df.columns])
    table = pa.Table.from_pandas(scalar, preserve_index=False)

    elt_type = pa.float32()
    fsl_type = pa.list_(elt_type, n)

    for col in list_cols:
        if col not in df.columns:
            continue
        arr = np.stack([np.asarray(r, dtype=np.float32) for r in df[col].to_numpy()])
        if arr.shape[1] != n:
            raise ValueError(
                f"column {col!r}: expected {n} components, got {arr.shape[1]}"
            )
        flat = pa.array(arr.reshape(-1), type=elt_type)
        fsl = pa.FixedSizeListArray.from_arrays(flat, n)
        table = table.append_column(col, fsl)
    return table
```

In `_write_partitions`, detect whether the chunk is already a `pa.Table` (from the helper) or a DataFrame and handle both cases. Keep the DataFrame path unchanged for non-PDF datasets.

- [ ] **Step 3: Run test**

Run: `pytest test/test_pdf_converter.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/data/converter.py test/test_pdf_converter.py
git commit -m "phase10: fixed-size list PDF columns + pdf_spec kwarg on convert_survey"
```

---

### Task 7: DatasetView convenience — `.load_pdf()`

**Files:**
- Modify: `oneuniverse/data/dataset_view.py`
- Test: `test/test_pdf_reader.py` (extend)

**Why:** Callers should not need to pass both `df` and `PdfSpec` manually — `DatasetView` already owns the manifest, so it can hand back a ready-to-use `ProbabilisticRedshift`.

- [ ] **Step 1: Extend failing test**

```python
# test/test_pdf_reader.py (append)
def test_datasetview_load_pdf(tmp_path):
    # Re-use the converter fixture from test_pdf_converter.
    from test.test_pdf_converter import _PdfLoader
    from oneuniverse.data.converter import convert_survey
    from oneuniverse.data.dataset_view import DatasetView

    df, grid = make_gaussian_pdf_catalog(n_rows=32, n_grid=51, seed=3)
    spec = PdfSpec(
        parameterisation="interp", n_components=len(grid),
        grid=list(grid), grid_kind="z",
    )
    out = tmp_path / "pdf2"
    convert_survey(loader=_PdfLoader(df, grid), out_dir=out, overwrite=True,
                   pdf_spec=spec)

    dv = DatasetView(out)
    pz = dv.load_pdf()
    assert len(pz) == 32
    assert pz.spec == spec
```

Run: `pytest test/test_pdf_reader.py::test_datasetview_load_pdf -v` → FAIL.

- [ ] **Step 2: Add method to `DatasetView`**

```python
# oneuniverse/data/dataset_view.py — inside class DatasetView
def load_pdf(self):
    """Return a :class:`ProbabilisticRedshift` bound to this dataset.

    Raises
    ------
    ValueError
        If the manifest has no ``pdf_spec`` (dataset is not probabilistic).
    """
    from oneuniverse.data.pdf import ProbabilisticRedshift
    spec = self.manifest.pdf_spec
    if spec is None:
        raise ValueError(
            f"DatasetView({self.survey_path}): manifest has no pdf_spec; "
            f"dataset is not probabilistic."
        )
    df = self.to_pandas()
    return ProbabilisticRedshift.from_dataframe(df, spec)
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_pdf_reader.py::test_datasetview_load_pdf -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/data/dataset_view.py test/test_pdf_reader.py
git commit -m "phase10: DatasetView.load_pdf convenience"
```

---

### Task 8: `quant` and `mixmod` parameterisations

**Files:**
- Modify: `oneuniverse/data/pdf.py`
- Test: `test/test_pdf_reader.py` (extend)

**Why:** `quant` is the Rubin/LSST DP convention; `mixmod` is the Euclid/BPZ convention. Both need `mean`/`std`/`cdf`/`ppf`/`sample` to work so downstream code stays parameterisation-agnostic.

- [ ] **Step 1: Failing tests**

```python
# test/test_pdf_reader.py (append)
def test_quant_reader_recovers_moments():
    rng = np.random.default_rng(0)
    n = 10
    mu = rng.uniform(0.2, 1.8, size=n)
    sigma = rng.uniform(0.02, 0.08, size=n)
    levels = np.linspace(0.01, 0.99, 41)
    from scipy.stats import norm
    qvals = np.stack([norm.ppf(levels, loc=mu[i], scale=sigma[i]) for i in range(n)])
    spec = PdfSpec(
        parameterisation="quant", n_components=len(levels),
        grid=None, grid_kind="quantile",
        quant_levels=list(levels.astype(float)),
    )
    pz = ProbabilisticRedshift(spec, values=qvals, grid=np.asarray(levels))
    np.testing.assert_allclose(pz.mean(), mu, atol=5e-2)
    np.testing.assert_allclose(pz.std(), sigma, rtol=0.2)


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
```

Run: `pytest test/test_pdf_reader.py -v` → FAIL.

- [ ] **Step 2: Extend reader**

Replace the `if spec.parameterisation != "interp"` guard with per-parameterisation storage. Add:

```python
# inside ProbabilisticRedshift.__init__ — replace the guard with:
if spec.parameterisation == "interp":
    # values: (n_rows, n_grid) p(z) samples
    if values.ndim != 2 or values.shape[1] != len(grid):
        raise ValueError("interp: values/grid mismatch")
    self.values = np.asarray(values, dtype=np.float64)
    self.grid = np.asarray(grid, dtype=np.float64)
    self._mixmod = None
elif spec.parameterisation == "quant":
    # values: (n_rows, n_levels) z(q) at the common quant_levels
    if spec.quant_levels is None:
        raise ValueError("quant: spec.quant_levels required")
    self.values = np.asarray(values, dtype=np.float64)
    self.grid = np.asarray(spec.quant_levels, dtype=np.float64)
    self._mixmod = None
elif spec.parameterisation == "mixmod":
    # values ignored; use from_mixmod instead
    self.values = np.asarray(values, dtype=np.float64)
    self.grid = np.asarray(grid, dtype=np.float64)
    self._mixmod = None
else:
    raise ValueError(f"unsupported parameterisation {spec.parameterisation!r}")

self.spec = spec


@classmethod
def from_mixmod(cls, spec, mu, sigma, w):
    obj = cls.__new__(cls)
    obj.spec = spec
    obj._mixmod = (np.asarray(mu, float), np.asarray(sigma, float), np.asarray(w, float))
    obj.values = obj._mixmod[0]
    obj.grid = np.arange(spec.n_components, dtype=np.float64)
    return obj
```

Override the moment/CDF/PPF/sample methods with per-parameterisation branches (keep `interp` as the primary path; `quant` uses trapezoid rule on `grid=levels`, `values=z(q)`; `mixmod` has closed-form mean/var and samples component-by-component). Full per-branch code:

```python
def mean(self):
    if self.spec.parameterisation == "interp":
        dz = self.grid[1] - self.grid[0]
        return (self.values * self.grid[None, :]).sum(axis=1) * dz
    if self.spec.parameterisation == "quant":
        # E[z] = ∫ z(q) dq via trapezoid over the common levels
        return np.trapz(self.values, self.grid, axis=1)
    mu, sigma, w = self._mixmod
    return (w * mu).sum(axis=1)


def std(self):
    if self.spec.parameterisation == "interp":
        dz = self.grid[1] - self.grid[0]
        m = self.mean()
        var = (self.values * (self.grid[None, :] - m[:, None]) ** 2).sum(axis=1) * dz
        return np.sqrt(np.maximum(var, 0.0))
    if self.spec.parameterisation == "quant":
        m = self.mean()
        var = np.trapz((self.values - m[:, None]) ** 2, self.grid, axis=1)
        return np.sqrt(np.maximum(var, 0.0))
    mu, sigma, w = self._mixmod
    m = self.mean()
    second = (w * (mu ** 2 + sigma ** 2)).sum(axis=1)
    return np.sqrt(np.maximum(second - m ** 2, 0.0))


def cdf(self):
    if self.spec.parameterisation == "interp":
        dz = self.grid[1] - self.grid[0]
        c = np.cumsum(self.values, axis=1) * dz
        c /= np.maximum(c[:, -1:], 1e-300)
        return c
    if self.spec.parameterisation == "quant":
        # The *grid* here is already the quantile levels → cdf is defined
        # by (values, grid) inversely; callers use ppf.
        return np.broadcast_to(self.grid, self.values.shape).copy()
    raise NotImplementedError("mixmod.cdf requires scipy.stats; deferred")


def ppf(self, q):
    q = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if self.spec.parameterisation == "interp":
        c = self.cdf()
        out = np.empty((c.shape[0], q.size), dtype=np.float64)
        for i in range(c.shape[0]):
            out[i] = np.interp(q, c[i], self.grid)
        return out[:, 0] if q.size == 1 else out
    if self.spec.parameterisation == "quant":
        # Direct interpolation of z(q).
        out = np.empty((self.values.shape[0], q.size), dtype=np.float64)
        for i in range(self.values.shape[0]):
            out[i] = np.interp(q, self.grid, self.values[i])
        return out[:, 0] if q.size == 1 else out
    raise NotImplementedError("mixmod.ppf requires scipy.stats; deferred")


def sample(self, n_per, seed=None):
    rng = np.random.default_rng(seed)
    if self.spec.parameterisation in ("interp", "quant"):
        q = rng.uniform(0.0, 1.0, size=(self.values.shape[0], n_per))
        if self.spec.parameterisation == "interp":
            c = self.cdf()
            out = np.empty_like(q)
            for i in range(self.values.shape[0]):
                out[i] = np.interp(q[i], c[i], self.grid)
            return out
        out = np.empty_like(q)
        for i in range(self.values.shape[0]):
            out[i] = np.interp(q[i], self.grid, self.values[i])
        return out
    mu, sigma, w = self._mixmod
    n_rows, K = mu.shape
    out = np.empty((n_rows, n_per), dtype=np.float64)
    for i in range(n_rows):
        comp = rng.choice(K, size=n_per, p=w[i])
        out[i] = rng.normal(mu[i, comp], sigma[i, comp])
    return out
```

- [ ] **Step 3: Run full reader suite**

Run: `pytest test/test_pdf_reader.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/data/pdf.py test/test_pdf_reader.py
git commit -m "phase10: quant + mixmod ProbabilisticRedshift parameterisations"
```

---

### Task 9: PDF-aware weights in `combine/weights`

**Files:**
- Create: `oneuniverse/combine/weights/pdf.py`
- Modify: `oneuniverse/combine/weights/__init__.py`
- Modify: `oneuniverse/combine/weights/registry.py`
- Test: `test/test_pdf_weights.py`

**Why:** Probabilistic datasets still need to flow through `WeightedCatalog.fill_defaults`. Without a `("photometric", "phot_pdf")` default, callers must construct weights manually. Also: an inverse-variance weight over the PDF width is a cheap, physically motivated default.

- [ ] **Step 1: Failing test**

```python
# test/test_pdf_weights.py
import numpy as np
import pandas as pd

from oneuniverse.combine.weights import (
    PdfWidthIVarWeight, PdfMeanRedshiftWeight, default_weight_for,
)


def _df():
    return pd.DataFrame({
        "z_pdf_mean": np.array([0.3, 0.5, 0.9], dtype=np.float32),
        "z_pdf_std": np.array([0.04, 0.08, 0.02], dtype=np.float32),
    })


def test_pdf_width_ivar():
    w = PdfWidthIVarWeight(std_column="z_pdf_std")
    got = w(_df())
    np.testing.assert_allclose(got, 1.0 / _df()["z_pdf_std"].to_numpy(dtype=np.float64) ** 2)


def test_pdf_mean_redshift_weight_reads_mean():
    w = PdfMeanRedshiftWeight(mean_column="z_pdf_mean")
    got = w(_df())
    np.testing.assert_allclose(got, _df()["z_pdf_mean"].to_numpy(dtype=np.float64))


def test_default_weight_for_phot_pdf_registered():
    w = default_weight_for("photometric", "phot_pdf")
    got = w(_df())
    assert got.shape == (3,)
```

Run: `pytest test/test_pdf_weights.py -v` → FAIL.

- [ ] **Step 2: Implement weights + register default**

```python
# oneuniverse/combine/weights/pdf.py
"""PDF-aware per-object weights."""
from __future__ import annotations

import numpy as np
import pandas as pd

from oneuniverse.combine.weights.base import Weight


class PdfWidthIVarWeight(Weight):
    """``w = 1 / z_pdf_std**2`` — objective IVar on the PDF width."""

    def __init__(self, std_column: str = "z_pdf_std", name: str = "ivar(pdf_std)"):
        self.std_column = std_column
        self.name = name

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        if self.std_column not in df.columns:
            raise KeyError(f"PdfWidthIVarWeight: missing '{self.std_column}'")
        s = df[self.std_column].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 0, 1.0 / (s * s), 0.0)


class PdfMeanRedshiftWeight(Weight):
    """Pass-through of the PDF's first moment (``z_pdf_mean``) as a weight.

    Useful when downstream code wants to multiply by ``<z>`` (e.g. as a
    radial weight); stays in the weight system so composition rules are
    uniform.
    """

    def __init__(self, mean_column: str = "z_pdf_mean", name: str = "pdf_mean"):
        self.mean_column = mean_column
        self.name = name

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_column not in df.columns:
            raise KeyError(f"PdfMeanRedshiftWeight: missing '{self.mean_column}'")
        return df[self.mean_column].to_numpy(dtype=np.float64)
```

```python
# oneuniverse/combine/weights/registry.py — add factory + entry
def _ivar_pdf_width() -> Weight:
    from oneuniverse.combine.weights.pdf import PdfWidthIVarWeight
    return PdfWidthIVarWeight(std_column="z_pdf_std")

_DEFAULTS: Mapping[Key, Factory] = MappingProxyType({
    ("spectroscopic", "spec"): _ivar_spec,
    ("photometric", "phot"): _ivar_phot,
    ("peculiar_velocity", "pec"): _ivar_pec,
    ("photometric", "phot_pdf"): _ivar_pdf_width,
})
```

```python
# oneuniverse/combine/weights/__init__.py — extend exports
from oneuniverse.combine.weights.pdf import (
    PdfMeanRedshiftWeight, PdfWidthIVarWeight,
)

__all__ = [
    "Weight", "ProductWeight", "ConstantWeight", "ColumnWeight",
    "InverseVarianceWeight", "FKPWeight", "QualityMaskWeight",
    "PdfMeanRedshiftWeight", "PdfWidthIVarWeight",
    "default_weight_for",
]
```

- [ ] **Step 3: Run test**

Run: `pytest test/test_pdf_weights.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/combine/weights/pdf.py \
        oneuniverse/combine/weights/__init__.py \
        oneuniverse/combine/weights/registry.py \
        test/test_pdf_weights.py
git commit -m "phase10: PdfWidthIVar + PdfMeanRedshift weights; phot_pdf default"
```

---

### Task 10: Diagnostic figures + z_type="phot_pdf" validation

**Files:**
- Create: `test/test_visual_pdf.py`
- Modify: `oneuniverse/data/schema.py` — extend `Z_TYPE_VALUES`

**Why:** Visual testing is required for data-infrastructure work (user standing rule). Produces a 3-panel figure — PDF overlays, CDF round-trip, recovered-vs-input mean scatter — so any regression is obvious.

- [ ] **Step 1: Extend `Z_TYPE_VALUES`**

```python
# oneuniverse/data/schema.py — change
Z_TYPE_VALUES: Tuple[str, ...] = ("spec", "phot", "phot_pdf", "pv", "none")
```

- [ ] **Step 2: Failing visual test**

```python
# test/test_visual_pdf.py
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oneuniverse.data.converter import convert_survey
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.pdf import PdfSpec
from test.fixtures.pdf_catalog import make_gaussian_pdf_catalog
from test.test_pdf_converter import _PdfLoader


OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_phase10_visual_end_to_end(tmp_path):
    df, grid = make_gaussian_pdf_catalog(n_rows=300, n_grid=201, seed=5)
    spec = PdfSpec(
        parameterisation="interp", n_components=len(grid),
        grid=list(grid), grid_kind="z",
    )
    out = tmp_path / "pdf_viz"
    convert_survey(loader=_PdfLoader(df, grid), out_dir=out, overwrite=True,
                   pdf_spec=spec)

    dv = DatasetView(out)
    pz = dv.load_pdf()
    df_read = dv.to_pandas()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    # (a) a handful of PDFs
    for i in range(5):
        ax[0].plot(pz.grid, pz.values[i], alpha=0.7, lw=1)
    ax[0].set_xlabel("z"); ax[0].set_ylabel("p(z)"); ax[0].set_title("5 example photo-z PDFs")

    # (b) CDF monotonicity sanity check
    cdf = pz.cdf()
    ax[1].plot(pz.grid, cdf[:20].T, color="tab:blue", alpha=0.3, lw=0.6)
    ax[1].set_xlabel("z"); ax[1].set_ylabel("CDF(z)"); ax[1].set_title("CDFs (first 20)")

    # (c) recovered mean vs. input z_true
    ax[2].scatter(df_read["z_true"], pz.mean(), s=8, alpha=0.6)
    zmin = min(df_read["z_true"].min(), pz.mean().min())
    zmax = max(df_read["z_true"].max(), pz.mean().max())
    ax[2].plot([zmin, zmax], [zmin, zmax], "k--", lw=0.8)
    ax[2].set_xlabel("z_true"); ax[2].set_ylabel("<z> from PDF")
    ax[2].set_title("PDF mean vs input truth")
    fig.tight_layout()
    fig.savefig(OUT / "phase10_pdf_overview.png", dpi=110)
    plt.close(fig)
```

Run: `pytest test/test_visual_pdf.py -v` → PASS. Inspect `test_output/phase10_pdf_overview.png`.

- [ ] **Step 3: Commit**

```bash
git add oneuniverse/data/schema.py test/test_visual_pdf.py
git commit -m "phase10: phot_pdf z_type + visual end-to-end test"
```

---

### Task 11: Close Phase 10

**Why:** Mirrors every earlier phase closeout: plans/README updated, memory pointer refreshed, test count recorded.

- [ ] **Step 1: Run full test suite**

Run: `pytest` from `Packages/oneuniverse` → all green; record count.

- [ ] **Step 2: Update `plans/README.md`**

Add row:

```
| 10 | Probabilistic redshifts (photo-z PDFs, FixedSizeList parquet columns, PdfSpec manifest, ProbabilisticRedshift reader, PDF-aware weights) | **complete (YYYY-MM-DD, N/N tests green)** |
```

- [ ] **Step 3: Update memory file `project_oneuniverse_stabilisation.md`**

Append a "Phase 10 complete" block summarising: `PdfSpec`, `ProbabilisticRedshift` (interp/quant/mixmod), `z_pdf_*` schema, `DatasetView.load_pdf()`, `PdfWidthIVarWeight`, `default_weight_for("photometric", "phot_pdf")`.

- [ ] **Step 4: Final commit**

```bash
git add plans/README.md
git commit -m "phase10: close-out — update plans/README"
```
