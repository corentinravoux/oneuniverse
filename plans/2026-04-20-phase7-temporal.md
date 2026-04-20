# Phase 7 — Temporal Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional epoch-of-observation support to POINT datasets and add a new `LIGHTCURVE` geometry, both pushed cleanly through the OUF 2.x manifest, partition-stats, and `DatasetView` query stack.

**Architecture:** Extend `Manifest` with an optional `temporal: TemporalSpec` block; extend `PartitionStats` with `t_min`/`t_max`; extend `DatasetView.scan(t_range=…)`; add `DataGeometry.LIGHTCURVE` with object-table + epoch-partition layout mirroring `SIGHTLINE`; ship a `write_ouf_lightcurve_dataset` writer and reader support.

**Tech Stack:** Python 3.10+, dataclasses, pyarrow/pyarrow.dataset, pandas, pytest, healpy, numpy, matplotlib (for the diagnostic).

**Pre-reading:**
- `oneuniverse/data/format_spec.py` — geometry enum, required column tables.
- `oneuniverse/data/manifest.py` — `Manifest`, `PartitionSpec`, `PartitionStats`, `PartitioningSpec`.
- `oneuniverse/data/dataset_view.py` — `_select_partitions`, `_range_expr`, `_spatial_expr`.
- `oneuniverse/data/converter.py` — the writer that produces OUF POINT datasets (for pattern reference, especially partition-stats collection).

**Format version bump:** OUF format `2.0.0` → `2.1.0`; schema `2.0.0` → `2.1.0`. Additive; the reader accepts both.

---

## File Structure

**Create:**
- `oneuniverse/data/temporal.py` — `TemporalSpec` frozen dataclass + small helpers.
- `oneuniverse/data/_converter_lightcurve.py` — `write_ouf_lightcurve_dataset()`.
- `test/test_temporal_spec.py` — dataclass tests.
- `test/test_partition_stats_time.py` — partition-stats time fields + range pruning.
- `test/test_dataset_view_time.py` — `scan(t_range=…)` pruning + pushdown.
- `test/test_lightcurve_geometry.py` — geometry enum, required columns, manifest validation.
- `test/test_lightcurve_roundtrip.py` — write then read a synthetic LIGHTCURVE.
- `test/test_visual_temporal.py` — diagnostic figure (skymap + MJD hist + lightcurve peek).

**Modify:**
- `oneuniverse/data/format_spec.py` — add `DataGeometry.LIGHTCURVE`, `LIGHTCURVE_OBJECT_REQUIRED_COLUMNS`, `LIGHTCURVE_DATA_REQUIRED_COLUMNS`, `DEFAULT_PARTITION_ROWS[LIGHTCURVE]`, bump `FORMAT_VERSION`/`SCHEMA_VERSION` to `2.1.0`.
- `oneuniverse/data/manifest.py` — add `TemporalSpec` import, `PartitionStats.t_min/t_max`, `Manifest.temporal: Optional[TemporalSpec]`, version-compatibility rule (`2.x`).
- `oneuniverse/data/dataset_view.py` — add `t_range` parameter on `scan`/`read`, extend `_select_partitions` + `_range_expr`, add `objects_table()` method covering both SIGHTLINE and LIGHTCURVE.
- `oneuniverse/data/converter.py` — when the input DataFrame has a `t_obs` column for a POINT survey, fill `PartitionStats.t_min/t_max` and attach a `TemporalSpec` to the manifest.
- `oneuniverse/data/__init__.py` — export `TemporalSpec`, `write_ouf_lightcurve_dataset`.

---

## Conventions and shared snippets

Each task ends with a commit step. Commit messages follow the repo's existing style (seen in Phase-6 commits `f7f1281`, `211eec9`): short imperative subject, optional body.

All new tests live in `Packages/oneuniverse/test/`. All `pytest` invocations assume the working directory is `Packages/oneuniverse/`.

**Shared synthetic fixture** (used by several tests — define it once in each test file that needs it; do not add a conftest fixture unless you're already editing conftest for another reason):

```python
import numpy as np
import pandas as pd

def _make_transient_point_df(n: int = 1000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mjd = rng.uniform(58000.0, 60000.0, size=n)   # 2017-09-04 .. 2023-02-25
    return pd.DataFrame({
        "ra":    rng.uniform(0.0, 360.0, size=n),
        "dec":   rng.uniform(-60.0, 60.0, size=n),
        "z":     rng.uniform(0.01, 0.5, size=n),
        "z_type": np.array(["spec"] * n, dtype=object),
        "z_err":  rng.uniform(1e-4, 1e-3, size=n),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.array([f"syn{i:05d}" for i in range(n)], dtype=object),
        "_original_row_index": np.arange(n, dtype=np.int64),
        "t_obs": mjd,
    })
```

---

## Task 1: `TemporalSpec` dataclass

**Files:**
- Create: `oneuniverse/data/temporal.py`
- Test: `test/test_temporal_spec.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_temporal_spec.py
import pytest
from oneuniverse.data.temporal import TemporalSpec


def test_temporal_spec_defaults():
    spec = TemporalSpec(t_min=58000.0, t_max=60000.0)
    assert spec.time_column == "t_obs"
    assert spec.time_reference == "TDB"
    assert spec.time_unit == "MJD"
    assert spec.cadence is None


def test_temporal_spec_frozen():
    spec = TemporalSpec(t_min=0.0, t_max=1.0)
    with pytest.raises((AttributeError, TypeError)):
        spec.t_min = 2.0  # type: ignore[misc]


def test_temporal_spec_validates_range():
    with pytest.raises(ValueError, match="t_min"):
        TemporalSpec(t_min=10.0, t_max=5.0)


def test_temporal_spec_validates_time_reference():
    with pytest.raises(ValueError, match="time_reference"):
        TemporalSpec(t_min=0.0, t_max=1.0, time_reference="GARBAGE")


def test_temporal_spec_roundtrip_dict():
    spec = TemporalSpec(t_min=58000.0, t_max=60000.0, cadence=7.0)
    as_dict = spec.to_dict()
    back = TemporalSpec.from_dict(as_dict)
    assert back == spec
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest test/test_temporal_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: oneuniverse.data.temporal`.

- [ ] **Step 3: Implement `TemporalSpec`**

```python
# oneuniverse/data/temporal.py
"""Temporal metadata block attached to OUF 2.1 manifests."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

_ALLOWED_TIME_REFERENCES = frozenset({"TDB", "TAI", "UTC", "TT"})


@dataclass(frozen=True)
class TemporalSpec:
    """Describes the time axis carried by a temporal dataset.

    Attributes
    ----------
    t_min, t_max : float
        Inclusive bounds of the epoch column across the full dataset, in
        ``time_unit``.
    time_column : str
        Name of the column holding the epoch. Default ``"t_obs"``.
    time_unit : str
        Default ``"MJD"``. Free-form — readers that do not recognise a
        unit should raise rather than guess.
    time_reference : str
        One of ``"TDB"``, ``"TAI"``, ``"UTC"``, ``"TT"``. Default ``"TDB"``.
    cadence : float or None
        Nominal cadence in the same unit as ``time_unit``. ``None`` for
        irregularly-sampled data.
    """

    t_min: float
    t_max: float
    time_column: str = "t_obs"
    time_unit: str = "MJD"
    time_reference: str = "TDB"
    cadence: Optional[float] = None

    def __post_init__(self) -> None:
        if self.t_max < self.t_min:
            raise ValueError(
                f"TemporalSpec: t_min={self.t_min} > t_max={self.t_max}"
            )
        if self.time_reference not in _ALLOWED_TIME_REFERENCES:
            raise ValueError(
                f"TemporalSpec: time_reference={self.time_reference!r} "
                f"not in {sorted(_ALLOWED_TIME_REFERENCES)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TemporalSpec":
        return cls(**raw)
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_temporal_spec.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/temporal.py test/test_temporal_spec.py
git commit -m "feat(temporal): add TemporalSpec dataclass for OUF 2.1 manifests"
```

---

## Task 2: `PartitionStats` time fields

**Files:**
- Modify: `oneuniverse/data/manifest.py`
- Test: `test/test_partition_stats_time.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_partition_stats_time.py
from pathlib import Path

from oneuniverse.data.manifest import (
    ColumnSpec, LoaderSpec, Manifest, OriginalFileSpec,
    PartitionSpec, PartitionStats, write_manifest, read_manifest,
)
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.temporal import TemporalSpec


def _minimal_manifest(tmp_path: Path, stats: PartitionStats, temporal=None) -> Path:
    m = Manifest(
        oneuniverse_format_version="2.1.0",
        oneuniverse_schema_version="2.1.0",
        geometry=DataGeometry.POINT,
        survey_name="syn",
        survey_type="transient",
        created_utc="2026-04-20T00:00:00Z",
        original_files=[OriginalFileSpec(
            path="src.csv", sha256="0" * 16, n_rows=None,
            size_bytes=1, format="csv",
        )],
        partitions=[PartitionSpec(
            name="part_0000.parquet", n_rows=10,
            sha256="f" * 16, size_bytes=100, stats=stats,
        )],
        partitioning=None,
        schema=[ColumnSpec(name="t_obs", dtype="float64")],
        conversion_kwargs={},
        loader=LoaderSpec(name="syn", version="0"),
        temporal=temporal,
    )
    out = tmp_path / "manifest.json"
    write_manifest(out, m)
    return out


def test_partition_stats_accepts_time_fields():
    stats = PartitionStats(t_min=58000.0, t_max=60000.0)
    assert stats.t_min == 58000.0
    assert stats.t_max == 60000.0


def test_partition_stats_time_defaults_none():
    stats = PartitionStats()
    assert stats.t_min is None
    assert stats.t_max is None


def test_manifest_roundtrip_preserves_time_stats(tmp_path):
    stats = PartitionStats(ra_min=0, ra_max=10, t_min=58000.0, t_max=60000.0)
    path = _minimal_manifest(tmp_path, stats)
    m = read_manifest(path)
    assert m.partitions[0].stats.t_min == 58000.0
    assert m.partitions[0].stats.t_max == 60000.0


def test_manifest_roundtrip_preserves_temporal_spec(tmp_path):
    temporal = TemporalSpec(t_min=58000.0, t_max=60000.0, cadence=3.0)
    path = _minimal_manifest(tmp_path, PartitionStats(), temporal=temporal)
    m = read_manifest(path)
    assert m.temporal == temporal
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest test/test_partition_stats_time.py -v`
Expected: FAIL — `PartitionStats` has no `t_min`/`t_max`; `Manifest` has no `temporal`.

- [ ] **Step 3: Extend `PartitionStats` and `Manifest`**

Modify `oneuniverse/data/manifest.py`:

```python
# at top, alongside existing imports
from oneuniverse.data.temporal import TemporalSpec

# replace the existing PartitionStats block
@dataclass(frozen=True)
class PartitionStats:
    ra_min: Optional[float] = None
    ra_max: Optional[float] = None
    dec_min: Optional[float] = None
    dec_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    t_min: Optional[float] = None
    t_max: Optional[float] = None

# in the Manifest dataclass, add as the LAST field (after `extra`)
    temporal: Optional[TemporalSpec] = None
```

Update `_from_dict` in the same file (the `Manifest(...)` constructor call at the bottom):

```python
# before the final return Manifest(...), decode temporal
temporal_raw = raw.get("temporal")
temporal = TemporalSpec.from_dict(temporal_raw) if temporal_raw else None

# then append temporal=temporal, to the Manifest(**...) call
```

Update `_to_dict` so that `TemporalSpec` serialises cleanly — `asdict`
already handles nested dataclasses; the only change needed is to keep
`temporal` as `None` (not an empty dict) when absent:

```python
def _to_dict(m: Manifest) -> Dict[str, Any]:
    d = asdict(m)
    d["geometry"] = m.geometry.value
    if m.temporal is None:
        d["temporal"] = None
    return d
```

Update the format-version compatibility rule:

```python
if not (isinstance(fmt, str) and (fmt.startswith("2.0") or fmt.startswith("2.1"))):
    raise ManifestValidationError(
        f"{path}: oneuniverse_format_version={fmt!r} is not compatible "
        f"with this library (expected 2.0.x or 2.1.x)."
    )
```

Bump the `FORMAT_VERSION` and `SCHEMA_VERSION` module constants at the
top of `manifest.py` to `"2.1.0"`.

- [ ] **Step 4: Run tests**

Run: `pytest test/test_partition_stats_time.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full suite, confirm no regressions**

Run: `pytest -q`
Expected: all pre-existing tests still pass; the `2.0.x` datasets they
create continue to be accepted because the reader matches `2.0` OR `2.1`.
Any existing test that compared the format string to `"2.0.0"` literally
should be updated to `.startswith("2.")` — fix in place if encountered.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/manifest.py test/test_partition_stats_time.py
git commit -m "feat(temporal): partition-stats t_min/t_max, Manifest.temporal, bump to 2.1.0"
```

---

## Task 3: `DatasetView.scan(t_range=…)` partition pruning

**Files:**
- Modify: `oneuniverse/data/dataset_view.py`
- Test: `test/test_dataset_view_time.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_dataset_view_time.py
import datetime as dt
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.manifest import (
    ColumnSpec, LoaderSpec, Manifest, OriginalFileSpec,
    PartitionSpec, PartitionStats, write_manifest,
)
from oneuniverse.data.format_spec import DataGeometry, ONEUNIVERSE_SUBDIR
from oneuniverse.data.temporal import TemporalSpec


def _write_point_partition(path: Path, df: pd.DataFrame) -> None:
    import pyarrow as pa
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def _build_dataset(tmp_path: Path, partitions: list[tuple[str, pd.DataFrame]]) -> Path:
    survey_dir = tmp_path / "syn"
    ou_dir = survey_dir / ONEUNIVERSE_SUBDIR
    ou_dir.mkdir(parents=True)
    specs = []
    for name, df in partitions:
        _write_point_partition(ou_dir / name, df)
        specs.append(PartitionSpec(
            name=name, n_rows=len(df), sha256="0" * 16,
            size_bytes=(ou_dir / name).stat().st_size,
            stats=PartitionStats(
                t_min=float(df["t_obs"].min()),
                t_max=float(df["t_obs"].max()),
            ),
        ))
    m = Manifest(
        oneuniverse_format_version="2.1.0",
        oneuniverse_schema_version="2.1.0",
        geometry=DataGeometry.POINT,
        survey_name="syn", survey_type="transient",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        original_files=[OriginalFileSpec(
            path="src.csv", sha256="0" * 16, n_rows=None, size_bytes=1, format="csv",
        )],
        partitions=specs,
        partitioning=None,
        schema=[ColumnSpec(name=c, dtype=str(partitions[0][1][c].dtype)) for c in partitions[0][1].columns],
        conversion_kwargs={},
        loader=LoaderSpec(name="syn", version="0"),
        temporal=TemporalSpec(
            t_min=min(p.stats.t_min for p in specs),
            t_max=max(p.stats.t_max for p in specs),
        ),
    )
    write_manifest(ou_dir / "manifest.json", m)
    return survey_dir


def test_t_range_prunes_partition_by_stats(tmp_path):
    early = pd.DataFrame({"ra": [0.0], "dec": [0.0], "t_obs": [58000.0]})
    late = pd.DataFrame({"ra": [0.0], "dec": [0.0], "t_obs": [60000.0]})
    root = _build_dataset(tmp_path, [("part_0000.parquet", early), ("part_0001.parquet", late)])

    view = DatasetView.from_path(root)
    kept = view._select_partitions(t_range=(59500.0, 61000.0))
    assert [p.name for p in kept] == ["part_0001.parquet"]


def test_t_range_pushdown_filters_rows(tmp_path):
    df = pd.DataFrame({
        "ra": [0.0, 0.0, 0.0],
        "dec": [0.0, 0.0, 0.0],
        "t_obs": [58000.0, 59000.0, 60000.0],
    })
    root = _build_dataset(tmp_path, [("part_0000.parquet", df)])
    view = DatasetView.from_path(root)
    out = view.read(columns=["t_obs"], t_range=(58500.0, 59500.0))
    assert list(out["t_obs"]) == [59000.0]


def test_t_range_none_returns_all(tmp_path):
    df = pd.DataFrame({
        "ra": [0.0, 0.0],
        "dec": [0.0, 0.0],
        "t_obs": [58000.0, 60000.0],
    })
    root = _build_dataset(tmp_path, [("part_0000.parquet", df)])
    view = DatasetView.from_path(root)
    out = view.read(columns=["t_obs"])
    assert len(out) == 2
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest test/test_dataset_view_time.py -v`
Expected: FAIL — `scan` does not know `t_range`.

- [ ] **Step 3: Extend `_select_partitions`**

In `oneuniverse/data/dataset_view.py`, modify `_select_partitions`:

```python
def _select_partitions(
    self,
    *,
    ra_range: Optional[Range] = None,
    dec_range: Optional[Range] = None,
    z_range: Optional[Range] = None,
    t_range: Optional[Range] = None,
    healpix_cells: Optional[Iterable[int]] = None,
) -> List[PartitionSpec]:
    cell_filter = (
        {int(c) for c in healpix_cells}
        if healpix_cells is not None else None
    )
    keep: List[PartitionSpec] = []
    for part in self.manifest.partitions:
        if (
            cell_filter is not None
            and part.healpix_cell is not None
            and part.healpix_cell not in cell_filter
        ):
            continue
        if not _range_overlaps(ra_range,  part.stats.ra_min,  part.stats.ra_max):
            continue
        if not _range_overlaps(dec_range, part.stats.dec_min, part.stats.dec_max):
            continue
        if not _range_overlaps(z_range,   part.stats.z_min,   part.stats.z_max):
            continue
        if not _range_overlaps(t_range,   part.stats.t_min,   part.stats.t_max):
            continue
        keep.append(part)
    return keep
```

- [ ] **Step 4: Extend `_range_expr` to include `t_range`**

Replace `_range_expr` so it covers the temporal column too. The
dataset's own time-column name lives in `manifest.temporal.time_column`;
pass it in from the caller.

```python
def _range_expr(
    base: Optional[pc.Expression],
    ra_range: Optional[Range],
    dec_range: Optional[Range],
    z_range: Optional[Range],
    t_range: Optional[Range] = None,
    time_column: Optional[str] = None,
) -> Optional[pc.Expression]:
    parts: List[pc.Expression] = []
    if base is not None:
        parts.append(base)
    cols: List[Tuple[str, Optional[Range]]] = [
        ("ra", ra_range), ("dec", dec_range), ("z", z_range),
    ]
    if t_range is not None and time_column is not None:
        cols.append((time_column, t_range))
    for col, rng in cols:
        if rng is None:
            continue
        lo, hi = rng
        if lo is not None:
            parts.append(pc.field(col) >= lo)
        if hi is not None:
            parts.append(pc.field(col) <= hi)
    if not parts:
        return None
    expr = parts[0]
    for p in parts[1:]:
        expr = expr & p
    return expr
```

- [ ] **Step 5: Thread `t_range` through `scan` and `read`**

In `DatasetView.scan`, add the `t_range` parameter and pass the time
column:

```python
def scan(
    self,
    columns: Optional[Sequence[str]] = None,
    filter: Optional[pc.Expression] = None,
    *,
    ra_range: Optional[Range] = None,
    dec_range: Optional[Range] = None,
    z_range: Optional[Range] = None,
    t_range: Optional[Range] = None,
    cone: Optional[Cone] = None,
    skypatch: Optional[SkyPatch] = None,
    healpix_cells: Optional[Iterable[int]] = None,
) -> pa.Table:
    cells = self._resolve_cells(cone, skypatch, healpix_cells)
    partitions = self._select_partitions(
        ra_range=ra_range, dec_range=dec_range, z_range=z_range,
        t_range=t_range, healpix_cells=cells,
    )
    dataset = self._build_dataset(partitions)
    if dataset is None:
        return _empty_table(self.columns, columns)
    time_col = (
        self.manifest.temporal.time_column
        if self.manifest.temporal is not None else None
    )
    expr = _range_expr(filter, ra_range, dec_range, z_range, t_range, time_col)
    expr = _spatial_expr(expr, cone, skypatch)
    cols = list(columns) if columns is not None else None
    return dataset.to_table(columns=cols, filter=expr)
```

Mirror the new parameter in `read`:

```python
def read(
    self,
    columns: Optional[Sequence[str]] = None,
    filter: Optional[pc.Expression] = None,
    *,
    ra_range: Optional[Range] = None,
    dec_range: Optional[Range] = None,
    z_range: Optional[Range] = None,
    t_range: Optional[Range] = None,
    cone: Optional[Cone] = None,
    skypatch: Optional[SkyPatch] = None,
    healpix_cells: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    return self.scan(
        columns=columns, filter=filter,
        ra_range=ra_range, dec_range=dec_range, z_range=z_range,
        t_range=t_range,
        cone=cone, skypatch=skypatch, healpix_cells=healpix_cells,
    ).to_pandas()
```

- [ ] **Step 6: Run tests**

Run: `pytest test/test_dataset_view_time.py -v`
Expected: 3 passed.

- [ ] **Step 7: Run full suite**

Run: `pytest -q`
Expected: all previous tests still pass.

- [ ] **Step 8: Commit**

```bash
git add oneuniverse/data/dataset_view.py test/test_dataset_view_time.py
git commit -m "feat(temporal): DatasetView.scan(t_range=...) partition pruning + pushdown"
```

---

## Task 4: Converter writes temporal metadata for POINT

**Files:**
- Modify: `oneuniverse/data/converter.py`
- Test: add two tests to `test/test_dataset_view_time.py`

- [ ] **Step 1: Add the failing test**

Append to `test/test_dataset_view_time.py`:

```python
def test_converter_fills_temporal_when_t_obs_present(tmp_path):
    """convert_survey must populate Manifest.temporal and per-partition
    t_min/t_max when the DataFrame carries a t_obs column."""
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.manifest import read_manifest

    df = _make_transient_point_df(n=500, seed=1)
    from oneuniverse.data.format_spec import DataGeometry, ONEUNIVERSE_SUBDIR
    survey_dir = tmp_path / "syn"
    write_ouf_dataset(
        df, survey_dir,
        survey_name="syn", survey_type="transient",
        geometry=DataGeometry.POINT, loader_name="syn", loader_version="0",
    )
    m = read_manifest(survey_dir / ONEUNIVERSE_SUBDIR / "manifest.json")
    assert m.temporal is not None
    assert m.temporal.time_column == "t_obs"
    assert m.temporal.t_min == float(df["t_obs"].min())
    assert m.temporal.t_max == float(df["t_obs"].max())
    for p in m.partitions:
        assert p.stats.t_min is not None
        assert p.stats.t_max is not None


def test_converter_omits_temporal_when_t_obs_absent(tmp_path):
    """A POINT survey without a t_obs column still produces a valid
    manifest, with temporal=None."""
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.manifest import read_manifest
    from oneuniverse.data.format_spec import DataGeometry, ONEUNIVERSE_SUBDIR

    df = _make_transient_point_df(n=100, seed=2).drop(columns=["t_obs"])
    survey_dir = tmp_path / "syn_notime"
    write_ouf_dataset(
        df, survey_dir,
        survey_name="syn_notime", survey_type="spectroscopic",
        geometry=DataGeometry.POINT, loader_name="syn", loader_version="0",
    )
    m = read_manifest(survey_dir / ONEUNIVERSE_SUBDIR / "manifest.json")
    assert m.temporal is None
    for p in m.partitions:
        assert p.stats.t_min is None
```

Add the shared helper to the top of `test/test_dataset_view_time.py`:

```python
def _make_transient_point_df(n=1000, seed=7):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ra": rng.uniform(0.0, 360.0, size=n),
        "dec": rng.uniform(-60.0, 60.0, size=n),
        "z": rng.uniform(0.01, 0.5, size=n),
        "z_type": np.array(["spec"] * n, dtype=object),
        "z_err": rng.uniform(1e-4, 1e-3, size=n),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.array([f"syn{i:05d}" for i in range(n)], dtype=object),
        "_original_row_index": np.arange(n, dtype=np.int64),
        "t_obs": rng.uniform(58000.0, 60000.0, size=n),
    })
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest test/test_dataset_view_time.py::test_converter_fills_temporal_when_t_obs_present -v`
Expected: FAIL — the converter does not yet fill temporal.

- [ ] **Step 3: Locate the existing partition-stats code**

Run: `grep -n "PartitionStats" oneuniverse/data/converter.py`
Read the surrounding 20 lines to find where ra/dec/z stats are computed
per partition. The extension plugs into the same loop.

- [ ] **Step 4: Wire temporal stats into the converter**

Inside the converter's partition-writing loop, when the geometry is
POINT and the DataFrame contains the column `t_obs`, add:

```python
# in the per-partition stats computation (alongside ra_min/ra_max …)
t_min = float(chunk["t_obs"].min()) if "t_obs" in chunk.columns else None
t_max = float(chunk["t_obs"].max()) if "t_obs" in chunk.columns else None

stats = PartitionStats(
    ra_min=ra_min, ra_max=ra_max,
    dec_min=dec_min, dec_max=dec_max,
    z_min=z_min, z_max=z_max,
    t_min=t_min, t_max=t_max,
)
```

After the partition loop, when building the final `Manifest`, add:

```python
temporal = None
if "t_obs" in df.columns and geometry is DataGeometry.POINT:
    temporal = TemporalSpec(
        t_min=float(df["t_obs"].min()),
        t_max=float(df["t_obs"].max()),
    )

manifest = Manifest(
    ...,
    temporal=temporal,
)
```

Add the matching imports at the top of `converter.py`:

```python
from oneuniverse.data.temporal import TemporalSpec
```

- [ ] **Step 5: Run tests**

Run: `pytest test/test_dataset_view_time.py -v`
Expected: all 5 tests pass (the 3 from Task 3 + the 2 added here).

- [ ] **Step 6: Full suite**

Run: `pytest -q`
Expected: all pre-existing tests pass — the change is additive.

- [ ] **Step 7: Commit**

```bash
git add oneuniverse/data/converter.py test/test_dataset_view_time.py
git commit -m "feat(temporal): converter fills TemporalSpec and partition t_min/t_max for POINT"
```

---

## Task 5: `DataGeometry.LIGHTCURVE` geometry enum and column contracts

**Files:**
- Modify: `oneuniverse/data/format_spec.py`
- Test: `test/test_lightcurve_geometry.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_lightcurve_geometry.py
import pytest

from oneuniverse.data.format_spec import (
    DataGeometry, GEOMETRY_COLUMNS, validate_columns,
    DEFAULT_PARTITION_ROWS, FORMAT_VERSION, SCHEMA_VERSION,
)


def test_lightcurve_enum_value():
    assert DataGeometry.LIGHTCURVE.value == "lightcurve"


def test_lightcurve_object_required_columns_present():
    required = GEOMETRY_COLUMNS[DataGeometry.LIGHTCURVE]["objects"]
    # one row per logical object — lightcurves, not pixels
    for col in ("object_id", "ra", "dec", "z", "n_epochs", "mjd_min", "mjd_max"):
        assert col in required


def test_lightcurve_data_required_columns_present():
    required = GEOMETRY_COLUMNS[DataGeometry.LIGHTCURVE]["data"]
    for col in ("object_id", "mjd"):
        assert col in required


def test_validate_columns_lightcurve_objects_ok():
    missing = validate_columns(
        ["object_id", "ra", "dec", "z", "n_epochs", "mjd_min", "mjd_max"],
        DataGeometry.LIGHTCURVE, table_type="objects",
    )
    assert missing == []


def test_validate_columns_lightcurve_data_missing():
    missing = validate_columns(
        ["object_id"], DataGeometry.LIGHTCURVE, table_type="data",
    )
    assert "mjd" in missing


def test_default_partition_rows_has_lightcurve():
    assert DataGeometry.LIGHTCURVE in DEFAULT_PARTITION_ROWS


def test_format_version_bumped():
    assert FORMAT_VERSION == "2.1.0"
    assert SCHEMA_VERSION == "2.1.0"
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest test/test_lightcurve_geometry.py -v`
Expected: all FAIL — `LIGHTCURVE` not defined.

- [ ] **Step 3: Extend `format_spec.py`**

Modify `oneuniverse/data/format_spec.py`:

```python
# bump version constants
FORMAT_VERSION: str = "2.1.0"
SCHEMA_VERSION: str = "2.1.0"


# add to the DataGeometry enum (below HEALPIX)
class DataGeometry(str, Enum):
    POINT = "point"
    SIGHTLINE = "sightline"
    HEALPIX = "healpix"
    LIGHTCURVE = "lightcurve"
    """One row per (object, epoch). Multi-epoch photometry / time-domain
    surveys (ZTF, LSST, SN Ia discovery pipelines in lightcurve form).
    Tables: objects.parquet (per-object metadata)
          + part_*.parquet (per-epoch data, keyed by object_id + mjd)."""


# add the two column tables below HEALPIX_REQUIRED_COLUMNS
LIGHTCURVE_OBJECT_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "object_id",       # int64, unique per lightcurve
    "ra",              # float64, degrees (ICRS)
    "dec",             # float64, degrees (ICRS)
    "z",               # float64, redshift (NaN if unknown)
    "z_type",          # string, "spec"/"phot"/"none"
    "z_err",           # float64
    "n_epochs",        # int32
    "mjd_min",         # float64
    "mjd_max",         # float64
    "_healpix32",      # int64, NSIDE=32 NESTED cell (for spatial pruning)
)

LIGHTCURVE_DATA_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "object_id",       # int64, foreign key to objects table
    "mjd",             # float64, observation epoch (MJD, TDB by default)
    "filter",          # string, photometric band label
    "flux",            # float32, flux (arbitrary-but-declared units)
    "flux_err",        # float32
    "flag",            # int16, survey-defined quality flag
)


# add to GEOMETRY_COLUMNS
GEOMETRY_COLUMNS: Dict[DataGeometry, Dict[str, Tuple[str, ...]]] = {
    DataGeometry.POINT: {"data": POINT_REQUIRED_COLUMNS},
    DataGeometry.SIGHTLINE: {
        "objects": SIGHTLINE_OBJECT_REQUIRED_COLUMNS,
        "data": SIGHTLINE_DATA_REQUIRED_COLUMNS,
    },
    DataGeometry.HEALPIX: {"data": HEALPIX_REQUIRED_COLUMNS},
    DataGeometry.LIGHTCURVE: {
        "objects": LIGHTCURVE_OBJECT_REQUIRED_COLUMNS,
        "data": LIGHTCURVE_DATA_REQUIRED_COLUMNS,
    },
}


# add to DEFAULT_PARTITION_ROWS
DEFAULT_PARTITION_ROWS: Dict[DataGeometry, int] = {
    DataGeometry.POINT:      200_000,
    DataGeometry.SIGHTLINE:  2_000_000,
    DataGeometry.HEALPIX:    500_000,
    DataGeometry.LIGHTCURVE: 1_000_000,   # ~6 float cols per epoch ≈ 15 MB
}
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_lightcurve_geometry.py -v`
Expected: 7 passed.

- [ ] **Step 5: Full suite**

Run: `pytest -q`
Expected: all pre-existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/format_spec.py test/test_lightcurve_geometry.py
git commit -m "feat(temporal): DataGeometry.LIGHTCURVE + required columns (objects + epochs)"
```

---

## Task 6: `write_ouf_lightcurve_dataset` writer

**Files:**
- Create: `oneuniverse/data/_converter_lightcurve.py`
- Test: `test/test_lightcurve_roundtrip.py`
- Modify: `oneuniverse/data/__init__.py` (export)

- [ ] **Step 1: Write the failing test**

```python
# test/test_lightcurve_roundtrip.py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from oneuniverse.data._converter_lightcurve import write_ouf_lightcurve_dataset
from oneuniverse.data.manifest import read_manifest
from oneuniverse.data.format_spec import DataGeometry, ONEUNIVERSE_SUBDIR


def _make_lightcurve_dataset(n_obj=5, n_epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    objects = pd.DataFrame({
        "object_id": np.arange(n_obj, dtype=np.int64),
        "ra":  rng.uniform(0.0, 360.0, size=n_obj),
        "dec": rng.uniform(-60.0, 60.0, size=n_obj),
        "z":   rng.uniform(0.01, 0.5, size=n_obj),
        "z_type": np.array(["spec"] * n_obj, dtype=object),
        "z_err": rng.uniform(1e-4, 1e-3, size=n_obj),
    })
    rows = []
    for oid in objects["object_id"]:
        mjd = np.sort(rng.uniform(58000.0, 60000.0, size=n_epochs))
        for t in mjd:
            rows.append({
                "object_id": int(oid),
                "mjd": float(t),
                "filter": rng.choice(["g", "r", "i"]),
                "flux":  float(rng.normal(100.0, 5.0)),
                "flux_err": 1.0,
                "flag": 0,
            })
    epochs = pd.DataFrame(rows)
    return objects, epochs


def test_write_lightcurve_produces_manifest_and_tables(tmp_path):
    objects, epochs = _make_lightcurve_dataset(n_obj=3, n_epochs=10)
    survey_dir = tmp_path / "syn_lc"
    write_ouf_lightcurve_dataset(
        objects=objects, epochs=epochs,
        survey_path=survey_dir,
        survey_name="syn_lc", survey_type="transient",
        loader_name="syn", loader_version="0",
    )
    ou_dir = survey_dir / ONEUNIVERSE_SUBDIR
    assert (ou_dir / "manifest.json").exists()
    assert (ou_dir / "objects.parquet").exists()
    assert any(ou_dir.glob("part_*.parquet"))

    m = read_manifest(ou_dir / "manifest.json")
    assert m.geometry is DataGeometry.LIGHTCURVE
    assert m.temporal is not None
    assert m.temporal.time_column == "mjd"

    obj_tbl = pq.read_table(ou_dir / "objects.parquet").to_pandas()
    assert set(obj_tbl.columns) >= {"object_id", "ra", "dec", "z", "n_epochs", "mjd_min", "mjd_max"}
    # n_epochs matches the epoch table
    expected = epochs.groupby("object_id").size().reset_index(name="n_epochs")
    merged = obj_tbl[["object_id", "n_epochs"]].merge(expected, on="object_id", suffixes=("", "_truth"))
    assert (merged["n_epochs"] == merged["n_epochs_truth"]).all()


def test_write_lightcurve_rejects_orphan_epoch_rows(tmp_path):
    objects, epochs = _make_lightcurve_dataset(n_obj=2, n_epochs=3)
    # inject an epoch row with no matching object
    orphan = epochs.iloc[:1].copy()
    orphan["object_id"] = 999
    epochs = pd.concat([epochs, orphan], ignore_index=True)
    import pytest
    with pytest.raises(ValueError, match="orphan"):
        write_ouf_lightcurve_dataset(
            objects=objects, epochs=epochs,
            survey_path=tmp_path / "syn_bad",
            survey_name="syn_bad", survey_type="transient",
            loader_name="syn", loader_version="0",
        )


def test_write_lightcurve_partition_stats_cover_mjd(tmp_path):
    objects, epochs = _make_lightcurve_dataset(n_obj=4, n_epochs=30, seed=3)
    survey_dir = tmp_path / "syn_lc2"
    write_ouf_lightcurve_dataset(
        objects=objects, epochs=epochs,
        survey_path=survey_dir,
        survey_name="syn_lc2", survey_type="transient",
        loader_name="syn", loader_version="0",
    )
    m = read_manifest(survey_dir / ONEUNIVERSE_SUBDIR / "manifest.json")
    assert all(p.stats.t_min is not None and p.stats.t_max is not None for p in m.partitions)
    overall_min = min(p.stats.t_min for p in m.partitions)
    overall_max = max(p.stats.t_max for p in m.partitions)
    assert overall_min == float(epochs["mjd"].min())
    assert overall_max == float(epochs["mjd"].max())
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest test/test_lightcurve_roundtrip.py -v`
Expected: ImportError on `_converter_lightcurve`.

- [ ] **Step 3: Implement the writer**

```python
# oneuniverse/data/_converter_lightcurve.py
"""Writer for OUF 2.1 LIGHTCURVE datasets.

Layout (same as SIGHTLINE):
    {survey_path}/oneuniverse/
      manifest.json
      objects.parquet      # per-object metadata (object_id, ra, dec, z, n_epochs, …)
      part_0000.parquet    # per-epoch rows (object_id, mjd, filter, flux, …)
      part_0001.parquet
      …

Partitioning is by row count, sorted by (object_id, mjd) so that epoch
partitions are contiguous per object.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse.data._atomic import atomic_write_bytes
from oneuniverse.data._hashing import hash_bytes, hash_file
from oneuniverse.data.format_spec import (
    COMPRESSION, DEFAULT_PARTITION_ROWS, LIGHTCURVE_DATA_REQUIRED_COLUMNS,
    LIGHTCURVE_OBJECT_REQUIRED_COLUMNS, ONEUNIVERSE_SUBDIR,
    validate_columns, DataGeometry,
)
from oneuniverse.data.manifest import (
    ColumnSpec, LoaderSpec, Manifest, OriginalFileSpec,
    PartitionSpec, PartitionStats, write_manifest,
)
from oneuniverse.data.temporal import TemporalSpec


def _healpix32_for(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    import healpy as hp
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    return hp.ang2pix(32, theta, phi, nest=True).astype(np.int64)


def _build_objects_table(
    objects: pd.DataFrame, epochs: pd.DataFrame,
) -> pd.DataFrame:
    per_obj = epochs.groupby("object_id").agg(
        n_epochs=("mjd", "size"),
        mjd_min=("mjd", "min"),
        mjd_max=("mjd", "max"),
    ).reset_index()
    merged = objects.merge(per_obj, on="object_id", how="left")
    if merged["n_epochs"].isna().any():
        missing = merged.loc[merged["n_epochs"].isna(), "object_id"].tolist()
        raise ValueError(
            f"write_ouf_lightcurve_dataset: objects {missing!r} have no "
            f"epochs in the epoch table."
        )
    merged["n_epochs"] = merged["n_epochs"].astype("int32")
    if "_healpix32" not in merged.columns:
        merged["_healpix32"] = _healpix32_for(
            merged["ra"].to_numpy(), merged["dec"].to_numpy(),
        )
    missing_cols = validate_columns(
        list(merged.columns), DataGeometry.LIGHTCURVE, table_type="objects",
    )
    if missing_cols:
        raise ValueError(
            f"write_ouf_lightcurve_dataset: objects table missing "
            f"required columns {missing_cols!r}"
        )
    return merged


def _check_no_orphan_epochs(objects: pd.DataFrame, epochs: pd.DataFrame) -> None:
    known = set(objects["object_id"].astype(np.int64).tolist())
    seen = set(epochs["object_id"].astype(np.int64).tolist())
    orphans = seen - known
    if orphans:
        raise ValueError(
            f"write_ouf_lightcurve_dataset: {len(orphans)} epoch rows refer "
            f"to orphan object_id(s) not present in the objects table: "
            f"{sorted(list(orphans))[:5]}"
        )


def _partition_epoch_rows(
    epochs: pd.DataFrame, partition_rows: int,
) -> List[pd.DataFrame]:
    epochs = epochs.sort_values(["object_id", "mjd"]).reset_index(drop=True)
    parts = []
    for start in range(0, len(epochs), partition_rows):
        parts.append(epochs.iloc[start : start + partition_rows].copy())
    return parts


def _schema_from_frame(df: pd.DataFrame) -> List[ColumnSpec]:
    return [ColumnSpec(name=str(c), dtype=str(df[c].dtype)) for c in df.columns]


def write_ouf_lightcurve_dataset(
    *,
    objects: pd.DataFrame,
    epochs: pd.DataFrame,
    survey_path: Union[str, Path],
    survey_name: str,
    survey_type: str,
    loader_name: str,
    loader_version: str,
    partition_rows: int = DEFAULT_PARTITION_ROWS[DataGeometry.LIGHTCURVE],
    conversion_kwargs: Union[dict, None] = None,
) -> Path:
    """Write a LIGHTCURVE OUF 2.1 dataset.

    Parameters
    ----------
    objects
        Per-object table. Must contain LIGHTCURVE_OBJECT_REQUIRED_COLUMNS
        (``object_id``, ``ra``, ``dec``, ``z``, ``z_type``, ``z_err``).
        ``n_epochs``, ``mjd_min``, ``mjd_max``, ``_healpix32`` are filled
        in from the epoch table if not provided.
    epochs
        Per-(object, epoch) table. Must contain
        LIGHTCURVE_DATA_REQUIRED_COLUMNS.

    Returns
    -------
    Path
        The survey root directory (parent of ``oneuniverse/``).
    """
    survey_path = Path(survey_path)
    ou_dir = survey_path / ONEUNIVERSE_SUBDIR
    ou_dir.mkdir(parents=True, exist_ok=True)

    _check_no_orphan_epochs(objects, epochs)

    missing = validate_columns(
        list(epochs.columns), DataGeometry.LIGHTCURVE, table_type="data",
    )
    if missing:
        raise ValueError(
            f"write_ouf_lightcurve_dataset: epoch table missing required "
            f"columns {missing!r}"
        )

    objects_final = _build_objects_table(objects, epochs)
    obj_path = ou_dir / "objects.parquet"
    obj_bytes = pa.Table.from_pandas(objects_final, preserve_index=False)
    tmp = pa.BufferOutputStream()
    pq.write_table(obj_bytes, tmp, compression=COMPRESSION)
    atomic_write_bytes(obj_path, tmp.getvalue().to_pybytes())

    partitions = _partition_epoch_rows(epochs, partition_rows)
    part_specs: List[PartitionSpec] = []
    for i, chunk in enumerate(partitions):
        name = f"part_{i:04d}.parquet"
        buf = pa.BufferOutputStream()
        pq.write_table(
            pa.Table.from_pandas(chunk, preserve_index=False),
            buf, compression=COMPRESSION,
        )
        data = buf.getvalue().to_pybytes()
        atomic_write_bytes(ou_dir / name, data)
        stats = PartitionStats(
            t_min=float(chunk["mjd"].min()),
            t_max=float(chunk["mjd"].max()),
        )
        part_specs.append(PartitionSpec(
            name=name, n_rows=len(chunk),
            sha256=hash_bytes(data),
            size_bytes=(ou_dir / name).stat().st_size,
            stats=stats,
        ))

    temporal = TemporalSpec(
        t_min=float(epochs["mjd"].min()),
        t_max=float(epochs["mjd"].max()),
        time_column="mjd",
    )

    manifest = Manifest(
        oneuniverse_format_version="2.1.0",
        oneuniverse_schema_version="2.1.0",
        geometry=DataGeometry.LIGHTCURVE,
        survey_name=survey_name,
        survey_type=survey_type,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        original_files=[],
        partitions=part_specs,
        partitioning=None,
        schema=_schema_from_frame(epochs),
        conversion_kwargs=dict(conversion_kwargs or {}),
        loader=LoaderSpec(name=loader_name, version=loader_version),
        temporal=temporal,
        extra={"n_objects": int(len(objects_final))},
    )
    write_manifest(ou_dir / "manifest.json", manifest)
    return survey_path
```

- [ ] **Step 4: Export from `oneuniverse.data`**

Append to `oneuniverse/data/__init__.py`:

```python
from oneuniverse.data.temporal import TemporalSpec  # noqa: F401
from oneuniverse.data._converter_lightcurve import write_ouf_lightcurve_dataset  # noqa: F401
```

- [ ] **Step 5: Run tests**

Run: `pytest test/test_lightcurve_roundtrip.py -v`
Expected: 3 passed.

- [ ] **Step 6: Full suite**

Run: `pytest -q`
Expected: all pre-existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add oneuniverse/data/_converter_lightcurve.py oneuniverse/data/__init__.py \
        test/test_lightcurve_roundtrip.py
git commit -m "feat(temporal): write_ouf_lightcurve_dataset writer (objects + epoch partitions)"
```

---

## Task 7: `DatasetView.objects_table()` for SIGHTLINE and LIGHTCURVE

**Files:**
- Modify: `oneuniverse/data/dataset_view.py`
- Test: append to `test/test_lightcurve_roundtrip.py`

Motivation: LIGHTCURVE (and SIGHTLINE) datasets have two on-disk tables;
the existing `scan` returns the `part_*.parquet` rows. Callers that want
the per-object metadata must be able to reach `objects.parquet` without
re-reading all epochs.

- [ ] **Step 1: Write the failing test**

Append to `test/test_lightcurve_roundtrip.py`:

```python
def test_dataset_view_objects_table_lightcurve(tmp_path):
    from oneuniverse.data.dataset_view import DatasetView
    objects, epochs = _make_lightcurve_dataset(n_obj=4, n_epochs=5)
    survey_dir = tmp_path / "syn_view_lc"
    write_ouf_lightcurve_dataset(
        objects=objects, epochs=epochs,
        survey_path=survey_dir,
        survey_name="syn_view_lc", survey_type="transient",
        loader_name="syn", loader_version="0",
    )
    view = DatasetView.from_path(survey_dir)
    obj = view.objects_table()
    assert len(obj) == 4
    assert "n_epochs" in obj.columns

    epo = view.scan(columns=["object_id", "mjd"], t_range=(59000.0, 60000.0))
    # only epochs inside the window
    assert all(t >= 59000.0 and t <= 60000.0 for t in epo.column("mjd").to_pylist())


def test_dataset_view_objects_table_on_point_raises(tmp_path):
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.dataset_view import DatasetView
    import numpy as np, pandas as pd
    df = pd.DataFrame({
        "ra": [0.0], "dec": [0.0], "z": [0.1], "z_type": ["spec"], "z_err": [0.01],
        "galaxy_id": np.array([0], dtype=np.int64),
        "survey_id": ["syn0"],
        "_original_row_index": np.array([0], dtype=np.int64),
    })
    survey_dir = tmp_path / "syn_point"
    write_ouf_dataset(
        df, survey_dir,
        survey_name="syn_point", survey_type="spectroscopic",
        geometry=__import__("oneuniverse.data.format_spec", fromlist=["DataGeometry"]).DataGeometry.POINT,
        loader_name="syn", loader_version="0",
    )
    view = DatasetView.from_path(survey_dir)
    import pytest
    with pytest.raises(ValueError, match="objects_table"):
        view.objects_table()
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest test/test_lightcurve_roundtrip.py -v`
Expected: the two new tests FAIL (`DatasetView.objects_table` missing).

- [ ] **Step 3: Implement `objects_table()`**

In `oneuniverse/data/dataset_view.py`, add inside `DatasetView`:

```python
def objects_table(
    self,
    columns: Optional[Sequence[str]] = None,
) -> pa.Table:
    """Return the per-object (``objects.parquet``) table for SIGHTLINE
    and LIGHTCURVE geometries.

    Raises
    ------
    ValueError
        If the dataset's geometry has no objects table (POINT, HEALPIX).
    """
    if self.geometry not in {DataGeometry.SIGHTLINE, DataGeometry.LIGHTCURVE}:
        raise ValueError(
            f"objects_table() is only defined for SIGHTLINE and LIGHTCURVE "
            f"geometries, got {self.geometry.value!r}"
        )
    from oneuniverse.data.format_spec import OBJECTS_FILENAME
    import pyarrow.parquet as pq
    obj_path = self.ou_dir / OBJECTS_FILENAME
    table = pq.read_table(obj_path, columns=list(columns) if columns else None)
    return table
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_lightcurve_roundtrip.py -v`
Expected: all 5 tests pass.

- [ ] **Step 5: Full suite**

Run: `pytest -q`

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/dataset_view.py test/test_lightcurve_roundtrip.py
git commit -m "feat(temporal): DatasetView.objects_table() for SIGHTLINE/LIGHTCURVE geometries"
```

---

## Task 8: Visual diagnostic — temporal dataset

**Files:**
- Create: `test/test_visual_temporal.py`

This task satisfies the standing instruction that data-infrastructure
work produces a diagnostic figure (see `feedback_visual_testing`
memory). The test is marked so it runs locally but is skipped in
headless CI if matplotlib cannot import a backend.

- [ ] **Step 1: Write the test**

```python
# test/test_visual_temporal.py
"""Diagnostic figure for Phase 7 — skymap of transient events + MJD histogram
for a POINT dataset; object skymap + per-object lightcurve peek for a
LIGHTCURVE dataset. Output is written to test_output/ so it can be
inspected manually."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.data.converter import write_ouf_dataset  # noqa: E402
from oneuniverse.data.dataset_view import DatasetView  # noqa: E402
from oneuniverse.data.format_spec import DataGeometry  # noqa: E402
from oneuniverse.data._converter_lightcurve import write_ouf_lightcurve_dataset  # noqa: E402


OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _make_transient_point(n=5000, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ra":    rng.uniform(0.0, 360.0, n),
        "dec":   rng.uniform(-60.0, 60.0, n),
        "z":     rng.uniform(0.01, 0.5, n),
        "z_type": np.array(["spec"] * n, dtype=object),
        "z_err": rng.uniform(1e-4, 1e-3, n),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.array([f"syn{i:06d}" for i in range(n)], dtype=object),
        "_original_row_index": np.arange(n, dtype=np.int64),
        "t_obs": rng.uniform(58000.0, 60000.0, n),
    })


def test_visual_temporal_point(tmp_path):
    df = _make_transient_point()
    survey_dir = tmp_path / "viz_point"
    write_ouf_dataset(
        df, survey_dir,
        survey_name="viz_point", survey_type="transient",
        geometry=DataGeometry.POINT, loader_name="syn", loader_version="0",
    )
    view = DatasetView.from_path(survey_dir)
    sub = view.read(columns=["ra", "dec", "t_obs"],
                    t_range=(59000.0, 59500.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ra_wrap = np.where(df["ra"] > 180, df["ra"] - 360, df["ra"])
    axes[0].scatter(np.radians(ra_wrap), np.radians(df["dec"]),
                    s=2, alpha=0.4, color="#888", label="all")
    ra_sub = np.where(sub["ra"] > 180, sub["ra"] - 360, sub["ra"])
    axes[0].scatter(np.radians(ra_sub), np.radians(sub["dec"]),
                    s=4, color="#d62728", label="t_range window")
    axes[0].set_xlabel("RA (rad)")
    axes[0].set_ylabel("Dec (rad)")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Transient events — windowed vs full")

    axes[1].hist(df["t_obs"], bins=80, color="#3a7bd5", alpha=0.6, label="all")
    axes[1].axvspan(59000.0, 59500.0, color="#d62728", alpha=0.2, label="window")
    axes[1].set_xlabel("MJD")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].set_title(f"{len(sub):,} events in window / {len(df):,} total")

    fig.tight_layout()
    out = OUTPUT_DIR / "phase7_temporal_point.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert out.exists()
    assert out.stat().st_size > 10_000


def test_visual_temporal_lightcurve(tmp_path):
    rng = np.random.default_rng(2)
    n_obj = 10
    n_epochs = 30
    objects = pd.DataFrame({
        "object_id": np.arange(n_obj, dtype=np.int64),
        "ra":  rng.uniform(0.0, 360.0, n_obj),
        "dec": rng.uniform(-60.0, 60.0, n_obj),
        "z":   rng.uniform(0.01, 0.2, n_obj),
        "z_type": np.array(["spec"] * n_obj, dtype=object),
        "z_err": rng.uniform(1e-4, 1e-3, n_obj),
    })
    rows = []
    for oid in objects["object_id"]:
        mjd = np.sort(rng.uniform(58000.0, 60000.0, n_epochs))
        for i, t in enumerate(mjd):
            rows.append({
                "object_id": int(oid),
                "mjd": float(t),
                "filter": ("g", "r", "i")[i % 3],
                "flux":  float(100.0 + 20.0 * np.sin(0.01 * t) + rng.normal(0, 2)),
                "flux_err": 1.0,
                "flag": 0,
            })
    epochs = pd.DataFrame(rows)
    survey_dir = tmp_path / "viz_lc"
    write_ouf_lightcurve_dataset(
        objects=objects, epochs=epochs,
        survey_path=survey_dir,
        survey_name="viz_lc", survey_type="transient",
        loader_name="syn", loader_version="0",
    )
    view = DatasetView.from_path(survey_dir)
    obj_tbl = view.objects_table().to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(obj_tbl["ra"], obj_tbl["dec"], s=40, c="#3a7bd5", edgecolor="k")
    for _, row in obj_tbl.iterrows():
        axes[0].annotate(str(row["object_id"]), (row["ra"], row["dec"]))
    axes[0].set_xlabel("RA (deg)")
    axes[0].set_ylabel("Dec (deg)")
    axes[0].set_title(f"{len(obj_tbl)} lightcurve objects")

    pick = int(obj_tbl["object_id"].iloc[0])
    lc = view.scan(columns=["mjd", "flux", "filter"],
                   filter=__import__("pyarrow.compute", fromlist=["field"]).field("object_id") == pick
                  ).to_pandas()
    for band, sub in lc.groupby("filter"):
        axes[1].errorbar(sub["mjd"], sub["flux"],
                         yerr=1.0, fmt="o", label=str(band), ms=3)
    axes[1].set_xlabel("MJD")
    axes[1].set_ylabel("flux")
    axes[1].legend()
    axes[1].set_title(f"Lightcurve of object_id={pick}")

    fig.tight_layout()
    out = OUTPUT_DIR / "phase7_temporal_lightcurve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert out.exists()
    assert out.stat().st_size > 10_000
```

- [ ] **Step 2: Run the visual tests**

Run: `pytest test/test_visual_temporal.py -v`
Expected: 2 passed. Inspect the generated images under `test/test_output/phase7_temporal_point.png` and `test/test_output/phase7_temporal_lightcurve.png`.

- [ ] **Step 3: Commit**

```bash
git add test/test_visual_temporal.py
git commit -m "test(temporal): diagnostic figures for POINT(t_obs) and LIGHTCURVE datasets"
```

---

## Task 9: Phase-status update

**Files:**
- Modify: `plans/README.md`
- Modify: `/home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md`

- [ ] **Step 1: Update the plan index**

In `plans/README.md`, insert the new plans in the list and add a row to the status table:

```markdown
- [`2026-04-20-temporal-subobject-roadmap.md`](2026-04-20-temporal-subobject-roadmap.md) — roadmap for Phase 7 (temporal) + Phase 8 (sub-object).
- [`2026-04-20-phase7-temporal.md`](2026-04-20-phase7-temporal.md) — detailed task-by-task plan for Phase 7.
- [`2026-04-20-phase8-subobject.md`](2026-04-20-phase8-subobject.md) — detailed task-by-task plan for Phase 8.
```

Status table row to add (fill test count once the task is complete):

```markdown
| 7 | Temporal data (t_obs on POINT + LIGHTCURVE geometry) | **complete (2026-04-20, N/N tests green)** |
```

- [ ] **Step 2: Update the memory file**

Append a Phase-7-complete block to `project_oneuniverse_stabilisation.md`
with the same shape used for Phase-6 entries: list each task commit, note
the total test count, and record the new manifest format version.

- [ ] **Step 3: Commit**

```bash
git add plans/README.md
git commit -m "docs(plans): mark Phase 7 (temporal) complete"
```

---

## Self-review checklist

- [ ] Every new column declared in `format_spec.py` appears by name in at least one test.
- [ ] `Manifest.temporal` is optional (`None` for non-temporal datasets); no existing 2.0.x test needed to be rewritten beyond the version-string match.
- [ ] `DatasetView.scan` parameters are kwargs-only and additive — no caller signatures broken.
- [ ] `write_ouf_lightcurve_dataset` validates both its inputs (orphan epochs, missing columns) before writing anything.
- [ ] `objects_table()` raises rather than returning an empty table for geometries without an objects file.
- [ ] The diagnostic test is skipped when matplotlib is absent.
- [ ] Format version bump is compatible — the reader accepts `2.0.x` *and* `2.1.x`.
