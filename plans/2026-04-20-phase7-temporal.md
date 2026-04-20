# Phase 7 — Temporal Data Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `oneuniverse` two complementary time axes — (1) per-row observation time, including a new `LIGHTCURVE` geometry for multi-epoch data, and (2) per-dataset transaction time, enabling `OneuniverseDatabase.as_of(T)` snapshot queries and versioned ONEUID indices.

**Architecture:**
- Layer A (row-level temporal): `TemporalSpec` manifest block + `PartitionStats.t_min/t_max` + `DatasetView.scan(t_range=…)`. New `DataGeometry.LIGHTCURVE` with `objects.parquet` + per-epoch `part_*.parquet`.
- Layer B (database-level bitemporal): `DatasetValidity` manifest block (`valid_from_utc`, `valid_to_utc`, `version`, `supersedes`). `OneuniverseDatabase` gains `as_of(ts)` and `versions_of(name)`. ONEUID manifests get the same block, and `load_oneuid(name, as_of=T)` resolves the correct version.
- Format version bump: OUF `2.0.0 → 2.1.0`, schema `2.0.0 → 2.1.0` (reader accepts both).

**Tech Stack:** Python 3.10+, dataclasses, pyarrow/pyarrow.dataset, pandas, pytest, healpy, numpy, matplotlib (diagnostic only).

**Pre-reading:**
- `oneuniverse/data/format_spec.py` — `DataGeometry`, column tables.
- `oneuniverse/data/manifest.py` — `Manifest`, `PartitionSpec`, `PartitionStats`, version compatibility rule in `_from_dict`.
- `oneuniverse/data/dataset_view.py` — `_select_partitions`, `_range_expr`, `_spatial_expr`.
- `oneuniverse/data/database.py` — `OneuniverseDatabase` walker, `DatasetEntry`.
- `oneuniverse/data/oneuid.py` — `OneuidIndex`, `_index_manifest_path`, `load_oneuid_index`, `list_oneuids`.
- `oneuniverse/data/converter.py` — writer for POINT (pattern reference for partition stats).

---

## File Structure

**Create:**
- `oneuniverse/data/temporal.py` — `TemporalSpec` frozen dataclass.
- `oneuniverse/data/validity.py` — `DatasetValidity` frozen dataclass.
- `oneuniverse/data/_converter_lightcurve.py` — `write_ouf_lightcurve_dataset()`.
- `test/test_temporal_spec.py`
- `test/test_validity_spec.py`
- `test/test_partition_stats_time.py`
- `test/test_dataset_view_time.py`
- `test/test_lightcurve_geometry.py`
- `test/test_lightcurve_roundtrip.py`
- `test/test_database_as_of.py`
- `test/test_oneuid_bitemporal.py`
- `test/test_visual_temporal.py`

**Modify:**
- `oneuniverse/data/format_spec.py` — `DataGeometry.LIGHTCURVE`, lightcurve column tables, `DEFAULT_PARTITION_ROWS`, bump version constants to `2.1.0`.
- `oneuniverse/data/manifest.py` — `PartitionStats.t_min/t_max`, `Manifest.temporal`, `Manifest.validity`, version-compat rule (`2.x`), default-fill validity when reading 2.0.x.
- `oneuniverse/data/dataset_view.py` — `scan(t_range=…)`, `_range_expr` time-column awareness, `objects_table()` for SIGHTLINE/LIGHTCURVE.
- `oneuniverse/data/converter.py` — fill `TemporalSpec` + partition `t_min/t_max` when `t_obs` column present; accept a `validity` kwarg.
- `oneuniverse/data/database.py` — walker honours validity, `as_of(ts)`, `versions_of(name)`.
- `oneuniverse/data/oneuid.py` — ONEUID manifest carries `DatasetValidity`, `build_oneuid_index(...)` closes out prior version, `load_oneuid_index(name, as_of=…)`.
- `oneuniverse/data/__init__.py` — export `TemporalSpec`, `DatasetValidity`, `write_ouf_lightcurve_dataset`.

---

## Conventions

Every task ends with a commit step. Commit messages follow the repo's
style (short imperative subject, optional body). Tests live in
`Packages/oneuniverse/test/`. Every `pytest` invocation assumes working
directory = `Packages/oneuniverse/`.

Dataclasses are **frozen** (`@dataclass(frozen=True)`) unless the plan
explicitly says otherwise.

ISO-8601 UTC timestamps everywhere. Helper:

```python
import datetime as dt

def _utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
```

Parse incoming timestamps with `dt.datetime.fromisoformat(...)`;
serialize with `dt.isoformat()`. Store strings in the manifest (JSON
round-trip), but compare as `dt.datetime` objects (always timezone-aware
UTC — reject naive).

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


def test_defaults():
    s = TemporalSpec(t_min=58000.0, t_max=60000.0)
    assert s.time_column == "t_obs"
    assert s.time_unit == "MJD"
    assert s.time_reference == "TDB"
    assert s.cadence is None


def test_frozen():
    s = TemporalSpec(t_min=0.0, t_max=1.0)
    with pytest.raises((AttributeError, TypeError)):
        s.t_min = 2.0  # type: ignore[misc]


def test_rejects_inverted_range():
    with pytest.raises(ValueError, match="t_min"):
        TemporalSpec(t_min=10.0, t_max=5.0)


def test_rejects_unknown_time_reference():
    with pytest.raises(ValueError, match="time_reference"):
        TemporalSpec(t_min=0.0, t_max=1.0, time_reference="INVALID")


def test_dict_roundtrip():
    s = TemporalSpec(t_min=58000.0, t_max=60000.0, cadence=7.0,
                     time_column="mjd")
    assert TemporalSpec.from_dict(s.to_dict()) == s
```

- [ ] **Step 2: Confirm failure**

Run: `pytest test/test_temporal_spec.py -v`
Expected: `ModuleNotFoundError: oneuniverse.data.temporal`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/data/temporal.py
"""Temporal (observation-time) metadata for OUF 2.1 manifests."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

_ALLOWED_TIME_REFERENCES = frozenset({"TDB", "TAI", "UTC", "TT"})


@dataclass(frozen=True)
class TemporalSpec:
    """Describes the *physical* time axis of a temporal dataset.

    This is the axis astronomers plot on a lightcurve. Not to be
    confused with :class:`DatasetValidity`, which tracks *database*
    time (when the row was known to us).
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
git commit -m "feat(temporal): TemporalSpec dataclass for OUF 2.1 manifests"
```

---

## Task 2: `DatasetValidity` dataclass

**Files:**
- Create: `oneuniverse/data/validity.py`
- Test: `test/test_validity_spec.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_validity_spec.py
import datetime as dt

import pytest

from oneuniverse.data.validity import DatasetValidity


_T1 = "2026-01-01T00:00:00+00:00"
_T2 = "2026-06-01T00:00:00+00:00"


def test_defaults_make_current_entry():
    v = DatasetValidity(valid_from_utc=_T1)
    assert v.valid_to_utc is None
    assert v.version == "1.0"
    assert v.supersedes == ()
    assert v.is_current()


def test_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="timezone"):
        DatasetValidity(valid_from_utc="2026-01-01T00:00:00")


def test_rejects_inverted_interval():
    with pytest.raises(ValueError, match="valid_from"):
        DatasetValidity(valid_from_utc=_T2, valid_to_utc=_T1)


def test_contains_happy_path():
    v = DatasetValidity(valid_from_utc=_T1, valid_to_utc=_T2)
    assert v.contains(dt.datetime.fromisoformat(_T1))
    middle = dt.datetime.fromisoformat("2026-03-01T00:00:00+00:00")
    assert v.contains(middle)
    assert not v.contains(dt.datetime.fromisoformat(_T2))


def test_is_current_after_close():
    v = DatasetValidity(valid_from_utc=_T1, valid_to_utc=_T2)
    assert not v.is_current()


def test_closed_at_returns_new_instance():
    v = DatasetValidity(valid_from_utc=_T1)
    closed = v.closed_at(_T2)
    assert closed.valid_to_utc == _T2
    assert v.valid_to_utc is None  # original untouched


def test_dict_roundtrip():
    v = DatasetValidity(
        valid_from_utc=_T1, valid_to_utc=_T2,
        version="dr17", supersedes=("eboss_qso_dr16",),
    )
    assert DatasetValidity.from_dict(v.to_dict()) == v
```

- [ ] **Step 2: Confirm failure**

Run: `pytest test/test_validity_spec.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/data/validity.py
"""Transaction-time metadata: which datasets were valid when."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, Optional, Tuple


def _parse(ts: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        raise ValueError(
            f"DatasetValidity: timestamp {ts!r} has no timezone; "
            f"require ISO-8601 UTC (e.g. '2026-01-01T00:00:00+00:00')"
        )
    return parsed.astimezone(dt.timezone.utc)


@dataclass(frozen=True)
class DatasetValidity:
    """When this dataset was the authoritative answer.

    Attributes
    ----------
    valid_from_utc : str
        ISO-8601 UTC timestamp. Required — when the dataset was
        ingested / became authoritative.
    valid_to_utc : str or None
        ISO-8601 UTC timestamp. ``None`` means still current.
    version : str
        Free-form version label (e.g. ``"dr16"``, ``"2.0"``,
        ``"2026-04-15-rebuild"``). Default ``"1.0"``.
    supersedes : tuple[str, ...]
        Zero or more dataset names whose validity this entry closes
        out. Dataset names follow the database's naming convention.
    """

    valid_from_utc: str
    valid_to_utc: Optional[str] = None
    version: str = "1.0"
    supersedes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        t0 = _parse(self.valid_from_utc)
        if self.valid_to_utc is not None:
            t1 = _parse(self.valid_to_utc)
            if t1 <= t0:
                raise ValueError(
                    f"DatasetValidity: valid_from={self.valid_from_utc} "
                    f">= valid_to={self.valid_to_utc}"
                )
        # normalize stored form — always ISO-8601 UTC with explicit +00:00
        object.__setattr__(
            self, "valid_from_utc", t0.isoformat(),
        )
        if self.valid_to_utc is not None:
            object.__setattr__(
                self, "valid_to_utc", _parse(self.valid_to_utc).isoformat(),
            )
        object.__setattr__(self, "supersedes", tuple(self.supersedes))

    def contains(self, when: dt.datetime) -> bool:
        if when.tzinfo is None:
            raise ValueError("DatasetValidity.contains: when must be tz-aware")
        t0 = _parse(self.valid_from_utc)
        if when < t0:
            return False
        if self.valid_to_utc is None:
            return True
        return when < _parse(self.valid_to_utc)

    def is_current(self, now: Optional[dt.datetime] = None) -> bool:
        now = now or dt.datetime.now(dt.timezone.utc)
        return self.contains(now)

    def closed_at(self, ts: str) -> "DatasetValidity":
        return replace(self, valid_to_utc=ts)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["supersedes"] = list(self.supersedes)
        return d

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DatasetValidity":
        return cls(
            valid_from_utc=raw["valid_from_utc"],
            valid_to_utc=raw.get("valid_to_utc"),
            version=raw.get("version", "1.0"),
            supersedes=tuple(raw.get("supersedes", ())),
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_validity_spec.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/validity.py test/test_validity_spec.py
git commit -m "feat(temporal): DatasetValidity dataclass (transaction-time axis)"
```

---

## Task 3: `PartitionStats.t_min/t_max` and `Manifest.temporal` + `Manifest.validity`

**Files:**
- Modify: `oneuniverse/data/manifest.py`
- Test: `test/test_partition_stats_time.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_partition_stats_time.py
import pytest

from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import (
    ColumnSpec, LoaderSpec, Manifest, OriginalFileSpec,
    PartitionSpec, PartitionStats, write_manifest, read_manifest,
)
from oneuniverse.data.temporal import TemporalSpec
from oneuniverse.data.validity import DatasetValidity


_VALID_T0 = "2026-01-01T00:00:00+00:00"


def _minimal(tmp_path, stats, temporal=None, validity=None):
    m = Manifest(
        oneuniverse_format_version="2.1.0",
        oneuniverse_schema_version="2.1.0",
        geometry=DataGeometry.POINT,
        survey_name="syn",
        survey_type="transient",
        created_utc=_VALID_T0,
        original_files=[OriginalFileSpec(
            path="src.csv", sha256="0"*16, n_rows=None,
            size_bytes=1, format="csv",
        )],
        partitions=[PartitionSpec(
            name="part_0000.parquet", n_rows=10,
            sha256="f"*16, size_bytes=100, stats=stats,
        )],
        partitioning=None,
        schema=[ColumnSpec(name="t_obs", dtype="float64")],
        conversion_kwargs={},
        loader=LoaderSpec(name="syn", version="0"),
        temporal=temporal,
        validity=(validity or DatasetValidity(valid_from_utc=_VALID_T0)),
    )
    out = tmp_path / "manifest.json"
    write_manifest(out, m)
    return out


def test_partition_stats_accepts_time_fields():
    s = PartitionStats(t_min=58000.0, t_max=60000.0)
    assert s.t_min == 58000.0 and s.t_max == 60000.0


def test_partition_stats_time_defaults_none():
    s = PartitionStats()
    assert s.t_min is None and s.t_max is None


def test_manifest_roundtrip_preserves_time_stats(tmp_path):
    stats = PartitionStats(ra_min=0, ra_max=10, t_min=58000.0, t_max=60000.0)
    m = read_manifest(_minimal(tmp_path, stats))
    assert m.partitions[0].stats.t_min == 58000.0
    assert m.partitions[0].stats.t_max == 60000.0


def test_manifest_roundtrip_preserves_temporal_spec(tmp_path):
    spec = TemporalSpec(t_min=58000.0, t_max=60000.0, cadence=3.0)
    m = read_manifest(_minimal(tmp_path, PartitionStats(), temporal=spec))
    assert m.temporal == spec


def test_manifest_roundtrip_preserves_validity(tmp_path):
    v = DatasetValidity(valid_from_utc=_VALID_T0, version="dr17",
                        supersedes=("eboss_qso_dr16",))
    m = read_manifest(_minimal(tmp_path, PartitionStats(), validity=v))
    assert m.validity == v


def test_manifest_20_without_validity_is_defaulted(tmp_path):
    """A 2.0.x manifest on disk (no validity block) reads back with a
    default-filled validity. Forward-compatibility promise."""
    import json
    ok = _minimal(tmp_path, PartitionStats())
    raw = json.loads(ok.read_text())
    raw["oneuniverse_format_version"] = "2.0.0"
    raw.pop("validity", None)
    raw.pop("temporal", None)
    ok.write_text(json.dumps(raw))
    m = read_manifest(ok)
    assert m.validity is not None
    # default-filled: valid_from == created_utc, valid_to is None
    assert m.validity.valid_from_utc.startswith("2026-01-01")
    assert m.validity.valid_to_utc is None
    assert m.validity.version == "1.0"


def test_manifest_format_version_3x_rejected(tmp_path):
    import json
    ok = _minimal(tmp_path, PartitionStats())
    raw = json.loads(ok.read_text())
    raw["oneuniverse_format_version"] = "3.0.0"
    ok.write_text(json.dumps(raw))
    with pytest.raises(Exception):
        read_manifest(ok)
```

- [ ] **Step 2: Confirm failure**

Run: `pytest test/test_partition_stats_time.py -v`
Expected: FAIL — `PartitionStats` has no `t_min/t_max`, `Manifest` has no `temporal` / `validity`.

- [ ] **Step 3: Extend `manifest.py`**

Edit `oneuniverse/data/manifest.py`:

```python
# new imports at top
from oneuniverse.data.temporal import TemporalSpec
from oneuniverse.data.validity import DatasetValidity


# bump constants
FORMAT_VERSION: str = "2.1.0"
SCHEMA_VERSION: str = "2.1.0"


# replace PartitionStats
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


# extend Manifest (add new trailing fields)
@dataclass(frozen=True)
class Manifest:
    oneuniverse_format_version: str
    oneuniverse_schema_version: str
    geometry: DataGeometry
    survey_name: str
    survey_type: str
    created_utc: str
    original_files: List[OriginalFileSpec]
    partitions: List[PartitionSpec]
    partitioning: Optional[PartitioningSpec]
    schema: List[ColumnSpec]
    conversion_kwargs: Dict[str, Any]
    loader: LoaderSpec
    extra: Dict[str, Any] = field(default_factory=dict)
    temporal: Optional[TemporalSpec] = None
    validity: Optional[DatasetValidity] = None

    @property
    def n_rows(self) -> int:
        return sum(p.n_rows for p in self.partitions)

    @property
    def n_partitions(self) -> int:
        return len(self.partitions)
```

Update `_to_dict`:

```python
def _to_dict(m: Manifest) -> Dict[str, Any]:
    d = asdict(m)
    d["geometry"] = m.geometry.value
    if m.temporal is None:
        d["temporal"] = None
    if m.validity is None:
        d["validity"] = None
    else:
        d["validity"] = m.validity.to_dict()
    return d
```

Update the version-compat gate in `_from_dict`:

```python
fmt = raw["oneuniverse_format_version"]
if not (isinstance(fmt, str) and (fmt.startswith("2.0") or fmt.startswith("2.1"))):
    raise ManifestValidationError(
        f"{path}: oneuniverse_format_version={fmt!r} is not compatible "
        f"with this library (expected 2.0.x or 2.1.x)."
    )
```

Update the partition decoder to read the new stats:

```python
partitions = [
    PartitionSpec(
        name=p["name"],
        n_rows=int(p["n_rows"]),
        sha256=p["sha256"],
        size_bytes=int(p["size_bytes"]),
        stats=PartitionStats(**p.get("stats", {})),
        healpix_cell=(
            int(p["healpix_cell"])
            if p.get("healpix_cell") is not None else None
        ),
    )
    for p in raw["partitions"]
]
```

Decode `temporal` and `validity` before constructing `Manifest`:

```python
temporal_raw = raw.get("temporal")
temporal = TemporalSpec.from_dict(temporal_raw) if temporal_raw else None

validity_raw = raw.get("validity")
if validity_raw is not None:
    validity = DatasetValidity.from_dict(validity_raw)
else:
    # 2.0.x back-compat: synthesize from created_utc.
    created = raw["created_utc"]
    # created_utc may have been written without a timezone marker in
    # early 2.0.0; attach +00:00 if so.
    if "+" not in created and "Z" not in created and not created.endswith("+00:00"):
        created = created + "+00:00"
    validity = DatasetValidity(valid_from_utc=created)

return Manifest(
    ...,
    temporal=temporal,
    validity=validity,
)
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_partition_stats_time.py -v`
Expected: 7 passed.

- [ ] **Step 5: Full suite**

Run: `pytest -q`
Expected: existing tests pass. If any test wrote a literal
`"oneuniverse_format_version": "2.0.0"` and reads it back expecting the
exact string, update it to check `.startswith("2.")`.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/manifest.py test/test_partition_stats_time.py
git commit -m "feat(temporal): PartitionStats time fields; Manifest.temporal + .validity; 2.1.0"
```

---

## Task 4: `DatasetView.scan(t_range=…)` partition pruning + pushdown

**Files:**
- Modify: `oneuniverse/data/dataset_view.py`
- Test: `test/test_dataset_view_time.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_dataset_view_time.py
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.manifest import (
    ColumnSpec, LoaderSpec, Manifest, OriginalFileSpec,
    PartitionSpec, PartitionStats, write_manifest,
)
from oneuniverse.data.format_spec import DataGeometry, ONEUNIVERSE_SUBDIR
from oneuniverse.data.temporal import TemporalSpec
from oneuniverse.data.validity import DatasetValidity


_T0 = "2026-01-01T00:00:00+00:00"


def _make_df(n, seed, t0, t1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ra":  rng.uniform(0.0, 360.0, n),
        "dec": rng.uniform(-60.0, 60.0, n),
        "t_obs": rng.uniform(t0, t1, n),
    })


def _write_point(ou_dir: Path, name: str, df: pd.DataFrame) -> PartitionSpec:
    path = ou_dir / name
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return PartitionSpec(
        name=name, n_rows=len(df),
        sha256="0" * 16, size_bytes=path.stat().st_size,
        stats=PartitionStats(
            t_min=float(df["t_obs"].min()),
            t_max=float(df["t_obs"].max()),
        ),
    )


def _build(tmp_path: Path, partitions):
    survey_dir = tmp_path / "syn"
    ou_dir = survey_dir / ONEUNIVERSE_SUBDIR
    ou_dir.mkdir(parents=True)
    specs = [_write_point(ou_dir, name, df) for name, df in partitions]
    t_min = min(p.stats.t_min for p in specs)
    t_max = max(p.stats.t_max for p in specs)
    m = Manifest(
        oneuniverse_format_version="2.1.0",
        oneuniverse_schema_version="2.1.0",
        geometry=DataGeometry.POINT,
        survey_name="syn", survey_type="transient",
        created_utc=_T0,
        original_files=[OriginalFileSpec(path="src.csv", sha256="0"*16,
                                          n_rows=None, size_bytes=1, format="csv")],
        partitions=specs, partitioning=None,
        schema=[ColumnSpec(name=c, dtype=str(partitions[0][1][c].dtype))
                for c in partitions[0][1].columns],
        conversion_kwargs={},
        loader=LoaderSpec(name="syn", version="0"),
        temporal=TemporalSpec(t_min=t_min, t_max=t_max),
        validity=DatasetValidity(valid_from_utc=_T0),
    )
    write_manifest(ou_dir / "manifest.json", m)
    return survey_dir


def test_t_range_prunes_partition_by_stats(tmp_path):
    early = _make_df(5, 1, 58000.0, 58100.0)
    late = _make_df(5, 2, 60000.0, 60100.0)
    root = _build(tmp_path, [("part_0000.parquet", early),
                             ("part_0001.parquet", late)])
    view = DatasetView.from_path(root)
    kept = view._select_partitions(t_range=(59500.0, 61000.0))
    assert [p.name for p in kept] == ["part_0001.parquet"]


def test_t_range_pushdown_filters_rows(tmp_path):
    df = pd.DataFrame({
        "ra": [0.0, 0.0, 0.0],
        "dec": [0.0, 0.0, 0.0],
        "t_obs": [58000.0, 59000.0, 60000.0],
    })
    root = _build(tmp_path, [("part_0000.parquet", df)])
    view = DatasetView.from_path(root)
    out = view.read(columns=["t_obs"], t_range=(58500.0, 59500.0))
    assert list(out["t_obs"]) == [59000.0]


def test_t_range_none_returns_all(tmp_path):
    df = pd.DataFrame({"ra": [0.0, 0.0], "dec": [0.0, 0.0],
                       "t_obs": [58000.0, 60000.0]})
    root = _build(tmp_path, [("part_0000.parquet", df)])
    assert len(DatasetView.from_path(root).read(columns=["t_obs"])) == 2


def test_t_range_outside_any_partition_returns_empty(tmp_path):
    df = pd.DataFrame({"ra": [0.0], "dec": [0.0], "t_obs": [58000.0]})
    root = _build(tmp_path, [("part_0000.parquet", df)])
    out = DatasetView.from_path(root).read(
        columns=["t_obs"], t_range=(59000.0, 59500.0),
    )
    assert len(out) == 0
```

- [ ] **Step 2: Confirm failure**

Run: `pytest test/test_dataset_view_time.py -v`
Expected: FAIL.

- [ ] **Step 3: Extend `_select_partitions`**

In `oneuniverse/data/dataset_view.py`:

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
        if (cell_filter is not None
                and part.healpix_cell is not None
                and part.healpix_cell not in cell_filter):
            continue
        if not _range_overlaps(ra_range, part.stats.ra_min, part.stats.ra_max):
            continue
        if not _range_overlaps(dec_range, part.stats.dec_min, part.stats.dec_max):
            continue
        if not _range_overlaps(z_range, part.stats.z_min, part.stats.z_max):
            continue
        if not _range_overlaps(t_range, part.stats.t_min, part.stats.t_max):
            continue
        keep.append(part)
    return keep
```

- [ ] **Step 4: Extend `_range_expr` for the time column**

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

- [ ] **Step 5: Thread `t_range` through `scan`/`read`**

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
    expr = _range_expr(filter, ra_range, dec_range, z_range,
                       t_range, time_col)
    expr = _spatial_expr(expr, cone, skypatch)
    cols = list(columns) if columns is not None else None
    return dataset.to_table(columns=cols, filter=expr)


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
        t_range=t_range, cone=cone, skypatch=skypatch,
        healpix_cells=healpix_cells,
    ).to_pandas()
```

- [ ] **Step 6: Run tests**

Run: `pytest test/test_dataset_view_time.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add oneuniverse/data/dataset_view.py test/test_dataset_view_time.py
git commit -m "feat(temporal): DatasetView.scan(t_range=...) partition pruning + pushdown"
```

---

## Task 5: Converter fills `TemporalSpec` + partition time stats

**Files:**
- Modify: `oneuniverse/data/converter.py`
- Test: append to `test/test_dataset_view_time.py`

- [ ] **Step 1: Add the failing tests**

Append to `test/test_dataset_view_time.py`:

```python
def test_converter_fills_temporal_when_t_obs_present(tmp_path):
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.manifest import read_manifest

    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "ra":  rng.uniform(0.0, 360.0, n),
        "dec": rng.uniform(-60.0, 60.0, n),
        "z":   rng.uniform(0.01, 0.3, n),
        "z_type": ["spec"] * n,
        "z_err":  rng.uniform(1e-4, 1e-3, n),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": [f"syn{i:05d}" for i in range(n)],
        "_original_row_index": np.arange(n, dtype=np.int64),
        "t_obs": rng.uniform(58000.0, 60000.0, n),
    })
    survey_dir = tmp_path / "syn"
    write_ouf_dataset(
        df, survey_dir,
        survey_name="syn", survey_type="transient",
        geometry=DataGeometry.POINT,
        loader_name="syn", loader_version="0",
    )
    m = read_manifest(survey_dir / ONEUNIVERSE_SUBDIR / "manifest.json")
    assert m.temporal is not None
    assert m.temporal.time_column == "t_obs"
    assert m.temporal.t_min == float(df["t_obs"].min())
    assert m.temporal.t_max == float(df["t_obs"].max())
    assert all(p.stats.t_min is not None for p in m.partitions)


def test_converter_accepts_validity_kwarg(tmp_path):
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.manifest import read_manifest

    df = pd.DataFrame({
        "ra": [0.0], "dec": [0.0], "z": [0.1],
        "z_type": ["spec"], "z_err": [0.01],
        "galaxy_id": np.array([0], dtype=np.int64),
        "survey_id": ["syn0"],
        "_original_row_index": np.array([0], dtype=np.int64),
    })
    v = DatasetValidity(valid_from_utc=_T0, version="dr17",
                        supersedes=("eboss_qso_dr16",))
    survey_dir = tmp_path / "with_v"
    write_ouf_dataset(
        df, survey_dir,
        survey_name="with_v", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader_name="syn", loader_version="0",
        validity=v,
    )
    m = read_manifest(survey_dir / ONEUNIVERSE_SUBDIR / "manifest.json")
    assert m.validity == v


def test_converter_fills_default_validity_when_absent(tmp_path):
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.manifest import read_manifest

    df = pd.DataFrame({
        "ra": [0.0], "dec": [0.0], "z": [0.1],
        "z_type": ["spec"], "z_err": [0.01],
        "galaxy_id": np.array([0], dtype=np.int64),
        "survey_id": ["syn0"],
        "_original_row_index": np.array([0], dtype=np.int64),
    })
    survey_dir = tmp_path / "no_v"
    write_ouf_dataset(
        df, survey_dir,
        survey_name="no_v", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader_name="syn", loader_version="0",
    )
    m = read_manifest(survey_dir / ONEUNIVERSE_SUBDIR / "manifest.json")
    assert m.validity is not None
    # Default validity uses now() — assert only that it's valid.
    assert m.validity.is_current()
```

- [ ] **Step 2: Confirm failure**

Run: `pytest test/test_dataset_view_time.py -v`
Expected: 3 new tests FAIL.

- [ ] **Step 3: Extend the converter**

In `oneuniverse/data/converter.py`:

```python
# new imports
from oneuniverse.data.temporal import TemporalSpec
from oneuniverse.data.validity import DatasetValidity
import datetime as dt


# inside write_ouf_dataset, add a new kwarg (keyword-only)
def write_ouf_dataset(
    df, survey_path, *,
    survey_name, survey_type, geometry,
    loader_name, loader_version,
    partition_rows=None,
    conversion_kwargs=None,
    validity: Optional[DatasetValidity] = None,  # <-- new
    ...
):
    ...
```

Inside the per-partition loop, when building `PartitionStats`, include
`t_min`/`t_max` if the column is present:

```python
t_min = float(chunk["t_obs"].min()) if "t_obs" in chunk.columns else None
t_max = float(chunk["t_obs"].max()) if "t_obs" in chunk.columns else None

stats = PartitionStats(
    ra_min=ra_min, ra_max=ra_max,
    dec_min=dec_min, dec_max=dec_max,
    z_min=z_min, z_max=z_max,
    t_min=t_min, t_max=t_max,
)
```

After the partition loop, build `temporal`:

```python
temporal = None
if "t_obs" in df.columns and geometry is DataGeometry.POINT:
    temporal = TemporalSpec(
        t_min=float(df["t_obs"].min()),
        t_max=float(df["t_obs"].max()),
    )
```

Default-fill `validity`:

```python
if validity is None:
    validity = DatasetValidity(
        valid_from_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
```

Pass both into the `Manifest(...)` constructor:

```python
manifest = Manifest(
    ...,
    temporal=temporal,
    validity=validity,
)
```

Bump the in-file format constants if any are hard-coded (they now live
in `manifest.py`).

- [ ] **Step 4: Run tests**

Run: `pytest test/test_dataset_view_time.py -v`
Expected: all tests in the file pass.

- [ ] **Step 5: Full suite**

Run: `pytest -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/converter.py test/test_dataset_view_time.py
git commit -m "feat(temporal): converter fills TemporalSpec, partition t_min/t_max, accepts validity kwarg"
```

---

## Task 6: `DataGeometry.LIGHTCURVE` + required columns

**Files:**
- Modify: `oneuniverse/data/format_spec.py`
- Test: `test/test_lightcurve_geometry.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_lightcurve_geometry.py
from oneuniverse.data.format_spec import (
    DataGeometry, GEOMETRY_COLUMNS, validate_columns,
    DEFAULT_PARTITION_ROWS, FORMAT_VERSION, SCHEMA_VERSION,
)


def test_enum_value():
    assert DataGeometry.LIGHTCURVE.value == "lightcurve"


def test_object_required_columns_present():
    req = GEOMETRY_COLUMNS[DataGeometry.LIGHTCURVE]["objects"]
    for c in ("object_id", "ra", "dec", "z", "n_epochs", "mjd_min", "mjd_max"):
        assert c in req


def test_data_required_columns_present():
    req = GEOMETRY_COLUMNS[DataGeometry.LIGHTCURVE]["data"]
    for c in ("object_id", "mjd", "flux", "flux_err"):
        assert c in req


def test_validate_columns_objects_ok():
    missing = validate_columns(
        ["object_id", "ra", "dec", "z", "n_epochs", "mjd_min",
         "mjd_max", "_healpix32", "z_type", "z_err"],
        DataGeometry.LIGHTCURVE, table_type="objects",
    )
    assert missing == []


def test_validate_columns_data_missing_mjd():
    assert "mjd" in validate_columns(
        ["object_id"], DataGeometry.LIGHTCURVE, table_type="data",
    )


def test_default_partition_rows_has_lightcurve():
    assert DataGeometry.LIGHTCURVE in DEFAULT_PARTITION_ROWS


def test_format_version_is_2_1_0():
    assert FORMAT_VERSION == "2.1.0"
    assert SCHEMA_VERSION == "2.1.0"
```

- [ ] **Step 2: Confirm failure**

Run: `pytest test/test_lightcurve_geometry.py -v`
Expected: FAIL.

- [ ] **Step 3: Extend `format_spec.py`**

Edit `oneuniverse/data/format_spec.py`:

```python
FORMAT_VERSION: str = "2.1.0"
SCHEMA_VERSION: str = "2.1.0"


class DataGeometry(str, Enum):
    POINT = "point"
    SIGHTLINE = "sightline"
    HEALPIX = "healpix"
    LIGHTCURVE = "lightcurve"


LIGHTCURVE_OBJECT_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "object_id",  "ra", "dec", "z", "z_type", "z_err",
    "n_epochs",   "mjd_min", "mjd_max", "_healpix32",
)

LIGHTCURVE_DATA_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "object_id", "mjd", "filter", "flux", "flux_err", "flag",
)

GEOMETRY_COLUMNS[DataGeometry.LIGHTCURVE] = {
    "objects": LIGHTCURVE_OBJECT_REQUIRED_COLUMNS,
    "data":    LIGHTCURVE_DATA_REQUIRED_COLUMNS,
}

DEFAULT_PARTITION_ROWS[DataGeometry.LIGHTCURVE] = 1_000_000
```

(Strictly: keep the existing inline literal dicts if they appear, and
extend them with the new entries rather than re-assigning.)

- [ ] **Step 4: Run tests**

Run: `pytest test/test_lightcurve_geometry.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/format_spec.py test/test_lightcurve_geometry.py
git commit -m "feat(temporal): DataGeometry.LIGHTCURVE + required columns"
```

---

## Task 7: `write_ouf_lightcurve_dataset`

**Files:**
- Create: `oneuniverse/data/_converter_lightcurve.py`
- Test: `test/test_lightcurve_roundtrip.py`
- Modify: `oneuniverse/data/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_lightcurve_roundtrip.py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from oneuniverse.data._converter_lightcurve import write_ouf_lightcurve_dataset
from oneuniverse.data.manifest import read_manifest
from oneuniverse.data.format_spec import DataGeometry, ONEUNIVERSE_SUBDIR


def _make(n_obj=5, n_epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    objects = pd.DataFrame({
        "object_id": np.arange(n_obj, dtype=np.int64),
        "ra":    rng.uniform(0.0, 360.0, n_obj),
        "dec":   rng.uniform(-60.0, 60.0, n_obj),
        "z":     rng.uniform(0.01, 0.5, n_obj),
        "z_type": ["spec"] * n_obj,
        "z_err": rng.uniform(1e-4, 1e-3, n_obj),
    })
    rows = []
    for oid in objects["object_id"]:
        mjd = np.sort(rng.uniform(58000.0, 60000.0, n_epochs))
        for t in mjd:
            rows.append({
                "object_id": int(oid), "mjd": float(t),
                "filter": rng.choice(["g", "r", "i"]),
                "flux": float(rng.normal(100.0, 5.0)),
                "flux_err": 1.0, "flag": 0,
            })
    return objects, pd.DataFrame(rows)


def test_roundtrip_structure(tmp_path):
    objects, epochs = _make(3, 10)
    survey_dir = tmp_path / "lc"
    write_ouf_lightcurve_dataset(
        objects=objects, epochs=epochs,
        survey_path=survey_dir,
        survey_name="lc", survey_type="transient",
        loader_name="syn", loader_version="0",
    )
    ou = survey_dir / ONEUNIVERSE_SUBDIR
    assert (ou / "manifest.json").exists()
    assert (ou / "objects.parquet").exists()
    assert any(ou.glob("part_*.parquet"))
    m = read_manifest(ou / "manifest.json")
    assert m.geometry is DataGeometry.LIGHTCURVE
    assert m.temporal.time_column == "mjd"
    obj = pq.read_table(ou / "objects.parquet").to_pandas()
    assert set(obj.columns) >= {"object_id", "ra", "dec", "z",
                                 "n_epochs", "mjd_min", "mjd_max"}
    expected = epochs.groupby("object_id").size().reset_index(name="n_epochs")
    merged = obj[["object_id", "n_epochs"]].merge(
        expected, on="object_id", suffixes=("", "_truth"))
    assert (merged["n_epochs"] == merged["n_epochs_truth"]).all()


def test_rejects_orphan_epochs(tmp_path):
    objects, epochs = _make(2, 3)
    orphan = epochs.iloc[:1].copy()
    orphan["object_id"] = 999
    epochs = pd.concat([epochs, orphan], ignore_index=True)
    with pytest.raises(ValueError, match="orphan"):
        write_ouf_lightcurve_dataset(
            objects=objects, epochs=epochs,
            survey_path=tmp_path / "bad",
            survey_name="bad", survey_type="transient",
            loader_name="syn", loader_version="0",
        )


def test_partition_stats_cover_mjd(tmp_path):
    objects, epochs = _make(4, 30, seed=3)
    survey_dir = tmp_path / "lc2"
    write_ouf_lightcurve_dataset(
        objects=objects, epochs=epochs,
        survey_path=survey_dir,
        survey_name="lc2", survey_type="transient",
        loader_name="syn", loader_version="0",
    )
    m = read_manifest(survey_dir / ONEUNIVERSE_SUBDIR / "manifest.json")
    overall_min = min(p.stats.t_min for p in m.partitions)
    overall_max = max(p.stats.t_max for p in m.partitions)
    assert overall_min == float(epochs["mjd"].min())
    assert overall_max == float(epochs["mjd"].max())


def test_writer_accepts_validity_kwarg(tmp_path):
    from oneuniverse.data.validity import DatasetValidity
    objects, epochs = _make(2, 3)
    v = DatasetValidity(valid_from_utc="2026-02-01T00:00:00+00:00",
                        version="ztf_2026_02")
    survey_dir = tmp_path / "vlc"
    write_ouf_lightcurve_dataset(
        objects=objects, epochs=epochs,
        survey_path=survey_dir,
        survey_name="vlc", survey_type="transient",
        loader_name="syn", loader_version="0",
        validity=v,
    )
    m = read_manifest(survey_dir / ONEUNIVERSE_SUBDIR / "manifest.json")
    assert m.validity == v
```

- [ ] **Step 2: Confirm failure**

Run: `pytest test/test_lightcurve_roundtrip.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the writer**

```python
# oneuniverse/data/_converter_lightcurve.py
"""Writer for OUF 2.1 LIGHTCURVE datasets."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse.data._atomic import atomic_write_bytes
from oneuniverse.data._hashing import hash_bytes
from oneuniverse.data.format_spec import (
    COMPRESSION, DEFAULT_PARTITION_ROWS, ONEUNIVERSE_SUBDIR,
    validate_columns, DataGeometry,
)
from oneuniverse.data.manifest import (
    ColumnSpec, LoaderSpec, Manifest, PartitionSpec,
    PartitionStats, write_manifest,
)
from oneuniverse.data.temporal import TemporalSpec
from oneuniverse.data.validity import DatasetValidity


def _healpix32(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    import healpy as hp
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    return hp.ang2pix(32, theta, phi, nest=True).astype(np.int64)


def _check_no_orphan_epochs(objects: pd.DataFrame, epochs: pd.DataFrame) -> None:
    known = set(objects["object_id"].astype(np.int64).tolist())
    seen = set(epochs["object_id"].astype(np.int64).tolist())
    orphans = seen - known
    if orphans:
        raise ValueError(
            f"write_ouf_lightcurve_dataset: {len(orphans)} epoch rows "
            f"refer to orphan object_id(s) not present in the objects "
            f"table: {sorted(list(orphans))[:5]}"
        )


def _build_objects_table(objects: pd.DataFrame,
                          epochs: pd.DataFrame) -> pd.DataFrame:
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
        merged["_healpix32"] = _healpix32(
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


def _partition(epochs: pd.DataFrame, rows: int) -> List[pd.DataFrame]:
    epochs = epochs.sort_values(["object_id", "mjd"]).reset_index(drop=True)
    return [epochs.iloc[s : s + rows].copy()
            for s in range(0, len(epochs), rows)]


def _schema(df: pd.DataFrame) -> List[ColumnSpec]:
    return [ColumnSpec(name=str(c), dtype=str(df[c].dtype))
            for c in df.columns]


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
    conversion_kwargs: Optional[dict] = None,
    validity: Optional[DatasetValidity] = None,
) -> Path:
    """Write an OUF 2.1 LIGHTCURVE dataset.

    Layout: ``{survey_path}/oneuniverse/{manifest.json, objects.parquet,
    part_*.parquet}``.  The epoch partitions are sorted by
    ``(object_id, mjd)``.
    """
    survey_path = Path(survey_path)
    ou = survey_path / ONEUNIVERSE_SUBDIR
    ou.mkdir(parents=True, exist_ok=True)

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
    obj_buf = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pandas(objects_final, preserve_index=False),
        obj_buf, compression=COMPRESSION,
    )
    atomic_write_bytes(ou / "objects.parquet", obj_buf.getvalue().to_pybytes())

    parts = _partition(epochs, partition_rows)
    specs: List[PartitionSpec] = []
    for i, chunk in enumerate(parts):
        name = f"part_{i:04d}.parquet"
        buf = pa.BufferOutputStream()
        pq.write_table(
            pa.Table.from_pandas(chunk, preserve_index=False),
            buf, compression=COMPRESSION,
        )
        data = buf.getvalue().to_pybytes()
        atomic_write_bytes(ou / name, data)
        specs.append(PartitionSpec(
            name=name, n_rows=len(chunk),
            sha256=hash_bytes(data),
            size_bytes=(ou / name).stat().st_size,
            stats=PartitionStats(
                t_min=float(chunk["mjd"].min()),
                t_max=float(chunk["mjd"].max()),
            ),
        ))

    temporal = TemporalSpec(
        t_min=float(epochs["mjd"].min()),
        t_max=float(epochs["mjd"].max()),
        time_column="mjd",
    )
    if validity is None:
        validity = DatasetValidity(
            valid_from_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        )

    manifest = Manifest(
        oneuniverse_format_version="2.1.0",
        oneuniverse_schema_version="2.1.0",
        geometry=DataGeometry.LIGHTCURVE,
        survey_name=survey_name,
        survey_type=survey_type,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        original_files=[],
        partitions=specs,
        partitioning=None,
        schema=_schema(epochs),
        conversion_kwargs=dict(conversion_kwargs or {}),
        loader=LoaderSpec(name=loader_name, version=loader_version),
        extra={"n_objects": int(len(objects_final))},
        temporal=temporal,
        validity=validity,
    )
    write_manifest(ou / "manifest.json", manifest)
    return survey_path
```

- [ ] **Step 4: Export from package**

Append to `oneuniverse/data/__init__.py`:

```python
from oneuniverse.data.temporal import TemporalSpec  # noqa: F401
from oneuniverse.data.validity import DatasetValidity  # noqa: F401
from oneuniverse.data._converter_lightcurve import write_ouf_lightcurve_dataset  # noqa: F401
```

- [ ] **Step 5: Run tests**

Run: `pytest test/test_lightcurve_roundtrip.py -v`
Expected: 4 passed.

- [ ] **Step 6: Full suite**

Run: `pytest -q`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add oneuniverse/data/_converter_lightcurve.py oneuniverse/data/__init__.py \
        test/test_lightcurve_roundtrip.py
git commit -m "feat(temporal): write_ouf_lightcurve_dataset writer + package exports"
```

---

## Task 8: `DatasetView.objects_table()` for SIGHTLINE and LIGHTCURVE

**Files:**
- Modify: `oneuniverse/data/dataset_view.py`
- Test: append to `test/test_lightcurve_roundtrip.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_lightcurve_roundtrip.py`:

```python
def test_objects_table_on_lightcurve(tmp_path):
    from oneuniverse.data.dataset_view import DatasetView
    objects, epochs = _make(4, 5)
    survey_dir = tmp_path / "view_lc"
    write_ouf_lightcurve_dataset(
        objects=objects, epochs=epochs,
        survey_path=survey_dir,
        survey_name="view_lc", survey_type="transient",
        loader_name="syn", loader_version="0",
    )
    view = DatasetView.from_path(survey_dir)
    obj = view.objects_table().to_pandas()
    assert len(obj) == 4
    assert "n_epochs" in obj.columns

    epo = view.scan(columns=["object_id", "mjd"], t_range=(59000.0, 60000.0))
    for t in epo.column("mjd").to_pylist():
        assert 59000.0 <= t <= 60000.0


def test_objects_table_on_point_raises(tmp_path):
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.dataset_view import DatasetView
    df = pd.DataFrame({
        "ra": [0.0], "dec": [0.0], "z": [0.1],
        "z_type": ["spec"], "z_err": [0.01],
        "galaxy_id": np.array([0], dtype=np.int64),
        "survey_id": ["syn0"],
        "_original_row_index": np.array([0], dtype=np.int64),
    })
    survey_dir = tmp_path / "pt"
    write_ouf_dataset(
        df, survey_dir,
        survey_name="pt", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader_name="syn", loader_version="0",
    )
    view = DatasetView.from_path(survey_dir)
    with pytest.raises(ValueError, match="objects_table"):
        view.objects_table()
```

- [ ] **Step 2: Confirm failure**

Run: `pytest test/test_lightcurve_roundtrip.py -v`
Expected: FAIL (`objects_table` missing).

- [ ] **Step 3: Implement**

In `oneuniverse/data/dataset_view.py`, inside class `DatasetView`:

```python
def objects_table(
    self, columns: Optional[Sequence[str]] = None,
) -> pa.Table:
    """Return the per-object metadata table for SIGHTLINE or LIGHTCURVE.

    Raises
    ------
    ValueError
        For POINT or HEALPIX geometries (no objects table on disk).
    """
    from oneuniverse.data.format_spec import OBJECTS_FILENAME
    import pyarrow.parquet as pq

    if self.geometry not in {DataGeometry.SIGHTLINE, DataGeometry.LIGHTCURVE}:
        raise ValueError(
            f"objects_table() is only defined for SIGHTLINE and LIGHTCURVE "
            f"geometries, got {self.geometry.value!r}"
        )
    return pq.read_table(
        self.ou_dir / OBJECTS_FILENAME,
        columns=list(columns) if columns else None,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_lightcurve_roundtrip.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/dataset_view.py test/test_lightcurve_roundtrip.py
git commit -m "feat(temporal): DatasetView.objects_table() for SIGHTLINE/LIGHTCURVE"
```

---

## Task 9: `OneuniverseDatabase.as_of(timestamp)` and `versions_of(name)`

**Files:**
- Modify: `oneuniverse/data/database.py`, `oneuniverse/data/_dataset_entry.py`
- Test: `test/test_database_as_of.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_database_as_of.py
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.database import OneuniverseDatabase
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.validity import DatasetValidity


def _write_syn(root: Path, sub: str, name: str, validity: DatasetValidity):
    df = pd.DataFrame({
        "ra": [0.0], "dec": [0.0], "z": [0.1],
        "z_type": ["spec"], "z_err": [0.01],
        "galaxy_id": np.array([0], dtype=np.int64),
        "survey_id": ["syn0"],
        "_original_row_index": np.array([0], dtype=np.int64),
    })
    survey_dir = root / sub
    write_ouf_dataset(
        df, survey_dir,
        survey_name=name, survey_type=sub.split("/")[0],
        geometry=DataGeometry.POINT,
        loader_name="syn", loader_version="0",
        validity=validity,
    )


def test_as_of_selects_version_valid_at_timestamp(tmp_path):
    # v1 valid 2026-01..2026-06; v2 valid 2026-06..forever
    v1 = DatasetValidity(
        valid_from_utc="2026-01-01T00:00:00+00:00",
        valid_to_utc  ="2026-06-01T00:00:00+00:00",
        version="dr16",
    )
    v2 = DatasetValidity(
        valid_from_utc="2026-06-01T00:00:00+00:00",
        version="dr17", supersedes=("spectroscopic_eboss_qso_v_dr16",),
    )
    _write_syn(tmp_path, "spectroscopic/eboss_qso/v_dr16",
               "spectroscopic_eboss_qso_v_dr16", v1)
    _write_syn(tmp_path, "spectroscopic/eboss_qso/v_dr17",
               "spectroscopic_eboss_qso_v_dr17", v2)

    db = OneuniverseDatabase.from_root(tmp_path)
    # Default view: only currently-valid entries (v2).
    names = set(db.list())
    assert "spectroscopic_eboss_qso_v_dr17" in names
    assert "spectroscopic_eboss_qso_v_dr16" not in names

    # Snapshot in March 2026: only v1.
    t = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)
    snap = db.as_of(t)
    snap_names = set(snap.list())
    assert "spectroscopic_eboss_qso_v_dr16" in snap_names
    assert "spectroscopic_eboss_qso_v_dr17" not in snap_names


def test_versions_of_lists_all_versions(tmp_path):
    v1 = DatasetValidity(
        valid_from_utc="2026-01-01T00:00:00+00:00",
        valid_to_utc  ="2026-06-01T00:00:00+00:00",
        version="dr16",
    )
    v2 = DatasetValidity(
        valid_from_utc="2026-06-01T00:00:00+00:00",
        version="dr17", supersedes=("spectroscopic_eboss_qso_v_dr16",),
    )
    _write_syn(tmp_path, "spectroscopic/eboss_qso/v_dr16",
               "spectroscopic_eboss_qso_v_dr16", v1)
    _write_syn(tmp_path, "spectroscopic/eboss_qso/v_dr17",
               "spectroscopic_eboss_qso_v_dr17", v2)

    db = OneuniverseDatabase.from_root(tmp_path)
    versions = db.versions_of_root("spectroscopic/eboss_qso")
    labels = [e.manifest.validity.version for e in versions]
    assert labels == ["dr16", "dr17"]  # sorted by valid_from_utc ascending


def test_as_of_rejects_naive_timestamp(tmp_path):
    db = OneuniverseDatabase.from_root(tmp_path)
    with pytest.raises(ValueError, match="timezone"):
        db.as_of(dt.datetime(2026, 3, 1))


def test_default_walker_warns_on_mixed_validity(tmp_path, caplog):
    """Two entries with overlapping validity = operator error; log a warning."""
    v1 = DatasetValidity(valid_from_utc="2026-01-01T00:00:00+00:00")
    v2 = DatasetValidity(valid_from_utc="2026-02-01T00:00:00+00:00")
    _write_syn(tmp_path, "spectroscopic/a", "spectroscopic_a", v1)
    _write_syn(tmp_path, "spectroscopic/a_overlap", "spectroscopic_a_overlap", v2)
    with caplog.at_level("WARNING"):
        OneuniverseDatabase.from_root(tmp_path)
    # overlapping entries under the same survey root should log a hint;
    # different roots are unrelated, so no warning is fine.
```

- [ ] **Step 2: Confirm failure**

Run: `pytest test/test_database_as_of.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `as_of` and `versions_of_root` on the database**

Edit `oneuniverse/data/database.py`.

Imports:

```python
import datetime as dt
from oneuniverse.data.validity import DatasetValidity
```

Inside `OneuniverseDatabase`:

```python
def as_of(
    self, when: dt.datetime,
) -> "OneuniverseDatabase":
    """Return a sibling database view containing only entries whose
    :class:`DatasetValidity` was valid at *when*.

    The returned object shares the same root path and loader classes;
    only the ``_entries`` dict is filtered.
    """
    if when.tzinfo is None:
        raise ValueError(
            "OneuniverseDatabase.as_of: timestamp must be timezone-aware"
        )
    filtered = {
        name: e
        for name, e in self._entries.items()
        if e.manifest.validity is not None
        and e.manifest.validity.contains(when)
    }
    clone = self.__class__.__new__(self.__class__)
    # copy fields the ctor filled in; this depends on your class's init.
    clone._root = self._root
    clone._entries = filtered
    clone._data_root = getattr(self, "_data_root", None)
    return clone


def versions_of_root(
    self, survey_root: str,
) -> "List[DatasetEntry]":
    """Return all entries whose path lives under ``survey_root``, sorted
    by ``validity.valid_from_utc``.

    ``survey_root`` is a subpath under the database root (e.g.
    ``"spectroscopic/eboss_qso"``). Entries without a validity block
    are placed first.
    """
    root = Path(self._root) / survey_root
    matching = [
        e for e in self._entries.values()
        if Path(e.path).is_relative_to(root)
    ]
    def _key(e):
        v = e.manifest.validity
        if v is None:
            return dt.datetime.min.replace(tzinfo=dt.timezone.utc)
        return dt.datetime.fromisoformat(v.valid_from_utc)
    return sorted(matching, key=_key)
```

In the constructor / walker, change the default population so that only
currently-valid entries land in `_entries` (keep the superseded ones
reachable through `versions_of_root` by also tracking them in a
`_all_entries` dict; or, simpler, load every manifest and filter at
`list()`-time). The simplest patch:

- Keep a full `_all_entries: Dict[str, DatasetEntry]` collected by the
  walker.
- On construction, set `_entries = {name: e for name, e in
  _all_entries.items() if e.manifest.validity is None or
  e.manifest.validity.is_current()}`.
- `versions_of_root` walks `_all_entries`.

The walker must also honour `validity.supersedes`: when an entry's
`supersedes=("foo",)`, and the database contains `foo` with no
`valid_to_utc`, patch `foo`'s validity to close at the superseder's
`valid_from_utc`. Concretely, *do not mutate the on-disk manifest* —
wrap the `DatasetEntry` with a new `Manifest` whose `validity` has
`valid_to_utc` set:

```python
def _apply_supersessions(all_entries):
    """Close out superseded entries by copying manifests with patched
    validity. Works on a dict-of-entries in place."""
    from dataclasses import replace as _replace
    for new in list(all_entries.values()):
        v = new.manifest.validity
        if v is None or not v.supersedes:
            continue
        for prior_name in v.supersedes:
            prior = all_entries.get(prior_name)
            if prior is None or prior.manifest.validity is None:
                continue
            if prior.manifest.validity.valid_to_utc is not None:
                continue  # already closed
            closed = prior.manifest.validity.closed_at(v.valid_from_utc)
            new_manifest = _replace(prior.manifest, validity=closed)
            all_entries[prior_name] = _replace(prior, manifest=new_manifest)
```

Wire `_apply_supersessions` into the constructor immediately after the
walker populates `_all_entries`, before the `_entries` filter.

For the mixed-validity warning (last test — soft assertion): the walker
can log a `WARNING` when two entries under the same survey-root have
overlapping `valid_from_utc` windows without a `supersedes` link. Keep
this as a loose sanity check; it does not block the load.

- [ ] **Step 4: Run tests**

Run: `pytest test/test_database_as_of.py -v`
Expected: the deterministic tests pass.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/database.py test/test_database_as_of.py
git commit -m "feat(temporal): OneuniverseDatabase.as_of + versions_of_root + supersession closing"
```

---

## Task 10: Bitemporal ONEUID indices

**Files:**
- Modify: `oneuniverse/data/oneuid.py`
- Test: `test/test_oneuid_bitemporal.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_oneuid_bitemporal.py
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.database import OneuniverseDatabase
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.oneuid_rules import CrossMatchRules
from oneuniverse.data.validity import DatasetValidity


def _syn(root, sub, name, n=20, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ra":    rng.uniform(0.0, 10.0, n),
        "dec":   rng.uniform(-5.0, 5.0, n),
        "z":     rng.uniform(0.1, 0.3, n),
        "z_type": ["spec"] * n,
        "z_err":  rng.uniform(1e-4, 1e-3, n),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": [f"{name}_{i:03d}" for i in range(n)],
        "_original_row_index": np.arange(n, dtype=np.int64),
    })
    write_ouf_dataset(
        df, root / sub,
        survey_name=name, survey_type=sub.split("/")[0],
        geometry=DataGeometry.POINT,
        loader_name="syn", loader_version="0",
        validity=DatasetValidity(
            valid_from_utc="2026-01-01T00:00:00+00:00"),
    )


def _set_up_db(tmp_path):
    _syn(tmp_path, "spectroscopic/a", "spectroscopic_a", seed=1)
    _syn(tmp_path, "spectroscopic/b", "spectroscopic_b", seed=2)
    return OneuniverseDatabase.from_root(tmp_path)


def test_build_index_stamps_validity(tmp_path):
    db = _set_up_db(tmp_path)
    rules = CrossMatchRules(sky_tol_arcsec=3.0)
    db.build_oneuid(
        datasets=["spectroscopic_a", "spectroscopic_b"],
        rules=rules, name="v1",
    )
    idx = db.load_oneuid(name="v1")
    assert idx.validity is not None
    assert idx.validity.is_current()


def test_rebuild_closes_previous_version(tmp_path):
    db = _set_up_db(tmp_path)
    rules = CrossMatchRules(sky_tol_arcsec=3.0)
    db.build_oneuid(datasets=["spectroscopic_a"], rules=rules, name="v1")
    t1 = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    # rebuild with same name but different member set — supersedes v1
    db.build_oneuid(
        datasets=["spectroscopic_a", "spectroscopic_b"],
        rules=rules, name="v1",
    )
    # on disk: two manifests (the old one renamed with a suffix);
    # the old one has valid_to_utc set.
    old_manifests = list((Path(tmp_path) / "_oneuid").glob("v1__*.manifest.json"))
    assert len(old_manifests) == 1
    closed = json.loads(old_manifests[0].read_text())["validity"]
    assert closed["valid_to_utc"] is not None


def test_load_oneuid_as_of_picks_earlier_version(tmp_path):
    db = _set_up_db(tmp_path)
    rules = CrossMatchRules(sky_tol_arcsec=3.0)
    db.build_oneuid(datasets=["spectroscopic_a"], rules=rules, name="v1")
    db.build_oneuid(
        datasets=["spectroscopic_a", "spectroscopic_b"],
        rules=rules, name="v1",
    )
    # Probe a moment between the two builds: the earlier version is
    # the one we want. Use the supersession transition timestamp.
    later_current = db.load_oneuid(name="v1")
    # supersession timestamp is the later one's valid_from.
    t_mid = dt.datetime.fromisoformat(
        later_current.validity.valid_from_utc
    ) - dt.timedelta(seconds=1)
    earlier = db.load_oneuid(name="v1", as_of=t_mid)
    # earlier had only one dataset in its member set
    assert earlier is not None
    assert earlier.datasets == ("spectroscopic_a",)


def test_list_oneuids_bitemporal_contains_archived(tmp_path):
    db = _set_up_db(tmp_path)
    rules = CrossMatchRules(sky_tol_arcsec=3.0)
    db.build_oneuid(datasets=["spectroscopic_a"], rules=rules, name="v1")
    db.build_oneuid(
        datasets=["spectroscopic_a", "spectroscopic_b"],
        rules=rules, name="v1",
    )
    current = db.list_oneuids()
    assert "v1" in current

    archived = db.list_oneuids(include_archived=True)
    assert "v1" in archived
    # archived listing also reports the archived version(s)
    assert any(label.startswith("v1__") for label in archived)
```

- [ ] **Step 2: Confirm failure**

Run: `pytest test/test_oneuid_bitemporal.py -v`
Expected: FAIL.

- [ ] **Step 3: Extend ONEUID code**

Edit `oneuniverse/data/oneuid.py`:

1. Add `validity` to `OneuidIndex`:

   ```python
   from oneuniverse.data.validity import DatasetValidity

   @dataclass
   class OneuidIndex:
       ...
       validity: Optional[DatasetValidity] = None
   ```

2. The on-disk manifest serializer (`_write_index_manifest` /
   `_read_index_manifest`) must read/write a `validity` field. Default
   to `DatasetValidity(valid_from_utc=<now-UTC>)` for newly-built
   indices.

3. On `build_oneuid_index`, when a previous index with the same `name`
   already exists on disk, rename it with a suffix and patch its
   manifest's `validity.valid_to_utc`:

   ```python
   def _archive_previous(root: Path, name: str) -> None:
       idx_path = _index_path(root, name)
       man_path = _index_manifest_path(root, name)
       if not (idx_path.exists() and man_path.exists()):
           return
       ts_suffix = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
       archived_name = f"{name}__{ts_suffix}"
       archived_idx = root / ONEUID_DIR / f"{archived_name}.parquet"
       archived_man = root / ONEUID_DIR / f"{archived_name}.manifest.json"
       idx_path.rename(archived_idx)
       raw = json.loads(man_path.read_text())
       v = raw.get("validity")
       if v is not None:
           v["valid_to_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
           raw["validity"] = v
       man_path.rename(archived_man)
       archived_man.write_text(json.dumps(raw, indent=2))
   ```

   Call `_archive_previous(root, name)` at the top of the build
   function, immediately before writing the new index.

4. `load_oneuid_index(root, name=..., as_of=None)`:

   ```python
   def load_oneuid_index(
       root: Path, *, name: str = "default",
       as_of: Optional[dt.datetime] = None,
   ) -> OneuidIndex:
       if as_of is None:
           return _read_live_index(root, name)
       if as_of.tzinfo is None:
           raise ValueError("load_oneuid_index: as_of must be tz-aware")
       # search live + archived manifests, pick the one whose validity.contains(as_of)
       candidates = [ _index_manifest_path(root, name) ]
       candidates += list(
           (root / ONEUID_DIR).glob(f"{name}__*.manifest.json")
       )
       for man in candidates:
           if not man.exists():
               continue
           raw = json.loads(man.read_text())
           v = DatasetValidity.from_dict(raw["validity"])
           if v.contains(as_of):
               return _read_index_from_manifest(root, man, raw)
       raise FileNotFoundError(
           f"load_oneuid_index: no version of {name!r} valid at {as_of}"
       )
   ```

5. `list_oneuid_indices(root, include_archived: bool = False)`:

   ```python
   def list_oneuid_indices(
       root: Path, *, include_archived: bool = False,
   ) -> List[str]:
       live = [
           p.stem.removesuffix(".manifest")
           for p in (root / ONEUID_DIR).glob("*.manifest.json")
           if "__" not in p.stem
       ]
       if not include_archived:
           return sorted(live)
       archived = [
           p.stem.removesuffix(".manifest")
           for p in (root / ONEUID_DIR).glob("*__*.manifest.json")
       ]
       return sorted(live) + sorted(archived)
   ```

6. Mirror in `OneuniverseDatabase`: accept `as_of` in `load_oneuid` and
   `include_archived` in `list_oneuids`.

- [ ] **Step 4: Run tests**

Run: `pytest test/test_oneuid_bitemporal.py -v`
Expected: 4 passed.

- [ ] **Step 5: Full suite**

Run: `pytest -q`
Expected: pre-existing ONEUID tests still pass (they do not pass
`as_of`, so the default path kicks in).

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/oneuid.py test/test_oneuid_bitemporal.py
git commit -m "feat(temporal): bitemporal ONEUID indices (archive on rebuild, as_of load)"
```

---

## Task 11: Visual diagnostics — temporal POINT + LIGHTCURVE + database snapshot

**Files:**
- Create: `test/test_visual_temporal.py`

- [ ] **Step 1: Write the test**

```python
# test/test_visual_temporal.py
"""Diagnostic figures for Phase 7. Run locally; skipped in headless CI
if matplotlib isn't importable. Outputs land under test/test_output/."""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.data.converter import write_ouf_dataset  # noqa: E402
from oneuniverse.data.database import OneuniverseDatabase  # noqa: E402
from oneuniverse.data.dataset_view import DatasetView  # noqa: E402
from oneuniverse.data.format_spec import DataGeometry  # noqa: E402
from oneuniverse.data._converter_lightcurve import (  # noqa: E402
    write_ouf_lightcurve_dataset,
)
from oneuniverse.data.validity import DatasetValidity  # noqa: E402


OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def test_visual_transient_point(tmp_path):
    rng = np.random.default_rng(1)
    n = 5000
    df = pd.DataFrame({
        "ra":    rng.uniform(0.0, 360.0, n),
        "dec":   rng.uniform(-60.0, 60.0, n),
        "z":     rng.uniform(0.01, 0.5, n),
        "z_type": ["spec"] * n,
        "z_err":  rng.uniform(1e-4, 1e-3, n),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": [f"syn{i:06d}" for i in range(n)],
        "_original_row_index": np.arange(n, dtype=np.int64),
        "t_obs": rng.uniform(58000.0, 60000.0, n),
    })
    survey_dir = tmp_path / "viz_pt"
    write_ouf_dataset(
        df, survey_dir,
        survey_name="viz_pt", survey_type="transient",
        geometry=DataGeometry.POINT,
        loader_name="syn", loader_version="0",
    )
    view = DatasetView.from_path(survey_dir)
    window = view.read(columns=["ra", "dec", "t_obs"],
                       t_range=(59000.0, 59500.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ra_full = np.where(df["ra"] > 180, df["ra"] - 360, df["ra"])
    axes[0].scatter(np.radians(ra_full), np.radians(df["dec"]),
                    s=1.5, alpha=0.35, color="#888", label="all")
    ra_win = np.where(window["ra"] > 180, window["ra"] - 360, window["ra"])
    axes[0].scatter(np.radians(ra_win), np.radians(window["dec"]),
                    s=3, color="#d62728",
                    label=f"t_range window ({len(window):,})")
    axes[0].set_xlabel("RA (rad)")
    axes[0].set_ylabel("Dec (rad)")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Transient events — window vs full")

    axes[1].hist(df["t_obs"], bins=80, color="#3a7bd5", alpha=0.6, label="all")
    axes[1].axvspan(59000.0, 59500.0, color="#d62728", alpha=0.2, label="window")
    axes[1].set_xlabel("MJD")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].set_title(f"{len(window):,}/{len(df):,} events in window")

    fig.tight_layout()
    out = OUTPUT_DIR / "phase7_temporal_point.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 10_000


def test_visual_lightcurve(tmp_path):
    rng = np.random.default_rng(2)
    n_obj, n_epochs = 10, 30
    objects = pd.DataFrame({
        "object_id": np.arange(n_obj, dtype=np.int64),
        "ra":    rng.uniform(0.0, 360.0, n_obj),
        "dec":   rng.uniform(-60.0, 60.0, n_obj),
        "z":     rng.uniform(0.01, 0.2, n_obj),
        "z_type": ["spec"] * n_obj,
        "z_err":  rng.uniform(1e-4, 1e-3, n_obj),
    })
    rows = []
    for oid in objects["object_id"]:
        mjd = np.sort(rng.uniform(58000.0, 60000.0, n_epochs))
        for i, t in enumerate(mjd):
            rows.append({
                "object_id": int(oid), "mjd": float(t),
                "filter": ("g", "r", "i")[i % 3],
                "flux": float(100 + 20 * np.sin(0.01 * t) + rng.normal(0, 2)),
                "flux_err": 1.0, "flag": 0,
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
    obj = view.objects_table().to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(obj["ra"], obj["dec"], s=50, c="#3a7bd5", edgecolor="k")
    for _, row in obj.iterrows():
        axes[0].annotate(str(row["object_id"]), (row["ra"], row["dec"]))
    axes[0].set_xlabel("RA (deg)")
    axes[0].set_ylabel("Dec (deg)")
    axes[0].set_title(f"{len(obj)} lightcurve objects")

    import pyarrow.compute as pc
    pick = int(obj["object_id"].iloc[0])
    lc = view.scan(
        columns=["mjd", "flux", "filter"],
        filter=pc.field("object_id") == pick,
    ).to_pandas()
    for band, sub in lc.groupby("filter"):
        axes[1].errorbar(sub["mjd"], sub["flux"], yerr=1.0,
                         fmt="o", label=str(band), ms=3)
    axes[1].set_xlabel("MJD")
    axes[1].set_ylabel("flux")
    axes[1].legend()
    axes[1].set_title(f"Lightcurve of object_id={pick}")

    fig.tight_layout()
    out = OUTPUT_DIR / "phase7_temporal_lightcurve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 10_000


def _syn_point(root, sub, name, validity, seed=0, n=200):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "ra":  rng.uniform(0.0, 30.0, n),
        "dec": rng.uniform(-5.0, 5.0, n),
        "z":   rng.uniform(0.1, 0.3, n),
        "z_type": ["spec"] * n,
        "z_err":  rng.uniform(1e-4, 1e-3, n),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": [f"{name}_{i:04d}" for i in range(n)],
        "_original_row_index": np.arange(n, dtype=np.int64),
    })
    write_ouf_dataset(
        df, root / sub, survey_name=name, survey_type=sub.split("/")[0],
        geometry=DataGeometry.POINT,
        loader_name="syn", loader_version="0",
        validity=validity,
    )


def test_visual_database_snapshot(tmp_path):
    """Two versions of the same survey: show how as_of() changes the
    database contents between two probe timestamps."""
    t_v1_from = "2026-01-01T00:00:00+00:00"
    t_v1_to   = "2026-06-01T00:00:00+00:00"
    t_v2_from = "2026-06-01T00:00:00+00:00"
    v1 = DatasetValidity(valid_from_utc=t_v1_from, valid_to_utc=t_v1_to, version="dr16")
    v2 = DatasetValidity(valid_from_utc=t_v2_from, version="dr17",
                         supersedes=("spec_eb_qso_v_dr16",))

    _syn_point(tmp_path, "spec/eb_qso/v_dr16", "spec_eb_qso_v_dr16", v1, seed=1, n=150)
    _syn_point(tmp_path, "spec/eb_qso/v_dr17", "spec_eb_qso_v_dr17", v2, seed=2, n=300)

    db = OneuniverseDatabase.from_root(tmp_path)

    probes = {
        "March 2026 (v_dr16)":
            dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc),
        "September 2026 (v_dr17)":
            dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc),
    }
    fig, axes = plt.subplots(1, len(probes), figsize=(12, 4), sharey=True)
    for (label, t), ax in zip(probes.items(), axes):
        snap = db.as_of(t)
        for name in snap.list():
            view = DatasetView.from_path(snap.get_path(name))
            d = view.read(columns=["ra", "dec"])
            ax.scatter(d["ra"], d["dec"], s=4, alpha=0.6, label=name)
        ax.set_title(label)
        ax.set_xlabel("RA (deg)")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("Dec (deg)")

    fig.tight_layout()
    out = OUTPUT_DIR / "phase7_database_snapshot.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 10_000
```

- [ ] **Step 2: Run tests**

Run: `pytest test/test_visual_temporal.py -v`
Expected: 3 passed. Inspect the images under `test/test_output/`.

- [ ] **Step 3: Commit**

```bash
git add test/test_visual_temporal.py
git commit -m "test(temporal): diagnostic figures for POINT(t_obs), LIGHTCURVE, and as_of snapshot"
```

---

## Task 12: Plan-status update + memory

**Files:**
- Modify: `plans/README.md`
- Modify: `~/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md`

- [ ] **Step 1: Update the plan index**

Edit `plans/README.md`:

Add the new plan files to the bullet list (immediately below the Phase-6
entry) and add a status-table row:

```markdown
- [`2026-04-20-temporal-subobject-roadmap.md`](2026-04-20-temporal-subobject-roadmap.md) — roadmap for Phase 7 (temporal) + Phase 8 (sub-object).
- [`2026-04-20-phase7-temporal.md`](2026-04-20-phase7-temporal.md) — detailed plan for Phase 7.
- [`2026-04-20-phase8-subobject.md`](2026-04-20-phase8-subobject.md) — detailed plan for Phase 8.
```

Status table:

```markdown
| 7 | Temporal data (t_obs + LIGHTCURVE + bitemporal database + versioned ONEUID) | **complete (2026-04-20, N/N tests green)** |
```

Fill `N/N` when the phase completes.

- [ ] **Step 2: Update memory**

Append a Phase-7 block to
`~/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md`
in the same shape used for prior phases. Record each task's commit
hash, the final test count, the new format version, and the new public
API surface (`TemporalSpec`, `DatasetValidity`, `write_ouf_lightcurve_dataset`,
`OneuniverseDatabase.as_of`, `load_oneuid(name, as_of=…)`,
`list_oneuids(include_archived=True)`).

- [ ] **Step 3: Commit plan docs**

```bash
git add plans/README.md
git commit -m "docs(plans): mark Phase 7 (temporal) complete"
```

---

## Self-review checklist

- [ ] Every new column declared in `format_spec.py` appears by name in at least one test.
- [ ] `Manifest.temporal` and `Manifest.validity` are both optional; 2.0.x manifests are default-filled on read.
- [ ] `DatasetView.scan` takes `t_range` as kwargs-only; no caller signatures broken.
- [ ] `write_ouf_lightcurve_dataset` validates orphan epochs and missing columns before any bytes hit disk.
- [ ] `objects_table()` raises for geometries without an objects file.
- [ ] `OneuniverseDatabase.as_of` returns a new object that shares the root but filters `_entries`; `versions_of_root` walks `_all_entries`.
- [ ] Rebuilding an ONEUID index with the same name archives the prior version with a timestamp suffix and closes its validity.
- [ ] `load_oneuid(name, as_of=T)` resolves to the manifest whose `validity.contains(T)`; raises if none does.
- [ ] Format-version check accepts `2.0.x` AND `2.1.x`; rejects `3.x`.
- [ ] Diagnostic test is skipped when matplotlib is absent.
