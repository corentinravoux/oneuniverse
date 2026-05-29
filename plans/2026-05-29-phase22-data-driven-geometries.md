# Phase 22 — Data-Driven Geometry Expansion (CUBE + GW_SKYMAP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the two remaining **observational** geometries that
today's POINT / SIGHTLINE / HEALPIX / LIGHTCURVE set cannot
represent: `CUBE` (per-row N-D arrays for IFU / HI / 21cm intensity
maps) and `GW_SKYMAP` (per-event HEALPix probability map with
optional 3D distance extras). **No mock geometry** — `PARTICLE` is
owned by Pillar 3.

**Architecture:** Two new `DataGeometry` enum values, two new
required-column tuples (data + objects), two new manifest sub-specs
(`CubeSpec`, `GwSkymapSpec`). Writer reuses `_write_partitions` (the
non-HEALPix-partitioned path) — both geometries write one row per
object with a `list<f4>` payload column for the array data, via the
Phase 17 mini-language. OUF bumps 2.4.0 → 2.5.0; 2.0–2.4 still parse
via the existing back-compat clause.

**Tech Stack:** Python 3.9+, numpy, pyarrow, pandas. No new deps.

---

## File Structure

**New files:**
- `oneuniverse/data/cube_spec.py` — `CubeSpec` dataclass (axes, units,
  WCS hint).
- `oneuniverse/data/gwskymap_spec.py` — `GwSkymapSpec` dataclass
  (NSIDE, ordering, distance-extras flag).
- `test/test_cube_geometry.py` — enum + required-columns + writer/
  reader round-trip.
- `test/test_gwskymap_geometry.py` — same shape, for GW_SKYMAP.
- `test/test_cube_spec.py` — CubeSpec round-trip.
- `test/test_gwskymap_spec.py` — GwSkymapSpec round-trip.
- `test/test_visual_phase22.py` — diagnostic figure.

**Modified files:**
- `oneuniverse/data/format_spec.py` — add `CUBE` + `GW_SKYMAP` enum
  values, required columns, partition rows; bump
  `FORMAT_VERSION` / `SCHEMA_VERSION` 2.4.0 → 2.5.0.
- `oneuniverse/data/manifest.py` — bump version constants + accept
  `2.5.x`; wire `cube`, `gwskymap` Manifest fields.
- `test/test_lightcurve_geometry.py` — version assertion bump.
- `test/test_manifest_phase16.py` — version assertion bump.
- `oneuniverse/CLAUDE.md`, `plans/README.md`,
  `research/schema_generalisation_audit.md` — Phase 22 close-out.

---

## Pre-flight

- [ ] **Step 0: Baseline.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest -q 2>&1 | tail -3
```

Expected: `499 passed, 2 skipped` (Phase 21 baseline).

---

## Task 1: `CUBE` + `GW_SKYMAP` enum values + required columns

**Files:**
- Modify: `oneuniverse/data/format_spec.py`
- Create: `test/test_cube_geometry.py`
- Create: `test/test_gwskymap_geometry.py`

- [ ] **Step 1: Failing test (CUBE)**

```python
# test/test_cube_geometry.py
"""Phase 22 T1/T3 — CUBE geometry scaffold."""
import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.format_spec import (
    CUBE_DATA_REQUIRED_COLUMNS,
    DEFAULT_PARTITION_ROWS,
    DataGeometry,
    GEOMETRY_COLUMNS,
    validate_columns,
)


def test_cube_enum_value():
    assert DataGeometry.CUBE.value == "cube"


def test_cube_in_geometry_columns():
    assert DataGeometry.CUBE in GEOMETRY_COLUMNS
    assert "data" in GEOMETRY_COLUMNS[DataGeometry.CUBE]


def test_cube_required_columns_contents():
    cols = set(CUBE_DATA_REQUIRED_COLUMNS)
    assert {"cube_id", "ra", "dec", "shape", "cube"} <= cols


def test_cube_default_partition_rows_present():
    assert DataGeometry.CUBE in DEFAULT_PARTITION_ROWS


def test_validate_columns_accepts_cube_data():
    df = pd.DataFrame({
        "cube_id": np.array([0], dtype="i8"),
        "ra":      np.array([10.0], dtype="f8"),
        "dec":     np.array([0.0], dtype="f8"),
        "shape":   [np.array([3, 3, 4], dtype="i4")],
        "cube":    [np.zeros(36, dtype="f4")],
    })
    assert validate_columns(list(df.columns), DataGeometry.CUBE, "data") == []
```

- [ ] **Step 2: Failing test (GW_SKYMAP)**

```python
# test/test_gwskymap_geometry.py
"""Phase 22 T1/T3 — GW_SKYMAP geometry scaffold."""
import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.data.format_spec import (
    DEFAULT_PARTITION_ROWS,
    DataGeometry,
    GEOMETRY_COLUMNS,
    GW_SKYMAP_DATA_REQUIRED_COLUMNS,
    validate_columns,
)


def test_gwskymap_enum_value():
    assert DataGeometry.GW_SKYMAP.value == "gw_skymap"


def test_gwskymap_in_geometry_columns():
    assert DataGeometry.GW_SKYMAP in GEOMETRY_COLUMNS


def test_gwskymap_required_columns_contents():
    cols = set(GW_SKYMAP_DATA_REQUIRED_COLUMNS)
    assert {"event_id", "event_name", "map_nside", "map_nest", "prob"} <= cols


def test_gwskymap_default_partition_rows_present():
    assert DataGeometry.GW_SKYMAP in DEFAULT_PARTITION_ROWS


def test_validate_columns_accepts_gwskymap_data():
    nside = 8
    df = pd.DataFrame({
        "event_id":   np.array([0], dtype="i8"),
        "event_name": np.array(["GW230529"], dtype=object),
        "map_nside":  np.array([nside], dtype="i4"),
        "map_nest":   np.array([True], dtype="bool"),
        "prob":       [np.zeros(hp.nside2npix(nside), dtype="f4")],
    })
    assert validate_columns(list(df.columns), DataGeometry.GW_SKYMAP, "data") == []
```

- [ ] **Step 3: Add the enum values + tuples**

In `oneuniverse/data/format_spec.py`, extend `DataGeometry`:

```python
    CUBE = "cube"
    """One row per observational cube (IFU / HI / 21cm intensity).
    Tables: part_*.parquet only. Columns include a per-row variable-
    length `cube` payload + `shape` triple."""

    GW_SKYMAP = "gw_skymap"
    """One row per gravitational-wave event with a per-row HEALPix
    probability map. Tables: part_*.parquet only."""
```

Add the required-column tuples just after `LIGHTCURVE_DATA_REQUIRED_COLUMNS`:

```python
CUBE_DATA_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "cube_id",          # int64, unique per cube
    "ra",               # float64, cube reference RA (deg, ICRS)
    "dec",              # float64, cube reference Dec (deg, ICRS)
    "shape",            # int32[3] — (n_ra, n_dec, n_chan)
    "cube",             # list<float32> — flattened cube payload
)

GW_SKYMAP_DATA_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "event_id",         # int64, unique per event
    "event_name",       # string, e.g. "GW230529_181500"
    "map_nside",        # int32, fixed HEALPix NSIDE per row
    "map_nest",         # bool, ordering
    "prob",             # f4[12 * nside²] or list<f4>
)
```

Extend `GEOMETRY_COLUMNS`:

```python
    DataGeometry.CUBE: {
        "data": CUBE_DATA_REQUIRED_COLUMNS,
    },
    DataGeometry.GW_SKYMAP: {
        "data": GW_SKYMAP_DATA_REQUIRED_COLUMNS,
    },
```

Extend `DEFAULT_PARTITION_ROWS`:

```python
    DataGeometry.CUBE:      1_000,    # one row per cube; small partitions
    DataGeometry.GW_SKYMAP: 100,      # one row per event
```

- [ ] **Step 4: Run tests**

```bash
pytest test/test_cube_geometry.py test/test_gwskymap_geometry.py -q
```

Expected: all enum/registry tests green; writer-side tests come in
T3.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/format_spec.py \
        test/test_cube_geometry.py test/test_gwskymap_geometry.py
git commit -m "phase22/T1: DataGeometry.CUBE + GW_SKYMAP enum values + required columns (data-only; no mocks)"
```

---

## Task 2: `CubeSpec` + `GwSkymapSpec` sub-specs

**Files:**
- Create: `oneuniverse/data/cube_spec.py`, `oneuniverse/data/gwskymap_spec.py`
- Create: `test/test_cube_spec.py`, `test/test_gwskymap_spec.py`

- [ ] **Step 1: Failing tests**

```python
# test/test_cube_spec.py
"""Phase 22 T2 — CubeSpec sub-spec."""
import pytest

from oneuniverse.data.cube_spec import CubeSpec


def test_defaults():
    spec = CubeSpec(
        axes=("ra", "dec", "wavelength"),
        axis_units=("deg", "deg", "angstrom"),
        wavelength_convention="vacuum",
    )
    assert spec.axes == ("ra", "dec", "wavelength")
    assert spec.wavelength_convention == "vacuum"


def test_axes_axis_units_must_match_length():
    with pytest.raises(ValueError, match="length"):
        CubeSpec(
            axes=("ra", "dec", "wavelength"),
            axis_units=("deg", "deg"),
        )


def test_to_from_dict_roundtrip():
    spec = CubeSpec(
        axes=("ra", "dec", "frequency"),
        axis_units=("deg", "deg", "MHz"),
        wavelength_convention="vacuum",
    )
    d = spec.to_dict()
    assert CubeSpec.from_dict(d) == spec
```

```python
# test/test_gwskymap_spec.py
"""Phase 22 T2 — GwSkymapSpec sub-spec."""
import pytest

from oneuniverse.data.gwskymap_spec import GwSkymapSpec


def test_defaults():
    spec = GwSkymapSpec(map_nside=32)
    assert spec.map_nside == 32
    assert spec.map_nest is True
    assert spec.has_distance_extras is False


def test_rejects_non_power_of_two_nside():
    with pytest.raises(ValueError, match="power of two"):
        GwSkymapSpec(map_nside=30)


def test_to_from_dict_roundtrip():
    spec = GwSkymapSpec(
        map_nside=64, map_nest=False, has_distance_extras=True,
    )
    d = spec.to_dict()
    assert GwSkymapSpec.from_dict(d) == spec
```

- [ ] **Step 2: Implement sub-specs**

```python
# oneuniverse/data/cube_spec.py
"""Observational CUBE metadata for OUF 2.5.

Declares the axis layout, axis units, and (for SIGHTLINE-style λ
axes) the wavelength convention of an observed cube. Pure
observational — no cosmological assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class CubeSpec:
    """Per-cube axis metadata.

    Parameters
    ----------
    axes
        Ordered tuple of axis names, e.g. ``("ra", "dec", "wavelength")``
        or ``("ra", "dec", "frequency")``.
    axis_units
        Same-length tuple of axis units (``"deg"``, ``"angstrom"``,
        ``"MHz"``, …).
    wavelength_convention
        ``"vacuum"`` or ``"air"`` when a spectral axis is present;
        ``None`` for non-spectral cubes (frequency-only).
    """

    axes: Tuple[str, ...]
    axis_units: Tuple[str, ...]
    wavelength_convention: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.axes) != len(self.axis_units):
            raise ValueError(
                f"CubeSpec: axes (len {len(self.axes)}) and axis_units "
                f"(len {len(self.axis_units)}) must have equal length"
            )
        object.__setattr__(self, "axes", tuple(str(a) for a in self.axes))
        object.__setattr__(
            self, "axis_units", tuple(str(u) for u in self.axis_units),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axes": list(self.axes),
            "axis_units": list(self.axis_units),
            "wavelength_convention": self.wavelength_convention,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CubeSpec":
        return cls(
            axes=tuple(d["axes"]),
            axis_units=tuple(d["axis_units"]),
            wavelength_convention=d.get("wavelength_convention"),
            extra=dict(d.get("extra", {})),
        )
```

```python
# oneuniverse/data/gwskymap_spec.py
"""Per-event GW sky-localisation map metadata for OUF 2.5.

GW LIGO/Virgo BAYESTAR / LALInference outputs typically ship as
multi-order MOC HEALPix; consumers rasterise via
:func:`oneuniverse.data.moc.rasterise_moc_to_healpix` to a
fixed-NSIDE numpy array before writing. This spec records that NSIDE
+ ordering + whether per-pixel 3-D distance extras (DISTMU /
DISTSIGMA / DISTNORM) are also stored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@dataclass(frozen=True)
class GwSkymapSpec:
    map_nside: int
    map_nest: bool = True
    has_distance_extras: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_power_of_two(int(self.map_nside)):
            raise ValueError(
                f"GwSkymapSpec.map_nside must be a power of two, "
                f"got {self.map_nside!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "map_nside": int(self.map_nside),
            "map_nest": bool(self.map_nest),
            "has_distance_extras": bool(self.has_distance_extras),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GwSkymapSpec":
        return cls(
            map_nside=int(d["map_nside"]),
            map_nest=bool(d.get("map_nest", True)),
            has_distance_extras=bool(d.get("has_distance_extras", False)),
            extra=dict(d.get("extra", {})),
        )
```

- [ ] **Step 3: Run tests**

```bash
pytest test/test_cube_spec.py test/test_gwskymap_spec.py -q
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/data/cube_spec.py oneuniverse/data/gwskymap_spec.py \
        test/test_cube_spec.py test/test_gwskymap_spec.py
git commit -m "phase22/T2: CubeSpec + GwSkymapSpec sub-specs (observational axis metadata only)"
```

---

## Task 3: Manifest wiring + OUF 2.5.0 bump

**Files:**
- Modify: `oneuniverse/data/manifest.py`
- Modify: `oneuniverse/data/format_spec.py` (version constants)
- Modify: `test/test_lightcurve_geometry.py` (version assertion)
- Modify: `test/test_manifest_phase16.py` (version assertion)
- Create: `test/test_manifest_phase22.py`

- [ ] **Step 1: Failing test**

```python
# test/test_manifest_phase22.py
"""Phase 22 T3 — Manifest carries cube / gwskymap and bumps to OUF 2.5."""
import json

import pytest

from oneuniverse.data.cube_spec import CubeSpec
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.gwskymap_spec import GwSkymapSpec
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


def _minimal_manifest(**overrides) -> Manifest:
    defaults = dict(
        oneuniverse_format_version=FORMAT_VERSION,
        oneuniverse_schema_version=FORMAT_VERSION,
        geometry=DataGeometry.CUBE,
        survey_name="fixture", survey_type="ifu",
        created_utc="2026-05-29T00:00:00+00:00",
        original_files=[OriginalFileSpec(
            path="raw.fits", sha256="0123456789abcdef",
            n_rows=1, size_bytes=100, format="fits",
        )],
        partitions=[PartitionSpec(
            name="data/part_0000.parquet",
            n_rows=1, sha256="fedcba9876543210", size_bytes=50,
            stats=PartitionStats(),
        )],
        partitioning=None, schema=[], conversion_kwargs={},
        loader=LoaderSpec(name="fixture_loader", version="0.0"),
    )
    defaults.update(overrides)
    return Manifest(**defaults)


def test_version_constants_bumped():
    assert FORMAT_VERSION == "2.5.0"


def test_manifest_carries_cube_spec(tmp_path):
    spec = CubeSpec(
        axes=("ra", "dec", "wavelength"),
        axis_units=("deg", "deg", "angstrom"),
        wavelength_convention="vacuum",
    )
    m = _minimal_manifest(cube=spec)
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read.cube == spec


def test_manifest_carries_gwskymap_spec(tmp_path):
    spec = GwSkymapSpec(map_nside=32)
    m = _minimal_manifest(
        geometry=DataGeometry.GW_SKYMAP, gwskymap=spec,
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read.gwskymap == spec


def test_reads_2_4_manifest_with_compat_defaults(tmp_path):
    payload = {
        "oneuniverse_format_version": "2.4.0",
        "oneuniverse_schema_version": "2.4.0",
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
    assert read.cube is None
    assert read.gwskymap is None
```

- [ ] **Step 2: Bump constants + accept 2.5.x**

In `oneuniverse/data/manifest.py`:

```python
FORMAT_VERSION: str = "2.5.0"
SCHEMA_VERSION: str = "2.5.0"
```

Extend the version-compat clause to include `2.5`:

```python
        and (
            fmt.startswith("2.0") or fmt.startswith("2.1")
            or fmt.startswith("2.2") or fmt.startswith("2.3")
            or fmt.startswith("2.4") or fmt.startswith("2.5")
        )
```

and the error message:

```python
        raise ManifestValidationError(
            f"{path}: oneuniverse_format_version={fmt!r} is not compatible "
            f"with this library (expected 2.0.x / 2.1.x / 2.2.x / 2.3.x "
            f"/ 2.4.x / 2.5.x)."
        )
```

In `oneuniverse/data/format_spec.py`:

```python
FORMAT_VERSION: str = "2.5.0"
SCHEMA_VERSION: str = "2.5.0"
```

In `test/test_lightcurve_geometry.py`:

```python
def test_format_version_is_2_5_0():
    assert FORMAT_VERSION == "2.5.0"
    assert SCHEMA_VERSION == "2.5.0"
```

In `test/test_manifest_phase16.py`:

```python
def test_version_constants_bumped():
    assert FORMAT_VERSION == "2.5.0"
```

- [ ] **Step 3: Wire `cube` + `gwskymap` Manifest fields**

In `oneuniverse/data/manifest.py`, add imports:

```python
from oneuniverse.data.cube_spec import CubeSpec
from oneuniverse.data.gwskymap_spec import GwSkymapSpec
```

Extend `Manifest`:

```python
    tomographic_nz: Optional[TomographicNzSpec] = None
    classification_pdf: Optional[ClassificationPdfSpec] = None
    # Phase 22 additions.
    cube: Optional[CubeSpec] = None
    gwskymap: Optional[GwSkymapSpec] = None
```

Extend `_to_dict`:

```python
    d["cube"] = m.cube.to_dict() if m.cube is not None else None
    d["gwskymap"] = m.gwskymap.to_dict() if m.gwskymap is not None else None
```

Extend `_from_dict` (just before the `return Manifest(...)`):

```python
    cube_raw = raw.get("cube")
    cube = CubeSpec.from_dict(cube_raw) if cube_raw else None
    gwskymap_raw = raw.get("gwskymap")
    gwskymap = GwSkymapSpec.from_dict(gwskymap_raw) if gwskymap_raw else None
```

and extend the constructor:

```python
        cube=cube,
        gwskymap=gwskymap,
```

- [ ] **Step 4: Run tests**

```bash
pytest test/test_manifest_phase22.py test/test_manifest.py test/test_manifest_phase16.py test/test_manifest_phase18.py test/test_lightcurve_geometry.py -q
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/manifest.py oneuniverse/data/format_spec.py \
        test/test_manifest_phase22.py test/test_manifest_phase16.py \
        test/test_lightcurve_geometry.py
git commit -m "phase22/T3: Manifest gains cube + gwskymap sub-specs; bump to OUF 2.5.0"
```

---

## Task 4: Writer / reader round-trip

**Files:**
- Extend `test/test_cube_geometry.py` and `test/test_gwskymap_geometry.py`
  with end-to-end round-trip tests.

- [ ] **Step 1: Append round-trip tests**

To `test/test_cube_geometry.py`:

```python
from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.cube_spec import CubeSpec
from oneuniverse.data.manifest import LoaderSpec, read_manifest


def test_cube_writer_reader_roundtrip(tmp_path):
    n_cubes = 3
    shape = (3, 3, 4)
    npx = shape[0] * shape[1] * shape[2]
    df = pd.DataFrame({
        "cube_id": np.arange(n_cubes, dtype="i8"),
        "ra":  np.linspace(10.0, 12.0, n_cubes).astype("f8"),
        "dec": np.zeros(n_cubes, dtype="f8"),
        "shape": [np.array(shape, dtype="i4") for _ in range(n_cubes)],
        "cube":  [
            np.arange(npx, dtype="f4") + i * npx
            for i in range(n_cubes)
        ],
    })
    out = tmp_path / "ifu" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="ifu", survey_type="ifu",
        geometry=DataGeometry.CUBE,
        loader=LoaderSpec(name="ifu_fixture", version="0"),
        column_dtypes={"cube": "list<f4>", "shape": "i4[3]"},
    )
    m = read_manifest(out / "manifest.json")
    assert m.geometry is DataGeometry.CUBE
    # Re-read the parquet to confirm round-trip of the list payload.
    import pyarrow.parquet as pq
    parts = sorted((out).rglob("part_*.parquet"))
    assert parts
    table = pq.read_table(parts[0])
    out_cubes = table.column("cube").to_pylist()
    assert [len(c) for c in out_cubes] == [npx, npx, npx]
```

To `test/test_gwskymap_geometry.py`:

```python
from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.gwskymap_spec import GwSkymapSpec
from oneuniverse.data.manifest import LoaderSpec, read_manifest


def test_gwskymap_writer_reader_roundtrip(tmp_path):
    nside = 8
    npix = hp.nside2npix(nside)
    df = pd.DataFrame({
        "event_id":   np.array([0, 1], dtype="i8"),
        "event_name": np.array(["GW230529", "GW230601"], dtype=object),
        "map_nside":  np.array([nside, nside], dtype="i4"),
        "map_nest":   np.array([True, True], dtype="bool"),
        "prob":       [
            np.zeros(npix, dtype="f4"),
            np.linspace(0, 1, npix, dtype="f4") / npix,
        ],
    })
    out = tmp_path / "gw" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="gw", survey_type="gw",
        geometry=DataGeometry.GW_SKYMAP,
        loader=LoaderSpec(name="gw_fixture", version="0"),
        column_dtypes={"prob": "list<f4>"},
    )
    m = read_manifest(out / "manifest.json")
    assert m.geometry is DataGeometry.GW_SKYMAP
    import pyarrow.parquet as pq
    parts = sorted(out.rglob("part_*.parquet"))
    assert parts
    table = pq.read_table(parts[0])
    out_probs = table.column("prob").to_pylist()
    assert [len(p) for p in out_probs] == [npix, npix]
```

- [ ] **Step 2: Run**

```bash
pytest test/test_cube_geometry.py test/test_gwskymap_geometry.py -q
```

Expected: green.

- [ ] **Step 3: Commit**

```bash
git add test/test_cube_geometry.py test/test_gwskymap_geometry.py
git commit -m "phase22/T4: CUBE + GW_SKYMAP writer/reader round-trip via column_dtypes (list<f4> + i4[3])"
```

---

## Task 5: Visual diagnostic

**Files:**
- Create: `test/test_visual_phase22.py`

```python
# test/test_visual_phase22.py
"""Phase 22 visual diagnostic — synthetic IFU cube + GW skymap."""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_phase22_visual(tmp_path):
    # Synthetic IFU cube
    nra, ndec, nchan = 24, 24, 30
    rng = np.random.default_rng(0)
    grid_ra = np.arange(nra)[:, None, None]
    grid_dec = np.arange(ndec)[None, :, None]
    grid_ch = np.arange(nchan)[None, None, :]
    blob = np.exp(
        -0.5 * ((grid_ra - nra/2) ** 2 + (grid_dec - ndec/2) ** 2) / 30.0
    )
    line = np.exp(-0.5 * ((grid_ch - nchan/2) ** 2) / 4.0)
    cube = (blob * line + 0.1 * rng.normal(size=(nra, ndec, nchan))).astype("f4")

    # Synthetic GW skymap
    nside = 64
    npix = hp.nside2npix(nside)
    lon, lat = hp.pix2ang(nside, np.arange(npix), nest=True, lonlat=True)
    centre = hp.ang2vec(20.0, 0.0, lonlat=True)
    vecs = np.array(hp.pix2vec(nside, np.arange(npix), nest=True))
    cos_sep = vecs.T @ centre
    sep = np.arccos(np.clip(cos_sep, -1.0, 1.0))
    prob = np.exp(-0.5 * (sep / np.radians(5.0)) ** 2)
    prob /= prob.sum()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].imshow(cube.sum(axis=2).T, origin="lower", cmap="viridis")
    ax[0].set_xlabel("RA pixel")
    ax[0].set_ylabel("Dec pixel")
    ax[0].set_title("IFU cube — collapsed flux")

    ax[1].plot(cube.sum(axis=(0, 1)) / cube.sum(), lw=1.0)
    ax[1].set_xlabel("channel")
    ax[1].set_ylabel("relative flux")
    ax[1].set_title("Spectral axis (line at mid-channel)")

    sc = ax[2].scatter(lon, lat, c=prob, s=2, cmap="magma")
    ax[2].set_xlabel("lon [deg]")
    ax[2].set_ylabel("lat [deg]")
    ax[2].set_title("GW probability skymap")
    plt.colorbar(sc, ax=ax[2], label="P(pixel)")

    fig.tight_layout()
    out_png = OUT / "phase22_cube_and_gwskymap.png"
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

- [ ] **Step 2: Run**

```bash
pytest test/test_visual_phase22.py -v
```

- [ ] **Step 3: Commit**

```bash
git add test/test_visual_phase22.py \
        test/test_output/phase22_cube_and_gwskymap.png
git commit -m "phase22/T5: visual diagnostic — synthetic IFU cube + GW skymap"
```

---

## Task 6: Docs + close-out

- [ ] **Step 1: CLAUDE.md**

Under "OUF 2.x" section bump the title to "OUF 2.5". Under the
sub-spec list append:

```
- `CubeSpec` (Phase 22) — observed-cube axis metadata (axes,
  axis_units, optional wavelength_convention). Used by CUBE
  datasets (IFU MaNGA/SAMI/MUSE, HI WALLABY, 21cm CHIME/HERA).
- `GwSkymapSpec` (Phase 22) — per-event HEALPix NSIDE + ordering +
  has-distance-extras flag. Used by GW_SKYMAP datasets (LIGO/Virgo
  BAYESTAR/LALInference outputs after MOC rasterisation).
- New `DataGeometry` values: `CUBE` (observed N-D arrays) and
  `GW_SKYMAP` (per-event probability maps). `PARTICLE` and mock
  geometries are **owned by Pillar 3**, not Pillar 1.
```

- [ ] **Step 2: plans/README.md**

```
| 22 | Data-driven geometry expansion: `CUBE` (observed IFU/HI/21cm) + `GW_SKYMAP` (event probability maps) — **no mocks** (`PARTICLE` is Pillar 3) | **complete (2026-05-29, NNN/NNN tests green)** |
```

- [ ] **Step 3: research/schema_generalisation_audit.md**

Replace the existing Phase 22 bullet with:

```
- **Phase 22 — Data-driven geometry expansion.** Landed 2026-05-29.
  Adds `DataGeometry.CUBE` (observed IFU / HI / 21cm cubes) +
  `DataGeometry.GW_SKYMAP` (per-event probability maps), with
  `CubeSpec` + `GwSkymapSpec` sub-specs on Manifest. `PARTICLE`
  geometry and mock readers reassigned to Pillar 3. OUF 2.5.0. See
  [`../plans/2026-05-29-phase22-data-driven-geometries.md`](../plans/2026-05-29-phase22-data-driven-geometries.md).
```

- [ ] **Step 4: Full suite + memory**

```bash
pytest -q 2>&1 | tail -3
```

Replace `NNN/NNN` in plans/README.md.

Append to memory file
`/home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md`:

```markdown
## Phase 22 — Data-driven geometry expansion (complete 2026-05-29)

- New `DataGeometry` values: `CUBE` (observed cubes, IFU/HI/21cm) and
  `GW_SKYMAP` (per-event HEALPix probability maps).
- `CubeSpec` (axes / axis_units / wavelength_convention) and
  `GwSkymapSpec` (map_nside / map_nest / has_distance_extras)
  attach to Manifest.
- Writer reuses `_write_partitions` + Phase 17 mini-language
  (`list<f4>`, `f4[N]`, `i4[3]`) — no new code path.
- `PARTICLE` geometry and mock readers reassigned to Pillar 3.
- OUF bump 2.4.0 → 2.5.0; 2.0–2.4 still parse.
- Tests: NNN/NNN green.
- Per-phase plan:
  `plans/2026-05-29-phase22-data-driven-geometries.md`.
```

- [ ] **Step 5: Final commit**

```bash
git add oneuniverse/CLAUDE.md plans/README.md \
        research/schema_generalisation_audit.md \
        /home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md
git commit -m "phase22: close-out — CUBE + GW_SKYMAP geometries, NNN tests green; no mocks (Pillar 3 owns PARTICLE)"
```

---

## Self-review checklist

- [ ] No cosmology metadata added.
- [ ] No mock / `PARTICLE` geometry added.
- [ ] `CUBE` + `GW_SKYMAP` round-trip writer → manifest → reader.
- [ ] OUF 2.0–2.4 manifests still parse.
- [ ] Visual PNG ≥ 30 kB.

## Spec-coverage map

| Requirement | Task |
|---|---|
| `DataGeometry.CUBE` + `GW_SKYMAP` + required columns | T1 |
| `CubeSpec` + `GwSkymapSpec` sub-specs | T2 |
| Manifest integration + OUF 2.5 bump | T3 |
| Writer/reader round-trip | T4 |
| Visual | T5 |
| Docs + close-out | T6 |
