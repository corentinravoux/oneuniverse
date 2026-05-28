# Phase 16 — Observational Metadata Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add observational metadata (`CoordinateSpec`, `SpectrumSpec`, extensible `z_type` registry, `ColumnDef.frame/epoch/wavelength_convention/nullable`) to OUF without introducing any cosmological assumption, bumping OUF 2.1.0 → 2.2.0 with backward-compat for 2.1 manifests.

**Architecture:** Three new lightweight modules (`ztypes.py`, `coordinate_spec.py`, `spectrum_spec.py`) plus targeted extensions to `schema.ColumnDef` and `manifest.Manifest`. Every loader gains observational annotations where the source survey publishes them; nothing is invented. Cosmology stays out per [[feedback_no_cosmology_in_pillar1]].

**Tech Stack:** Python 3.9+, dataclasses, pyarrow, pandas, pytest. Same stack as existing `pdf.py` / `temporal.py` / `validity.py`. Pattern follows `PdfSpec` for sub-spec serialisation (`to_dict` / `from_dict`).

---

## File Structure

**New files:**
- `oneuniverse/data/ztypes.py` — `Z_TYPE_REGISTRY: Set[str]`, `register_z_type()`, `is_registered()`, `assert_valid()`.
- `oneuniverse/data/coordinate_spec.py` — `CoordinateSpec` dataclass with `from_dict` / `to_dict`.
- `oneuniverse/data/spectrum_spec.py` — `SpectrumSpec` dataclass with `from_dict` / `to_dict`.
- `test/test_ztype_registry.py` — registry semantics.
- `test/test_coordinate_spec.py` — `CoordinateSpec` round-trip.
- `test/test_spectrum_spec.py` — `SpectrumSpec` round-trip.
- `test/test_columndef_metadata.py` — `ColumnDef` new fields.
- `test/test_manifest_phase16.py` — `Manifest` integration + 2.1 ↔ 2.2 compat.
- `test/test_visual_phase16_metadata.py` — diagnostic figure rendering manifest metadata.

**Modified files:**
- `oneuniverse/data/schema.py:26-34` — `ColumnDef` gains four optional fields.
- `oneuniverse/data/schema.py:58` — `Z_TYPE_VALUES` aliases the registry.
- `oneuniverse/data/manifest.py:33-34` — bump `FORMAT_VERSION`, `SCHEMA_VERSION` to `2.2.0`.
- `oneuniverse/data/manifest.py:100-119` — `Manifest` gains `coordinate`, `spectrum`, `observed_z_types`.
- `oneuniverse/data/manifest.py:164-170` — `_to_dict` serialises new sub-specs.
- `oneuniverse/data/manifest.py:196-288` — `_from_dict` reads new sub-specs with 2.0/2.1/2.2 compat.
- `oneuniverse/data/converter.py` — writer populates `observed_z_types` and validates against registry.
- `oneuniverse/data/surveys/*.py` — loaders declare observational specs where known.
- `oneuniverse/CLAUDE.md` — bump Z_TYPE_VALUES text, mention OUF 2.2.
- `plans/README.md` — mark Phase 16 in progress.

---

## Pre-flight

- [ ] **Step 0a: Confirm baseline is green.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest -q 2>&1 | tail -5
```

Expected: `365 passed, 1 skipped` (the post-Phase-15 baseline). If not, stop — fix baseline before continuing.

- [ ] **Step 0b: Create a worktree if not already in one.**

Per `superpowers:using-git-worktrees`. Branch name: `phase16-observational-metadata`.

---

## Task 1: `Z_TYPE_REGISTRY` extensible module

**Files:**
- Create: `oneuniverse/data/ztypes.py`
- Create: `test/test_ztype_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_ztype_registry.py
"""Phase 16 T1 — z_type registry is extensible and validatable."""
import pytest

from oneuniverse.data.ztypes import (
    Z_TYPE_REGISTRY,
    assert_valid,
    is_registered,
    register_z_type,
)


def test_legacy_values_are_registered():
    for v in ("spec", "phot", "phot_pdf", "pv", "none"):
        assert is_registered(v)


def test_register_new_value_is_idempotent():
    register_z_type("cluster_z", description="z from cluster member consensus")
    register_z_type("cluster_z", description="z from cluster member consensus")
    assert is_registered("cluster_z")


def test_register_rejects_bad_names():
    with pytest.raises(ValueError, match="lowercase"):
        register_z_type("Spec")
    with pytest.raises(ValueError, match="lowercase"):
        register_z_type("z-type")


def test_assert_valid_passes_for_known_values():
    register_z_type("spec_lya")
    assert_valid(["spec", "spec_lya", "phot"])


def test_assert_valid_rejects_unknown():
    with pytest.raises(ValueError, match="unregistered"):
        assert_valid(["spec", "made_up"])


def test_registry_is_set_like():
    assert isinstance(Z_TYPE_REGISTRY, set)
    register_z_type("xcorr_z")
    assert "xcorr_z" in Z_TYPE_REGISTRY
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_ztype_registry.py -v
```

Expected: `ImportError: No module named 'oneuniverse.data.ztypes'`.

- [ ] **Step 3: Implement the module**

```python
# oneuniverse/data/ztypes.py
"""Extensible registry of `z_type` tag values.

The CORE `z_type` column carries a short string label describing what
kind of redshift each row holds (``spec``, ``phot``, ``phot_pdf``,
``pv``, ``none`` for the legacy set). Phase 16 promotes the set to a
runtime registry so surveys can declare new variants
(``spec_lya``, ``cluster_z``, ``xcorr_z``, …) without editing core
schema code.

This module is **observational metadata only**: a `z_type` value is a
label for what a column contains, not a cosmological choice. Frame
disambiguation (CMB vs heliocentric) lives in :class:`ColumnDef.frame`
and :class:`CoordinateSpec`.
"""
from __future__ import annotations

import re
from typing import Iterable, Set

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

Z_TYPE_REGISTRY: Set[str] = {
    "spec",
    "phot",
    "phot_pdf",
    "pv",
    "none",
}

_DESCRIPTIONS: dict = {
    "spec": "spectroscopic redshift",
    "phot": "photometric point estimate",
    "phot_pdf": "photometric redshift with PDF on disk",
    "pv": "peculiar-velocity-derived redshift",
    "none": "no redshift available",
}


def register_z_type(name: str, *, description: str = "") -> None:
    """Add ``name`` to :data:`Z_TYPE_REGISTRY`. Idempotent.

    Raises
    ------
    ValueError
        If ``name`` is not lowercase ASCII matching ``[a-z][a-z0-9_]*``.
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(
            f"z_type names must be lowercase ASCII matching "
            f"[a-z][a-z0-9_]*; got {name!r}"
        )
    Z_TYPE_REGISTRY.add(name)
    if description and name not in _DESCRIPTIONS:
        _DESCRIPTIONS[name] = description


def is_registered(name: str) -> bool:
    return name in Z_TYPE_REGISTRY


def assert_valid(values: Iterable[str]) -> None:
    """Raise :class:`ValueError` if any value is not registered."""
    unknown = sorted({v for v in values if v not in Z_TYPE_REGISTRY})
    if unknown:
        raise ValueError(
            f"unregistered z_type value(s): {unknown!r}; "
            f"call register_z_type() first, or use one of "
            f"{sorted(Z_TYPE_REGISTRY)!r}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_ztype_registry.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/ztypes.py test/test_ztype_registry.py
git commit -m "phase16/T1: extensible z_type registry"
```

---

## Task 2: `ColumnDef` gains observational metadata fields

**Files:**
- Modify: `oneuniverse/data/schema.py:26-34`
- Create: `test/test_columndef_metadata.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_columndef_metadata.py
"""Phase 16 T2 — ColumnDef carries observational metadata."""
import pytest

from oneuniverse.data.schema import ColumnDef


def test_columndef_defaults():
    c = ColumnDef("z", "f4", "", "redshift")
    assert c.frame is None
    assert c.epoch is None
    assert c.wavelength_convention is None
    assert c.nullable is False


def test_columndef_accepts_frame():
    c = ColumnDef(
        "z_helio", "f4", "", "heliocentric z",
        frame="heliocentric",
    )
    assert c.frame == "heliocentric"


def test_columndef_accepts_epoch():
    c = ColumnDef(
        "ra", "f8", "deg", "ICRS at GAIA DR3 epoch",
        epoch=2016.0,
    )
    assert c.epoch == 2016.0


def test_columndef_accepts_wavelength_convention():
    c = ColumnDef(
        "loglam", "f4", "", "log wavelength",
        wavelength_convention="vacuum",
    )
    assert c.wavelength_convention == "vacuum"


def test_columndef_accepts_nullable():
    c = ColumnDef("z_phot", "f4", "", "photo-z", nullable=True)
    assert c.nullable is True


def test_columndef_remains_frozen():
    c = ColumnDef("z", "f4", "", "redshift")
    with pytest.raises(Exception):  # FrozenInstanceError
        c.frame = "cmb"  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_columndef_metadata.py -v
```

Expected: AttributeError or TypeError on `frame` / `epoch` / `wavelength_convention` / `nullable` — fields don't exist yet.

- [ ] **Step 3: Extend `ColumnDef`**

Edit `oneuniverse/data/schema.py` lines 26–34 to:

```python
@dataclass(frozen=True)
class ColumnDef:
    """Definition of a single catalog column."""

    name: str
    dtype: str  # numpy dtype string: "f8", "f4", "i8", "i1", "U32", …
    unit: str  # astropy-compatible unit string, "" if dimensionless
    description: str
    required: bool = True  # within its group
    # Phase 16 observational-metadata annotations. All optional. No
    # cosmology assumption — only what the survey publishes.
    frame: Optional[str] = None
    # One of: "heliocentric", "cmb", "lsr", "galactocentric", "icrs",
    # "galactic", "ecliptic", "observer", "AB", "Vega", … Loader-defined.
    epoch: Optional[float] = None
    # Decimal-year epoch for position columns (e.g. 2016.0 for GAIA DR3).
    wavelength_convention: Optional[str] = None
    # "vacuum" | "air" for spectral columns; None for non-spectral.
    nullable: bool = False
    # True if the column is allowed to contain NaN / missing rows.
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_columndef_metadata.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Confirm pre-existing schema tests still pass**

```bash
pytest test/test_pdf_schema.py test/test_manifest.py -q
```

Expected: all green (no regressions from adding optional fields).

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/schema.py test/test_columndef_metadata.py
git commit -m "phase16/T2: ColumnDef gains frame/epoch/wavelength_convention/nullable"
```

---

## Task 3: Annotate existing CORE / spec / PV columns where unambiguous

**Files:**
- Modify: `oneuniverse/data/schema.py:39-55` (CORE), `:60-72` (SPECTROSCOPIC), `:82-88` (PV), `:153-163` (SNIA)
- Test: `test/test_columndef_metadata.py` (extend)

- [ ] **Step 1: Add an assertion test on the annotations**

Append to `test/test_columndef_metadata.py`:

```python
def test_core_z_columns_have_no_frame_by_default():
    from oneuniverse.data.schema import CORE_COLUMNS
    by_name = {c.name: c for c in CORE_COLUMNS}
    # CORE z has no fixed frame — the loader / manifest fixes it.
    assert by_name["z"].frame is None


def test_spec_zhelio_is_heliocentric():
    from oneuniverse.data.schema import SPECTROSCOPIC_COLUMNS
    by_name = {c.name: c for c in SPECTROSCOPIC_COLUMNS}
    assert by_name["z_helio"].frame == "heliocentric"
    assert by_name["z_cmb"].frame == "cmb"
    assert by_name["cz_cmb"].frame == "cmb"


def test_snia_zcmb_is_cmb():
    from oneuniverse.data.schema import SNIA_COLUMNS
    by_name = {c.name: c for c in SNIA_COLUMNS}
    assert by_name["z_cmb"].frame == "cmb"


def test_ra_dec_nullable_is_false():
    from oneuniverse.data.schema import CORE_COLUMNS
    by_name = {c.name: c for c in CORE_COLUMNS}
    assert by_name["ra"].nullable is False
    assert by_name["dec"].nullable is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_columndef_metadata.py::test_spec_zhelio_is_heliocentric -v
```

Expected: assertion failure — `frame` is `None`.

- [ ] **Step 3: Annotate the columns**

In `oneuniverse/data/schema.py`, update these specific entries:

- Line 63: `ColumnDef("z_helio", "f4", "", "Heliocentric redshift", required=False, frame="heliocentric")`
- Line 64: `ColumnDef("z_cmb", "f4", "", "CMB-frame redshift", required=False, frame="cmb")`
- Line 65: `ColumnDef("cz_cmb", "f4", "km/s", "CMB-frame recession velocity", required=False, frame="cmb")`
- Line 154: `ColumnDef("z_cmb", "f4", "", "CMB-frame redshift", frame="cmb")`

Leave CORE `z`, `ra`, `dec` unannotated — the manifest's `CoordinateSpec` (Task 4) is the authoritative source for ICRS / epoch.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_columndef_metadata.py -v
```

Expected: all 10 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/schema.py test/test_columndef_metadata.py
git commit -m "phase16/T3: annotate z_helio / z_cmb / cz_cmb / SNIa z_cmb frames"
```

---

## Task 4: `CoordinateSpec` dataclass + serialisation

**Files:**
- Create: `oneuniverse/data/coordinate_spec.py`
- Create: `test/test_coordinate_spec.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_coordinate_spec.py
"""Phase 16 T4 — CoordinateSpec sub-spec."""
import pytest

from oneuniverse.data.coordinate_spec import CoordinateSpec


def test_defaults():
    spec = CoordinateSpec()
    assert spec.frame == "icrs"
    assert spec.epoch is None
    assert spec.proper_motion_available is False
    assert spec.parallax_available is False


def test_rejects_unknown_frame():
    with pytest.raises(ValueError, match="frame"):
        CoordinateSpec(frame="middle_earth")


def test_to_dict_and_from_dict_roundtrip():
    spec = CoordinateSpec(
        frame="icrs",
        epoch=2016.0,
        proper_motion_available=True,
        parallax_available=True,
    )
    d = spec.to_dict()
    assert d == {
        "frame": "icrs",
        "epoch": 2016.0,
        "proper_motion_available": True,
        "parallax_available": True,
    }
    assert CoordinateSpec.from_dict(d) == spec


def test_from_dict_tolerates_missing_optional_fields():
    spec = CoordinateSpec.from_dict({"frame": "galactic"})
    assert spec.frame == "galactic"
    assert spec.epoch is None
    assert spec.proper_motion_available is False
    assert spec.parallax_available is False


def test_is_frozen():
    spec = CoordinateSpec(frame="icrs")
    with pytest.raises(Exception):
        spec.frame = "galactic"  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_coordinate_spec.py -v
```

Expected: `ImportError: No module named 'oneuniverse.data.coordinate_spec'`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/data/coordinate_spec.py
"""Observational coordinate metadata for OUF 2.2.

`CoordinateSpec` is **observational only**: it records what frame and
epoch the survey published, plus whether proper-motion / parallax
columns are present. It does **not** assume any cosmology. Frame
conversion (e.g. ICRS → galactic) and epoch propagation (PM-correction
to a later epoch) happen in downstream pillars at use-time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

_ALLOWED_FRAMES = frozenset({"icrs", "galactic", "ecliptic"})


@dataclass(frozen=True)
class CoordinateSpec:
    """Coordinate-frame metadata declared at dataset level.

    Parameters
    ----------
    frame
        One of ``"icrs"`` (default), ``"galactic"``, ``"ecliptic"``.
    epoch
        Decimal-year position epoch, e.g. ``2016.0`` for GAIA DR3.
        ``None`` means the survey did not declare an epoch.
    proper_motion_available
        True if the dataset carries PM columns.
    parallax_available
        True if the dataset carries a parallax column.
    """

    frame: str = "icrs"
    epoch: Optional[float] = None
    proper_motion_available: bool = False
    parallax_available: bool = False

    def __post_init__(self) -> None:
        if self.frame not in _ALLOWED_FRAMES:
            raise ValueError(
                f"unknown coordinate frame {self.frame!r}; "
                f"allowed: {sorted(_ALLOWED_FRAMES)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "epoch": (float(self.epoch) if self.epoch is not None else None),
            "proper_motion_available": bool(self.proper_motion_available),
            "parallax_available": bool(self.parallax_available),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CoordinateSpec":
        return cls(
            frame=d.get("frame", "icrs"),
            epoch=(float(d["epoch"]) if d.get("epoch") is not None else None),
            proper_motion_available=bool(d.get("proper_motion_available", False)),
            parallax_available=bool(d.get("parallax_available", False)),
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_coordinate_spec.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/coordinate_spec.py test/test_coordinate_spec.py
git commit -m "phase16/T4: CoordinateSpec sub-spec dataclass"
```

---

## Task 5: `SpectrumSpec` dataclass + serialisation

**Files:**
- Create: `oneuniverse/data/spectrum_spec.py`
- Create: `test/test_spectrum_spec.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_spectrum_spec.py
"""Phase 16 T5 — SpectrumSpec sub-spec (SIGHTLINE datasets only)."""
import pytest

from oneuniverse.data.spectrum_spec import SpectrumSpec


def test_defaults():
    spec = SpectrumSpec(wavelength_convention="vacuum")
    assert spec.wavelength_convention == "vacuum"
    assert spec.log_binned is True
    assert spec.rest_frame_corrected is False
    assert spec.wavelength_unit == "angstrom"


def test_rejects_unknown_convention():
    with pytest.raises(ValueError, match="wavelength_convention"):
        SpectrumSpec(wavelength_convention="ether")


def test_rejects_unknown_unit():
    with pytest.raises(ValueError, match="wavelength_unit"):
        SpectrumSpec(wavelength_convention="vacuum", wavelength_unit="parsec")


def test_to_dict_and_from_dict_roundtrip():
    spec = SpectrumSpec(
        wavelength_convention="air",
        log_binned=False,
        rest_frame_corrected=True,
        wavelength_unit="nanometer",
    )
    d = spec.to_dict()
    assert d == {
        "wavelength_convention": "air",
        "log_binned": False,
        "rest_frame_corrected": True,
        "wavelength_unit": "nanometer",
    }
    assert SpectrumSpec.from_dict(d) == spec


def test_from_dict_tolerates_missing_optional_fields():
    spec = SpectrumSpec.from_dict({"wavelength_convention": "vacuum"})
    assert spec.log_binned is True
    assert spec.rest_frame_corrected is False
    assert spec.wavelength_unit == "angstrom"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_spectrum_spec.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

```python
# oneuniverse/data/spectrum_spec.py
"""Observational spectrum metadata for OUF 2.2 SIGHTLINE datasets.

Pure observational. Tells consumers whether wavelengths are vacuum or
air, log- or linear-binned, in what unit, and whether already
rest-frame-corrected. No cosmological choice; just a column-axis label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

_ALLOWED_CONVENTIONS = frozenset({"vacuum", "air"})
_ALLOWED_UNITS = frozenset({"angstrom", "nanometer", "micron"})


@dataclass(frozen=True)
class SpectrumSpec:
    """Wavelength-axis metadata for a SIGHTLINE dataset.

    Parameters
    ----------
    wavelength_convention
        ``"vacuum"`` (BOSS+/DESI/Euclid) or ``"air"`` (legacy SDSS,
        some VIPERS).
    log_binned
        True if pixels are uniform in ``log10(lambda)``.
    rest_frame_corrected
        True if the wavelength axis has already been divided by
        ``(1+z)``.
    wavelength_unit
        One of ``"angstrom"`` (default), ``"nanometer"``, ``"micron"``.
    """

    wavelength_convention: str
    log_binned: bool = True
    rest_frame_corrected: bool = False
    wavelength_unit: str = "angstrom"

    def __post_init__(self) -> None:
        if self.wavelength_convention not in _ALLOWED_CONVENTIONS:
            raise ValueError(
                f"unknown wavelength_convention "
                f"{self.wavelength_convention!r}; "
                f"allowed: {sorted(_ALLOWED_CONVENTIONS)}"
            )
        if self.wavelength_unit not in _ALLOWED_UNITS:
            raise ValueError(
                f"unknown wavelength_unit {self.wavelength_unit!r}; "
                f"allowed: {sorted(_ALLOWED_UNITS)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wavelength_convention": self.wavelength_convention,
            "log_binned": bool(self.log_binned),
            "rest_frame_corrected": bool(self.rest_frame_corrected),
            "wavelength_unit": self.wavelength_unit,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpectrumSpec":
        return cls(
            wavelength_convention=d["wavelength_convention"],
            log_binned=bool(d.get("log_binned", True)),
            rest_frame_corrected=bool(d.get("rest_frame_corrected", False)),
            wavelength_unit=d.get("wavelength_unit", "angstrom"),
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_spectrum_spec.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/spectrum_spec.py test/test_spectrum_spec.py
git commit -m "phase16/T5: SpectrumSpec sub-spec dataclass"
```

---

## Task 6: Wire new sub-specs into `Manifest` + OUF 2.2.0 bump

**Files:**
- Modify: `oneuniverse/data/manifest.py:27-34` (imports + version constants)
- Modify: `oneuniverse/data/manifest.py:100-119` (Manifest dataclass)
- Modify: `oneuniverse/data/manifest.py:164-170` (`_to_dict`)
- Modify: `oneuniverse/data/manifest.py:196-288` (`_from_dict` + 2.1 compat)
- Create: `test/test_manifest_phase16.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_manifest_phase16.py
"""Phase 16 T6 — Manifest carries CoordinateSpec / SpectrumSpec /
observed_z_types and round-trips OUF 2.2 ↔ 2.1.
"""
import json

import pytest

from oneuniverse.data.coordinate_spec import CoordinateSpec
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
from oneuniverse.data.spectrum_spec import SpectrumSpec


def _minimal_manifest(**overrides) -> Manifest:
    defaults = dict(
        oneuniverse_format_version=FORMAT_VERSION,
        oneuniverse_schema_version="2.2.0",
        geometry=DataGeometry.POINT,
        survey_name="fixture",
        survey_type="spectroscopic",
        created_utc="2026-05-28T00:00:00+00:00",
        original_files=[
            OriginalFileSpec(
                path="raw.fits", sha256="0123456789abcdef",
                n_rows=10, size_bytes=4096, format="fits",
            ),
        ],
        partitions=[
            PartitionSpec(
                name="data/part_0000.parquet",
                n_rows=10, sha256="fedcba9876543210", size_bytes=2048,
                stats=PartitionStats(),
            ),
        ],
        partitioning=None,
        schema=[],
        conversion_kwargs={},
        loader=LoaderSpec(name="fixture_loader", version="0.0"),
    )
    defaults.update(overrides)
    return Manifest(**defaults)


def test_version_constants_bumped():
    assert FORMAT_VERSION == "2.2.0"


def test_manifest_carries_coordinate_spec(tmp_path):
    m = _minimal_manifest(
        coordinate=CoordinateSpec(frame="icrs", epoch=2016.0,
                                  proper_motion_available=True),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read.coordinate == m.coordinate


def test_manifest_carries_spectrum_spec(tmp_path):
    m = _minimal_manifest(
        geometry=DataGeometry.SIGHTLINE,
        spectrum=SpectrumSpec(
            wavelength_convention="vacuum",
            log_binned=True,
            rest_frame_corrected=False,
        ),
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read.spectrum == m.spectrum


def test_manifest_carries_observed_z_types(tmp_path):
    m = _minimal_manifest(observed_z_types=("spec", "phot"))
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read.observed_z_types == ("spec", "phot")


def test_reads_phase15_2_1_manifest_with_compat_defaults(tmp_path):
    """A 2.1.0 manifest written before Phase 16 must still parse."""
    payload = {
        "oneuniverse_format_version": "2.1.0",
        "oneuniverse_schema_version": "2.1.0",
        "geometry": "point",
        "survey_name": "legacy",
        "survey_type": "spectroscopic",
        "created_utc": "2026-04-15T00:00:00+00:00",
        "original_files": [{
            "path": "raw.fits", "sha256": "0123456789abcdef",
            "n_rows": 1, "size_bytes": 100, "format": "fits",
        }],
        "partitions": [{
            "name": "data/part_0000.parquet", "n_rows": 1,
            "sha256": "fedcba9876543210", "size_bytes": 50,
        }],
        "partitioning": None,
        "schema": [],
        "conversion_kwargs": {},
        "loader": {"name": "legacy_loader", "version": "0.0"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    m = read_manifest(path)
    # Phase 16 fields default cleanly when absent.
    assert m.coordinate is None
    assert m.spectrum is None
    assert m.observed_z_types == ()


def test_unknown_format_version_still_rejected(tmp_path):
    payload = {
        "oneuniverse_format_version": "3.0.0",
        "oneuniverse_schema_version": "3.0.0",
        "geometry": "point",
        "survey_name": "future",
        "survey_type": "spectroscopic",
        "created_utc": "2030-01-01T00:00:00+00:00",
        "original_files": [],
        "partitions": [],
        "partitioning": None,
        "schema": [],
        "conversion_kwargs": {},
        "loader": {"name": "future_loader", "version": "0.0"},
    }
    from oneuniverse.data.manifest import ManifestValidationError

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ManifestValidationError):
        read_manifest(path)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_manifest_phase16.py -v
```

Expected: `FORMAT_VERSION == "2.1.0"`, no `coordinate` / `spectrum` / `observed_z_types` fields.

- [ ] **Step 3: Bump versions and import the new sub-specs**

Edit `oneuniverse/data/manifest.py` lines 27–34 from:

```python
from oneuniverse.data._atomic import atomic_write_text
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.pdf import PdfSpec
from oneuniverse.data.temporal import TemporalSpec
from oneuniverse.data.validity import DatasetValidity

FORMAT_VERSION: str = "2.1.0"
SCHEMA_VERSION: str = "2.1.0"
```

to:

```python
from oneuniverse.data._atomic import atomic_write_text
from oneuniverse.data.coordinate_spec import CoordinateSpec
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.pdf import PdfSpec
from oneuniverse.data.spectrum_spec import SpectrumSpec
from oneuniverse.data.temporal import TemporalSpec
from oneuniverse.data.validity import DatasetValidity

FORMAT_VERSION: str = "2.2.0"
SCHEMA_VERSION: str = "2.2.0"
```

- [ ] **Step 4: Extend `Manifest` dataclass with three new fields**

Edit `oneuniverse/data/manifest.py` line 119 (right after `pdf_spec`) to append:

```python
    pdf_spec: Optional[PdfSpec] = None
    # Phase 16: observational metadata. All None by default for
    # forward-compat with 2.1.x manifests.
    coordinate: Optional[CoordinateSpec] = None
    spectrum: Optional[SpectrumSpec] = None
    observed_z_types: tuple = ()
```

(Also update the `Optional[...]` import line to include `Tuple` if needed — file already imports from `typing`.)

- [ ] **Step 5: Extend `_to_dict` to serialise the new fields**

Replace `_to_dict` (lines 164–170) with:

```python
def _to_dict(m: Manifest) -> Dict[str, Any]:
    d = asdict(m)
    d["geometry"] = m.geometry.value
    d["temporal"] = m.temporal.to_dict() if m.temporal is not None else None
    d["validity"] = m.validity.to_dict() if m.validity is not None else None
    d["pdf_spec"] = m.pdf_spec.to_dict() if m.pdf_spec is not None else None
    d["coordinate"] = (
        m.coordinate.to_dict() if m.coordinate is not None else None
    )
    d["spectrum"] = m.spectrum.to_dict() if m.spectrum is not None else None
    d["observed_z_types"] = list(m.observed_z_types)
    return d
```

- [ ] **Step 6: Extend `_from_dict` to read the new fields with compat**

In `oneuniverse/data/manifest.py`, change the version check at lines 200–208 to accept 2.2 as well:

```python
    fmt = raw["oneuniverse_format_version"]
    if not (
        isinstance(fmt, str)
        and (fmt.startswith("2.0") or fmt.startswith("2.1") or fmt.startswith("2.2"))
    ):
        raise ManifestValidationError(
            f"{path}: oneuniverse_format_version={fmt!r} is not compatible "
            f"with this library (expected 2.0.x / 2.1.x / 2.2.x)."
        )
```

Just before the final `return Manifest(...)` block (line 271), add:

```python
    coord_raw = raw.get("coordinate")
    coordinate = CoordinateSpec.from_dict(coord_raw) if coord_raw else None
    spec_raw = raw.get("spectrum")
    spectrum = SpectrumSpec.from_dict(spec_raw) if spec_raw else None
    observed_z_types = tuple(raw.get("observed_z_types", ()))
```

Then extend the return value:

```python
    return Manifest(
        oneuniverse_format_version=fmt,
        oneuniverse_schema_version=raw["oneuniverse_schema_version"],
        geometry=geometry,
        survey_name=raw["survey_name"],
        survey_type=raw["survey_type"],
        created_utc=raw["created_utc"],
        original_files=original_files,
        partitions=partitions,
        partitioning=partitioning,
        schema=schema,
        conversion_kwargs=raw["conversion_kwargs"],
        loader=loader,
        extra=raw.get("extra", {}),
        temporal=temporal,
        validity=validity,
        pdf_spec=pdf_spec,
        coordinate=coordinate,
        spectrum=spectrum,
        observed_z_types=observed_z_types,
    )
```

- [ ] **Step 7: Run test to verify it passes**

```bash
pytest test/test_manifest_phase16.py -v
```

Expected: 6 passed.

- [ ] **Step 8: Run full manifest test module to confirm no regression**

```bash
pytest test/test_manifest.py test/test_pdf_manifest.py test/test_temporal_spec.py test/test_validity_spec.py -q
```

Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add oneuniverse/data/manifest.py test/test_manifest_phase16.py
git commit -m "phase16/T6: Manifest gains coordinate/spectrum/observed_z_types; bump to OUF 2.2.0"
```

---

## Task 7: Writer validates `z_type` against registry + populates `observed_z_types`

**Files:**
- Modify: `oneuniverse/data/converter.py` (locate the writer chunk-loop)
- Create: `test/test_converter_phase16_ztype.py`

- [ ] **Step 1: Locate the converter chunk loop**

```bash
grep -n "_chunk_to_table\|observed_z_types\|z_type" oneuniverse/data/converter.py | head -20
```

Note the line numbers for: where `_chunk_to_table` is called, where partitions are written, where the final `Manifest` is constructed. (Skipping listing here because line numbers shift across patches; the engineer must locate them at run-time. The change in Step 3 is localized to those two points.)

- [ ] **Step 2: Write the failing test**

```python
# test/test_converter_phase16_ztype.py
"""Phase 16 T7 — writer validates z_type and records observed values."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec, read_manifest


def _df(z_types):
    n = len(z_types)
    return pd.DataFrame({
        "ra": np.linspace(0.0, 1.0, n).astype("f8"),
        "dec": np.linspace(0.0, 1.0, n).astype("f8"),
        "z": np.full(n, 0.5, dtype="f4"),
        "z_type": np.array(z_types, dtype=object),
        "z_err": np.full(n, 0.01, dtype="f4"),
    })


def test_writer_records_observed_z_types(tmp_path):
    df = _df(["spec", "spec", "phot", "none"])
    out = tmp_path / "fixture" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="fixture", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="fixture_loader", version="0"),
    )
    m = read_manifest(out / "manifest.json")
    assert set(m.observed_z_types) == {"spec", "phot", "none"}


def test_writer_rejects_unknown_z_type(tmp_path):
    df = _df(["spec", "made_up"])
    out = tmp_path / "fixture" / "oneuniverse"
    out.mkdir(parents=True)
    with pytest.raises(ValueError, match="unregistered"):
        write_ouf_dataset(
            df=df, out_dir=out,
            survey_name="fixture", survey_type="spectroscopic",
            geometry=DataGeometry.POINT,
            loader=LoaderSpec(name="fixture_loader", version="0"),
        )


def test_writer_accepts_newly_registered_z_type(tmp_path):
    from oneuniverse.data.ztypes import register_z_type
    register_z_type("spec_lya", description="Lyman-alpha z")
    df = _df(["spec_lya", "spec_lya"])
    out = tmp_path / "fixture" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="fixture", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="fixture_loader", version="0"),
    )
    m = read_manifest(out / "manifest.json")
    assert "spec_lya" in m.observed_z_types
```

- [ ] **Step 3: Implement converter changes**

Two edits to `oneuniverse/data/converter.py`:

1. Near the top of `write_ouf_dataset` (after the input DataFrame is sanitised but before chunking), insert:

```python
from oneuniverse.data.ztypes import assert_valid as _assert_z_types

if "z_type" in df.columns:
    # Phase 16: every z_type value must be registered. Fail loudly
    # rather than silently writing a manifest that breaks downstream.
    _assert_z_types(set(df["z_type"].dropna().unique().tolist()))
    _observed_z_types = tuple(sorted({str(v) for v in df["z_type"].dropna().unique()}))
else:
    _observed_z_types = ()
```

2. In the `Manifest(...)` constructor at the end of the function, add:

```python
        observed_z_types=_observed_z_types,
```

(Leave `coordinate` and `spectrum` at default `None`; loaders supply them via Task 8.)

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_converter_phase16_ztype.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run the full converter suite**

```bash
pytest test/test_converter.py test/test_converter_phase12.py test/test_converter_pdf.py test/test_pdf_converter.py -q 2>&1 | tail -10
```

(File names may differ slightly — pick all converter tests that exist.)

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/converter.py test/test_converter_phase16_ztype.py
git commit -m "phase16/T7: writer validates z_type against registry, records observed_z_types"
```

---

## Task 8: Loader-side `CoordinateSpec` / `SpectrumSpec` declarations

Loaders pass observational specs into `convert_survey` (or the
underlying `write_ouf_dataset`). Touch only the loaders that ingest
real data today (the test-only DESI DR1 fixture loader is left as-is).

**Files:**
- Modify: `oneuniverse/data/surveys/eboss/qso/loader.py`
- Modify: `oneuniverse/data/surveys/desi/qso/loader.py`
- Modify: `oneuniverse/data/surveys/sdss/mgs/loader.py`
- Modify: `oneuniverse/data/surveys/desi/bgs/loader.py`
- Modify: `oneuniverse/data/surveys/desi/pv/loader.py`
- Modify: `oneuniverse/data/surveys/sixdfgs/loader.py`
- Modify: `oneuniverse/data/surveys/cosmicflows/cf4/loader.py`
- Modify: `oneuniverse/data/surveys/pantheonplus/loader.py`
- Modify: `oneuniverse/data/surveys/des/dr2/loader.py`
- Modify: `oneuniverse/data/converter.py` (`convert_survey` accepts and forwards new specs)
- Create: `test/test_loader_specs_phase16.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_loader_specs_phase16.py
"""Phase 16 T8 — loaders declare CoordinateSpec / SpectrumSpec where
the source survey publishes the info.
"""
import pytest

from oneuniverse.data.surveys.eboss.qso.loader import EbossQsoLoader
from oneuniverse.data.surveys.sdss.mgs.loader import SdssMgsLoader
from oneuniverse.data.surveys.pantheonplus.loader import PantheonPlusLoader


def test_eboss_qso_declares_icrs_and_vacuum():
    spec = EbossQsoLoader.coordinate_spec()
    assert spec.frame == "icrs"
    sspec = EbossQsoLoader.spectrum_spec()
    assert sspec.wavelength_convention == "vacuum"


def test_sdss_mgs_declares_icrs_and_air():
    spec = SdssMgsLoader.coordinate_spec()
    assert spec.frame == "icrs"
    sspec = SdssMgsLoader.spectrum_spec()
    assert sspec.wavelength_convention == "air"


def test_pantheonplus_declares_icrs_no_spectrum():
    spec = PantheonPlusLoader.coordinate_spec()
    assert spec.frame == "icrs"
    assert PantheonPlusLoader.spectrum_spec() is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_loader_specs_phase16.py -v
```

Expected: `AttributeError: ... has no attribute 'coordinate_spec'`.

- [ ] **Step 3: Add the classmethod hooks to `BaseSurveyLoader`**

Locate `oneuniverse/data/_base_loader.py`. After the class declaration of `BaseSurveyLoader`, add:

```python
    # ── Phase 16: observational metadata hooks ────────────────────────
    # Subclasses override to declare what frame / epoch / wavelength
    # convention the underlying survey publishes. Default: None
    # (no claim, manifest carries no observational metadata).

    @classmethod
    def coordinate_spec(cls):
        """Return :class:`CoordinateSpec` or ``None``."""
        return None

    @classmethod
    def spectrum_spec(cls):
        """Return :class:`SpectrumSpec` or ``None``."""
        return None
```

- [ ] **Step 4: Implement loader overrides — eBOSS QSO**

Append to `oneuniverse/data/surveys/eboss/qso/loader.py`:

```python
from oneuniverse.data.coordinate_spec import CoordinateSpec
from oneuniverse.data.spectrum_spec import SpectrumSpec

# Inside EbossQsoLoader:
    @classmethod
    def coordinate_spec(cls):
        return CoordinateSpec(frame="icrs")

    @classmethod
    def spectrum_spec(cls):
        # BOSS/eBOSS spectra published in vacuum wavelengths, log-binned.
        return SpectrumSpec(
            wavelength_convention="vacuum",
            log_binned=True,
            rest_frame_corrected=False,
        )
```

- [ ] **Step 5: Implement remaining loader overrides**

For each loader file listed under "Files" above, add equivalent
overrides. Concrete mapping:

| Loader file | `coordinate_spec()` | `spectrum_spec()` |
|---|---|---|
| `desi/qso/loader.py` | `CoordinateSpec(frame="icrs")` | `SpectrumSpec(wavelength_convention="vacuum", log_binned=True)` |
| `sdss/mgs/loader.py` | `CoordinateSpec(frame="icrs")` | `SpectrumSpec(wavelength_convention="air", log_binned=True)` |
| `desi/bgs/loader.py` | `CoordinateSpec(frame="icrs")` | `SpectrumSpec(wavelength_convention="vacuum", log_binned=True)` |
| `desi/pv/loader.py` | `CoordinateSpec(frame="icrs")` | `None` (catalog only) |
| `sixdfgs/loader.py` | `CoordinateSpec(frame="icrs")` | `None` |
| `cosmicflows/cf4/loader.py` | `CoordinateSpec(frame="icrs")` | `None` |
| `pantheonplus/loader.py` | `CoordinateSpec(frame="icrs")` | `None` |
| `des/dr2/loader.py` | `CoordinateSpec(frame="icrs")` | `None` |

Each implementation block looks like:

```python
from oneuniverse.data.coordinate_spec import CoordinateSpec
# (omit the spectrum import where spectrum_spec returns None)

# Inside the Loader class:
    @classmethod
    def coordinate_spec(cls):
        return CoordinateSpec(frame="icrs")
```

- [ ] **Step 6: Forward specs through `convert_survey`**

In `oneuniverse/data/converter.py`, locate `convert_survey`. After
resolving the `loader_instance`, fetch the specs and pass them to the
inner writer:

```python
coord = loader_instance.coordinate_spec()
spec = loader_instance.spectrum_spec()
write_ouf_dataset(
    ...,
    coordinate=coord,
    spectrum=spec,
)
```

Then in `write_ouf_dataset` signature, add:

```python
def write_ouf_dataset(
    ...,
    coordinate: Optional["CoordinateSpec"] = None,
    spectrum: Optional["SpectrumSpec"] = None,
):
```

(Use string forward-refs to avoid a circular import; add the imports under `TYPE_CHECKING`.)

And in the `Manifest(...)` constructor at the end of `write_ouf_dataset`:

```python
        coordinate=coordinate,
        spectrum=spectrum,
```

- [ ] **Step 7: Run test to verify it passes**

```bash
pytest test/test_loader_specs_phase16.py -v
```

Expected: 3 passed.

- [ ] **Step 8: Run the full loader-side suite**

```bash
pytest test/test_loaders.py test/test_eboss.py test/test_desi_qso.py test/test_pantheonplus.py -q
```

(Tests may not all exist; run whichever do. The goal is no regression.)

Expected: green.

- [ ] **Step 9: Commit**

```bash
git add oneuniverse/data/_base_loader.py \
        oneuniverse/data/surveys/eboss/qso/loader.py \
        oneuniverse/data/surveys/desi/qso/loader.py \
        oneuniverse/data/surveys/sdss/mgs/loader.py \
        oneuniverse/data/surveys/desi/bgs/loader.py \
        oneuniverse/data/surveys/desi/pv/loader.py \
        oneuniverse/data/surveys/sixdfgs/loader.py \
        oneuniverse/data/surveys/cosmicflows/cf4/loader.py \
        oneuniverse/data/surveys/pantheonplus/loader.py \
        oneuniverse/data/surveys/des/dr2/loader.py \
        oneuniverse/data/converter.py \
        test/test_loader_specs_phase16.py
git commit -m "phase16/T8: loaders declare CoordinateSpec/SpectrumSpec; converter forwards them"
```

---

## Task 9: Visual diagnostic test (per visual-testing memory)

**Files:**
- Create: `test/test_visual_phase16_metadata.py`

- [ ] **Step 1: Write the test**

```python
# test/test_visual_phase16_metadata.py
"""Phase 16 T9 — diagnostic figure showing observational metadata in
a written manifest. Per [[feedback_visual_testing]]."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.data.converter import write_ouf_dataset  # noqa: E402
from oneuniverse.data.coordinate_spec import CoordinateSpec  # noqa: E402
from oneuniverse.data.format_spec import DataGeometry  # noqa: E402
from oneuniverse.data.manifest import LoaderSpec, read_manifest  # noqa: E402
from oneuniverse.data.spectrum_spec import SpectrumSpec  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_phase16_visual(tmp_path):
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "ra": rng.uniform(0, 360, n).astype("f8"),
        "dec": rng.uniform(-30, 30, n).astype("f8"),
        "z": rng.uniform(0.1, 1.0, n).astype("f4"),
        "z_type": rng.choice(["spec", "phot"], size=n).astype(object),
        "z_err": np.full(n, 0.01, dtype="f4"),
    })
    out = tmp_path / "phase16_viz" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="phase16_viz", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="phase16_viz", version="0"),
        coordinate=CoordinateSpec(frame="icrs", epoch=2016.0,
                                  proper_motion_available=True),
        spectrum=SpectrumSpec(wavelength_convention="vacuum",
                              log_binned=True),
    )
    m = read_manifest(out / "manifest.json")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sc = ax[0].scatter(df["ra"], df["dec"], c=df["z"], s=4, alpha=0.7)
    plt.colorbar(sc, ax=ax[0], label="z")
    ax[0].set_xlabel("RA [deg]")
    ax[0].set_ylabel("Dec [deg]")
    ax[0].set_title(
        f"frame={m.coordinate.frame}  epoch={m.coordinate.epoch}\n"
        f"PM available: {m.coordinate.proper_motion_available}"
    )

    labels = sorted(set(df["z_type"]))
    counts = [int((df["z_type"] == lbl).sum()) for lbl in labels]
    ax[1].bar(labels, counts)
    ax[1].set_ylabel("rows")
    ax[1].set_title(
        f"observed_z_types = {tuple(sorted(m.observed_z_types))}\n"
        f"spectrum: {m.spectrum.wavelength_convention} / "
        f"log_binned={m.spectrum.log_binned}"
    )

    fig.tight_layout()
    out_png = OUT / "phase16_observational_metadata.png"
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
pytest test/test_visual_phase16_metadata.py -v
```

Expected: pass; `test/test_output/phase16_observational_metadata.png` created.

- [ ] **Step 3: Open the figure and sanity-check**

```bash
xdg-open test/test_output/phase16_observational_metadata.png 2>/dev/null || \
  echo "Inspect test/test_output/phase16_observational_metadata.png manually."
```

Expected: a 2-panel figure; left = RA/Dec scatter with frame+epoch in the title; right = z_type bar chart with observed_z_types in the title.

- [ ] **Step 4: Commit**

```bash
git add test/test_visual_phase16_metadata.py test/test_output/phase16_observational_metadata.png
git commit -m "phase16/T9: visual diagnostic for observational metadata"
```

---

## Task 10: Documentation + plan-README close-out

**Files:**
- Modify: `oneuniverse/CLAUDE.md`
- Modify: `plans/README.md`
- Modify: `research/schema_generalisation_audit.md`

- [ ] **Step 1: Update `oneuniverse/CLAUDE.md`**

Locate the line:

```
`Z_TYPE_VALUES = {"spec", "phot", "phot_pdf", "pv", "none"}`.
```

Replace with:

```
`Z_TYPE_REGISTRY = {"spec", "phot", "phot_pdf", "pv", "none", …}` —
extensible at runtime via
`oneuniverse.data.ztypes.register_z_type(name)`. Manifest stamps
`observed_z_types` automatically (Phase 16).
```

In the "OUF 2.1" heading, change `2.1` → `2.2`. Under the sub-spec
list under `manifest.json`, add:

```
`CoordinateSpec` (frame / epoch / PM-parallax availability),
`SpectrumSpec` (vacuum-or-air, log-binning, rest-frame state,
λ-unit; SIGHTLINE only).
```

- [ ] **Step 2: Update `plans/README.md`**

In the Phase status table, change the "16" row from `planned` to:

```
| 16 | Observational metadata expansion | **complete (YYYY-MM-DD, NNN/NNN tests green; OUF → 2.2.0)** |
```

(Fill in the date and test count after Task 11.)

- [ ] **Step 3: Update `research/schema_generalisation_audit.md`**

Find the "Phase 16 — Observational metadata expansion" line in the
"Suggested staging into phases" section. Replace with:

```
- **Phase 16 — Observational metadata expansion.** Landed YYYY-MM-DD.
  Adds `CoordinateSpec`, `SpectrumSpec`, extensible `z_type` registry,
  `ColumnDef` gains `frame`/`epoch`/`wavelength_convention`/`nullable`.
  No cosmology. OUF 2.2.0. See
  [`../plans/2026-05-28-phase16-observational-metadata.md`](../plans/2026-05-28-phase16-observational-metadata.md).
```

- [ ] **Step 4: Regenerate Sphinx autosummary (if docs extra installed)**

```bash
pip install -e ".[docs]" 2>&1 | tail -2
make -C docs clean && make -C docs html 2>&1 | tail -5
```

Expected: `build succeeded` (with warnings tolerated as long as none
mention the new modules).

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/CLAUDE.md plans/README.md \
        research/schema_generalisation_audit.md
git commit -m "docs(phase16): close-out — OUF 2.2.0, observational metadata, plans/README + audit"
```

---

## Task 11: Phase close-out — full suite green, memory update

- [ ] **Step 1: Run the full suite**

```bash
pytest -q 2>&1 | tail -5
```

Expected: `>= 376 passed` (365 baseline + 11 new from Phase 16 modules).
Record the exact count for the close-out commit.

- [ ] **Step 2: Update memory file**

Edit `/home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md`. Append a section:

```markdown
## Phase 16 — Observational metadata expansion (complete YYYY-MM-DD)

- `CoordinateSpec` (frame / epoch / PM-parallax) and `SpectrumSpec`
  (vacuum/air, log-binning, rest-frame, λ-unit) on Manifest.
- `ColumnDef` gains `frame`, `epoch`, `wavelength_convention`,
  `nullable`.
- `Z_TYPE_REGISTRY` extensible; Manifest stamps `observed_z_types`.
- OUF format bumped 2.1.0 → 2.2.0; 2.1 manifests still parse.
- No cosmology metadata anywhere — Pillar 1 is data-only as per
  [[no-cosmology-in-pillar1]].
- Tests: NNN/NNN green, suite ~Mmin.
```

- [ ] **Step 3: Update plans/README.md Phase 16 row with final test count**

Edit the row added in Task 10, Step 2 to fill in the actual numbers.

- [ ] **Step 4: Final commit**

```bash
git add plans/README.md \
        /home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md
git commit -m "phase16: close-out — OUF 2.2.0, NNN tests green"
```

- [ ] **Step 5: Hand off**

Per `superpowers:finishing-a-development-branch`, choose between PR,
merge to main, or further work. Default for this repo: merge directly
into the working branch after green suite.

---

## Self-review checklist (pre-merge sanity)

- [ ] No cosmology metadata fields anywhere (no `H0`, `Omega_m`,
      `little_h`, `distance_kind`, `fiducial`, etc).
- [ ] All new sub-specs follow the `to_dict` / `from_dict` pattern
      from `PdfSpec`.
- [ ] 2.1.x manifests load without raising and get defaulted-None
      for the new fields.
- [ ] `Z_TYPE_REGISTRY` is `set`, not frozen — runtime
      `register_z_type` works.
- [ ] Loader-side `coordinate_spec()` / `spectrum_spec()` return
      `None` when the survey does not declare the corresponding
      metadata (CF4, Pantheon+, 6dFGS, DESI PV → spectrum None).
- [ ] Visual figure `phase16_observational_metadata.png` exists,
      ≥ 30 kB, ≥ 800 × 200 px.
- [ ] Total tests pass; suite wall-clock is still ≤ 4 min.

---

## Spec-coverage map (confirms each requirement maps to a task)

| Requirement (from `research/schema_generalisation_audit.md`) | Task |
|---|---|
| `Z_TYPE_REGISTRY` extensible + observed-set on Manifest | T1, T6, T7 |
| `ColumnDef.frame / epoch / wavelength_convention / nullable` | T2, T3 |
| `CoordinateSpec` on Manifest | T4, T6, T8 |
| `SpectrumSpec` on Manifest (SIGHTLINE only) | T5, T6, T8 |
| Per-column z_helio / z_cmb / cz_cmb frame annotation | T3 |
| OUF 2.1 → 2.2 bump with backward-compat reader | T6 |
| Loader-side declaration of observational specs | T8 |
| No cosmology anywhere | self-review |
| Visual diagnostic | T9 |
| Docs + CLAUDE.md + plans/README + audit doc | T10 |
| Memory updated | T11 |
