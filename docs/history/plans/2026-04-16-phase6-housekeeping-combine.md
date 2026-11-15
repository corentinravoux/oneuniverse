# Phase 6 — Housekeeping + `weight/` → `combine/` redesign

> **For agentic workers:** Execute task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Close out Pillar 1 stabilisation. Two concerns — data-layer
cleanups (consolidate database state, trim dtype, delete legacy
shims) and a focused redesign of the weighting sub-package around a
principled default scheme.

**Architecture:**

- `OneuniverseDatabase` collapses its three parallel dicts into one
  `DatasetEntry` map.
- ONEUID index stores `dataset` as `pd.Categorical` and `row_index`
  as `int32` when the max row count allows.
- Module-level `_data_root` in `_config.py` moves onto the database
  instance.
- The survey registry freezes after import.
- `oneuniverse.weight` → `oneuniverse.combine`, restructured with a
  `weights/` subpackage (one file per weight family) and a new
  `registry.py` providing `default_weight_for(survey_type, ztype)`.
  `WeightedCatalog` mandates an `OneuidIndex`; raw-DataFrame
  construction becomes an error.
- Legacy shims drop: `load_universal`, legacy
  `_oneuid_index.parquet` layout, `WeightedCatalog.crossmatch()` self-
  builder, `oneuniverse.weight.crossmatch` module.

**Tech Stack:** unchanged from Phase 5.

**Context — what Phases 1–5 delivered:**

- OUF 2.0 format, HEALPix partitioning, `DatasetView`, named ONEUID
  indices, streaming hydration — all green on 198/198 tests.
- ONEUID owns cross-matching via
  `oneuniverse.data.oneuid_crossmatch`. `weight/crossmatch.py` is now
  a one-way shim ready for deletion.
- `WeightedCatalog.from_oneuid` is the recommended constructor since
  Phase 4 (emits `DeprecationWarning` on the self-builder).

---

## File Structure

### Data-layer housekeeping

- Modify: `oneuniverse/data/database.py`
  — replace `_loaders` / `_manifests` / `_paths` with a single
    `_entries: Dict[str, DatasetEntry]`; freeze the survey registry
    reference on construction.
- Create: `oneuniverse/data/_dataset_entry.py`
  — frozen dataclass `DatasetEntry(loader, manifest, path)`.
- Modify: `oneuniverse/data/_config.py`
  — remove module-level `_data_root`; `data_root` becomes a database
    kwarg / class attribute.
- Modify: `oneuniverse/data/oneuid.py`
  — ONEUID index tables use categorical `dataset` and `int32`
    `row_index`; delete `load_universal` and the legacy `_oneuid_index.parquet`
    read path; update `OneuidQuery` accordingly.
- Modify: `oneuniverse/data/_registry.py`
  — freeze via `MappingProxyType` after import.

### `weight/` → `combine/` redesign

- Delete: `oneuniverse/weight/` (the entire package).
- Create: `oneuniverse/combine/`
  - `__init__.py`
  - `weights/__init__.py`
  - `weights/base.py`        (Weight, ProductWeight)
  - `weights/ivar.py`        (InverseVarianceWeight)
  - `weights/fkp.py`         (FKPWeight)
  - `weights/quality.py`     (QualityMaskWeight, ColumnWeight, ConstantWeight)
  - `weights/registry.py`    (default_weight_for)
  - `strategies.py`          (combine_weights + strategies)
  - `catalog.py`             (WeightedCatalog — requires OneuidIndex)
  - `measurements.py`        (CombinedMeasurements)
- Modify: `oneuniverse/__init__.py`
  — re-exports point at `combine`, not `weight`.
- Modify: `test/test_weight.py` → `test/test_combine.py` (renamed +
  reshaped).
- Create: `test/test_combine_registry.py` — `default_weight_for`.
- Create: `test/test_combine_catalog_contract.py` — `WeightedCatalog`
  rejects raw DataFrames.

---

## Task 1: `DatasetEntry` consolidation

**Files:**
- Create: `oneuniverse/data/_dataset_entry.py`
- Modify: `oneuniverse/data/database.py`

- [ ] **Step 1: Write failing test** (`test/test_database_entry.py`)

```python
from oneuniverse.data._dataset_entry import DatasetEntry

def test_entry_frozen():
    e = DatasetEntry(loader=None, manifest=None, path=None)
    with pytest.raises(Exception):  # FrozenInstanceError or TypeError
        e.loader = object()

def test_entry_attrs(db_two):
    db = db_two
    any_name = next(iter(db))
    e = db.entry(any_name)
    assert e.path == db.get_path(any_name)
    assert e.manifest is db.get_manifest(any_name)
```

- [ ] **Step 2: Add the dataclass**

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DatasetEntry:
    loader: object
    manifest: object
    path: Path
```

- [ ] **Step 3: Refactor `OneuniverseDatabase`**

Replace `self._loaders`, `self._manifests`, `self._paths` with
`self._entries: Dict[str, DatasetEntry]`. Update `get_path`,
`get_manifest`, `get_loader`, `__iter__`, `__contains__`, `view` to
read through `_entries`. Add `entry(name) -> DatasetEntry`.

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest test/ -q
```

Expected: all 198 tests still green; `test_database_entry.py` adds 2
new cases.

- [ ] **Step 5: Commit**

---

## Task 2: Compact dtypes on ONEUID index

**Files:**
- Modify: `oneuniverse/data/oneuid.py`

- [ ] **Step 1: Write failing test**

```python
def test_dataset_column_categorical(db_two):
    idx = db_two.build_oneuid(
        rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3),
    )
    assert idx.table["dataset"].dtype.name == "category"

def test_row_index_int32_when_small(db_two):
    idx = db_two.build_oneuid(
        rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3),
    )
    assert idx.table["row_index"].dtype == np.int32
```

- [ ] **Step 2: In `build_oneuid_index`, after assembling the table**

```python
table["dataset"] = pd.Categorical(table["dataset"])
if table["row_index"].max() < np.iinfo(np.int32).max:
    table["row_index"] = table["row_index"].astype(np.int32)
```

- [ ] **Step 3: Fix any downstream code that depended on `object`
  dtype** (mostly `np.isin`, `.to_numpy()`, `.groupby` — expected
  no-ops, but run the suite to confirm).

- [ ] **Step 4: Run tests + commit**

---

## Task 3: Delete legacy shims

**Files:**
- Modify: `oneuniverse/data/oneuid.py`
  — delete `ONEUID_INDEX_FILENAME_LEGACY`, `_read_legacy_index`,
    legacy branch in `load_oneuid_index` and `list_oneuids`, and
    `load_universal`.
- Delete: `oneuniverse/weight/crossmatch.py` (module removed in
  later task when `weight/` folds into `combine/`).

- [ ] **Step 1: Write failing test — the deletion is visible**

```python
def test_load_universal_gone():
    from oneuniverse.data import oneuid as mod
    assert not hasattr(mod, "load_universal")

def test_legacy_oneuid_filename_gone():
    from oneuniverse.data import oneuid as mod
    assert not hasattr(mod, "ONEUID_INDEX_FILENAME_LEGACY")
```

- [ ] **Step 2: Remove the legacy branches**

In `load_oneuid_index`, drop the back-compat block and raise
`FileNotFoundError` directly when the new path is missing. Drop
`_read_legacy_index`. In `list_oneuids`, drop the fallback that
returned `["default"]` when only the legacy file existed.

- [ ] **Step 3: Delete `load_universal` + its references**

Drop the function. Delete the
`TestLoadUniversal` class in `test/test_oneuid.py` (its replacement
is `TestOneuidQuery` + `test_oneuid_streaming.py`).

- [ ] **Step 4: Run tests + commit**

```bash
python3 -m pytest test/ -q
```

Expected: test count decreases by the removed `TestLoadUniversal`
tests (4 tests gone), adds the two new `test_legacy_*` cases. Net
green.

---

## Task 4: Per-database `data_root`

**Files:**
- Modify: `oneuniverse/data/_config.py`
- Modify: `oneuniverse/data/database.py`
- Modify: `oneuniverse/data/converter.py` (if it reads module state)

- [ ] **Step 1: Remove `_data_root` mutable module-level state**

In `_config.py`, keep only:

```python
_ENV_VAR = "ONEUNIVERSE_DATA_ROOT"

def env_data_root() -> Optional[Path]:
    env = os.environ.get(_ENV_VAR)
    return Path(env) if env else None
```

Delete `_data_root`, `set_data_root`, and `get_data_root`.

- [ ] **Step 2: Database owns the root**

```python
class OneuniverseDatabase:
    def __init__(self, root: Path | str, *, data_root: Optional[Path] = None):
        self.root = Path(root)
        self.data_root = Path(data_root) if data_root else env_data_root()
```

Update `resolve_survey_path` to take an explicit `data_root` arg
(instead of reading the module). Any caller that still expects the
old helper is fixed up here.

- [ ] **Step 3: Run tests + commit**

---

## Task 5: Freeze survey registry

**Files:**
- Modify: `oneuniverse/data/_registry.py`

- [ ] **Step 1: Test**

```python
def test_registry_frozen_after_import():
    from oneuniverse.data._registry import REGISTRY
    with pytest.raises(TypeError):
        REGISTRY["new"] = object()
```

- [ ] **Step 2: Implement**

Internally use `_REGISTRY: Dict[str, Type[BaseSurveyLoader]]`, then
expose `REGISTRY = MappingProxyType(_REGISTRY)`. Keep `_REGISTRY`
mutable for tests (our test fixtures write into it), but the public
name is the frozen view.

- [ ] **Step 3: Run tests + commit**

---

## Task 6: Create `oneuniverse/combine` skeleton

**Files:**
- Create: `oneuniverse/combine/__init__.py`
- Create: `oneuniverse/combine/measurements.py`
- Create: `oneuniverse/combine/strategies.py`
- Create: `oneuniverse/combine/weights/__init__.py`
- Create: `oneuniverse/combine/weights/base.py`
- Create: `oneuniverse/combine/weights/ivar.py`
- Create: `oneuniverse/combine/weights/fkp.py`
- Create: `oneuniverse/combine/weights/quality.py`
- Create: `oneuniverse/combine/weights/registry.py`

- [ ] **Step 1: Port each class to its own file** — copy the
  implementation verbatim from the Phase-5 `weight/base.py` / etc.,
  splitting it so each file has one responsibility:

  - `weights/base.py`   → `Weight`, `ProductWeight`
  - `weights/ivar.py`   → `InverseVarianceWeight`
  - `weights/fkp.py`    → `FKPWeight`
  - `weights/quality.py`→ `QualityMaskWeight`, `ColumnWeight`,
                          `ConstantWeight`
  - `strategies.py`     → `combine_weights`, private helpers
  - `measurements.py`   → `CombinedMeasurements`

- [ ] **Step 2: `combine/weights/__init__.py`** re-exports the union

```python
from .base import Weight, ProductWeight
from .ivar import InverseVarianceWeight
from .fkp import FKPWeight
from .quality import ColumnWeight, ConstantWeight, QualityMaskWeight
from .registry import default_weight_for

__all__ = [
    "Weight", "ProductWeight",
    "InverseVarianceWeight", "FKPWeight",
    "ColumnWeight", "ConstantWeight", "QualityMaskWeight",
    "default_weight_for",
]
```

- [ ] **Step 3: `combine/__init__.py`** re-exports the public API

```python
from .catalog import WeightedCatalog
from .measurements import CombinedMeasurements
from .strategies import combine_weights
from .weights import (
    ColumnWeight, ConstantWeight, FKPWeight, InverseVarianceWeight,
    ProductWeight, QualityMaskWeight, Weight, default_weight_for,
)
```

- [ ] **Step 4: No tests yet — this task is purely structural.**

- [ ] **Step 5: Commit**

---

## Task 7: `default_weight_for(survey_type, ztype)`

**Files:**
- Modify: `oneuniverse/combine/weights/registry.py`
- Create: `test/test_combine_registry.py`

- [ ] **Step 1: Write tests**

```python
import pytest
import pandas as pd, numpy as np
from oneuniverse.combine import default_weight_for, InverseVarianceWeight

def test_spectroscopic_spec_returns_ivar_on_z_err():
    w = default_weight_for("spectroscopic", "spec")
    assert isinstance(w, InverseVarianceWeight)
    assert w.error_column == "z_err"

def test_peculiar_velocity_returns_ivar_on_velocity_error_with_floor():
    w = default_weight_for("peculiar_velocity", "pv")
    assert isinstance(w, InverseVarianceWeight)
    assert w.error_column == "velocity_error"
    assert w.floor > 0  # σ_* floor to absorb non-linear dispersion

def test_photometric_phot_returns_ivar_on_z_err():
    w = default_weight_for("photometric", "phot")
    assert isinstance(w, InverseVarianceWeight)
    assert w.error_column == "z_err"

def test_unknown_pair_raises_or_returns_constant_one():
    # Design choice: raise with a clear message, not silently return 1.
    with pytest.raises(KeyError):
        default_weight_for("something_weird", "none")
```

- [ ] **Step 2: Implement**

```python
from typing import Dict, Tuple

from .ivar import InverseVarianceWeight

# (survey_type, ztype) -> Weight factory
_DEFAULTS: Dict[Tuple[str, str], callable] = {
    ("spectroscopic", "spec"): lambda: InverseVarianceWeight("z_err"),
    ("photometric",   "phot"): lambda: InverseVarianceWeight("z_err"),
    ("peculiar_velocity", "pv"): lambda: InverseVarianceWeight(
        "velocity_error", floor=250.0,  # km/s — Howlett 2019
    ),
}

def default_weight_for(survey_type: str, ztype: str):
    try:
        return _DEFAULTS[(survey_type, ztype)]()
    except KeyError:
        raise KeyError(
            f"No default weight for (survey_type={survey_type!r}, "
            f"ztype={ztype!r}). Register one or pass an explicit Weight."
        ) from None
```

- [ ] **Step 3: Run + commit**

---

## Task 8: `WeightedCatalog` requires `OneuidIndex`

**Files:**
- Create: `oneuniverse/combine/catalog.py`
- Create: `test/test_combine_catalog_contract.py`

- [ ] **Step 1: Write tests**

```python
from oneuniverse.combine import WeightedCatalog

def test_raw_dataframes_rejected():
    with pytest.raises(TypeError, match="OneuidIndex"):
        WeightedCatalog({"A": pd.DataFrame({"ra": [0.0]})})

def test_from_oneuid_still_works(_db_with_oneuid):
    db, index, *_ = _db_with_oneuid
    wc = WeightedCatalog.from_oneuid(index, db)
    assert wc._match is not None


def test_fill_defaults_populates(_db_with_oneuid):
    db, index, ds_a, ds_b = _db_with_oneuid
    wc = WeightedCatalog.from_oneuid(index, db)
    wc.fill_defaults(db, ztype="spec")
    # Every dataset should have at least one weight registered.
    assert wc._weights[ds_a] and wc._weights[ds_b]
```

- [ ] **Step 2: Implement**

```python
class WeightedCatalog:
    def __init__(self, catalogs: Dict[str, pd.DataFrame]) -> None:
        raise TypeError(
            "WeightedCatalog can no longer be built from raw DataFrames. "
            "Build a ONEUID index with `database.build_oneuid()` and use "
            "`WeightedCatalog.from_oneuid(index, database)`."
        )

    @classmethod
    def from_oneuid(cls, index, database):
        wc = cls.__new__(cls)
        wc._init_internal(index, database)
        return wc

    def _init_internal(self, index, database):
        ...  # logic lifted from Phase-4 from_oneuid

    def fill_defaults(self, database, *, ztype: str) -> "WeightedCatalog":
        """Populate default weights for every registered dataset."""
        from .weights import default_weight_for
        for name in self.catalogs:
            entry = database.entry(name)
            survey_type = entry.manifest.survey_type
            self.add_weight(name, default_weight_for(survey_type, ztype))
        return self
```

`crossmatch()` is **deleted** (not deprecated) in this task.
Everything else from Phase-4 `from_oneuid` is kept.

- [ ] **Step 3: Run + commit**

---

## Task 9: Wire tests + root `__init__.py` + delete old `weight/`

**Files:**
- Modify: `oneuniverse/__init__.py`
- Delete: `oneuniverse/weight/` directory (all files)
- Rename: `test/test_weight.py` → `test/test_combine.py`
- Update imports in any other test file that still imports
  `oneuniverse.weight` (expected: `test/test_oneuid.py` has one
  `from oneuniverse.weight import combine_weights` — switch to
  `from oneuniverse.combine import combine_weights`).

- [ ] **Step 1: Update `oneuniverse/__init__.py` exports**

Drop any `from oneuniverse.weight import …` lines. Add:

```python
from oneuniverse.combine import (  # noqa: F401
    WeightedCatalog,
    combine_weights,
    default_weight_for,
)
```

- [ ] **Step 2: Rename/rewrite the weight test**

Port each test case to the new imports (`oneuniverse.combine` instead
of `oneuniverse.weight`). Drop the now-impossible
`test_crossmatch_populates_long_table` style (raw DataFrames rejected
at construction); equivalent coverage lives in
`test_combine_catalog_contract.py`. Keep all weight-primitive tests
— they're still valid.

- [ ] **Step 3: Delete `oneuniverse/weight/`**

```bash
rm -r oneuniverse/weight
```

- [ ] **Step 4: Run full suite + commit**

```bash
python3 -m pytest test/ -q
```

Expected: everything green, with the old `test_weight.py` replaced
by `test_combine.py` + `test_combine_registry.py` +
`test_combine_catalog_contract.py`.

---

## Task 10: Integration + documentation sync

- [ ] **Step 1: Full suite one more time**

```bash
python3 -m pytest test/ -q
```

Expected: 195+ tests green, no deprecation warnings from the new
public API (old `WeightedCatalog.crossmatch()` warning goes with the
code).

- [ ] **Step 2: Update `plans/README.md`** — mark Phase 6 complete.

- [ ] **Step 3: Update memory** — add the Phase 6 bullet to
  `project_oneuniverse_stabilisation.md`.

- [ ] **Step 4: Roadmap sync** — note that Pillar 1 stabilisation is
  done; Phases 1–6 all green.

---

## Out of scope for Phase 6

- Pillar 2 / 3 work.
- New survey loaders.
- Any change to the converter's on-disk format beyond dtype tweaks
  on the ONEUID index.
- Performance tuning beyond the dtype compaction task.
