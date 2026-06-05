# `oneuniverse.measure` — Map × Catalog (FieldMap subtype) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Add **map-based cross-correlation** probes (galaxy × CMBκ, × tSZ y, × HI) to `oneuniverse.measure` by introducing the **`FieldMap`** DataProduct subtype (HEALPix/voxel field + mask) and `build_map_cross` (PointSet × FieldMap), emitting cosmology-free `MeasurementSet`s.

**Architecture:** Second non-PointSet subtype. Reuse the galaxy-clustering builder for the catalog side, `assign_regions`, `MeasurementSet`, `MeasurementSpec`. New: `FieldMap(values, mask, nside, nest)` from an external HEALPix array or P1's GW_SKYMAP/HEALPIX geometry, and a multi-product MeasurementSet pairing a PointSet with a FieldMap. **Cosmology-free** — C_ℓ theory + z→r are P2.

**Tech Stack:** numpy, healpy; reuses `oneuniverse.measure.build_galaxy_clustering`, `DatasetView`. No cosmology engine.

---

## Reused (already built)
`oneuniverse.measure`: `build_galaxy_clustering`, `assign_regions`, `MeasurementSet`, `MeasurementSpec`, `ProductMetadata`, `Provenance`, `DataProduct` base, fixture pattern.

## Confirmed P1 APIs
- `DataGeometry.HEALPIX` / `GW_SKYMAP` / `CUBE`; `CubeSpec` / `GwSkymapSpec` on the manifest. GW_SKYMAP rows carry `prob (list<f4>)` + `map_nside`. External κ/y maps are plain HEALPix `.fits`/`.npy` arrays.

## File structure (new / modified)
| File | Responsibility |
|---|---|
| Modify `measure/dataproduct.py` | add `FieldMap` subtype (`kind="fieldmap"`) |
| Create `measure/fieldmap.py` | `fieldmap_from_healpix(values, mask, nside)` + `fieldmap_from_view(view)` |
| Create `measure/mapcross.py` | `build_map_cross(catalog_view, fieldmap, ...)` |
| Modify `test/fixtures/measure_ouf.py` | `synthetic_healpix_map(nside, seed)` helper |
| Tests | one per task |

---

## Task 1: `FieldMap` subtype
**Files:** Modify `measure/dataproduct.py`; Test `test/test_measure_fieldmap_type.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_fieldmap_type.py
import numpy as np

from oneuniverse.measure.dataproduct import FieldMap
from oneuniverse.measure.metadata import ProductMetadata, Provenance


def test_fieldmap_holds_values_and_mask():
    import healpy as hp
    nside = 16
    vals = np.random.default_rng(0).standard_normal(hp.nside2npix(nside))
    mask = np.ones_like(vals, dtype=bool)
    fm = FieldMap(
        values=vals, mask=mask, nside=nside, nest=True,
        region_map=np.array([], dtype=np.int64),
        metadata=ProductMetadata(frame="galactic", epoch=2000.0,
                                 length_unit="dimensionless", nside_region=8),
        provenance=Provenance(dataset_ids=("cmb_kappa",)))
    assert fm.kind == "fieldmap"
    assert fm.npix == hp.nside2npix(16)
    assert fm.values.shape == fm.mask.shape
```

- [ ] **Step 2: Run — FAIL. Step 3: Add `FieldMap` to `measure/dataproduct.py`**

```python
@dataclass(kw_only=True)
class FieldMap(DataProduct):
    values: np.ndarray = None         # HEALPix vector (or flattened voxel grid)
    mask: np.ndarray = None
    nside: int = 0
    nest: bool = True
    axes: object = None               # WCS/axis metadata for cubes (optional)

    kind: ClassVar[str] = "fieldmap"

    @property
    def npix(self) -> int:
        return int(self.values.shape[0])
```

- [ ] **Step 4: Run — PASS. Step 5: Commit** `measure/map-T1: FieldMap DataProduct subtype`.

---

## Task 2: FieldMap ingest
**Files:** Create `measure/fieldmap.py`; extend fixture (`synthetic_healpix_map(nside, seed)` returns `(values, mask)`); Test `test/test_measure_fieldmap_ingest.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_fieldmap_ingest.py
import sys
from pathlib import Path

import numpy as np

from oneuniverse.measure.fieldmap import fieldmap_from_healpix

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_healpix_map  # noqa: E402


def test_fieldmap_from_healpix(tmp_path):
    vals, mask = synthetic_healpix_map(nside=32, seed=1)
    fm = fieldmap_from_healpix(vals, mask=mask, nside=32, frame="galactic")
    assert fm.kind == "fieldmap" and fm.nside == 32
    assert fm.metadata.frame == "galactic"
    assert fm.values.shape == fm.mask.shape
    # masked pixels are flagged, not deleted (alignment preserved)
    assert fm.mask.dtype == bool
```

- [ ] **Step 2:** Add `synthetic_healpix_map(nside, seed)` to the fixture (`hp.nside2npix` gaussian values + a half-sky boolean mask).

- [ ] **Step 3: Run — FAIL. Step 4: Implement `measure/fieldmap.py`**

```python
"""Ingest a HEALPix field (CMBκ / tSZ y / HI) into a measure FieldMap."""
from __future__ import annotations

from typing import Optional

import numpy as np

from oneuniverse.measure.dataproduct import FieldMap
from oneuniverse.measure.metadata import ProductMetadata, Provenance


def fieldmap_from_healpix(values, *, mask: Optional[np.ndarray] = None,
                          nside: int, nest: bool = True,
                          frame: str = "galactic",
                          dataset_id: str = "map") -> FieldMap:
    values = np.asarray(values, float)
    if mask is None:
        mask = np.ones(values.shape, dtype=bool)
    mask = np.asarray(mask, bool)
    meta = ProductMetadata(frame=frame, epoch=2000.0,
                           length_unit="dimensionless", nside_region=0)
    return FieldMap(values=values, mask=mask, nside=int(nside), nest=nest,
                    region_map=np.array([], dtype=np.int64), metadata=meta,
                    provenance=Provenance(dataset_ids=(dataset_id,)))
```

(`fieldmap_from_view(view)` for P1 GW_SKYMAP/HEALPIX geometry is a thin wrapper — read the `prob`/value column + `map_nside` from the manifest. Add it in this task if the GW_SKYMAP fixture is reused; otherwise the external-array path above is the MVP.)

- [ ] **Step 5: Run — PASS. Step 6: Commit** `measure/map-T2: fieldmap_from_healpix (external map ingest)`.

---

## Task 3: `build_map_cross`
**Files:** Create `measure/mapcross.py`; Test `test/test_measure_mapcross.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_mapcross.py
import sys
from pathlib import Path

import numpy as np

from oneuniverse.combine.weights import ColumnWeight
from oneuniverse.measure import MeasurementSet
from oneuniverse.measure.fieldmap import fieldmap_from_healpix
from oneuniverse.measure.mapcross import build_map_cross

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_healpix_map, synthetic_point_view  # noqa: E402


def test_build_map_cross(tmp_path):
    gview = synthetic_point_view(tmp_path, n=4000, seed=3, name="gal")
    vals, mask = synthetic_healpix_map(nside=64, seed=4)
    fm = fieldmap_from_healpix(vals, mask=mask, nside=64, dataset_id="cmbk")
    ms = build_map_cross(
        gview, fm, gal_tracer="gal", map_tracer="kappa",
        z_range=(0.1, 1.0), gal_weights_columns=("weight_comp",),
        nside_region=4)
    assert isinstance(ms, MeasurementSet)
    assert set(ms.products) == {"gal", "kappa"}
    assert ("gal", "kappa") in ms.spec.pairs
    assert ms.spec.statistic == "cl"
    assert ms.spec.estimator_family == "cross"
    ms.check_invariants()
```

- [ ] **Step 2: Run — FAIL. Step 3: Implement `measure/mapcross.py`**

```python
"""Map × catalog cross-correlation MeasurementSet. Cosmology-free (C_ℓ is P2)."""
from __future__ import annotations

from typing import Tuple

from oneuniverse.combine.weights import ColumnWeight
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.clustering import build_galaxy_clustering
from oneuniverse.measure.dataproduct import FieldMap
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.spec import MeasurementSpec


def build_map_cross(catalog_view: DatasetView, fieldmap: FieldMap, *,
                    gal_tracer: str = "gal", map_tracer: str = "kappa",
                    z_range: Tuple[float, float] = (0.0, 2.0),
                    gal_weights_columns: Tuple[str, ...] = ("weight_comp",),
                    nz_edges=None, nside_region: int = 8) -> MeasurementSet:
    import numpy as np
    if nz_edges is None:
        nz_edges = np.linspace(0.0, 2.0, 21)
    gal_ms = build_galaxy_clustering(
        catalog_view, tracer=gal_tracer, z_range=z_range,
        weights=[ColumnWeight(c) for c in gal_weights_columns],
        nz_edges=nz_edges, randoms="generate", n_randoms=0,
        nside_region=nside_region)
    gal_ps = gal_ms.products[gal_tracer]
    spec = MeasurementSpec(
        tracers=(gal_tracer, map_tracer),
        pairs=((gal_tracer, map_tracer),),
        statistic="cl", estimator_family="cross")
    return MeasurementSet(products={gal_tracer: gal_ps, map_tracer: fieldmap},
                          spec=spec, metadata=gal_ms.metadata)
```

(`build_galaxy_clustering` with `randoms="generate", n_randoms=0` yields an empty randoms frame; if that errors, pass `randoms="none"` — verify the `randoms` arg handling and adjust. The FieldMap's `region_map` is empty (sky map, not jackknifed per-pixel here); `check_invariants` only length-checks products with a non-None `catalog`, so the map is skipped — confirm.)

- [ ] **Step 4: Run — PASS. Step 5: Commit** `measure/map-T3: build_map_cross (PointSet × FieldMap)`.

---

## Task 4: Visual + close-out
- [ ] `scripts/build_measure_mapcross_demo.py`: the HEALPix map (mollview) with the galaxy footprint overlaid → `test/test_output/measure_map_cross.png`. Visual test.
- [ ] Full suite green; `CLAUDE.md` + `plans/README.md` + memory updates (FieldMap subtype + map×catalog).
- [ ] Commit `measure/map-T4: map×catalog demo + docs`.

## Success criteria
- `FieldMap` subtype carries a HEALPix field + mask + NSIDE; cosmology-free.
- `build_map_cross` emits a 2-product MeasurementSet (PointSet × FieldMap) with a `cl` cross spec. `check_invariants` tolerates the map product. Full suite green.
- **All three DataProduct subtypes (PointSet / Sightline / FieldMap) now exist** — the Universal DataProduct is complete across the probe space.

## Maps to requirements research
[`research/2026-06-05-p1-to-p2-measurement-requirements.md`](../research/2026-06-05-p1-to-p2-measurement-requirements.md) §2 geometry A(FieldMap), F(fields/maps); §3 CMBκ×g / tSZ×g / HI rows; §4 statistic = cl.

## Self-review
- [ ] `FieldMap(values, mask, nside, nest)` signature consistent across T1–T3.
- [ ] `build_map_cross` reuses `build_galaxy_clustering` (no duplicated catalog spine).
- [ ] `randoms` arg handling for the n_randoms=0 / "none" case verified before T3.
- [ ] `check_invariants` confirmed to skip the FieldMap product (catalog is None).
- [ ] No cosmology in `measure/mapcross.py`.
