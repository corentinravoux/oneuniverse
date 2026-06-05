# `oneuniverse.measure` — Lyα Forest (Sightline subtype) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Add the **Lyα forest** probe to `oneuniverse.measure` by introducing the **`Sightline`** DataProduct subtype (per-LOS δ_F(λ) + mask + continuum) and `build_lya` (P_1D / P_3D), emitting cosmology-free `MeasurementSet`s.

**Architecture:** First use of a non-PointSet subtype. Reuse `assign_regions`, `MeasurementSet`, `MeasurementSpec`, `ProductMetadata`, `Provenance`. New: `Sightline(los, delta, mask, continuum, ...)` read from P1's SIGHTLINE geometry (per-LOS metadata via `objects_table`, per-pixel arrays via `read`). **Cosmology-free** — r_∥/r_⊥ and the power-spectrum estimator are P2.

**Tech Stack:** numpy, pandas, healpy; reuses `DatasetView.objects_table()` + `read()` for SIGHTLINE. No cosmology engine.

---

## Reused (already built)
`oneuniverse.measure`: `assign_regions`, `MeasurementSet`, `MeasurementSpec`, `ProductMetadata`, `Provenance`, `DataProduct` base, fixture pattern.

## Confirmed P1 APIs
- `DataGeometry.SIGHTLINE`; `DatasetView.objects_table()` returns one row per sightline (los metadata), valid only for SIGHTLINE/LIGHTCURVE; `read()` returns the per-pixel rows / var-len arrays. `SpectrumSpec` (vacuum/air, log-binning, rest-frame) on the manifest.
- Lyα native picca: per-LOS `LOGLAM, DELTA, WEIGHT, CONT` partitioned HEALPix NSIDE=16 NEST.

## File structure (new / modified)
| File | Responsibility |
|---|---|
| Modify `measure/dataproduct.py` | add `Sightline` subtype (`kind="sightline"`) |
| Create `measure/sightline.py` | `sightline_from_view(view)` (objects + per-pixel arrays) |
| Create `measure/lya.py` | `build_lya(view, *, statistic)` |
| Modify `test/fixtures/measure_ouf.py` | `synthetic_sightline_view(...)` (SIGHTLINE geometry) |
| Tests | one per task |

---

## Task 1: `Sightline` subtype
**Files:** Modify `measure/dataproduct.py`; Test `test/test_measure_sightline_type.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_sightline_type.py
import numpy as np

from oneuniverse.measure.dataproduct import Sightline
from oneuniverse.measure.metadata import ProductMetadata, Provenance


def test_sightline_holds_los_and_pixels():
    import pandas as pd
    los = pd.DataFrame({"los_id": [0, 1], "ra": [10.0, 11.0],
                        "dec": [0.0, 1.0], "z_qso": [2.3, 2.5]})
    sl = Sightline(
        los=los, delta=[np.zeros(3), np.zeros(4)],
        mask=[np.ones(3), np.ones(4)], continuum=[np.ones(3), np.ones(4)],
        region_map=np.array([0, 1], dtype=np.int64),
        metadata=ProductMetadata(frame="icrs", epoch=2000.0,
                                 length_unit="deg", nside_region=8),
        provenance=Provenance(dataset_ids=("lya",)))
    assert sl.kind == "sightline"
    assert sl.n_sightlines == 2
    assert len(sl.delta[1]) == 4
```

- [ ] **Step 2: Run — FAIL. Step 3: Add `Sightline` to `measure/dataproduct.py`**

```python
@dataclass(kw_only=True)
class Sightline(DataProduct):
    los: pd.DataFrame = None          # los_id, ra, dec, z_qso, region_id
    delta: object = None              # list/array of per-LOS δ_F(λ)
    mask: object = None
    continuum: object = None
    resolution: object = None

    kind: ClassVar[str] = "sightline"

    @property
    def n_sightlines(self) -> int:
        return len(self.los)
```

- [ ] **Step 4: Run — PASS. Step 5: Commit** `measure/lya-T1: Sightline DataProduct subtype`.

---

## Task 2: `sightline_from_view`
**Files:** Create `measure/sightline.py`; extend fixture (`synthetic_sightline_view`: SIGHTLINE geometry, `objects_df` = one row per LOS with `los_id, ra, dec, z_qso`; per-pixel `df` with `los_id, loglam, delta, weight, cont`); Test `test/test_measure_sightline_read.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_sightline_read.py
import sys
from pathlib import Path

from oneuniverse.measure.sightline import sightline_from_view

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_sightline_view  # noqa: E402


def test_sightline_from_view(tmp_path):
    view = synthetic_sightline_view(tmp_path, n_los=12, n_pix=20, seed=1)
    sl = sightline_from_view(view)
    assert sl.kind == "sightline" and sl.n_sightlines == 12
    assert {"los_id", "ra", "dec"} <= set(sl.los.columns)
    assert len(sl.delta) == 12                 # one delta array per LOS
```

- [ ] **Step 2:** Add `synthetic_sightline_view` to the fixture (model on `test/test_format.py` SIGHTLINE usage: `write_ouf_dataset(..., geometry=DataGeometry.SIGHTLINE, objects_df=los_df, ...)` with a per-pixel `df` keyed by `los_id`).

- [ ] **Step 3: Run — FAIL. Step 4: Implement `measure/sightline.py`**

```python
"""Read a SIGHTLINE OUF dataset into a measure Sightline product."""
from __future__ import annotations

import numpy as np

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.measure.dataproduct import Sightline
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.regions import assign_regions


def sightline_from_view(view: DatasetView, *, nside_region: int = 16,
                        id_column: str = "los_id") -> Sightline:
    if view.geometry is not DataGeometry.SIGHTLINE:
        raise ValueError(
            f"sightline_from_view: expected SIGHTLINE, got "
            f"{view.geometry.value!r}")
    los = view.objects_table().to_pandas()
    pix = view.read()                       # per-pixel rows keyed by los_id
    ids = los[id_column].to_numpy()
    delta = [pix.loc[pix[id_column] == i, "delta"].to_numpy() for i in ids]
    mask = [pix.loc[pix[id_column] == i, "weight"].to_numpy() for i in ids]
    cont = [pix.loc[pix[id_column] == i, "cont"].to_numpy() for i in ids]
    region = assign_regions(los["ra"].to_numpy(), los["dec"].to_numpy(),
                            nside=nside_region)
    los = los.copy(); los["region_id"] = region
    meta = ProductMetadata(frame="icrs", epoch=2000.0, length_unit="deg",
                           nside_region=int(nside_region))
    return Sightline(los=los, delta=delta, mask=mask, continuum=cont,
                     region_map=region, metadata=meta,
                     provenance=Provenance(dataset_ids=(view.survey_name,)))
```

(If P1 stores δ as one var-len list per LOS row rather than one row per pixel, replace the per-id `groupby` with a direct column read — verify against the SIGHTLINE fixture before Step 4.)

- [ ] **Step 5: Run — PASS. Step 6: Commit** `measure/lya-T2: sightline_from_view (objects + per-LOS δ/mask/continuum)`.

---

## Task 3: `build_lya`
**Files:** Create `measure/lya.py`; Test `test/test_measure_lya.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_lya.py
import sys
from pathlib import Path

from oneuniverse.measure import MeasurementSet
from oneuniverse.measure.lya import build_lya

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_sightline_view  # noqa: E402


def test_build_lya_p1d(tmp_path):
    view = synthetic_sightline_view(tmp_path, n_los=20, n_pix=24, seed=2)
    ms = build_lya(view, tracer="lya", statistic="p1d", nside_region=16)
    assert isinstance(ms, MeasurementSet)
    sl = ms.products["lya"]
    assert sl.kind == "sightline" and sl.n_sightlines == 20
    assert ms.spec.statistic == "p1d"
    assert ms.spec.estimator_family == "lya"
    ms.check_invariants()
```

- [ ] **Step 2: Run — FAIL. Step 3: Implement `measure/lya.py`**

```python
"""Lyα forest connection. Cosmology-free (r_∥/r_⊥ + P(k) are P2)."""
from __future__ import annotations

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.sightline import sightline_from_view
from oneuniverse.measure.spec import MeasurementSpec


def build_lya(view: DatasetView, *, tracer: str = "lya",
              statistic: str = "p1d", nside_region: int = 16) -> MeasurementSet:
    sl = sightline_from_view(view, nside_region=nside_region)
    spec = MeasurementSpec(tracers=(tracer,), pairs=((tracer, tracer),),
                           statistic=statistic, estimator_family="lya")
    return MeasurementSet(products={tracer: sl}, spec=spec,
                          metadata=sl.metadata)
```

(Generalise `MeasurementSet.check_invariants` Task: it must tolerate a `Sightline` product — the catalog-length check uses `getattr(p, "catalog", None)` which is `None` for Sightline, so it already skips; verify `len(p.region_map) == n_sightlines`. If a stricter per-subtype check is wanted, add `Sightline`-aware branch — note but keep minimal.)

- [ ] **Step 4: Run — PASS. Step 5: Commit** `measure/lya-T3: build_lya (Sightline MeasurementSet, P1D/P3D spec)`.

---

## Task 4: Visual + close-out
- [ ] `scripts/build_measure_lya_demo.py`: a few example δ_F(λ) sightlines + the LOS sky distribution coloured by region → `test/test_output/measure_lya.png`. Visual test.
- [ ] Full suite green; `CLAUDE.md` + `plans/README.md` + memory updates (Sightline subtype + Lyα connection).
- [ ] Commit `measure/lya-T4: Lyα demo + docs`.

## Success criteria
- `Sightline` subtype carries per-LOS δ/mask/continuum + LOS metadata + shared region_map; cosmology-free.
- `build_lya` emits a Sightline MeasurementSet with `p1d`/`p3d` spec. `check_invariants` tolerates the non-PointSet subtype. Full suite green.

## Maps to requirements research
[`research/2026-06-05-p1-to-p2-measurement-requirements.md`](../research/2026-06-05-p1-to-p2-measurement-requirements.md) §2 geometry A(Sightline); §3 Lyα P_1D / P_3D rows; §4 statistic ∈ {p1d, p3d}.

## Self-review
- [ ] `Sightline(los, delta, mask, continuum)` signature consistent across T1–T3.
- [ ] `DataGeometry.SIGHTLINE` + `objects_table()` per-LOS contract verified against the fixture before T2.
- [ ] `check_invariants` confirmed to skip catalog-length for Sightline (catalog is None).
- [ ] No cosmology in `measure/lya.py`.
