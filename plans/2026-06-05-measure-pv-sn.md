# `oneuniverse.measure` — Peculiar Velocities + Supernovae Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Add the **peculiar-velocity** and **SN Ia** probes to `oneuniverse.measure`: distance-indicator atoms (μ / η / v_pec / σ_v), light curves, and a **row-correlated covariance handle** (Pantheon+-style), via `build_peculiar_velocity` / `build_sn_hubble` — cosmology-free `MeasurementSet`s.

**Architecture:** Reuse the spine (`select_clean`, `footprint_from_positions`, `assign_regions`, `MeasurementSet`, `PointSet`). New surface: distance atoms validated onto the PointSet catalog, a `CovarianceHandle` (cov_id → external matrix), and a light-curve carrier read from P1's LIGHTCURVE geometry. **Cosmology-free** — μ→distance and the Hubble fit are P2.

**Tech Stack:** numpy, pandas, pyarrow; reuses `DatasetView.read`/`objects_table`. No cosmology engine.

---

## Reused (already built)
`oneuniverse.measure`: `select_clean`, `footprint_from_positions`, `assign_regions`, `MeasurementSet`, `MeasurementSpec`, `PointSet`, `ProductMetadata`, `Provenance`, fixture pattern.

## Confirmed P1 APIs
- PV/SN columns live as ordinary OUF POINT columns (per the survey loaders): distance modulus, log-distance-ratio, peculiar velocity, errors, indicator type. `z_type="pv"` is a valid Z_TYPE.
- LIGHTCURVE geometry: `DatasetView.objects_table()` (one row per source) + `read()` (per-epoch rows). `DataGeometry.LIGHTCURVE`.

## File structure (new / modified)
| File | Responsibility |
|---|---|
| Create `measure/distances.py` | `attach_distances(cat, *, columns)` — validate μ/η/v_pec/σ_v + provenance |
| Create `measure/covariance.py` | `CovarianceHandle` (cov_id + lazy external matrix loader) |
| Create `measure/lightcurve.py` | `LightcurveSet` carrier + `lightcurves_from_view` (LIGHTCURVE geom) |
| Create `measure/pvsn.py` | `build_peculiar_velocity(...)`, `build_sn_hubble(...)` |
| Modify `test/fixtures/measure_ouf.py` | `synthetic_pv_view`, `synthetic_sn_view`, `synthetic_lightcurve_view` |
| Tests | one per task |

---

## Task 1: Distance-indicator atoms
**Files:** Create `measure/distances.py`; extend fixture (`synthetic_pv_view` with `mu, mu_err, eta, v_pec, sigma_v, dist_indicator`, `z_type="pv"`); Test `test/test_measure_distances.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_distances.py
import sys
from pathlib import Path

from oneuniverse.measure.distances import attach_distances

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_pv_view  # noqa: E402


def test_attach_distances_validates(tmp_path):
    view = synthetic_pv_view(tmp_path, n=1500, seed=1)
    cat = view.read()
    out, prov = attach_distances(cat, columns=("mu", "mu_err", "v_pec",
                                               "sigma_v"))
    assert {"mu", "v_pec", "sigma_v"} <= set(out.columns)
    assert "mu" in prov
    import pytest
    with pytest.raises(ValueError, match="distance column"):
        attach_distances(cat.drop(columns=["v_pec"]),
                         columns=("mu", "v_pec"))
```

- [ ] **Step 2: Run — FAIL. Step 3: Implement `measure/distances.py`**

```python
"""Distance-indicator atoms for PV/SN (μ, η, v_pec, σ_v). No cosmology."""
from __future__ import annotations

from typing import Sequence, Tuple

import pandas as pd


def attach_distances(catalog: pd.DataFrame, *, columns: Sequence[str]
                     ) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    """Validate that the requested distance-indicator columns are present."""
    missing = [c for c in columns if c not in catalog.columns]
    if missing:
        raise ValueError(f"attach_distances: missing distance column(s) "
                         f"{missing}")
    return catalog.copy(), tuple(columns)
```

- [ ] **Step 4: Run — PASS. Step 5: Commit** `measure/pvsn-T1: distance-indicator atoms (attach_distances)`.

---

## Task 2: Covariance handle (row-correlated SN cov)
**Files:** Create `measure/covariance.py`; Test `test/test_measure_covariance.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_covariance.py
import numpy as np

from oneuniverse.measure.covariance import CovarianceHandle


def test_covariance_handle_lazy_load(tmp_path):
    cov = np.diag(np.arange(1, 6, dtype=float))
    p = tmp_path / "cov.npy"; np.save(p, cov)
    h = CovarianceHandle(cov_id="sn5", path=str(p), n=5)
    assert h.n == 5
    assert not h.is_loaded
    mat = h.matrix()
    assert mat.shape == (5, 5) and h.is_loaded
    np.testing.assert_allclose(np.diag(mat), np.arange(1, 6))
```

- [ ] **Step 2: Run — FAIL. Step 3: Implement `measure/covariance.py`**

```python
"""Row-correlated covariance handle (e.g. Pantheon+ 1701x1701). Lazy load."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CovarianceHandle:
    cov_id: str
    path: str
    n: int
    _cache: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def is_loaded(self) -> bool:
        return self._cache is not None

    def matrix(self) -> np.ndarray:
        if self._cache is None:
            self._cache = np.load(self.path)
            if self._cache.shape != (self.n, self.n):
                raise ValueError(
                    f"CovarianceHandle({self.cov_id}): matrix shape "
                    f"{self._cache.shape} != ({self.n},{self.n})")
        return self._cache
```

- [ ] **Step 4: Run — PASS. Step 5: Commit** `measure/pvsn-T2: CovarianceHandle (lazy row-correlated covariance)`.

---

## Task 3: Light-curve carrier
**Files:** Create `measure/lightcurve.py`; extend fixture (`synthetic_lightcurve_view`: LIGHTCURVE geometry — objects table + per-epoch flux rows); Test `test/test_measure_lightcurve.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_lightcurve.py
import sys
from pathlib import Path

from oneuniverse.measure.lightcurve import LightcurveSet, lightcurves_from_view

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_lightcurve_view  # noqa: E402


def test_lightcurves_from_view(tmp_path):
    view = synthetic_lightcurve_view(tmp_path, n_obj=20, n_epoch=8, seed=2)
    lc = lightcurves_from_view(view)
    assert isinstance(lc, LightcurveSet)
    assert lc.n_objects == 20
    # epochs for one object
    one = lc.for_object(lc.object_ids[0])
    assert {"t", "flux", "band"} <= set(one.columns)
    assert len(one) == 8
```

- [ ] **Step 2:** Add `synthetic_lightcurve_view` to the fixture using `write_ouf_dataset(..., geometry=DataGeometry.LIGHTCURVE, objects_df=<one row per source>)` with a per-epoch `df` carrying `source_id, t, flux, flux_err, band`. (See `test/test_format.py` / any LIGHTCURVE test for the exact objects_df contract.)

- [ ] **Step 3: Run — FAIL. Step 4: Implement `measure/lightcurve.py`**

```python
"""Per-source light curves from P1's LIGHTCURVE geometry."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry


@dataclass
class LightcurveSet:
    objects: pd.DataFrame            # one row per source (objects_table)
    epochs: pd.DataFrame             # per-epoch rows (source_id, t, flux, band)
    id_column: str = "source_id"

    @property
    def object_ids(self) -> np.ndarray:
        return self.objects[self.id_column].to_numpy()

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    def for_object(self, oid) -> pd.DataFrame:
        return self.epochs[self.epochs[self.id_column] == oid]


def lightcurves_from_view(view: DatasetView, *, id_column: str = "source_id"
                          ) -> LightcurveSet:
    if view.geometry is not DataGeometry.LIGHTCURVE:
        raise ValueError(
            f"lightcurves_from_view: expected LIGHTCURVE, got "
            f"{view.geometry.value!r}")
    objects = view.objects_table().to_pandas()
    epochs = view.read()
    return LightcurveSet(objects=objects, epochs=epochs, id_column=id_column)
```

- [ ] **Step 5: Run — PASS. Step 6: Commit** `measure/pvsn-T3: LightcurveSet + lightcurves_from_view (LIGHTCURVE geometry)`.

---

## Task 4: `build_peculiar_velocity`
**Files:** Create `measure/pvsn.py`; Test `test/test_measure_pv.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_pv.py
import sys
from pathlib import Path

from oneuniverse.measure import MeasurementSet
from oneuniverse.measure.pvsn import build_peculiar_velocity

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_pv_view  # noqa: E402


def test_build_peculiar_velocity(tmp_path):
    view = synthetic_pv_view(tmp_path, n=3000, seed=3)
    ms = build_peculiar_velocity(
        view, tracer="pv", z_range=(0.0, 0.1),
        distance_columns=("mu", "mu_err", "v_pec", "sigma_v"),
        nside_region=4)
    assert isinstance(ms, MeasurementSet)
    ps = ms.products["pv"]
    assert {"v_pec", "sigma_v"} <= set(ps.catalog.columns)
    assert ms.spec.estimator_family == "velocity"
    assert "v_pec" in ps.provenance.weight_recipe or \
        "v_pec" in ps.provenance.extra.get("distance_columns", ())
    ms.check_invariants()
```

- [ ] **Step 2: Run — FAIL. Step 3: Implement `build_peculiar_velocity`**

```python
"""PV + SN connections. Cosmology-free (μ->distance, Hubble fit are P2)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.covariance import CovarianceHandle
from oneuniverse.measure.distances import attach_distances
from oneuniverse.measure.dataproduct import PointSet
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.regions import assign_regions
from oneuniverse.measure.select import select_clean
from oneuniverse.measure.spec import MeasurementSpec
from oneuniverse.measure.window import footprint_from_positions


def build_peculiar_velocity(view: DatasetView, *, tracer: str = "pv",
                            z_range: Tuple[float, float] = (0.0, 0.1),
                            distance_columns: Sequence[str] = (
                                "mu", "mu_err", "v_pec", "sigma_v"),
                            nside_window: int = 128, nside_region: int = 8
                            ) -> MeasurementSet:
    cat = select_clean(view, z_range=z_range)
    cat, dcols = attach_distances(cat, columns=distance_columns)
    win = footprint_from_positions(cat["ra"].to_numpy(),
                                   cat["dec"].to_numpy(), nside=nside_window)
    region = assign_regions(cat["ra"].to_numpy(), cat["dec"].to_numpy(),
                            nside=nside_region)
    cat = cat.copy(); cat["region_id"] = region
    meta = ProductMetadata(frame="icrs", epoch=2000.0, length_unit="deg",
                           nside_region=int(nside_region))
    prov = Provenance(dataset_ids=(view.survey_name,),
                      extra={"distance_columns": tuple(dcols)})
    ps = PointSet(catalog=cat, randoms=None, nz=None, window=win,
                  region_map=region, metadata=meta, provenance=prov)
    spec = MeasurementSpec(tracers=(tracer,), pairs=((tracer, tracer),),
                           statistic="velocity_correlation",
                           estimator_family="velocity")
    return MeasurementSet(products={tracer: ps}, spec=spec, metadata=meta)
```

- [ ] **Step 4: Run — PASS. Step 5: Commit** `measure/pvsn-T4: build_peculiar_velocity`.

---

## Task 5: `build_sn_hubble`
**Files:** Modify `measure/pvsn.py`; extend fixture (`synthetic_sn_view`: `zHD, mu, mu_err` + a saved cov.npy); Test `test/test_measure_sn.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_sn.py
import sys
from pathlib import Path

import numpy as np

from oneuniverse.measure.covariance import CovarianceHandle
from oneuniverse.measure.pvsn import build_sn_hubble

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_sn_view  # noqa: E402


def test_build_sn_hubble_with_cov(tmp_path):
    view, n = synthetic_sn_view(tmp_path, n=200, seed=4)
    cov = np.diag(np.full(n, 0.01))
    p = tmp_path / "sncov.npy"; np.save(p, cov)
    ms = build_sn_hubble(view, tracer="sn",
                         distance_columns=("mu", "mu_err"),
                         covariance=CovarianceHandle("sn", str(p), n),
                         nside_region=2)
    ps = ms.products["sn"]
    assert {"mu", "z"} <= set(ps.catalog.columns)
    assert ms.spec.statistic == "hubble"
    assert ps.provenance.extra["cov_id"] == "sn"
    ms.check_invariants()
```

- [ ] **Step 2: Run — FAIL. Step 3: Implement `build_sn_hubble`** (same pattern; `covariance: Optional[CovarianceHandle]=None` recorded in `provenance.extra["cov_id"]`; `MeasurementSpec(statistic="hubble", estimator_family="sn")`).

- [ ] **Step 4: Run — PASS. Step 5: Commit** `measure/pvsn-T5: build_sn_hubble (+ CovarianceHandle wiring)`.

---

## Task 6: Visual + close-out
- [ ] `scripts/build_measure_pvsn_demo.py`: PV sky map coloured by v_pec + SN Hubble scatter (μ vs z) → `test/test_output/measure_pv_sn.png`. Visual test.
- [ ] Full suite green; `CLAUDE.md` + `plans/README.md` + memory updates.
- [ ] Commit `measure/pvsn-T6: PV/SN demo + docs`.

## Success criteria
- `build_peculiar_velocity` carries v_pec/σ_v/μ atoms on a cosmology-free PointSet.
- `build_sn_hubble` carries μ + z + a `CovarianceHandle` (lazy external matrix).
- `LightcurveSet` reads P1 LIGHTCURVE geometry. Spine reused. Full suite green.

## Maps to requirements research
[`research/2026-06-05-p1-to-p2-measurement-requirements.md`](../research/2026-06-05-p1-to-p2-measurement-requirements.md) §2 atoms C(distances, light curves) + I(correlated covariance, cov_id); §3 PV + SN rows; §4 statistic ∈ {velocity, hubble}.

## Self-review
- [ ] `attach_distances` returns `(df, cols)`; `CovarianceHandle.matrix()` lazy; `lightcurves_from_view` -> `LightcurveSet` — matched at call sites.
- [ ] `DataGeometry.LIGHTCURVE` + `objects_table()` verified before T3.
- [ ] No cosmology in `measure/pvsn.py`.
