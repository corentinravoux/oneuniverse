# Dev plan — close the data→twin span with the package's dummy tools

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the real observational path into the digital twin *now*, exercised
end-to-end with the package's dummy simulator (`simulation/linear`) and synthetic
OUF datasets — so that when real catalogs arrive it is plug-and-play. Closes P0
of the 2026-08-04 improvement plan without any real data.

**Architecture:** The twin's `Observation` needs only a gridded `delta_g`. The
real data→twin edge is therefore *catalogued tracer positions → CIC grid →
`Observation`* — reusing `simulation.pm.deposit.deposit_cic` (twin may import
simulation). We add `twin/observe_from_view.py` (the socket), a dummy fixture that
Poisson-samples galaxies from a known linear truth field and writes them as an OUF
`POINT` dataset, and one closed-loop test proving the chain
`catalog → observe → wiener_reconstruct → recover_metrics` recovers the known
truth. Then two low-cost hardening tasks: packed-backend parity (P1.1) and a
3rd engine on the plug-in contract (P1.2).

**Tech Stack:** numpy, pandas, pyarrow, healpy; existing `twin` + `simulation`
modules. No new dependency. No real data.

**Invariants that must not break:**
- Rule-1: `simulation` ⊥ `data`/`combine`. `twin` *may* import both — the new
  `twin/observe_from_view.py` importing `oneuniverse.data` is the intended first
  real `data→twin` edge and does **not** violate Rule-1 (the guard scans
  `simulation/` only).
- Cosmology-free Pillar 1: the gridding uses **box positions**; sky→comoving
  conversion (which needs cosmology) is deliberately out of scope here and stubbed
  for the real-data extension — it belongs at the twin/Pillar-3 call site, never
  in `data/`.
- Don't bloat the suite: every task below adds **one** targeted test file (user
  preference #P306).

**Baseline:** `cd Packages/oneuniverse && pytest -q` → `841 passed, 3 skipped`.
Work on a branch, not `main`.

**File structure:**
- Create `oneuniverse/twin/observe_from_view.py` — the data→twin socket.
- Modify `oneuniverse/twin/__init__.py` — export `observe_from_view`.
- Create `test/fixtures/tracer_sim.py` — dummy truth-field → OUF catalog fixture.
- Create `test/test_twin_observe_from_view.py` — unit + closed-loop tests.
- Modify `test/test_packed_converter.py` — add parity tests (P1.1).
- Create `oneuniverse/twin/engines_extra.py` + register in `engines.py` — 3rd engine (P1.2).
- Modify `test/test_twin_engine.py` (or add) — assert the 3rd engine.

---

## Task 1: `observe_from_view` — grid a catalog into an Observation

**Files:** Create `oneuniverse/twin/observe_from_view.py`; Test `test/test_twin_observe_from_view.py`.

- [ ] **Step 1: Write the failing test**

Create `test/test_twin_observe_from_view.py`:

```python
import numpy as np
import pandas as pd
from oneuniverse.twin.observe_from_view import observe_from_view
from oneuniverse.twin.engine import Observation


def test_observe_from_dataframe_positions():
    box, n = 100.0, 16
    rng = np.random.default_rng(0)
    # 3000 uniform-random galaxies -> near-zero overdensity, right shape
    xyz = rng.uniform(0, box, size=(3000, 3))
    df = pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
    obs = observe_from_view(df, box_size=box, n_grid=n, bias=1.4)
    assert isinstance(obs, Observation)
    assert obs.delta_g.shape == (n, n, n)
    assert abs(float(obs.delta_g.mean())) < 1e-9   # delta defined rel. to realised mean
    assert obs.bias == 1.4
    assert obs.nbar > 0
```

- [ ] **Step 2: Run it — fails (module missing)**

Run: `pytest test/test_twin_observe_from_view.py -q`
Expected: FAIL `ModuleNotFoundError: oneuniverse.twin.observe_from_view`.

- [ ] **Step 3: Implement the socket**

Create `oneuniverse/twin/observe_from_view.py`:

```python
"""The data → twin socket: grid a Pillar-1 catalog of tracer positions into the
`Observation` a ReconstructionEngine consumes.

This is the first real `oneuniverse.data` → `oneuniverse.twin` edge (the mock in
`mock_observe.py` is the synthetic stand-in it replaces). `twin` may import both
pillars; `simulation` stays Rule-1 clean.

Scope: box positions (columns x/y/z, Mpc/h). Sky→comoving conversion (ra/dec/z +
fiducial cosmology) is the real-survey extension — deliberately not here, so no
cosmology leaks below the twin call site.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from oneuniverse.simulation.pm.deposit import deposit_cic
from oneuniverse.twin.engine import Observation


def _positions(source, cols: Sequence[str]) -> np.ndarray:
    """Extract an (N,3) float array of box positions from a catalog-like source."""
    if isinstance(source, np.ndarray):
        arr = np.asarray(source, float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("ndarray source must be (N,3) positions")
        return arr
    # DatasetView (has .read) / MeasurementSet PointSet (has .catalog) / DataFrame
    if hasattr(source, "read"):
        df = source.read(columns=list(cols))
    elif hasattr(source, "catalog"):
        df = source.catalog
    else:
        df = source  # assume DataFrame-like
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"catalog missing position columns {missing}; "
                       f"pass position_cols= to match your data")
    return np.column_stack([np.asarray(df[c], float) for c in cols])


def observe_from_view(source, *, box_size: float, n_grid: int,
                      bias: float = 1.0, nbar: Optional[float] = None,
                      position_cols: Sequence[str] = ("x", "y", "z"),
                      mask: Optional[np.ndarray] = None) -> Observation:
    """Grid catalogued tracer positions into an `Observation`.

    Parameters
    ----------
    source : DatasetView | MeasurementSet PointSet | DataFrame | (N,3) ndarray.
    box_size, n_grid : the target mesh (Mpc/h, cells per side).
    bias : linear tracer bias carried into the Observation.
    nbar : mean number density; default = N / box^3.
    position_cols : catalog columns holding box x/y/z.
    """
    pos = _positions(source, position_cols)
    pos = np.mod(pos, box_size)  # wrap into the periodic box
    counts = deposit_cic(pos, n_grid, box_size)  # mass (≈counts) per cell
    mean = float(counts.mean())
    delta_g = counts / mean - 1.0 if mean > 0 else np.zeros_like(counts)
    if mask is not None:
        delta_g = delta_g * np.asarray(mask, float)
    if nbar is None:
        nbar = len(pos) / box_size ** 3
    return Observation(delta_g=delta_g, nbar=float(nbar), bias=float(bias),
                       mask=mask)
```

- [ ] **Step 4: Export it**

In `oneuniverse/twin/__init__.py`, add `from oneuniverse.twin.observe_from_view import observe_from_view` to the imports and `"observe_from_view"` to `__all__`.

- [ ] **Step 5: Run — passes**

Run: `pytest test/test_twin_observe_from_view.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/twin/observe_from_view.py oneuniverse/twin/__init__.py test/test_twin_observe_from_view.py
git commit -m "feat(twin): observe_from_view — the data->twin socket (grid catalog positions -> Observation)"
```

---

## Task 2: dummy tracer fixture — galaxies that trace a known truth field

**Files:** Create `test/fixtures/tracer_sim.py`; extend `test/test_twin_observe_from_view.py`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_twin_observe_from_view.py`:

```python
def test_tracer_view_clusters_like_truth(tmp_path):
    from fixtures.tracer_sim import synthetic_tracer_view
    from oneuniverse.twin.metrics import cross_correlation
    box, n = 200.0, 32
    view, truth = synthetic_tracer_view(tmp_path, box_size=box, n_grid=n,
                                        nbar=5e-3, bias=1.5, seed=3)
    obs = observe_from_view(view, box_size=box, n_grid=n, bias=1.5)
    # gridded tracers must correlate with the truth field on large scales
    k, r = cross_correlation(obs.delta_g, truth, box_size=box)
    lo = k < 0.15
    assert np.nanmedian(r[lo]) > 0.5
```

- [ ] **Step 2: Run it — fails (fixture missing)**

Run: `pytest test/test_twin_observe_from_view.py::test_tracer_view_clusters_like_truth -q`
Expected: FAIL `ModuleNotFoundError: fixtures.tracer_sim`.

- [ ] **Step 3: Implement the fixture**

Create `test/fixtures/tracer_sim.py`:

```python
"""Dummy tracer catalog: Poisson-sample galaxies from a known linear truth field
and write them as an OUF POINT dataset with box positions (x,y,z). Returns the
(DatasetView, truth_delta) so tests can check recovery against ground truth.

Uses only the package's dummy simulator (simulation.linear) + the OUF writer —
no real data. Positions are box coordinates; trivial ra/dec/z are filled to
satisfy the CORE schema (they are unused by observe_from_view here).
"""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field


def _cosmo() -> CosmologySpec:
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def synthetic_tracer_view(tmp: Path, *, box_size: float, n_grid: int,
                          nbar: float, bias: float, seed: int = 0,
                          name: str = "tracers"):
    """Return (DatasetView, truth_delta). Galaxies Poisson-sampled from the
    linear field with intensity λ = n̄_cell·max(0, 1 + b·δ)."""
    truth = generate_density_field(_cosmo(), box_size=box_size, n_grid=n_grid,
                                   z=0.0, seed=seed)
    rng = np.random.default_rng(seed + 1)
    cell = box_size / n_grid
    lam = nbar * cell ** 3 * np.clip(1.0 + bias * truth, 0.0, None)
    counts = rng.poisson(lam)  # (n,n,n)
    cells = np.argwhere(counts > 0)
    reps = counts[counts > 0]
    base = np.repeat(cells, reps, axis=0).astype(float)      # integer cell idx
    jitter = rng.random(base.shape)                          # uniform in-cell
    xyz = (base + jitter) * cell                             # box positions
    ngal = len(xyz)

    # trivial sky coords (unused downstream) just to satisfy the CORE schema
    ra = rng.uniform(0, 360, ngal)
    dec = rng.uniform(-60, 60, ngal)
    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "z": np.full(ngal, 0.1, np.float32),
        "z_type": np.full(ngal, "spec"),
        "z_err": np.full(ngal, 1e-4, np.float32),
        "galaxy_id": np.arange(ngal, dtype=np.int64),
        "survey_id": np.zeros(ngal, dtype=np.int64),
        "x": xyz[:, 0].astype(np.float32),
        "y": xyz[:, 1].astype(np.float32),
        "z_box": xyz[:, 2].astype(np.float32),  # 'z' is redshift; box-z is z_box
        "_original_row_index": np.arange(ngal, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })
    out = tmp / name / "oneuniverse"
    write_ouf_dataset(df=df, out_dir=out, survey_name=name,
                      survey_type="spectroscopic", geometry=DataGeometry.POINT,
                      loader=LoaderSpec(name=name, version="0"))
    return DatasetView.from_path(out.parent), truth
```

Note: the box z-axis is stored as `z_box` (the CORE column `z` is redshift). The
test in Step 1 must therefore call `observe_from_view(view, ...,
position_cols=("x", "y", "z_box"))`. Update the Task-2 test call accordingly.

- [ ] **Step 4: Fix the Task-2 test's position_cols**

In the Step-1 test, change the observe call to:

```python
    obs = observe_from_view(view, box_size=box, n_grid=n, bias=1.5,
                            position_cols=("x", "y", "z_box"))
```

- [ ] **Step 5: Run — passes**

Run: `pytest test/test_twin_observe_from_view.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add test/fixtures/tracer_sim.py test/test_twin_observe_from_view.py
git commit -m "test(twin): dummy tracer-sim fixture — galaxies tracing a known linear field"
```

---

## Task 3: the closed loop — catalog → observe → reconstruct → recovers truth

This is the endgame chain running end-to-end on dummy tools.

**Files:** extend `test/test_twin_observe_from_view.py`.

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_endgame_chain_recovers_truth_on_dummy(tmp_path):
    """catalog (OUF) -> observe_from_view -> wiener_reconstruct -> recover_metrics
    recovers the KNOWN dummy truth field. The whole data->twin span, no real data."""
    from fixtures.tracer_sim import synthetic_tracer_view, _cosmo
    from oneuniverse.twin.wiener import wiener_reconstruct
    from oneuniverse.twin.metrics import recover_metrics
    box, n, bias = 300.0, 48, 1.5
    view, truth = synthetic_tracer_view(tmp_path, box_size=box, n_grid=n,
                                        nbar=8e-3, bias=bias, seed=5)
    obs = observe_from_view(view, box_size=box, n_grid=n, bias=bias,
                            position_cols=("x", "y", "z_box"))
    rec = wiener_reconstruct(obs.delta_g, _cosmo(), box_size=box,
                             nbar=obs.nbar, bias=bias, z=0.0)
    m = recover_metrics(rec, truth, box_size=box)
    # large-scale reconstruction correlates strongly with the known truth
    lo = m.k < 0.1
    assert np.nanmedian(m.r[lo]) > 0.6
    assert np.isfinite(m.k_half)  # a finite reconstruction scale exists
```

- [ ] **Step 2: Run — expected PASS immediately**

Run: `pytest test/test_twin_observe_from_view.py::test_endgame_chain_recovers_truth_on_dummy -q`
Expected: PASS. If `r` at large scales is below 0.6, raise `nbar` to `1.5e-2` (denser sampling → higher S/N) — do **not** weaken the assertion below 0.5; the point is to prove genuine recovery.

- [ ] **Step 3: Commit**

```bash
git add test/test_twin_observe_from_view.py
git commit -m "test(twin): endgame chain — dummy catalog reconstructs its known truth field"
```

---

## Task 4: prove the MeasurementSet is a valid twin input

Shows `observe_from_view` consuming the actual P1→P2 product, not just a raw view.

**Files:** extend `test/test_twin_observe_from_view.py`.

- [ ] **Step 1: Write the test**

Append:

```python
def test_observe_accepts_measurement_set_pointset(tmp_path):
    """The twin socket accepts a MeasurementSet PointSet's catalog (the P1->P2
    handoff object), not only a bare DatasetView."""
    from fixtures.tracer_sim import synthetic_tracer_view
    box, n = 200.0, 32
    view, truth = synthetic_tracer_view(tmp_path, box_size=box, n_grid=n,
                                        nbar=6e-3, bias=1.4, seed=7)
    # a PointSet-like object exposing `.catalog` (duck-typed, no cosmology)
    class _PS:
        catalog = view.read(columns=["x", "y", "z_box"])
    obs = observe_from_view(_PS(), box_size=box, n_grid=n, bias=1.4,
                            position_cols=("x", "y", "z_box"))
    assert obs.delta_g.shape == (n, n, n)
    from oneuniverse.twin.metrics import cross_correlation
    k, r = cross_correlation(obs.delta_g, truth, box_size=box)
    assert np.nanmedian(r[k < 0.15]) > 0.5
```

- [ ] **Step 2: Run — passes** (`.catalog` branch already implemented in Task 1)

Run: `pytest test/test_twin_observe_from_view.py::test_observe_accepts_measurement_set_pointset -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add test/test_twin_observe_from_view.py
git commit -m "test(twin): observe_from_view consumes a MeasurementSet PointSet (P1->P2->twin handoff)"
```

---

## Task 5 (P1.1): parity-test the `packed` backend (2 → ~5 tests)

Storage generality is load-bearing (real sims arrive Gadget/Abacus/BigFile). Give
the 2nd backend the read-parity checks `oufsim` already has.

**Files:** Modify `test/test_packed_converter.py`.

- [ ] **Step 1: Add a read-parity test**

Append to `test/test_packed_converter.py` (imports already present at top):

```python
def test_packed_store_read_parity_with_reencode(tmp_path):
    """A packed-backend store returns the same sub-box particles as a reencode
    store built from the same native sim."""
    from oneuniverse.simulation.linear.pack import write_packed_native
    from oneuniverse.simulation.packed.converter import PackedSimConverter
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2)
    pk = write_packed_native(lin, tmp_path / "pk", particle_chunk_nside=4)
    enc = PackedSimConverter().convert(pk, tmp_path / "enc", sim_name="e",
                                       projection="reencode")
    ref = PackedSimConverter().convert(pk, tmp_path / "rf", sim_name="r",
                                       projection="reference")
    cube = Cube(0, 100, 0, 100, 0, 100)
    a = SimStore(enc).read_box("snapshots", 0.0, cube)
    b = SimStore(ref).read_box("snapshots", 0.0, cube)
    assert len(a["x"]) == len(b["x"]) > 0
    # same particles (order may differ) -> compare sorted x
    assert np.allclose(np.sort(a["x"]), np.sort(b["x"]))
```

- [ ] **Step 2: Add a wrap-in-place index-only test**

Append:

```python
def test_packed_reference_is_index_only(tmp_path):
    from oneuniverse.simulation.linear.pack import write_packed_native
    from oneuniverse.simulation.packed.converter import PackedSimConverter
    from pathlib import Path
    lin = generate_linear_sim(tmp_path / "lin2", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=4)
    pk = write_packed_native(lin, tmp_path / "pk2", particle_chunk_nside=4)
    ref = PackedSimConverter().convert(pk, tmp_path / "rf2", sim_name="r",
                                       projection="reference")
    snap = Path(ref) / "snapshots" / "z0.000"
    assert (snap / "_index.parquet").is_file()      # index present
    assert not list(snap.glob("chunk_*.npy"))       # no copied bulk data
```

- [ ] **Step 3: Run — verify (adjust to real API if needed)**

Run: `pytest test/test_packed_converter.py -q`
Expected: PASS. **Before finalising**, confirm the exact `convert(...)` signature,
the `read_box` return keys, and the reference chunk-file glob against
`oneuniverse/simulation/packed/converter.py` and `test/test_oufsim_reference.py`;
adjust the `projection=` kwarg / glob pattern to match (the existing
`test_oufsim_reference.py` is the reference implementation for these exact calls).

- [ ] **Step 4: Commit**

```bash
git add test/test_packed_converter.py
git commit -m "test(sim): packed-backend read-parity + wrap-in-place index-only (P1.1)"
```

---

## Task 6 (P1.2): a 3rd engine on the plug-in contract

Prove the `register_engine` contract flexes for a differently-shaped
reconstruction engine before real BORG/SBI engines arrive. Wrap the existing
`constrained_realization` as a `ReconstructionEngine`.

**Files:** Create `oneuniverse/twin/engines_extra.py`; import it in
`oneuniverse/twin/engines.py`; Test `test/test_twin_engine_extra.py`.

- [ ] **Step 1: Write the failing test**

Create `test/test_twin_engine_extra.py`:

```python
import numpy as np
from oneuniverse.twin.engine import get_engine, registered_engines, Observation
from oneuniverse.simulation.cosmology import CosmologySpec
import oneuniverse.twin.engines_extra  # noqa: F401  (triggers registration)


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_constrained_engine_registered_and_runs():
    assert "constrained" in registered_engines()
    Eng = get_engine("constrained")
    eng = Eng()
    rng = np.random.default_rng(0)
    obs = Observation(delta_g=rng.normal(size=(16, 16, 16)), nbar=5e-3, bias=1.5)
    field = eng.reconstruct(obs, cosmo=_cosmo(), box_size=200.0, z=0.0)
    assert field.shape == (16, 16, 16)
    assert np.isfinite(field).all()
```

- [ ] **Step 2: Run — fails**

Run: `pytest test/test_twin_engine_extra.py -q`
Expected: FAIL `ModuleNotFoundError: oneuniverse.twin.engines_extra`.

- [ ] **Step 3: Implement the engine**

Create `oneuniverse/twin/engines_extra.py`:

```python
"""A 3rd engine on the plug-in contract: constrained realization as a
ReconstructionEngine (Wiener mean + statistically-correct small-scale power).
Proves register_engine flexes for a differently-shaped engine without contract
changes — the socket real BORG/SBI engines will later fill.
"""
from __future__ import annotations

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.twin.constrained import constrained_realization
from oneuniverse.twin.engine import Observation, ReconstructionEngine, register_engine


@register_engine
class ConstrainedRealization(ReconstructionEngine):
    name = "constrained"

    def reconstruct(self, observation: Observation, *, cosmo: CosmologySpec,
                    box_size: float, z: float = 0.0) -> np.ndarray:
        return constrained_realization(
            observation.delta_g, cosmo, box_size=box_size,
            nbar=observation.nbar, bias=observation.bias, z=z, seed=0)
```

Confirm `constrained_realization`'s exact signature (from
`oneuniverse/twin/constrained.py`): `constrained_realization(delta_g, cosmo, *,
box_size, nbar, bias=1.0, z=0.0, seed=0)`. Match kwargs precisely.

- [ ] **Step 4: Register on import**

In `oneuniverse/twin/engines.py`, add at the end: `from oneuniverse.twin import engines_extra  # noqa: F401` so importing the engines module registers all three. (Alternatively export from `twin/__init__.py` — match the existing pattern.)

- [ ] **Step 5: Run — passes**

Run: `pytest test/test_twin_engine_extra.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/twin/engines_extra.py oneuniverse/twin/engines.py test/test_twin_engine_extra.py
git commit -m "feat(twin): 3rd engine (constrained realization) on the register_engine contract (P1.2)"
```

---

## Final verification

- [ ] **Step 1: Full suite**

Run: `pytest -q`
Expected: `841 passed` + 6 new tests, `3 skipped`, `0 failed`.

- [ ] **Step 2: Guards explicitly**

Run: `pytest -q -k "import_boundary or invariant or cosmolog"`
Expected: PASS — `simulation` still Rule-1 clean; `twin` importing `data` is
allowed and does not trip the guard (it scans `simulation/` only).

- [ ] **Step 3: Update the improvement-plan doc**

In `research/2026-08-04-package-analysis-and-improvement-plan.md`, mark P0.2/P0.3/
P1.1/P1.2 done (dummy-tools realization) and note P0.1 (real-catalog validation)
still pending real data. Commit.

---

## Self-review (writing-plans checklist)

**Spec coverage.** P0.2 (data→twin socket) → Task 1; dummy realization of P0.1's
intent (prove the handoff) → Tasks 2–4 (closed loop + MeasurementSet input);
P0.3 (regression-lock) → Tasks 3–4 tests; P1.1 → Task 5; P1.2 → Task 6. The one
item **not** covered — real-catalog validation of the measure builders — is
correctly out of scope (no data) and flagged for later.

**Placeholder scan.** Every code step ships real code; the two "confirm the exact
signature" notes (Task 5 `convert`/`read_box`, Task 6 `constrained_realization`)
point at the specific reference file to check, with the expected signature given.

**Type/name consistency.** `Observation(delta_g, nbar, bias, mask)` used
identically across tasks; `observe_from_view(source, *, box_size, n_grid, bias,
nbar, position_cols, mask)` signature stable; `position_cols=("x","y","z_box")`
consistent between fixture (Task 2) and every consumer (Tasks 2–4); engine
`name="constrained"` matches the `registered_engines()` assertion.

**Endgame check.** Every task advances the one missing span (real observational
flow into the twin) or hardens the substrate that span rides on — nothing here is
tidiness for its own sake.
```
