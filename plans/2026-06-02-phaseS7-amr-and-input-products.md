# Phase S7 — AMR octree layout + input/IC products

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cover the two storage shapes the linear demo has not yet
exercised: (1) a **non-Cartesian AMR octree** field layout (`sim_kind=
"amr"`), and (2) the **input / initial-conditions** side of the format
(`has_input=True`) — the white-noise realisation + IC descriptor a run
starts from. Both use the existing OUF-Sim store stack (parquet + index +
manifest) with new partition keys (Morton octree-node ids; IC grid tiles).

**Architecture:** A toy AMR is built by **refining cells around density
peaks** of the regular linear field into one level of 2× sub-cells — a
1-level octree. Stored as base-grid `.npy` tiles (unrefined) + a refined-
node table keyed by **Morton code + level**, with an octree-node index for
`Cube` partial access (select base tiles **and** refined nodes overlapping
the cube). The IC product stores the seeded white-noise field (the thing a
forward model integrates from) + a JSON `ic_descriptor` (seed, P(k) model,
box, grid), setting `has_input=True`. No sampler is run (Rule 4) — the IC
is the *deterministic* realisation the dummy already generates.

**Tech Stack:** numpy, pyarrow, healpy, pyyaml. Reuse `oufsim/_morton.py`
(S6). **Rule 1:** no `oneuniverse.data` / `combine` imports.

---

## File Structure

- Create: `linear/amr.py` — toy 1-level octree refinement around peaks.
- Create: `linear/ic.py` — IC (white-noise) realisation + descriptor.
- Modify: `linear/generate.py` — `with_amr`, `with_ic` flags + native files.
- Modify: `oufsim/write.py` — `_write_amr` (octree-node index) + `_write_ic`.
- Modify: `oufsim/read.py` — `read_amr_box` partial access.
- Modify: `linear/converter.py` — declare `fields`(amr-tagged) + ic products.
- Tests: `test/test_lin_amr.py`, `test/test_lin_ic.py`,
  `test/test_oufsim_amr.py`, extend `test/test_oufsim_store.py`.

## Pre-flight

- [ ] **Step 0: Baseline green.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest test/test_lin_*.py test/test_oufsim_*.py -q 2>&1 | tail -3
```

---

## Task 1: Toy AMR refinement generator

**Files:** Create `linear/amr.py`; Test `test/test_lin_amr.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_lin_amr.py
"""Phase S7 T1 — toy AMR refinement."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.amr import refine_field
from oneuniverse.simulation.linear.gaussian_field import generate_density_field


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81)


def test_refines_only_above_threshold():
    c = _cosmo()
    d = generate_density_field(c, box_size=200.0, n_grid=32, z=0.0, seed=4)
    amr = refine_field(d, threshold=1.5)
    # refined nodes correspond to base cells above threshold
    assert amr["n_refined"] == int((d > 1.5).sum())
    # each refined node carries 8 sub-cell values (1 level, 2x)
    assert amr["subcells"].shape[1] == 8
    # parent cell indices in range
    assert amr["parent_ix"].max() < 32
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `refine_field(delta, threshold, level=1)`:
  find cells `delta > threshold`; for each, create 8 sub-cells whose values
  are the parent value plus a deterministic trilinear-ish perturbation from
  the 6 neighbours (toy). Return dict: `parent_ix/iy/iz`,
  `subcells (n,8)`, `morton` (interleaved key of parent cell),
  `n_refined`, `level`. Pure numpy.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS7/T1: toy 1-level AMR refinement around density peaks`

---

## Task 2: AMR storage in the OUF-Sim store

**Files:** Modify `oufsim/write.py`; Test `test/test_oufsim_amr.py`.

- [ ] **Step 1: Failing test** — `write_oufsim_store(..., with_amr=True)`
  (or native AMR present) creates `fields_amr/z*/` with base tiles +
  `refined.parquet` + `_index.parquet` carrying per-node Morton + level +
  cell bbox; manifest `sim_kind` records amr capability; round-trip reads
  the refined table back.

```python
def test_amr_product_written(tmp_path):
    ...
    s = SimStore(store)
    assert "fields_amr" in s.layout or "fields" in s.products
    idx = s._index_rows(s.layout["fields_amr"]["z0.000"]["index"])
    assert all("morton" in r and "level" in r for r in idx)
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `_write_amr(prod_dir, base_field, amr, box,
  tile_cells)`: write base tiles (reuse `_write_field_tiles`) + a
  `refined.parquet` (parent cell + 8 subcells) + an index with `morton`,
  `level`, and the refined cell bbox in Mpc/h. Add `fields_amr` to the
  layout + a `ProductDecl("fields", "linear AMR octree", ("octree_node",),
  ("delta",))`.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS7/T2: AMR octree storage — base tiles + Morton-keyed refined-node index`

---

## Task 3: AMR partial-access read

**Files:** Modify `oufsim/read.py`; Test `test/test_oufsim_amr.py` (extend).

- [ ] **Step 1: Failing test** — `read_amr_box(z, cube)` returns
  `(base_subgrid, refined_rows)` where refined_rows are only nodes whose
  cell bbox overlaps the cube, and base is the stitched sub-grid (reusing
  `read_field_box`). `last_read_stats` shows nodes pruned.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `read_amr_box`: call `read_field_box` for the
  base; filter the refined index by `cube_overlaps_bbox`; read only those
  refined rows. Record `nodes_total`/`nodes_read`.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS7/T3: AMR partial-access read (base sub-grid + pruned refined nodes)`

---

## Task 4: Initial-conditions (input) product

**Files:** Create `linear/ic.py`; Modify `linear/generate.py`,
`oufsim/write.py`; Test `test/test_lin_ic.py`, `test/test_oufsim_store.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_lin_ic.py
"""Phase S7 T4 — initial-conditions product."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.ic import white_noise_ic


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81)


def test_ic_is_reproducible_and_described():
    field, desc = white_noise_ic(_cosmo(), box_size=200.0, n_grid=32, seed=7)
    field2, _ = white_noise_ic(_cosmo(), box_size=200.0, n_grid=32, seed=7)
    np.testing.assert_array_equal(field, field2)
    assert desc["seed"] == 7 and desc["n_grid"] == 32
    assert desc["pk_model"] == "eisenstein_hu_nowiggle"
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `white_noise_ic(cosmo, box_size, n_grid, seed)`
  → the seeded standard-normal field (the unscaled IC the GRF colours) +
  a descriptor dict. Add `with_ic` to `generate_linear_sim` writing
  `ic/field.npy` + `ic/descriptor.json`. In the writer, `_write_ic` emits an
  `ic` product (grid tiles + JSON descriptor) and sets `has_input=True` in
  the manifest. Product kind: `checkpoints` or `ic_posterior` per the
  envisioned slot — use `ic_posterior` (the deterministic IC realisation
  occupies the IC-posterior slot's mean).
- [ ] **Step 4: Run** `pytest test/test_lin_ic.py test/test_oufsim_store.py -v` — PASS.
- [ ] **Step 5: Commit** `phaseS7/T4: initial-conditions product — white-noise realisation + descriptor (has_input=True)`

---

## Task 5: Converter wiring + demo + close-out

**Files:** Modify `linear/converter.py`, `scripts/build_demo_oufsim.py`,
`CLAUDE.md`, `plans/README.md`, memory; Test
`test/test_oufsim_store.py` (extend), `test/test_visual_amr.py`.

- [ ] **Step 1:** `LinearSimConverter.declare_products` adds the AMR-tagged
  field + IC products when present; `detect`/`convert` unchanged.
- [ ] **Step 2:** Extend the demo to generate `with_amr=True, with_ic=True`;
  add an AMR refinement-map plot (base field + refined-cell overlay) and an
  IC white-noise slice plot.
- [ ] **Step 3:** Visual test asserts the AMR + IC plots exist.
- [ ] **Step 4:** Full suite green.
- [ ] **Step 5:** Docs — `CLAUDE.md` (AMR + IC products), `plans/README.md`
  (S7 → complete), memory append. With S5–S7 done, **all 9 `PRODUCT_KINDS`
  + AMR layout + both input/output sides** are exercised.
- [ ] **Step 6: Commit** `phaseS7/T5: converter wiring + AMR/IC demo plots + docs; AMR + input side complete`

---

## Self-review checklist

- [ ] AMR refines only above threshold; octree-node index round-trips.
- [ ] `read_amr_box` prunes refined nodes by cube overlap.
- [ ] IC field reproducible; descriptor complete; `has_input=True`.
- [ ] All 9 PRODUCT_KINDS now exercised by the linear backend.
- [ ] Rule 1 guard green; Rule 4 honoured (no sampler — IC is deterministic).

## Maps to pinned Pillar-3 rules

| Rule | Where |
|---|---|
| 2 — partial access | octree-node index gives `Cube` access over AMR |
| 4 — no mini-sim runs | IC is the stored deterministic realisation, not a sampled run |
| 5 — optimisation | AMR base tiles stay memmap-able; refined nodes Morton-keyed |
