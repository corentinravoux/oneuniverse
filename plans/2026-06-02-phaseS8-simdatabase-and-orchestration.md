# Phase S8 — SimDatabase + lineage + region-selection orchestration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the OUF-Sim **control plane** — a `SimDatabase` that
catalogs many OUF-Sim stores, tracks **lineage** (a parent sim → a zoom /
constrained child via a `RegionSpec`), and turns a region-selection query
into a `SimulationRequest`. This is the digital-twin loop's bookkeeping:
"given what we have simulated, decide and record what to simulate next."
**No simulation is run** (Rule 4) — the database stores, links, selects,
and emits requests as metadata only.

**Architecture:** Mirror Pillar 1's `database.py` discipline. The catalog
is a parquet table of discovered manifests; lineage + requests are parquet
edge/record tables written atomically alongside. `SimDatabase` reads
each store's `manifest.json` (S4) and exposes query / link / request APIs
over the existing `RegionSpec` + `SimulationRequest` types (S2). The
region selector reuses `Cube`/`Cone` + the store's product indexes to
size the requested region against existing data.

**Tech Stack:** pyarrow, numpy, healpy, pyyaml. **Rule 1:** no
`oneuniverse.data` / `combine` imports. **Rule 4:** stores/links/selects/
emits only — never dispatches a run, sampler, or solver.

---

## File Structure

- Create: `oufsim/database.py` — `SimDatabase` (discover / catalog / query /
  link / request / persist).
- Create: `oufsim/lineage.py` — lineage edge model + traversal helpers.
- Modify: `oufsim/__init__.py` — export `SimDatabase`.
- Create: `scripts/orchestrate_demo.py` — register demo sim, select a
  region, emit a request, draw the lineage graph.
- Tests: `test/test_simdatabase_*` (one per task), `test/test_visual_lineage.py`.

## Pre-flight

- [ ] **Step 0: Baseline green.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest test/test_oufsim_*.py test/test_sim_*.py -q 2>&1 | tail -3
```

---

## Task 1: Discover + catalog stores

**Files:** Create `oufsim/database.py`; Test `test/test_simdatabase_catalog.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_simdatabase_catalog.py
"""Phase S8 T1 — discover + catalog OUF-Sim stores."""
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import write_oufsim_store
from oneuniverse.simulation.oufsim.database import SimDatabase


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_discovers_stores(tmp_path):
    root = tmp_path / "root"
    for name in ("simA", "simB"):
        native = generate_linear_sim(tmp_path / name, _cosmo(),
                                     box_size=150.0, n_grid=16,
                                     redshifts=(0.0,), seed=1)
        write_oufsim_store(native, root, sim_name=name)
    db = SimDatabase(root)
    db.scan()
    assert set(db.sim_names()) == {"simA", "simB"}
    rec = db.get("simA")
    assert rec["box_size"] == 150.0 and "snapshots" in rec["products"]
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — `SimDatabase(root)`; `scan()` globs
  `*/oufsim/manifest.json`, reads each (S4 `read_json`), and builds an
  in-memory catalog keyed by `sim_name` with `{box_size, n_grid, redshifts,
  products, cosmology, store_path}`. `sim_names()`, `get(name)`.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS8/T1: SimDatabase.scan — discover + catalog OUF-Sim manifests`

---

## Task 2: Query the catalog

**Files:** Modify `oufsim/database.py`; Test `test/test_simdatabase_query.py`.

- [ ] **Step 1: Failing test** — `db.query(product="lightcone")`,
  `db.query(box_min=100)`, `db.query(omega_m=0.31)` each return the matching
  sim names.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — `query(**filters)` filtering the catalog on
  product membership, box-size range, redshift coverage, and cosmology
  fields (exact / tolerance).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS8/T2: SimDatabase.query — filter by product/box/z/cosmology`

---

## Task 3: Lineage links

**Files:** Create `oufsim/lineage.py`; Modify `oufsim/database.py`; Test
`test/test_simdatabase_lineage.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_simdatabase_lineage.py
"""Phase S8 T3 — parent->child lineage via RegionSpec."""
from oneuniverse.simulation.region import RegionSpec
# ... build db with parent "box" and child "zoom" ...


def test_link_and_traverse(tmp_path):
    db = ...  # two registered sims
    region = RegionSpec(region_id="z1", kind="zoom",
                        eulerian_bbox=(0, 50, 0, 50, 0, 50),
                        lagrangian_patch=None, cone=None, z=0.0,
                        mass=None, refs=())
    db.link(parent="box", child="zoom", region=region)
    assert db.children_of("box") == ["zoom"]
    assert db.parent_of("zoom") == "box"
    assert db.ancestors("zoom") == ["box"]
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — `lineage.py` holds `LineageEdge(parent, child,
  region)`; `SimDatabase.link/children_of/parent_of/ancestors/descendants`
  over the edge set (simple DAG traversal). Validate parent+child exist in
  the catalog.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS8/T3: lineage edges (parent->child via RegionSpec) + DAG traversal`

---

## Task 4: Region selection → SimulationRequest

**Files:** Modify `oufsim/database.py`; Test `test/test_simdatabase_request.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_simdatabase_request.py
"""Phase S8 T4 — emit a SimulationRequest from a region selection."""
from oneuniverse.simulation.request import SimulationRequest
from oneuniverse.simulation.selectors import Cube


def test_request_region_emits_pending(tmp_path):
    db = ...  # parent "box" registered
    req = db.request_region(parent="box", selector=Cube(0, 40, 0, 40, 0, 40),
                            ic_strategy="zoom_from_parent_ic", physics="dm")
    assert isinstance(req, SimulationRequest)
    assert req.status == "pending"
    # selection sized against the parent's data, no run executed
    assert req.ic_strategy == "zoom_from_parent_ic"
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — `request_region(parent, selector, ic_strategy,
  physics)` builds a `RegionSpec` from the selector, sizes it against the
  parent store's product indexes (how many particles/halos fall inside →
  recorded as request metadata), and returns a `SimulationRequest(status=
  "pending", ...)`. Append to `db.requests`. **No dispatch.**
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS8/T4: request_region -> pending SimulationRequest (metadata only, no run)`

---

## Task 5: Request lifecycle (metadata transitions)

**Files:** Modify `oufsim/database.py`; Test `test/test_simdatabase_lifecycle.py`.

- [ ] **Step 1: Failing test** — `db.set_status(req_id, "dispatched")` then
  `"running"` then `"ingested"` succeed in order; an illegal jump (e.g.
  `pending`→`ingested`) raises.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — a status state machine over
  `SimulationRequest._STATUSES` (`pending`→`dispatched`→`running`→
  `ingested`); reject out-of-order transitions. On `ingested`, optionally
  record the produced child store path (closing the lineage loop).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS8/T5: request lifecycle state machine (pending->dispatched->running->ingested)`

---

## Task 6: Persist catalog + lineage + requests

**Files:** Modify `oufsim/database.py`; Test `test/test_simdatabase_persist.py`.

- [ ] **Step 1: Failing test** — `db.save()` writes
  `sims_catalog.parquet` + `lineage.parquet` + `requests.parquet` under the
  root; `SimDatabase(root); db.load()` restores names, edges, requests.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — atomic parquet writes (reuse
  `oufsim/_io.py`); `save()`/`load()` round-trip the three tables. Catalog
  is rebuildable from `scan()` but persisted for fast open.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS8/T6: persist catalog + lineage + requests (parquet, atomic)`

---

## Task 7: Orchestration demo + close-out

**Files:** Create `scripts/orchestrate_demo.py`,
`test/test_visual_lineage.py`; Modify `oufsim/__init__.py`, `CLAUDE.md`,
`plans/README.md`, memory.

- [ ] **Step 1:** `orchestrate_demo.py`: open the `linsim_demo` store via
  `SimDatabase`, select a sub-cube region, emit a `SimulationRequest`,
  register a (placeholder) zoom child + lineage edge, `save()`, and draw a
  **lineage graph** (parent → zoom child, annotated with the region) +
  print the request record. Writes into the demo dir.
- [ ] **Step 2:** Run it; confirm the catalog/lineage/requests parquet files
  and the lineage plot are produced; no run is dispatched.
- [ ] **Step 3:** Visual test asserts the lineage plot exists.
- [ ] **Step 4:** Full suite green.
- [ ] **Step 5:** Export `SimDatabase`; docs — `CLAUDE.md` (control plane),
  `plans/README.md` (S8 → complete), memory append. **Pillar 3 substrate
  complete end to end: generate → store → optimise read/write → orchestrate
  next region.** Remaining work = `future` real-format backends.
- [ ] **Step 6: Commit** `phaseS8/T7: orchestration demo + lineage plot + docs; SimDatabase control plane complete`

---

## Self-review checklist

- [ ] `scan()` discovers every `*/oufsim/manifest.json`; catalog queryable.
- [ ] Lineage is a valid DAG; ancestors/descendants traverse correctly.
- [ ] `request_region` emits a pending request sized against real data,
      **without dispatching** anything (Rule 4).
- [ ] Lifecycle rejects illegal status jumps.
- [ ] Catalog/lineage/requests persist + reload.
- [ ] Rule 1 guard green.

## Maps to pinned Pillar-3 rules

| Rule | Where |
|---|---|
| 1 — minimal coupling | database reads only OUF-Sim manifests + parquet |
| 2 — partial access | region sizing uses the product indexes, not full loads |
| 4 — no mini-sim runs | emits requests + lineage only; never runs a sim |
