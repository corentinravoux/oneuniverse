# Phases S9–S14 — Digital-twin integration roadmap

> Turn the validated S8 *skeleton* into a structurally-complete data-driven
> digital-twin engine. **Linear** (cheap full-volume sim — coarse large
> scales) and **fast-PM** (the higher-fidelity mini-sim) are the reference
> plug-in simulators standing in for real codes (Gadget/BORG/…). The focus
> is **code structure / interfaces that would work with real simulators**,
> validated on the dummy — not real-simulation application. Each phase closes
> a specific weakness from the 2026-06-02 audit.

**Backends-agnostic principle:** every mechanism below is built against the
`ForwardEngine` / `ReconstructionEngine` contracts and the `SimStore`
partial-access API, so swapping linear/PM for a real code is a plugin change,
not a rewrite. linear/PM are *current plugins*, not assumptions.

**Sequence:** S9 → S10 → S11 → S12 → S13 → S14. Each is one phase, TDD,
committed task-by-task, validated on the linear+PM dummy.

---

## S9 — Data-driven dispatch (the junction)  · closes audit gap #1 (critical) · ✅ DONE 2026-06-02

**Problem.** `SimDatabase.dispatch` regenerates a Zel'dovich IC from a *seed*;
the twin's **constrained realization** (the data-informed IC) is never used.
The data→sim coupling — the defining feature of the *data-driven* twin —
isn't in the loop.

**Goal.** The orchestration runs `data → ReconstructionEngine → constrained
IC → ForwardEngine resim`, end to end, through a clean IC-source interface.

**Structure / tasks.**
- `ICSource` abstraction (in `resim/`): yields an IC field for a region from
  one of `{seed (fresh), parent-extracted, reconstruction(data)}`.
- Refactor the PM IC setup so `run_coupled` / `PMForwardEngine` accept a
  **provided IC field** (treat a density field as the linear field at
  z_start → Zel'dovich-displace → run), not only a seed.
- `SimDatabase.dispatch(request, ic_source=...)`: when `ic_strategy=
  "constrained_from_posterior"`, build the IC from a `ReconstructionEngine`
  (`WienerReconstruction` / `constrained_realization`) applied to mock data.

**Validated with linear+PM.** truth (linear) → mock-observe → constrained
realization → `dispatch` PM resim from that IC → the resim large-scale field
tracks the truth **where the data constrained it** (Gate-2 style). The IC
provenance is recorded as "constrained_from_posterior".

**Success.** A `SimDatabase` dispatch whose IC demonstrably came from *data*
(not a seed), and whose result correlates with the data-constrained truth at
large k. The end-to-end data-driven loop exists in the orchestrator.

---

## S10 — True zoom: multi-resolution refined ICs  · closes gap #2 (critical)

**Problem.** `run_coupled` uses the *parent* resolution — a same-res isolated
re-run, no fidelity gain. Real resimulation = *higher resolution* in the
region.

**Goal.** The zoom region is resimulated at **higher resolution** than the
parent: parent large scales preserved + **new small-scale power added**
(MUSIC/Panphasia idea), delivering an actual fidelity gain.

**Structure / tasks.**
- `refine_ic(coarse_sub, factor, cosmo, box)`: Fourier-upsample the extracted
  coarse IC sub-region to a finer grid (zero-pad in k) **and add high-k modes**
  for `k > parent Nyquist`, drawn from the same P(k) with new phases.
- `run_coupled(..., zoom_factor=R)` runs the PM at `n_zoom = R·n_region` in
  the buffer box.
- Reference for validation: a higher-resolution full-box PM.

**Validated with linear+PM.** linear (coarse) full → extract region →
`refine_ic`(×2) → PM zoom at 2× mesh → the zoom field's P(k) **resolves
higher k** than the parent-res run, while large scales still match the
high-res full reference (r(k) high at low–mid k).

**Success.** Zoom P(k) extends beyond the parent Nyquist with the right
amplitude; large-scale modes preserved. Actual fidelity gain demonstrated.

---

## S11 — COLA far-field coupling + partial-access resim  · closes gaps #3, #5 · ⚠️ PARTIAL 2026-06-02

> **Status (honest):** partial-access **store wiring done** (`run_coupled_from_store`
> — the resim consumes the parent IC from a `SimStore`). The **COLA far-field
> coupling is DEFERRED**: an experiment showed a *naive* far-field/external-tide
> injection helps at small buffer (+0.09) but **hurts at larger buffer (−0.08)**
> — it double-counts modes near the buffer cut. A correct COLA needs the
> LPT-subtraction frame (evolve the residual relative to the 2LPT trajectory,
> subtract the LPT force — no double-count), which is real work. Shipping the
> inconsistent version would violate the "no unphysical aspects" rule, so it is
> deferred to **S11b**. Physical tie-in: true region-local partial access (memory
> bounded to the buffer) *requires* the same global-LPT / local-residual split —
> so S11b unlocks both the smaller-buffer accuracy **and** the bounded-memory
> resim at once.

**Problem.** `far_field_potential` (S8.2) is built but **unused** — super-
buffer tides are dropped, so buffers must be large (sCOLA-*lite*). And
`run_coupled` builds the **whole-box** IC in memory (violates partial access,
Rule 2/5 — won't scale).

**Goal.** Real COLA-frame coupling (far-field as the long-range background →
**smaller buffers**), and the resim reads ICs/far-field for **only the buffer
region** via `SimStore` partial access.

**Structure / tasks.**
- COLA frame: evolve the PM **residual** relative to the far-field/2LPT
  large-scale trajectory (subtract the long-range part the far-field already
  supplies — avoid double-counting; high-pass the local PM force *or* low-pass
  the far-field at the buffer scale).
- Partial access: `run_coupled` obtains the parent IC + `far_field_box` for
  the buffer region through `SimStore.read_*` (no full-box regeneration).

**Validated with linear+PM.** linear full-sim's `gr_fields` φ(a) = the far
field; COLA-coupled PM resim **converges at a smaller buffer** than the
uncoupled S8.5 version (same target accuracy, less buffer). Peak memory of the
resim is bounded by the buffer region, not the full box.

**Success.** Buffer-for-fixed-accuracy drops measurably vs S8.5; resim peak
memory ≈ buffer size, not box size.

---

## S12 — Merge: multi-region tiling + conservation  · closes gaps #4, #11

**Problem.** No merge — one region only; can't rebuild a global field.

**Goal.** Stitch multiple resimulated regions into a coherent global field
with feathered overlaps + a conservation check.

**Structure / tasks.**
- `merge_fields(tiles, overlaps, feather="cosine")`: blend overlapping
  resimulated sub-fields into a global grid.
- Tile the volume → resimulate each tile (S11 coupling) → merge → global field.
- Conservation: total mass preserved within tolerance; seam continuity.

**Validated with linear+PM.** tile a box into a few overlapping regions,
PM-resimulate each, merge → global field matches the full-box PM reference
within tolerance; no visible seam; mass conserved.

**Success.** Tiled+merged global field r(k) vs full reference within tol;
seam discontinuity below threshold; mass conserved to ~%.

---

## S13 — Orchestration completeness: persistence + lineage + ensemble  · closes gap #9

**Problem.** `SimDatabase` state is in-memory only; no ensemble mode.

**Goal.** Persistent catalog + bitemporal lineage + **ensemble-over-prior**
orchestration (for mock suites / covariance / forecasting; the structure a
future SBI consumer would also use).

**Structure / tasks.**
- `save()/load()` catalog + lineage + requests to parquet (atomic, reuse
  `oufsim/_io`). Bitemporal lineage edges (valid-time).
- Ensemble mode: `request_ensemble(parent, n_realisations, vary=...)` → N
  `SimulationRequest`s (kind="ensemble", varying phase/parameters);
  `dispatch_ensemble` runs N resims and catalogs them with lineage.

**Validated with linear+PM.** a DB round-trips through save/load;
`request_ensemble` + `dispatch_ensemble` produces N PM realisations cataloged
under one parent with lineage edges.

**Success.** DB persists + reloads identically; an ensemble of N dummy resims
is produced, cataloged, and lineage-linked.

---

## S14 — Generality proof: external-style store-boundary contract  · closes gap #6

**Problem.** Generality is asserted with 2 in-house engines called directly;
there is no *external-style* boundary proving a real code would plug in the
same way.

**Goal.** Prove the engine contract across an **external-style boundary**
using PM as a stand-in for a real code: IC in via partial-access store read,
products out via store ingest — no orchestrator coupling to the engine's
internals.

**Structure / tasks.**
- Harden `ForwardEngine` to the **store boundary**: `consume` reads its IC
  from a `SimStore` (partial access), `run` writes products back via
  `write_oufsim_store` ingest. The orchestrator only sees store paths +
  the contract, never the engine's internals.
- Drive PM through this as the "external" engine (stand-in for Gadget/BORG);
  drive linear through it too — two engines over the *same* store boundary.

**Validated with linear+PM.** PM plugs through the hardened store-boundary
contract (IC from store → products to store) with no orchestrator change,
proving a real code would plug in identically.

**Success.** ≥2 engines satisfy the *store-boundary* contract (the real
generality test) — a real backend becomes a drop-in plugin.

---

## Out of scope (deferred — future)

- **Simulation-based inference (SBI)** (audit gap #7): deferred to the future
  bucket (owner decision 2026-06-02). The ensemble-over-prior orchestration
  (S13) is the structure a future SBI consumer would build on, but the SBI
  pipeline itself (summaries → trained posterior estimator) is not in this
  roadmap.
- Hydro / baryons / RT generation (gap #8): the two dummy simulators are
  DM/linear only; multi-physics *generation* waits for real codes (storage
  side already supports the product kinds).
- Real-format backends (Gadget/Abacus/BORG ingest): `future` bucket. S14
  proves the *contract* they'll use; the readers themselves are later.
- P1+P2 community track: deferred several months (owner decision).

## Coverage after S9–S14 (vs decided science cases)

| Science case | After S9–S14 |
|---|---|
| Data-driven digital twin (P1+P3) | ✅ wired end to end (S9) |
| True zoom / higher-fidelity resim | ✅ (S10) |
| Real scale-coupling (COLA) + partial access | ✅ (S11) |
| Multi-region global twin (merge/tile) | ✅ (S12) |
| Orchestration persistence + ensemble | ✅ (S13) |
| Generality (external-style store boundary) | ✅ (S14) |
| SBI | ⏳ future (S13 ensemble is the substrate it will use) |
| Multi-physics generation | ⏳ future (real backends) |

After S9–S14 the **code structure** covers every decided science case using
linear+PM as stand-ins; only real-simulator *application*, multi-physics, and
SBI remain, which are intentionally future.
