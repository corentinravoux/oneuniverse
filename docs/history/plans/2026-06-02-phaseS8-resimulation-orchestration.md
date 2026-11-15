# Phase S8 — Resimulation orchestration (the digital-twin core)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. This is a **master plan over six sub-phases (S8.0–S8.6)**; each sub-phase is sized like a normal phase and should be executed (and committed) in order. Do not start S8.x+1 until S8.x's verification gate passes.

**Goal:** Build and *empirically test* the machinery to (a) run a cheap
**full-volume** simulation that carries the large scales, (b) **extract a
region's initial conditions**, (c) **re-simulate that sub-volume** with a
higher-fidelity **mini-simulator** using the full-sim's large-scale forces
as a background + **buffer/overlap** to suppress border effects, (d)
**merge** the result back, and (e) **orchestrate** the whole loop from
`SimDatabase`. The success criterion is the user's: *the mini-sim and the
full-sim must agree on the large-scale density of the shared volume* — as a
**pre-run necessary** test and a **post-run sufficient** test.

**Architecture & physics basis:** This is, by construction, **sCOLA**
(spatially-split COLA) + **phase-consistent zoom ICs** + a
**separate-universe external tidal background**. See the critical
feasibility study — **read it before executing**:
[`research/2026-06-02-resimulation-orchestration-feasibility.md`](../research/2026-06-02-resimulation-orchestration-feasibility.md).
The full-sim's global potential mesh φ(x; a) (`gr_fields`, S5) is the
**long-range-force provider** every mini-sim consumes; the mini-sim solves
only the near field — the same scale-split as TreePM/AMR (which is why the
user's AMR intuition is correct).

**Honest scope (from the research verdict):** the achievable deliverable is
a **large-scale-consistent local zoom engine with a quantified error
budget** — *not* a full-sim replacement and *not* a way to update the large
scales from mini-sims (the small→large back-reaction is not deterministically
recoverable). **S8.5 Gate 2/3 is a real go/no-go**: if the post-run
large-scale agreement fails tolerance even at large buffers, report that
honestly as the result.

**Tech stack:** numpy, scipy (FFT/KDTree), pyarrow, healpy. **Rule 1:** no
`oneuniverse.data` / `combine` imports. **Rule 4 update (user directive
2026-06-02):** dummy **fast-PM mini-sim runs are now in scope** for
feasibility testing; heavy *real-code* runs (Gadget/Abacus/JaxPM/BORG) stay
in `future`.

---

## Module plan (created across the sub-phases)

- `oneuniverse/simulation/pm/` — fast PM mini-simulator (`deposit.py`,
  `poisson.py`, `integrator.py`, `run.py`, optional `cola.py`).
- `oneuniverse/simulation/resim/` — orchestration: `farfield.py`,
  `ic_extract.py`, `coupling.py`, `merge.py`, `verify.py`.
- Reuse: `oufsim/database.py` (`SimDatabase`) as the S8.6 control plane.

---

## Sub-phase S8.0 — Research & critical feasibility study  ✅ (this session)

**Deliverable:** the feasibility study (link above). Maps the idea to
sCOLA/COLA/zoom-ICs/separate-universe/TreePM; decomposes the operations;
defines Gates 1–3; gives the achievable/approximate/impossible verdict.
**No code.** Status: **complete 2026-06-02.** Everything below implements +
tests its conclusions.

---

## Sub-phase S8.1 — Fast PM mini-simulator

A vanilla particle-mesh N-body: CIC deposit → FFT Poisson → KDK leapfrog.
Validated against linear theory + Zel'dovich. Standalone (no orchestration).

**Files:** `oneuniverse/simulation/pm/{deposit,poisson,integrator,run}.py`;
tests `test/test_pm_*.py`.

- [ ] **T1 — CIC deposit + FFT Poisson.** Test: a single point mass yields a
  monopole potential whose gradient ∝ 1/r² at large separation; deposited
  mass equals total mass (CIC conserves mass).
  Impl: `deposit_cic(pos, n_grid, box)`; `solve_poisson(delta_k, box)` with
  `φ_k = -δ_k / k²`, `k=0` → 0; force `g = -∇φ` via `i k φ_k`.
  Commit: `phaseS8.1/T1: CIC deposit + FFT Poisson solver`.

- [ ] **T2 — KDK leapfrog with cosmological factors.** Test: a single
  growing-mode perturbation grows as `D(a)` to <2% over a→a in the linear
  regime. Impl: kick/drift with `a`-dependent factors (Quinn et al. /
  FastPM-style); use the S3 growth `D(a)`.
  Commit: `phaseS8.1/T2: KDK leapfrog integrator (cosmological kick/drift)`.

- [ ] **T3 — `run_pm` driver + linear validation.** Test: from Zel'dovich
  ICs at z=9 (S3), evolve to z=0; binned low-k `P(k)` matches
  `linear_power(z=0)` within a few % (large scales) — the headline PM
  validation. Impl: `run_pm(pos, vel, box, n_grid, a_start, a_end, n_steps)`.
  Commit: `phaseS8.1/T3: run_pm + linear-growth P(k) validation`.

- [ ] **T4 — COLA frame (optional accuracy).** Test: COLA-framed run with
  ~10 steps reproduces the full-PM large-scale `P(k)` (subtract 2LPT, evolve
  residual; large scales exact). Impl: `pm/cola.py`.
  Commit: `phaseS8.1/T4: COLA frame — 2LPT far-field + PM residual`.

**Gate:** PM reproduces linear growth + Zel'dovich. If not, fix before S8.2.

---

## Sub-phase S8.2 — Full-volume far-field provider

Export the full-sim's **global potential mesh φ(x; a)** + 2LPT displacement
at requested scale factors — the long-range force every mini-sim consumes.
Reuses the `gr_fields` φ product (S5).

**Files:** `oneuniverse/simulation/resim/farfield.py`; tests
`test/test_resim_farfield.py`.

- [ ] **T1 — potential mesh.** Test: `∇²φ ≈ δ` to FFT tolerance.
  Impl: `potential_mesh(delta, box)` (reuse S5 `gr_fields`).
  Commit: `phaseS8.2/T1: global potential mesh phi from delta`.

- [ ] **T2 — far-field export over scale factors.** Test: φ(a) scales as the
  growth `D(a)` (linear); 2LPT displacement shape correct.
  Impl: `export_farfield(cosmo, field, box, scale_factors)` → φ(a) + Ψ_2LPT.
  Commit: `phaseS8.2/T2: far-field export phi(a) + 2LPT displacement`.

- [ ] **T3 — sub-region far-field service (partial access).** Test: φ
  restricted to a sub-cube equals the full φ on that cube (uses S4 tile
  index). Impl: `farfield_box(store, z, cube)`.
  Commit: `phaseS8.2/T3: serve far-field over a sub-region via tile index`.

---

## Sub-phase S8.3 — Region IC extraction + linkage + PRE-RUN verification (Gate 1)

Extract the Lagrangian sub-region white noise from the parent (phase
consistent), build the mini-sim IC inheriting the parent's large-scale
modes, and verify the **necessary** condition before running anything.

**Files:** `oneuniverse/simulation/resim/ic_extract.py`,
`resim/verify.py`; tests `test/test_resim_ic_extract.py`,
`test/test_resim_gate1.py`.

- [ ] **T1 — extract sub-region white noise.** Test: the extracted
  white-noise sub-cube equals the parent white-noise slice exactly (phase
  consistency). Impl: `extract_region_whitenoise(parent_wn, lagrangian_cube,
  box, n_grid)` (the S7 IC product is the parent white noise).
  Commit: `phaseS8.3/T1: extract phase-consistent sub-region white noise`.

- [ ] **T2 — build mini IC with buffer.** Test: the mini IC's low-k modes
  equal the parent's on the shared volume; buffer halo carries parent modes.
  Impl: `build_mini_ic(parent_wn, cube, buffer, cosmo, box, n_grid)` →
  colour with P(k); attach buffer of parent field.
  Commit: `phaseS8.3/T2: build_mini_ic (sub-region + buffer, parent modes)`.

- [ ] **T3 — Gate 1: pre-run consistency.** Test: `gate1_consistency(mini_ic,
  full_field, cube)` returns `r(k)→1` and `P_mini/P_full ∈ [1−ε,1+ε]` for
  `k < k_buffer`; assert it passes for a correctly-extracted IC and **fails**
  for a deliberately phase-scrambled IC. Impl: smooth both, FFT, cross-power
  / auto-power ratios. Impl in `resim/verify.py`.
  Commit: `phaseS8.3/T3: Gate 1 pre-run large-scale consistency check`.

**Gate 1:** IC extraction is correct (necessary condition). This is a unit
test on the linkage, exact by construction.

---

## Sub-phase S8.4 — Far-field coupling + buffers/overlap

Run the mini-sim with the full-sim far field as background (COLA frame),
inside a buffer, with isolated boundary conditions — the dynamical hard part.

**Files:** `oneuniverse/simulation/resim/coupling.py`; tests
`test/test_resim_coupling.py`.

- [ ] **T1 — COLA-frame mini-sim (no double-count).** Test: in the linear
  regime, the coupled mini-sim's large-scale displacement equals the
  full-sim's on the shared volume (far field carried exactly; the PM
  residual must not re-add the long-range force). Impl: `run_coupled(mini_ic,
  farfield, cube, buffer, ...)` — subtract the long-range component the far
  field already supplies.
  Commit: `phaseS8.4/T1: COLA-frame coupling (far-field background, no double-count)`.

- [ ] **T2 — buffer + isolated BC.** Test: the **inner** (non-buffer) region
  result is insensitive (within tol) to doubling the buffer once buffer ≳ a
  few × rms displacement; periodic vs zero-padded BC compared. Impl: buffer
  padding + optional zero-padded (open) Poisson solve.
  Commit: `phaseS8.4/T2: buffer + isolated BC; inner-region buffer-insensitivity`.

- [ ] **T3 — uniform tidal background (optional, separate-universe).** Test:
  imposing a known uniform tidal tensor `T_ij` produces the expected
  anisotropic distortion sign/scaling. Impl: anisotropic drift factor (mark
  ADVANCED; may defer). Impl: `coupling.py` tidal hook.
  Commit: `phaseS8.4/T3: uniform external tidal background (anisotropic, optional)`.

---

## Sub-phase S8.5 — Extract, merge + POST-RUN verification (Gate 2/3 — go/no-go)

Extract the inner region, merge overlapping mini-sims, and run the
**sufficient** test + the convergence sweep. **This decides whether the
method works on real numbers.**

**Files:** `oneuniverse/simulation/resim/merge.py`, extend `resim/verify.py`;
tests `test/test_resim_merge.py`, `test/test_resim_gate2.py`;
`scripts/resim_feasibility.py`.

- [ ] **T1 — inner-region extraction.** Test: the returned region excludes
  the buffer and matches the requested cube. Impl: `trim_buffer(result,
  cube, buffer)`.
  Commit: `phaseS8.5/T1: trim buffer -> inner region`.

- [ ] **T2 — merge overlapping mini-sims (field-level).** Test: density
  continuity across the overlap (no step at the seam beyond tol); total mass
  conserved within tol after feathered blend. Impl: `merge_fields(tiles,
  overlap, feather="cosine")`.
  Commit: `phaseS8.5/T2: feathered field-level merge across overlaps`.

- [ ] **T3 — Gate 2: post-run large-scale agreement.** Test: evolved low-k
  `r(k)` and `P_mini/P_full` on the inner region within tolerance for a
  chosen buffer (record the **feasibility number**). Impl: `gate2_dynamical(
  mini_result, full_result, cube)`.
  Commit: `phaseS8.5/T3: Gate 2 post-run large-scale consistency (feasibility number)`.

- [ ] **T4 — Gate 3: convergence + error budget.** Test: Gate-2 error
  **decreases** as buffer width grows (monotone within noise). Driver
  `scripts/resim_feasibility.py` sweeps buffer width (and tidal treatment)
  on the demo data, writes `RESIM_FEASIBILITY.md` + an error-vs-buffer plot.
  **Critical-physicist clause:** if error does not reach the target even at
  the largest buffer, the report must say so plainly.
  Commit: `phaseS8.5/T4: Gate 3 convergence sweep + honest error-budget report`.

**Gate 2/3:** the empirical verdict. Proceed to S8.6 only if resimulation
hits a usable tolerance for some buffer; otherwise document the negative
result and stop.

---

## Sub-phase S8.6 — `SimDatabase` orchestration control plane

The bookkeeping that drives the loop: catalog stores, track lineage, turn a
region selection into a `SimulationRequest`, **dispatch the dummy mini-sim**,
ingest + merge the child, and update the catalog/lineage. (Absorbs the
former standalone "SimDatabase" plan; now wired to S8.1–S8.5.)

**Files:** `oneuniverse/simulation/oufsim/database.py`,
`oufsim/lineage.py`; `scripts/orchestrate_demo.py`; tests
`test/test_simdatabase_*.py`, `test/test_visual_lineage.py`.

- [ ] **T1 — discover + catalog stores** (`scan()` over
  `*/oufsim/manifest.json`; query by product/box/z/cosmology).
  Commit: `phaseS8.6/T1: SimDatabase scan + query`.
- [ ] **T2 — lineage DAG** (`link(parent, child, region)`,
  `ancestors`/`descendants`).
  Commit: `phaseS8.6/T2: lineage edges + DAG traversal`.
- [ ] **T3 — region → request** (`request_region(parent, selector,
  ic_strategy, physics)` → pending `SimulationRequest`, sized against the
  parent's product indexes).
  Commit: `phaseS8.6/T3: region selection -> pending SimulationRequest`.
- [ ] **T4 — dispatch + ingest the dummy mini-sim** (extract IC → run_coupled
  → trim → merge → register child store + lineage edge; status
  pending→dispatched→running→ingested). Test: the child store exists and
  Gate-2 passes for the dispatched region. **(Rule-4 update: dummy runs
  allowed.)**
  Commit: `phaseS8.6/T4: dispatch+ingest dummy mini-sim; lifecycle to ingested`.
- [ ] **T5 — persist** catalog + lineage + requests (parquet, atomic);
  reload round-trip.
  Commit: `phaseS8.6/T5: persist catalog/lineage/requests`.
- [ ] **T6 — orchestration demo + close-out.** `orchestrate_demo.py`: open
  `linsim_demo`, select a sub-cube, run the full extract→mini-sim→merge→
  verify loop, draw the lineage graph + a full-vs-mini large-scale overlay.
  Full suite green; docs (`CLAUDE.md`, `plans/README.md`), memory.
  Commit: `phaseS8.6/T6: orchestration demo + lineage/overlay plots; Pillar-3 core complete`.

---

## Self-review checklist (whole program)

- [ ] PM reproduces linear growth + Zel'dovich (S8.1).
- [ ] Far-field φ(a) served per sub-region (S8.2).
- [ ] Gate 1 passes for correct ICs, fails for scrambled (S8.3).
- [ ] Coupled mini-sim carries large scales without double-counting; inner
      region buffer-insensitive (S8.4).
- [ ] Gate 2 feasibility number recorded; Gate 3 error decreases with buffer
      — **or** the negative result is documented (S8.5).
- [ ] Orchestrator drives extract→run→merge→verify and records lineage (S8.6).
- [ ] Rule 1 guard green throughout. Heavy real-code runs remain `future`.

## Maps to pinned Pillar-3 rules

| Rule | Where |
|---|---|
| 1 — minimal coupling | `pm/` + `resim/` use only numpy/scipy/pyarrow/healpy |
| 2 — partial access | far-field + IC served per sub-region via the S4 indexes |
| 3 — MPI/GPU | mini-sims are embarrassingly parallel (sCOLA tiling) |
| 4 — mini-sim runs | **relaxed for the dummy fast-PM** (feasibility); real codes still future |
| 5 — optimisation | buffer/overlap cost is the central tunable; Gate 3 quantifies it |
