# Twin-coupling roadmap (refactored 2026-06-02)

> **For agentic workers:** master plan over the near-term coupling phases.
> All work is **dummy + mock data** (clean skeleton before any real-dataset
> loading). Each phase gets a detailed TDD plan at execution time.

**Decision (owner, 2026-06-02):** twin/coupling (P1+P3) is the focus for the
next several months; the community product (P1+P2) waits. Build a **clean
skeleton on simplified datasets + mocks** before touching real survey
loading. **Sequence: C2 → C3 → C4 → C5 → S5 → S6 → S7 → S8** — the coupling
skeleton first, then the simulation-substrate phases, then the big
resimulation stage (its S8.1 fast-PM mini-sim doubles as the second forward
engine).

**Through-line:** the twin loop is `data → reconstruct → field → forward →
products → compare`. C1 proved it (Wiener + r(k)). The C-series makes that
loop a **clean, contract-driven, validated skeleton** — still all synthetic.

---

## Active: the C-series (coupling skeleton, dummy + mock)

### C1 — minimal mock challenge ✅ (done, 616 tests)
truth → mock Poisson tracers → Wiener reconstruct → r(k) vs truth.
`oneuniverse/twin/`.

### C2 — engine contracts + first plugins
Make the generality contract real (the ADR `ForwardEngine`), with **both
roles** the loop needs:
- `oneuniverse/twin/engine.py`: `ReconstructionEngine` (data → constrained
  field) and `ForwardEngine` (field/IC + far-field → products) ABCs +
  light `EngineRun` / `ProductBundle` types.
- Refactor C1's Wiener into `WienerReconstruction(ReconstructionEngine)`.
- Wrap the linear sim as `LinearForwardEngine(ForwardEngine)`.
- **Proves generality on the dummy with one engine of each role.**
- Tests: each engine satisfies the ABC; the C1 loop runs through the
  contract unchanged.

### C3 — validation harness
Promote C1's metric into a first-class, reusable, regression-tested module:
- `oneuniverse/twin/validation.py`: `recover_metrics(rec, truth, box)` →
  r(k), power ratio, `k_half` (scale where r=0.5), bias/transfer.
- Every later increment reports through it → the twin becomes a *measurable*
  program (the methods-paper spine).
- Tests: metrics on known fields; k_half monotone with n̄.

### C4 — mock data / selection layer
The first realistic "data" shape — still synthetic, no `oneuniverse.data`
import yet:
- `oneuniverse/twin/mock_survey.py`: synthetic selection mask, mock survey
  geometry (masked box / wedge), mock n(z).
- Masked (apodised / non-periodic) Wiener so reconstruction works under a
  real-ish geometry (mode coupling from the mask).
- Tests: masked recon still recovers large scales inside the footprint;
  graceful outside.

### C5 — constrained realization (Hoffman–Ribak)
Turn the Wiener *mean* into a proper constrained *realization* (restore the
unconstrained small-scale variance) — the correct IC for resimulation:
- `oneuniverse/twin/constrained.py`: `constrained_realization(...)` =
  Wiener mean + (random − Wiener[random]).
- Tests: large scales match the data constraint; small-scale power restored
  to P(k); ensemble mean ≈ Wiener mean.

---

## After the C-series: S5 → S6 → S7 → S8 (owner, 2026-06-02)

Once the coupling skeleton (C2–C5) is in place, run the simulation-substrate
phases in order, then the big resimulation stage:

- **S5** — OUF-Sim write-path optimisation + full product coverage
  (tree/phase_space/gr_fields/checkpoints/ic_posterior).
- **S6** — OUF-Sim read-path optimisation (benchmark + tests).
- **S7** — AMR octree layout + input/IC products.
- **S8** — **resimulation orchestration (the big stage)**, six sub-phases
  (S8.0 research ✅; sCOLA + zoom ICs + separate-universe; feasible as
  controlled approximation). S8.1 fast-PM mini-sim is the **second
  `ForwardEngine`** + the forward half of the loop; then the `resim/`
  machinery (farfield, ic_extract, coupling, merge, verify) wiring C5's
  constrained IC → PM → Gate-2/3. Full plan:
  [`2026-06-02-phaseS8-resimulation-orchestration.md`](2026-06-02-phaseS8-resimulation-orchestration.md).

Full sequence: **C2 → C3 → C4 → C5 → S5 → S6 → S7 → S8**.

---

## Deferred

- **Track A — P1+P2** (community cross-correlation facilitator: `onemeasure`
  / MeasurementSet → flip): **deferred several months** (owner decision).

---

## Definition of done for this skeleton (defend against scope creep)

- The twin loop runs **end to end through the engine contracts** on dummy +
  mock data: mock survey → constrained realization → forward (linear; PM
  skeleton) → validation metrics.
- **≥2 engines** satisfy the contract (one reconstruction, one+ forward).
- Every step reports standard **validation metrics**; all regression-tested.
- **No real-dataset loading yet**; `oneuniverse.simulation` stays Rule-1
  clean; `twin/` may import both pillars but uses only mocks for now.
