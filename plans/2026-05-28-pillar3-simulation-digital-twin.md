# Pillar 3 — Simulation and Digital Twin of the Universe

**Date:** 2026-05-28
**Scope:** Constrained Bayesian forward modelling of the actual 3D
matter density + velocity field of our Universe, fed by every
observation Pillar 1 standardises, with mini-simulation zoom-ins for
high-resolution astrophysics. This is the long-term scientific
motivation for the entire stack.

This document is a **large-scope roadmap**, not a task plan.
Implementation lives in a future `oneuniverse.simulation` subpackage
(or sibling package); concrete phases are scoped only after Pillars 1
and 2 reach minimum-viable.

---

## 1. Mission

Build a "digital twin" of the observed Universe by inferring the
underlying initial conditions (primordial density field) consistent
with every cosmological observation in Pillar 1. Forward-evolve the
inferred IC with an N-body simulation, apply per-survey observation
models, compare against the corresponding `MeasurementSet`, iterate
until consistent. Once large-scale structure is fixed, run
mini-simulations: zoom-in hydrodynamics in selected regions
(filaments, clusters, voids) at higher resolution. Update
incrementally when new data arrives.

This is the reason the package is called *one*universe.

## 2. Boundary clarity

| In scope | Out of scope |
|---|---|
| Forward-model the matter field from IC | Ingest / standardise raw catalogs |
| Per-survey observation models (selection, photo-z error, fiber assignment, shear systematics) | Cross-match across surveys |
| Compare synthetic to data `MeasurementSet`s | Compute estimator from data alone |
| Sample initial conditions (HMC / BORG-style) | Theory P(k) from CLASS / CAMB |
| Mini-simulation zoom-ins (hydro, RT) | Generic N-body library |
| Incremental updates as new surveys land | Long-tail engineering of legacy estimators |
| Cosmology baseline declaration + sampling | (cosmology kept here, never in Pillar 1) |
| `PARTICLE` geometry for N-body / hydro snapshots (AbacusSummit, MillenniumTNG, Quijote, …) | (Pillar 1 geometries are data-only) |
| Mock-catalog geometry + readers (Buzzard, Flagship lightcones) | |

## 3. Scientific framing

Every observable is a biased + noisy projection of the same
underlying field δ(x, z):

- **Galaxy positions** → δ via galaxy bias + RSD + survey window.
- **Peculiar velocities** → ∇⁻¹δ via continuity equation + linear/
  non-linear bias.
- **Lyman-α forest δ** → δ along QSO sightlines + FGPA / hydro.
- **Weak lensing shear** → projected δ along line of sight + IA bias.
- **CMB lensing κ** → integrated δ across the past lightcone.
- **SZ / X-ray clusters** → δ_cluster = ν · δ_field, with non-linear
  bias.
- **SNe Ia + standard sirens** → distance ladder anchoring
  H(z) + peculiar-velocity foregrounds.
- **HI / 21 cm** → δ_HI ∝ δ in linear regime, biased in non-linear.

Combining them via a single posterior on the underlying field breaks
degeneracies no single probe can resolve. Pillar 3 is where that
combination happens.

## 4. Architectural shape

```
                    ┌────────────────────┐
                    │ Pillar 1           │
                    │ MeasurementSet(s)  │  ← data side
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Pillar 3          │
                    │  oneuniverse.      │
                    │  simulation        │
                    │                    │
                    │  ┌──────────────┐  │
                    │  │ IC sampler   │  │
                    │  │ (HMC / NUTS) │  │
                    │  └──────┬───────┘  │
                    │         │          │
                    │  ┌──────▼───────┐  │
                    │  │ Forward      │  │
                    │  │ model        │  │
                    │  │ (N-body,     │  │
                    │  │  PM / COLA / │  │
                    │  │  full)       │  │
                    │  └──────┬───────┘  │
                    │         │          │
                    │  ┌──────▼───────┐  │
                    │  │ Observation  │  │  ← per-survey
                    │  │ model        │  │
                    │  │ (synthetic   │  │
                    │  │  Measurement │  │
                    │  │  Set)        │  │
                    │  └──────┬───────┘  │
                    │         │          │
                    │  ┌──────▼───────┐  │
                    │  │ Likelihood   │  │
                    │  │ vs data MS   │  │
                    │  └──────┬───────┘  │
                    └─────────┼──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Posterior over IC  │
                    │ → constrained      │
                    │   realisations     │
                    │ → mini-sim zoom-in │
                    └────────────────────┘
```

The bottom box is where Pillar 3 delivers science: posterior
samples of the initial conditions consistent with our actual
observations, plus the constrained realisations of structure today
that flow from them.

## 5. Subsystems

### 5.1 Initial-conditions sampler

Sample a high-dimensional Gaussian random field (the primordial
density field). State of the art:

- **HMC** with the forward model differentiable (BORG, BORG-PM).
- **NUTS** for problem sizes that fit (smaller volumes).
- **Variational inference** (e.g. SVI on a latent IC representation)
  for speed.
- **Flow-based posterior approximations** (normalising flows
  conditioned on data) — research-grade but actively progressing.

Likely starting point: wrap an existing implementation
(BORG-DESI, JaxPM-based samplers) rather than build from scratch.

### 5.2 Forward model

Evolve δ_IC at high redshift to δ(z=0) over the survey lightcone.
Tiered options:

| Tier | Method | Cost | Use case |
|---|---|---|---|
| L0 | Linear / Lagrangian PT | seconds | Smoke tests, large scales only |
| L1 | COLA / 2LPT | minutes | Quasi-non-linear, BAO scale |
| L2 | JaxPM (differentiable PM) | hours | Field-level inference |
| L3 | Full N-body (Gadget/HACC) | days | Final-pass realisation |
| L4 | Hydro / RT mini-sim | days–weeks | Zoom-in astrophysics |

L2 is the workhorse for inference. L3 is for the final realisation
once IC is converged. L4 is the "mini-simulation" deliverable.

### 5.3 Per-survey observation models

Convert the forward-modelled field into a synthetic `MeasurementSet`
that has the **same structure** as a data `MeasurementSet`.

```python
class ObservationModel(ABC):
    @abstractmethod
    def apply(self, field: Field, ms_template: MeasurementSet) -> MeasurementSet:
        """Take a forward-modelled field, return a synthetic Measurement-
        Set with the same window / region map / n(z) as ms_template."""
```

One observation model per survey class:
- `GalaxyPositionObsModel` — bias + RSD + selection.
- `PeculiarVelocityObsModel` — linear v + intrinsic scatter + Malmquist.
- `LyaForestObsModel` — FGPA or hydro emulator on sightlines.
- `WeakLensingObsModel` — projection + shape noise + IA.
- `CMBLensingObsModel` — projection over lightcone + noise.
- `ClusterObsModel` — halo finder + selection function.
- `SNIaObsModel` — host + intrinsic + PV + extinction.

This list is exactly the per-survey loaders in Pillar 1, mirrored
on the simulation side. The symmetry is the design principle.

### 5.4 Likelihood comparison

Given a data `MeasurementSet` and a synthetic `MeasurementSet` from
the forward + obs model, compute log L. Options:

- **Field-level** log L (compare the full observed field against the
  forward-modelled prediction at each pixel / cell). Highest
  information; costliest. BORG-style.
- **Summary statistics** (P(k), ξ(r), C_ℓ, n(z)) — cheaper, lossy.
  Useful as a sanity-check sidecar.
- **Simulation-based inference** (compress to learned summaries,
  use neural ratio / posterior estimators) — middle ground.

Pillar 3 should support all three with a common interface; the
field-level path is the target.

### 5.5 Mini-simulation zoom-in

Once the IC posterior is converged at large scales:
1. Pick a region of interest (cluster, void, filament, observed
   transient host).
2. Re-simulate at higher resolution with full hydro / RT physics.
3. Compare against the observed properties of the region (X-ray
   temperature, SZ signal, galaxy stellar masses, ...).
4. Iterate astrophysics nuisance parameters.

This is where the "twin" delivers science specific to one system,
not just population statistics.

### 5.6 Incremental updates

When a new survey DR lands:
1. Ingest via Pillar 1 → new `MeasurementSet`.
2. Re-run inference *starting from the previous posterior* as
   prior (or warm-start the IC sampler).
3. Update the constrained realisation.

Incrementality is the user-facing feature that distinguishes the
digital twin from per-paper re-analysis. Engineering-wise it requires
posterior serialisation and resume hooks throughout.

## 6. Where cosmology lives

**Only here, alongside Pillar 2.** Pillar 3 owns:
- Fiducial baseline (H₀, Ωₘ, Ω_b, σ₈, n_s, ...) — declared per run.
- Distance-conversion machinery (z → r_comoving → r_proper).
- Cosmological prior on the IC sampling (slow / fast parameter
  split when sampling cosmology jointly with IC).
- Cosmoprimo / pyCCL / CLASS engine choice.

Pillar 1 never sees these. Pillar 2 may or may not, depending on
whether the estimator is theory-aware.

## 7. Roadmap (large strokes)

Pillar 3 is **post-Pillar-1-Phase-21** (need `MeasurementSet`) and
benefits from Pillar 2 Phase A (field-level inference in flip) being
stable.

### Phase α — Synthetic data side of `MeasurementSet`

- Define `oneuniverse.simulation.ObservationModel` ABC.
- Implement smoke-test obs model: Gaussian random field → galaxy
  catalog with known bias, n(z), and window from a Pillar 1
  `MeasurementSet` template.
- End-to-end: synthetic IC → synthetic MS round-trips through a
  Pillar 2 estimator and recovers the input bias / n(z).

### Phase β — L2 forward model (JaxPM / similar)

- Wrap a differentiable PM forward model.
- End-to-end: known IC → forward → obs model → synthetic MS;
  gradient computable through the whole stack.

### Phase γ — Single-tracer field-level inference

- HMC sampling of IC conditioned on a single survey's
  `MeasurementSet` (likely DESI BGS or eBOSS LRG).
- Demonstrate posterior over the IC consistent with the data.
- Validate against published BORG-DESI / similar results.

### Phase δ — Multi-tracer joint inference

- Joint IC posterior conditioned on galaxy + PV + Lyα `Measurement-
  Set`s simultaneously.
- Validate degeneracy-breaking on synthetic data first.
- First real-data demo (likely eBOSS+CF4 or DESI+CF4).

### Phase ε — Mini-simulation zoom-in

- Pick a converged IC realisation; select a 50–100 Mpc region.
- Re-simulate with hydro (Gadget / AREPO / GIZMO).
- Compare against an observed cluster's X-ray / SZ / galaxy
  properties.

### Phase ζ — Incremental updates + posterior serialisation

- Save IC posterior chains in HDF5 / Zarr.
- Resume HMC from previous chain when a new MeasurementSet arrives.
- Demonstrate end-to-end: DR-1 inference, then re-condition on DR-2
  in less compute than a fresh run.

### Phase η — CMB lensing / weak lensing joint

- Add `WeakLensingObsModel` + `CMBLensingObsModel`.
- Joint posterior across spectroscopic + photometric + κ.
- Output: full 3D matter field constrained by every cosmological
  probe.

This is the **digital twin** by any reasonable definition.

## 8. External tools / collaborations

The phases above are too ambitious for one developer in a vacuum.
Pillar 3 should integrate (not reinvent):

- **JaxPM / JAX-cosmo** — differentiable forward model.
- **BORG / borg-public** — IC sampling reference.
- **JaxNetCosmo / pmwd** — alternative differentiable PM.
- **Gadget-4 / AREPO / GIZMO** — N-body / hydro for zoom-ins.
- **Cosmoprimo** — cosmology + distances.
- **Cobaya** — wrapping for joint sampling with nuisance parameters.

Plan a collaboration / dependency assessment **before** starting
Phase β. Pillar 3 is years of work; don't re-implement what's
already production-grade elsewhere.

## 9. Deliverables checklist (Pillar 3 v1)

A reasonable "Pillar 3 v1 done" definition:

- [ ] `oneuniverse.simulation.ObservationModel` ABC + at least three
      concrete obs models (galaxy positions, peculiar velocities,
      Lyα).
- [ ] L2 forward model (JaxPM-or-equivalent) integrated.
- [ ] Single-tracer field-level inference on a real Pillar 1
      MeasurementSet recovers a published result.
- [ ] Multi-tracer joint inference produces a tighter constraint
      than any single probe alone (synthetic + real-data demos).
- [ ] One mini-simulation zoom-in matches an observed cluster's
      bulk properties.
- [ ] Posterior serialisation + incremental resume works end-to-end.
- [ ] Documentation + tutorials.

## 10. Risks + open questions

- **Compute scale.** Field-level inference on Gpc/h volumes at
  Mpc resolution is at the edge of academic-scale compute.
  Triage: smaller boxes first; lean on collaborators with HPC
  access.
- **Differentiability.** L2 forward model must be differentiable
  to make HMC tractable; JaxPM is the obvious choice but limits
  physics. L3 (full N-body) is non-differentiable → SBI or
  ensemble-Kalman alternatives.
- **Observation-model accuracy.** Each survey's selection function
  is non-trivial; getting them wrong silently biases the posterior.
  Heavy validation per survey before joint inference.
- **Cosmology in the loop.** Sampling cosmology jointly with IC is
  ~10× cost. Fix cosmology first; release the prior only when
  single-tracer single-cosmology inference is solid.
- **Reproducibility.** Mini-sims at L3/L4 are not bit-reproducible
  across hardware; need provenance tracking sufficient to compare
  realisations.
- **Naming.** `oneuniverse.simulation` is a candidate. Alternatives:
  separate package `digitaltwin` or `onesim`. Decide before Phase α.

## 11. What this enables (the dream)

- A live, queryable representation of the matter field of our actual
  Universe, updated as new data lands.
- Zoom-in physics on specific structures (the Local Group, the Coma
  cluster, the Boötes void, an observed TDE host) with realistic
  cosmological boundary conditions.
- Forecast tools that don't rely on mock-catalog assumptions:
  "what would survey X measure for cosmology Y, given the actual
  large-scale structure of our Universe?"
- A common posterior to which every new probe can be added.

## 12. References

- [`2026-05-28-pillar1-data-combine-measure.md`](2026-05-28-pillar1-data-combine-measure.md)
  — Pillar 1 roadmap (data side of MeasurementSet).
- [`2026-05-28-pillar2-external-interfaces.md`](2026-05-28-pillar2-external-interfaces.md)
  — Pillar 2 roadmap (estimators and likelihoods).
- [[project_digital_twin_vision]] — original digital-twin framing.
- [`../research/digital_twin_research.md`](../research/digital_twin_research.md)
  — prior literature synthesis on constrained reconstructions.
