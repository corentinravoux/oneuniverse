# oneuniverse — capability notebooks

Each notebook states a **claim** and validates it against either linear theory
or a controlled mock with known ground truth. All are executed (figures + output
embedded). Real eBOSS DR16Q / DESI DR1 data are used where present; otherwise a
synthetic stand-in is generated so the notebooks always run.

Rebuild + re-execute:

```bash
cd notebooks && python3 _build_notebooks.py
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

| # | Notebook | Claim validated |
|---|---|---|
| 01 | **Data ingestion** | Two real surveys → one schema; OUF partial-access prunes a cone to a small fraction of partitions; the same quasar is re-identified across eBOSS↔DESI at sub-arcsecond separation. |
| 02 | **Selection: weights, n(z), randoms** | FKP weights down-weight the cosmic-variance regime; generated randoms reproduce the data n(z) (KS test) and lie inside the angular footprint. |
| 03 | **Measurement recovery** | A field drawn with the Eisenstein–Hu P(k) is recovered to ~1% (cosmic-variance-limited); lognormal tracers recover their input bias via the matter cross-spectrum; the `MeasurementSet` round-trips to disk, cosmology-free. |
| 04 | **Probe gallery** | One container (PointSet / Sightline / FieldMap) produces analysis-ready measurements for clustering, weak lensing, peculiar velocities, supernovae, Lyα, and galaxy×map — and expresses builder-less probes via optional slots. |
| 05 | **Simulation storage** | Sub-volume reads touch ~1 chunk; wrap-in-place is ≈10–15% of a re-encode; the stored field's P(k) is lossless to round-off. |
| 06 | **PM gravity & resimulation** | The particle-mesh solver reproduces linear growth on large scales (r→1, T→1); the TreePM force split lets a resimulated sub-volume match the truth at a much smaller buffer than the naive coupling. |
| 07 | **Twin reconstruction** | A Wiener filter reconstructs the large-scale field from shot-noise-limited tracers; the reconstruction scale k(r=0.5) tightens monotonically with tracer density n̄ — the depth–fidelity trade-off of constrained simulation. |

**Scope honesty.** Notebooks 05–07 demonstrate the storage, gravity, and
reconstruction *machinery* on a linear-theory + fast-PM + Wiener stand-in; the
physics is a controlled toy, not a production N-body / Bayesian-inference
pipeline. The data layer (01–04) uses the real surveys.

Estimators (P(k), ξ, C_ℓ, f σ₈) are external tools that consume the
`MeasurementSet`; the small power-spectrum estimators used here for *validation*
are written inline and are not part of the package's public API.
