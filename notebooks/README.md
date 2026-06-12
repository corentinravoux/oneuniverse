# oneuniverse — the visual tour

Six executed notebooks presenting **everything the package offers**, with the
real eBOSS DR16Q / DESI DR1 quasar catalogs wherever they are on disk (and a
synthetic stand-in otherwise, so the notebooks always run). Each notebook is
visual-first — sky maps, schema diagrams, pipeline drawings, byte-range maps —
and every scientific statement is validated against linear theory or a
controlled mock with known ground truth.

Rebuild + re-execute:

```bash
cd notebooks && python3 _build_notebooks.py
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

| # | Notebook | What you see |
|---|---|---|
| 01 | **One universe of data** | Real eBOSS+DESI on a Mollweide sky; the same quasars re-identified across surveys (sub-arcsecond peak); the anatomy of **OUF 2.6** (identity-only manifest + partition-index sidecar, painted on the sky); selector-driven partial reads benchmarked. |
| 02 | **The SQL face** | The exported schema drawn; real eBOSS → `catalog.sqlite` queried in pure SQL (GROUP-BY n(z), HEALPix cone as `WHERE IN`); ONEUID as a JOIN; the simulation chunk index as a relational bbox query (chunk map with the hit set highlighted); zero-copy DuckDB DDL. |
| 03 | **Measurements for every probe** | The nine-step pipeline drawn; real-eBOSS clustering `MeasurementSet` with KS-verified randoms; the weak-lensing **photo-z kernel waterfall** → tomographic n(z); a four-probe gallery (PV, SN, Lyα, map×catalog); the generality slots (lens-system time-delay links, named weights, covariance plans); save→reload→SQL round-trip. |
| 04 | **Simulation storage** | The chunk index and a box read drawn together (hit chunks highlighted); **wrap-in-place as a byte-range map** (~13% of a copy); bit-lossless P(k); bounded-memory scaling. |
| 05 | **Gravity & resimulation** | Structure *growing* across four PM snapshots, with σ(δ) landing on the exact linear D(a) curve; r(k)/T(k)/stochasticity panels; the **TreePM force split drawn in k-space**; buffer-convergence + per-buffer coupling gains. |
| 06 | **The constrained twin** | The Wiener gain G(k) family vs survey depth; truth → tracers → Wiener mean → **constrained realization** (Hoffman–Ribak) side by side; the depth → reconstructable-scale law k½(n̄). |

**Scope honesty.** Notebooks 04–06 demonstrate the storage, gravity and
reconstruction *machinery* on a linear-theory + fast-PM + Wiener stand-in; the
physics is a controlled toy, not a production N-body / Bayesian pipeline. The
data and SQL layers (01–03) use the real surveys. Estimators (P(k), ξ, C_ℓ)
are external tools that consume the `MeasurementSet`; the inline P(k) code here
exists only to *validate* the package against theory.
