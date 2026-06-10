# oneuniverse

**A toolkit for turning messy, heterogeneous cosmology survey catalogs into one
clean, queryable, analysis-ready dataset — and for storing and resimulating the
cosmic web that produced them.**

If you work with survey data (galaxies, quasars, peculiar velocities,
supernovae, weak-lensing shapes, Lyman-α forests, CMB/HI maps) you spend a lot
of time on the same plumbing: reading FITS files into a common shape, matching
the same object across surveys, applying the right weights, generating randoms,
building n(z), and packaging it all for an estimator. `oneuniverse` does that
plumbing once, consistently, and hands you a single object — a
**`MeasurementSet`** — that any downstream estimator can consume.

It is deliberately **cosmology-free** on the data side: cuts, weights, frames,
and metadata are stored verbatim, but H₀ / Ωₘ / a distance model never touch
your catalog. You choose the cosmology later, at the estimator call. That makes
the same prepared data reusable under any fiducial.

Pure Python (numpy / pandas / pyarrow / healpy). 781 tests.

```bash
pip install -e ".[dev]"
pytest -q            # ~5 min; tests needing real survey files auto-skip
```

---

## What can I do with it?

| I want to… | Use | Returns |
|---|---|---|
| Load a survey catalog into a standard table | `load_catalog("eboss_qso")` | a `pandas.DataFrame` |
| Read a *huge* catalog without loading it all | `DatasetView(...).read(cone=…, z_range=…)` | only the rows you asked for |
| Match the same object across surveys | `database.build_oneuid(...)` | a cross-match index (ONEUID) |
| Weight a catalog (FKP, completeness, systematics, shear, PIP) | `oneuniverse.combine.weights` | per-object weights |
| Carry a photo-z PDF per object | `DatasetView.load_pdf()` | a `qp`-style kernel |
| **Build a ready-to-estimate measurement** | `oneuniverse.measure.build_*` | a **`MeasurementSet`** |
| Store a simulation so you can query sub-volumes | `write_oufsim_store(...)` / `SimStore` | partial-access sim store |
| Run a quick particle-mesh sim / resimulate a region | `oneuniverse.simulation` | density / particle fields |
| Constrain a simulation to look like your data | `oneuniverse.twin` | reconstructed field |
| **Get it all as a SQL database** | `data.sql.export_sql` / `oufsim.sql.export_sim_sql` / `ms.to_sql()` | a portable SQLite file |

The rest of this README walks through each, from a user's point of view.

### SQL export

Both on-disk formats export to standard SQL — SQLite (stdlib, single portable
file) or zero-copy DuckDB views over the existing parquet:

```python
from oneuniverse.data.sql import export_sql, attach_sql_ddl
export_sql([survey_path, ...], "catalog.sqlite")     # datasets, partitions,
                                                      # objects, PDFs (BLOBs),
                                                      # ONEUID, sub-object links
print(attach_sql_ddl([survey_path]))                  # DuckDB views, no copy

from oneuniverse.simulation.oufsim.sql import export_sim_sql
export_sim_sql(store, "sim.sqlite")                   # sims, chunk index,
                                                      # halos/lightcone/tree
ms.to_sql("measurement.sqlite")                       # a MeasurementSet
```

Bulk simulation products stay index-only in SQL (the `sim_chunks` table answers
*which file holds box X* — the same pruning `SimStore` does, in pure SQL);
catalog-sized products materialise fully. Design + DDL:
`research/2026-06-10-structural-review-and-sql-design.md` §5.

---

## 1 · Load and standardise a survey catalog

Every survey ships its own FITS layout, column names, redshift conventions, and
masks. `load_catalog` reads one and returns a **standard table** with canonical
columns (`ra, dec, z, z_type, z_err, galaxy_id, survey_id, …`):

```python
from oneuniverse.data import load_catalog
df = load_catalog("eboss_qso")                 # standardised DataFrame
df = load_catalog("desi_qso", columns=["ra", "dec", "z"])
```

**Available right now:** `eboss_qso` (eBOSS DR16Q), `desi_qso` (DESI DR1), and
`dummy` (a synthetic catalog for trying things out). Seven more survey loaders
(`des_dr2`, `desi_bgs`, `sdss_mgs`, `sixdfgs`, `pantheonplus`, `desi_pv`,
`cosmicflows4`) are **scaffolds** — registered but not yet implemented; calling
them raises `NotImplementedError`. See `REVIEW.md` (H1).

To persist a catalog in the package's on-disk format (**OUF 2.5** — a
`manifest.json` + HEALPix-partitioned parquet), use `convert_survey(...)` or the
lower-level `write_ouf_dataset(...)`. The format supports point catalogs,
sightlines (Lyα δ), HEALPix/GW sky-maps, data cubes, and light curves.

## 2 · Read a huge catalog efficiently (partial access)

You rarely need the whole sky. `DatasetView` is a lazy reader over an OUF
dataset on disk: it prunes HEALPix partitions so a sky-cone or redshift slice
touches only the relevant files.

```python
from oneuniverse.data import DatasetView
from oneuniverse.data.selection import Cone

view = DatasetView.from_path(survey_path)
patch = view.read(cone=Cone(ra=185, dec=15, radius=10), z_range=(0.8, 2.2))
# a 10° cone over a 40k-object catalog returns ~1k rows, reading 1 partition.
```

## 3 · Cross-match and link surveys

The same quasar may sit in eBOSS *and* DESI. **ONEUID** assigns one stable
identity across surveys (bitemporally — old versions are archived, not
overwritten), and **sub-object links** record hierarchies (cluster→members,
QSO→DLAs, lens→images, GW→host):

```python
from oneuniverse.data import database
database.build_oneuid(datasets, rules, name="eboss_x_desi")
database.build_subobject_links(rules, parents, children, name="qso_dla")
```

## 4 · Weight a catalog

Clustering and lensing need weights. The weight primitives are composable and
keyed to the columns your survey actually provides:

```python
from oneuniverse.combine.weights import FKPWeight, ColumnWeight
w_fkp  = FKPWeight(nbar=lambda z: nbar_of_z(z), P0=1e4)
w_total = w_fkp(df) * ColumnWeight("weight_systot")(df)
```

Available: FKP, completeness, redshift-failure, imaging-systematics, **shear**
(metacal/lensfit responses), and **PIP/bitwise** (DESI fiber-collision
realisations). `WeightedCatalog` combines weights across cross-matched surveys.

## 5 · Photo-z PDFs

Photometric surveys give a *distribution* p(z) per object, not a point. OUF
stores it; `DatasetView.load_pdf()` reconstructs a `qp`-style kernel you can
sample or stack into a tomographic n(z):

```python
pz = view.load_pdf()
pz.mean(), pz.std(), pz.sample(n_per=10)
```

## 6 · Build a ready-to-estimate measurement — the `MeasurementSet`

This is the centrepiece. A **`MeasurementSet`** bundles everything an estimator
needs and nothing it shouldn't: the (weighted, cleaned) catalog, matched
randoms, the window/footprint, n(z), a shared jackknife region map, and full
provenance — **with no cosmology baked in**. One function call does the whole
preparation pipeline:

```python
from oneuniverse.measure import build_galaxy_clustering
ms = build_galaxy_clustering(
        view, tracer="qso", z_range=(0.8, 2.2),
        weights=[FKPWeight(nbar=..., P0=1e4)],
        nz_edges=np.linspace(0.7, 2.3, 33),
        randoms="generate")          # or randoms=<a randoms DatasetView> to ingest
ms.summary()                          # JSON-safe description: products + atoms, no cosmology
ms.check_invariants()                 # asserts the shared-region + cosmology-free contract
```

There is one builder per probe, all producing the same `MeasurementSet` shape:

| Builder | Probe | Carries |
|---|---|---|
| `build_galaxy_clustering` | 3D clustering / RSD / BAO | catalog + randoms + n(z) + window + weights |
| `build_cosmic_shear`, `build_3x2pt` | weak lensing | shapes + calibration + **photo-z kernel** + tomographic n(z) |
| `build_peculiar_velocity`, `build_sn_hubble` | PV, supernovae | distances (μ, v_pec, σ_v) + covariance handle |
| `build_lya` | Lyman-α forest | per-sightline δ_F(λ) + mask + continuum |
| `build_map_cross` | galaxy × CMBκ / tSZ / HI | a HEALPix field + mask, paired with a catalog |

Under the hood there is **one container** (a *DataProduct* with three geometry
flavours: point set, sightline, field map) general enough to also express probes
that don't have a one-call builder yet — galaxy clusters, strong-lens time
delays, redshift-less radio sources, gravitational-wave sirens, line-intensity
maps — through optional slots (sub-object links, named weight families,
covariance plans, beam/interloper metadata, …). A coverage test exercises 12
such cases.

**Important boundary:** the measure layer *prepares and validates* the
measurement; it does **not** compute P(k), ξ(r), or C_ℓ. Those are computed by
external estimators (`flip`, `pycorr`, `picca`, …); the converters from a
`MeasurementSet` to each tool live in a **separate package**.

**Tried on real data:** `build_galaxy_clustering` runs end-to-end on the real
eBOSS DR16Q and DESI DR1 quasar catalogs — genuine survey footprint, real n(z),
randoms that match the mask (`test/test_measure_real_desi_eboss.py`).

## 7 · Store and query a simulation (OUF-Sim)

A simulation snapshot can be terabytes; you usually want one sub-volume.
**OUF-Sim** is a storage layer that indexes a sim so you can read a box, cone,
or field tile without touching the rest — and it can **wrap the native files in
place** (storing only an index, ≈14% of a re-encode) instead of copying them.

```python
from oneuniverse.simulation.oufsim import write_oufsim_store, SimStore
store = write_oufsim_store(native_dir, out, sim_name="run",
                           field_projection="reference")     # wrap, don't copy
SimStore(store).read_box("snapshots", z=0.0, cube=cube)      # partial access
```

It is **multi-backend** (add a new sim format via a small adapter + converter),
with MPI/GPU read hooks and memory-budgeted streaming.

## 8 · Run a fast simulation / resimulate a region

A built-in particle-mesh (PM) mini-simulation lets you generate fields and test
the machinery without an external N-body code. **Resimulation** re-runs just a
sub-volume at higher effective resolution, coupling it to the large-scale field
with a TreePM force split (which reaches a target accuracy at a ~4× smaller
buffer than the naive approach). Field-validation estimators — cross-correlation
r(k), transfer T(k), stochasticity — quantify the agreement.

```python
from oneuniverse.simulation.pm.run import run_pm
from oneuniverse.simulation.validation import validate_field
```

## 9 · Constrain a simulation to your data (the twin)

The `twin` layer couples observations to simulations: turn a field into mock
biased tracers, reconstruct the underlying field (Wiener), and verify the
recovery against the truth with r(k). It is the minimal version of constrained
forward modelling.

```python
from oneuniverse.twin.mock_observe import mock_tracer_field
from oneuniverse.twin.verify import cross_correlation
```

---

## Honest status

- **Survey loaders:** 3 work (eBOSS, DESI, dummy); 7 are scaffolds.
- **Measure layer:** prepares & validates `MeasurementSet`s for all probes;
  does not compute estimators (that's downstream). On-disk save/load for a
  `MeasurementSet` is not yet implemented (REVIEW.md H2).
- **Simulation & twin:** the storage, I/O, and orchestration are real and
  tested; the *physics* (linear field + fast-PM + Wiener) is a **toy stand-in**
  for real N-body / Bayesian inference.
- **Tests:** 781 passing, 0 TODO/FIXME, with import + cosmology-free guards.

A frank external review of every known limitation lives in
**[REVIEW.md](REVIEW.md)**. Design notes and roadmaps are in
**[plans/README.md](plans/README.md)**; runnable demonstrations of everything
above are in **[notebooks/](notebooks/)** (executed, with figures).

## Where things live

```
oneuniverse/
  data/         load, standardise, OUF format, partial-access reader, cross-match, loaders
  combine/      weights + weighted/combined catalogs
  measure/      DataProduct + MeasurementSet + one builder per probe
  simulation/   OUF-Sim store, fast-PM, resimulation
  twin/         data ↔ simulation coupling
notebooks/  scripts/  test/  plans/  research/  docs/
```

Author: Corentin Ravoux. Companion to the `flip` estimator (arXiv:2501.16852).
