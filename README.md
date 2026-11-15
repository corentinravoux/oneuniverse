# oneuniverse

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21653036.svg)](https://doi.org/10.5281/zenodo.21653036)
[![tests](https://github.com/corentinravoux/oneuniverse/actions/workflows/tests.yml/badge.svg)](https://github.com/corentinravoux/oneuniverse/actions/workflows/tests.yml)
[![docs](https://github.com/corentinravoux/oneuniverse/actions/workflows/docs.yml/badge.svg)](https://corentinravoux.github.io/oneuniverse/)

**One on-disk format and one cross-survey identity for cosmology survey
catalogs, and a cosmology-free `MeasurementSet` to hand to your estimator.**

`oneuniverse` is the data and orchestration layer of a three-pillar cosmology
stack. It ships no estimators and no samplers.

- **Pillar 1, data.** `oneuniverse.data` converts raw survey files into OUF 3.0,
  a partitioned parquet format you query lazily and partially (cone, sky patch,
  redshift range). It gives every object a stable identity (`source_row`,
  `object_id`) and cross-matches surveys into a ONEUID index. `oneuniverse.combine`
  adds weights and combines measurements of the same object. None of it needs a
  cosmology: cuts, weights and metadata are stored verbatim, and you choose the
  fiducial cosmology later, in the estimator.
- **Pillar 1 to 2, measure.** `oneuniverse.measure` builds a `MeasurementSet`
  (catalog, randoms, window, n(z), jackknife regions, provenance) that external
  estimators such as flip, pycorr or picca consume.
- **Pillar 3, simulation and twin.** `oneuniverse.simulation` is a partial-access
  store (OUF-Sim 1.0) and a request/lineage catalog for simulation products.
  `oneuniverse.twin` holds the engine contracts that couple data and simulation.
  `oneuniverse.reference` is a small physics engine that plugs into them.

Both formats are also readable as SQL: DuckDB views over the parquet, nothing
copied (extra `[sql]`).

## Example

<!-- example:start -->
```python
import numpy as np
from oneuniverse import data, combine, measure

# Pillar 1: convert raw survey files into OUF datasets, open them lazily
data.list_surveys()                        # {"desi_qso": SurveyMeta, "eboss_qso": SurveyMeta}
data.convert("eboss_qso", source=raw / "eboss_qso", out=db_root / "eboss_qso")
data.convert("desi_qso", source=raw / "desi_qso", out=db_root / "desi_qso")
db   = data.Database(db_root)
view = db["eboss_qso"]                     # DatasetView: partial reads, nothing loaded yet
df   = view.read(cone=data.Cone(185, 15, 10), z=(0.8, 2.2), columns=["ra", "dec", "z"])
idx  = db.build_oneuid(["eboss_qso", "desi_qso"], rules=data.CrossMatchRules(sky_tol_arcsec=1), name="qso")
q    = db.oneuid_query("qso")
both = q.hydrate_table(q.select(cone=data.Cone(185, 15, 10)), columns=["ra", "dec", "z"])

# combine: one weight per dataset, defaults filled in from the survey kind
fkp = combine.FKPWeight(nbar=lambda z: 3e-4 * np.ones_like(z), P0=3e4)
wc  = combine.WeightedCatalog.from_oneuid(idx, db).fill_defaults()
wc.add_weight("desi_qso", fkp)

# measure: a cosmology-free MeasurementSet for an external estimator
ms = measure.galaxy_clustering(view, z=(0.8, 2.2), weight=fkp, nz_edges=np.linspace(0.8, 2.2, 15))
measure.save(ms, out_path);  ms2 = measure.load(out_path)
```
<!-- example:end -->

`raw` and `db_root` are directories and `out_path` a new directory; the rest is
the whole flow. The block is run by `test/test_readme_example.py` on synthetic
survey files.

## What is real and what is a stand-in

- **Real.** Two survey loaders, eBOSS DR16Q (`eboss_qso`) and DESI DR1 QSO
  (`desi_qso`), are tested on synthetic files that mimic both formats. Gated
  real-catalog tests (`test/data/test_real_surveys.py`, enabled by
  `ONEUNIVERSE_DATA_ROOT`) exist and have not yet been run for this release.
  Everything else is tested on synthetic data generated in the test run.
- **Stand-in.** `oneuniverse.reference` is a toy: Eisenstein-Hu linear theory,
  Zel'dovich and particle-mesh evolution, a Wiener filter, lognormal mocks. It
  exists to validate the pipelines and the engine contracts, not to produce
  science. Real engines plug in through `twin.register_engine` and
  `simulation.register` (converters).
- **Not included.** Estimators (power spectra, correlation functions,
  likelihoods) and samplers.

## Install

```bash
git clone https://github.com/corentinravoux/oneuniverse.git
cd oneuniverse
pip install -e ".[dev,sql]"      # tests, DuckDB SQL face
```

Python >= 3.11. Hard dependencies: numpy, pyarrow, pandas, healpy, scipy,
astropy. Extras: `sql`, `mpi`, `dev`, `docs`; `fits` is reserved and unused
in 2.0.

```bash
pytest -p no:cacheprovider -q    # default run, skips the slow ensemble tests
pytest -m slow                   # ensemble physics, memory, real-data and example tests
```

## Documentation

- Guide and API reference: <https://corentinravoux.github.io/oneuniverse/>
  (sources in `docs_site/`; build with `pip install -e ".[docs,sql]"` then
  `mkdocs build --strict`).

## Citation

Cite the software through [`CITATION.cff`](CITATION.cff) (DOI
[10.5281/zenodo.21653036](https://doi.org/10.5281/zenodo.21653036)).

## License

MIT, see [`LICENSE`](LICENSE).
