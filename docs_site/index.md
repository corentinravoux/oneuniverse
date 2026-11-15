# oneuniverse

`oneuniverse` is the data and orchestration layer of a three-pillar cosmology
stack. It turns heterogeneous survey catalogs into one lazily queried on-disk
format with one cross-survey identity, builds a cosmology-free `MeasurementSet`
for external estimators, and stores simulation products for partial access. It
ships no estimators and no samplers.

| Subpackage | Role |
|---|---|
| [`data`](api/data.md) | OUF 3.0 format, survey loaders, `DatasetView`, `Database`, ONEUID, sub-object links, SQL face |
| [`combine`](api/combine.md) | `Weight` primitives and `WeightedCatalog` |
| [`measure`](api/measure.md) | `MeasurementSet`, `build_pointset`, presets, photo-z attachment |
| [`simulation`](api/simulation.md) | OUF-Sim 1.0 store, `SimDatabase`, request/lineage catalog, converters |
| [`twin`](api/twin.md) | Forward and reconstruction engine contracts, data-to-twin bridge |
| [`reference`](api/reference.md) | Toy reference physics that plugs into `simulation` and `twin` |
| [`_core`](api/core.md) | Errors, registry, geometry, HEALPix helpers shared by all |

## Where to start

- [Getting started](getting-started.md): install, convert a survey, query it,
  cross-match, weight, build a `MeasurementSet`.
- [Architecture](architecture.md): the pillars, the dependency rules, identity.
- [File formats](formats.md): OUF 3.0, OUF-Sim 1.0, the measurement format and
  the versioned sidecars.
- [SQL face](sql.md): DuckDB views over both formats.
- [API reference](api/index.md).

## Real and stand-in

Two loaders, eBOSS DR16Q and DESI DR1 QSO, are tested on synthetic files that
mimic both formats; gated real-catalog tests (`ONEUNIVERSE_DATA_ROOT`) exist and
have not yet been run for this release. The reference
engine in `oneuniverse.reference` is a toy for validating pipelines and engine
contracts. Everything else is tested on synthetic data.

## Citation

Cite the software through `CITATION.cff` in the repository (DOI
[10.5281/zenodo.21653036](https://doi.org/10.5281/zenodo.21653036)). The
archived pre-rewrite plans and review are in `docs/history/` in the repository.
