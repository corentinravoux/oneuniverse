# Getting started

## Install

```bash
git clone https://github.com/corentinravoux/oneuniverse.git
cd oneuniverse
pip install -e ".[dev,sql]"
```

Python >= 3.11. The hard dependencies are numpy, pyarrow, pandas, healpy, scipy
and astropy. Optional extras: `sql` (DuckDB), `mpi`, `dev`, `docs`; `fits`
(fitsio) is reserved and unused in 2.0.

## Convert a survey

Two loaders are built in: `eboss_qso` (eBOSS DR16Q) and `desi_qso` (DESI DR1
QSO). `convert` reads the raw files found in `source` and writes an OUF 3.0
dataset to `out`.

```python
from oneuniverse import data

data.list_surveys()                    # {"desi_qso": SurveyMeta, "eboss_qso": SurveyMeta}
data.convert("eboss_qso", source=raw / "eboss_qso", out=db_root / "eboss_qso")
data.convert("desi_qso", source=raw / "desi_qso", out=db_root / "desi_qso")
```

`overwrite=True` replaces an existing dataset (the old manifest is kept in
`_history/`). Other keywords: `partition_nside`, `batch_size`, and loader
keywords. To add a survey, subclass `SurveyLoader` (or `FitsCatalogLoader`) and
pass it to `data.register`, or ship it as an entry point in the group
`oneuniverse.survey_loaders`.

## Read part of a catalog

A `Database` is a root directory of datasets. Indexing it gives a
`DatasetView`: it opens the manifest and the partition index, and reads parquet
only for the partitions a selection touches.

```python
db   = data.Database(db_root)
view = db["eboss_qso"]
df   = view.read(cone=data.Cone(185, 15, 10), z=(0.8, 2.2), columns=["ra", "dec", "z"])
```

`read` returns a `pandas.DataFrame`; `table` returns a `pyarrow.Table`; `scan`
yields record batches for catalogs larger than memory. The selectors are `cone`,
`patch` (`SkyPatch`), `z`, `t` and `ranges`, and `filter` takes an Arrow
expression. Every point dataset has the core columns `ra`, `dec`, `z`, `z_type`, `z_err`,
`object_id`, `survey`, `source_row`, `hpix32`.

## Cross-match surveys

```python
idx = db.build_oneuid(["eboss_qso", "desi_qso"],
                      rules=data.CrossMatchRules(sky_tol_arcsec=1), name="qso")
q    = db.oneuid_query("qso")
both = q.hydrate_table(q.select(cone=data.Cone(185, 15, 10)), columns=["ra", "dec", "z"])
```

The index stores `(dataset, source_row)` plus the partition and offset of every
member, so `hydrate` is a take per partition, never a scan. It records the
version of each dataset; if a dataset was overwritten, hydration raises
`FormatError` and the index must be rebuilt. `data.SubobjectLinks` links
parent and child objects (host galaxy and supernova, cluster and member).

## Weight and combine

```python
import numpy as np
from oneuniverse import combine

fkp = combine.FKPWeight(nbar=lambda z: 3e-4 * np.ones_like(z), P0=3e4)
wc  = combine.WeightedCatalog.from_oneuid(idx, db).fill_defaults()
wc.add_weight("desi_qso", fkp)
```

`fill_defaults` gives each dataset the default weight for its survey kind and
`z_type` (`combine.register_default` adds more). Weights compose with `*`.
`WeightedCatalog.combine` merges the measurements of one object with a strategy
(`ivar`, `best`, `mean`, `priority`).

## Build a MeasurementSet

```python
from oneuniverse import measure

ms = measure.galaxy_clustering(view, z=(0.8, 2.2), weight=fkp,
                               nz_edges=np.linspace(0.8, 2.2, 15))
measure.save(ms, out_path)
ms = measure.load(out_path)
```

`galaxy_clustering` runs the full spine (`build_pointset`): selection, cleaning,
weights, window, jackknife regions, n(z), randoms and provenance. Without a
`mask` the window is the footprint of the data, a fallback only: pass a survey
mask for science. `cosmic_shear` and `three_by_two` are the other presets;
`build_sightlines` handles Lyman-alpha sightlines and `attach_photoz` attaches
photo-z PDFs. A `MeasurementSet` never contains a cosmology: it rejects
cosmology-derived column names.

## Simulations and the twin

`oneuniverse.simulation` stores simulation products in OUF-Sim and reads them
only through a selector (`SimStore.read_box` with a `Cube`, `read_cone` with a
`Cone`); `scan_full()` is the explicit whole-box read. `oneuniverse.twin` defines the `ForwardEngine` and
`ReconstructionEngine` contracts, `observe_from_view` (data to twin) and
`run_engine_to_store` (engine to store). `oneuniverse.reference` provides toy
engines for both; import `oneuniverse.reference.engines` to register them.
See [Architecture](architecture.md) and [File formats](formats.md).
