# File formats

Four formats are written by the package. Each carries a version and every
reader checks it.

## OUF 3.0 (survey datasets)

`format_version` is `"3.0.0"`. One dataset is one directory:

```
<dataset>/manifest.json        identity of the dataset
<dataset>/_index.parquet       one row per partition
<dataset>/data/cell=<n>/part-<i>.parquet
<dataset>/_history/<created_utc>.manifest.json
```

- **`manifest.json`** holds the geometry (`point`, `sightline` or `lightcurve`),
  the survey (name, kind, reference), the loader (name, version, the keywords
  it was run with), the Arrow schema, the partitioning (`nside`, `nest`), the
  original files with their sha256 and row counts, the validity (`version`,
  `valid_from`, `valid_to`), a `dataset_uuid` and free `extensions`.
- **`_index.parquet`** has one row per partition file: its path, row count,
  HEALPix cell, and min/max of `ra`, `dec` and `z`, plus optional statistics of
  extra columns. `DatasetView` prunes partitions with it before opening any
  parquet file.
- **`data/cell=<n>/`** are the partitions. `n` is numbered at the manifest's
  `partitioning.nside`. Rows are sorted by `(hpix32, source_row)`, where
  `hpix32` is the NEST pixel at nside 32 computed from `ra` and `dec`.
- **`_history/`** keeps the previous manifest when a dataset is overwritten;
  `validity.version` increments. The data files of old versions are not kept.

Core columns of a point dataset: `ra`, `dec`, `z`, `z_type`, `z_err`,
`object_id`, `survey`, `source_row`, `hpix32`. `z_type` is a registry
(`data.z_types()`, `data.register_z_type`); a value outside it raises
`SchemaError`. A sightline dataset adds list columns `wavelength`, `delta` and
`weight`; a lightcurve dataset adds `mjd`, `flux`, `flux_err` and `band`.
Optional column groups per survey kind (spectroscopic, photometric, peculiar
velocity, ...) are listed in `oneuniverse.data.schema`.

### Identity

`source_row` is the row number in the original file; loaders keep it through
cuts, and it is the only row identity and the recommended join key. `object_id`
is `(hash32(survey) << 32) | source_row`.

## Versioned sidecars (ONEUID and links)

ONEUID indexes and sub-object links live next to the datasets, under the
database root:

```
<root>/_<kind>/<name>/<version>/table.parquet
                               /meta.json
<root>/_<kind>/<name>/current.json
```

`kind` is `oneuid` or `links`. The version id is the UTC write time to the
microsecond, and `current.json` points to the live one, so every earlier version
stays readable. A publish is built in `<path>.tmp` and renamed. Rows store
`(dataset, source_row)` plus `(partition, offset)` computed at build time, which
makes hydration a `take` per partition. `meta.json` records `format_version`
(currently `1`), the rules, and the version of each dataset the index was built
on. If a dataset changed since, reading raises `FormatError` ("rebuild the
index"). A links set also records the ONEUID version it was computed on.

`Database(root, as_of=<timezone-aware datetime>)` resolves datasets and
sidecars as they were at that time. A database opened `as_of` is read-only. A
`DatasetView` opened before its dataset was overwritten raises `FormatError`
(its `dataset_uuid` no longer matches).

## OUF-Sim 1.0 (simulation stores)

`OUFSIM_FORMAT_VERSION` is `"1.0.0"`. One store is one directory:

```
<store>/manifest.json
<store>/layout.json
<store>/products/<name>/z=<z:.3f>/{_index.parquet, chunk_<id>.parquet | tile_<id>.npy | pix_<id>.parquet}
```

- **`manifest.json`** (`OUFSimManifest`) holds the box size, the cosmology
  (`CosmologySpec`), units (`UnitFrameSpec`), provenance (`ProvenanceSpec`,
  including the code), the declared products (`ProductDecl`, kinds `particles`,
  `halos`, `field`, `lightcone`, `checkpoint`), the native-file description, the
  creation time, and an optional parent and lineage.
- **`layout.json`** lists the committed products. It is the per-product commit
  point: a product directory not listed there is an orphan that a retry
  replaces. The store directory itself is published atomically.
- Catalogs are chunked on an `n_side`^3 periodic grid (`chunk_<id>.parquet`),
  fields are cut in tiles (`tile_<id>.npy`), lightcones in NEST pixels
  (`pix_<id>.parquet`). Each redshift has an `_index.parquet` with the bounding
  box of every chunk.

`SimStore` reads only through a selector: `read_box(product, z, cube)` (periodic:
a cube across the boundary is split), `read_cone`, `read_field_box`; the
whole-box read is the explicit `scan_full`. A `SimDatabase` root holds several
stores plus `_simdb/requests.json` and `_simdb/lineage.parquet`, the
request/lineage catalog.

## Measurement format

`measure.save(ms, path)` writes `MEASURE_FORMAT_VERSION` `"1.0.0"`:

```
path/measurement_set.json
path/regions.npy
path/products/<name>/catalog.parquet [randoms.parquet] [window.npz] [nz.npz]
                     [values.npy mask.npy]               (FieldMap)
                     extras/<key>.npy | <key>.parquet
```

The tree is built in `<path>.tmp` and published by rename, so a failed save
leaves no partial set. Point sets, field maps and sightlines, both kinds of
extras, window systematics, n(z) and provenance all round-trip. `measure.load`
accepts any `1.x` file and rejects another major version.
