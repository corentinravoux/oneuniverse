# Database Technologies for the `oneuniverse` Cosmology Catalog

**Date:** 2026-04-01
**Purpose:** Research report on Python-native database and spatial indexing technologies for the `oneuniverse` package, which must store and unify heterogeneous galaxy survey catalogs with efficient spatial queries, cross-matching, and weighted statistical access.

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [Storage Format Comparison](#2-storage-format-comparison)
3. [Spatial Indexing Strategies](#3-spatial-indexing-strategies)
4. [Heterogeneous Schema Handling](#4-heterogeneous-schema-handling)
5. [Weights, Masks, and Flags](#5-weights-masks-and-flags)
6. [Quantitative Comparison Summary](#6-quantitative-comparison-summary)
7. [Recommendation](#7-recommendation)
8. [Key Packages and Installation](#8-key-packages-and-installation)
9. [Concrete Code Sketch](#9-concrete-code-sketch)

---

## 1. Problem Framing

The dominant access patterns for cosmology catalogs are:

| Access pattern | Characteristic |
|---|---|
| Cone search (spatial query) | Filter rows by angular distance on sphere |
| Redshift slice | Range filter on a single column |
| Cross-catalog join | Match objects between two catalogs by position |
| Bulk statistical read | Read 1–N columns for all/most objects |
| Per-object lookup | Retrieve all columns for one known object |
| Weighted catalog arithmetic | Apply masks, weights, compute means/variances |

For cosmology survey catalogs the dominant patterns are **cone search, cross-match, and bulk reads**. This profile strongly favors **columnar storage with a spatial index** and is hostile to row-oriented storage.

---

## 2. Storage Format Comparison

### 2.1 HDF5 (h5py / astropy.table)

**Architecture.** Hierarchical binary format. Columnar if columns stored as separate 1-D datasets under a group (rather than compound row records).

**Strengths:**
- Hierarchical structure maps naturally to multi-survey databases: `/sdss/main/`, `/desi/edr/`.
- Metadata and provenance as HDF5 attributes on groups.
- `astropy.table.Table.read('catalog.hdf5', path='sdss/main')` reads units and descriptions automatically.
- Partial I/O via dataset slicing: `f['sdss/ra'][mask]`.
- Chunked storage + compression (GZIP, LZ4, Blosc via `hdf5plugin`).
- SWMR (Single Writer Multiple Reader) for concurrent access.

**Weaknesses:**
- No built-in query engine. Cone search requires loading a HEALPix pixel column and doing a boolean mask, then fancy-indexed read — **very slow for large sparse masks**.
- Fancy indexing on HDF5 is fast for contiguous slices but slow for scattered indices.
- Schema evolution is awkward (adding a column requires rewriting or a separate dataset).
- No native missing value support.

**Verdict.** Excellent for bulk reads and hierarchical organization. Poor for cone searches without careful chunking. Best used for auxiliary dense array products (N-body snapshots, power spectra, simulation cubes).

---

### 2.2 Parquet (PyArrow / pandas)

**Architecture.** Apache Parquet is a columnar format with row groups. Each row group stores every column separately with per-column statistics (min/max) and optional dictionary encoding, Bloom filters, and RLE compression.

**Strengths:**
- True columnar: reading only `ra, dec, z` from a 50-column catalog touches only 3/50 of the bytes.
- **Predicate pushdown:** row group statistics allow skipping entire row groups for redshift slices without reading data.
- Excellent compression (Zstandard or Snappy): 5–15 GB for a typical 100M-row, 50-column catalog.
- DuckDB, Spark, Polars, PyArrow all read Parquet natively — cross-catalog joins with no data import.
- **Partition-by-column:** partition by `survey=sdss/healpix_order8=12345/` for spatial + survey-level skipping.
- Heterogeneous schemas handled via `read_parquet(..., union_by_name=True)`.

**Weaknesses:**
- No built-in spatial index; you engineer the spatial structure via HEALPix partitioning.
- Appending rows to an existing file is not supported (create new files).
- Provenance metadata stored in file-level key-value map or a sidecar JSON/YAML.

**Verdict.** Best general-purpose format for this use case. Primary recommendation for catalog storage.

---

### 2.3 Zarr

**Architecture.** Stores N-dimensional arrays as directories of compressed chunk files (or cloud key-value objects).

**Strengths:**
- Excellent for cloud-native parallel access: chunks map to S3/GCS objects.
- Perfect for dense array data: simulation cubes, power spectrum grids, imaging.
- `dask.array` integration for lazy computation.

**Weaknesses for this use case:**
- No query engine, no predicate pushdown, no spatial index.
- A catalog (table) must be stored as a group of 1-D arrays — same random-access problems as HDF5.
- The Python ecosystem for tabular catalog work is thin compared to Parquet.

**Verdict.** Not recommended as primary catalog format. Use as complement for auxiliary dense array products.

---

### 2.4 SQLite vs DuckDB

#### SQLite

- **Row-oriented B-tree database.** Reading only `ra, dec, z` from a 50-column table at 100M rows deserializes all 50 columns for every row scanned.
- `spatialite` extension has an R-tree for 2-D Euclidean geometry; not ideal for sphere.
- **Not designed for 100M–1B row analytical queries.** Avoid.

#### DuckDB

**Architecture.** Columnar, vectorized, OLAP-oriented embedded SQL engine. Can query Parquet, CSV, JSON directly without importing.

**Strengths:**
- Query Parquet files directly: `SELECT * FROM read_parquet('catalogs/**/*.parquet')`.
- Vectorized execution over columnar data: 10–100× faster than SQLite for analytical aggregations.
- `ASOF JOIN`, window functions, lateral joins available.
- DuckDB spatial extension: `ST_Distance_Spheroid` and Haversine for great-circle distance.
- Results as PyArrow tables, pandas DataFrames, or NumPy arrays.
- DuckDB v1.x supports Parquet row group filtering, Bloom filters, parallel multi-file reads.
- Cross-catalog joins: load both as views over Parquet, hash join on HEALPix integer key.

```python
import duckdb
con = duckdb.connect()
con.execute("LOAD spatial;")
result = con.execute("""
    SELECT ra, dec, z, r_mag, stellar_mass
    FROM read_parquet('catalogs/survey=sdss/**/*.parquet', union_by_name=True)
    WHERE z BETWEEN 0.1 AND 0.5
      AND hpx_order8 IN (SELECT unnest(?))
""", [list(cone_pixels)]).fetchdf()
```

**Verdict.** Best query engine for ad-hoc analysis, redshift slices, and cross-catalog joins. Use alongside Parquet (DuckDB is the query layer; Parquet is the storage layer).

---

### 2.5 FITS (fitsio / astropy.io.fits)

**Strengths:**
- Universal: every astronomical software and archive speaks FITS.
- `fitsio` supports column-by-column reads (skips non-requested columns at C level).
- Rich header keywords for metadata and provenance.

**Weaknesses:**
- Row-oriented on disk: 3–10× slower than Parquet for multi-column reads at 100M rows.
- No query engine, no joins, no predicate pushdown.
- Compression requires `.fits.gz` which forces sequential decompression.

**Verdict.** Use FITS for interoperability and data release. **Convert incoming FITS to Parquet on ingest** for internal analysis.

---

### 2.6 AstroPy-Native (Table, QTable, SkyCoord)

These are **in-memory** structures, not storage formats.

- **`QTable`:** Associates `astropy.units.Quantity` with columns; arithmetic tracks units. Essential when mixing photometry across filters or velocities in different units.
- **`SkyCoord.match_to_catalog_sky(other)`:** BallTree-based cross-matching. For N,M ~ 10^6, takes 10–30 s. Returns closest match with angular separation.
- **`SkyCoord.search_around_sky(other, seplimit)`:** Many-to-many match within separation limit.

```python
from astropy.coordinates import SkyCoord
import astropy.units as u

sdss = SkyCoord(ra=sdss_ra*u.deg, dec=sdss_dec*u.deg, frame='icrs')
desi = SkyCoord(ra=desi_ra*u.deg, dec=desi_dec*u.deg, frame='icrs')
idx, sep, _ = sdss.match_to_catalog_sky(desi)
matched = sep < 1*u.arcsec
```

**Verdict.** Use `QTable` as the in-memory working structure after loading from disk. Use `SkyCoord.match_to_catalog_sky` for cross-matching up to ~10^7 objects.

---

## 3. Spatial Indexing Strategies

### 3.1 HEALPix Pixelization (Recommended)

HEALPix divides the sphere into 12 × 4^k **equal-area** pixels at order k. At order 8 (nside=256) each pixel is ~0.21 deg²; at order 12 (nside=4096) ~0.82 arcmin².

**Usage as spatial index:**
1. At ingest: `healpy.ang2pix(nside, ra_rad, dec_rad, lonlat=True)` stored as integer column or partition key.
2. Cone search: `healpy.query_disc(nside, vec, r_rad, inclusive=True)` → set of overlapping pixels.
3. Filter catalog to rows in those pixels (fast integer membership test).
4. Refine with exact great-circle distance.

**Multi-resolution (two-level index):**
- Order 5 (nside=32, ~1.8 deg²): Parquet file partitioning. A 10-deg cone touches ~30 files.
- Order 12 (nside=4096, ~0.82 arcmin²): per-row column for fine filtering within a partition.

**Always use NESTED ordering** for spatial indexing — nearby pixels on the sphere have nearby integer IDs, enabling range queries.

```python
import healpy as hp
import numpy as np

vec = hp.ang2vec(ra0, dec0, lonlat=True)
target_pixels = hp.query_disc(4096, vec, np.radians(1/60), inclusive=True, nest=True)
```

**HEALPix is the de-facto standard** for all modern large surveys (SDSS, DES, DESI, Euclid, LSST). Equal-area pixels are crucial for cosmology: number counts per pixel are directly comparable across the sky.

---

### 3.2 k-d Trees (scipy.spatial.cKDTree)

Project (RA, Dec) → 3-D unit vectors (x, y, z), build k-d tree in 3-D.

- Very fast for in-memory cross-matching: O(N log N) build, O(log N) query.
- Must be built in RAM: ~10–12 GB for 100M objects. Not persistent.
- Euclidean approximation breaks for cones > 10 deg.

```python
from scipy.spatial import cKDTree
xyz = np.column_stack([np.cos(dec_r)*np.cos(ra_r),
                       np.cos(dec_r)*np.sin(ra_r),
                       np.sin(dec_r)])
tree = cKDTree(xyz)
r_chord = 2 * np.sin(np.radians(1/60) / 2)  # 1 arcmin chord distance
idx = tree.query_ball_point(xyz_query, r=r_chord)
```

---

### 3.3 BallTree (sklearn / astropy internals)

Supports the `haversine` metric directly on (Dec, RA) pairs in radians — **exact great-circle distances, no projection error**. This is what `astropy.coordinates.match_to_catalog_sky` uses internally.

```python
from sklearn.neighbors import BallTree
import numpy as np

X = np.radians(np.column_stack([dec, ra]))  # (N, 2) in radians
tree = BallTree(X, metric='haversine')
r_query = np.radians(1/3600)  # 1 arcsec
indices = tree.query_radius(X_query, r=r_query)
```

**Recommendation:** Prefer BallTree via `SkyCoord.match_to_catalog_sky` for correctness and convenience.

---

### 3.4 HTM (Hierarchical Triangular Mesh)

Recursively subdivides the sphere into spherical triangles. Not equal-area (vary ~2×), less active Python ecosystem than HEALPix, largely legacy (SDSS-era `esutil`/`smatch`).

**Verdict.** Superseded by HEALPix for new projects. Avoid unless interfacing with legacy code.

---

### 3.5 MOC (Multi-Order Coverage) and mocpy

MOC describes arbitrary sky regions as unions of HEALPix pixels at multiple orders. `mocpy` (backed by Rust since v0.12) implements MOC operations in Python.

**Use case:** Survey footprints are naturally MOCs. Intersection of two footprints is one line:

```python
from mocpy import MOC
import astropy.units as u

sdss_moc = MOC.from_fits('sdss_footprint.moc.fits')
desi_moc = MOC.from_fits('desi_footprint.moc.fits')
overlap   = sdss_moc.intersection(desi_moc)

in_overlap = overlap.contains_lonlat(table['ra']*u.deg, table['dec']*u.deg)
```

**Recommendation.** Use `mocpy` for all footprint operations. Store survey footprints as MOC FITS files alongside catalogs.

---

### 3.6 LSDB / HiPSCat (Recommended Spatial Layer)

**LSDB** (LINCC Frameworks, designed for LSST-scale catalogs) is built on Dask + a HEALPix-partitioned Parquet layout called **HiPSCat**. Each Parquet partition corresponds to one HEALPix pixel at a chosen order, with objects sorted by fine-grained HEALPix pixel within each file.

```python
import lsdb

sdss = lsdb.read_hipscat('catalogs/sdss/')
desi = lsdb.read_hipscat('catalogs/desi/')

# Cone search — lazy, Dask-backed
cone = sdss.cone_search(ra=185.0, dec=15.0, radius_arcsec=600)

# Cross-match — exploits shared HEALPix partitioning → O(N log N)
matched = sdss.crossmatch(desi, n_neighbors=1, radius_arcsec=1.0)

result = matched.compute()  # pandas DataFrame
```

- HiPSCat catalogs are just Parquet files in a specific directory layout + JSON metadata — readable by any Parquet reader.
- Designed for Rubin LSST (10^10 objects) but works well for hundreds of millions.

**LSDB + HiPSCat is the single most compelling specialized tool** for this use case.

---

## 4. Heterogeneous Schema Handling

Three practical options:

**Option A: Per-survey Parquet files with union-by-name queries (simplest)**
Each survey is a separate directory. DuckDB's `read_parquet(..., union_by_name=True)` fills missing columns with NULL. Add a `survey` string column at ingest.

**Option B: Core columns + per-survey extension tables**
Define a core schema (ra, dec, z_best, z_type, survey_id, object_id) in every catalog; per-survey extra columns in separate Parquet files joined on `object_id`.

**Option C: HiPSCat with per-survey catalogs (cleanest)**
LSDB supports cross-matching between catalogs with different schemas. The crossmatch returns a merged table with prefixed column names (`left_`, `right_`).

---

## 5. Weights, Masks, and Flags

**Best practices:**
1. **Store weights as columns.** `weight_completeness`, `weight_fiber_collisions`, `weight_photometric_systematic` as float32 in the same Parquet file — no join overhead.
2. **Store boolean flags as bitfields.** A single `flags` int64 column with documented bit positions.
3. **Store geometric mask as MOC.** The angular window function belongs in a MOC file, not in the per-object table.
4. **HEALPix systematic maps.** Per-pixel maps (depth, PSF size, extinction) as HEALPix FITS files. Join at analysis time by pixel lookup:

```python
import healpy as hp
depth_map = hp.read_map('depth_map_nside256.fits')
nside = hp.get_nside(depth_map)
hpx = hp.ang2pix(nside, obj_ra, obj_dec, lonlat=True, nest=False)
obj_depth = depth_map[hpx]  # vectorized, fast
```

---

## 6. Quantitative Comparison Summary

| Criterion | FITS | HDF5 | Parquet | Zarr | SQLite | DuckDB+Parquet | LSDB+HiPSCat |
|---|---|---|---|---|---|---|---|
| Columnar storage | partial | yes | yes | yes | no | yes | yes |
| Redshift slice speed | slow | medium | fast | medium | very slow | very fast | very fast |
| Cone search (built-in) | no | no | no | no | partial | manual† | yes |
| Cross-catalog join | manual | manual | manual | manual | poor | fast (SQL) | native |
| Heterogeneous schemas | per-file | groups | union_by_name | groups | one schema | union_by_name | per-catalog |
| Compression | gz only | good | excellent | good | poor | excellent | excellent |
| 100M row performance | fair | good | excellent | good | poor | excellent | excellent |
| Spatial index | no | no | partition-only | no | R-tree | HEALPix manual | HEALPix native |
| astropy integration | native | native | via PyArrow | limited | via pandas | via pandas | native |
| Cloud-native | no | no | yes | yes | no | yes | yes |

† Requires HEALPix pre-filter column + SQL refinement step.

---

## 7. Recommendation

### Architecture: Parquet + DuckDB + HEALPix + LSDB/HiPSCat

**Storage layout (HiPSCat-compatible):**

```
catalogs/
  sdss_dr17/
    _metadata                     # Parquet dataset metadata
    catalog_info.json             # HiPSCat metadata (nside, total_rows, etc.)
    Norder=5/
      Dir=0/
        Npix=1234.parquet         # ~500k–2M rows, spatially contiguous
    sdss_dr17_footprint.moc.fits
  desi_edr/
    ...
  cosmicflows4/
    ...
```

**Every Parquet file contains:**
- `ra`, `dec` (float64, ICRS degrees)
- `z`, `z_err`, `z_type` (float32 + int8 flag)
- `hpx_order5`, `hpx_order12` (int32, NESTED, computed at ingest)
- `survey_id` (dictionary-encoded string)
- `object_id` (int64, unique within survey)
- Survey-specific columns (heterogeneous; use `union_by_name` for cross-survey queries)
- `weight_*` columns (float32)
- `flags` (int64 bitfield)

**Query layer:**
- Spatial queries / cross-matching → **LSDB** (`cone_search`, `crossmatch`)
- Ad-hoc SQL / redshift slices → **DuckDB** on Parquet
- In-memory analysis with units → **`QTable`** + **`SkyCoord`**
- Footprint operations → **`mocpy`**
- Systematic maps → **`healpy`** pixel lookup

### Decision tree

```
Incoming data (FITS, CSV)?
  → fitsio / astropy read → convert to Parquet at ingest → store as HiPSCat

Spatial query / cross-matching?
  → LSDB (cone_search, crossmatch)
  → Fallback: healpy.query_disc + DuckDB filter on hpx_order12

Analytical query (redshift bins, statistics)?
  → DuckDB on Parquet (predicate pushdown handles z-slices natively)

In-memory arithmetic with units?
  → QTable; SkyCoord for coordinate operations

Survey footprint / mask operations?
  → mocpy MOC intersection and containment

Systematic maps (depth, PSF, extinction)?
  → healpy.read_map; join to catalog by pixel lookup
```

### Formats to avoid

- **SQLite:** row-oriented, no sphere geometry, poor at scale.
- **Zarr:** correct for dense arrays, not for catalogs.
- **HTM:** superseded by HEALPix.
- **FITS as primary format:** use for data release / interoperability only; convert to Parquet on ingest.

---

## 8. Key Packages and Installation

| Package | Role | Install |
|---|---|---|
| `pyarrow` ≥ 17 | Parquet I/O, Arrow in-memory | `pip install pyarrow` |
| `duckdb` ≥ 1.1 | SQL query engine over Parquet | `pip install duckdb` |
| `lsdb` ≥ 0.4 | HiPSCat spatial catalog | `pip install lsdb` |
| `hipscat` ≥ 0.3 | HiPSCat format I/O | installed with lsdb |
| `healpy` ≥ 1.17 | HEALPix pixelization | `pip install healpy` |
| `astropy` ≥ 6.1 | Table, QTable, SkyCoord, units | `pip install astropy` |
| `mocpy` ≥ 0.14 | MOC footprint operations | `pip install mocpy` |
| `fitsio` ≥ 1.2 | Fast FITS ingest (C-backed) | `pip install fitsio` |
| `h5py` ≥ 3.11 | HDF5 for auxiliary products | `pip install h5py` |
| `scikit-learn` ≥ 1.5 | BallTree for cross-matching | `pip install scikit-learn` |

---

## 9. Concrete Code Sketch

```python
# ---- INGEST ----
import fitsio, numpy as np, healpy as hp
import pyarrow as pa, pyarrow.parquet as pq

def ingest_fits_to_hipscat(fits_path, out_dir, survey_name,
                            nside_partition=32, nside_fine=4096):
    """Convert a FITS catalog to HiPSCat-compatible partitioned Parquet."""
    with fitsio.FITS(fits_path) as f:
        ra  = f[1].read_column('RA').astype(np.float64)
        dec = f[1].read_column('DEC').astype(np.float64)

    hpx_part = hp.ang2pix(nside_partition, ra, dec, lonlat=True, nest=True)
    hpx_fine  = hp.ang2pix(nside_fine,     ra, dec, lonlat=True, nest=True)

    table = pa.table({
        'ra': ra, 'dec': dec,
        'hpx_part': hpx_part.astype(np.int32),
        'hpx_fine':  hpx_fine.astype(np.int32),
        'survey': np.full(len(ra), survey_name, dtype=object),
    })
    pq.write_to_dataset(table, root_path=out_dir,
                        partition_cols=['hpx_part'],
                        compression='zstd', write_statistics=True)


# ---- SPATIAL QUERY ----
import lsdb

sdss = lsdb.read_hipscat('catalogs/sdss_dr17/')
desi = lsdb.read_hipscat('catalogs/desi_edr/')

cone_sdss  = sdss.cone_search(ra=185.0, dec=15.0, radius_arcsec=300)
df_cone    = cone_sdss.compute()

matched    = sdss.crossmatch(desi, n_neighbors=1, radius_arcsec=1.0)
df_matched = matched.compute()


# ---- AD-HOC SQL ANALYSIS ----
import duckdb

con    = duckdb.connect()
result = con.execute("""
    SELECT ra, dec, z, r_mag, weight_completeness
    FROM read_parquet('catalogs/sdss_dr17/**/*.parquet')
    WHERE z BETWEEN 0.2 AND 0.4
      AND hpx_fine IN (SELECT unnest(?))
""", [target_fine_pixels.tolist()]).fetch_arrow_table()


# ---- IN-MEMORY ANALYSIS ----
from astropy.table import QTable
import astropy.units as u

qt = QTable.from_pandas(result.to_pandas())
qt['ra'].unit  = u.deg
qt['dec'].unit = u.deg
weighted_mean_z = np.average(qt['z'], weights=qt['weight_completeness'])


# ---- FOOTPRINT / MASK ----
from mocpy import MOC

sdss_moc     = MOC.from_fits('catalogs/sdss_dr17/footprint.moc.fits')
in_footprint = sdss_moc.contains_lonlat(qt['ra'], qt['dec'])
qt_masked    = qt[in_footprint]


# ---- SYSTEMATIC MAP LOOKUP ----
depth_map = hp.read_map('catalogs/sdss_dr17/depth_nside256.fits', nest=False)
hpx_depth = hp.ang2pix(256, qt['ra'].value, qt['dec'].value, lonlat=True, nest=False)
qt['depth'] = depth_map[hpx_depth]
```

---

*This architecture reflects how DESI, Rubin/LSST, and LINCC Frameworks are building catalog infrastructure as of 2025–2026.*
