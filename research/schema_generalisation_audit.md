# Schema Generalisation Audit — oneuniverse vs the Survey Landscape

**Date:** 2026-05-28
**Purpose:** Audit `oneuniverse` Pillar 1 (data + orchestration) against
the real-world diversity of cosmology survey datasets catalogued in
[`survey_landscape_review.md`](survey_landscape_review.md), and propose
concrete API changes so the package can ingest every class of survey we
expect to onboard (Pillar 1 generalisation, post-stabilisation).

Scope: schema, manifest, converter, dataset_view, ONEUID/sub-object,
PDF, surveys/, combine/. No estimator or forward-model concerns.

---

## 1. Schema (`oneuniverse/data/schema.py`)

**Currently supports.** CORE, SPECTROSCOPIC, PHOTOMETRIC, PV, QSO,
SNIA, PROBABILISTIC_REDSHIFT column groups as static tuples of frozen
`ColumnDef(name, dtype, unit, description, required)`. `z_type` is a
hard-coded enum `{spec, phot, phot_pdf, pv, none}`. Scalar per-object
properties; multi-band photometry as explicit
`psfmag_u/g/r/i/z + extinction_u/g/r/i/z`; named BOSS-style weights
(`w_fkp/comp/cp/noz/sys/tot`); DLA prefix counts (`n_dla`).

**Does not handle.**
- Per-row **variable-length** arrays (Lyα δ, ZTF light curves, GAIA XP
  spectra).
- **Masked / nullable** array dtypes; NaN is ad-hoc.
- **`z_type` extensibility** — to add `spec_uncertain`, `cluster_z`,
  `xcorr_z`, `clustering_z`, `redrock_template` requires code change.
- **Shear catalogs** (`e1/e2`, `R11/R22`, calibration biases, per-
  component uncertainties).
- **Variable filter set per survey/partition** — fixed column names
  assume the filter list is known at code-write time.
- **Particle snapshots, IFU cubes, mesh fields, primary pixel maps**
  (only POINT, SIGHTLINE, HEALPIX, LIGHTCURVE geometries declared).
- **Classification PDFs** (galaxy-type, AGN-vs-galaxy, transient
  classifier outputs).
- **Per-column units machine-readable** + frame-of-reference tags
  (comoving vs redshift vs observer; AB vs Vega; vacuum vs air).

**Suggested API.**

```python
# oneuniverse/data/ztypes.py
Z_TYPE_REGISTRY: Set[str] = {
    "spec", "phot", "phot_pdf", "pv", "none",
    "spec_uncertain", "cluster_z", "xcorr_z", "clustering_z",
    "spec_lya", "redrock_template",
}
def register_z_type(name: str) -> None: ...
```

```python
@dataclass(frozen=True)
class ColumnDef:
    name: str
    dtype: str
    unit: Optional[str] = None
    description: str = ""
    required: bool = False
    frame: Optional[str] = None          # "comoving", "redshift", "observer", "AB", "Vega"
    wavelength_convention: Optional[str] = None  # "vacuum", "air"
    epoch: Optional[float] = None        # e.g. 2016.0 for GAIA
    nullable: bool = False
```

Promote `z_type` validation: writer rejects rows whose value is not in
`Z_TYPE_REGISTRY`; manifest stores the observed subset for readers.

---

## 2. Format Spec (`oneuniverse/data/format_spec.py`)

**Currently supports.** Four `DataGeometry` values: POINT, SIGHTLINE,
HEALPIX, LIGHTCURVE. HEALPix NSIDE=32 NEST partitioning for POINT;
auto-coarsening (Phase 12 F3); per-partition `PartitionStats` with
hard-coded ra/dec/z/t ranges.

**Does not handle.**
- **Strip / wedge footprints** efficiently (HEALPix waste for WiggleZ
  strips, DESI petals).
- **HEALPix probability maps as the row payload** (GW skymaps).
- **N-D cubes** (HI, 21cm, IFU): no geometry covers them.
- **Multi-scale hierarchies** beyond two levels (LIGHTCURVE/SIGHTLINE
  are flat parent+child).
- **Per-partition stats on arbitrary columns** (S/N, color,
  magnitude, EBV) — pushdown limited to ra/dec/z/t.

**Suggested API.**

```python
class DataGeometry(str, Enum):
    POINT = "point"
    SIGHTLINE = "sightline"
    HEALPIX = "healpix"        # pixel map sidecar
    LIGHTCURVE = "lightcurve"
    GW_SKYMAP = "gw_skymap"    # row payload = HEALPix prob map
    CUBE = "cube"              # N-D array per row (HI, IFU)
    PARTICLE = "particle"      # mock snapshots
```

```python
@dataclass(frozen=True)
class PartitionStats:
    ra_min: Optional[float] = None
    ra_max: Optional[float] = None
    dec_min: Optional[float] = None
    dec_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    t_min: Optional[float] = None
    t_max: Optional[float] = None
    extra_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
```

Reader emits `pa.compute.field(col) >= lo & field(col) <= hi` pushdown
for any column in `extra_ranges`.

---

## 3. Manifest (`oneuniverse/data/manifest.py`)

**Currently supports.** Typed `Manifest` (format 2.1.0); file
provenance via sha256; `PartitioningSpec`, `TemporalSpec`,
`DatasetValidity`, `PdfSpec`; catch-all `extra: Dict[str, Any]`.

**Does not handle.**
- **Hierarchy depth** (only one parent→child level via SubobjectLinks).
- **Per-partition quality metrics** (mean S/N, completeness fraction).
- **Dynamic column sets** (per-partition filter availability).
- **Multi-component PDF shapes beyond scalar n_components** (e.g.
  mixture with per-component covariance matrix).
- **Coordinate frame / epoch** of the dataset as a whole.
- **λ convention** (vacuum/air) per spectrum dataset.
- **Bitemporal granularity** below dataset level (per-row
  `epoch_of_position`, `valid_from/to`).

> **Out of scope.** No cosmology metadata (H₀, Ωₘ, fiducial baseline,
> distance-model selection) belongs in Pillar 1. The package stores
> what surveys deliver — comoving conversion, dipole modelling, and
> distance-modulus interpretation live in Pillars 2 (`flip`) and 3
> (forward models).

**Suggested API.** Observational metadata only — frame, epoch,
wavelength convention. No cosmology.

```python
@dataclass(frozen=True)
class CoordinateSpec:
    frame: str = "icrs"                # or "galactic", "ecliptic"
    epoch: Optional[float] = None      # e.g. 2016.0 for GAIA DR3
    proper_motion_available: bool = False
    parallax_available: bool = False

@dataclass(frozen=True)
class SpectrumSpec:
    wavelength_convention: str         # "vacuum" or "air"
    log_binned: bool = True
    rest_frame_corrected: bool = False
    wavelength_unit: str = "angstrom"  # or "nanometer", "micron"

# Manifest gains:
coordinate: Optional[CoordinateSpec] = None
spectrum: Optional[SpectrumSpec] = None  # only for SIGHTLINE
```

Redshift-frame disambiguation (heliocentric vs CMB vs LSR) is
**per-column** observational metadata, expressed via the new
`ColumnDef.frame` field — not a dataset-level cosmology choice. A
single dataset can carry both `z_helio` and `z_cmb` columns with
different `frame` annotations.

Audit `extra` usage across the codebase quarterly; promote any key
used by ≥2 surveys to a typed field.

---

## 4. Converter (`oneuniverse/data/converter.py`)

**Currently supports.** `write_ouf_dataset()` for the four declared
geometries; `_chunk_to_table()` handles `FixedSizeList[float32,
n_components]` for PDF columns; auto `_healpix32`; partition by row
count or HEALPix cell; sha256 + stats per partition; zstd compression.

**Does not handle.**
- **Variable-length list columns** (`pa.list_(pa.float32())` for
  per-LOS variable pixel counts).
- **`LargeList` / nested struct columns** (filter-set lists, alert
  history arrays).
- **Masked nullable arrays** at PyArrow level.
- **Ragged photometry** (per-row filter set varying).
- **Multi-object hierarchies** beyond flat parent+child (no
  `parent_galaxy_id → spectrum_id → line_id` chain).
- **Alternative codecs** (lz4, snappy) for downstream consumers.

**Suggested API.** Extend `_chunk_to_table(chunk, pdf_spec, *,
list_columns=None, struct_columns=None)`:

```python
def _chunk_to_table(
    chunk: pd.DataFrame,
    pdf_spec: Optional[PdfSpec] = None,
    *,
    list_columns: Mapping[str, pa.DataType] = (),
    struct_columns: Mapping[str, pa.StructType] = (),
) -> pa.Table:
    """Convert DataFrame to PyArrow table.

    list_columns and struct_columns let loaders declare per-row
    variable-length or composite payloads (Lya delta arrays,
    multi-filter photometry, alert history).
    """
```

`write_ouf_dataset` grows `column_dtype_map: Mapping[str, str]` to
override automatic detection (e.g. `{"delta": "list<f4>"}`).

---

## 5. DatasetView (`oneuniverse/data/dataset_view.py`)

**Currently supports.** Lazy reader; partition pruning via ra/dec/z/t
ranges + `healpix_cells`; column projection; `pa.compute` pushdown
filter; cone + sky-patch selection; `load_pdf() →
ProbabilisticRedshift`; manifest-driven NSIDE resolution (Phase 12 D5).

**Does not handle.**
- **Streaming / iterator API** for out-of-core scans (`scan_iter`).
- **Map-based spatial filtering** (cone + per-pixel probability
  threshold from a HEALPix map).
- **Cross-geometry queries** (point catalog ⨯ mask map at the same
  NSIDE).
- **Per-partition column stats other than ra/dec/z/t** (requires
  `PartitionStats.extra_ranges`).
- **Alternative filter expression languages** (SQL, astropy mask
  syntax). Not strictly needed but ergonomic for end users.

**Suggested API.**

```python
def scan_iter(
    self,
    *,
    partition_size: int = 10_000,
    columns: Optional[Sequence[str]] = None,
    **filter_kwargs,
) -> Iterator[pa.Table]:
    """Yield chunks of (partition_size) rows for out-of-core scans."""

def select_by_map(
    self,
    map_array: np.ndarray,
    map_nside: int,
    *,
    threshold: float,
    nest: bool = True,
) -> "DatasetView":
    """Filter rows whose _healpix32 cell maps onto pixels with
    map_array >= threshold (resampled to map_nside)."""
```

---

## 6. ONEUID & Sub-object (`oneuid.py`, `subobject.py`, `database.py`)

**Currently supports.** Multi-survey cross-match via `CrossMatchRules`
(sky tolerance, per-pair z tolerance, rejection); SubobjectRules with
ambiguous-match acceptance + confidence; bitemporal validity
(`valid_from/to_utc`); auto-archive on rebuild; symmetric z-type pair
rules.

**Does not handle.**
- **Three-level hierarchy** (cluster → galaxy → spectrum / line)
  needs two separate link sidecars with no transitive query API.
- **Probabilistic matching** ("P(match | ra, dec, z)" from a posterior
  or a classifier output).
- **Cross-matching point catalogs against HEALPix probability maps**
  (galaxy ⨯ GW skymap; galaxy ⨯ cluster contamination map).
- **Attribute-based rules** ("only match if Δcolor < 0.1",
  "only match if both S/N > 100", "only match if PM-propagated
  positions agree").
- **Tuple / byte-payload IDs** (current model assumes int64
  `galaxy_id`).

**Suggested API.**

```python
@dataclass(frozen=True)
class SubobjectLinks:
    ...
    score_column: Optional[str] = None       # float32 match confidence
    relation_type: Literal["containment", "causality", "association"] = "association"
    relation_metadata: Mapping[str, Any] = field(default_factory=dict)
    next_level: Optional[str] = None         # name of next SubobjectLinks for chaining

@dataclass(frozen=True)
class CrossMatchRules:
    ...
    attribute_filters: Sequence["AttributeFilter"] = ()  # extensible
```

```python
def build_subobject_links_to_map(
    parent_dataset: DatasetView,
    event_map_dataset: DatasetView,       # row-per-event with HEALPix payload
    *,
    overlap_threshold: float = 0.5,
    score_kind: str = "containment",      # or "iou", "max_overlap"
) -> SubobjectLinks: ...
```

`galaxy_id` widened to `Union[int64, bytes]` or a structured tuple
field for non-int IDs.

---

## 7. PDF (`oneuniverse/data/pdf.py`)

**Currently supports.** Three parameterisations: `interp` (gridded
p(z)), `quant` (z(q)), `mixmod` (Gaussian mixture). Unlimited
`n_components` in code; ProbabilisticRedshift vectorises moment, CDF,
PPF, sampling.

**Does not handle.**
- **Sample-based PDFs** (`qp` "sample" mode; common in DES SOMPZ and
  GW samples).
- **Histogram PDFs** (binned with edges, common in KiDS-1000 n(z)).
- **Per-bin tomographic n(z)** as a dataset-level object, not row-level.
- **Classification PDFs** (categorical, multi-class).
- **Multi-dimensional posteriors** (P(z, type), P(z, template), P(z,
  SED)).
- **Truncated / sparse grids** (grid_mask).
- **Custom column names** (z_pdf_values is hard-coded).

**Suggested API.**

```python
@dataclass(frozen=True)
class PdfSpec:
    parameterisation: Literal["interp", "quant", "mixmod", "sample", "hist"]
    n_components: int
    grid: Optional[Sequence[float]] = None       # interp + hist edges
    quantile_levels: Optional[Sequence[float]] = None
    grid_kind: str = "z"
    value_column: str = "z_pdf_values"
    sigma_column: Optional[str] = "z_pdf_sigma"
    weights_column: Optional[str] = "z_pdf_weights"
    grid_mask: Optional[Sequence[bool]] = None   # sparse / truncated
    axis_labels: Tuple[str, ...] = ("z",)        # multi-D posteriors

@dataclass(frozen=True)
class ClassificationPdfSpec:
    parameterisation: Literal["categorical", "mixture"]
    classes: Tuple[str, ...]
    value_column: str = "class_pdf_values"

@dataclass(frozen=True)
class TomographicNzSpec:
    """Per-bin n(z) at dataset level, not per row."""
    bins: Sequence[Tuple[float, float]]
    grid: Sequence[float]
    values: Sequence[Sequence[float]]   # shape (n_bins, n_grid)
    bin_assignment_column: str = "tomo_bin"
```

Align with `qp` (Malz & Marshall 2018; Schmidt+ 2020) so RAIL outputs
roundtrip with no conversion.

---

## 8. Survey Loaders (`oneuniverse/data/surveys/*.py`)

**Currently provides.** eBOSS QSO, DESI BGS, SDSS MGS, DESI PV, CF4,
Pantheon+, DES DR2, test-only DESI DR1. Each maps native FITS/CSV to
CORE + extended group columns. Per-survey default weights;
multi-band photometry renaming; QSO-specific columns
(`z_pipe/pca/vi`, BAL flags, DLA counts).

**Does not handle.**
- **IFU / spectrum payloads** in loader output (returns scalars only).
- **Per-visit / pixel-level metadata** (only survey-level).
- **Shear catalogs** (DES Y3 shape, KiDS lensfit, HSC HSM).
- **Dynamic filter sets** (must hardcode every band column).
- **Calibration systematics** as columns (zero-point offsets,
  per-template biases).
- **PIP `BITWEIGHTS` int64 arrays.**

**Suggested API.** Add `BaseSurveyLoader.characteristic_fields_optional`
to declare dynamic column expectations:

```python
class BaseSurveyLoader:
    characteristic_fields: Mapping[str, ColumnSpec] = {}
    characteristic_fields_optional: Mapping[str, ColumnSpec] = {}

# Example:
class DESY6Loader(BaseSurveyLoader):
    characteristic_fields_optional = {
        "magnitude_bands": ColumnSpec("list[str]", "", "Filter names"),
        "magnitudes":      ColumnSpec("f4[n_bands]", "mag", "Per-band mags"),
        "bitweights":      ColumnSpec("i8[64]", "", "PIP 64-bit weights"),
    }
```

Converter respects `list[str]` / `f4[n_bands]` / `i8[n_bits]` dtype
mini-language for variable shapes.

---

## 9. Combine / Weights (`oneuniverse/combine/`)

**Currently supports.** `WeightedCatalog` registry; `ColumnWeight`,
`ConstantWeight`, `InverseVarianceWeight`, `FKPWeight`,
`QualityMaskWeight`, `HealpixMapWeight`, `PdfWidthIVarWeight`,
`PdfMeanRedshiftWeight`, BOSS-named wrappers
(`FiberCollisionWeight/ZFailureWeight/CompletenessWeight`),
`boss_total_weight`. Composition via `*` (ProductWeight). Public
registration `register_default(survey_type, z_type, factory)`.

**Does not handle.**
- **Shear weights** (e1/e2 errors, R-matrix propagation, selection
  bias subtraction).
- **Density-dependent weights** (scale-dependent bias correction from
  local n_g).
- **Time-dependent weights** (variability surveys, transient cuts).
- **Classification-aware weights** (`p_galaxy`, `p_AGN`).
- **PIP bitweight expansion** (`BITWEIGHTS` int64 → 64 boolean
  realisations).
- **Sub-species registry keys** (e.g. `(DESI, BGS_BRIGHT, spec)` vs
  `(DESI, BGS_FAINT, spec)`).

**Suggested API.**

```python
class ShearWeight(Weight):
    """Propagates shear measurement errors and metacal/lensfit response."""
    def __init__(self, e1_err_col, e2_err_col, R11_col, R22_col,
                 c1_col=None, c2_col=None, m_col=None): ...

class PipBitweightWeight(Weight):
    """Expand BITWEIGHTS int64[64] into 64 realisations, average."""
    def __init__(self, bitweights_col: str = "BITWEIGHTS"): ...

class ClassificationWeight(Weight):
    """Weight by P(class | features) from a classifier."""
    def __init__(self, prob_col: str, target_class: str): ...

class TemporalWeight(Weight):
    def __init__(self, time_column: str, weight_func: Callable[[np.ndarray], np.ndarray]): ...
```

Extend registry key to `(survey_type, sub_kind, z_type)`; default
`sub_kind=None`.

---

## 10. Cross-cutting gaps + prioritised roadmap

### High priority (unblock entire survey classes)

1. **Extensible `z_type` registry** — unblocks Lyα, cluster_z,
   xcorr_z, RAIL outputs.
2. **`CoordinateSpec` + per-column `frame` annotation in Manifest** —
   required for GAIA epoch propagation, PV surveys to record
   heliocentric vs CMB *as observational metadata*, Pantheon+
   `zHD/zHEL` distinction. **No cosmology fields** — H₀/Ωₘ/distance
   conversion belong in Pillars 2/3, not here.
3. **Variable-length list columns in converter** — unblocks Lyα
   pixels, ZTF light curves, GAIA XP spectra, multi-filter photometry.
4. **Generic `PartitionStats.extra_ranges`** — unblocks pushdown on
   S/N, EBV, mag — needed for narrow-strip / wedge surveys where
   ra/dec/z pruning alone is wasteful.
5. **`SpectrumSpec`** (vacuum/air, rest-frame state) — required for
   any consumer that crosses spec surveys.

### Medium priority (unlock real-world use cases)

6. **Map-based ONEUID / SubobjectLinks** — GW skymaps, cluster
   contamination maps, photometric depth masks as match constraints.
7. **Multi-level hierarchy** (`relation_type`, `next_level`) — galaxy
   → spectrum → emission line, cluster → galaxy → spectrum.
8. **Dynamic filter-set metadata** (`dynamic_columns`) — KiDS/DES/
   HSC/Rubin/J-PAS without hardcoding every band.
9. **Shear column group + `ShearWeight`** — DES Y3, KiDS-1000,
   HSC-Y3, Rubin shapes.
10. **`PdfSpec` polymorphism + tomographic n(z) dataset kind** —
    qp alignment; KiDS/DES/HSC tomographic bin n(z).
11. **Sample-based + histogram PDFs** — `qp` "sample" and "hist"
    parameterisations.

### Lower priority (polish + future-proofing)

12. **Multi-dimensional PDFs** (P(z, type), P(z, template)).
13. **Per-row bitemporal columns** + alert ingestion-time tracking.
14. **`ClassificationPdf` + classification weight families** —
    galaxy-type, AGN-vs-galaxy, transient classifier outputs.
15. **`CUBE` / `PARTICLE` geometries** — IFU, HI cubes, mock
    snapshots.
16. **`extra` dict audit** — promote frequently-used keys to typed
    fields; eliminate junk-drawer drift.
17. **Tuple / byte-payload `galaxy_id`** — composite IDs like
    `PLATE-MJD-FIBERID`.

### Cleanup opportunities flagged during audit

- `z_pdf_values` / `z_pdf_sigma` / `z_pdf_weights` are hard-coded in
  both `_chunk_to_table` and `ProbabilisticRedshift`. Aliasing via
  `PdfSpec.value_column` etc. would let surveys declare native names.
- `_log_summary` in converter has a `Path(None) / config.data_filename`
  short-circuit (Phase 12 D3 fix); revisit once `loader=` overload is
  used by ≥3 surveys.
- `PartitionStats` four hard-coded axes is the single biggest source of
  pushdown limitation — refactor to `Dict[str, Tuple[float, float]]`
  is straightforward but touches `manifest.py`, `dataset_view.py`,
  `converter.py` stats builder, and every reader test.
- `extra: dict` appears on `Manifest`, `PartitioningSpec`, `PdfSpec`,
  and arguably `LoaderSpec`. Standardise on a single typed
  escape-hatch helper.

---

## 11. Suggested staging into phases

A natural Phase 16+ decomposition (each producing a working OUF
version bump + green test suite):

- **Phase 16 — Observational metadata expansion.** Landed 2026-05-28.
  Adds `CoordinateSpec`, `SpectrumSpec`, extensible `z_type` registry,
  `ColumnDef` gains `frame`/`epoch`/`wavelength_convention`/`nullable`.
  No cosmology. OUF 2.2.0. See
  [`../plans/2026-05-28-phase16-observational-metadata.md`](../plans/2026-05-28-phase16-observational-metadata.md).
- **Phase 17 — Variable-length columns + generic partition stats.**
  Landed 2026-05-29. Adds `_chunk_to_table(column_dtypes=...)` with a
  small dtype mini-language (`f4[N]`, `i8[N]`, `list<f4>`,
  `large_list<f4>`), `PartitionStats.extra_ranges`,
  `write_ouf_dataset(extra_stats_columns=...)`,
  `DatasetView.extra_filters`. OUF 2.3.0. See
  [`../plans/2026-05-28-phase17-variable-length-and-partition-stats.md`](../plans/2026-05-28-phase17-variable-length-and-partition-stats.md).
- **Phase 18 — PDF polymorphism + tomographic n(z).** `sample` +
  `hist` parameterisations, sparse grid_mask, axis_labels, tomographic
  n(z) as a manifest-level object. RAIL `qp` roundtrip.
- **Phase 19 — Shear + weight expansion.** SHEAR_COLUMNS group,
  `ShearWeight`, `PipBitweightWeight`, registry sub_kind. Unblocks DES
  Y3, KiDS-1000, HSC-Y3, DESI bitweights.
- **Phase 20 — Map-based ONEUID / sub-object.** Probabilistic match,
  map overlap scoring, attribute filters, multi-level chains. Unblocks
  GW × galaxy, cluster member chains, deblender hierarchies.
- **Phase 21 — Geometry expansion** (`CUBE`, `PARTICLE`, `GW_SKYMAP`):
  optional; pursue once a concrete consumer appears.

Phase 13 (real-survey loader writes for the existing schema) remains
postponed per the 2026-05-22 forward plan. Phases 16–20 are
prerequisites for clean BOSS+/DESI/Euclid/Rubin loaders; Phase 13
absorbs naturally afterwards.
