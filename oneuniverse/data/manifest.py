"""Typed manifest for the oneuniverse file format (OUF) v2.

A :class:`Manifest` is the authoritative description of a converted
survey on disk. It is written once by the converter and read by every
downstream consumer (database scanner, ONEUID builder, query engine).

Design goals
------------

- **Single source of truth.** Every field required downstream is declared
  here; no scattered ``dict.get("x", default)`` calls.
- **Validation at the boundary.** :func:`read_manifest` raises
  :class:`ManifestValidationError` on any malformed file. No silent
  defaults.
- **Bump-proof.** ``oneuniverse_format_version`` is pinned; reading a
  different major version raises rather than silently coercing.
- **Auditable.** Content hashes (sha256 prefix) on original files and
  partitions so consumers can detect drift.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from oneuniverse.data._atomic import atomic_write_text
from oneuniverse.data.classification_pdf import ClassificationPdfSpec
from oneuniverse.data.coordinate_spec import CoordinateSpec
from oneuniverse.data.cube_spec import CubeSpec
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.gwskymap_spec import GwSkymapSpec
from oneuniverse.data.pdf import PdfSpec
from oneuniverse.data.spectrum_spec import SpectrumSpec
from oneuniverse.data.temporal import TemporalSpec
from oneuniverse.data.tomographic_nz import TomographicNzSpec
from oneuniverse.data.validity import DatasetValidity

# Single source of truth for the format version lives in format_spec (the
# 2026-06-10 review found these two modules each pinning their own copy —
# the same drift that produced the LIGHTCURVE 2.1.0 mislabel, bug F1).
from oneuniverse.data.format_spec import (  # noqa: F401  (re-export)
    FORMAT_VERSION,
    SCHEMA_VERSION,
)

#: OUF 2.6: per-partition stats live in this parquet sidecar next to
#: manifest.json; the manifest itself stays identity-only and stable.
PARTITION_INDEX_FILENAME = "_index.parquet"


class ManifestValidationError(ValueError):
    """Raised when a manifest file is malformed or format-incompatible."""


# ── Sub-specs ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str                              # numpy dtype string
    unit: Optional[str] = None
    description: Optional[str] = None
    required: bool = False


@dataclass(frozen=True)
class OriginalFileSpec:
    path: str                               # relative to survey_path
    sha256: str                             # 16-hex-char prefix
    n_rows: Optional[int]
    size_bytes: int
    format: str


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
    # Phase 17: generic per-column min/max for arbitrary axes
    # (S/N, EBV, magnitude, ...). Empty by default for forward-compat.
    extra_ranges: Dict[str, tuple] = field(default_factory=dict)


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    n_rows: int
    sha256: str
    size_bytes: int
    stats: PartitionStats = field(default_factory=PartitionStats)
    healpix_cell: Optional[int] = None


@dataclass(frozen=True)
class PartitioningSpec:
    scheme: str                             # e.g. "healpix32"
    column: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoaderSpec:
    name: str
    version: str


# ── Manifest ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Manifest:
    oneuniverse_format_version: str
    oneuniverse_schema_version: str
    geometry: DataGeometry
    survey_name: str
    survey_type: str
    created_utc: str
    original_files: List[OriginalFileSpec]
    partitions: List[PartitionSpec]
    partitioning: Optional[PartitioningSpec]
    schema: List[ColumnSpec]
    conversion_kwargs: Dict[str, Any]
    loader: LoaderSpec
    # Geometry- or survey-specific extras that do not deserve a
    # first-class field (e.g. healpix_nside, n_sightlines).
    extra: Dict[str, Any] = field(default_factory=dict)
    temporal: Optional[TemporalSpec] = None
    validity: Optional[DatasetValidity] = None
    pdf_spec: Optional[PdfSpec] = None
    # Phase 16: observational metadata. All None / empty by default for
    # forward-compat with 2.1.x manifests.
    coordinate: Optional[CoordinateSpec] = None
    spectrum: Optional[SpectrumSpec] = None
    observed_z_types: tuple = ()
    # Phase 18 additions.
    tomographic_nz: Optional[TomographicNzSpec] = None
    classification_pdf: Optional[ClassificationPdfSpec] = None
    # Phase 22 additions.
    cube: Optional[CubeSpec] = None
    gwskymap: Optional[GwSkymapSpec] = None

    @property
    def n_rows(self) -> int:
        return sum(p.n_rows for p in self.partitions)

    @property
    def n_partitions(self) -> int:
        return len(self.partitions)


# ── I/O ─────────────────────────────────────────────────────────────────


def write_manifest(path: Union[str, Path], manifest: Manifest) -> None:
    """Atomically write a manifest to *path* (…/manifest.json).

    OUF 2.6: per-partition stats are written to the ``_index.parquet``
    sidecar next to the manifest; the JSON stays identity-only (stable,
    diffable, O(1) regardless of partition count). The sidecar is written
    first, so a crash in between leaves the previous manifest intact.
    ``read_manifest`` reconstructs ``Manifest.partitions`` transparently —
    no downstream consumer changes.
    """
    path = Path(path)
    payload = _to_dict(manifest)
    _write_partition_index(path.parent / PARTITION_INDEX_FILENAME,
                           manifest.partitions)
    payload.pop("partitions", None)
    payload["partition_index"] = PARTITION_INDEX_FILENAME
    payload["n_partitions"] = len(manifest.partitions)
    payload["n_rows_total"] = sum(p.n_rows for p in manifest.partitions)
    text = json.dumps(payload, indent=2, sort_keys=False, default=str)
    atomic_write_text(path, text)


_INDEX_SCHEMA_COLS = ("name", "n_rows", "sha256", "size_bytes",
                      "healpix_cell", "ra_min", "ra_max", "dec_min",
                      "dec_max", "z_min", "z_max", "t_min", "t_max",
                      "extra_ranges_json")


def _write_partition_index(path: Path, partitions: List[PartitionSpec]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        ("name", pa.string()), ("n_rows", pa.int64()),
        ("sha256", pa.string()), ("size_bytes", pa.int64()),
        ("healpix_cell", pa.int64()),
        ("ra_min", pa.float64()), ("ra_max", pa.float64()),
        ("dec_min", pa.float64()), ("dec_max", pa.float64()),
        ("z_min", pa.float64()), ("z_max", pa.float64()),
        ("t_min", pa.float64()), ("t_max", pa.float64()),
        ("extra_ranges_json", pa.string()),
    ])
    cols: Dict[str, list] = {c: [] for c in _INDEX_SCHEMA_COLS}
    for p in partitions:
        s = p.stats
        cols["name"].append(p.name)
        cols["n_rows"].append(int(p.n_rows))
        cols["sha256"].append(p.sha256)
        cols["size_bytes"].append(int(p.size_bytes))
        cols["healpix_cell"].append(
            None if p.healpix_cell is None else int(p.healpix_cell))
        for f in ("ra_min", "ra_max", "dec_min", "dec_max",
                  "z_min", "z_max", "t_min", "t_max"):
            cols[f].append(getattr(s, f))
        cols["extra_ranges_json"].append(
            json.dumps(s.extra_ranges) if s.extra_ranges else None)
    pq.write_table(pa.table(cols, schema=schema), path)


def _read_partition_index(path: Path,
                          manifest_path: Path) -> List[PartitionSpec]:
    import pyarrow.parquet as pq
    if not path.is_file():
        raise ManifestValidationError(
            f"{manifest_path}: declares partition_index "
            f"'{path.name}' but the sidecar file is missing")
    out = []
    for r in pq.read_table(path).to_pylist():
        er = (json.loads(r["extra_ranges_json"])
              if r.get("extra_ranges_json") else {})
        out.append(PartitionSpec(
            name=r["name"], n_rows=int(r["n_rows"]), sha256=r["sha256"],
            size_bytes=int(r["size_bytes"]),
            healpix_cell=(None if r["healpix_cell"] is None
                          else int(r["healpix_cell"])),
            stats=PartitionStats(
                ra_min=r["ra_min"], ra_max=r["ra_max"],
                dec_min=r["dec_min"], dec_max=r["dec_max"],
                z_min=r["z_min"], z_max=r["z_max"],
                t_min=r["t_min"], t_max=r["t_max"],
                extra_ranges={k: (float(v[0]), float(v[1]))
                              for k, v in er.items()}),
        ))
    return out


def read_manifest(path: Union[str, Path]) -> Manifest:
    """Read and validate a manifest file.

    Raises
    ------
    ManifestValidationError
        On any malformed, missing, or format-incompatible manifest.
    """
    path = Path(path)
    if not path.is_file():
        raise ManifestValidationError(f"Manifest file not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ManifestValidationError(f"{path}: invalid JSON ({e})") from e
    if not isinstance(raw, dict):
        raise ManifestValidationError(f"{path}: top-level must be a JSON object")
    return _from_dict(raw, path)


# ── (De)serialization internals ─────────────────────────────────────────


def _to_dict(m: Manifest) -> Dict[str, Any]:
    d = asdict(m)
    d["geometry"] = m.geometry.value
    d["temporal"] = m.temporal.to_dict() if m.temporal is not None else None
    d["validity"] = m.validity.to_dict() if m.validity is not None else None
    d["pdf_spec"] = m.pdf_spec.to_dict() if m.pdf_spec is not None else None
    d["coordinate"] = (
        m.coordinate.to_dict() if m.coordinate is not None else None
    )
    d["spectrum"] = m.spectrum.to_dict() if m.spectrum is not None else None
    d["observed_z_types"] = list(m.observed_z_types)
    d["tomographic_nz"] = (
        m.tomographic_nz.to_dict() if m.tomographic_nz is not None else None
    )
    d["classification_pdf"] = (
        m.classification_pdf.to_dict()
        if m.classification_pdf is not None else None
    )
    d["cube"] = m.cube.to_dict() if m.cube is not None else None
    d["gwskymap"] = m.gwskymap.to_dict() if m.gwskymap is not None else None
    return d


_REQUIRED_TOP_KEYS = (
    "oneuniverse_format_version",
    "oneuniverse_schema_version",
    "geometry",
    "survey_name",
    "survey_type",
    "created_utc",
    "original_files",
    # "partitions" is required as EITHER an embedded array (<=2.5) OR a
    # "partition_index" sidecar reference (2.6+) — checked in _from_dict.
    "schema",
    "conversion_kwargs",
    "loader",
)


def _require(raw: Dict[str, Any], key: str, path: Path) -> Any:
    if key not in raw:
        raise ManifestValidationError(
            f"{path}: missing required manifest key '{key}'"
        )
    return raw[key]


def _load_partition_stats(raw: Dict[str, Any]) -> PartitionStats:
    """Build a :class:`PartitionStats` from JSON-decoded dict.

    JSON has no tuple type, so ``extra_ranges`` values arrive as
    2-element lists; we normalise them back to tuples.
    """
    er = {
        k: (float(v[0]), float(v[1]))
        for k, v in raw.get("extra_ranges", {}).items()
    }
    return PartitionStats(
        ra_min=raw.get("ra_min"), ra_max=raw.get("ra_max"),
        dec_min=raw.get("dec_min"), dec_max=raw.get("dec_max"),
        z_min=raw.get("z_min"), z_max=raw.get("z_max"),
        t_min=raw.get("t_min"), t_max=raw.get("t_max"),
        extra_ranges=er,
    )


def _from_dict(raw: Dict[str, Any], path: Path) -> Manifest:
    for key in _REQUIRED_TOP_KEYS:
        _require(raw, key, path)

    fmt = raw["oneuniverse_format_version"]
    parts_v = fmt.split(".") if isinstance(fmt, str) else []
    if not (len(parts_v) >= 2 and parts_v[0] == "2"
            and parts_v[1].isdigit() and int(parts_v[1]) <= 6):
        raise ManifestValidationError(
            f"{path}: oneuniverse_format_version={fmt!r} is not compatible "
            f"with this library (expected 2.0.x – 2.6.x)."
        )

    geo = raw["geometry"]
    try:
        geometry = DataGeometry(geo)
    except ValueError as e:
        raise ManifestValidationError(
            f"{path}: unknown geometry {geo!r}"
        ) from e

    original_files = [OriginalFileSpec(**spec) for spec in raw["original_files"]]
    if "partition_index" in raw:               # OUF 2.6+: parquet sidecar
        partitions = _read_partition_index(
            path.parent / raw["partition_index"], path)
    elif "partitions" in raw:                  # OUF <=2.5: embedded array
        partitions = [
            PartitionSpec(
                name=p["name"],
                n_rows=int(p["n_rows"]),
                sha256=p["sha256"],
                size_bytes=int(p["size_bytes"]),
                stats=_load_partition_stats(p.get("stats", {})),
                healpix_cell=(
                    int(p["healpix_cell"])
                    if p.get("healpix_cell") is not None
                    else None
                ),
            )
            for p in raw["partitions"]
        ]
    else:
        raise ManifestValidationError(
            f"{path}: missing required manifest key 'partitions' "
            f"(or 'partition_index' for format >= 2.6)")
    partitioning_raw = raw.get("partitioning")
    partitioning = (
        PartitioningSpec(
            scheme=partitioning_raw["scheme"],
            column=partitioning_raw["column"],
            extra=partitioning_raw.get("extra", {}),
        )
        if partitioning_raw is not None
        else None
    )
    schema = [ColumnSpec(**c) for c in raw["schema"]]
    loader = LoaderSpec(**raw["loader"])

    temporal_raw = raw.get("temporal")
    temporal = TemporalSpec.from_dict(temporal_raw) if temporal_raw else None

    validity_raw = raw.get("validity")
    if validity_raw is not None:
        validity = DatasetValidity.from_dict(validity_raw)
    elif fmt.startswith("2.0"):
        # Forward-compatibility: 2.0.x manifests have no validity block;
        # synthesize one from created_utc so downstream code can rely on
        # .validity being non-None. 2.1.x authors opt in explicitly.
        created = raw["created_utc"]
        if (
            "+" not in created
            and "Z" not in created
            and not created.endswith("+00:00")
        ):
            created = created + "+00:00"
        validity = DatasetValidity(valid_from_utc=created)
    else:
        validity = None

    pdf_raw = raw.get("pdf_spec")
    pdf_spec = PdfSpec.from_dict(pdf_raw) if pdf_raw is not None else None

    coord_raw = raw.get("coordinate")
    coordinate = CoordinateSpec.from_dict(coord_raw) if coord_raw else None
    spec_raw = raw.get("spectrum")
    spectrum = SpectrumSpec.from_dict(spec_raw) if spec_raw else None
    observed_z_types = tuple(raw.get("observed_z_types", ()))

    tnz_raw = raw.get("tomographic_nz")
    tomographic_nz = (
        TomographicNzSpec.from_dict(tnz_raw) if tnz_raw else None
    )
    cpd_raw = raw.get("classification_pdf")
    classification_pdf = (
        ClassificationPdfSpec.from_dict(cpd_raw) if cpd_raw else None
    )
    cube_raw = raw.get("cube")
    cube = CubeSpec.from_dict(cube_raw) if cube_raw else None
    gwsky_raw = raw.get("gwskymap")
    gwskymap = GwSkymapSpec.from_dict(gwsky_raw) if gwsky_raw else None

    return Manifest(
        oneuniverse_format_version=fmt,
        oneuniverse_schema_version=raw["oneuniverse_schema_version"],
        geometry=geometry,
        survey_name=raw["survey_name"],
        survey_type=raw["survey_type"],
        created_utc=raw["created_utc"],
        original_files=original_files,
        partitions=partitions,
        partitioning=partitioning,
        schema=schema,
        conversion_kwargs=raw["conversion_kwargs"],
        loader=loader,
        extra=raw.get("extra", {}),
        temporal=temporal,
        validity=validity,
        pdf_spec=pdf_spec,
        coordinate=coordinate,
        spectrum=spectrum,
        observed_z_types=observed_z_types,
        tomographic_nz=tomographic_nz,
        classification_pdf=classification_pdf,
        cube=cube,
        gwskymap=gwskymap,
    )
