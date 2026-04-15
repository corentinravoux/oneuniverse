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
from oneuniverse.data.format_spec import DataGeometry

FORMAT_VERSION: str = "2.0.0"
SCHEMA_VERSION: str = "2.0.0"


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


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    n_rows: int
    sha256: str
    size_bytes: int
    stats: PartitionStats = field(default_factory=PartitionStats)


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

    @property
    def n_rows(self) -> int:
        return sum(p.n_rows for p in self.partitions)

    @property
    def n_partitions(self) -> int:
        return len(self.partitions)


# ── I/O ─────────────────────────────────────────────────────────────────


def write_manifest(path: Union[str, Path], manifest: Manifest) -> None:
    """Atomically write a manifest to *path* (…/manifest.json)."""
    path = Path(path)
    payload = _to_dict(manifest)
    text = json.dumps(payload, indent=2, sort_keys=False)
    atomic_write_text(path, text)


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
    return d


_REQUIRED_TOP_KEYS = (
    "oneuniverse_format_version",
    "oneuniverse_schema_version",
    "geometry",
    "survey_name",
    "survey_type",
    "created_utc",
    "original_files",
    "partitions",
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


def _from_dict(raw: Dict[str, Any], path: Path) -> Manifest:
    for key in _REQUIRED_TOP_KEYS:
        _require(raw, key, path)

    fmt = raw["oneuniverse_format_version"]
    if not (isinstance(fmt, str) and fmt.startswith("2.")):
        raise ManifestValidationError(
            f"{path}: oneuniverse_format_version={fmt!r} is not compatible "
            f"with this library (expected 2.x)."
        )

    geo = raw["geometry"]
    try:
        geometry = DataGeometry(geo)
    except ValueError as e:
        raise ManifestValidationError(
            f"{path}: unknown geometry {geo!r}"
        ) from e

    original_files = [OriginalFileSpec(**spec) for spec in raw["original_files"]]
    partitions = [
        PartitionSpec(
            name=p["name"],
            n_rows=int(p["n_rows"]),
            sha256=p["sha256"],
            size_bytes=int(p["size_bytes"]),
            stats=PartitionStats(**p.get("stats", {})),
        )
        for p in raw["partitions"]
    ]
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
    )
