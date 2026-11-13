"""OUF 3.0 manifest dataclasses, JSON (de)serialisation and partition index I/O."""
from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse._core.errors import FormatError
from oneuniverse._core.fs import atomic_write_bytes, atomic_write_json
from oneuniverse.data.schema import Geometry

FORMAT_VERSION = "3.0.0"
MANIFEST_FILENAME = "manifest.json"
INDEX_FILENAME = "_index.parquet"
DATA_DIR = "data"
DATASET_UUID_KEY = "oneuniverse.dataset_uuid"  # parquet key-value metadata of every data file
EXTENSION_VERSIONS: dict[str, int] = {
    "coordinate": 1, "spectrum": 1, "temporal": 1, "pdf": 1}

INDEX_SCHEMA = pa.schema([
    ("path", pa.string()), ("cell", pa.int32()), ("n_rows", pa.int64()),
    ("ra_min", pa.float64()), ("ra_max", pa.float64()),
    ("dec_min", pa.float64()), ("dec_max", pa.float64()),
    ("z_min", pa.float64()), ("z_max", pa.float64()),
])


@dataclass(frozen=True, slots=True)
class SurveyMeta:
    name: str
    kind: str
    description: str = ""
    reference: str = ""
    url: str = ""


@dataclass(frozen=True, slots=True)
class LoaderInfo:
    name: str
    version: str
    kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OriginalFile:
    path: str
    sha256: str
    n_rows: int | None = None


@dataclass(frozen=True, slots=True)
class Validity:
    valid_from: str
    valid_to: str | None
    version: int


@dataclass(frozen=True, slots=True)
class Partitioning:
    nside: int
    nest: bool = True


def _require(d: Mapping, keys: tuple[str, ...], where: str) -> None:
    for k in keys:
        if k not in d:
            raise FormatError(f"manifest {where} is missing required key {k!r}")


def _schema_to_dict(schema: pa.Schema) -> dict:
    return {
        "ipc_base64": base64.b64encode(schema.serialize().to_pybytes()).decode("ascii"),
        "columns": [{"name": f.name, "type": str(f.type)} for f in schema],
    }


def _schema_from_dict(d: Mapping) -> pa.Schema:
    if "ipc_base64" not in d:
        raise FormatError("manifest schema is missing required key 'ipc_base64'")
    try:
        raw = base64.b64decode(d["ipc_base64"], validate=True)
        return pa.ipc.read_schema(pa.py_buffer(raw))
    except (ValueError, pa.ArrowInvalid) as e:
        raise FormatError(f"manifest schema cannot be decoded: {e}") from e


def _check_extensions(extensions: Mapping[str, Mapping]) -> None:
    for name, ext in extensions.items():
        if not isinstance(ext, Mapping) or "version" not in ext:
            raise FormatError(f"extension {name!r} must carry a 'version' key")
        v = ext["version"]
        if isinstance(v, bool) or not isinstance(v, int):
            raise FormatError(f"extension {name!r} version must be an int, got {v!r}")
        if name in EXTENSION_VERSIONS and v != EXTENSION_VERSIONS[name]:
            raise FormatError(
                f"extension {name!r} version {v} unsupported "
                f"(expected {EXTENSION_VERSIONS[name]})")


_REQUIRED = ("format_version", "geometry", "survey", "loader", "created_utc",
             "schema", "partitioning", "original_files", "n_rows", "validity",
             "dataset_uuid", "extensions")


@dataclass(frozen=True, slots=True, eq=False)
class Manifest:
    """Dataset manifest. ``dataset_uuid`` identifies one publish: the writer stamps it
    into every data file, so readers detect a dataset rewritten under an open view.
    Not hashable (carries a schema and mutable dicts);
    equality compares the Arrow schema with ``Schema.equals``."""

    format_version: str
    geometry: Geometry
    survey: SurveyMeta
    loader: LoaderInfo
    created_utc: str
    schema: pa.Schema
    partitioning: Partitioning
    original_files: tuple[OriginalFile, ...]
    n_rows: int
    validity: Validity
    dataset_uuid: str
    extensions: dict[str, dict]

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _check_extensions(self.extensions)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Manifest):
            return NotImplemented
        for f in Manifest.__dataclass_fields__:
            a, b = getattr(self, f), getattr(other, f)
            if f == "schema":
                if not a.equals(b):
                    return False
            elif a != b:
                return False
        return True

    def extension(self, name: str) -> dict | None:
        return self.extensions.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "geometry": Geometry(self.geometry).value,
            "survey": {"name": self.survey.name, "kind": self.survey.kind,
                       "description": self.survey.description,
                       "reference": self.survey.reference, "url": self.survey.url},
            "loader": {"name": self.loader.name, "version": self.loader.version,
                       "kwargs": dict(self.loader.kwargs)},
            "created_utc": self.created_utc,
            "schema": _schema_to_dict(self.schema),
            "partitioning": {"nside": self.partitioning.nside,
                             "nest": self.partitioning.nest},
            "original_files": [
                {"path": o.path, "sha256": o.sha256, "n_rows": o.n_rows}
                for o in self.original_files],
            "n_rows": self.n_rows,
            "validity": {"valid_from": self.validity.valid_from,
                         "valid_to": self.validity.valid_to,
                         "version": self.validity.version},
            "dataset_uuid": self.dataset_uuid,
            "extensions": {k: dict(v) for k, v in self.extensions.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Manifest:
        _require(d, ("format_version",), "")
        fv = str(d["format_version"])
        try:
            major = int(fv.split(".")[0])
        except ValueError as e:
            raise FormatError(f"unparseable manifest format_version {fv!r}") from e
        if major != 3:
            raise FormatError(
                f"unsupported OUF format_version {fv!r}; this reader handles 3.x only")
        _require(d, _REQUIRED, "")
        _require(d["survey"], ("name", "kind"), "survey")
        _require(d["loader"], ("name", "version"), "loader")
        _require(d["partitioning"], ("nside",), "partitioning")
        _require(d["validity"], ("valid_from", "valid_to", "version"), "validity")
        try:
            geometry = Geometry(d["geometry"])
        except ValueError as e:
            raise FormatError(f"unknown geometry {d['geometry']!r}") from e
        s, lo, p, v = d["survey"], d["loader"], d["partitioning"], d["validity"]
        files = []
        for o in d["original_files"]:
            _require(o, ("path", "sha256"), "original_files entry")
            files.append(OriginalFile(o["path"], o["sha256"], o.get("n_rows")))
        return cls(
            format_version=fv, geometry=geometry,
            survey=SurveyMeta(s["name"], s["kind"], s.get("description", ""),
                              s.get("reference", ""), s.get("url", "")),
            loader=LoaderInfo(lo["name"], lo["version"], dict(lo.get("kwargs", {}))),
            created_utc=d["created_utc"],
            schema=_schema_from_dict(d["schema"]),
            partitioning=Partitioning(p["nside"], p.get("nest", True)),
            original_files=tuple(files), n_rows=d["n_rows"],
            validity=Validity(v["valid_from"], v["valid_to"], v["version"]),
            dataset_uuid=str(d["dataset_uuid"]),
            extensions={k: dict(e) for k, e in d["extensions"].items()},
        )


def write_manifest(path: str | Path, m: Manifest) -> None:
    """Atomically write *m* as JSON."""
    atomic_write_json(path, m.to_dict())


def read_manifest(path: str | Path) -> Manifest:
    """Read a manifest; FormatError on bad JSON, non-3 major or missing keys."""
    path = Path(path)
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise FormatError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(d, dict):
        raise FormatError(f"{path} does not contain a JSON object")
    return Manifest.from_dict(d)


def write_partition_index(path: str | Path, table: pa.Table) -> None:
    """Validate required INDEX_SCHEMA columns and atomically write parquet."""
    for f in INDEX_SCHEMA:
        if f.name not in table.schema.names:
            raise FormatError(f"partition index is missing required column {f.name!r}")
        t = table.schema.field(f.name).type
        if t != f.type:
            raise FormatError(
                f"partition index column {f.name!r} has type {t}, expected {f.type}")
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    atomic_write_bytes(path, sink.getvalue().to_pybytes())


def read_partition_index(path: str | Path) -> pa.Table:
    return pq.read_table(path)
