"""DuckDB SQL face over an OUF 3.0 database: views only, no data copied.

``connect(root)`` returns a DuckDB connection holding

* catalog tables ``datasets``, ``partitions``, ``versions`` (small, built at
  attach time from manifests, partition indexes and manifest history);
* one view per dataset over its hive-partitioned parquet (``cell`` is a column);
* ``oneuid_<name>`` / ``links_<name>`` views over the versioned sidecars.

Timestamps in the catalog tables are VARCHAR, ISO-8601 UTC exactly as stored in
the manifests (they sort lexicographically). ``duckdb`` is imported lazily, only
by :func:`connect`; :func:`cone_where` is pure Python.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

from oneuniverse._core.errors import NotFoundError, ValidationError
from oneuniverse._core.geometry import Cone
from oneuniverse._core.healpix import cone_pixels
from oneuniverse.data.database import Database
from oneuniverse.data.manifest import DATA_DIR, INDEX_FILENAME, read_partition_index
from oneuniverse.data.sidecar import VersionedSidecar

__all__ = ["cone_where", "connect"]

_RAD = np.pi / 180.0
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CATALOG = ("datasets", "partitions", "versions")

_DATASETS = pa.schema([
    ("name", pa.string()), ("geometry", pa.string()), ("kind", pa.string()),
    ("survey", pa.string()), ("n_rows", pa.int64()), ("nside", pa.int32()),
    ("format_version", pa.string()), ("created_utc", pa.string()), ("version", pa.int64()),
    ("valid_from", pa.string()), ("valid_to", pa.string()), ("path", pa.string())])
_PARTITIONS = pa.schema([
    ("dataset", pa.string()), ("partition", pa.int32()), ("cell", pa.int32()),
    ("path", pa.string()), ("n_rows", pa.int64()),
    ("ra_min", pa.float64()), ("ra_max", pa.float64()),
    ("dec_min", pa.float64()), ("dec_max", pa.float64()),
    ("z_min", pa.float64()), ("z_max", pa.float64())])
_VERSIONS = pa.schema([
    ("dataset", pa.string()), ("version", pa.int64()), ("created_utc", pa.string()),
    ("valid_from", pa.string()), ("valid_to", pa.string())])

_IMPORT_MSG = "the SQL face needs duckdb: pip install oneuniverse[sql]"


def _import_duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(_IMPORT_MSG) from exc
    return duckdb


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _lit(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def cone_where(cone: Cone, nside: int, *, cell_column: str = "cell",
               ra_column: str = "ra", dec_column: str = "dec") -> str:
    """SQL predicate for cone membership: partition-cell pruning AND closed haversine.

    Mirrors ``view.cone_expr`` (float64, same operation order): ``cell IN (<NEST cover
    at *nside*>)`` then ``2 asin(sqrt(a)) <= radius`` in radians. *nside* is the
    dataset's partition nside (``DatasetView.nside``). An empty cover gives ``FALSE``.
    """
    pix = cone_pixels(nside, cone.ra, cone.dec, cone.radius, nest=True)
    if len(pix) == 0:
        return "FALSE"
    cell, ra, dec = _q(cell_column), _q(ra_column), _q(dec_column)
    ra0, dec0 = float(np.radians(cone.ra)), float(np.radians(cone.dec))
    ra_r = f"(CAST({ra} AS DOUBLE) * {_RAD!r})"
    dec_r = f"(CAST({dec} AS DOUBLE) * {_RAD!r})"
    a = (f"(pow(sin(({dec_r} - {dec0!r}) / 2.0), 2.0) + "
         f"(cos({dec_r}) * {float(np.cos(dec0))!r}) * pow(sin(({ra_r} - {ra0!r}) / 2.0), 2.0))")
    sep = f"(asin(sqrt(greatest(least({a}, 1.0), 0.0))) * 2.0)"
    cells = ", ".join(str(int(p)) for p in pix)
    return f"{cell} IN ({cells}) AND {sep} <= {float(np.radians(cone.radius))!r}"


def _view_name(name: str) -> str:
    return name.replace("__", "_").replace("-", "_")


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(UTC)


def _check_as_of(as_of):
    if as_of is not None and (not isinstance(as_of, datetime) or as_of.tzinfo is None
                              or as_of.utcoffset() is None):
        raise ValidationError(f"as_of needs a timezone-aware datetime, got {as_of!r}")


def _intervals(db: Database, name: str) -> list[tuple]:
    """``(manifest, valid_from, valid_to)`` per version, oldest first (live one open)."""
    ms = db.history(name)
    out = []
    for i, m in enumerate(ms):
        nxt = ms[i + 1].created_utc if i + 1 < len(ms) else None
        out.append((m, m.created_utc, nxt))
    return out


def _register(con, table_name: str, table: pa.Table) -> None:
    con.register("_ou_tmp", table)
    try:
        con.execute(f"CREATE OR REPLACE TABLE {_q(table_name)} AS SELECT * FROM _ou_tmp")
    finally:
        con.unregister("_ou_tmp")


def _sidecar_names(db: Database, kind: str, prefix: str, taken: dict[str, str]) -> dict[str, str]:
    out = {}
    for name in VersionedSidecar.names(db.root, kind):
        view = f"{prefix}_{re.sub(r'[^A-Za-z0-9_]', '_', name)}"
        _claim(taken, view, f"{kind} sidecar {name!r}")
        out[name] = view
    return out


def _claim(taken: dict[str, str], view: str, what: str) -> None:
    if not _IDENT.match(view):
        raise ValidationError(f"{what}: {view!r} is not a valid SQL identifier")
    key = view.lower()
    if key in taken:
        raise ValidationError(f"{what}: view name {view!r} collides with {taken[key]}")
    taken[key] = what


def _drop_stale(con, keep: set[str]) -> None:
    """Drop every view and table of the main schema whose name is not in *keep*."""
    rows = con.execute("SELECT table_name, table_type FROM information_schema.tables "
                       "WHERE table_schema = 'main'").fetchall()
    for name, kind in rows:
        if name not in keep:
            con.execute(f"DROP {'VIEW' if kind == 'VIEW' else 'TABLE'} {_q(name)}")


def connect(root, *, as_of: datetime | None = None, database: str = ":memory:"):
    """DuckDB connection with catalog tables and views over the database at *root*.

    *as_of* (timezone-aware) mirrors ``Database.as_of``: a dataset is kept when its
    live manifest was created at or before *as_of*; a dataset whose live version
    postdates *as_of* gets no view and no ``datasets`` row (its archived rows stay in
    ``versions``, whose data files are not retained). Sidecars (``oneuid_*``,
    ``links_*``) select the version live at *as_of* and are always built, so an as_of
    sidecar may refer to rows of a dataset version that is no longer readable.

    ``oneuid_*.partition`` is the partition ordinal (row number in the dataset's
    ``_index.parquet``), not the HEALPix cell: join it to ``partitions.partition``
    (with ``dataset``), not to ``partitions.cell``. ``partition`` and ``offset`` are
    SQL keywords: quote them (``"partition"``, ``"offset"``). ``source_row`` is the
    recommended join key to a dataset view.

    Glob characters (``[``, ``*``, ``?``) in *root* are interpreted by DuckDB in the
    dataset view paths: avoid them. With a persistent *database* file, every view and
    table of the main schema that this call does not recreate is dropped; an
    in-memory database starts empty. Returns the real ``duckdb.DuckDBPyConnection``.
    """
    duckdb = _import_duckdb()
    _check_as_of(as_of)
    root = Path(root)
    db = Database(root)

    taken: dict[str, str] = {t: f"catalog table {t!r}" for t in _CATALOG}
    chosen = {}
    for name in (db.as_of(as_of) if as_of is not None else db).names():
        _claim(taken, _view_name(name), f"dataset {name!r}")
        chosen[name] = _intervals(db, name)[-1]
    oneuids = _sidecar_names(db, "oneuid", "oneuid", taken)
    links = _sidecar_names(db, "links", "links", taken)

    ds_rows, part_rows, ver_rows = [], [], []
    for name, (m, vfrom, vto) in chosen.items():
        p = db.path(name)
        ds_rows.append({
            "name": name, "geometry": m.geometry.value, "kind": m.survey.kind,
            "survey": m.survey.name, "n_rows": int(m.n_rows), "nside": m.partitioning.nside,
            "format_version": m.format_version, "created_utc": m.created_utc,
            "version": int(m.validity.version), "valid_from": vfrom, "valid_to": vto,
            "path": str(p)})
        idx = read_partition_index(p / INDEX_FILENAME).to_pydict()
        n = len(idx["cell"])
        part_rows.append(pa.table({
            "dataset": [name] * n, "partition": list(range(n)), "cell": idx["cell"],
            "path": [str(p / x) for x in idx["path"]], "n_rows": idx["n_rows"],
            **{c: idx[c] for c in ("ra_min", "ra_max", "dec_min", "dec_max", "z_min", "z_max")},
        }, schema=_PARTITIONS))
    for name in db.names():
        for hm, hfrom, hto in _intervals(db, name):
            if as_of is not None and _utc(hfrom) > as_of:
                continue
            ver_rows.append({"dataset": name, "version": int(hm.validity.version),
                             "created_utc": hm.created_utc, "valid_from": hfrom,
                             "valid_to": hto})

    con = duckdb.connect(database)
    created = set(_CATALOG)
    _register(con, "datasets", pa.Table.from_pylist(ds_rows, schema=_DATASETS))
    _register(con, "partitions",
              pa.concat_tables(part_rows) if part_rows else _PARTITIONS.empty_table())
    _register(con, "versions", pa.Table.from_pylist(ver_rows, schema=_VERSIONS))

    for name in chosen:
        glob = db.path(name) / DATA_DIR / "*" / "*.parquet"
        created.add(_view_name(name))
        con.execute(f"CREATE OR REPLACE VIEW {_q(_view_name(name))} AS SELECT * FROM "
                    f"read_parquet({_lit(str(glob))}, hive_partitioning = true)")

    def sidecar_views(kind: str, names: dict[str, str], select: str) -> None:
        for name, view in names.items():
            sc = VersionedSidecar(root, kind, name)
            try:
                meta = sc.meta(as_of=as_of)
            except NotFoundError:
                continue  # did not exist at as_of
            table = sc.dir / meta["version"] / "table.parquet"
            created.add(view)
            con.execute(f"CREATE OR REPLACE VIEW {_q(view)} AS SELECT {select} "
                        f"FROM read_parquet({_lit(str(table))})")

    sidecar_views("oneuid", oneuids,
                  "oneuid, CAST(dataset AS VARCHAR) AS dataset, source_row, partition, "
                  '"offset", ra, dec, z, CAST(z_type AS VARCHAR) AS z_type')
    sidecar_views("links", links,
                  "parent_oneuid, child_oneuid, sep_arcsec, dz, confidence")
    if database != ":memory:":
        _drop_stale(con, created)
    return con
