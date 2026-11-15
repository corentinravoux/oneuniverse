"""DuckDB SQL face over OUF-Sim stores: tables and views only, no data copied.

``connect(store_root)`` returns a DuckDB connection holding

* catalog tables ``sims``, ``products``, ``chunks``, ``tiles`` (small, built at attach
  time from manifests, ``layout.json`` and the ``_index.parquet`` files);
* one view per catalog product in the ``chunks`` layout and per redshift, over its
  parquet chunk files.

``store_root`` is either one store (a directory holding ``manifest.json``) or a
``SimDatabase`` root (subdirectories holding ``manifest.json``; ``_``-prefixed names such
as ``_simdb`` and staging directories are skipped). Nothing is written. Only committed
``layout.json`` entries are exposed. Products in the ``reference`` projection (rows live
in the native files) and ``sky`` products get catalog rows but no view. Timestamps are
VARCHAR ISO-8601 UTC as stored. ``duckdb`` is imported lazily, only by :func:`connect`;
:func:`box_where` is pure Python.
"""
from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse._core.errors import NotFoundError, ValidationError
from oneuniverse._core.fs import is_staging
from oneuniverse._core.geometry import Cube
from oneuniverse.simulation.manifest import MANIFEST_FILENAME
from oneuniverse.simulation.store.index import split_periodic
from oneuniverse.simulation.store.read import INDEX_FILENAME, SimStore

__all__ = ["box_where", "connect"]

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CATALOG = ("sims", "products", "chunks", "tiles")
_COUNT_KEYS = ("n_chunks", "n_tiles", "n_pix")
_IMPORT_MSG = "the SQL face needs duckdb: pip install oneuniverse[sql]"

_SIMS = pa.schema([
    ("name", pa.string()), ("box_size", pa.float64()), ("h", pa.float64()),
    ("omega_m", pa.float64()), ("sigma8", pa.float64()), ("code", pa.string()),
    ("created_utc", pa.string()), ("parent", pa.string())])
_PRODUCTS = pa.schema([
    ("sim", pa.string()), ("name", pa.string()), ("kind", pa.string()), ("z", pa.float64()),
    ("layout", pa.string()), ("n_chunks", pa.int64())])
_CHUNKS = pa.schema(
    [("sim", pa.string()), ("product", pa.string()), ("z", pa.float64()),
     ("chunk_id", pa.int32()), ("path", pa.string()),
     ("row_start", pa.int64()), ("row_stop", pa.int64())]
    + [(f"{a}_{m}", pa.float64()) for a in "xyz" for m in ("min", "max")])
_TILES = pa.schema(
    [("sim", pa.string()), ("product", pa.string()), ("z", pa.float64()),
     ("tile_id", pa.int32()), ("path", pa.string())]
    + [(f"{a}{e}", pa.int32()) for a in "ijk" for e in "01"])


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


def _box(part: Cube) -> str:
    terms = []
    for a, lo, hi in zip("xyz", part.lo, part.hi):
        terms.append(f"{a}_min <= {float(hi)!r} AND {a}_max >= {float(lo)!r}")
    return " AND ".join(terms)


def box_where(cube: Cube, box_size: float | None = None) -> str:
    """SQL predicate on the ``chunks`` table: closed chunk bbox intersects *cube*.

    ``x_min <= hi AND x_max >= lo`` per axis, the same closed inequality as
    ``store.index.overlapping_chunks``. Without *box_size* the cube is used as given:
    periodic wrap is not handled, so a cube crossing the box boundary must first be
    split with ``split_periodic``. With *box_size*, the cube is split exactly as
    ``overlapping_chunks`` does and the parts are OR-ed, giving the same chunk set.
    """
    if box_size is None:
        return _box(cube)
    parts = [_box(p) for p in split_periodic(cube, box_size)]
    return parts[0] if len(parts) == 1 else "(" + " OR ".join(f"({p})" for p in parts) + ")"


def _stores(root: Path) -> dict[str | None, Path]:
    """``{sim name: store dir}``: the root itself, or its store subdirectories."""
    if not root.is_dir():
        raise NotFoundError(f"no simulation store or database at {root}")
    if (root / MANIFEST_FILENAME).is_file():
        return {None: root}  # name read from the manifest
    found = {d.name: d for d in sorted(root.iterdir())
             if d.is_dir() and not d.name.startswith("_") and not is_staging(d)
             and (d / MANIFEST_FILENAME).is_file()}
    if not found:
        raise NotFoundError(f"{root} holds no {MANIFEST_FILENAME} and no store subdirectory")
    return found


def _ztag_ident(tag: str) -> str:
    return tag.replace(".", "_")


def _claim(taken: dict[str, str], view: str, what: str) -> None:
    if not _IDENT.match(view):
        raise ValidationError(f"{what}: {view!r} is not a valid SQL identifier")
    key = view.lower()
    if key in taken:
        raise ValidationError(f"{what}: view name {view!r} collides with {taken[key]}")
    taken[key] = what


def _register(con, name: str, table: pa.Table) -> None:
    con.register("_ou_tmp", table)
    try:
        con.execute(f"CREATE OR REPLACE TABLE {_q(name)} AS SELECT * FROM _ou_tmp")
    finally:
        con.unregister("_ou_tmp")


def _cat(rows: list[pa.Table], schema: pa.Schema) -> pa.Table:
    return pa.concat_tables(rows) if rows else schema.empty_table()


def _drop_stale(con, keep: set[str]) -> None:
    """Drop every view and table of the main schema whose name is not in *keep*."""
    rows = con.execute("SELECT table_name, table_type FROM information_schema.tables "
                       "WHERE table_schema = 'main'").fetchall()
    for name, kind in rows:
        if name not in keep:
            con.execute(f"DROP {'VIEW' if kind == 'VIEW' else 'TABLE'} {_q(name)}")


def connect(store_root, *, database: str = ":memory:"):
    """DuckDB connection with catalog tables and chunk views over *store_root*.

    View names: a single store gives ``<product>_z<tag>`` (``halos_z0_000``: the redshift
    tag ``0.000`` with ``.`` replaced by ``_``); a ``SimDatabase`` root gives
    ``<sim>__<product>_z<tag>`` (characters outside ``[A-Za-z0-9_]`` become ``_``).
    Invalid identifiers and collisions raise ``ValidationError``. Chunks with no rows
    (empty path) are in ``chunks`` but not in the views, and a product with no rows at
    all gets no view. ``products.n_chunks`` holds the product's ``n_chunks``, ``n_tiles``
    or ``n_pix`` (whichever its layout entry has). Glob characters (``[``, ``*``, ``?``)
    in the store paths are interpreted by DuckDB in the view file lists: avoid them.
    With a persistent *database* file, every view and table of the main schema that this
    call does not recreate is dropped; an in-memory database starts empty. Returns the
    real ``duckdb.DuckDBPyConnection``.
    """
    duckdb = _import_duckdb()
    root = Path(store_root)
    stores = _stores(root)
    multi = None not in stores

    taken: dict[str, str] = {t: f"catalog table {t!r}" for t in _CATALOG}
    sims, products, chunks, tiles, views = [], [], [], [], {}
    for key, d in stores.items():
        st = SimStore(d)
        m = st.manifest
        sim = key if multi else m.name
        sims.append({"name": sim, "box_size": float(m.box_size), "h": float(m.cosmology.h),
                     "omega_m": float(m.cosmology.omega_m),
                     "sigma8": float(m.cosmology.sigma8), "code": m.provenance.code,
                     "created_utc": m.created_utc, "parent": m.parent})
        kinds = {p.name: p.kind for p in m.products}
        for pname in st.products:
            for tag, entry in st.layout["products"][pname].items():
                z = float(tag)
                n = next((int(entry[k]) for k in _COUNT_KEYS if k in entry), None)
                products.append({"sim": sim, "name": pname, "kind": kinds[pname], "z": z,
                                 "layout": entry["layout"], "n_chunks": n})
                idx = pq.read_table(d / entry["dir"] / INDEX_FILENAME).to_pydict()
                n_rows = len(next(iter(idx.values())))
                base = d / entry["dir"] if entry["layout"] != "reference" else d
                paths = [str(base / p) if p else "" for p in idx["path"]]
                head = {"sim": [sim] * n_rows, "product": [pname] * n_rows, "z": [z] * n_rows}
                if entry["kind"] == "field" and "tile_id" in idx:
                    tiles.append(pa.table({**head, **idx, "path": paths}, schema=_TILES))
                elif "chunk_id" in idx:
                    chunks.append(pa.table({**head, **idx, "path": paths}, schema=_CHUNKS))
                if entry["layout"] == "chunks":
                    stem = f"{pname}_z{_ztag_ident(tag)}"
                    view = re.sub(r"[^A-Za-z0-9_]", "_", f"{sim}__{stem}") if multi else stem
                    _claim(taken, view, f"product {pname!r} z={tag} of sim {sim!r}")
                    views[view] = [p for p in paths if p]

    con = duckdb.connect(database)
    created = set(_CATALOG)
    _register(con, "sims", pa.Table.from_pylist(sims, schema=_SIMS))
    _register(con, "products", pa.Table.from_pylist(products, schema=_PRODUCTS))
    _register(con, "chunks", _cat(chunks, _CHUNKS))
    _register(con, "tiles", _cat(tiles, _TILES))
    for view, files in views.items():
        if not files:
            continue
        created.add(view)
        flist = ", ".join(_lit(p) for p in files)
        con.execute(f"CREATE OR REPLACE VIEW {_q(view)} AS SELECT * FROM read_parquet([{flist}])")
    if database != ":memory:":
        _drop_stale(con, created)
    return con
