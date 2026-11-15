"""Crash-safe, versioned, bitemporal sidecar store.

Layout::

    <root>/_<kind>/<name>/<version>/table.parquet   (zstd)
    <root>/_<kind>/<name>/<version>/meta.json
    <root>/_<kind>/<name>/current.json              {"version": <id>}

A version id is the UTC write time ``YYYY-MM-DDTHH:MM:SS.ffffffZ``; ids sort
lexicographically in chronological order. If the clock does not yield an id
strictly greater than the current one, the id is bumped by 1 microsecond.

A write performs, in order:

1. publish the new version directory (``atomic_dir``): table + meta;
2. swap ``current.json`` (``atomic_write_json``) -- the commit point;
3. patch the previous version's ``meta.json`` with ``valid_to = <new id>``.

Crash semantics:

* Failure during (1): nothing changed; ``current`` is untouched. A hard kill
  may leave a ``<version>.tmp`` / ``.old`` staging dir, which is ignored by
  readers (and makes the next write of that exact id refuse, see ``atomic_dir``).
* Failure at (2): the new directory exists but is an *orphan* (id > current).
  Readers ignore it; the next successful write removes it.
* Failure at (3): the new version is live. The previous meta lacks
  ``valid_to``; readers derive it from the next listed version id, so
  ``load(as_of=...)`` is unaffected.

A version ``v`` is live at ``when`` if ``v.valid_from <= when < v.valid_to``
(``valid_to`` None = open). ``as_of`` must be timezone-aware.
Concurrent writers are not supported.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse._core.errors import FormatError, NotFoundError, ValidationError
from oneuniverse._core.fs import (
    STAGING_SUFFIXES,
    _json_default,
    atomic_dir,
    atomic_write_json,
    is_staging,
)

SIDECAR_FORMAT_VERSION = 1

_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"
_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_SEG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RESERVED = frozenset(
    {"format_version", "kind", "name", "version", "valid_from", "valid_to", "n_rows"})


def _now() -> datetime:
    return datetime.now(UTC)


def _segment(value: str, what: str) -> str:
    if not isinstance(value, str) or not _SEG_RE.match(value):
        raise ValidationError(
            f"invalid {what} {value!r}: use letters, digits, '_', '-', '.', "
            f"not starting with '.' or '_'")
    if value.endswith(STAGING_SUFFIXES):
        raise ValidationError(
            f"invalid {what} {value!r}: must not end with a staging suffix {STAGING_SUFFIXES}")
    return value


def _read_json(path: Path, what: str) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise FormatError(f"cannot read {what} {path}: {exc}") from exc


def _read_meta(path: Path) -> dict:
    meta = _read_json(path, "sidecar meta")
    if not isinstance(meta, dict) or meta.get("format_version") != SIDECAR_FORMAT_VERSION:
        got = meta.get("format_version") if isinstance(meta, dict) else None
        raise FormatError(
            f"unsupported sidecar format_version {got!r} in {path} "
            f"(this version reads {SIDECAR_FORMAT_VERSION})")
    missing = [k for k in ("version", "valid_from") if k not in meta]
    if missing:
        raise FormatError(f"corrupt sidecar meta {path}: missing {missing}")
    return meta


def _fmt(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime(_FMT)


class VersionedSidecar:
    """Versioned table store at ``<root>/_<kind>/<name>/``."""

    def __init__(self, root, kind: str, name: str):
        self.root = Path(root)
        self.kind = _segment(kind, "kind")
        self.name = _segment(name, "name")
        self.dir = self.root / f"_{self.kind}" / self.name

    # ------------------------------------------------------------------ read
    def current(self) -> str | None:
        p = self.dir / "current.json"
        if not p.exists():
            return None
        cur = _read_json(p, "sidecar pointer")
        if not isinstance(cur, dict) or not isinstance(cur.get("version"), str):
            raise FormatError(f"corrupt sidecar pointer {p}: no 'version'")
        return cur["version"]

    def _ids(self) -> list[str]:
        """Version ids <= current, oldest first (orphans and staging ignored)."""
        cur = self.current()
        if cur is None or not self.dir.is_dir():
            return []
        return sorted(
            p.name for p in self.dir.iterdir()
            if p.is_dir() and not is_staging(p) and _ID_RE.match(p.name) and p.name <= cur)

    def _orphans(self) -> list[Path]:
        cur = self.current()
        if not self.dir.is_dir():
            return []
        return [p for p in self.dir.iterdir()
                if p.is_dir() and not is_staging(p) and _ID_RE.match(p.name)
                and (cur is None or p.name > cur)]

    def versions(self) -> list[dict]:
        """Metas of all versions up to current, oldest first (valid_to derived if missing)."""
        ids = self._ids()
        out = []
        for i, vid in enumerate(ids):
            meta = _read_meta(self.dir / vid / "meta.json")
            if meta.get("valid_to") is None and i + 1 < len(ids):
                meta["valid_to"] = ids[i + 1]
            out.append(meta)
        return out

    def meta(self, *, as_of: datetime | None = None) -> dict:
        """Meta of the current version, or of the one live at *as_of* (tz-aware)."""
        if as_of is not None and (as_of.tzinfo is None or as_of.utcoffset() is None):
            raise ValidationError("as_of must be timezone-aware")
        metas = self.versions()
        if not metas:
            raise NotFoundError(f"no version of {self.kind}/{self.name} under {self.root}")
        if as_of is None:
            return metas[-1]
        when = _fmt(as_of)
        live = [m for m in metas
                if m["valid_from"] <= when and (m.get("valid_to") is None or when < m["valid_to"])]
        if not live:
            raise NotFoundError(
                f"no version of {self.kind}/{self.name} is live at {as_of.isoformat()}")
        return live[-1]

    def load(self, *, as_of: datetime | None = None) -> tuple[pa.Table, dict]:
        """Load the current version, or the one live at *as_of* (tz-aware)."""
        meta = self.meta(as_of=as_of)
        table = pq.read_table(self.dir / meta["version"] / "table.parquet")
        return table, meta

    @staticmethod
    def names(root, kind: str) -> list[str]:
        base = Path(root) / f"_{_segment(kind, 'kind')}"
        if not base.is_dir():
            return []
        return sorted(p.name for p in base.iterdir()
                      if p.is_dir() and not is_staging(p) and (p / "current.json").exists())

    # ----------------------------------------------------------------- write
    def write(self, table: pa.Table, meta: dict) -> str:
        """Publish *table* as a new version and return its id."""
        bad = _RESERVED & set(meta)
        if bad:
            raise ValidationError(f"reserved meta keys: {sorted(bad)}")
        try:
            json.dumps(meta, allow_nan=False, default=_json_default)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"meta is not JSON-serialisable: {exc}") from exc

        self.dir.mkdir(parents=True, exist_ok=True)
        for orphan in self._orphans():
            shutil.rmtree(orphan)

        cur = self.current()
        now = _now()
        vid = _fmt(now)
        if cur is not None and vid <= cur:
            vid = _fmt(datetime.strptime(cur, _FMT).replace(tzinfo=UTC)
                       + timedelta(microseconds=1))
        full = {**meta, "format_version": SIDECAR_FORMAT_VERSION, "kind": self.kind,
                "name": self.name, "version": vid, "valid_from": vid, "valid_to": None,
                "n_rows": table.num_rows}

        # (1) new version directory
        with atomic_dir(self.dir / vid) as d:
            pq.write_table(table, d / "table.parquet", compression="zstd")
            atomic_write_json(d / "meta.json", full)
        # (2) commit point
        atomic_write_json(self.dir / "current.json", {"version": vid})
        # (3) close the previous version
        if cur is not None:
            p = self.dir / cur / "meta.json"
            prev = _read_meta(p)
            prev["valid_to"] = vid
            atomic_write_json(p, prev)
        return vid
