"""SimDatabase: a directory of stores plus persisted region requests and lineage.

Layout under ``root``: one subdirectory per store (holding ``manifest.json``) and
``_simdb/`` with ``requests.json`` and ``lineage.parquet``.  Engines are plain
callables; this module runs no physics.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse._core.errors import FormatError, NotFoundError, ValidationError
from oneuniverse._core.fs import (
    atomic_dir,
    atomic_write_bytes,
    atomic_write_json,
    is_staging,
)
from oneuniverse.simulation.execution import ExecutionPlan
from oneuniverse.simulation.manifest import (
    LAYOUT_FILENAME,
    MANIFEST_FILENAME,
    OUFSimManifest,
    read_manifest,
    write_manifest,
)
from oneuniverse.simulation.specs import RegionSpec, SimulationRequest
from oneuniverse.simulation.store.read import SimStore
from oneuniverse.simulation.store.write import ingest_field

SIMDB_DIRNAME = "_simdb"
REQUESTS_FILENAME = "requests.json"
LINEAGE_FILENAME = "lineage.parquet"
_LINEAGE_SCHEMA = pa.schema([("parent", pa.string()), ("child", pa.string()),
                             ("request_id", pa.string()), ("created_utc", pa.string())])

Engine = Callable[[SimulationRequest, SimStore], "tuple[np.ndarray, float]"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SimDatabase:
    """Stores under ``root`` with persisted requests and a parent-to-child lineage table."""

    def __init__(self, root):
        self.root = Path(root)
        self._stores: dict[str, OUFSimManifest] = {}
        self._requests: list[SimulationRequest] = []
        self._lineage: list[dict] = []

    # ---- discovery and persistence -------------------------------------------------
    @property
    def _dir(self) -> Path:
        return self.root / SIMDB_DIRNAME

    def scan(self) -> SimDatabase:
        """Discover stores under ``root`` and load persisted requests and lineage."""
        found: dict[str, OUFSimManifest] = {}
        if self.root.is_dir():
            for d in sorted(self.root.iterdir()):
                if (not d.is_dir() or d.name.startswith("_") or is_staging(d)
                        or not (d / MANIFEST_FILENAME).is_file()):
                    continue
                found[d.name] = read_manifest(d / MANIFEST_FILENAME)
        self._stores = found
        self.load()
        return self

    def load(self) -> SimDatabase:
        rpath, lpath = self._dir / REQUESTS_FILENAME, self._dir / LINEAGE_FILENAME
        self._requests = []
        if rpath.is_file():
            try:
                self._requests = [SimulationRequest.from_dict(d) for d in json.loads(rpath.read_text())]
            except json.JSONDecodeError as exc:
                raise FormatError(f"{rpath} is not valid JSON: {exc}") from exc
        self._lineage = pq.read_table(lpath).to_pylist() if lpath.is_file() else []
        return self

    def save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self._dir / REQUESTS_FILENAME, [r.to_dict() for r in self._requests])
        sink = pa.BufferOutputStream()
        pq.write_table(self.lineage(), sink)
        atomic_write_bytes(self._dir / LINEAGE_FILENAME, sink.getvalue().to_pybytes())

    # ---- accessors -----------------------------------------------------------------
    def stores(self) -> dict[str, OUFSimManifest]:
        return dict(sorted(self._stores.items()))

    def store(self, name: str) -> SimStore:
        if name not in self._stores:
            raise NotFoundError(f"no store {name!r} in {self.root}; have {sorted(self._stores)}")
        return SimStore(self.root / name)

    def lineage(self) -> pa.Table:
        return pa.Table.from_pylist(self._lineage, schema=_LINEAGE_SCHEMA)

    def requests(self) -> list[SimulationRequest]:
        return list(self._requests)

    # ---- requests ------------------------------------------------------------------
    def request_region(self, parent: str, region: RegionSpec, *, physics: str = "dm",
                       ic_strategy: str = "parent_field") -> SimulationRequest:
        """Record a pending request to re-simulate ``region`` of store ``parent``."""
        if parent not in self._stores:
            raise NotFoundError(f"no store {parent!r} in {self.root}; have {sorted(self._stores)}")
        outer = region.outer
        sides = [h - l for l, h in zip(outer.lo, outer.hi)]
        if not np.allclose(sides, sides[0], rtol=1e-9, atol=0.0):
            raise ValidationError(f"region.outer must be a cube with equal sides, got sides {sides}")
        req = SimulationRequest(uuid.uuid4().hex[:12], parent, region, physics, ic_strategy)
        self._requests.append(req)
        self.save()
        return req

    def _set(self, request: SimulationRequest, *, sync: SimulationRequest | None = None,
             **changes) -> SimulationRequest:
        new = replace(request, **changes)
        self._requests[self._requests.index(self._get(request.id))] = new
        self.save()
        if sync is not None:  # keep the caller's frozen handle in step (e.g. ``req.child``)
            for k, v in changes.items():
                object.__setattr__(sync, k, v)
        return new

    def _get(self, request_id: str) -> SimulationRequest:
        for r in self._requests:
            if r.id == request_id:
                return r
        raise NotFoundError(f"no request {request_id!r} in {self.root}")

    def dispatch(self, request: SimulationRequest, engine: Engine, *,
                 child_name: str | None = None, tile_cells: int = 32) -> SimStore:
        """Run ``engine(request, parent_store)`` and ingest its output as a child store.

        Engine contract: it returns ``(field, z)`` where ``field`` samples
        ``request.region.outer`` on a regular cubic grid.  If ``region.n_grid`` is
        set, ``field.shape`` must be ``(n_grid,) * 3`` (``ValidationError``
        otherwise); if not, any cubic grid is accepted.

        The child store ``child_name`` (default ``<parent>__<request.id>``) is built
        atomically with the parent's cosmology, units and provenance, ``box_size``
        equal to the outer side length, ``parent`` set, and ``lineage`` holding the
        request id and region (``outer.lo`` is the child origin in parent
        coordinates).  On success the request becomes ``ingested`` and a lineage row
        is appended; the passed ``request`` handle is updated in place to match
        (status, child, error).  On any engine or ingest error the request becomes ``failed``
        with ``error`` set, nothing is published, and the exception is re-raised.
        """
        caller = request
        request = self._get(request.id)
        parent = self.store(request.parent)
        name = child_name or f"{request.parent}__{request.id}"
        if name in self._stores or (self.root / name).exists():
            raise ValidationError(f"child store {name!r} already exists in {self.root}")
        pm = parent.manifest
        region = request.region
        outer = region.outer
        request = self._set(request, sync=caller, status="running")
        try:
            field, z = engine(request, parent)
            field = np.asarray(field)
            if region.n_grid is not None and field.shape != (region.n_grid,) * 3:
                raise ValidationError(
                    f"engine field has shape {field.shape}, request needs {(region.n_grid,) * 3}")
            manifest = OUFSimManifest(
                pm.format_version, name, outer.hi[0] - outer.lo[0], pm.cosmology, pm.units,
                replace(pm.provenance, extra={**pm.provenance.extra, "request_id": request.id,
                                              "note": f"resimulation of {request.parent}"}),
                (), {"format": "ingested", "projection": "reencode", "request_id": request.id},
                _now(), parent=request.parent,
                lineage={"request_id": request.id, "region": region.to_dict()})
            with atomic_dir(self.root / name) as tmp:
                atomic_write_json(tmp / LAYOUT_FILENAME, {"products": {}})
                write_manifest(tmp / MANIFEST_FILENAME, manifest)
                ingest_field(tmp, "field", z, field, tile_cells=tile_cells, plan=ExecutionPlan())
        except BaseException as exc:
            self._set(request, sync=caller, status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        self._stores[name] = read_manifest(self.root / name / MANIFEST_FILENAME)
        self._lineage.append({"parent": request.parent, "child": name, "request_id": request.id,
                              "created_utc": _now()})
        self._set(request, sync=caller, status="ingested", child=name)
        return SimStore(self.root / name)
