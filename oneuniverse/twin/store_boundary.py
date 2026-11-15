"""Run a :class:`ForwardEngine` across the OUF-Sim store boundary.

The engine sees arrays; the orchestrator moves them through stores: the initial
conditions are read from a ``SimStore`` and every product of the bundle is written
as a field product of a new store (:func:`create_field_store`), with parent and
lineage recorded in the manifest.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import oneuniverse
from oneuniverse._core.errors import ValidationError
from oneuniverse._core.geometry import Cube
from oneuniverse.simulation.execution import ExecutionPlan
from oneuniverse.simulation.manifest import OUFSIM_FORMAT_VERSION, OUFSimManifest
from oneuniverse.simulation.specs import ProvenanceSpec, UnitFrameSpec
from oneuniverse.simulation.store.build import create_field_store
from oneuniverse.simulation.store.read import SimStore
from oneuniverse.twin.engine import ForwardEngine


def run_engine_to_store(engine: ForwardEngine, *, store_root, name: str, cosmo, box: float,
                        n_grid: int, z: float,
                        ic_from: tuple[SimStore, str, float] | None = None,
                        seed: int | None = None, plan: ExecutionPlan | None = None) -> SimStore:
    """``engine.forward`` -> new store ``Path(store_root) / name`` (one field product per bundle field).

    ``ic_from=(store, product, z_ic)`` reads the initial conditions from a store through
    ``SimStore.read_field_box`` over the WHOLE box (an explicit whole-box read: an
    engine needs the full periodic IC); the new manifest then has
    ``parent = store.root.name`` and ``lineage = {"ic_from": {"store", "product", "z"}}``.
    The source box must equal ``box``. ``ValidationError`` if ``engine`` is not a
    ``ForwardEngine`` or the target exists.
    """
    if not isinstance(engine, ForwardEngine):
        raise ValidationError(f"{type(engine).__name__} is not a ForwardEngine")
    target = Path(store_root) / name
    if target.exists():
        raise ValidationError(f"store {target} already exists")
    ic = parent = None
    lineage: dict = {}
    if ic_from is not None:
        src, product, z_ic = ic_from
        if abs(src.manifest.box_size - float(box)) > 1e-9 * float(box):
            raise ValidationError(
                f"ic store box {src.manifest.box_size} differs from requested box {box}")
        ic = np.asarray(src.read_field_box(product, z_ic, Cube((0.0, 0.0, 0.0), (float(box),) * 3)),
                        dtype=np.float64)
        parent = src.root.name
        lineage = {"ic_from": {"store": parent, "product": product, "z": float(z_ic)}}
    bundle = engine.forward(cosmo=cosmo, box=float(box), n_grid=int(n_grid), z=float(z),
                            seed=seed, ic_field=ic)
    call = {"engine": engine.name, "box": float(box), "n_grid": int(n_grid), "z": float(z),
            "seed": seed, "ic_from": lineage.get("ic_from"), "cosmology": cosmo.to_dict()}
    chash = hashlib.sha256(json.dumps(call, sort_keys=True, default=str).encode()).hexdigest()[:16]
    prov = ProvenanceSpec(engine.name, oneuniverse.__version__, seed, chash,
                          {"engine_provenance": bundle.provenance})
    manifest = OUFSimManifest(OUFSIM_FORMAT_VERSION, name, float(box), cosmo, UnitFrameSpec(), prov, (),
                              {"format": "ingested", "projection": "reencode"},
                              datetime.now(UTC).isoformat(timespec="seconds"),
                              parent=parent, lineage=lineage)
    return create_field_store(target, manifest=manifest, fields=bundle.fields, z=bundle.z, plan=plan)
