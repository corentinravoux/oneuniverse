"""SimDatabase — the OUF-Sim control plane (orchestration).

Catalogs OUF-Sim stores, turns a region selection into a
``SimulationRequest``, dispatches the *dummy* resimulation (Rule 4 relaxed
for the fast-PM dummy — heavy real-code runs stay future), and records the
parent→child lineage. This is the bookkeeping that drives the
extract→run→merge→verify loop.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.oufsim._io import read_json
from oneuniverse.simulation.region import RegionSpec
from oneuniverse.simulation.request import SimulationRequest
from oneuniverse.simulation.resim.coupling import run_coupled


class SimDatabase:
    """A catalog + orchestration layer over a directory of OUF-Sim stores."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.catalog: Dict[str, dict] = {}
        self.requests: List[SimulationRequest] = []
        self.lineage: List[dict] = []

    # -- discovery --------------------------------------------------------
    def scan(self) -> "SimDatabase":
        for mf in self.root.glob("*/oufsim/manifest.json"):
            man = read_json(mf)
            name = man["sim_name"]
            seed = None
            ckpt = mf.parent / "checkpoints" / "descriptor.json"
            if ckpt.is_file():
                seed = read_json(ckpt).get("seed")
            self.catalog[name] = {
                "box_size": man.get("box_size"),
                "n_grid": man.get("n_grid"),
                "products": tuple(man.get("products", ())),
                "cosmology": man.get("cosmology"),
                "seed": seed, "store": mf.parent,
            }
        return self

    def sim_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self.catalog))

    def get(self, name: str) -> dict:
        return self.catalog[name]

    # -- region selection -> request --------------------------------------
    def request_region(self, parent: str, *, target_lo: float,
                       target_side: float, buffer: float,
                       physics: Tuple[str, ...] = ("dm",),
                       ic_strategy: str = "zoom_from_parent_ic"
                       ) -> SimulationRequest:
        rec = self.catalog[parent]
        thi = target_lo + target_side
        patch = (target_lo, thi, target_lo, thi, target_lo, thi)
        region = RegionSpec(region_id=f"{parent}_region",
                            kind="lagrangian", lagrangian_patch=patch)
        req = SimulationRequest(
            request_id=f"req_{len(self.requests):04d}",
            parent_sim=parent, region=region,
            target_resolution=float(rec["n_grid"]), physics=physics,
            cosmology=CosmologySpec.from_dict(rec["cosmology"]),
            ic_strategy=ic_strategy, status="pending",
            provenance={"buffer": float(buffer), "target_side": target_side},
        )
        self.requests.append(req)
        return req

    # -- dispatch the dummy resimulation ----------------------------------
    def dispatch(self, request: SimulationRequest, *, z_start: float = 9.0,
                 z_end: float = 0.0, n_steps: int = 15,
                 ic_field: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, str]:
        """Run the dummy resimulation for ``request``.

        ``ic_field`` (a z=0 density field, e.g. a data-driven constrained
        realization) makes this the **data-driven** path: the resim starts
        from the data-informed IC rather than a fresh seed. The IC provenance
        is recorded on the lineage edge.
        """
        rec = self.catalog[request.parent_sim]
        lp = request.region.lagrangian_patch
        tlo, thi = lp[0], lp[1]
        ic_source = ("constrained_from_posterior" if ic_field is not None
                     else "fresh_seed")
        res = run_coupled(
            request.cosmology, box=rec["box_size"], n_grid=rec["n_grid"],
            target_lo=tlo, target_side=thi - tlo,
            buffer=request.provenance["buffer"], z_start=z_start, z_end=z_end,
            seed=int(rec["seed"]) if rec["seed"] is not None else 0,
            ic_field=ic_field, n_steps=n_steps,
        )
        child = f"{request.parent_sim}_zoom"
        self.lineage.append({"parent": request.parent_sim, "child": child,
                             "region": request.region.region_id,
                             "ic_source": ic_source})
        self._set_status(request, "ingested")
        return res["inner"], child

    def _set_status(self, request: SimulationRequest, status: str) -> None:
        for i, r in enumerate(self.requests):
            if r.request_id == request.request_id:
                self.requests[i] = dataclasses.replace(r, status=status)

    # -- lineage ----------------------------------------------------------
    def children_of(self, name: str) -> List[str]:
        return [e["child"] for e in self.lineage if e["parent"] == name]

    def parent_of(self, child: str) -> Optional[str]:
        for e in self.lineage:
            if e["child"] == child:
                return e["parent"]
        return None
