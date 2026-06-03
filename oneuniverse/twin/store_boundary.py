"""Run a ForwardEngine across the OUF-Sim *store boundary*.

The generality proof: the orchestrator hands the engine **store paths**, not
in-memory arrays — IC read from a `SimStore` (partial access), product written
back via `ingest_field`. Any engine (linear, PM, or a real external code) that
satisfies the `ForwardEngine` contract plugs in identically; the orchestrator
never touches the engine's internals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.oufsim import SimStore
from oneuniverse.simulation.oufsim.write import ingest_field
from oneuniverse.simulation.selectors import Cube
from oneuniverse.twin.engine import ForwardEngine


def run_engine_to_store(engine: ForwardEngine, *, out_root: Union[str, Path],
                        sim_name: str, cosmo: CosmologySpec, box: float,
                        n_grid: int, ic_store: Optional[Union[str, Path]] = None,
                        z: float = 0.0, seed: int = 0, overwrite: bool = False,
                        **engine_kw) -> Path:
    """IC from ``ic_store`` (if given) → ``engine.forward`` → product store."""
    ic = None
    if ic_store is not None:
        ic, _ = SimStore(ic_store).read_field_box(z, Cube(0, box, 0, box,
                                                          0, box))
    bundle = engine.forward(cosmo=cosmo, box_size=box, n_grid=n_grid, z=z,
                            seed=seed, ic=ic, **engine_kw)
    return ingest_field(out_root, sim_name, cosmo=cosmo, box_size=box,
                        field=bundle.fields["delta"], z=z, overwrite=overwrite)
