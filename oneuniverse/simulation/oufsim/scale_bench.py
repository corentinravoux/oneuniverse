"""Scale-sweep: convert + read wall/peak vs grid size, and store size by
projection. Returns plain dicts so a script can plot + a test can assert
bounded growth (Rule 5)."""
from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Sequence

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.linear.pack import write_packed_native
from oneuniverse.simulation.oufsim import SimStore
from oneuniverse.simulation.packed.converter import PackedSimConverter
from oneuniverse.simulation.selectors import Cube


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def sweep(tmp: Path, cosmo: CosmologySpec, grids: Sequence[int],
          *, box: float = 300.0) -> List[Dict]:
    tmp = Path(tmp)
    out = []
    for ng in grids:
        lin = generate_linear_sim(tmp / f"lin{ng}", cosmo, box_size=box,
                                  n_grid=ng, redshifts=(0.0,), seed=2,
                                  with_lightcone=False)
        pk = write_packed_native(lin, tmp / f"pk{ng}", particle_chunk_nside=4)
        tracemalloc.start(); t0 = time.perf_counter()
        enc = PackedSimConverter().convert(pk, tmp / f"enc{ng}", sim_name="d",
                                           projection="reencode")
        wall = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        ref = PackedSimConverter().convert(pk, tmp / f"ref{ng}", sim_name="e",
                                           projection="reference")
        cube = Cube(0, box / 4, 0, box / 4, 0, box / 4)
        SimStore(enc).read_box("snapshots", 0.0, cube)
        out.append({
            "n_grid": ng, "n_particles": ng ** 3,
            "convert_wall_s": round(wall, 4), "convert_peak_mb": peak / 1e6,
            "store_reencode_mb": _dir_size(enc) / 1e6,
            "store_reference_mb": _dir_size(ref) / 1e6,
            "native_mb": _dir_size(pk) / 1e6,
        })
    return out
