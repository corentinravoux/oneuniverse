"""Phase S6 T3/T6 — Morton row order + predicate pushdown."""
import numpy as np
import pyarrow.parquet as pq

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _store(tmp_path, row_order):
    native = generate_linear_sim(tmp_path / ("n_" + row_order), _cosmo(),
                                 box_size=200.0, n_grid=32, redshifts=(0.0,),
                                 seed=2)
    return write_oufsim_store(native, tmp_path / row_order, sim_name="d",
                              particle_chunk_nside=2, row_order=row_order,
                              row_group_size=200)


def _median_rowgroup_volume(store):
    """Median row-group bounding-box volume in the first snapshot chunk.

    Smaller = tighter 3D clustering = more row-groups skippable by a Cube
    query. (Native Lagrangian order clusters x but spans full y,z; Morton
    clusters all three.)
    """
    s = SimStore(store)
    prod = store / s.layout["snapshots"]["z0.000"]["dir"]
    f = sorted(prod.glob("part_*.parquet"))[0]
    md = pq.ParquetFile(f).metadata
    vols = []
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        rng = {}
        for c in range(rg.num_columns):
            col = rg.column(c)
            if col.path_in_schema in ("x", "y", "z") and col.statistics:
                rng[col.path_in_schema] = col.statistics.max - col.statistics.min
        vols.append(rng["x"] * rng["y"] * rng["z"])
    return float(np.median(vols))


def test_morton_shrinks_rowgroup_volume(tmp_path):
    none = _median_rowgroup_volume(_store(tmp_path, "none"))
    mort = _median_rowgroup_volume(_store(tmp_path, "morton"))
    # Morton clustering -> tighter 3D row-group boxes -> better Cube pruning
    assert mort < 0.7 * none


def test_pushdown_matches_bruteforce(tmp_path):
    store = _store(tmp_path, "morton")
    s = SimStore(store)
    cube = Cube(0, 20, 0, 20, 0, 20)
    brute = s.read_box("snapshots", 0.0, cube, pushdown=False)
    pushed = s.read_box("snapshots", 0.0, cube, pushdown=True)
    assert len(pushed["x"]) == len(brute["x"])
    np.testing.assert_allclose(np.sort(pushed["x"]), np.sort(brute["x"]))
