"""S17 T5 — particle reference projection is index-only and reads match."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.linear.pack import write_packed_native
from oneuniverse.simulation.oufsim import SimStore
from oneuniverse.simulation.packed.converter import PackedSimConverter
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _dir_size(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def test_particle_reference_is_index_only_and_reads_match(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2,
                              with_lightcone=False)
    pk = write_packed_native(lin, tmp_path / "pk", particle_chunk_nside=4)
    enc = PackedSimConverter().convert(pk, tmp_path / "enc", sim_name="d",
                                       projection="reencode")
    ref = PackedSimConverter().convert(pk, tmp_path / "ref", sim_name="e",
                                       projection="reference")
    # snapshots dir of the reference store holds no copied parquet floats
    snap_ref = ref / "snapshots" / "z0.000"
    assert not any(f.suffix == ".parquet" and f.name.startswith("part_")
                   for f in snap_ref.iterdir())
    assert _dir_size(snap_ref) < 0.1 * _dir_size(enc / "snapshots" / "z0.000")
    # reads identical
    cube = Cube(10, 80, 10, 80, 10, 80)
    a = SimStore(enc).read_box("snapshots", 0.0, cube, columns=("x", "y", "z"))
    b = SimStore(ref).read_box("snapshots", 0.0, cube, columns=("x", "y", "z"))
    assert len(a["x"]) == len(b["x"])
    np.testing.assert_allclose(np.sort(a["x"]), np.sort(b["x"]))
