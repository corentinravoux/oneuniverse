"""S17 T3 — format-agnostic build_store reproduces the linear store reads."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.oufsim.build import NativeProduct, build_store
from oneuniverse.simulation.selectors import Cube
from oneuniverse.simulation.unit_frame import UnitFrameSpec


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_build_store_matches_write_oufsim_store(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2,
                              with_lightcone=False)
    ref_store = write_oufsim_store(lin, tmp_path / "ref", sim_name="d",
                                   particle_chunk_nside=4)
    parts = np.load(lin / "z0.000" / "particles.npy")
    field = np.load(lin / "z0.000" / "field.npy")
    products = [
        NativeProduct(name="snapshots", kind="catalog", z=0.0,
                      load=lambda parts=parts: {
                          "x": parts[:, 0], "y": parts[:, 1], "z": parts[:, 2],
                          "vx": parts[:, 3], "vy": parts[:, 4], "vz": parts[:, 5]},
                      columns=("x", "y", "z", "vx", "vy", "vz"), n_side=4),
        NativeProduct(name="fields", kind="field", z=0.0,
                      load=lambda field=field: field),
    ]
    built = build_store(tmp_path / "built", sim_name="d", cosmo=_cosmo(),
                        unit_frame=UnitFrameSpec(length_unit="Mpc/h",
                            mass_unit="Msun/h", velocity_unit="km/s peculiar",
                            frame="box"),
                        box_size=200.0, n_grid=32, redshifts=(0.0,),
                        products=products, code="test.builder")
    cube = Cube(0, 60, 0, 60, 0, 60)
    a = SimStore(ref_store).read_box("snapshots", 0.0, cube)
    b = SimStore(built).read_box("snapshots", 0.0, cube)
    assert len(a["x"]) == len(b["x"])
    fa, _ = SimStore(ref_store).read_field_box(0.0, cube)
    fb, _ = SimStore(built).read_field_box(0.0, cube)
    assert np.allclose(fa, fb)
