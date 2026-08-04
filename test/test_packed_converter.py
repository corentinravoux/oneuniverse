"""S17 T4 — a 2nd backend produces an equivalent store from a different format."""
import numpy as np

from oneuniverse.simulation.converter import detect_converter, get_converter
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.linear.pack import write_packed_native
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.packed.converter import PackedSimConverter  # noqa: F401
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_packed_converter_detects_and_registers(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2,
                              with_lightcone=False)
    pk = write_packed_native(lin, tmp_path / "pk", particle_chunk_nside=4)
    cls = detect_converter(pk)
    assert cls is not None and cls.code == PackedSimConverter.code
    assert get_converter("packed_npy") is PackedSimConverter


def test_packed_store_reads_match_linear(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2,
                              with_lightcone=False)
    pk = write_packed_native(lin, tmp_path / "pk", particle_chunk_nside=4)
    lin_store = write_oufsim_store(lin, tmp_path / "ls", sim_name="d",
                                   particle_chunk_nside=4)
    pk_store = PackedSimConverter().convert(pk, tmp_path / "ps", sim_name="d")
    cube = Cube(20, 90, 20, 90, 20, 90)
    a = SimStore(lin_store).read_box("snapshots", 0.0, cube,
                                     columns=("x", "y", "z"))
    b = SimStore(pk_store).read_box("snapshots", 0.0, cube,
                                    columns=("x", "y", "z"))
    # same set of particles in the cube (order may differ -> compare sorted)
    assert len(a["x"]) == len(b["x"])
    np.testing.assert_allclose(np.sort(a["x"]), np.sort(b["x"]))
    fa, _ = SimStore(lin_store).read_field_box(0.0, cube)
    fb, _ = SimStore(pk_store).read_field_box(0.0, cube)
    assert np.allclose(fa, fb)


def test_packed_store_read_parity_reference_vs_reencode(tmp_path):
    """A packed reference (wrap-in-place) store returns the same sub-box
    particles as a reencode store built from the same native sim."""
    from oneuniverse.simulation.linear.pack import write_packed_native
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2,
                              with_lightcone=False)
    pk = write_packed_native(lin, tmp_path / "pk", particle_chunk_nside=4)
    enc = PackedSimConverter().convert(pk, tmp_path / "enc", sim_name="e",
                                       projection="reencode")
    ref = PackedSimConverter().convert(pk, tmp_path / "rf", sim_name="r",
                                       projection="reference")
    cube = Cube(0, 100, 0, 100, 0, 100)
    a = SimStore(enc).read_box("snapshots", 0.0, cube, columns=("x", "y", "z"))
    b = SimStore(ref).read_box("snapshots", 0.0, cube, columns=("x", "y", "z"))
    assert len(a["x"]) == len(b["x"]) > 0
    np.testing.assert_allclose(np.sort(a["x"]), np.sort(b["x"]))


def test_packed_reference_is_index_only(tmp_path):
    """The packed reference projection copies no bulk data — only the sidecar
    index over the native slab (storage generality: wrap-in-place)."""
    from pathlib import Path
    from oneuniverse.simulation.linear.pack import write_packed_native
    lin = generate_linear_sim(tmp_path / "lin2", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=4,
                              with_lightcone=False)
    pk = write_packed_native(lin, tmp_path / "pk2", particle_chunk_nside=4)
    ref = PackedSimConverter().convert(pk, tmp_path / "rf2", sim_name="r",
                                       projection="reference")
    snap = Path(ref) / "snapshots" / "z0.000"
    assert (snap / "_index.parquet").is_file()      # index present
    assert not list(snap.glob("*.npy"))             # no copied bulk data
