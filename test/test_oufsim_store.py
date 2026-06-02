"""Phase S4 spike — OUF-Sim store round-trip + partial access."""
import numpy as np

from oneuniverse.simulation import detect_converter, get_converter
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import LinearSimConverter, generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.selectors import Cone, Cube


def _cosmo():
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
        t_cmb=2.7255,
    )


def _make_store(tmp_path):
    native = generate_linear_sim(
        tmp_path / "native", _cosmo(),
        box_size=200.0, n_grid=32, redshifts=(0.0, 0.5), seed=3,
    )
    store = write_oufsim_store(
        native, tmp_path / "store", sim_name="demo",
        particle_chunk_nside=4, field_tile_cells=16, lightcone_nside_part=2,
    )
    return native, store


def test_manifest_and_products(tmp_path):
    _, store = _make_store(tmp_path)
    s = SimStore(store)
    assert (store / "manifest.json").is_file()
    assert set(s.products) >= {"snapshots", "fields", "halos", "lightcone"}
    assert s.manifest["oufsim_format_version"].startswith("0.1")


def test_read_box_is_subset(tmp_path):
    _, store = _make_store(tmp_path)
    s = SimStore(store)
    cube = Cube(0.0, 50.0, 0.0, 50.0, 0.0, 50.0)
    sel = s.read_box("snapshots", 0.0, cube)
    # all returned points inside the cube
    assert sel["x"].max() <= 50.0 and sel["x"].min() >= 0.0
    assert sel["y"].max() <= 50.0 and sel["z"].max() <= 50.0
    # partial access actually pruned chunks
    assert s.last_read_stats["chunks_read"] < s.last_read_stats["chunks_total"]
    assert len(sel["x"]) > 0


def test_read_field_box_matches_full(tmp_path):
    native, store = _make_store(tmp_path)
    s = SimStore(store)
    full = np.load(native / "z0.000" / "field.npy")
    cube = Cube(0.0, 60.0, 0.0, 60.0, 0.0, 60.0)
    sub, (ix0, iy0, iz0) = s.read_field_box(0.0, cube)
    nx, ny, nz = sub.shape
    np.testing.assert_allclose(
        sub, full[ix0:ix0 + nx, iy0:iy0 + ny, iz0:iz0 + nz])
    assert s.last_read_stats["tiles_read"] < s.last_read_stats["tiles_total"]


def test_read_cone_prunes_pixels(tmp_path):
    _, store = _make_store(tmp_path)
    s = SimStore(store)
    cone = Cone(lon=10.0, lat=20.0, radius_deg=30.0)
    out = s.read_cone(cone)
    assert s.last_read_stats["pixels_read"] <= s.last_read_stats["pixels_total"]
    # every returned object within the cone
    if len(out.get("lon", [])):
        dlon = np.radians(out["lon"] - cone.lon)
        cosang = (np.sin(np.radians(20.0)) * np.sin(np.radians(out["lat"]))
                  + np.cos(np.radians(20.0)) * np.cos(np.radians(out["lat"]))
                  * np.cos(dlon))
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        assert ang.max() <= 30.0 + 1e-6


def test_converter_detect_and_convert(tmp_path):
    native = generate_linear_sim(
        tmp_path / "n2", _cosmo(),
        box_size=150.0, n_grid=16, redshifts=(0.0,), seed=1,
    )
    cls = detect_converter(native)
    assert cls is LinearSimConverter
    assert get_converter("oneuniverse.simulation.linear") is LinearSimConverter
    store = cls().convert(native, tmp_path / "s2", sim_name="c2")
    assert (store / "manifest.json").is_file()
