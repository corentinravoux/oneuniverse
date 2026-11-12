import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from astropy.io import fits
from fixtures.fits_like import write_fake_desi_dr1_fits, write_fake_eboss_dr16q_fits

from oneuniverse._core.errors import NotFoundError
from oneuniverse.data.loaders import get_loader, list_surveys, read_fits_batches
from oneuniverse.data.schema import Geometry, stable_object_id, validate_schema


def _concat(batches):
    return pa.concat_tables([pa.Table.from_batches([b]) for b in batches])


def test_registry_lists_two_real_surveys():
    s = list_surveys()
    assert set(s) >= {"eboss_qso", "desi_qso"} and s["eboss_qso"].kind == "spectroscopic"
    assert list(s) == sorted(s)
    with pytest.raises(NotFoundError):
        get_loader("nope")
    assert get_loader("desi_qso") is not get_loader("desi_qso")


def test_desi_batches_core_complete_and_filtered(tmp_path):
    src = write_fake_desi_dr1_fits(tmp_path, 5_000, 0)
    t = _concat(get_loader("desi_qso").batches(src.parent, batch_size=1_000))
    validate_schema(t.schema, Geometry.POINT)
    assert t.num_rows < 5_000 and set(t["z_type"].to_pylist()) == {"spec"}
    assert "mag_g" in t.column_names and "targetid" in t.column_names


def test_desi_mag_nan_for_nonpositive_flux(tmp_path, recwarn):
    src = write_fake_desi_dr1_fits(tmp_path, 3_000, 1)
    t = _concat(get_loader("desi_qso").batches(src.parent))
    raw = fits.getdata(src, 1)
    flux = raw["FLUX_G"][t["source_row"].to_numpy()]
    mag = t["mag_g"].to_numpy()
    assert np.all(np.isnan(mag[flux <= 0])) and (flux <= 0).any()
    assert np.allclose(mag[flux > 0], 22.5 - 2.5 * np.log10(flux[flux > 0]), atol=1e-5)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_source_row_survives_cut(tmp_path):
    src = write_fake_desi_dr1_fits(tmp_path, 3_000, 1)
    t = _concat(get_loader("desi_qso").batches(src.parent, batch_size=700))
    raw = fits.getdata(src, 1)
    rows = t["source_row"].to_numpy()
    assert np.allclose(raw["RA"][rows], t["ra"].to_numpy())
    assert np.all(np.char.strip(raw["SPECTYPE"][rows]) == "QSO") and np.all(raw["ZWARN"][rows] == 0)
    assert np.array_equal(t["object_id"].to_numpy(), stable_object_id("desi_qso", rows))


def test_source_row_contiguous_across_uneven_batches(tmp_path):
    src = write_fake_desi_dr1_fits(tmp_path, 3_001, 4)
    raw = fits.getdata(src, 1)
    expected = np.flatnonzero((np.char.strip(raw["SPECTYPE"]) == "QSO") & (raw["ZWARN"] == 0))
    for bs in (7, 700, 1_333):
        t = _concat(get_loader("desi_qso").batches(src.parent, batch_size=bs))
        assert np.array_equal(t["source_row"].to_numpy(), expected)
        assert np.allclose(t["z"].to_numpy(), raw["Z_RR"][expected])


def test_eboss_batches_and_dla_count(tmp_path):
    src = write_fake_eboss_dr16q_fits(tmp_path, 4_000, 0)
    t = _concat(get_loader("eboss_qso").batches(src.parent))
    validate_schema(t.schema, Geometry.POINT)
    raw = fits.getdata(src, 1)
    rows = t["source_row"].to_numpy()
    assert np.all(raw["IS_QSO_FINAL"][rows] == 1) and np.all(np.isnan(t["z_err"].to_numpy()))
    assert t["n_dla"].to_numpy().max() <= 5
    assert np.array_equal(t["n_dla"].to_numpy(), (raw["Z_DLA"][rows] > 0).sum(axis=1))
    assert t["n_dla"].type == pa.int8()


def test_loader_kwargs_z_cut(tmp_path):
    src = write_fake_desi_dr1_fits(tmp_path, 2_000, 2)
    t = _concat(get_loader("desi_qso").batches(src.parent, z_min=2.0))
    assert pc.min(t["z"]).as_py() >= 2.0
    raw = fits.getdata(src, 1)
    assert np.all(raw["Z_RR"][t["source_row"].to_numpy()] >= 2.0)


def test_unknown_option_and_missing_file(tmp_path):
    src = write_fake_desi_dr1_fits(tmp_path, 100, 0)
    with pytest.raises(TypeError):
        list(get_loader("desi_qso").batches(src.parent, bogus=1))
    with pytest.raises(NotFoundError, match="QSO_full.dat.fits"):
        list(get_loader("desi_qso").batches(tmp_path / "empty"))
    assert get_loader("desi_qso").original_files(src.parent) == [src]


def test_fits_batches_native_types(tmp_path):
    src = write_fake_eboss_dr16q_fits(tmp_path, 100, 0)
    start, t = next(read_fits_batches(src, 1, ["RA", "SDSS_NAME", "Z_DLA"], 100))
    assert start == 0 and t["RA"].type == pa.float64() and pa.types.is_string(t["SDSS_NAME"].type)
    assert pa.types.is_fixed_size_list(t["Z_DLA"].type)
    assert t["SDSS_NAME"][0].as_py() == t["SDSS_NAME"][0].as_py().strip()


def test_broken_entry_point_does_not_break_loading(monkeypatch):
    from oneuniverse.data.loaders import registry

    class _EP:
        name = "broken_plugin"

        def load(self):
            raise ImportError("boom")

    monkeypatch.setattr(registry, "entry_points", lambda group: [_EP()])
    with pytest.warns(RuntimeWarning, match="broken_plugin"):
        assert registry.load_entry_points() == []
    assert "broken_plugin" not in list_surveys()
