import pytest
from fixtures.fits_like import write_fake_desi_dr1_fits, write_fake_eboss_dr16q_fits

from oneuniverse._core.errors import FormatError
from oneuniverse.data import DatasetView, convert, get_loader


def test_convert_desi_end_to_end(tmp_path):
    src = write_fake_desi_dr1_fits(tmp_path / "raw", 6_000, 0)
    m = convert("desi_qso", source=src.parent, out=tmp_path / "db/desi_qso", batch_size=1_000)
    assert m.survey.name == "desi_qso" and m.loader.kwargs == {"batch_size": 1_000}
    assert m.original_files[0].path == "QSO_full.dat.fits" and m.original_files[0].n_rows == 6_000
    v = DatasetView.from_path(tmp_path / "db/desi_qso")
    assert v.n_rows == m.n_rows and v.manifest.extension("spectrum")["wavelength_convention"] == "vacuum"


def test_convert_default_kwargs_empty(tmp_path):
    src = write_fake_eboss_dr16q_fits(tmp_path / "raw", 500, 0)
    m = convert("eboss_qso", source=src.parent, out=tmp_path / "e")
    assert m.loader.kwargs == {}


def test_convert_accepts_loader_instance_and_overwrite(tmp_path):
    src = write_fake_eboss_dr16q_fits(tmp_path / "raw", 1_000, 0)
    convert(get_loader("eboss_qso"), source=src.parent, out=tmp_path / "e")
    with pytest.raises(FormatError):
        convert("eboss_qso", source=src.parent, out=tmp_path / "e")
    m = convert("eboss_qso", source=src.parent, out=tmp_path / "e", overwrite=True)
    assert m.validity.version == 2


def test_convert_bad_kwarg_leaves_nothing(tmp_path):
    src = write_fake_eboss_dr16q_fits(tmp_path / "raw", 100, 0)
    with pytest.raises(TypeError):
        convert("eboss_qso", source=src.parent, out=tmp_path / "e", bogus=1)
    assert not (tmp_path / "e").exists() and not (tmp_path / "e.tmp").exists()
