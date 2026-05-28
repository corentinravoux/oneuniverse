"""Phase 16 T2/T3 — ColumnDef carries observational metadata."""
import pytest

from oneuniverse.data.schema import ColumnDef


def test_columndef_defaults():
    c = ColumnDef("z", "f4", "", "redshift")
    assert c.frame is None
    assert c.epoch is None
    assert c.wavelength_convention is None
    assert c.nullable is False


def test_columndef_accepts_frame():
    c = ColumnDef(
        "z_helio", "f4", "", "heliocentric z",
        frame="heliocentric",
    )
    assert c.frame == "heliocentric"


def test_columndef_accepts_epoch():
    c = ColumnDef(
        "ra", "f8", "deg", "ICRS at GAIA DR3 epoch",
        epoch=2016.0,
    )
    assert c.epoch == 2016.0


def test_columndef_accepts_wavelength_convention():
    c = ColumnDef(
        "loglam", "f4", "", "log wavelength",
        wavelength_convention="vacuum",
    )
    assert c.wavelength_convention == "vacuum"


def test_columndef_accepts_nullable():
    c = ColumnDef("z_phot", "f4", "", "photo-z", nullable=True)
    assert c.nullable is True


def test_columndef_remains_frozen():
    c = ColumnDef("z", "f4", "", "redshift")
    with pytest.raises(Exception):  # FrozenInstanceError
        c.frame = "cmb"  # type: ignore[misc]


# ── T3 annotations ────────────────────────────────────────────────────


def test_core_z_columns_have_no_frame_by_default():
    from oneuniverse.data.schema import CORE_COLUMNS

    by_name = {c.name: c for c in CORE_COLUMNS}
    # CORE z has no fixed frame — the loader / manifest fixes it.
    assert by_name["z"].frame is None


def test_spec_zhelio_is_heliocentric():
    from oneuniverse.data.schema import SPECTROSCOPIC_COLUMNS

    by_name = {c.name: c for c in SPECTROSCOPIC_COLUMNS}
    assert by_name["z_helio"].frame == "heliocentric"
    assert by_name["z_cmb"].frame == "cmb"
    assert by_name["cz_cmb"].frame == "cmb"


def test_snia_zcmb_is_cmb():
    from oneuniverse.data.schema import SNIA_COLUMNS

    by_name = {c.name: c for c in SNIA_COLUMNS}
    assert by_name["z_cmb"].frame == "cmb"


def test_ra_dec_nullable_is_false():
    from oneuniverse.data.schema import CORE_COLUMNS

    by_name = {c.name: c for c in CORE_COLUMNS}
    assert by_name["ra"].nullable is False
    assert by_name["dec"].nullable is False
