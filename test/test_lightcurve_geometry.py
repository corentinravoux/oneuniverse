from oneuniverse.data.format_spec import (
    DataGeometry, GEOMETRY_COLUMNS, validate_columns,
    DEFAULT_PARTITION_ROWS, FORMAT_VERSION, SCHEMA_VERSION,
)


def test_enum_value():
    assert DataGeometry.LIGHTCURVE.value == "lightcurve"


def test_object_required_columns_present():
    req = GEOMETRY_COLUMNS[DataGeometry.LIGHTCURVE]["objects"]
    for c in ("object_id", "ra", "dec", "z", "n_epochs", "mjd_min", "mjd_max"):
        assert c in req


def test_data_required_columns_present():
    req = GEOMETRY_COLUMNS[DataGeometry.LIGHTCURVE]["data"]
    for c in ("object_id", "mjd", "flux", "flux_err"):
        assert c in req


def test_validate_columns_objects_ok():
    missing = validate_columns(
        ["object_id", "ra", "dec", "z", "n_epochs", "mjd_min",
         "mjd_max", "_healpix32", "z_type", "z_err"],
        DataGeometry.LIGHTCURVE, table_type="objects",
    )
    assert missing == []


def test_validate_columns_data_missing_mjd():
    assert "mjd" in validate_columns(
        ["object_id"], DataGeometry.LIGHTCURVE, table_type="data",
    )


def test_default_partition_rows_has_lightcurve():
    assert DataGeometry.LIGHTCURVE in DEFAULT_PARTITION_ROWS


def test_format_version_is_2_1_0():
    assert FORMAT_VERSION == "2.1.0"
    assert SCHEMA_VERSION == "2.1.0"
