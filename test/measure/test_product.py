import dataclasses
import json

import numpy as np
import pyarrow as pa
import pytest

from oneuniverse._core.errors import ValidationError
from oneuniverse.measure.atoms import Regions
from oneuniverse.measure.measurement_set import MeasurementSet, MeasurementSpec
from oneuniverse.measure.product import FieldMap, Product, Sightline


def _cat(n=10): return pa.table({"ra": np.zeros(n), "dec": np.zeros(n), "z": np.full(n, 0.5), "weight": np.ones(n), "region_id": np.zeros(n, np.int32), "source_row": np.arange(n), "partition": np.zeros(n, np.int32), "offset": np.arange(n)})


def test_product_validates_core_and_extras():
    p = Product(_cat()); assert p.n == 10 and p.kind == "pointset"
    with pytest.raises(ValidationError, match="region_id"): Product(_cat().drop(["region_id"]))
    with pytest.raises(TypeError): Product(_cat(), extras={"x": [1, 2]})
    assert p.with_extra("e1", np.zeros(10)).extras["e1"].shape == (10,)


def test_product_rejects_empty_catalog():
    with pytest.raises(ValidationError, match="empty"): Product(_cat(0))


def test_product_rejects_cosmology_columns():
    with pytest.raises(ValidationError, match="comoving_distance"): Product(_cat().append_column("comoving_distance", pa.array(np.zeros(10))))


def test_product_dtype_families_randoms_extras_keys():
    with pytest.raises(ValidationError, match="ra"):
        Product(_cat().set_column(0, "ra", pa.array(np.zeros(10, np.int64))))
    with pytest.raises(ValidationError, match="randoms"):
        Product(_cat(), randoms=_cat().select(["ra", "dec"]))
    with pytest.raises(ValidationError, match="chi"):
        Product(_cat(), randoms=_cat().append_column("chi", pa.array(np.zeros(10))))
    with pytest.raises(ValidationError, match="identifier"):
        Product(_cat(), extras={"not ok": np.zeros(3)})
    with pytest.raises(ValidationError, match="r_comoving"):
        Product(_cat(), extras={"r_comoving": np.zeros(3)})
    assert Product(_cat(), extras={"t": pa.table({"a": [1]})}).extras["t"].num_rows == 1


def test_product_provenance_json_normalised():
    p = Product(_cat(), provenance={"z": (0.1, np.float64(0.5)), "n": np.int64(3)})
    assert p.provenance == {"z": [0.1, 0.5], "n": 3}
    assert type(p.provenance["n"]) is int
    json.dumps(p.provenance)
    with pytest.raises(ValidationError, match="provenance"):
        Product(_cat(), provenance={"x": object()})


def test_fieldmap_and_sightline_kinds():
    fm = FieldMap(_cat(0), values=np.zeros(12 * 4 * 4), mask=np.ones(12 * 4 * 4), nside=4); assert fm.kind == "fieldmap"
    sl_cat = _cat(3).append_column("sightline_id", pa.array([1, 2, 3])).append_column("wavelength", pa.array([[3600.0, 3601.0]] * 3, pa.list_(pa.float32()))).append_column("delta", pa.array([[0.1, -0.1]] * 3, pa.list_(pa.float32()))).append_column("pixel_weight", pa.array([[1.0, 1.0]] * 3, pa.list_(pa.float32())))
    sl = Sightline(sl_cat); assert sl.kind == "sightline" and sl.n_pixels == 6
    with pytest.raises(ValidationError, match="wavelength"): Sightline(_cat(3))


def test_fieldmap_validates_lengths():
    with pytest.raises(ValidationError): FieldMap(_cat(0), values=np.zeros(5), mask=np.ones(192), nside=4)
    with pytest.raises(ValidationError): FieldMap(_cat(0), values=np.zeros(192), mask=np.ones(5), nside=4)
    with pytest.raises(ValidationError, match="core|region_id"):
        FieldMap(_cat(0).drop(["region_id"]), values=np.zeros(192), mask=np.ones(192), nside=4)


def test_measurement_set_invariants():
    reg = Regions(4, True, 2, np.zeros(12 * 16, np.int32), "test"); p = Product(_cat())
    ms = MeasurementSet({"gal": p}, MeasurementSpec("t", ("gal",), ("pk",)), reg); json.dumps(ms.summary())
    with pytest.raises(ValidationError, match="tracers"): MeasurementSet({"gal": p}, MeasurementSpec("t", ("other",), ("pk",)), reg)
    bad = Product(_cat().set_column(4, "region_id", pa.array(np.full(10, 7, np.int32))))
    with pytest.raises(ValidationError, match="region"): MeasurementSet({"gal": bad}, MeasurementSpec("t", ("gal",), ("pk",)), reg)
    with pytest.raises(ValidationError, match="pairs"): MeasurementSet({"gal": p}, MeasurementSpec("t", ("gal",), ("pk",), pairs=(("gal", "x"),)), reg)


def test_measurement_set_fieldmap_exempt_and_summary():
    reg = Regions(4, True, 2, np.zeros(12 * 16, np.int32), "test")
    fm = FieldMap(_cat(0), values=np.zeros(192), mask=np.ones(192), nside=4)
    ms = MeasurementSet({"k": fm}, MeasurementSpec("t", ("k",), ("cl",)), reg)
    s = ms.summary()
    assert s["products"]["k"]["kind"] == "fieldmap" and s["regions"]["n_regions"] == 2
    json.dumps(s)


def test_measurement_set_negative_region_and_spec_validation():
    reg = Regions(4, True, 2, np.zeros(12 * 16, np.int32), "test")
    neg = Product(_cat().set_column(4, "region_id", pa.array(np.full(10, -1, np.int32))))
    with pytest.raises(ValidationError, match="region"):
        MeasurementSet({"g": neg}, MeasurementSpec("t", ("g",), ("pk",)), reg)
    with pytest.raises(ValidationError): MeasurementSpec("t", ("g",), ("pk",), covariance="bogus")
    with pytest.raises(ValidationError): MeasurementSpec("t", "g", ("pk",))
    with pytest.raises(TypeError): dataclasses.replace(Product(_cat()), extras={"x": {"a": 1}})
