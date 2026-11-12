import numpy as np
import pyarrow as pa
import pytest

from oneuniverse._core.errors import SchemaError
from oneuniverse.data.schema import (Geometry, cast_to_core, core_schema,
                                     register_z_type, stable_object_id,
                                     validate_schema, validate_z_type_values,
                                     z_types)


def test_core_point_schema_types():
    s = core_schema(Geometry.POINT)
    assert s.field("ra").type == pa.float64() and s.field("hpix32").type == pa.int32()
    assert pa.types.is_dictionary(s.field("z_type").type) and s.field("z_type").type.index_type == pa.int8()


def test_validate_schema_missing_and_wrong_type():
    good = core_schema(Geometry.POINT); validate_schema(good, Geometry.POINT)
    with pytest.raises(SchemaError, match="missing.*dec"): validate_schema(good.remove(good.get_field_index("dec")), Geometry.POINT)
    bad = good.set(good.get_field_index("ra"), pa.field("ra", pa.string()))
    with pytest.raises(SchemaError, match="ra"): validate_schema(bad, Geometry.POINT)


def test_validate_schema_accepts_extra_and_optional_group_types():
    s = core_schema(Geometry.POINT).append(pa.field("mag", pa.float32())).append(pa.field("e1", pa.float32()))
    validate_schema(s, Geometry.POINT)
    with pytest.raises(SchemaError, match="e1"): validate_schema(s.set(s.get_field_index("e1"), pa.field("e1", pa.string())), Geometry.POINT)


def test_validate_schema_dictionary_value_type():
    s = core_schema(Geometry.POINT)
    bad = s.set(s.get_field_index("survey"), pa.field("survey", pa.dictionary(pa.int16(), pa.int32())))
    with pytest.raises(SchemaError, match="survey"): validate_schema(bad, Geometry.POINT)


def test_other_geometries_validate():
    for g in Geometry:
        validate_schema(core_schema(g), g)


def test_register_z_type_and_reject_unknown():
    register_z_type("grism"); assert "grism" in z_types()
    with pytest.raises(ValueError): register_z_type("spec")
    with pytest.raises(ValueError): register_z_type("Bad Name")


def test_validate_z_type_values():
    validate_z_type_values(pa.array(["spec", "phot"]).dictionary_encode())
    validate_z_type_values(pa.chunked_array([pa.array(["spec"])]))
    with pytest.raises(SchemaError, match="weird"):
        validate_z_type_values(pa.array(["spec", "weird"]).dictionary_encode())


def test_stable_object_id_is_deterministic_and_disjoint():
    a = stable_object_id("eboss_qso", np.arange(3)); b = stable_object_id("desi_qso", np.arange(3))
    assert a.dtype == np.int64 and np.array_equal(a, stable_object_id("eboss_qso", np.arange(3))) and not np.intersect1d(a, b).size
    with pytest.raises(ValueError): stable_object_id("x", np.array([2**32]))
    with pytest.raises(ValueError): stable_object_id("x", np.array([-1]))
    assert np.array_equal(a & 0xFFFFFFFF, np.arange(3))


def test_cast_to_core_dictionary_encodes():
    t = pa.table({"ra": [1.0], "dec": [2.0], "z": [0.5], "z_type": ["spec"], "z_err": [0.1], "object_id": [1], "survey": ["s"], "source_row": [0], "hpix32": [3], "mag": [20.0]})
    out = cast_to_core(t, Geometry.POINT)
    assert pa.types.is_dictionary(out.schema.field("survey").type) and out.schema.field("z_err").type == pa.float32() and out.schema.field("mag").type == pa.float64()
