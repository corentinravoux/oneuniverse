"""Phase 17 T1 — dtype mini-language."""
import pyarrow as pa
import pytest

from oneuniverse.data.dtype_lang import parse_dtype, is_variable_length


def test_scalar_f4():
    t = parse_dtype("f4")
    assert t.equals(pa.float32())


def test_scalar_i8():
    t = parse_dtype("i8")
    assert t.equals(pa.int64())


def test_fixed_size_list_f4_64():
    t = parse_dtype("f4[64]")
    assert isinstance(t, pa.FixedSizeListType)
    assert t.list_size == 64
    assert t.value_type.equals(pa.float32())


def test_fixed_size_list_i8_64():
    t = parse_dtype("i8[64]")
    assert isinstance(t, pa.FixedSizeListType)
    assert t.list_size == 64
    assert t.value_type.equals(pa.int64())


def test_variable_length_list_f4():
    t = parse_dtype("list<f4>")
    assert isinstance(t, pa.ListType)
    assert t.value_type.equals(pa.float32())


def test_large_list_f4():
    t = parse_dtype("large_list<f4>")
    assert isinstance(t, pa.LargeListType)
    assert t.value_type.equals(pa.float32())


def test_rejects_unknown_syntax():
    with pytest.raises(ValueError, match="dtype"):
        parse_dtype("array of floats")


def test_is_variable_length_classifies():
    assert is_variable_length("list<f4>")
    assert is_variable_length("large_list<f4>")
    assert not is_variable_length("f4[64]")
    assert not is_variable_length("f4")
