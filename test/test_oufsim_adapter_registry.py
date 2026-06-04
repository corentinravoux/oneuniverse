"""S17 T1 — native adapter row reads + format registry."""
import numpy as np
import pytest

from oneuniverse.simulation.oufsim.native import (
    NumpyFieldAdapter, get_adapter, register_adapter, NativeReaderAdapter,
)


def test_registry_resolves_by_format():
    assert isinstance(get_adapter("npy"), NumpyFieldAdapter)
    with pytest.raises(KeyError):
        get_adapter("does_not_exist")


def test_field_adapter_has_no_row_product(tmp_path):
    a = np.arange(8 * 8 * 8, dtype=np.float64).reshape(8, 8, 8)
    p = tmp_path / "f.npy"; np.save(p, a)
    ad = get_adapter("npy")
    sub = ad.read_field_region(p, (slice(0, 4), slice(0, 4), slice(0, 4)))
    assert sub.shape == (4, 4, 4)
    with pytest.raises(NotImplementedError):
        ad.read_rows(p, slice(0, 4))


def test_register_adapter_is_idempotent_by_format():
    @register_adapter
    class _Dummy(NativeReaderAdapter):
        native_format = "dummy_fmt_t1"
        def read_field_region(self, path, cell_slice):
            return np.zeros((1, 1, 1))
    assert get_adapter("dummy_fmt_t1").native_format == "dummy_fmt_t1"
