"""``create_field_store``: atomic multi-field store creation."""
import numpy as np
import pytest

from oneuniverse._core.errors import ValidationError
from oneuniverse.simulation import (
    CosmologySpec,
    Cube,
    OUFSimManifest,
    ProvenanceSpec,
    UnitFrameSpec,
    create_field_store,
)

COSMO = CosmologySpec(0.6736, 0.3153, 0.0493, 0.8111, 0.9649)


def _manifest(name="s"):
    return OUFSimManifest("1.0", name, 32.0, COSMO, UnitFrameSpec(), ProvenanceSpec("t", "1", 0, "h"),
                          (), {"format": "ingested"}, "2026-01-01T00:00:00+00:00")


def test_create_field_store_two_fields_and_refuses_existing(tmp_path):
    f = np.random.default_rng(0).standard_normal((8, 8, 8))
    st = create_field_store(tmp_path / "s", manifest=_manifest(), fields={"a": f, "b": 2 * f}, z=0.5)
    assert set(st.products) == {"a", "b"}
    assert np.allclose(st.read_field_box("b", 0.5, Cube((0, 0, 0), (32.0,) * 3)), 2 * f)
    with pytest.raises(ValidationError):
        create_field_store(tmp_path / "s", manifest=_manifest(), fields={"a": f}, z=0.5)
    with pytest.raises(ValidationError):
        create_field_store(tmp_path / "e", manifest=_manifest(), fields={}, z=0.5)
    assert not (tmp_path / "e").exists()
