"""Phase 16 T1 — z_type registry is extensible and validatable."""
import pytest

from oneuniverse.data.ztypes import (
    Z_TYPE_REGISTRY,
    assert_valid,
    is_registered,
    register_z_type,
)


def test_legacy_values_are_registered():
    for v in ("spec", "phot", "phot_pdf", "pv", "none"):
        assert is_registered(v)


def test_register_new_value_is_idempotent():
    register_z_type("cluster_z", description="z from cluster member consensus")
    register_z_type("cluster_z", description="z from cluster member consensus")
    assert is_registered("cluster_z")


def test_register_rejects_bad_names():
    with pytest.raises(ValueError, match="lowercase"):
        register_z_type("Spec")
    with pytest.raises(ValueError, match="lowercase"):
        register_z_type("z-type")


def test_assert_valid_passes_for_known_values():
    register_z_type("spec_lya")
    assert_valid(["spec", "spec_lya", "phot"])


def test_assert_valid_rejects_unknown():
    with pytest.raises(ValueError, match="unregistered"):
        assert_valid(["spec", "made_up"])


def test_registry_is_set_like():
    assert isinstance(Z_TYPE_REGISTRY, set)
    register_z_type("xcorr_z")
    assert "xcorr_z" in Z_TYPE_REGISTRY
