"""Phase 19 T5 — registry key widens to (survey_type, sub_kind, z_type)."""
import pytest

from oneuniverse.combine.weights import default_weight_for
from oneuniverse.combine.weights.registry import (
    register_default,
    unregister_default,
)


def test_backward_compat_two_key_call_still_works():
    w = default_weight_for("spectroscopic", "spec")
    assert w is not None


def test_explicit_none_sub_kind_matches_two_key_default():
    w = default_weight_for("spectroscopic", "spec", sub_kind=None)
    assert w is not None


def test_register_sub_kind_specific_default():
    from oneuniverse.combine.weights.ivar import InverseVarianceWeight

    register_default(
        "spectroscopic", "spec",
        lambda: InverseVarianceWeight(
            "z_spec_err", floor=0.01, name="ivar(z_spec,BGS_BRIGHT)",
        ),
        sub_kind="BGS_BRIGHT",
    )
    try:
        w = default_weight_for(
            "spectroscopic", "spec", sub_kind="BGS_BRIGHT",
        )
        assert "BGS_BRIGHT" in repr(w)
        # Fallback to default when sub_kind is unknown.
        fallback = default_weight_for(
            "spectroscopic", "spec", sub_kind="BGS_FAINT",
        )
        assert "BGS_BRIGHT" not in repr(fallback)
    finally:
        unregister_default(
            "spectroscopic", "spec", sub_kind="BGS_BRIGHT",
        )


def test_register_rejects_duplicate_sub_kind():
    from oneuniverse.combine.weights.ivar import InverseVarianceWeight

    register_default(
        "spectroscopic", "spec",
        lambda: InverseVarianceWeight("z_spec_err"),
        sub_kind="DUP",
    )
    try:
        with pytest.raises(ValueError, match="already"):
            register_default(
                "spectroscopic", "spec",
                lambda: InverseVarianceWeight("z_spec_err"),
                sub_kind="DUP",
            )
    finally:
        unregister_default("spectroscopic", "spec", sub_kind="DUP")
