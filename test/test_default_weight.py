"""Tests for oneuniverse.combine.weights.registry.default_weight_for."""
from __future__ import annotations

import pytest

from oneuniverse.combine import InverseVarianceWeight, default_weight_for


class TestDefaultWeightFor:
    def test_spec_returns_ivar_z_spec_err(self):
        w = default_weight_for("spectroscopic", "spec")
        assert isinstance(w, InverseVarianceWeight)
        assert w.error_column == "z_spec_err"

    def test_phot_returns_ivar_z_phot_err(self):
        w = default_weight_for("photometric", "phot")
        assert isinstance(w, InverseVarianceWeight)
        assert w.error_column == "z_phot_err"

    def test_pec_returns_ivar_velocity_error(self):
        w = default_weight_for("peculiar_velocity", "pec")
        assert isinstance(w, InverseVarianceWeight)
        assert w.error_column == "velocity_error"

    def test_returns_fresh_instance(self):
        w1 = default_weight_for("spectroscopic", "spec")
        w2 = default_weight_for("spectroscopic", "spec")
        assert w1 is not w2

    def test_unknown_pair_raises(self):
        with pytest.raises(KeyError, match="No default weight"):
            default_weight_for("gravitational_wave", "gw")

    def test_error_message_lists_known_pairs(self):
        with pytest.raises(KeyError) as exc:
            default_weight_for("xxx", "yyy")
        msg = str(exc.value)
        assert "spectroscopic" in msg
        assert "photometric" in msg
        assert "peculiar_velocity" in msg
