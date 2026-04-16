"""Tests for :class:`CrossMatchRules` (Phase 4 Task 1)."""

from __future__ import annotations

import pytest

from oneuniverse.data.oneuid_rules import CrossMatchRules


class TestDefaults:
    def test_default_sky_tol(self):
        r = CrossMatchRules()
        assert r.sky_tol_arcsec == 1.0

    def test_default_dz_tol(self):
        r = CrossMatchRules()
        assert r.dz_tol_for("spec", "spec") == pytest.approx(1e-3)

    def test_default_accepts_everything(self):
        r = CrossMatchRules()
        assert r.accepts("spec", "phot") is True
        assert r.accepts("phot", "phot") is True
        assert r.accepts("none", "spec") is True


class TestZtypeOverride:
    def test_pair_override(self):
        r = CrossMatchRules(dz_tol_by_ztype={("spec", "phot"): 5e-2})
        assert r.dz_tol_for("spec", "phot") == 5e-2

    def test_pair_override_symmetric(self):
        r = CrossMatchRules(dz_tol_by_ztype={("spec", "phot"): 5e-2})
        assert r.dz_tol_for("phot", "spec") == 5e-2

    def test_default_unchanged_for_other_pairs(self):
        r = CrossMatchRules(dz_tol_by_ztype={("spec", "phot"): 5e-2})
        assert r.dz_tol_for("spec", "spec") == pytest.approx(1e-3)

    def test_dz_tol_default_none_means_no_z_cut(self):
        r = CrossMatchRules(dz_tol_default=None)
        assert r.dz_tol_for("spec", "spec") is None


class TestReject:
    def test_reject_pair(self):
        r = CrossMatchRules(reject_ztype={("phot", "phot")})
        assert r.accepts("phot", "phot") is False

    def test_reject_pair_symmetric(self):
        r = CrossMatchRules(reject_ztype={("spec", "phot")})
        assert r.accepts("spec", "phot") is False
        assert r.accepts("phot", "spec") is False

    def test_reject_does_not_affect_other_pairs(self):
        r = CrossMatchRules(reject_ztype={("phot", "phot")})
        assert r.accepts("phot", "spec") is True
        assert r.accepts("spec", "spec") is True


class TestHash:
    def test_hash_is_deterministic(self):
        r1 = CrossMatchRules(sky_tol_arcsec=1.0, dz_tol_default=1e-3)
        r2 = CrossMatchRules(sky_tol_arcsec=1.0, dz_tol_default=1e-3)
        assert r1.hash() == r2.hash()

    def test_hash_order_invariant_for_pairs(self):
        r1 = CrossMatchRules(dz_tol_by_ztype={("spec", "phot"): 5e-2})
        r2 = CrossMatchRules(dz_tol_by_ztype={("phot", "spec"): 5e-2})
        assert r1.hash() == r2.hash()

    def test_hash_order_invariant_for_rejects(self):
        r1 = CrossMatchRules(reject_ztype={("phot", "spec")})
        r2 = CrossMatchRules(reject_ztype={("spec", "phot")})
        assert r1.hash() == r2.hash()

    def test_hash_changes_when_rules_change(self):
        r1 = CrossMatchRules(sky_tol_arcsec=1.0)
        r2 = CrossMatchRules(sky_tol_arcsec=2.0)
        assert r1.hash() != r2.hash()


class TestFrozen:
    def test_instance_is_hashable(self):
        r = CrossMatchRules()
        hash(r)

    def test_fields_immutable(self):
        r = CrossMatchRules()
        with pytest.raises((AttributeError, Exception)):
            r.sky_tol_arcsec = 2.0
