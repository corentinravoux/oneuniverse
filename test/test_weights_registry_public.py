"""Public register_default / unregister_default hooks on the weight registry."""
from __future__ import annotations

import pandas as pd
import pytest

from oneuniverse.combine.weights import ColumnWeight, default_weight_for
from oneuniverse.combine.weights.registry import (
    register_default, unregister_default,
)


def test_register_new_default():
    def _factory():
        return ColumnWeight("special_w", name="special")

    register_default("custom", "x", _factory)
    try:
        w = default_weight_for("custom", "x")
        df = pd.DataFrame({"special_w": [1.0, 2.0, 3.0]})
        got = w(df)
        assert list(got) == [1.0, 2.0, 3.0]
    finally:
        unregister_default("custom", "x")


def test_register_default_rejects_duplicate():
    def _factory():
        return ColumnWeight("a")

    register_default("custom2", "x", _factory)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_default("custom2", "x", _factory)
    finally:
        unregister_default("custom2", "x")


def test_unregister_missing_raises():
    with pytest.raises(KeyError):
        unregister_default("does_not_exist", "anything")
