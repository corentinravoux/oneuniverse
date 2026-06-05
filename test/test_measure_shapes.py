"""measure WL-T1 — shear-source shape atoms."""
import sys
from pathlib import Path

import pytest

from oneuniverse.measure.shapes import attach_shear

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_shear_view  # noqa: E402


def test_attach_shear_validates_and_weights(tmp_path):
    view = synthetic_shear_view(tmp_path, n=2000, seed=1, kind="metacal")
    cat = view.read()
    out, recipe = attach_shear(cat, kind="metacal")
    assert {"e1", "e2", "shear_weight"} <= set(out.columns)
    assert "weight" in out.columns and (out["weight"] >= 0).all()
    assert "metacal" in recipe.lower()
    with pytest.raises(ValueError, match="shape column"):
        attach_shear(cat.drop(columns=["e1"]), kind="metacal")
