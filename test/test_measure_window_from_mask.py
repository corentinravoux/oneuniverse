"""M1 fix — the window can be built from the survey's own angular mask."""
import healpy as hp
import numpy as np
import pytest

from oneuniverse.measure.window import Window, window_from_mask


def test_window_from_mask_uses_supplied_completeness():
    nside = 32; npix = hp.nside2npix(nside)
    comp = np.zeros(npix); comp[:npix // 2] = 1.0          # half-sky mask
    depth = np.random.rand(npix)
    win = window_from_mask(comp, nside=nside,
                           systematics={"depth": depth})
    assert isinstance(win, Window) and win.nside == nside
    np.testing.assert_array_equal(win.mask, comp)
    assert abs(win.covered_fraction() - 0.5) < 1e-9
    assert "depth" in win.systematics
    # a covered pixel centre is "contained"; an uncovered one is not
    th, ph = hp.pix2ang(nside, 0, nest=True)
    assert win.contains(np.degrees(ph), 90 - np.degrees(th))


def test_window_from_mask_validates_nside():
    with pytest.raises(ValueError, match="needs"):
        window_from_mask(np.ones(10), nside=32)
