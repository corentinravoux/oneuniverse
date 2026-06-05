"""measure WL-T2 — photo-z kernel attach."""
import sys
from pathlib import Path

import numpy as np

from oneuniverse.measure.photoz import attach_photoz

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_shear_view  # noqa: E402


def test_attach_photoz_returns_kernel(tmp_path):
    view = synthetic_shear_view(tmp_path, n=1500, seed=2, with_pdf=True)
    kernel = attach_photoz(view)
    assert kernel.mean().shape[0] == view.n_rows
    assert np.all(kernel.std() > 0)
