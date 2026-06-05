"""measure T5 — n(z) radial selection."""
import numpy as np

from oneuniverse.measure.nz import Nz, nz_from_spec_z


def test_nz_normalises_and_records_method():
    z = np.concatenate([np.full(100, 0.3), np.full(300, 0.5)])
    nz = nz_from_spec_z(z, edges=np.linspace(0.0, 1.0, 11))
    assert isinstance(nz, Nz)
    assert nz.method == "spec_hist"
    i3 = np.digitize(0.3, nz.edges) - 1
    i5 = np.digitize(0.5, nz.edges) - 1
    assert nz.counts[i5] > 2.5 * nz.counts[i3]
    assert np.isclose(np.trapz(nz.pdf(), nz.centers()), 1.0, atol=0.2)


def test_nz_weighted():
    z = np.array([0.3, 0.3, 0.5])
    w = np.array([1.0, 1.0, 4.0])
    nz = nz_from_spec_z(z, edges=np.linspace(0.2, 0.6, 5), weights=w)
    i5 = np.digitize(0.5, nz.edges) - 1
    i3 = np.digitize(0.3, nz.edges) - 1
    assert nz.counts[i5] == 4.0 and nz.counts[i3] == 2.0
