"""Test-only helpers for ``oneuniverse.reference`` tests."""
import numpy as np


def divergence_on_grid(v, box):
    """FFT divergence of a periodic vector field.

    ``v`` has shape (n, n, n, 3) on a grid of side ``box``; returns the real
    (n, n, n) field ``ifftn(i k · v_k)``. The derivative along an axis is set
    to zero on that axis' Nyquist plane, the same convention as the
    Zel'dovich displacement in ``reference.linear``.
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[0]
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=box / n)
    if n % 2 == 0:
        k1[n // 2] = 0.0
    ks = (k1[:, None, None], k1[None, :, None], k1[None, None, :])
    div_k = np.zeros((n, n, n), dtype=np.complex128)
    for i in range(3):
        div_k += 1j * ks[i] * np.fft.fftn(v[..., i])
    return np.fft.ifftn(div_k).real
