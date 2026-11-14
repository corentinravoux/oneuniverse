"""Periodic buffered resimulation of a sub-cube and its TreePM coupling.

A *tile* is a cubic target ``cube`` of the parent box padded by ``buffer`` on
every side. It is resimulated alone (a cheap mini-simulation) and only its
*inner* cells — the target cube — are kept; the buffer absorbs the boundary
error. Two couplings to the parent are provided: the plain buffered run
(:func:`run_buffered`, the baseline) and a TreePM split (:func:`run_treepm`)
whose long-range force comes from the full-box linear field.
:func:`buffer_convergence` scores either against the full-box particle-mesh
reference (:func:`run_full_reference`).

Every physical ingredient is reused from :mod:`.linear` (growth, Zel'dovich,
``k_grid``) and :mod:`.pm` (CIC, Poisson solvers, KDK conventions,
``p = a²ẋ``, ``vel = 100·p/a`` km/s); nothing here re-derives them.

Conventions
-----------
Tile geometry (periodic)
    Parent mesh: ``n`` cells per side, ``cell = box/n``, cell ``i`` covering
    ``[i·cell, (i+1)·cell)``. ``cube`` must be cubic, and ``cube.lo``,
    ``cube.hi`` and ``buffer`` must lie on the parent cell grid to within
    ``1e-9·cell`` (else :class:`~oneuniverse._core.errors.ValidationError`);
    ``buffer >= 0``. The tile starts at the parent cell containing
    ``lo − buffer``, ``i₀ = floor((lo − buffer)/cell)``, and spans
    ``n_sub = (side + 2·buffer)/cell`` cells per axis; it is taken with
    ``np.take(..., mode="wrap")`` on each axis, so a cube at (or across) the
    box edge gets its full buffer from the opposite side of the periodic box
    (the old non-periodic code lost those buffer particles). ``n_sub <= n``
    is required (a tile may not overlap itself). ``sub_box = n_sub·cell``;
    the tile's origin in parent coordinates is ``i₀·cell`` (possibly
    negative: positions are compared modulo ``box``). The inner cube occupies
    tile cells ``[pad, pad + n_in)`` with ``pad = buffer/cell``,
    ``n_in = side/cell``. :func:`inner_reference` is the same periodic slice
    with no buffer.
Density contrast of a tile
    ``δ = n/n̄ − 1`` on the tile mesh (``n_sub³`` parent-sized cells), CIC
    counts ``n``, with ``n̄ = N/n_sub³`` the mean particle density of the
    tile (``N`` = number of tile particles, one per Lagrangian tile cell
    times ``resolution_factor³``). It is *not* renormalised to the inner
    cube, so ``inner`` keeps the local over/under-density of the target
    relative to its tile.
Buffered run (:func:`run_buffered`)
    Zel'dovich ICs (:func:`~.pm.zeldovich_pm_ic_from_field`) are built from
    the extracted tile IC field alone on ``sub_box`` (its FFT treats the tile
    as periodic and drops the tile-mean mode), then evolved with
    :func:`~.pm.run_pm` with ``force="isolated"`` (open boundaries; the tile
    is not periodic). Inherited from :mod:`.pm`: the isolated solve sources
    ``ρ/ρ̄ − 1`` with ``ρ̄`` the tile mean, so the tile as a whole is treated
    as having the background density (no tile-scale collapse or expansion),
    and positions are wrapped into the tile, so matter leaving one face
    re-enters through the opposite one — both are boundary errors the buffer
    must absorb. Super-tile modes (tides and bulk flows beyond the tile) are
    absent: this is the irreducible truncation the TreePM coupling removes.
Refinement (``resolution_factor = r > 1``)
    The *whole parent* IC field is Fourier-interpolated to ``n·r`` cells
    (zero-padding its periodic spectrum, the even-``n`` Nyquist coefficient
    split evenly between ``±k_N``, with the half-cell phase shift that maps
    parent cell centres onto fine cell centres), then the tile is extracted
    on the fine grid; no small-scale power is added. Interpolating the
    periodic parent (rather than the non-periodic tile) avoids edge ringing;
    the cost is one ``(n·r)³`` array. The PM runs on the ``(n_sub·r)³``
    mesh with ``N = n_sub³·r³`` particles; the returned ``inner`` is
    deposited on the *parent* cells (``n_sub³`` tile mesh), so it is directly
    comparable to :func:`inner_reference` for every ``r``.
TreePM run (:func:`run_treepm`)
    Gaussian split at ``k_split`` (default ``2·2π/sub_box``):
    ``W_long(k) = exp(−k²/k_split²)`` and ``W_short = 1 − W_long`` (the
    ``exp(−k²/k_split²)`` form, not the old code's ``exp(−k²/(2k_split²))``).
    Long range: :func:`~.pm.pm_force` of the low-passed *full-box z = 0
    linear* field, scaled by ``D(a)`` (the linear growth of
    ``∇²φ = δ_lin·D``) and CIC-interpolated at the particles' parent
    positions (periodic). Short range: the tile's own isolated force with
    the high-pass applied to the free-space kernel on the zero-padded
    ``(2n_sub)³`` grid (the Hockney–Eastwood kernels of :mod:`.pm`), so the
    filter introduces no wrap-around at the tile faces. The two ranges are
    complementary in ``k``, hence no double counting. ICs: the full-box
    Zel'dovich growing mode (positions and velocities of the tile's
    Lagrangian cells), because the long-range field is the full-box linear
    field; starting the long modes from the tile-only displacement would
    seed a decaying mode. Particles are *not* wrapped into the tile: one
    that leaves the tile keeps moving under the long-range force only, and
    is neither deposited nor given a short-range force. Stepping is the
    :func:`~.pm.run_pm` KDK scheme (``n_steps`` uniform in ``a``, the same
    time integrals), re-run here because :func:`~.pm.run_pm` takes no custom
    force.
Benchmark (:func:`buffer_convergence`)
    One row per buffer: ``cell_corr`` (Pearson, :func:`~.fieldstats.cell_corr`)
    of the tile's inner field against :func:`inner_reference` of the full-box
    run with the same ``z_start, z_end, n_steps``, and ``r_lowk`` = mean of
    :func:`~.fieldstats.cross_r` (computed on the inner cube, side ``L``)
    over the bins whose mean ``k`` is below ``2·k_f``, ``k_f = 2π/L`` — in
    practice the first, ``k ≈ k_f`` bin; NaN if no bin qualifies.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyarrow as pa

from oneuniverse._core.errors import ValidationError
from oneuniverse._core.geometry import Cube
from oneuniverse.reference.fieldstats import cell_corr, cross_r
from oneuniverse.reference.linear import (
    _check_grid,
    _cubic,
    _require_lcdm,
    _scale_factor,
    growth_factor,
    k_grid,
)
from oneuniverse.reference.pm import (
    _isolated_kernels,
    _time_integral,
    deposit_cic,
    interpolate_cic,
    pm_force,
    run_pm,
    zeldovich_pm_ic,
    zeldovich_pm_ic_from_field,
)
from oneuniverse.simulation.specs import CosmologySpec

__all__ = [
    "buffer_convergence",
    "extract_buffered",
    "inner_reference",
    "run_buffered",
    "run_full_reference",
    "run_full_reference_from_ic",
    "run_treepm",
]

_METHODS = ("buffered", "treepm")
_ALIGN_TOL = 1e-9


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def _cells(value: float, cell: float, what: str) -> int:
    x = float(value) / cell
    i = round(x)
    if not np.isfinite(x) or abs(x - i) > _ALIGN_TOL:
        raise ValidationError(f"{what}={value} is not on the parent cell grid (cell={cell})")
    return int(i)


def _tile_geometry(n: int, box: float, cube: Cube, buffer: float):
    """``(start (3,) int, n_sub, pad, n_in)`` of a buffered tile on the parent grid."""
    if not isinstance(cube, Cube):
        raise ValidationError(f"cube must be a Cube, got {type(cube).__name__}")
    buffer = float(buffer)
    if not np.isfinite(buffer) or buffer < 0:
        raise ValidationError(f"buffer must be finite and >= 0, got {buffer}")
    cell = box / n
    lo = np.array([_cells(v, cell, "cube.lo") for v in cube.lo])
    hi = np.array([_cells(v, cell, "cube.hi") for v in cube.hi])
    sides = hi - lo
    if not np.all(sides == sides[0]):
        raise ValidationError(f"cube must be cubic, got sides {sides * cell}")
    pad = _cells(buffer, cell, "buffer")
    n_in = int(sides[0])
    n_sub = n_in + 2 * pad
    if n_sub > n:
        raise ValidationError(
            f"buffered tile ({n_sub} cells) exceeds the parent grid ({n} cells)")
    return lo - pad, n_sub, pad, n_in


def _take_wrap(field: np.ndarray, start, size: int) -> np.ndarray:
    out = field
    for ax in range(3):
        out = np.take(out, start[ax] + np.arange(size), axis=ax, mode="wrap")
    return np.ascontiguousarray(out)


def _inner(delta: np.ndarray, pad: int, n_in: int) -> np.ndarray:
    return np.ascontiguousarray(delta[pad:pad + n_in, pad:pad + n_in, pad:pad + n_in])


def extract_buffered(ic_field, *, box: float, cube: Cube, buffer: float):
    """``(sub_ic, sub_box)``: the periodic buffered tile of ``ic_field`` around ``cube``.

    See the module docstring for the alignment rule and index convention.
    """
    f = _cubic(ic_field, "ic_field")
    n, box = _check_grid(f.shape[0], box)
    start, n_sub, _, _ = _tile_geometry(n, box, cube, buffer)
    return _take_wrap(f, start, n_sub), n_sub * (box / n)


def inner_reference(delta_full, *, box: float, cube: Cube) -> np.ndarray:
    """The parent cells of ``cube`` (periodic), the comparison target of a tile's ``inner``."""
    d = _cubic(delta_full, "delta_full")
    n, box = _check_grid(d.shape[0], box)
    start, n_sub, _, _ = _tile_geometry(n, box, cube, 0.0)
    return _take_wrap(d, start, n_sub)


# --------------------------------------------------------------------------
# Full-box reference
# --------------------------------------------------------------------------

def _evolved_delta(cosmo, pos, vel, *, box, n_grid, z_start, z_end, n_steps):
    out = run_pm(pos, vel, cosmo, box=box, n_grid=n_grid, z_start=z_start, z_end=z_end,
                 n_steps=n_steps, force="periodic", snapshots=(z_end,))
    return out["snapshots"][z_end]


def run_full_reference(cosmo: CosmologySpec, *, box: float, n_grid: int, z_start: float,
                       z_end: float, n_steps: int, seed: int):
    """``(δ at z_end on the n_grid³ parent mesh, z = 0 IC field)`` of a full periodic PM run."""
    pos, vel, ic = zeldovich_pm_ic(cosmo, box=box, n_grid=n_grid, z_start=z_start, seed=seed)
    delta = _evolved_delta(cosmo, pos, vel, box=box, n_grid=n_grid, z_start=z_start,
                           z_end=z_end, n_steps=n_steps)
    return delta, ic


def run_full_reference_from_ic(cosmo: CosmologySpec, ic_field, *, box: float, n_grid: int,
                               z_start: float, z_end: float, n_steps: int):
    """As :func:`run_full_reference` from a given z = 0 linear field (returned unchanged)."""
    f = _cubic(ic_field, "ic_field")
    if f.shape[0] != n_grid:
        raise ValidationError(f"ic_field has {f.shape[0]} cells per side, n_grid={n_grid}")
    pos, vel = zeldovich_pm_ic_from_field(cosmo, f, box=box, z_start=z_start)
    delta = _evolved_delta(cosmo, pos, vel, box=box, n_grid=n_grid, z_start=z_start,
                           z_end=z_end, n_steps=n_steps)
    return delta, ic_field


# --------------------------------------------------------------------------
# Buffered resimulation
# --------------------------------------------------------------------------

def _fourier_refine(field, r: int) -> np.ndarray:
    """Band-limited interpolation of a periodic cell-centred grid to ``r×`` the cells per side."""
    out = np.asarray(field, dtype=np.float64)
    if r == 1:
        return out
    shift = 0.5 - 0.5 * r            # parent-centre → fine-centre offset, in fine cells
    for ax in range(3):
        src = np.moveaxis(out, ax, 0)
        n = src.shape[0]
        m = n * r
        fk = np.fft.fft(src, axis=0)
        fine = np.zeros((m,) + src.shape[1:], dtype=np.complex128)
        q = np.rint(np.fft.fftfreq(n, d=1.0 / n)).astype(np.int64)     # signed integer freqs
        keep = q != -(n // 2) if n % 2 == 0 else np.ones(n, dtype=bool)
        fine[q[keep] % m] = fk[keep]
        if n % 2 == 0:                                                  # split Nyquist ±k_N
            fine[n // 2] = 0.5 * fk[n // 2]
            fine[m - n // 2] = 0.5 * fk[n // 2]
        qf = np.rint(np.fft.fftfreq(m, d=1.0 / m))
        if n % 2 == 0:
            qf[n // 2] = n // 2
            qf[m - n // 2] = -(n // 2)
        phase = r * np.exp(2j * np.pi * qf * shift / m)
        fine *= phase.reshape((m,) + (1,) * (src.ndim - 1))
        out = np.moveaxis(np.fft.ifft(fine, axis=0).real, 0, ax)
    return np.ascontiguousarray(out)


def _resolution(resolution_factor) -> int:
    if (isinstance(resolution_factor, bool) or not isinstance(resolution_factor, (int, np.integer))
            or resolution_factor < 1):
        raise ValidationError(f"resolution_factor must be an integer >= 1, got {resolution_factor!r}")
    return int(resolution_factor)


def _tile_delta(pos: np.ndarray, n_sub: int, sub_box: float, n_particles: int) -> np.ndarray:
    """``n/n̄ − 1`` on the tile mesh, ``n̄ = n_particles/n_sub³``; particles outside the tile are dropped."""
    inside = np.all((pos >= 0.0) & (pos < sub_box), axis=1)
    counts = deposit_cic(pos, n_sub, sub_box, weights=inside.astype(np.float64))
    return counts * (n_sub ** 3 / n_particles) - 1.0


def run_buffered(cosmo: CosmologySpec, ic_field, *, box: float, cube: Cube, buffer: float,
                 z_start: float, z_end: float, n_steps: int, resolution_factor: int = 1) -> dict:
    """Resimulate ``cube`` with a periodic ``buffer`` as an isolated PM tile.

    Returns ``{"inner": δ of the inner cube cells (parent resolution),
    "n_particles": n_sub³·resolution_factor³, "sub_box": tile side}``. See
    the module docstring for the ICs, the force and the δ normalisation.
    """
    _require_lcdm(cosmo)
    f = _cubic(ic_field, "ic_field")
    n, box = _check_grid(f.shape[0], box)
    r = _resolution(resolution_factor)
    start, n_sub, pad, n_in = _tile_geometry(n, box, cube, buffer)
    sub_box = n_sub * (box / n)
    sub = _take_wrap(_fourier_refine(f, r), start * r, n_sub * r)
    pos, vel = zeldovich_pm_ic_from_field(cosmo, sub, box=sub_box, z_start=z_start)
    out = run_pm(pos, vel, cosmo, box=sub_box, n_grid=n_sub * r, z_start=z_start,
                 z_end=z_end, n_steps=n_steps, force="isolated")
    delta = _tile_delta(out["pos"], n_sub, sub_box, len(pos))
    return {"inner": _inner(delta, pad, n_in), "n_particles": len(pos),
            "sub_box": float(sub_box)}


# --------------------------------------------------------------------------
# TreePM coupling
# --------------------------------------------------------------------------

def _short_force(delta: np.ndarray, sub_box: float, w_high: np.ndarray):
    """Isolated force of ``delta`` with the high-pass ``w_high`` applied to the padded kernel."""
    n = delta.shape[0]
    m = 2 * n
    pad = np.zeros((m, m, m), dtype=np.float64)
    pad[:n, :n, :n] = delta
    pad_k = np.fft.fftn(pad) * w_high
    return tuple(np.fft.ifftn(pad_k * kern).real[:n, :n, :n]
                 for kern in _isolated_kernels(n, sub_box))


def _sample(grids, pos: np.ndarray, box: float) -> np.ndarray:
    return np.stack([interpolate_cic(g, pos, box) for g in grids], axis=1)


def run_treepm(cosmo: CosmologySpec, ic_field, *, box: float, cube: Cube, buffer: float,
               z_start: float, z_end: float, n_steps: int, k_split: float | None = None) -> dict:
    """Resimulate ``cube`` with a TreePM split: full-box linear long range + tile PM short range.

    Returns the :func:`run_buffered` keys plus ``"k_split"`` (h/Mpc). See the
    module docstring for the split kernel, the ICs and the boundary handling.
    """
    _require_lcdm(cosmo)
    f = _cubic(ic_field, "ic_field")
    n, box = _check_grid(f.shape[0], box)
    start, n_sub, pad, n_in = _tile_geometry(n, box, cube, buffer)
    cell = box / n
    sub_box = n_sub * cell
    origin = start * cell
    if k_split is None:
        k_split = 2.0 * 2.0 * np.pi / sub_box
    k_split = float(k_split)
    if not np.isfinite(k_split) or k_split <= 0:
        raise ValidationError(f"k_split must be positive and finite, got {k_split}")
    a_start, a_end = _scale_factor(z_start), _scale_factor(z_end)
    if z_end > z_start:
        raise ValidationError(f"z_end ({z_end}) must not exceed z_start ({z_start})")
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps < 1:
        raise ValidationError(f"n_steps must be an integer >= 1, got {n_steps}")
    om = cosmo.omega_m

    # long range: low-passed full-box linear force (z = 0), grown with D(a)
    _, _, _, kmag = k_grid(n, box)
    f_low = np.fft.ifftn(np.fft.fftn(f) * np.exp(-(kmag / k_split) ** 2)).real
    g_long = pm_force(f_low, box)
    # short range: high-pass of the tile's isolated force
    _, _, _, kpad = k_grid(2 * n_sub, 2 * sub_box)
    w_high = -np.expm1(-(kpad / k_split) ** 2)

    # ICs: full-box Zel'dovich growing mode of the tile's Lagrangian cells
    pos_full, vel_full = zeldovich_pm_ic_from_field(cosmo, f, box=box, z_start=z_start)
    g = (np.arange(n) + 0.5) * cell
    q_full = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    disp = np.mod(pos_full - q_full + 0.5 * box, box) - 0.5 * box

    def take(arr: np.ndarray) -> np.ndarray:
        return _take_wrap(arr.reshape(n, n, n, 3), start, n_sub).reshape(-1, 3)

    gt = (np.arange(n_sub) + 0.5) * cell
    q_tile = np.stack(np.meshgrid(gt, gt, gt, indexing="ij"), axis=-1).reshape(-1, 3)
    x = q_tile + take(disp)
    p = (a_start / 100.0) * take(vel_full)
    n_part = len(x)

    def acc(xs: np.ndarray, a: float) -> np.ndarray:
        inside = np.all((xs >= 0.0) & (xs < sub_box), axis=1)
        delta = _tile_delta(xs, n_sub, sub_box, n_part)
        f_short = _sample(_short_force(delta, sub_box, w_high), xs, sub_box) * inside[:, None]
        f_long = growth_factor(1.0 / a - 1.0, cosmo) * _sample(g_long, xs + origin, box)
        return 1.5 * om * (f_short + f_long)

    a_grid = np.linspace(a_start, a_end, int(n_steps) + 1)
    a_grid[-1] = a_end
    acc_x = acc(x, a_start)
    for i in range(int(n_steps)):
        a0, a1 = float(a_grid[i]), float(a_grid[i + 1])
        ah = 0.5 * (a0 + a1)
        p += acc_x * _time_integral(a0, ah, om, 2)
        x = x + p * _time_integral(a0, a1, om, 3)
        acc_x = acc(x, a1)
        p += acc_x * _time_integral(ah, a1, om, 2)
    delta = _tile_delta(x, n_sub, sub_box, n_part)
    return {"inner": _inner(delta, pad, n_in), "n_particles": n_part,
            "sub_box": float(sub_box), "k_split": k_split}


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------

def buffer_convergence(cosmo: CosmologySpec, ic_field, *, box: float, cube: Cube,
                       buffers: Sequence[float], method: str = "buffered",
                       **run_kw) -> pa.Table:
    """Inner-cube agreement with the full-box PM reference versus buffer size.

    ``run_kw`` is forwarded to :func:`run_buffered` / :func:`run_treepm` and
    must contain ``z_start``, ``z_end`` and ``n_steps`` (also used for the
    reference run). Returns a table with float64 columns ``buffer``,
    ``cell_corr``, ``r_lowk`` (see the module docstring), one row per buffer.
    """
    if method not in _METHODS:
        raise ValidationError(f"method must be one of {_METHODS}, got {method!r}")
    buffers = [float(b) for b in buffers]
    if not buffers:
        raise ValidationError("buffers must not be empty")
    missing = sorted({"z_start", "z_end", "n_steps"} - set(run_kw))
    if missing:
        raise ValidationError(f"buffer_convergence needs run_kw {missing}")
    f = _cubic(ic_field, "ic_field")
    full, _ = run_full_reference_from_ic(
        cosmo, f, box=box, n_grid=f.shape[0], z_start=run_kw["z_start"],
        z_end=run_kw["z_end"], n_steps=run_kw["n_steps"])
    ref = inner_reference(full, box=box, cube=cube)
    side = cube.hi[0] - cube.lo[0]
    run = run_buffered if method == "buffered" else run_treepm
    corr, r_low = [], []
    for b in buffers:
        inner = run(cosmo, f, box=box, cube=cube, buffer=b, **run_kw)["inner"]
        corr.append(cell_corr(inner, ref))
        k, r = cross_r(inner, ref, box=side)
        sel = k < 2.0 * (2.0 * np.pi / side)
        r_low.append(float(np.nanmean(r[sel])) if np.any(sel & np.isfinite(r)) else float("nan"))
    return pa.table({"buffer": pa.array(buffers, pa.float64()),
                     "cell_corr": pa.array(corr, pa.float64()),
                     "r_lowk": pa.array(r_low, pa.float64())})
