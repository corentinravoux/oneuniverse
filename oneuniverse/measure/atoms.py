"""Small immutable building blocks of a measurement: Nz, Window, Regions.

Arrays are stored as private copies with fixed dtypes. Equality of Nz and
Window is by identity (arrays make value equality ambiguous); Regions defines
value equality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import healpy as hp
import numpy as np
from scipy.cluster.vq import kmeans2

from oneuniverse._core import healpix
from oneuniverse._core.errors import ValidationError

_NZ_METHODS = ("spec_hist", "photoz_stack", "external")


def _set(obj, name, value):
    object.__setattr__(obj, name, value)


@dataclass(frozen=True, eq=False)
class Nz:
    """Redshift distribution as histogram counts on *edges*.

    counts has shape (k,) or (b, k) (one row per tomographic bin; *bins*
    optionally labels the rows).
    """
    edges: np.ndarray
    counts: np.ndarray
    method: Literal["spec_hist", "photoz_stack", "external"]
    bins: tuple[int, ...] | None = None

    def __post_init__(self):
        edges = np.array(self.edges, dtype=np.float64)
        counts = np.array(self.counts, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) \
                or np.any(np.diff(edges) <= 0):
            raise ValidationError("Nz edges must be 1-D, finite and strictly increasing")
        if counts.ndim not in (1, 2) or counts.shape[-1] != len(edges) - 1:
            raise ValidationError(
                f"Nz counts last axis must be len(edges)-1={len(edges) - 1}, got shape {counts.shape}")
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValidationError("Nz counts must be finite and >= 0")
        if self.method not in _NZ_METHODS:
            raise ValidationError(f"Nz method must be one of {_NZ_METHODS}, got {self.method!r}")
        bins = self.bins
        if bins is not None:
            bins = tuple(int(b) for b in bins)
            if counts.ndim != 2 or len(bins) != counts.shape[0]:
                raise ValidationError("Nz bins must label the rows of 2-D counts")
        _set(self, "edges", edges)
        _set(self, "counts", counts)
        _set(self, "bins", bins)

    def _density(self) -> np.ndarray:
        return self.counts / np.diff(self.edges)

    def pdf(self, z) -> np.ndarray:
        """Piecewise-constant density at *z* (0 outside the edges).

        Shape of z for 1-D counts, (b, len(z)) for 2-D counts. Not normalised
        unless the Nz is (see normalised()).
        """
        z = np.asarray(z, dtype=np.float64)
        idx = np.searchsorted(self.edges, z, side="right") - 1
        inside = (z >= self.edges[0]) & (z < self.edges[-1])
        # the last edge belongs to the last bin
        last = z == self.edges[-1]
        idx = np.where(last, len(self.edges) - 2, idx)
        inside = inside | last
        idx = np.clip(idx, 0, len(self.edges) - 2)
        return np.where(inside, self._density()[..., idx], 0.0)

    def normalised(self) -> "Nz":
        """Copy whose density integrates to 1 (per bin row when 2-D)."""
        total = (self.counts * 1.0).sum(axis=-1, keepdims=True)
        if np.any(total <= 0):
            raise ValidationError("cannot normalise an Nz with zero total counts")
        return Nz(self.edges, self.counts / total, self.method, self.bins)


@dataclass(frozen=True, eq=False)
class Window:
    """Survey window as a HEALPix weight map in [0, 1]."""
    nside: int
    nest: bool
    mask: np.ndarray
    source: Literal["mask", "footprint"]
    systematics: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        healpix._check_nside(self.nside)
        npix = 12 * self.nside ** 2
        mask = np.array(self.mask, dtype=np.float32)
        if mask.shape != (npix,):
            raise ValidationError(f"mask must have length {npix}, got shape {mask.shape}")
        if not np.all(np.isfinite(mask)) or mask.min() < 0 or mask.max() > 1:
            raise ValidationError("mask values must be finite and in [0, 1]")
        if self.source not in ("mask", "footprint"):
            raise ValidationError(f"Window source must be 'mask' or 'footprint', got {self.source!r}")
        systematics = {}
        for k, v in dict(self.systematics).items():
            a = np.array(v)
            if a.shape != (npix,):
                raise ValidationError(f"systematic {k!r} must have length {npix}, got shape {a.shape}")
            systematics[str(k)] = a
        _set(self, "nest", bool(self.nest))
        _set(self, "mask", mask)
        _set(self, "systematics", systematics)

    @property
    def area_sr(self) -> float:
        return float(self.mask.astype(np.float64).sum() * hp.nside2pixarea(self.nside))

    @property
    def fraction(self) -> float:
        return float(self.mask.astype(np.float64).sum() / len(self.mask))

    def contains(self, ra, dec) -> np.ndarray:
        pix = healpix.radec_to_pix(self.nside, ra, dec, nest=self.nest)
        return self.mask[pix] > 0

    @classmethod
    def from_mask(cls, array, *, nside: int, nest: bool = True) -> "Window":
        return cls(nside, nest, np.asarray(array), "mask")

    @classmethod
    def footprint(cls, ra, dec, *, nside: int) -> "Window":
        """Window of the pixels holding at least one point."""
        healpix._check_nside(nside)
        pix = healpix.radec_to_pix(nside, ra, dec, nest=True)
        mask = np.zeros(12 * nside ** 2, np.float32)
        mask[pix] = 1.0
        return cls(nside, True, mask, "footprint")


@dataclass(frozen=True, eq=False)
class Regions:
    """Partition of the window into jackknife regions (pixel -> region id)."""
    nside: int
    nest: bool
    n_regions: int
    pixel_region: np.ndarray
    method: str

    def __post_init__(self):
        healpix._check_nside(self.nside)
        pr = np.array(self.pixel_region, dtype=np.int32)
        if pr.shape != (12 * self.nside ** 2,):
            raise ValidationError(
                f"pixel_region must have length {12 * self.nside ** 2}, got shape {pr.shape}")
        if self.n_regions < 1 or (pr.size and pr.max() >= self.n_regions) or (pr.size and pr.min() < -1):
            raise ValidationError("pixel_region values must lie in [-1, n_regions)")
        _set(self, "nest", bool(self.nest))
        _set(self, "n_regions", int(self.n_regions))
        _set(self, "pixel_region", pr)

    def __eq__(self, other):
        if not isinstance(other, Regions):
            return NotImplemented
        return (self.nside == other.nside and self.nest == other.nest
                and self.n_regions == other.n_regions and self.method == other.method
                and np.array_equal(self.pixel_region, other.pixel_region))

    __hash__ = None

    def assign(self, ra, dec) -> np.ndarray:
        """Region id of each (ra, dec); -1 outside the window."""
        pix = healpix.radec_to_pix(self.nside, ra, dec, nest=self.nest)
        return self.pixel_region[pix].astype(np.int32)

    @classmethod
    def from_window(cls, window: Window, n_regions: int, *, nside: int | None = None,
                    seed: int = 0) -> "Regions":
        """k-means on unit vectors of masked pixels (deterministic for a seed).

        *nside* (default window.nside) must not exceed window.nside; the mask
        is coarsened by max (a coarse pixel is masked if any child is).

        Regions are k-means clusters of pixel centres, so they are not
        equal-area. The empty-cluster ValidationError only occurs when
        n_regions is close to the number of masked pixels.
        """
        nside = window.nside if nside is None else int(nside)
        healpix._check_nside(nside)
        if nside > window.nside:
            raise ValidationError(
                f"region nside {nside} exceeds window nside {window.nside}")
        if n_regions < 1:
            raise ValidationError("n_regions must be >= 1")
        mask = window.mask if window.nest else hp.reorder(window.mask, r2n=True)
        npix = 12 * nside ** 2
        if nside == window.nside:
            masked = mask > 0
        else:
            fine = np.flatnonzero(mask > 0)
            masked = np.zeros(npix, bool)
            masked[healpix.coarsen(fine, window.nside, nside)] = True
        pix = np.flatnonzero(masked)
        if n_regions > len(pix):
            raise ValidationError(
                f"n_regions={n_regions} exceeds the {len(pix)} masked pixels at nside={nside}")
        vec = np.array(hp.pix2vec(nside, pix, nest=True)).T
        _, labels = kmeans2(vec, n_regions, minit="++", seed=np.random.default_rng(seed))
        # kmeans2 may leave empty clusters; refuse rather than return ids with holes
        used = np.unique(labels)
        if len(used) != n_regions:
            raise ValidationError(
                f"k-means produced {len(used)} non-empty regions for n_regions={n_regions}; "
                "use fewer regions")
        pixel_region = np.full(npix, -1, np.int32)
        pixel_region[pix] = labels
        return cls(nside, True, n_regions, pixel_region, f"kmeans(seed={seed})")
