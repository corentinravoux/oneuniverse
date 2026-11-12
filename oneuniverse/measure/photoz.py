"""Photo-z attachment and tomographic n(z) for point-set Products.

Nothing here depends on a cosmology.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Literal

import numpy as np

from oneuniverse._core.errors import FormatError, ValidationError
from oneuniverse.data.pdf import PdfSpec
from oneuniverse.data.view import DatasetView
from oneuniverse.measure.atoms import Nz
from oneuniverse.measure.product import Product


def attach_photoz(product: Product, view: DatasetView, *, n_draws: int = 1,
                  seed: int = 0) -> Product:
    """Add ``z_draws`` (n, n_draws), ``z_pdf_mean`` and ``z_pdf_std`` (n,) extras.

    PDFs are read back with ``view.take`` at the catalog's (partition, offset),
    so extras row i is catalog row i whatever cuts the product went through.
    The view must be the version the product was built from (FormatError if
    not). Draws are deterministic for a seed.
    """
    if int(n_draws) < 1:
        raise ValidationError("n_draws must be >= 1")
    if list(view.version) != product.provenance.get("version"):
        raise FormatError(
            f"product built from another version of {product.provenance.get('dataset')!r}")
    ext = view.manifest.extension("pdf")
    if ext is None:
        raise FormatError("dataset has no 'pdf' extension")
    spec = PdfSpec.from_dict(ext)
    cat = product.catalog
    rows = view.take(cat["partition"].to_numpy(), cat["offset"].to_numpy(),
                     columns=[spec.value_column])
    pdf = view.load_pdf(rows)
    return (product
            .with_extra("z_draws", np.asarray(pdf.sample(int(n_draws), int(seed)), np.float64))
            .with_extra("z_pdf_mean", np.asarray(pdf.mean(), np.float64))
            .with_extra("z_pdf_std", np.asarray(pdf.std(), np.float64)))


def tomographic_nz(product: Product, *, bin_edges: Sequence[float], nz_edges: Sequence[float],
                   by: Literal["z_pdf_mean", "z"] = "z_pdf_mean") -> tuple[Product, Nz]:
    """Assign tomographic bins and stack the per-bin n(z) from the photo-z draws.

    ``tomo_bin`` (int32) is the bin of the *by* quantity in *bin_edges*, -1 outside.
    Nz counts have shape (n_bins, k): the catalog-weighted histogram of every draw
    of the bin's members, divided by n_draws; method "photoz_stack".
    Needs the ``z_draws`` extra (and ``z_pdf_mean`` when by="z_pdf_mean").
    """
    if by not in ("z_pdf_mean", "z"):
        raise ValidationError(f"by must be 'z_pdf_mean' or 'z', got {by!r}")
    if "z_draws" not in product.extras or (by == "z_pdf_mean" and "z_pdf_mean" not in product.extras):
        raise ValidationError("tomographic_nz needs the photo-z extras z_draws (and z_pdf_mean): "
                              "call attach_photoz first")
    be = np.asarray(bin_edges, np.float64)
    if be.ndim != 1 or be.size < 2 or np.any(np.diff(be) <= 0):
        raise ValidationError("bin_edges must be strictly increasing with >= 2 values")
    draws = np.asarray(product.extras["z_draws"], np.float64)
    q = (product.catalog["z"].to_numpy() if by == "z"
         else np.asarray(product.extras["z_pdf_mean"], np.float64))
    n_bins = be.size - 1
    inb = (q >= be[0]) & (q < be[-1])
    tomo = np.where(inb, np.searchsorted(be, q, side="right") - 1, -1).astype(np.int32)
    w = product.catalog["weight"].to_numpy()
    ne = np.asarray(nz_edges, np.float64)
    counts = np.zeros((n_bins, ne.size - 1))
    n_draws = draws.shape[1]
    for b in range(n_bins):
        m = tomo == b
        if m.any():
            counts[b] = np.histogram(draws[m].ravel(), ne,
                                     weights=np.repeat(w[m], n_draws))[0] / n_draws
    nz = Nz(ne, counts, "photoz_stack", tuple(range(n_bins)))
    return dataclasses.replace(product.with_extra("tomo_bin", tomo), nz=nz), nz

