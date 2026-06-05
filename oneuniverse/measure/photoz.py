"""Photo-z kernel: P1's per-object p(z) attached as the measure atom."""
from __future__ import annotations

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.pdf import ProbabilisticRedshift


def attach_photoz(view: DatasetView) -> ProbabilisticRedshift:
    """Return the per-object photo-z kernel (qp) from an OUF PDF dataset."""
    return view.load_pdf()
