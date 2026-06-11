"""The shared PointSet preparation spine (review S5).

Every PointSet builder runs the same first half of the 9-step transform —
select → clean → (weights) → window → region → metadata. Before this module
each builder re-implemented it (~80 duplicated lines across clustering /
lensing / pvsn); now they call :func:`prepare_pointset` and add only their
probe-specific atoms (n(z), randoms, shapes, photo-z, distances).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from oneuniverse.combine.weights import Weight
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.metadata import ProductMetadata
from oneuniverse.measure.regions import assign_regions
from oneuniverse.measure.select import select_clean
from oneuniverse.measure.weighting import assemble_weight
from oneuniverse.measure.window import Window, footprint_from_positions


def prepare_pointset(view: DatasetView, *,
                     z_range: Optional[Tuple[float, float]] = None,
                     weights: Optional[Sequence[Weight]] = None,
                     nside_window: int = 256, nside_region: int = 8,
                     quality_column: Optional[str] = None,
                     quality_min: float = 1.0,
                     frame: str = "icrs", epoch: float = 2000.0):
    """Common PointSet preparation. Returns
    ``(catalog, window, region_map, metadata, weight_recipe)``.

    The catalog is cleaned, optionally weight-assembled (``weight`` column +
    recipe), carries ``region_id``, and the window/region are derived from its
    positions. Probe builders attach their specific atoms on top.
    """
    cat = select_clean(view, z_range=z_range, quality_column=quality_column,
                       quality_min=quality_min)
    recipe: Tuple[str, ...] = ()
    if weights:
        cat, recipe = assemble_weight(cat, weights)
    win: Window = footprint_from_positions(
        cat["ra"].to_numpy(), cat["dec"].to_numpy(), nside=nside_window)
    region = assign_regions(cat["ra"].to_numpy(), cat["dec"].to_numpy(),
                            nside=nside_region)
    cat = cat.copy()
    cat["region_id"] = region
    meta = ProductMetadata(frame=frame, epoch=epoch, length_unit="deg",
                           nside_region=int(nside_region))
    return cat, win, region, meta, recipe
