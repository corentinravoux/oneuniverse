"""build_galaxy_clustering — the galaxy-clustering P1->P2 connection (9 steps)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

from oneuniverse.combine.weights import Weight
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.dataproduct import PointSet
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.nz import nz_from_spec_z
from oneuniverse.measure.randoms import generate_randoms, randoms_from_view
from oneuniverse.measure.regions import assign_regions
from oneuniverse.measure.select import select_clean
from oneuniverse.measure.spec import MeasurementSpec
from oneuniverse.measure.weighting import assemble_weight
from oneuniverse.measure.window import footprint_from_positions


def build_galaxy_clustering(
    view: DatasetView, *, tracer: str = "gal",
    z_range: Tuple[float, float],
    weights: Sequence[Weight],
    nz_edges,
    nside_window: int = 256,
    nside_region: int = 8,
    quality_column: Optional[str] = "quality", quality_min: float = 1.0,
    randoms: Union[str, DatasetView] = "generate",
    n_randoms: int = 0, seed: int = 0,
    statistic: str = "pk_multipole",
) -> MeasurementSet:
    """OUF POINT view -> galaxy-clustering MeasurementSet (cosmology-free)."""
    # 1-2 select + clean
    cat = select_clean(view, z_range=z_range, quality_column=quality_column,
                       quality_min=quality_min)
    # 3 weights
    cat, recipe = assemble_weight(cat, weights)
    w = cat["weight"].to_numpy()
    # 5 window
    win = footprint_from_positions(cat["ra"].to_numpy(), cat["dec"].to_numpy(),
                                   nside=nside_window)
    # 6 n(z) (weighted)
    nz = nz_from_spec_z(cat["z"].to_numpy(), edges=nz_edges, weights=w)
    # 4 randoms (ingest | generate)
    if isinstance(randoms, DatasetView):
        rnd, source = randoms_from_view(randoms)
    elif randoms == "generate":
        rnd, source = generate_randoms(win, nz, n_randoms=n_randoms, seed=seed)
    else:
        rnd, source = None, None
    # 8 region map (shared scheme; applied to data + randoms)
    region = assign_regions(cat["ra"].to_numpy(), cat["dec"].to_numpy(),
                            nside=nside_region)
    cat = cat.copy(); cat["region_id"] = region
    if rnd is not None:
        rnd = rnd.copy()
        rnd["region_id"] = assign_regions(rnd["ra"].to_numpy(),
                                          rnd["dec"].to_numpy(),
                                          nside=nside_region)
    meta = ProductMetadata(frame="icrs", epoch=2000.0, length_unit="deg",
                           nside_region=int(nside_region))
    prov = Provenance(dataset_ids=(view.survey_name,), weight_recipe=recipe,
                      randoms_source=source, nz_method=nz.method)
    ps = PointSet(catalog=cat, randoms=rnd, nz=nz, window=win,
                  region_map=region, metadata=meta, provenance=prov)
    spec = MeasurementSpec(tracers=(tracer,), pairs=((tracer, tracer),),
                           statistic=statistic, estimator_family="clustering")
    return MeasurementSet(products={tracer: ps}, spec=spec, metadata=meta)
