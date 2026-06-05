"""Read a SIGHTLINE OUF dataset into a measure Sightline product."""
from __future__ import annotations

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.measure.dataproduct import Sightline
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.regions import assign_regions


def sightline_from_view(view: DatasetView, *, nside_region: int = 16,
                        id_column: str = "sightline_id") -> Sightline:
    """Build a Sightline from per-LOS metadata + per-pixel δ/weight/continuum."""
    if view.geometry is not DataGeometry.SIGHTLINE:
        raise ValueError(
            f"sightline_from_view: expected SIGHTLINE, got "
            f"{view.geometry.value!r}")
    los = view.objects_table().to_pandas()
    pix = view.read()
    grp = pix.groupby(id_column)
    ids = los[id_column].to_numpy()
    delta = [grp.get_group(i)["delta"].to_numpy() for i in ids]
    mask = [grp.get_group(i)["weight"].to_numpy() for i in ids]
    cont = ([grp.get_group(i)["cont"].to_numpy() for i in ids]
            if "cont" in pix.columns else None)
    region = assign_regions(los["ra"].to_numpy(), los["dec"].to_numpy(),
                            nside=nside_region)
    los = los.copy(); los["region_id"] = region
    meta = ProductMetadata(frame="icrs", epoch=2000.0, length_unit="deg",
                           nside_region=int(nside_region))
    return Sightline(los=los, delta=delta, mask=mask, continuum=cont,
                     region_map=region, metadata=meta,
                     provenance=Provenance(dataset_ids=(view.survey_name,)))
