"""Per-source light curves from P1's LIGHTCURVE geometry."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry


@dataclass
class LightcurveSet:
    objects: pd.DataFrame            # one row per source (objects_table)
    epochs: pd.DataFrame             # per-epoch rows (object_id, mjd, flux, filter)
    id_column: str = "object_id"

    @property
    def object_ids(self) -> np.ndarray:
        return self.objects[self.id_column].to_numpy()

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    def for_object(self, oid) -> pd.DataFrame:
        return self.epochs[self.epochs[self.id_column] == oid]


def lightcurves_from_view(view: DatasetView, *, id_column: str = "object_id"
                          ) -> LightcurveSet:
    """Read a LIGHTCURVE OUF dataset into a LightcurveSet."""
    if view.geometry is not DataGeometry.LIGHTCURVE:
        raise ValueError(
            f"lightcurves_from_view: expected LIGHTCURVE, got "
            f"{view.geometry.value!r}")
    objects = view.objects_table().to_pandas()
    epochs = view.read()
    return LightcurveSet(objects=objects, epochs=epochs, id_column=id_column)
