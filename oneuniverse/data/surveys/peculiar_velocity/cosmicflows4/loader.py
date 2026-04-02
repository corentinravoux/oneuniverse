"""
CosmicFlows-4 peculiar velocity compilation loader.

Data expected at: {DATA_ROOT}/peculiar_velocity/cosmicflows4/

Column mapping (CF4 native → oneuniverse):
    RA, Dec (or SGL, SGB)  → ra, dec
    Vcmb                   → cz_cmb
    Dist                   → mu  (via 5 log10(Dist) + 25)
    e_Dist                 → mu_err
    Ty                     → mu_method (1=TF, 2=FP, 3=SBF, 4=SNIa, 5=misc)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data._registry import register


@register
class CosmicFlows4Loader(BaseSurveyLoader):

    config = SurveyConfig(
        name="cosmicflows4",
        survey_type="peculiar_velocity",
        description="CosmicFlows-4 — ~56K galaxies, z < 0.07, TF/FP/SBF/SNIa compilation",
        column_groups=("core", "peculiar_velocity"),
        characteristic_fields={
            "v_pec":     ("f4", "km/s", "Peculiar velocity from distance indicator"),
            "mu":        ("f4", "mag",  "Distance modulus"),
            "mu_err":    ("f4", "mag",  "Distance modulus uncertainty"),
            "mu_method": ("U8", "",     "Distance method: TF/FP/SBF/SNIa/misc"),
            "nest":      ("i4", "",     "Nest/group membership ID"),
            "sgl":       ("f4", "deg",  "Supergalactic longitude"),
            "sgb":       ("f4", "deg",  "Supergalactic latitude"),
        },
        data_filename="cf4_grouped.fits",
        data_format="fits",
        reference="Tully+2023",
        url="https://edd.ifa.hawaii.edu/CF4/",
        sky_fraction=0.90,
        z_range=(0.0, 0.07),
        n_objects_approx=56_000,
    )

    def _load_raw(self, data_path: Optional[Path] = None, **kwargs) -> pd.DataFrame:
        if data_path is None:
            raise FileNotFoundError(
                "No data root configured for cosmicflows4. "
                "Set ONEUNIVERSE_DATA_ROOT or pass data_path= explicitly."
            )
        raise NotImplementedError(
            f"CosmicFlows-4 loader not yet implemented. "
            f"Expected data at: {data_path / self.config.data_filename}"
        )
