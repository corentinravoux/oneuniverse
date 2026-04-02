"""
DESI Peculiar Velocity Survey loader.

Data expected at: {DATA_ROOT}/peculiar_velocity/desi_pv/

Column mapping (DESI PV native → oneuniverse):
    TARGET_RA, TARGET_DEC  → ra, dec
    Z                      → z_spec
    DN4000, SIGMA_STAR     → (FP pipeline inputs)
    MU_FP / MU_TF          → mu
    MU_ERR                 → mu_err
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data._registry import register


@register
class DESIPVLoader(BaseSurveyLoader):

    config = SurveyConfig(
        name="desi_pv",
        survey_type="peculiar_velocity",
        description="DESI Peculiar Velocity — ~150K FP+TF galaxies, z < 0.1",
        column_groups=("core", "spectroscopic", "peculiar_velocity"),
        characteristic_fields={
            "v_pec":      ("f4", "km/s", "Peculiar velocity"),
            "mu":         ("f4", "mag",  "Distance modulus (FP or TF)"),
            "mu_err":     ("f4", "mag",  "Distance modulus uncertainty"),
            "mu_method":  ("U8", "",     "FP or TF"),
            "dn4000":     ("f4", "",     "4000-A break index (FP input)"),
            "sigma_star": ("f4", "km/s", "Stellar velocity dispersion (FP input)"),
            "logw":       ("f4", "",     "Log HI linewidth (TF input)"),
        },
        data_filename="desi_pv_dr1.fits",
        data_format="fits",
        reference="DESI PV papers 2024–2025",
        url="https://data.desi.lbl.gov/",
        sky_fraction=0.34,
        z_range=(0.0, 0.10),
        n_objects_approx=150_000,
    )

    def _load_raw(self, data_path: Optional[Path] = None, **kwargs) -> pd.DataFrame:
        if data_path is None:
            raise FileNotFoundError(
                "No data root configured for desi_pv. "
                "Set ONEUNIVERSE_DATA_ROOT or pass data_path= explicitly."
            )
        raise NotImplementedError(
            f"DESI PV loader not yet implemented. "
            f"Expected data at: {data_path / self.config.data_filename}"
        )
