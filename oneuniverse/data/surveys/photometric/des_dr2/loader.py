"""
DES DR2 photometric catalog loader.

Data expected at: {DATA_ROOT}/photometric/des_dr2/

Column mapping (DES native → oneuniverse):
    RA, DEC         → ra, dec
    DNF_ZMC_SOF     → z_phot
    DNF_ZSIGMA_SOF  → z_phot_err
    BPZ_ZPHOT       → (alternative photo-z)
    MAG_AUTO_{G,R,I,Z,Y} → mag_g, mag_r, mag_i, mag_z
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data._registry import register


@register
class DESDR2Loader(BaseSurveyLoader):

    config = SurveyConfig(
        name="des_dr2",
        survey_type="photometric",
        description="DES DR2 — grizY photometry, photo-z, ~700M objects, 5000 deg²",
        column_groups=("core", "photometric"),
        characteristic_fields={
            "z_phot":     ("f4", "",    "DNF photometric redshift"),
            "z_phot_err": ("f4", "",    "DNF photo-z uncertainty"),
            "mag_g":      ("f4", "mag", "g-band MAG_AUTO (AB)"),
            "mag_r":      ("f4", "mag", "r-band MAG_AUTO (AB)"),
            "mag_i":      ("f4", "mag", "i-band MAG_AUTO (AB)"),
            "mag_z":      ("f4", "mag", "z-band MAG_AUTO (AB)"),
            "spread_model_i": ("f4", "", "Star-galaxy separator (i-band)"),
        },
        data_filename="des_dr2_gold.parquet",
        data_format="parquet",
        reference="DES Collaboration 2021, ApJS 255 20",
        url="https://des.ncsa.illinois.edu/releases/dr2",
        sky_fraction=0.12,
        z_range=(0.1, 1.5),
        n_objects_approx=700_000_000,
    )

    def _load_raw(self, data_path: Optional[Path] = None, **kwargs) -> pd.DataFrame:
        if data_path is None:
            raise FileNotFoundError(
                "No data root configured for des_dr2. "
                "Set ONEUNIVERSE_DATA_ROOT or pass data_path= explicitly."
            )
        raise NotImplementedError(
            f"DES DR2 loader not yet implemented. "
            f"Expected data at: {data_path / self.config.data_filename}"
        )
