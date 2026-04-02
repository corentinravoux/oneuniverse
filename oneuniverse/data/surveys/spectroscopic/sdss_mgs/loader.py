"""
SDSS Main Galaxy Sample loader.

Data expected at: {DATA_ROOT}/spectroscopic/sdss_mgs/sdss_mgs_dr17.fits

Column mapping (SDSS native → oneuniverse):
    RA, DEC         → ra, dec
    Z               → z, z_spec
    Z_ERR           → z_spec_err
    ZWARNING        → z_spec_qual  (keep 0 = good)
    PLUG_RA/DEC     → (ignored, use calibrated RA/DEC)
    SPECTROFLUX     → mag_r (via nanomaggies conversion)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data._registry import register


@register
class SDSSMGSLoader(BaseSurveyLoader):

    config = SurveyConfig(
        name="sdss_mgs",
        survey_type="spectroscopic",
        description="SDSS Main Galaxy Sample DR17 — r < 17.77, z ~ 0.01–0.25, ~700K gal",
        column_groups=("core", "spectroscopic"),
        characteristic_fields={
            "z_spec_qual": ("i1", "",     "ZWARNING flag (0 = good)"),
            "mag_r":       ("f4", "mag",  "r-band Petrosian magnitude (AB)"),
            "mag_g":       ("f4", "mag",  "g-band Petrosian magnitude (AB)"),
            "ebv":         ("f4", "mag",  "E(B-V) Galactic extinction (SF11)"),
        },
        data_filename="sdss_mgs_dr17.fits",
        data_format="fits",
        reference="York+2000 AJ 120 1579; Strauss+2002 AJ 124 1810",
        url="https://www.sdss.org/dr17/",
        sky_fraction=0.24,
        z_range=(0.01, 0.25),
        n_objects_approx=700_000,
    )

    def _load_raw(self, data_path: Optional[Path] = None, **kwargs) -> pd.DataFrame:
        if data_path is None:
            raise FileNotFoundError(
                "No data root configured for sdss_mgs. "
                "Set ONEUNIVERSE_DATA_ROOT or pass data_path= explicitly."
            )
        raise NotImplementedError(
            f"SDSS MGS loader not yet implemented. "
            f"Expected data at: {data_path / self.config.data_filename}"
        )
