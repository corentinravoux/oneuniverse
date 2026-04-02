"""
Pantheon+ Type Ia supernovae loader.

Data expected at: {DATA_ROOT}/snia/pantheonplus/

Column mapping (Pantheon+ native → oneuniverse):
    RA, DECL           → ra, dec
    zHD                → z_cmb  (Hubble-diagram redshift, CMB frame)
    zHEL               → z_helio
    MU_SH0ES           → mu
    MU_ERR_VPEC        → mu_err  (includes vpec uncertainty)
    x1, c              → (SALT2 stretch, color — characteristic fields)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data._registry import register


@register
class PantheonPlusLoader(BaseSurveyLoader):

    config = SurveyConfig(
        name="pantheonplus",
        survey_type="snia",
        description="Pantheon+ — ~1700 SNe Ia, z = 0.01–2.3, standardizable candles",
        column_groups=("core", "peculiar_velocity"),
        characteristic_fields={
            "mu":        ("f4", "mag",   "SALT2 distance modulus (SH0ES calibration)"),
            "mu_err":    ("f4", "mag",   "Distance modulus uncertainty (incl. vpec)"),
            "x1":        ("f4", "",      "SALT2 stretch parameter"),
            "c":         ("f4", "",      "SALT2 color parameter"),
            "host_logmass": ("f4", "",   "Host galaxy log stellar mass"),
            "is_calibrator": ("i1", "",  "1 if Cepheid-calibrated host"),
        },
        data_filename="Pantheon+SH0ES.dat",
        data_format="csv",
        reference="Scolnic+2022 ApJ 938 113; Brout+2022 ApJ 938 110",
        url="https://pantheonplussh0es.github.io/",
        sky_fraction=0.95,
        z_range=(0.01, 2.30),
        n_objects_approx=1_700,
    )

    def _load_raw(self, data_path: Optional[Path] = None, **kwargs) -> pd.DataFrame:
        if data_path is None:
            raise FileNotFoundError(
                "No data root configured for pantheonplus. "
                "Set ONEUNIVERSE_DATA_ROOT or pass data_path= explicitly."
            )
        raise NotImplementedError(
            f"Pantheon+ loader not yet implemented. "
            f"Expected data at: {data_path / self.config.data_filename}"
        )
