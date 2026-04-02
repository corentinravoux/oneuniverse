"""
DESI Bright Galaxy Survey loader.

Data expected at: {DATA_ROOT}/spectroscopic/desi_bgs/

Column mapping (DESI native → oneuniverse):
    TARGET_RA, TARGET_DEC  → ra, dec
    Z                      → z, z_spec
    ZERR                   → z_spec_err
    ZWARN                  → z_spec_qual
    WEIGHT_ZFAIL           → w_noz
    WEIGHT_SYS             → w_sys
    WEIGHT_COMP            → w_comp
    WEIGHT_FKP             → w_fkp
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data._registry import register


@register
class DESIBGSLoader(BaseSurveyLoader):

    config = SurveyConfig(
        name="desi_bgs",
        survey_type="spectroscopic",
        description="DESI Bright Galaxy Survey DR1 — r < 19.5, z ~ 0.01–0.6, ~15M gal",
        column_groups=("core", "spectroscopic"),
        characteristic_fields={
            "z_spec_qual": ("i1",  "",    "ZWARN flag (0 = good)"),
            "ntile":       ("i2",  "",    "Number of overlapping tiles (fiber passes)"),
            "w_fkp":       ("f4",  "",    "WEIGHT_FKP"),
            "w_comp":      ("f4",  "",    "WEIGHT_COMP"),
            "w_noz":       ("f4",  "",    "WEIGHT_ZFAIL"),
            "w_sys":       ("f4",  "",    "WEIGHT_SYS (imaging systematics)"),
        },
        data_filename="desi_bgs_dr1.fits",
        data_format="fits",
        reference="DESI Collaboration 2024, DR1",
        url="https://data.desi.lbl.gov/",
        sky_fraction=0.34,
        z_range=(0.01, 0.60),
        n_objects_approx=15_000_000,
    )

    def _load_raw(self, data_path: Optional[Path] = None, **kwargs) -> pd.DataFrame:
        if data_path is None:
            raise FileNotFoundError(
                "No data root configured for desi_bgs. "
                "Set ONEUNIVERSE_DATA_ROOT or pass data_path= explicitly."
            )
        raise NotImplementedError(
            f"DESI BGS loader not yet implemented. "
            f"Expected data at: {data_path / self.config.data_filename}"
        )
