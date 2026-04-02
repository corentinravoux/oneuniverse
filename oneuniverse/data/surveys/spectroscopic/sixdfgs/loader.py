"""
6dF Galaxy Survey loader.

Data expected at: {DATA_ROOT}/spectroscopic/sixdfgs/

Column mapping (6dFGS native → oneuniverse):
    RAJ2000, DEJ2000  → ra, dec
    cz                → cz_cmb (after CMB correction)
    zfinal            → z_spec
    e_cz              → z_spec_err (via cz_err / c)
    quality           → z_spec_qual  (keep >= 3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data._registry import register


@register
class SixdFGSLoader(BaseSurveyLoader):

    config = SurveyConfig(
        name="sixdfgs",
        survey_type="spectroscopic",
        description="6dF Galaxy Survey — K < 12.65, z ~ 0–0.2, ~136K gal (southern sky)",
        column_groups=("core", "spectroscopic"),
        characteristic_fields={
            "z_spec_qual": ("i1",  "",     "Quality flag (3+ = reliable)"),
            "mag_k":       ("f4",  "mag",  "K-band magnitude (Vega)"),
            "prog_id":     ("U8",  "",     "6dFGS program ID"),
        },
        data_filename="sixdfgs_dr3.fits",
        data_format="fits",
        reference="Jones+2009 MNRAS 399 683",
        url="http://www.6dfgs.net/",
        sky_fraction=0.41,
        z_range=(0.001, 0.20),
        n_objects_approx=136_000,
    )

    def _load_raw(self, data_path: Optional[Path] = None, **kwargs) -> pd.DataFrame:
        if data_path is None:
            raise FileNotFoundError(
                "No data root configured for sixdfgs. "
                "Set ONEUNIVERSE_DATA_ROOT or pass data_path= explicitly."
            )
        raise NotImplementedError(
            f"6dFGS loader not yet implemented. "
            f"Expected data at: {data_path / self.config.data_filename}"
        )
