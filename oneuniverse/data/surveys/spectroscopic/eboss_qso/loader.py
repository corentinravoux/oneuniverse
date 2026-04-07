"""
eBOSS DR16Q Superset quasar catalog loader.

Data: DR16Q_Superset_v3.fits (~1.44M rows, 97 columns)
Reference: Lyke et al. 2020, ApJS 250, 8

This catalog contains the superset of all SDSS/BOSS/eBOSS quasar targets.
The key classification column IS_QSO_FINAL selects confirmed quasars.

Column mapping (FITS → oneuniverse):
    RA, DEC              → ra, dec
    Z                    → z  (best available redshift)
    IS_QSO_FINAL         → is_qso (1=confirmed QSO, 0=not QSO, -2=bad, 2=ambiguous)
    SOURCE_Z             → source_z (VI, PIPE, DR6Q_HW, DR7QV_SCH, DR12QV)
    Z_PIPE               → z_pipe
    Z_PCA                → z_pca
    Z_VI                 → z_vi
    Z_QN                 → z_qn (QuasarNET redshift)
    Z_CONF               → z_conf (visual inspection confidence 0-3)
    ZWARNING             → zwarning (pipeline quality; 0=good)
    CLASS_PERSON         → class_person (VI class: 3=QSO, 30=BAL, 1=star, 4=galaxy)
    BAL_PROB             → bal_prob
    BI_CIV               → bi_civ
    Z_MGII/CIII/CIV/LYA → z_mgii, z_ciii, z_civ, z_lya
    PSFMAG[5]            → psfmag_u/g/r/i/z
    EXTINCTION[5]        → extinction_u/g/r/i/z
    PLATE, MJD, FIBERID  → plate, mjd, fiberid
    SN_MEDIAN_ALL        → sn_median
    THING_ID             → thing_id
    SDSS_NAME            → sdss_name
    BOSS_TARGET1         → boss_target1
    EBOSS_TARGET0/1/2    → eboss_target0/1/2
    Z_DLA[5], NHI_DLA[5], CONF_DLA[5] → n_dla (count of valid DLAs)

Sentinel values in the original FITS:
    -1   = not measured / not applicable
    -999 = catastrophic failure or missing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data._io import read_fits
from oneuniverse.data._registry import register

logger = logging.getLogger(__name__)

# Columns to read from the FITS file (selective I/O for speed)
_FITS_COLUMNS = [
    "RA", "DEC", "Z", "IS_QSO_FINAL", "SOURCE_Z",
    "Z_PIPE", "Z_PCA", "Z_VI", "Z_QN", "Z_CONF", "ZWARNING",
    "CLASS_PERSON", "BAL_PROB", "BI_CIV",
    "Z_MGII", "Z_CIII", "Z_CIV", "Z_LYA",
    "Z_HALPHA", "Z_HBETA",
    "PLATE", "MJD", "FIBERID", "THING_ID", "SDSS_NAME",
    "BOSS_TARGET1", "EBOSS_TARGET0", "EBOSS_TARGET1", "EBOSS_TARGET2",
    "SN_MEDIAN_ALL", "LAMBDA_EFF", "NSPEC",
    "PSFMAG", "EXTINCTION",
    "Z_DLA", "NHI_DLA", "CONF_DLA",
]

# Mapping from FITS column names to oneuniverse names
_COLUMN_MAP = {
    "RA": "ra",
    "DEC": "dec",
    "Z": "z",
    "IS_QSO_FINAL": "is_qso",
    "SOURCE_Z": "source_z",
    "Z_PIPE": "z_pipe",
    "Z_PCA": "z_pca",
    "Z_VI": "z_vi",
    "Z_QN": "z_qn",
    "Z_CONF": "z_conf",
    "ZWARNING": "zwarning",
    "CLASS_PERSON": "class_person",
    "BAL_PROB": "bal_prob",
    "BI_CIV": "bi_civ",
    "Z_MGII": "z_mgii",
    "Z_CIII": "z_ciii",
    "Z_CIV": "z_civ",
    "Z_LYA": "z_lya",
    "Z_HALPHA": "z_halpha",
    "Z_HBETA": "z_hbeta",
    "PLATE": "plate",
    "MJD": "mjd",
    "FIBERID": "fiberid",
    "THING_ID": "thing_id",
    "SDSS_NAME": "sdss_name",
    "BOSS_TARGET1": "boss_target1",
    "EBOSS_TARGET0": "eboss_target0",
    "EBOSS_TARGET1": "eboss_target1",
    "EBOSS_TARGET2": "eboss_target2",
    "SN_MEDIAN_ALL": "sn_median",
    "LAMBDA_EFF": "lambda_eff",
    "NSPEC": "nspec",
}


def _count_valid_dlas(z_dla: np.ndarray) -> np.ndarray:
    """Count valid DLA detections per sightline (sentinel: z_dla == -1)."""
    return np.sum(z_dla > 0, axis=1).astype(np.int8)


@register
class EbossQSOLoader(BaseSurveyLoader):

    config = SurveyConfig(
        name="eboss_qso",
        survey_type="spectroscopic",
        description=(
            "eBOSS DR16Q Superset v3 — 1.44M targets, ~920K confirmed QSOs, "
            "z ~ 0.1–7.0, SDSS/BOSS/eBOSS quasar catalog (Lyke+2020)"
        ),
        column_groups=("core", "spectroscopic", "qso"),
        characteristic_fields={
            "is_qso":       ("i1", "",     "QSO classification (1=QSO, 0=not, -2=bad, 2=ambig)"),
            "source_z":     ("U16", "",    "Best-z source: VI, PIPE, DR6Q_HW, DR7QV_SCH, DR12QV"),
            "z_conf":       ("i1", "",     "VI confidence: 0=no VI, 1=low, 2=medium, 3=high"),
            "bal_prob":     ("f4", "",     "BAL probability from CNN classifier"),
            "bi_civ":       ("f8", "km/s", "CIV balnicity index (>0 = traditional BAL)"),
            "class_person": ("i2", "",     "VI class: 3=QSO, 30=BAL QSO, 1=star, 4=galaxy"),
            "n_dla":        ("i1", "",     "Number of DLA absorbers (0-5)"),
            "thing_id":     ("i8", "",     "SDSS unique object identifier"),
        },
        data_subpath="spectroscopic/eboss/qso",
        data_filename="DR16Q_Superset_v3.fits",
        data_format="fits",
        reference="Lyke+2020 ApJS 250 8",
        url="https://www.sdss.org/dr16/algorithms/qso-catalog/",
        sky_fraction=0.24,
        z_range=(0.0, 7.0),
        n_objects_approx=920_000,
    )

    def _load_raw(
        self,
        data_path: Optional[Path] = None,
        qso_only: bool = True,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Load the eBOSS DR16Q catalog.

        Parameters
        ----------
        data_path : Path or None
            Directory containing DR16Q_Superset_v3.fits.
        qso_only : bool
            If True (default), keep only IS_QSO_FINAL == 1.
        z_min, z_max : float or None
            Redshift pre-filter applied before returning.
        """
        if data_path is None:
            raise FileNotFoundError(
                "No data root configured for eboss_qso. "
                "Set ONEUNIVERSE_DATA_ROOT or pass data_path= explicitly."
            )

        fits_path = data_path / self.config.data_filename
        if not fits_path.exists():
            raise FileNotFoundError(
                f"Expected FITS file not found: {fits_path}"
            )

        logger.info("Reading %s ...", fits_path)

        # Read FITS with selective column I/O
        row_filter = {"IS_QSO_FINAL": 1} if qso_only else None
        df, array_cols = read_fits(
            fits_path,
            columns=_FITS_COLUMNS,
            row_filter=row_filter,
        )

        # Rename columns
        df = df.rename(columns=_COLUMN_MAP)

        # Expand multi-element array columns
        if "PSFMAG" in array_cols:
            psfmag = array_cols["PSFMAG"]
            for i, band in enumerate("ugriz"):
                df[f"psfmag_{band}"] = psfmag[:, i].astype(np.float32)

        if "EXTINCTION" in array_cols:
            ext = array_cols["EXTINCTION"]
            for i, band in enumerate("ugriz"):
                df[f"extinction_{band}"] = ext[:, i].astype(np.float32)

        if "Z_DLA" in array_cols:
            df["n_dla"] = _count_valid_dlas(array_cols["Z_DLA"])

        # Apply redshift filter
        if z_min is not None:
            df = df.loc[df["z"] >= z_min].reset_index(drop=True)
        if z_max is not None:
            df = df.loc[df["z"] <= z_max].reset_index(drop=True)

        # Add oneuniverse standard columns
        n = len(df)
        df["z_type"] = np.zeros(n, dtype=np.int8)  # spectroscopic
        df["galaxy_id"] = np.arange(n, dtype=np.int64)
        df["survey_id"] = "eboss_qso"

        # Spectroscopic group columns
        df["z_spec"] = df["z"].astype(np.float32)
        df["z_spec_err"] = np.float32(0.0)  # not provided per-object in DR16Q

        logger.info(
            "Loaded %d objects from eBOSS DR16Q (qso_only=%s)",
            len(df), qso_only,
        )
        return df
