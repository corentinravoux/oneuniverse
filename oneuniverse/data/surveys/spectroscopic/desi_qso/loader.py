"""
DESI DR1 QSO catalog loader.

Data: QSO_full.dat.fits (~2.33M rows, 117 columns)
Reference: DESI Collaboration 2024, AJ 168 58 (DR1)

This is the full DESI DR1 quasar catalog produced by the QSO target
selection pipeline.  It contains all objects that were targeted as QSOs,
including those that turned out to be stars or galaxies.  The key
classification is SPECTYPE == 'QSO'.

Column mapping (FITS -> oneuniverse):
    RA, DEC              -> ra, dec
    Z_RR                 -> z  (Redrock best redshift)
    ZERR                 -> z_spec_err
    ZWARN                -> zwarning
    SPECTYPE             -> spectype  (QSO, STAR, GALAXY)
    SUBTYPE              -> subtype   (HIZ, LOZ, ...)
    DELTACHI2            -> deltachi2 (chi2 difference to 2nd best)
    TARGETID             -> targetid  (unique DESI target identifier)
    Z_QN                 -> z_qn      (QuasarNET redshift)
    Z_QN_CONF            -> z_qn_conf (QuasarNET confidence)
    IS_QSO_QN            -> is_qso_qn (QuasarNET classification)
    NTILE                -> ntile
    FLUX_G/R/Z           -> flux_g/r/z (nanomaggy)
    FLUX_IVAR_G/R/Z      -> flux_ivar_g/r/z
    FLUX_W1/W2           -> flux_w1/w2
    FLUX_IVAR_W1/W2      -> flux_ivar_w1/w2
    MW_TRANSMISSION_G/R/Z -> mw_transmission_g/r/z
    EBV                  -> ebv
    TSNR2_QSO            -> tsnr2_qso (template S/N)
    TSNR2_LYA            -> tsnr2_lya
    COADD_NUMEXP         -> coadd_numexp
    COADD_EXPTIME        -> coadd_exptime
    WEIGHT_ZFAIL         -> w_zfail
    COMP_TILE            -> comp_tile
    FRACZ_TILELOCID      -> fracz_tilelocid
    FRAC_TLOBS_TILES     -> frac_tlobs_tiles
    PROB_OBS             -> prob_obs
    DESI_TARGET          -> desi_target (targeting bitmask)
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

# Columns to read from the FITS file (selective I/O)
_FITS_COLUMNS = [
    "RA", "DEC", "Z_RR", "ZERR", "ZWARN",
    "SPECTYPE", "SUBTYPE", "DELTACHI2",
    "TARGETID", "Z_QN", "Z_QN_CONF", "IS_QSO_QN",
    "NTILE", "FIBER",
    "FLUX_G", "FLUX_R", "FLUX_Z",
    "FLUX_IVAR_G", "FLUX_IVAR_R", "FLUX_IVAR_Z",
    "FLUX_W1", "FLUX_W2",
    "FLUX_IVAR_W1", "FLUX_IVAR_W2",
    "MW_TRANSMISSION_G", "MW_TRANSMISSION_R", "MW_TRANSMISSION_Z",
    "MW_TRANSMISSION_W1", "MW_TRANSMISSION_W2",
    "EBV",
    "TSNR2_QSO", "TSNR2_LYA",
    "COADD_NUMEXP", "COADD_EXPTIME", "COADD_NUMNIGHT",
    "WEIGHT_ZFAIL", "COMP_TILE",
    "FRACZ_TILELOCID", "FRAC_TLOBS_TILES", "PROB_OBS",
    "DESI_TARGET", "MORPHTYPE", "PHOTSYS",
]

_COLUMN_MAP = {
    "RA": "ra",
    "DEC": "dec",
    "Z_RR": "z",
    "ZERR": "z_spec_err",
    "ZWARN": "zwarning",
    "SPECTYPE": "spectype",
    "SUBTYPE": "subtype",
    "DELTACHI2": "deltachi2",
    "TARGETID": "targetid",
    "Z_QN": "z_qn",
    "Z_QN_CONF": "z_qn_conf",
    "IS_QSO_QN": "is_qso_qn",
    "NTILE": "ntile",
    "FIBER": "fiber",
    "FLUX_G": "flux_g",
    "FLUX_R": "flux_r",
    "FLUX_Z": "flux_z",
    "FLUX_IVAR_G": "flux_ivar_g",
    "FLUX_IVAR_R": "flux_ivar_r",
    "FLUX_IVAR_Z": "flux_ivar_z",
    "FLUX_W1": "flux_w1",
    "FLUX_W2": "flux_w2",
    "FLUX_IVAR_W1": "flux_ivar_w1",
    "FLUX_IVAR_W2": "flux_ivar_w2",
    "MW_TRANSMISSION_G": "mw_transmission_g",
    "MW_TRANSMISSION_R": "mw_transmission_r",
    "MW_TRANSMISSION_Z": "mw_transmission_z",
    "MW_TRANSMISSION_W1": "mw_transmission_w1",
    "MW_TRANSMISSION_W2": "mw_transmission_w2",
    "EBV": "ebv",
    "TSNR2_QSO": "tsnr2_qso",
    "TSNR2_LYA": "tsnr2_lya",
    "COADD_NUMEXP": "coadd_numexp",
    "COADD_EXPTIME": "coadd_exptime",
    "COADD_NUMNIGHT": "coadd_numnight",
    "WEIGHT_ZFAIL": "w_zfail",
    "COMP_TILE": "comp_tile",
    "FRACZ_TILELOCID": "fracz_tilelocid",
    "FRAC_TLOBS_TILES": "frac_tlobs_tiles",
    "PROB_OBS": "prob_obs",
    "DESI_TARGET": "desi_target",
    "MORPHTYPE": "morphtype",
    "PHOTSYS": "photsys",
}


@register
class DESIQSOLoader(BaseSurveyLoader):

    config = SurveyConfig(
        name="desi_qso",
        survey_type="spectroscopic",
        description=(
            "DESI DR1 QSO full catalog — 2.33M targets, "
            "z ~ 0.1–6.0, quasar-selected objects (DESI Collaboration 2024)"
        ),
        column_groups=("core", "spectroscopic"),
        characteristic_fields={
            "spectype":       ("U6",  "",        "Redrock spectral type (QSO/STAR/GALAXY)"),
            "deltachi2":      ("f8",  "",        "Delta-chi2 to 2nd best template"),
            "targetid":       ("i8",  "",        "DESI unique target identifier"),
            "z_qn":           ("f8",  "",        "QuasarNET redshift"),
            "z_qn_conf":      ("f8",  "",        "QuasarNET confidence (0-1)"),
            "is_qso_qn":      ("i2",  "",        "QuasarNET classification (1=QSO)"),
            "tsnr2_qso":      ("f4",  "",        "Template S/N^2 QSO"),
            "tsnr2_lya":      ("f4",  "",        "Template S/N^2 Lya"),
            "w_zfail":        ("f8",  "",        "Redshift failure weight"),
            "comp_tile":      ("f8",  "",        "Tile completeness"),
            "prob_obs":       ("f8",  "",        "Observation probability from BITWEIGHTS"),
        },
        data_subpath="spectroscopic/desi/dr1/qso",
        data_filename="QSO_full.dat.fits",
        data_format="fits",
        reference="DESI Collaboration 2024, AJ 168 58",
        url="https://data.desi.lbl.gov/doc/releases/dr1/",
        sky_fraction=0.34,
        z_range=(0.0, 6.0),
        n_objects_approx=2_332_000,
    )

    def _load_raw(
        self,
        data_path: Optional[Path] = None,
        qso_only: bool = True,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        good_zwarn: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Load the DESI DR1 QSO catalog.

        Parameters
        ----------
        data_path : Path or None
            Directory containing QSO_full.dat.fits.
        qso_only : bool
            If True (default), keep only SPECTYPE == 'QSO'.
        z_min, z_max : float or None
            Redshift pre-filter.
        good_zwarn : bool
            If True (default), keep only ZWARN == 0.
        """
        if data_path is None:
            raise FileNotFoundError(
                "No data root configured for desi_qso. "
                "Set ONEUNIVERSE_DATA_ROOT or pass data_path= explicitly."
            )

        fits_path = data_path / self.config.data_filename
        if not fits_path.exists():
            raise FileNotFoundError(f"Expected FITS file not found: {fits_path}")

        logger.info("Reading %s ...", fits_path)

        # For DESI the row_filter is done post-read because SPECTYPE is
        # a string column and fitsio row filtering only works on numeric.
        df, _array_cols = read_fits(fits_path, columns=_FITS_COLUMNS)

        # Rename columns
        df = df.rename(columns=_COLUMN_MAP)

        # Quality cuts
        if qso_only:
            df = df[df["spectype"].str.strip() == "QSO"].reset_index(drop=True)
            logger.info("qso_only cut: %d rows", len(df))
        if good_zwarn:
            df = df[df["zwarning"] == 0].reset_index(drop=True)
            logger.info("good_zwarn cut: %d rows", len(df))

        # Redshift filter
        if z_min is not None:
            df = df[df["z"] >= z_min].reset_index(drop=True)
        if z_max is not None:
            df = df[df["z"] <= z_max].reset_index(drop=True)

        # Compute nanomaggy -> magnitude columns for convenience
        for band in ("g", "r", "z"):
            flux = df[f"flux_{band}"].to_numpy(dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                mag = 22.5 - 2.5 * np.log10(np.maximum(flux, 1e-30))
            mag[flux <= 0] = np.nan
            df[f"mag_{band}"] = mag.astype(np.float32)

        # Add oneuniverse standard columns
        n = len(df)
        df["z_type"] = np.full(n, "spec", dtype="<U4")
        df["galaxy_id"] = np.arange(n, dtype=np.int64)
        df["survey_id"] = "desi_qso"
        df["z_spec"] = df["z"].astype(np.float32)

        logger.info("Loaded %d objects from DESI DR1 QSO", len(df))
        return df
