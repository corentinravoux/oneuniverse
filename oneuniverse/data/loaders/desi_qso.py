"""DESI DR1 QSO full catalogue."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from oneuniverse.data.loaders.base import FitsCatalogLoader
from oneuniverse.data.loaders.registry import register
from oneuniverse.data.manifest import SurveyMeta


def _mag(flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=np.float64)
    out = np.full(flux.shape, np.nan)
    ok = np.isfinite(flux) & (flux > 0)
    out[ok] = 22.5 - 2.5 * np.log10(flux[ok])
    return out.astype(np.float32)


@register
class DesiQso(FitsCatalogLoader):
    meta = SurveyMeta(
        name="desi_qso", kind="spectroscopic",
        description="DESI DR1 QSO full catalogue",
        reference="DESI Collaboration 2025, DR1",
        url="https://data.desi.lbl.gov/doc/releases/dr1/")
    filename = "QSO_full.dat.fits"
    columns = {
        "RA": "ra", "DEC": "dec", "Z_RR": "z", "ZWARN": "zwarning", "SPECTYPE": "spectype",
        "SUBTYPE": "subtype", "DELTACHI2": "deltachi2", "TARGETID": "targetid",
        "Z_QN": "z_qn", "Z_QN_CONF": "z_qn_conf", "IS_QSO_QN": "is_qso_qn",
        "NTILE": "ntile", "FIBER": "fiber",
        "FLUX_G": "flux_g", "FLUX_R": "flux_r", "FLUX_Z": "flux_z",
        "FLUX_IVAR_G": "flux_ivar_g", "FLUX_IVAR_R": "flux_ivar_r", "FLUX_IVAR_Z": "flux_ivar_z",
        "FLUX_W1": "flux_w1", "FLUX_W2": "flux_w2",
        "FLUX_IVAR_W1": "flux_ivar_w1", "FLUX_IVAR_W2": "flux_ivar_w2",
        "MW_TRANSMISSION_G": "mw_transmission_g", "MW_TRANSMISSION_R": "mw_transmission_r",
        "MW_TRANSMISSION_Z": "mw_transmission_z", "MW_TRANSMISSION_W1": "mw_transmission_w1",
        "MW_TRANSMISSION_W2": "mw_transmission_w2", "EBV": "ebv",
        "TSNR2_QSO": "tsnr2_qso", "TSNR2_LYA": "tsnr2_lya",
        "COADD_NUMEXP": "coadd_numexp", "COADD_EXPTIME": "coadd_exptime",
        "COADD_NUMNIGHT": "coadd_numnight", "WEIGHT_ZFAIL": "w_zfail", "COMP_TILE": "comp_tile",
        "FRACZ_TILELOCID": "fracz_tilelocid", "FRAC_TLOBS_TILES": "frac_tlobs_tiles",
        "PROB_OBS": "prob_obs", "DESI_TARGET": "desi_target",
        "MORPHTYPE": "morphtype", "PHOTSYS": "photsys",
    }
    z_err_column = "ZERR"
    extensions = {
        "coordinate": {"version": 1, "frame": "icrs"},
        "spectrum": {"version": 1, "wavelength_convention": "vacuum",
                     "log_binned": True, "rest_frame": False},
    }

    def row_filter(self, native: pa.Table, *, qso_only: bool = True, good_zwarn: bool = True):
        mask = None
        if qso_only:
            mask = pc.equal(native["SPECTYPE"], "QSO")
        if good_zwarn:
            ok = pc.equal(native["ZWARN"], 0)
            mask = ok if mask is None else pc.and_(mask, ok)
        return mask

    def derive(self, table: pa.Table) -> pa.Table:
        for b in "grz":
            table = table.append_column(
                f"mag_{b}", pa.array(_mag(table[f"flux_{b}"].to_numpy()), pa.float32()))
        return table
