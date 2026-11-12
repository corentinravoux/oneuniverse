"""eBOSS DR16Q superset quasar catalogue."""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from oneuniverse.data.loaders.base import FitsCatalogLoader
from oneuniverse.data.loaders.registry import register
from oneuniverse.data.manifest import SurveyMeta


@register
class EbossQso(FitsCatalogLoader):
    meta = SurveyMeta(
        name="eboss_qso", kind="spectroscopic",
        description="eBOSS DR16Q superset v3 quasar catalogue",
        reference="Lyke et al. 2020, ApJS 250 8",
        url="https://www.sdss.org/dr16/algorithms/qso-catalog/")
    filename = "DR16Q_Superset_v3.fits"
    columns = {
        "RA": "ra", "DEC": "dec", "Z": "z", "IS_QSO_FINAL": "is_qso", "SOURCE_Z": "source_z",
        "Z_PIPE": "z_pipe", "Z_PCA": "z_pca", "Z_VI": "z_vi", "Z_QN": "z_qn", "Z_CONF": "z_conf",
        "ZWARNING": "zwarning", "CLASS_PERSON": "class_person", "BAL_PROB": "bal_prob",
        "BI_CIV": "bi_civ", "Z_MGII": "z_mgii", "Z_CIII": "z_ciii", "Z_CIV": "z_civ",
        "Z_LYA": "z_lya", "Z_HALPHA": "z_halpha", "Z_HBETA": "z_hbeta", "PLATE": "plate",
        "MJD": "mjd", "FIBERID": "fiberid", "THING_ID": "thing_id", "SDSS_NAME": "sdss_name",
        "BOSS_TARGET1": "boss_target1", "EBOSS_TARGET0": "eboss_target0",
        "EBOSS_TARGET1": "eboss_target1", "EBOSS_TARGET2": "eboss_target2",
        "SN_MEDIAN_ALL": "sn_median", "LAMBDA_EFF": "lambda_eff", "NSPEC": "nspec",
        "PSFMAG": "psfmag", "EXTINCTION": "extinction", "Z_DLA": "z_dla",
    }
    z_err_column = None
    extensions = {
        "coordinate": {"version": 1, "frame": "icrs"},
        "spectrum": {"version": 1, "wavelength_convention": "vacuum",
                     "log_binned": True, "rest_frame": False},
    }

    def row_filter(self, native: pa.Table, *, qso_only: bool = True):
        if not qso_only:
            return None
        return pc.equal(native["IS_QSO_FINAL"], 1)

    def derive(self, table: pa.Table) -> pa.Table:
        z_dla = np.asarray(table["z_dla"].combine_chunks().values.to_numpy(zero_copy_only=False))
        n_dla = (z_dla.reshape(table.num_rows, -1) > 0).sum(axis=1).astype(np.int8)
        i = table.schema.get_field_index("z_dla")
        return table.remove_column(i).append_column("n_dla", pa.array(n_dla, pa.int8()))
