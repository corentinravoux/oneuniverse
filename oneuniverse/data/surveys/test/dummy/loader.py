"""
Dummy survey loader — reference implementation of BaseSurveyLoader.

Generates a reproducible in-memory catalog with no external files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data._registry import register

_C_KMS = 299_792.458

# Generation defaults
_N_GAL = 5_000
_SEED = 42
_Z_MIN = 0.005
_Z_MAX = 0.15
_SIGMA_V = 300.0     # km/s intrinsic velocity dispersion
_SIGMA_MU_TF = 0.45  # mag  Tully-Fisher scatter


@register
class DummyLoader(BaseSurveyLoader):

    config = SurveyConfig(
        name="dummy",
        survey_type="test",
        description=(
            "Synthetic full-sky catalog (5 000 gal, z < 0.15) "
            "for testing and demos — seed 42, reproducible"
        ),
        column_groups=("core", "spectroscopic", "peculiar_velocity"),
        characteristic_fields={
            "v_pec":     ("f4", "km/s", "Mock peculiar velocity (Gaussian, sigma=300 km/s)"),
            "mu":        ("f4", "mag",  "Mock distance modulus (low-z approx)"),
            "mu_method": ("U8", "",     "Always 'TF' for the dummy catalog"),
        },
        data_format="in-memory",
        reference="—",
        sky_fraction=1.0,
        z_range=(0.005, 0.15),
        n_objects_approx=5_000,
    )

    def _load_raw(
        self,
        data_path: Optional[Path] = None,
        n_galaxies: int = _N_GAL,
        seed: int = _SEED,
        **kwargs,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = n_galaxies

        # Positions: uniform on the sphere
        ra = rng.uniform(0.0, 360.0, size=n)
        dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))

        # Redshifts: proportional to z^2 (comoving volume)
        u = rng.uniform(0.0, 1.0, size=n)
        z = (_Z_MIN**3 + u * (_Z_MAX**3 - _Z_MIN**3)) ** (1.0 / 3.0)
        z = z.astype(np.float32)

        # Spectroscopic redshift with ~100 km/s error
        z_spec_err = np.full(n, 100.0 / _C_KMS, dtype=np.float32)
        z_spec = (z + rng.normal(0, z_spec_err)).astype(np.float32)

        # Peculiar velocities
        v_pec = rng.normal(0.0, _SIGMA_V, size=n).astype(np.float32)
        cz = z.astype(np.float64) * _C_KMS
        v_pec_err = (np.log(10.0) / 5.0 * _SIGMA_MU_TF * cz).astype(np.float32)

        # CMB-frame (trivial for dummy)
        z_cmb = z.copy()
        cz_cmb = (z_cmb.astype(np.float64) * _C_KMS).astype(np.float32)

        # Distance modulus (rough low-z approx)
        d_l_mpc = cz / 67.36
        mu = (5.0 * np.log10(np.maximum(d_l_mpc, 1e-10)) + 25.0).astype(np.float32)
        mu_err = np.full(n, _SIGMA_MU_TF, dtype=np.float32)

        # HEALPix NSIDE=32 NESTED index (OUF 2.0 spatial partition key)
        healpix32 = hp.ang2pix(
            32,
            np.radians(90.0 - dec),
            np.radians(ra),
            nest=True,
        ).astype(np.int32)

        return pd.DataFrame({
            # Core
            "ra": ra,
            "dec": dec,
            "z": z,
            "z_type": np.array(["spec"] * n),
            "z_err": z_spec_err,
            "galaxy_id": np.arange(n, dtype=np.int64),
            "survey_id": np.array(["dummy"] * n),
            "_original_row_index": np.arange(n, dtype=np.int64),
            "_healpix32": healpix32,
            # Spectroscopic
            "z_spec": z_spec,
            "z_spec_err": z_spec_err,
            "z_helio": z_spec,
            "z_cmb": z_cmb,
            "cz_cmb": cz_cmb,
            # Peculiar velocity
            "v_pec": v_pec,
            "v_pec_err": v_pec_err,
            "mu": mu,
            "mu_err": mu_err,
            "mu_method": np.array(["TF"] * n),
        })
