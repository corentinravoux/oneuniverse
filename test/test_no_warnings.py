"""Suite-wide warning-cleanliness pin.

Runs a representative slice of the package under ``-W error``-equivalent
catch_warnings, failing on any FutureWarning / DeprecationWarning that
originates inside ``oneuniverse``. Catches the common offenders
(pandas groupby, numpy aliases, healpy deprecations).
"""
from __future__ import annotations

import warnings


def test_no_pkg_futurewarning_on_typical_workflow(tmp_path):
    import numpy as np
    import pandas as pd
    import healpy as hp

    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.dataset_view import DatasetView
    from oneuniverse.data.format_spec import (
        DataGeometry, HEALPIX_PARTITION_NSIDE,
    )
    from oneuniverse.data.manifest import LoaderSpec
    from oneuniverse.data.selection import Cone

    n = 200
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "z": np.full(n, 0.5, dtype=np.float32),
        "z_type": np.array(["spec"] * n, dtype="<U4"),
        "z_err": np.full(n, 1e-3, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.array(["fake"] * n, dtype="<U16"),
        "_original_row_index": np.arange(n, dtype=np.int64),
    })
    theta = np.radians(90.0 - df["dec"].to_numpy(dtype=np.float64))
    phi = np.radians(df["ra"].to_numpy(dtype=np.float64))
    df["_healpix32"] = hp.ang2pix(
        HEALPIX_PARTITION_NSIDE, theta, phi, nest=True,
    ).astype(np.int32)

    out_dir = tmp_path / "x" / "oneuniverse"
    out_dir.mkdir(parents=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_ouf_dataset(
            df=df, out_dir=out_dir,
            survey_name="x", survey_type="spectroscopic",
            geometry=DataGeometry.POINT,
            loader=LoaderSpec(name="x", version="0"),
            partition_nside=HEALPIX_PARTITION_NSIDE,
        )
        view = DatasetView.from_path(out_dir.parent)
        _ = view.read()
        _ = view.scan(cone=Cone(ra=180.0, dec=0.0, radius=5.0))

    bad = [
        w for w in caught
        if issubclass(w.category, (FutureWarning, DeprecationWarning))
        and "oneuniverse" in str(w.filename)
    ]
    assert bad == [], [
        f"{w.category.__name__} at {w.filename}:{w.lineno}: {w.message}"
        for w in bad
    ]
