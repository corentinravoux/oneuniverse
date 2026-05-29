"""Phase 17 T6 — DatasetView prunes + pushes down extra_filters."""
import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec


def _make(tmp_path):
    rng = np.random.default_rng(0)
    n = 5000
    ra = rng.uniform(0, 360, n).astype("f8")
    dec = rng.uniform(-60, 60, n).astype("f8")
    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "z": rng.uniform(0.0, 1.0, n).astype("f4"),
        "z_type": np.array(["spec"] * n, dtype=object),
        "z_err": np.full(n, 0.01, dtype="f4"),
        "galaxy_id": np.arange(n, dtype="i8"),
        "survey_id": np.array(["fix"] * n, dtype=object),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
        "snr": rng.uniform(1.0, 200.0, n).astype("f4"),
    })
    out = tmp_path / "x" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="x", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="x", version="0"),
        extra_stats_columns=["snr"],
    )
    return DatasetView.from_path(out.parent), df


def test_extra_filters_push_down_to_rows(tmp_path):
    view, df = _make(tmp_path)
    out = view.read(extra_filters={"snr": (50.0, None)})
    assert (out["snr"] >= 50.0).all()


def test_extra_filters_upper_bound(tmp_path):
    view, df = _make(tmp_path)
    out = view.read(extra_filters={"snr": (None, 20.0)})
    assert (out["snr"] <= 20.0).all()


def test_extra_filters_prune_partitions(tmp_path):
    """_select_partitions excludes partitions whose stats cannot overlap.

    The deterministic version of this test patches the manifest with
    hand-built ``extra_ranges`` so the prune outcome is stable; the
    higher-level integration is already covered by the row-level tests.
    """
    from dataclasses import replace

    view, _ = _make(tmp_path)
    parts = view.manifest.partitions
    # Tag the first half of partitions with snr in [0, 50] and the
    # second half with snr in [150, 200].
    half = len(parts) // 2
    new_parts = []
    for i, p in enumerate(parts):
        er = (0.0, 50.0) if i < half else (150.0, 200.0)
        new_parts.append(
            replace(p, stats=replace(p.stats, extra_ranges={"snr": er})),
        )
    new_manifest = replace(view.manifest, partitions=new_parts)
    object.__setattr__(view, "manifest", new_manifest)
    full = view._select_partitions()
    pruned = view._select_partitions(extra_filters={"snr": (180.0, None)})
    assert 0 < len(pruned) < len(full)
