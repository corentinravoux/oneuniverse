import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from oneuniverse.data._converter_lightcurve import write_ouf_lightcurve_dataset
from oneuniverse.data.manifest import read_manifest
from oneuniverse.data.format_spec import DataGeometry, ONEUNIVERSE_SUBDIR


def _make(n_obj=5, n_epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    objects = pd.DataFrame({
        "object_id": np.arange(n_obj, dtype=np.int64),
        "ra":    rng.uniform(0.0, 360.0, n_obj),
        "dec":   rng.uniform(-60.0, 60.0, n_obj),
        "z":     rng.uniform(0.01, 0.5, n_obj),
        "z_type": ["spec"] * n_obj,
        "z_err": rng.uniform(1e-4, 1e-3, n_obj),
    })
    rows = []
    for oid in objects["object_id"]:
        mjd = np.sort(rng.uniform(58000.0, 60000.0, n_epochs))
        for t in mjd:
            rows.append({
                "object_id": int(oid), "mjd": float(t),
                "filter": rng.choice(["g", "r", "i"]),
                "flux": float(rng.normal(100.0, 5.0)),
                "flux_err": 1.0, "flag": 0,
            })
    return objects, pd.DataFrame(rows)


def test_roundtrip_structure(tmp_path):
    objects, epochs = _make(3, 10)
    survey_dir = tmp_path / "lc"
    write_ouf_lightcurve_dataset(
        objects=objects, epochs=epochs,
        survey_path=survey_dir,
        survey_name="lc", survey_type="transient",
        loader_name="syn", loader_version="0",
    )
    ou = survey_dir / ONEUNIVERSE_SUBDIR
    assert (ou / "manifest.json").exists()
    assert (ou / "objects.parquet").exists()
    assert any(ou.glob("part_*.parquet"))
    m = read_manifest(ou / "manifest.json")
    assert m.geometry is DataGeometry.LIGHTCURVE
    assert m.temporal.time_column == "mjd"
    obj = pq.read_table(ou / "objects.parquet").to_pandas()
    assert set(obj.columns) >= {"object_id", "ra", "dec", "z",
                                "n_epochs", "mjd_min", "mjd_max"}
    expected = epochs.groupby("object_id").size().reset_index(name="n_epochs")
    merged = obj[["object_id", "n_epochs"]].merge(
        expected, on="object_id", suffixes=("", "_truth"))
    assert (merged["n_epochs"] == merged["n_epochs_truth"]).all()


def test_rejects_orphan_epochs(tmp_path):
    objects, epochs = _make(2, 3)
    orphan = epochs.iloc[:1].copy()
    orphan["object_id"] = 999
    epochs = pd.concat([epochs, orphan], ignore_index=True)
    with pytest.raises(ValueError, match="orphan"):
        write_ouf_lightcurve_dataset(
            objects=objects, epochs=epochs,
            survey_path=tmp_path / "bad",
            survey_name="bad", survey_type="transient",
            loader_name="syn", loader_version="0",
        )


def test_partition_stats_cover_mjd(tmp_path):
    objects, epochs = _make(4, 30, seed=3)
    survey_dir = tmp_path / "lc2"
    write_ouf_lightcurve_dataset(
        objects=objects, epochs=epochs,
        survey_path=survey_dir,
        survey_name="lc2", survey_type="transient",
        loader_name="syn", loader_version="0",
    )
    m = read_manifest(survey_dir / ONEUNIVERSE_SUBDIR / "manifest.json")
    overall_min = min(p.stats.t_min for p in m.partitions)
    overall_max = max(p.stats.t_max for p in m.partitions)
    assert overall_min == float(epochs["mjd"].min())
    assert overall_max == float(epochs["mjd"].max())


def test_writer_accepts_validity_kwarg(tmp_path):
    from oneuniverse.data.validity import DatasetValidity
    objects, epochs = _make(2, 3)
    v = DatasetValidity(valid_from_utc="2026-02-01T00:00:00+00:00",
                        version="ztf_2026_02")
    survey_dir = tmp_path / "vlc"
    write_ouf_lightcurve_dataset(
        objects=objects, epochs=epochs,
        survey_path=survey_dir,
        survey_name="vlc", survey_type="transient",
        loader_name="syn", loader_version="0",
        validity=v,
    )
    m = read_manifest(survey_dir / ONEUNIVERSE_SUBDIR / "manifest.json")
    assert m.validity == v
