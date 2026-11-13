import dataclasses
import json

import numpy as np
import pyarrow as pa
import pytest
from fixtures.synthetic import write_synthetic_dataset, write_synthetic_sightlines

from oneuniverse._core.errors import FormatError
from oneuniverse.data import DatasetView
from oneuniverse.measure.io import MEASURE_FORMAT_VERSION, load, save
from oneuniverse.measure.measurement_set import MeasurementSet, MeasurementSpec
from oneuniverse.measure.pipeline import build_sightlines
from oneuniverse.measure.presets import galaxy_clustering
from oneuniverse.measure.product import FieldMap, Product


@pytest.fixture(scope="module")
def view(tmp_path_factory):
    d = tmp_path_factory.mktemp("io")
    write_synthetic_dataset(d / "ds", 5_000, 0, ra_range=(100, 160), dec_range=(-10, 30))
    write_synthetic_sightlines(d / "sl", 200, 0)
    return DatasetView.from_path(d / "ds"), DatasetView.from_path(d / "sl")


def _ms_all_kinds(view):
    gv, sv = view
    gc = galaxy_clustering(gv, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), regions=6)
    reg = gc.regions
    gal = gc.products["gal"]
    npix = 12 * gal.window.nside ** 2
    win = dataclasses.replace(gal.window, systematics={"stars": np.linspace(0, 1, npix).astype(np.float32)})
    gal = dataclasses.replace(gal, window=win, provenance={**gal.provenance, "tup": (1, 2)})
    gal = gal.with_extra("vec", np.arange(7.0)).with_extra(
        "tab", pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
    nside = 8
    kappa = FieldMap(gal.catalog.slice(0, 0), values=np.random.default_rng(0).normal(size=12 * nside ** 2),
                     mask=np.ones(12 * nside ** 2), nside=nside, provenance={"src": "test"})
    lya = build_sightlines(sv, z=(2.0, 2.5), regions=reg)
    spec = MeasurementSpec("all", ("gal", "kappa", "lya"), ("pk", "cl"), pairs=(("gal", "kappa"),),
                           notes={"a": 1})
    return MeasurementSet({"gal": gal, "kappa": kappa, "lya": lya}, spec, reg)


def test_round_trip_drops_nothing(view, tmp_path):
    ms = _ms_all_kinds(view); save(ms, tmp_path / "ms"); back = load(tmp_path / "ms")
    assert set(back.products) == set(ms.products) and back.spec == ms.spec
    assert back.regions == ms.regions
    for name, p in ms.products.items():
        q = back.products[name]
        assert type(q) is type(p) and q.catalog.equals(p.catalog)
        assert (q.randoms is None) == (p.randoms is None)
        if p.randoms is not None:
            assert q.randoms.equals(p.randoms)
        assert set(q.extras) == set(p.extras)
        for k, v in p.extras.items():
            assert (v.equals(q.extras[k]) if isinstance(v, pa.Table) else np.array_equal(v, q.extras[k]))
        assert (q.window is None) == (p.window is None) and (q.nz is None) == (p.nz is None)
        if p.window:
            assert np.array_equal(q.window.mask, p.window.mask) and q.window.source == p.window.source
            assert (q.window.nside, q.window.nest) == (p.window.nside, p.window.nest)
            assert set(q.window.systematics) == set(p.window.systematics)
            for k, v in p.window.systematics.items():
                assert np.array_equal(q.window.systematics[k], v)
        if p.nz:
            assert np.array_equal(q.nz.counts, p.nz.counts) and np.array_equal(q.nz.edges, p.nz.edges)
            assert q.nz.method == p.nz.method and q.nz.bins == p.nz.bins
        assert q.provenance == p.provenance
    fm = back.products["kappa"]
    assert isinstance(fm, FieldMap) and fm.nside == 8 and fm.nest
    assert np.array_equal(fm.values, ms.products["kappa"].values)
    assert np.array_equal(fm.mask, ms.products["kappa"].mask)
    assert back.products["gal"].extras["tab"].equals(ms.products["gal"].extras["tab"])
    assert back.products["lya"].n_pixels == ms.products["lya"].n_pixels


def test_round_trip_tomographic_nz(tmp_path):
    from oneuniverse.measure.atoms import Nz
    cat = pa.table({"ra": [0.0], "dec": [0.0], "z": [0.5], "weight": [1.0], "region_id": np.zeros(1, np.int32),
                    "source_row": [0], "partition": np.zeros(1, np.int32), "offset": [0]})
    from oneuniverse.measure.atoms import Regions
    nz = Nz(np.linspace(0, 1, 4), np.ones((2, 3)), "photoz_stack", (0, 1))
    reg = Regions(4, True, 2, np.zeros(192, np.int32), "t")
    ms = MeasurementSet({"s": Product(cat, nz=nz)}, MeasurementSpec("n", ("s",), ("xi",)), reg)
    save(ms, tmp_path / "m"); q = load(tmp_path / "m").products["s"]
    assert q.nz.bins == (0, 1) and np.array_equal(q.nz.counts, nz.counts)


def test_save_refuses_unknown_extra_type(view, tmp_path):
    ms = _ms_all_kinds(view)
    with pytest.raises(TypeError):
        dataclasses.replace(ms.products["gal"], extras={"x": {"a": 1}})


def test_save_is_atomic_and_versioned(view, tmp_path):
    ms = _ms_all_kinds(view); save(ms, tmp_path / "ms")
    d = json.loads((tmp_path / "ms/measurement_set.json").read_text())
    assert d["format_version"] == MEASURE_FORMAT_VERSION == "1.0.0"
    with pytest.raises(FormatError):
        save(ms, tmp_path / "ms")
    save(ms, tmp_path / "ms", overwrite=True)
    assert not (tmp_path / "ms.tmp").exists()
    d["format_version"] = "2.0.0"; (tmp_path / "ms/measurement_set.json").write_text(json.dumps(d))
    with pytest.raises(FormatError):
        load(tmp_path / "ms")


def test_load_rejects_missing_file_and_corrupt_json(view, tmp_path):
    ms = _ms_all_kinds(view); save(ms, tmp_path / "ms")
    (tmp_path / "ms/products/gal/nz.npz").unlink()
    with pytest.raises(FormatError, match="nz"):
        load(tmp_path / "ms")
    (tmp_path / "ms/measurement_set.json").write_text("{not json")
    with pytest.raises(FormatError) as e:
        load(tmp_path / "ms")
    assert e.value.__cause__ is not None
    with pytest.raises(FormatError):
        load(tmp_path / "nope")
