import json

import numpy as np
import pytest
from fixtures.synthetic import write_synthetic_dataset

from oneuniverse._core.errors import ValidationError
from oneuniverse.combine.weights import ShearWeight
from oneuniverse.data import DatasetView
from oneuniverse.data.pdf import PdfSpec
from oneuniverse.measure.measurement_set import MeasurementSet, MeasurementSpec
from oneuniverse.measure.presets import cosmic_shear, galaxy_clustering, three_by_two


@pytest.fixture(scope="module")
def view(tmp_path_factory):
    p = tmp_path_factory.mktemp("m") / "ds"
    write_synthetic_dataset(p, 20_000, 0, ra_range=(100, 160), dec_range=(-10, 30))
    return DatasetView.from_path(p)


def _gc(view, **kw):
    return galaxy_clustering(view, z=(0.3, 1.0), nz_edges=np.linspace(0.3, 1.0, 8), **kw)


def test_galaxy_clustering_preset(view):
    ms = _gc(view)
    assert ms.spec.tracers == ("gal",) and ms.spec.statistics == ("pk_multipoles",)
    assert ms.products["gal"].randoms is not None


def test_cosmic_shear_preset(tmp_path):
    spec = PdfSpec("interp", grid=tuple(np.linspace(0, 3, 31)))
    write_synthetic_dataset(tmp_path / "s", 20_000, 0, with_pdf=spec, with_shear=True)
    v = DatasetView.from_path(tmp_path / "s")
    ms = cosmic_shear(v, z=(0.2, 2.0), shear_weight=ShearWeight("metacal"), tomo_edges=[0.2, 0.8, 2.0],
                      nz_edges=np.linspace(0, 3, 31), n_draws=5)
    src = ms.products["src"]
    assert {"e1", "e2", "tomo_bin", "z_draws"} <= set(src.extras)
    assert src.nz.counts.shape[0] == 2 and ms.spec.statistics == ("xi_plus", "xi_minus")
    assert src.extras["e1"].shape == (src.n,) and src.randoms is None


def test_three_by_two_requires_shared_regions(view):
    lens = _gc(view, regions=10)
    src = _gc(view, regions=10, name="src")
    with pytest.raises(ValidationError, match="regions"):
        three_by_two(lens, _gc(view, regions=12))
    src2 = MeasurementSet({"src": src.products["gal"]},
                          MeasurementSpec("src", ("src",), ("xi_plus",)), lens.regions)
    ms = three_by_two(lens, src2)
    assert ms.spec.pairs == (("gal", "gal"), ("gal", "src"), ("src", "src"))
    assert set(ms.products) == {"gal", "src"}
    assert ms.spec.statistics == ("pk_multipoles", "xi_plus")


def test_three_by_two_rejects_same_tracer_name(view):
    a = _gc(view, regions=10)
    with pytest.raises(ValidationError, match="differ"):
        three_by_two(a, a)


def test_preset_regions_match_catalog_region_id(view):
    ms = _gc(view)
    cat = ms.products["gal"].catalog
    got = ms.regions.assign(cat["ra"].to_numpy(), cat["dec"].to_numpy())
    assert np.array_equal(got, cat["region_id"].to_numpy())


def test_cosmic_shear_seed_is_explicit_keyword():
    import inspect
    p = inspect.signature(cosmic_shear).parameters["seed"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default == 0


def test_two_products_with_different_regions_cannot_share_a_set(view):
    a = _gc(view, regions=10, seed=1)
    b = _gc(view, regions=10, seed=2)
    assert a.regions != b.regions
    with pytest.raises(ValidationError, match="region_id inconsistent with regions"):
        MeasurementSet({"gal": b.products["gal"]}, MeasurementSpec("x", ("gal",), ("pk",)), a.regions)


def test_notes_normalised_and_checked():
    s = MeasurementSpec("n", ("a",), ("x",), notes={"t": (1, 2)})
    assert s.notes == {"t": [1, 2]}
    with pytest.raises(ValidationError):
        MeasurementSpec("n", ("a",), ("x",), notes={"o": object()})


_DETERMINISM = """
import sys, json
sys.path.insert(0, sys.argv[1])
import json

import numpy as np
from fixtures.synthetic import write_synthetic_dataset
from oneuniverse.combine.weights import ShearWeight
from oneuniverse.data import DatasetView
from oneuniverse.data.pdf import PdfSpec
from oneuniverse.measure.presets import cosmic_shear
spec = PdfSpec("interp", grid=tuple(np.linspace(0, 3, 31)))
write_synthetic_dataset(sys.argv[2], 3000, 0, with_pdf=spec, with_shear=True)
v = DatasetView.from_path(sys.argv[2])
ms = cosmic_shear(v, z=(0.2, 2.0), shear_weight=ShearWeight("metacal"), tomo_edges=[0.2, 0.8, 2.0],
                  nz_edges=np.linspace(0, 3, 31), n_draws=2)
p = ms.products["src"]
prov = {k: v for k, v in p.provenance.items() if k != "version"}
print(json.dumps([p.catalog.column_names, prov, p.catalog["weight"].to_pylist()[:50]], sort_keys=True))
"""


def test_cosmic_shear_deterministic_across_hash_seeds(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    tdir = str(Path(__file__).resolve().parents[1])
    outs = []
    for i, hs in enumerate(("1", "2")):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        r = subprocess.run([sys.executable, "-c", _DETERMINISM, tdir, str(tmp_path / f"d{i}")],
                           env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip().splitlines()[-1])
    assert outs[0] == outs[1]
    prov = json.loads(outs[0])[1]
    assert prov["nz_method"] == "photoz_stack" and prov["tomo_edges"] == [0.2, 0.8, 2.0] and prov["n_draws"] == 2


def test_cosmic_shear_provenance_present(tmp_path):
    spec = PdfSpec("interp", grid=tuple(np.linspace(0, 3, 31)))
    write_synthetic_dataset(tmp_path / "s", 3000, 0, with_pdf=spec, with_shear=True)
    v = DatasetView.from_path(tmp_path / "s")
    ms = cosmic_shear(v, z=(0.2, 2.0), shear_weight=ShearWeight("metacal"), tomo_edges=[0.2, 2.0],
                      nz_edges=np.linspace(0, 3, 31), n_draws=2)
    assert ms.products["src"].provenance["n_draws"] == 2
