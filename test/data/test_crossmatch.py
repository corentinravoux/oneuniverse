import time

import numpy as np
import pyarrow as pa
import pytest

from oneuniverse._core.errors import SchemaError, ValidationError
from oneuniverse._core.healpix import radec_to_pix
from oneuniverse.data.crossmatch import CrossMatchRules, components, match_pair


def _cat(n, seed, *, survey, jitter_arcsec=0.0, z_jitter=0.0, base=None):
    rng = np.random.default_rng(seed + (1000 if base is not None else 0))
    if base is None:
        ra = rng.uniform(0, 360, n)
        dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
        z = rng.uniform(0.1, 2.0, n)
        src = np.arange(n)
    else:
        perm = rng.permutation(n)
        ra0, dec0, z0 = (base.column(c).to_numpy()[perm] for c in ("ra", "dec", "z"))
        j = jitter_arcsec / 3600.0
        dec = np.clip(dec0 + rng.uniform(-j, j, n), -90, 90)
        ra = (ra0 + rng.uniform(-j, j, n) / np.maximum(np.cos(np.radians(dec0)), 1e-3)) % 360
        z = z0 + z_jitter * np.where(rng.random(n) < 0.5, -1.0, 1.0)
        src = perm
    return pa.table({"ra": ra, "dec": dec, "z": z, "z_type": ["spec"] * n,
                     "hpix32": radec_to_pix(32, ra, dec), "source_row": src})


def test_rules_hash_and_matrices():
    r = CrossMatchRules(dz_tol=1e-2, dz_tol_by_pair=(("spec", "phot", 0.1),), reject_pairs=(("phot", "phot"),))
    assert r.hash() == CrossMatchRules.from_dict(r.to_dict()).hash() and len(r.hash()) == 16
    tol, acc = r.matrices(["spec", "phot", "none"])
    assert tol[0, 1] == tol[1, 0] == 0.1 and tol[0, 0] == 1e-2 and np.isnan(tol[0, 2]) and not acc[1, 1] and acc[0, 1]
    r2 = CrossMatchRules(dz_tol=1e-2, dz_tol_by_pair=(("phot", "spec", 0.1),), reject_pairs=(("phot", "phot"),))
    assert r2.hash() == r.hash()


def test_rules_validation():
    with pytest.raises(ValidationError):
        CrossMatchRules(sky_tol_arcsec=0)
    with pytest.raises(ValidationError):
        CrossMatchRules(dz_tol=-1)
    with pytest.raises(ValidationError):
        CrossMatchRules(reject_pairs=(("spec", "bogus"),)).matrices(["spec"])


def test_match_pair_recovers_true_pairs_across_blocks():
    a = _cat(20_000, 0, survey="a")
    b = _cat(20_000, 0, survey="b", jitter_arcsec=0.3, z_jitter=1e-4, base=a)
    p = match_pair(a, b, CrossMatchRules(sky_tol_arcsec=1.0), block_nside=8)
    truth = b.column("source_row").to_numpy()
    assert len(p.left) == 20_000 and np.array_equal(np.sort(p.left), np.arange(20_000)) and np.all(truth[p.right] == p.left)
    assert p.sep_arcsec.max() < 1.0
    assert p.left.dtype == np.int64 and p.sep_arcsec.dtype == np.float64


def test_match_pair_dz_and_reject():
    a = _cat(500, 1, survey="a")
    b = _cat(500, 1, survey="b", base=a, z_jitter=0.05)
    assert len(match_pair(a, b, CrossMatchRules(dz_tol=1e-3)).left) == 0
    assert len(match_pair(a, b, CrossMatchRules(dz_tol=None)).left) == 500
    b2 = b.set_column(b.schema.get_field_index("z_type"), "z_type", pa.array(["phot"] * 500))
    assert len(match_pair(a, b2, CrossMatchRules(dz_tol=None, reject_pairs=(("spec", "phot"),))).left) == 0


def test_match_pair_block_boundary_and_pole():
    ra_a, dec_a = [359.9999, 0.0, 10.0], [0.0, 0.0, 89.9998]
    ra_b, dec_b = [0.0001, 190.0], [0.0, 89.9998]
    a = pa.table({"ra": ra_a, "dec": dec_a, "z": [1.0] * 3, "z_type": ["spec"] * 3, "hpix32": radec_to_pix(32, ra_a, dec_a)})
    b = pa.table({"ra": ra_b, "dec": dec_b, "z": [1.0] * 2, "z_type": ["spec"] * 2, "hpix32": radec_to_pix(32, ra_b, dec_b)})
    p = match_pair(a, b, CrossMatchRules(sky_tol_arcsec=2.0))
    assert sorted(zip(p.left.tolist(), p.right.tolist())) == [(0, 0), (1, 0), (2, 1)]


def test_match_pair_nan_z_and_none_type():
    ra, dec = [10.0, 20.0, 30.0], [5.0, 5.0, 5.0]
    hp32 = radec_to_pix(32, ra, dec)
    a = pa.table({"ra": ra, "dec": dec, "z": [float("nan"), float("nan"), 1.0],
                  "z_type": ["spec", "none", "none"], "hpix32": hp32})
    b = pa.table({"ra": ra, "dec": dec, "z": [1.0, 1.0, float("nan")],
                  "z_type": ["spec", "spec", "spec"], "hpix32": hp32})
    p = match_pair(a, b, CrossMatchRules(dz_tol=1e-2))
    # row 0: nan z with finite tol fails; rows 1,2: "none" -> no z cut
    assert p.left.tolist() == [1, 2]
    assert len(match_pair(a, b, CrossMatchRules(dz_tol=None)).left) == 3


def test_match_pair_errors_and_empty():
    a = _cat(10, 0, survey="a")
    with pytest.raises(SchemaError):
        match_pair(a.drop_columns(["hpix32"]), a, CrossMatchRules())
    with pytest.raises(ValidationError):
        match_pair(a, a, CrossMatchRules(sky_tol_arcsec=1e6))
    bad = a.set_column(a.schema.get_field_index("z_type"), "z_type", pa.array(["zzz"] * 10))
    with pytest.raises(SchemaError):
        match_pair(a, bad, CrossMatchRules())
    p = match_pair(a.slice(0, 0), a, CrossMatchRules())
    assert len(p.left) == 0 and p.left.dtype == np.int64


def test_match_pair_performance_smoke():
    n = 200_000
    a = _cat(n, 5, survey="a")
    rng = np.random.default_rng(9)
    k = n // 5
    other = _cat(n - k, 77, survey="b")
    ra = np.concatenate([a.column("ra").to_numpy()[:k], other.column("ra").to_numpy()])
    dec = np.concatenate([a.column("dec").to_numpy()[:k], other.column("dec").to_numpy()])
    z = np.concatenate([a.column("z").to_numpy()[:k], other.column("z").to_numpy()])
    b = pa.table({"ra": ra, "dec": dec, "z": z, "z_type": ["spec"] * n, "hpix32": radec_to_pix(32, ra, dec)})
    t0 = time.perf_counter()
    p = match_pair(a, b, CrossMatchRules(dz_tol=None))
    dt = time.perf_counter() - t0
    print(f"match_pair 2x200k: {dt:.2f}s")
    assert dt < 5.0 and len(p.left) >= k
    _ = rng


def test_components_refuses_same_survey_merge():
    survey_of = np.array([0, 1, 0], np.int16)
    left = np.array([0, 2])
    right = np.array([1, 1])
    sep = np.array([0.5, 0.2])
    lab = components(3, left, right, sep, survey_of=survey_of, allow_same_survey_merge=False)
    assert lab[1] == lab[2] and lab[0] != lab[1]
    lab = components(3, left, right, sep, survey_of=survey_of, allow_same_survey_merge=True)
    assert len(set(lab)) == 1


def test_components_labels_first_appearance():
    survey_of = np.array([0, 1, 0, 1], np.int16)
    lab = components(4, np.array([2]), np.array([3]), np.array([1.0]), survey_of=survey_of, allow_same_survey_merge=False)
    assert lab.tolist() == [0, 1, 2, 2]


def test_separate_rules_normalised_and_roundtrip():
    r1 = CrossMatchRules(separate=(("b", "a"), ("c", "d")))
    r2 = CrossMatchRules(separate=(("d", "c"), ("a", "b"), ("a", "b")))
    assert r1.hash() == r2.hash() and r1.to_dict() == r2.to_dict()
    assert CrossMatchRules.from_dict(r1.to_dict()).hash() == r1.hash()
    assert r1.hash() != CrossMatchRules().hash()
    with pytest.raises(ValidationError):
        CrossMatchRules(separate=(("a",),))
    with pytest.raises(ValidationError):
        CrossMatchRules(separate=(("a", "a"),))


def test_components_forbid_blocks_transitive_merge():
    survey_of = np.array([0, 1, 2], np.int16)
    left, right, sep = np.array([0, 1]), np.array([1, 2]), np.array([0.1, 0.2])
    lab = components(3, left, right, sep, survey_of=survey_of, allow_same_survey_merge=False)
    assert len(set(lab)) == 1
    forbid = np.array([1 << 2, 0, 1 << 0], np.int64)
    lab = components(3, left, right, sep, survey_of=survey_of, allow_same_survey_merge=False,
                     forbid=forbid)
    assert lab[0] == lab[1] and lab[2] != lab[1]
    # forbid is also honoured when same-survey merges are allowed
    lab = components(3, left, right, sep, survey_of=survey_of, allow_same_survey_merge=True,
                     forbid=forbid)
    assert lab[0] == lab[1] and lab[2] != lab[1]


def test_null_z_type_is_none_code():
    from oneuniverse.data.crossmatch import _prepare
    from oneuniverse.data.schema import register_z_type, z_types
    name = "aaa_test"
    if name not in z_types():
        register_z_type(name)
    try:
        names = sorted(z_types())
        assert names.index(name) < names.index("none")
        ra, dec = [10.0, 20.0], [5.0, 5.0]
        t = pa.table({"ra": ra, "dec": dec, "z": [0.1, 0.2], "z_type": pa.array([None, "spec"], pa.string()),
                      "hpix32": radec_to_pix(32, ra, dec)})
        _, _, code, _ = _prepare(t, names)
        assert code[0] == names.index("none") and code[1] == names.index("spec")
    finally:
        from oneuniverse.data import schema
        schema._registered_z_types.discard(name)
