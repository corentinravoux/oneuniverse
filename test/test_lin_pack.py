"""S17 T2 — packed_npy native format + adapter."""
import json

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.linear.pack import write_packed_native
from oneuniverse.simulation.oufsim.native import get_adapter


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_packed_native_is_chunk_sorted_with_ranges(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2)
    pk = write_packed_native(lin, tmp_path / "pk", particle_chunk_nside=4)
    hdr = json.loads((pk / "header.json").read_text())
    assert hdr["native_format"] == "packed_npy"
    ci = hdr["snapshots"]["z0.000"]["chunk_index"]
    # contiguous, non-overlapping, covering all rows
    assert ci[0]["row_start"] == 0
    assert all(ci[i]["row_stop"] == ci[i + 1]["row_start"]
               for i in range(len(ci) - 1))
    assert ci[-1]["row_stop"] == 32 ** 3


def test_packed_adapter_reads_field_and_rows(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2)
    pk = write_packed_native(lin, tmp_path / "pk", particle_chunk_nside=4)
    hdr = json.loads((pk / "header.json").read_text())
    ad = get_adapter("packed_npy")
    # field region matches the linear field
    fpath = pk / hdr["fields"]["z0.000"]["file"]
    sub = ad.read_field_region(fpath, (slice(0, 8), slice(0, 8), slice(0, 8)))
    ref = np.load(lin / "z0.000" / "field.npy")[:8, :8, :8]
    assert np.allclose(sub, ref)
    # row read of the first chunk returns named columns inside that chunk bbox
    c0 = hdr["snapshots"]["z0.000"]["chunk_index"][0]
    ppath = pk / hdr["snapshots"]["z0.000"]["file"]
    cols = ad.read_rows(ppath, slice(c0["row_start"], c0["row_stop"]),
                        columns=("x", "y", "z"))
    assert set(cols) == {"x", "y", "z"}
    assert cols["x"].min() >= c0["xlo"] - 1e-6
    assert cols["x"].max() <= c0["xhi"] + 1e-6
