#!/usr/bin/env python3
"""Build a small dummy linear simulation, store it in OUF-Sim, profile the
pipeline for optimisation hotspots, and emit diagnostic plots.

Run:
    python3 scripts/build_demo_oufsim.py

Outputs (under OUT_ROOT/linsim_demo/):
    _native/        native linear-sim layout (config.yaml + per-z files)
    oufsim/         OUF-Sim store (manifest.json + parquet/tiles + indexes)
    plots/          diagnostic PNGs
    RUN_SUMMARY.json, OPTIMIZATION_FINDINGS.md
"""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import time
import tracemalloc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import healpy as hp  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from oneuniverse.simulation.cosmology import CosmologySpec  # noqa: E402
from oneuniverse.simulation.linear import (  # noqa: E402
    generate_linear_sim, linear_power,
)
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store  # noqa: E402
from oneuniverse.simulation.selectors import Cone, Cube  # noqa: E402

OUT_ROOT = Path("/home/ravoux/Documents/Science/Cosmography/oneuniverse_simulation")
SIM_NAME = "linsim_demo"
BOX = 512.0          # Mpc/h
NGRID = 128          # cells per side -> 2,097,152 particles
REDSHIFTS = (0.0, 0.5, 1.0, 2.0)
SEED = 42
COSMO = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                      sigma8=0.81, t_cmb=2.7255)


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    base = OUT_ROOT / SIM_NAME
    native_dir = base / "_native"
    plots = base / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    timings = {}

    # --- 1. generate native dummy sim --------------------------------------
    t0 = time.perf_counter()
    generate_linear_sim(native_dir, COSMO, box_size=BOX, n_grid=NGRID,
                        redshifts=REDSHIFTS, seed=SEED, with_lightcone=True)
    timings["generate_s"] = round(time.perf_counter() - t0, 3)

    # --- 2. convert to OUF-Sim store (clean wall time) ---------------------
    t0 = time.perf_counter()
    store = write_oufsim_store(native_dir, OUT_ROOT, sim_name=SIM_NAME,
                               particle_chunk_nside=4, field_tile_cells=32,
                               lightcone_nside_part=2, overwrite=True)
    timings["convert_s"] = round(time.perf_counter() - t0, 3)

    # --- 3. profiling pass: tracemalloc peak + cProfile hotspots -----------
    tmp_store = OUT_ROOT / (SIM_NAME + "_profile")
    tracemalloc.start()
    write_oufsim_store(native_dir, OUT_ROOT, sim_name=SIM_NAME + "_profile",
                       particle_chunk_nside=4, field_tile_cells=32,
                       lightcone_nside_part=2, overwrite=True)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings["convert_peak_mb"] = round(peak / 1e6, 1)

    prof = cProfile.Profile()
    prof.enable()
    write_oufsim_store(native_dir, OUT_ROOT, sim_name=SIM_NAME + "_profile",
                       particle_chunk_nside=4, field_tile_cells=32,
                       lightcone_nside_part=2, overwrite=True)
    prof.disable()
    sbuf = io.StringIO()
    pstats.Stats(prof, stream=sbuf).sort_stats("cumulative").print_stats(18)
    hotspots = sbuf.getvalue()
    import shutil
    shutil.rmtree(tmp_store, ignore_errors=True)

    # --- 4. partial-access measurement -------------------------------------
    s = SimStore(store)
    access = {}
    t0 = time.perf_counter()
    cube = Cube(0, 64, 0, 64, 0, 64)
    sel = s.read_box("snapshots", 0.0, cube)
    access["box_read_s"] = round(time.perf_counter() - t0, 4)
    access["box_chunks"] = s.last_read_stats
    access["box_n_returned"] = int(len(sel["x"]))
    sub, _ = s.read_field_box(0.0, cube)
    access["field_tiles"] = s.last_read_stats
    cone = Cone(lon=45.0, lat=10.0, radius_deg=25.0)
    cobj = s.read_cone(cone)
    access["cone_pixels"] = s.last_read_stats
    access["cone_n_returned"] = int(len(cobj.get("lon", [])))

    # --- 5. plots ----------------------------------------------------------
    _plot_power_spectrum(plots / "01_power_spectrum.png")
    _plot_field_slice(native_dir, plots / "02_field_slice.png")
    _plot_particles_halos(native_dir, plots / "03_particles_halos.png")
    _plot_mass_function(native_dir, plots / "04_halo_mass_function.png")
    _plot_lightcone(native_dir, plots / "05_lightcone_mollview.png")
    _plot_partial_access(s, plots / "06_partial_access.png")

    # --- 6. summary + findings --------------------------------------------
    summary = {
        "sim_name": SIM_NAME, "box_Mpc_h": BOX, "n_grid": NGRID,
        "n_particles_per_snapshot": NGRID ** 3, "redshifts": list(REDSHIFTS),
        "seed": SEED,
        "native_size_mb": round(_dir_size_mb(native_dir), 1),
        "store_size_mb": round(_dir_size_mb(store), 1),
        "timings": timings, "partial_access": access,
    }
    (base / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    (base / "cprofile_convert.txt").write_text(hotspots)
    print(json.dumps(summary, indent=2))
    print("\n--- cProfile (convert, cumulative top) ---\n")
    print(hotspots)


def _plot_power_spectrum(path):
    k = np.logspace(-2.5, 0.5, 200)
    fig, ax = plt.subplots(figsize=(6, 5))
    for z in REDSHIFTS:
        ax.loglog(k, linear_power(k, COSMO, z=z), label=f"z={z}")
    ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel("P(k) [(Mpc/h)$^3$]")
    ax.set_title("Eisenstein–Hu linear P(k)"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_field_slice(native_dir, path):
    field = np.load(native_dir / "z0.000" / "field.npy")
    sl = field[:, :, field.shape[2] // 2]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sl.T, origin="lower", extent=(0, BOX, 0, BOX), cmap="magma")
    ax.set_xlabel("x [Mpc/h]"); ax.set_ylabel("y [Mpc/h]")
    ax.set_title("density field slice (z=0)")
    fig.colorbar(im, ax=ax, label=r"$\delta$")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_particles_halos(native_dir, path):
    parts = np.load(native_dir / "z0.000" / "particles.npy")
    halos = pq.read_table(native_dir / "z0.000" / "halos.parquet")
    slab = parts[:, 2] < BOX / NGRID * 4
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(parts[slab, 0], parts[slab, 1], s=0.5, alpha=0.25, color="0.4")
    hx = halos.column("x").to_numpy(); hy = halos.column("y").to_numpy()
    ax.scatter(hx, hy, s=8, color="tab:red", marker="x", label="halos")
    ax.set_xlim(0, BOX); ax.set_ylim(0, BOX)
    ax.set_xlabel("x [Mpc/h]"); ax.set_ylabel("y [Mpc/h]")
    ax.set_title("Zel'dovich particles (slab) + halos (z=0)"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_mass_function(native_dir, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for z in REDSHIFTS:
        m = pq.read_table(native_dir / f"z{z:.3f}" / "halos.parquet") \
            .column("mass").to_numpy()
        if len(m) == 0:
            continue
        ax.hist(np.log10(m), bins=25, histtype="step", label=f"z={z}")
    ax.set_xlabel(r"$\log_{10}(M\,[M_\odot/h])$"); ax.set_ylabel("N halos")
    ax.set_yscale("log"); ax.set_title("toy halo mass function"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_lightcone(native_dir, path):
    lc = pq.read_table(native_dir / "lightcone.parquet")
    pix = lc.column("_healpix32").to_numpy()
    nside = 32
    m = np.zeros(hp.nside2npix(nside))
    np.add.at(m, pix, 1.0)
    m[m == 0] = hp.UNSEEN
    hp.mollview(m, nest=True, title="lightcone halo counts (HEALPix NSIDE32)",
                unit="N", cmap="viridis")
    hp.graticule()
    plt.savefig(path, dpi=110); plt.close("all")


def _plot_partial_access(store, path):
    sizes = [32, 64, 128, 256, 512]
    frac_chunks, frac_tiles = [], []
    for L in sizes:
        cube = Cube(0, L, 0, L, 0, L)
        store.read_box("snapshots", 0.0, cube)
        st = store.last_read_stats
        frac_chunks.append(st["chunks_read"] / st["chunks_total"])
        store.read_field_box(0.0, cube)
        st = store.last_read_stats
        frac_tiles.append(st["tiles_read"] / st["tiles_total"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sizes, frac_chunks, "o-", label="particle chunks read")
    ax.plot(sizes, frac_tiles, "s-", label="field tiles read")
    ax.set_xlabel("cube side [Mpc/h]")
    ax.set_ylabel("fraction of partitions touched")
    ax.set_title("partial access: data touched vs query size"); ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
