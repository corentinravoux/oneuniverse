#!/usr/bin/env python3
"""S17 demo — scale sweep + wrap-vs-reencode diagnostic figure.

Runs the multi-backend scale sweep (packed_npy converter, reencode vs
reference projection) and renders a 2-panel figure:
  left  — convert peak memory + wall time vs particle count (bounded-memory)
  right — store size by projection (native / reference / reencode)

Outputs RUN_SUMMARY.json into the science demo dir and the figure into the
repo's test/test_output (committed diagnostic, visual-testing convention).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.oufsim.scale_bench import sweep

DEMO = Path("/home/ravoux/Documents/Science/Cosmography/"
            "oneuniverse_simulation/s17_demo")
FIG = Path(__file__).resolve().parent.parent / "test" / "test_output" / "s17_scaling.png"


def main():
    cosmo = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                          sigma8=0.81, t_cmb=2.7255)
    DEMO.mkdir(parents=True, exist_ok=True)
    rows = sweep(DEMO / "work", cosmo, grids=(32, 48, 64))
    (DEMO / "RUN_SUMMARY.json").write_text(json.dumps(rows, indent=2))

    npart = [r["n_particles"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax0 = ax[0]
    ax0.plot(npart, [r["convert_peak_mb"] for r in rows], "o-", color="C0",
             label="convert peak (MB)")
    ax0.set_xlabel("particles"); ax0.set_ylabel("peak memory [MB]", color="C0")
    ax0.tick_params(axis="y", labelcolor="C0")
    ax0b = ax0.twinx()
    ax0b.plot(npart, [r["convert_wall_s"] for r in rows], "s--", color="C3",
              label="convert wall (s)")
    ax0b.set_ylabel("wall time [s]", color="C3")
    ax0b.tick_params(axis="y", labelcolor="C3")
    ax0.set_title("convert cost vs particle count (bounded memory)")
    ax0.grid(alpha=.3)

    ax1 = ax[1]
    import numpy as np
    x = np.arange(len(rows)); w = 0.25
    ax1.bar(x - w, [r["native_mb"] for r in rows], w, label="native")
    ax1.bar(x, [r["store_reference_mb"] for r in rows], w, label="reference")
    ax1.bar(x + w, [r["store_reencode_mb"] for r in rows], w, label="reencode")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{r['n_grid']}³" for r in rows])
    ax1.set_ylabel("store size [MB]"); ax1.set_xlabel("grid")
    ax1.set_title("storage: wrap-in-place (reference) vs re-encode")
    ax1.legend(); ax1.grid(alpha=.3, axis="y")

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=110)
    print("rows:", json.dumps(rows, indent=2))
    print("figure:", FIG)


if __name__ == "__main__":
    main()
