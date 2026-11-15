"""Build the linear-theory reference simulation and convert it to an OUF-Sim store.

Refuses an existing --out; writer ValidationErrors are shown as a clear message.

Example:
    python scripts/build_reference_store.py --out /data/ref_store --box 512 --n-grid 128
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from oneuniverse._core.errors import ValidationError
from oneuniverse.reference.native import LinearSimConverter, generate_linear_sim
from oneuniverse.simulation import CosmologySpec, ExecutionPlan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="store directory (must not exist)")
    p.add_argument("--box", type=float, default=512.0, help="box size [Mpc/h] (default 512)")
    p.add_argument("--n-grid", type=int, default=128, help="grid cells per side (default 128)")
    p.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    p.add_argument("--redshifts", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"error: destination {args.out} already exists; refusing to overwrite")
    cosmo = CosmologySpec(0.6736, 0.3153, 0.0493, 0.8111, 0.9649)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            native = generate_linear_sim(Path(tmp) / "native", cosmo, box=args.box,
                                         n_grid=args.n_grid, redshifts=args.redshifts,
                                         seed=args.seed)
            LinearSimConverter().convert(native, args.out, plan=ExecutionPlan())
    except ValidationError as e:
        raise SystemExit(f"error: {e}") from e
    print(args.out)


if __name__ == "__main__":
    main()
