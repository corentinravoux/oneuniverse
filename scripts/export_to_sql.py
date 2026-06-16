#!/usr/bin/env python3
"""Export an OUF or OUF-Sim directory to a SQLite database.

Examples
--------
    # OUF survey dir (the parent of oneuniverse/, or the oneuniverse/ dir itself)
    python scripts/export_to_sql.py /path/to/survey -o survey.sqlite
    python scripts/export_to_sql.py /path/to/survey/oneuniverse -o survey.sqlite

    # OUF-Sim store
    python scripts/export_to_sql.py /path/to/simstore -o sim.sqlite --sim

    # zero-copy DuckDB DDL instead of materialising
    python scripts/export_to_sql.py /path/to/survey --attach
"""
import argparse
import sys
from pathlib import Path


def _make_view(source: Path):
    from oneuniverse.data.dataset_view import DatasetView
    # Accept either the survey dir (parent of oneuniverse/) or the oneuniverse/ dir.
    if source.name == "oneuniverse":
        return DatasetView.from_ou_dir(source)
    if (source / "oneuniverse").is_dir():
        return DatasetView.from_path(source)
    # A bare oneuniverse-style dir (manifest.json directly inside).
    if (source / "manifest.json").exists():
        return DatasetView.from_ou_dir(source)
    raise SystemExit(f"no OUF dataset found at {source}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path,
                   help="OUF survey/oneuniverse dir, or an OUF-Sim store with --sim")
    p.add_argument("-o", "--out", type=Path,
                   help="output .sqlite path (default: <source>.sqlite)")
    p.add_argument("--sim", action="store_true", help="treat source as an OUF-Sim store")
    p.add_argument("--attach", action="store_true",
                   help="print zero-copy DuckDB attach DDL instead of materialising")
    args = p.parse_args(argv)

    if args.sim:
        from oneuniverse.simulation.oufsim.sql import export_sim_sql
        out = args.out or args.source.with_suffix(".sqlite")
        export_sim_sql(args.source, out)
        print(f"wrote {out}")
        return 0

    view = _make_view(args.source)
    if args.attach:
        from oneuniverse.data.sql import attach_sql_ddl
        print(attach_sql_ddl([view]))
        return 0
    from oneuniverse.data.sql import export_sql
    out = args.out or args.source.with_suffix(".sqlite")
    export_sql([view], out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
