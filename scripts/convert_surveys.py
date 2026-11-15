"""Convert raw survey files to OUF datasets with ``oneuniverse.data.convert``.

Example:
    python scripts/convert_surveys.py --data-root /data/raw --db-root /data/ouf \
        --surveys eboss_qso desi_qso
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oneuniverse.data import Database, convert, list_surveys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True,
                   help="directory holding one sub-directory of raw files per survey")
    p.add_argument("--db-root", type=Path, required=True,
                   help="Database root receiving one OUF dataset per survey")
    p.add_argument("--surveys", nargs="+", default=["eboss_qso", "desi_qso"],
                   help="registered survey names (default: eboss_qso desi_qso)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing dataset (previous manifest kept in _history)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    known = set(list_surveys())
    unknown = sorted(set(args.surveys) - known)
    if unknown:
        raise SystemExit(f"unknown survey(s) {unknown}; registered: {sorted(known)}")
    args.db_root.mkdir(parents=True, exist_ok=True)
    for survey in args.surveys:
        manifest = convert(survey, source=args.data_root / survey, out=args.db_root / survey,
                           overwrite=args.overwrite)
        print(f"{survey}: {manifest.n_rows:,} rows")
    print(json.dumps(Database(args.db_root).summary(), indent=2))


if __name__ == "__main__":
    main()
