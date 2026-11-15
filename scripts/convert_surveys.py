"""Convert raw survey files to OUF datasets with ``oneuniverse.data.convert``.

Each survey's raw files are read from ``<data-root>/<archive_path>``, where
``archive_path`` is the loader's default archive sub-directory (the survey name
when the loader declares none); ``--source NAME=PATH`` overrides it.

Example:
    python scripts/convert_surveys.py --data-root /data/raw --db-root /data/ouf \
        --surveys eboss_qso desi_qso --source desi_qso=/scratch/desi_qso
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oneuniverse.data import Database, convert, list_surveys
from oneuniverse.data.loaders import get_loader


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True,
                   help="survey archive root; each survey is read from <data-root>/<archive_path>")
    p.add_argument("--db-root", type=Path, required=True,
                   help="Database root receiving one OUF dataset per survey")
    p.add_argument("--surveys", nargs="+", default=["eboss_qso", "desi_qso"],
                   help="registered survey names (default: eboss_qso desi_qso)")
    p.add_argument("--source", action="append", default=[], metavar="NAME=PATH",
                   help="raw-file directory of one survey, overriding the archive layout")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing dataset (previous manifest kept in _history)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    known = set(list_surveys())
    unknown = sorted(set(args.surveys) - known)
    if unknown:
        raise SystemExit(f"unknown survey(s) {unknown}; registered: {sorted(known)}")
    sources = {}
    for item in args.source:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--source expects NAME=PATH, got {item!r}")
        sources[name] = Path(path)
    stray = sorted(set(sources) - set(args.surveys))
    if stray:
        raise SystemExit(f"--source given for survey(s) not converted: {stray}")
    args.db_root.mkdir(parents=True, exist_ok=True)
    for survey in args.surveys:
        source = sources.get(survey)
        if source is None:
            source = args.data_root / (get_loader(survey).archive_path or survey)
        manifest = convert(survey, source=source, out=args.db_root / survey,
                           overwrite=args.overwrite)
        print(f"{survey}: {manifest.n_rows:,} rows")
    print(json.dumps(Database(args.db_root).summary(), indent=2))


if __name__ == "__main__":
    main()
