"""Database facade over a root directory of OUF 3.0 datasets."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from oneuniverse._core.errors import FormatError, NotFoundError, ValidationError
from oneuniverse._core.fs import is_staging
from oneuniverse.data.manifest import MANIFEST_FILENAME, Manifest, read_manifest
from oneuniverse.data.sidecar import VersionedSidecar
from oneuniverse.data.view import DatasetView

_HISTORY = "_history"
_MAX_DEPTH = 2


def _parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(UTC)


class Database:
    """Datasets found under *root*.

    Discovery walks *root* to depth 2: a directory holding ``manifest.json`` is a
    dataset (not descended into); names starting with ``_`` and ``<x>.tmp`` /
    ``<x>.old`` staging directories are skipped. The dataset name is its relative
    path with ``/`` replaced by ``__``; two datasets mapping to the same name
    (``a/b`` and ``a__b``) are a collision: FormatError when strict, else both are
    excluded and recorded in :attr:`errors`. Symlinked directories are not followed. With ``strict=False`` datasets whose
    manifest cannot be read are skipped and recorded in :attr:`errors`.
    ``as_of_time`` is the restriction time given to :meth:`as_of`, else ``None``.
    """

    def __init__(self, root, *, as_of: datetime | None = None, strict: bool = True) -> None:
        self.root = Path(root)
        self.strict = strict
        self.errors: dict[str, str] = {}
        self.as_of_time = self._check_when(as_of) if as_of is not None else None
        self._paths: dict[str, Path] = {}
        self._manifests: dict[str, Manifest] = {}
        self._discover()

    @staticmethod
    def _check_when(when: datetime) -> datetime:
        if not isinstance(when, datetime) or when.tzinfo is None:
            raise ValidationError(f"as_of needs a timezone-aware datetime, got {when!r}")
        return when

    def _candidates(self):
        def walk(d: Path, depth: int):
            try:
                entries = sorted(p for p in d.iterdir() if p.is_dir())
            except OSError as e:
                raise FormatError(f"cannot list {d}: {e}") from e
            for p in entries:
                if p.name.startswith("_") or is_staging(p) or p.is_symlink():
                    continue
                if (p / MANIFEST_FILENAME).is_file():
                    yield p
                elif depth < _MAX_DEPTH:
                    yield from walk(p, depth + 1)
        if not self.root.is_dir():
            raise NotFoundError(f"database root {self.root} is not a directory")
        yield from walk(self.root, 1)

    def _discover(self) -> None:
        groups: dict[str, list[Path]] = {}
        for p in self._candidates():
            groups.setdefault("__".join(p.relative_to(self.root).parts), []).append(p)
        found: dict[str, tuple[Path, Manifest]] = {}
        for name, paths in groups.items():
            problems: list[str] = []
            m = None
            err: Exception | None = None
            for p in paths:
                try:
                    m = read_manifest(p / MANIFEST_FILENAME)
                except (FormatError, OSError, KeyError, ValueError) as e:
                    err = e
                    problems.append(f"dataset {p}: unreadable manifest: {e}")
            clash = len(paths) > 1
            if clash:
                problems.insert(0, f"dataset name {name!r} is ambiguous: "
                                + ", ".join(str(q) for q in paths))
            if problems:
                if self.strict:
                    raise FormatError("; ".join(problems)) from err
                self.errors[name] = "; ".join(problems)
                continue
            found[name] = (paths[0], m)
        when = self.as_of_time
        for name in sorted(found):
            p, m = found[name]
            if when is not None and _parse_utc(m.validity.valid_from) > when:
                continue
            self._paths[name], self._manifests[name] = p, m

    # ------------------------------------------------------------- mapping
    def names(self) -> list[str]:
        return sorted(self._paths)

    def __contains__(self, name: object) -> bool:
        return name in self._paths

    def __iter__(self):
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._paths)

    def _need(self, name: str) -> None:
        if name not in self._paths:
            raise NotFoundError(f"no dataset {name!r} in {self.root}; available: {self.names()}")

    def path(self, name: str) -> Path:
        self._need(name)
        return self._paths[name]

    def manifest(self, name: str) -> Manifest:
        self._need(name)
        return self._manifests[name]

    def __getitem__(self, name: str) -> DatasetView:
        """A fresh view each call (views carry a stale-guard, so none is cached)."""
        return DatasetView.from_path(self.path(name))

    # ------------------------------------------------------------- history
    def history(self, name: str) -> list[Manifest]:
        """Archived manifests plus the live one, oldest first."""
        p = self.path(name)
        hist = p / _HISTORY
        out = [self._manifests[name]]
        if hist.is_dir():
            out += [read_manifest(f) for f in sorted(hist.glob("*.manifest.json"))]
        return sorted(out, key=lambda m: _parse_utc(m.created_utc))

    def as_of(self, when: datetime) -> Database:
        """New Database restricted to datasets that existed at *when*.

        Inclusion/exclusion only: a dataset is kept when its **live** manifest was
        created at or before *when* (``validity.valid_from <= when``). A dataset
        whose live version postdates *when* is excluded even if an archived version
        existed then, because the data of old versions is not kept.
        """
        return Database(self.root, as_of=self._check_when(when), strict=self.strict)

    # ------------------------------------------------------------- oneuid
    def build_oneuid(self, datasets, *, rules, name: str = "default"):
        from oneuniverse.data.oneuid import build_oneuid
        return build_oneuid(self, datasets, rules=rules, name=name)

    def load_oneuid(self, name: str = "default", *, as_of=None):
        from oneuniverse.data.oneuid import load_oneuid
        return load_oneuid(self, name, as_of=as_of)

    def list_oneuids(self) -> list[str]:
        return VersionedSidecar.names(self.root, "oneuid")

    def oneuid_query(self, name: str = "default"):
        from oneuniverse.data.oneuid import OneuidQuery
        return OneuidQuery(self.load_oneuid(name), self)

    # ------------------------------------------------------------- summary
    def summary(self) -> dict:
        datasets = {}
        for n, m in self._manifests.items():
            datasets[n] = {"n_rows": int(m.n_rows), "geometry": m.geometry.value,
                           "kind": m.survey.kind, "version": int(m.validity.version)}
        return {"datasets": dict(sorted(datasets.items())),
                "oneuids": VersionedSidecar.names(self.root, "oneuid"),
                "links": VersionedSidecar.names(self.root, "links")}
