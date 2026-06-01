"""Sim-side provenance declaration — run history + ingest trail."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ProvenanceSpec:
    code: str
    code_version: Optional[str]
    git_hash: Optional[str]
    original_paths: Tuple[str, ...]
    ingested_utc: str
    converter: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "code_version": self.code_version,
            "git_hash": self.git_hash,
            "original_paths": list(self.original_paths),
            "ingested_utc": self.ingested_utc,
            "converter": self.converter,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProvenanceSpec":
        return cls(
            code=d["code"],
            code_version=d.get("code_version"),
            git_hash=d.get("git_hash"),
            original_paths=tuple(d.get("original_paths", ())),
            ingested_utc=d["ingested_utc"],
            converter=d["converter"],
        )
