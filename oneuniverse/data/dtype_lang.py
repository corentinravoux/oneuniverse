"""Tiny dtype mini-language for OUF 2.3 variable-length payloads.

Grammar:

    f4               -> pa.float32()
    f8               -> pa.float64()
    i1 i2 i4 i8      -> pa.int8 / int16 / int32 / int64
    u1 u2 u4 u8      -> pa.uint8 / uint16 / uint32 / uint64
    U<N>             -> pa.string()   (string columns; N kept for the loader)
    <scalar>[N]      -> pa.FixedSizeList(scalar, N)
    list<scalar>     -> pa.list_(scalar)
    large_list<scalar> -> pa.large_list(scalar)

Whitespace is rejected. Used by ``_chunk_to_table`` to coerce list /
fixed-size list / variable-length list columns to the right pyarrow
type at write time.
"""
from __future__ import annotations

import re
from typing import Dict

import pyarrow as pa

_SCALAR_MAP: Dict[str, pa.DataType] = {
    "f4": pa.float32(),
    "f8": pa.float64(),
    "i1": pa.int8(),
    "i2": pa.int16(),
    "i4": pa.int32(),
    "i8": pa.int64(),
    "u1": pa.uint8(),
    "u2": pa.uint16(),
    "u4": pa.uint32(),
    "u8": pa.uint64(),
}

_FIXED_RE = re.compile(r"^([a-z]\d)\[(\d+)\]$")
_LIST_RE = re.compile(r"^list<([a-z]\d)>$")
_LARGE_LIST_RE = re.compile(r"^large_list<([a-z]\d)>$")


def parse_dtype(spec: str) -> pa.DataType:
    """Parse a dtype mini-language string into a pyarrow type."""
    if not isinstance(spec, str) or " " in spec:
        raise ValueError(f"invalid dtype string {spec!r} (no whitespace)")
    if spec in _SCALAR_MAP:
        return _SCALAR_MAP[spec]
    if spec.startswith("U") and spec[1:].isdigit():
        return pa.string()
    m = _FIXED_RE.match(spec)
    if m:
        scalar, n = m.group(1), int(m.group(2))
        if scalar not in _SCALAR_MAP:
            raise ValueError(f"unknown scalar {scalar!r} in dtype {spec!r}")
        return pa.list_(_SCALAR_MAP[scalar], n)
    m = _LIST_RE.match(spec)
    if m:
        return pa.list_(_SCALAR_MAP[m.group(1)])
    m = _LARGE_LIST_RE.match(spec)
    if m:
        return pa.large_list(_SCALAR_MAP[m.group(1)])
    raise ValueError(
        f"unsupported dtype {spec!r}; allowed forms: f4 / i8 / U32 / "
        f"f4[N] / list<f4> / large_list<f4>"
    )


def is_variable_length(spec: str) -> bool:
    """Return True iff ``spec`` produces a variable-length pyarrow type."""
    return spec.startswith("list<") or spec.startswith("large_list<")
