"""Stable hashes for instances, configs, and scenario files."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(data: Any) -> str:
    """Serialize ``data`` with sorted keys and no insignificant whitespace."""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(data: Any) -> str:
    """Return the hex SHA-256 of :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 of raw bytes."""
    return hashlib.sha256(payload).hexdigest()
