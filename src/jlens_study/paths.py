"""Canonical repository and artifact paths."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(
    os.environ.get("JLENS_ARTIFACT_ROOT", REPO_ROOT / "artifacts")
).expanduser()
UPSTREAM_JLENS = REPO_ROOT / "references" / "upstream" / "anthropic-jacobian-lens"


def split_lens_dir(method: str) -> Path:
    """Return the canonical v2 split-lens directory for ``j`` or ``r``."""
    if method not in {"j", "r"}:
        raise ValueError(f"unknown lens method: {method!r}")
    return ARTIFACT_ROOT / "lenses" / method / "v2_split"


LEGACY_J_RUNS = ARTIFACT_ROOT / "runs" / "j" / "legacy"
COMPARISONS = ARTIFACT_ROOT / "comparisons"
