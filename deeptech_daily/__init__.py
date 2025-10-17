"""Compatibility package exposing the DeepTech Daily tooling."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
LEGACY_PACKAGE_DIR = PACKAGE_ROOT.parent / "deeptech-daily"

__all__ = []
__path__ = [str(LEGACY_PACKAGE_DIR)]
