# File: omnibioai_model_registry/package/validate.py
from __future__ import annotations
from pathlib import Path
from .layout import REQUIRED_FILES
from ..errors import ValidationError


def validate_package_files(version_dir: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (version_dir / f).exists()]
    if missing:
        raise ValidationError(f"Model package missing required files: {missing}. In: {version_dir}")
