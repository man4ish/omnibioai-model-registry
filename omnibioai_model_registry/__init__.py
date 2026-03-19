# File: omnibioai_model_registry/__init__.py
from .api import (
    ModelRegistry,
    promote_model,
    register_model,
    resolve_model,
    verify_model_ref,
)

__all__ = [
    "ModelRegistry",
    "register_model",
    "resolve_model",
    "promote_model",
    "verify_model_ref",
]
