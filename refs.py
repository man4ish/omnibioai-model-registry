# File: omnibioai_model_registry/refs.py
from dataclasses import dataclass
from .errors import InvalidModelRef


@dataclass(frozen=True)
class ModelRef:
    model_name: str
    selector: str  # alias or version


def parse_model_ref(model_ref: str) -> ModelRef:
    """
    Accepts:
      - "human_pbmc@production"
      - "human_pbmc@2026-02-13_001"
    """
    if not model_ref or "@" not in model_ref:
        raise InvalidModelRef(f"Invalid model_ref '{model_ref}'. Expected '<model_name>@<alias_or_version>'.")
    model_name, selector = model_ref.split("@", 1)
    model_name = model_name.strip()
    selector = selector.strip()
    if not model_name or not selector:
        raise InvalidModelRef(f"Invalid model_ref '{model_ref}'. Expected '<model_name>@<alias_or_version>'.")
    return ModelRef(model_name=model_name, selector=selector)
