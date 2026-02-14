# File: omnibioai_model_registry/config.py
import os
from dataclasses import dataclass
from .errors import RegistryNotConfigured


@dataclass(frozen=True)
class RegistryConfig:
    root: str
    backend: str = "localfs"  # future: s3, azure_blob
    strict_verify: bool = True


def load_config() -> RegistryConfig:
    root = os.getenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "").strip()
    if not root:
        raise RegistryNotConfigured(
            "OMNIBIOAI_MODEL_REGISTRY_ROOT is not set. "
            "Example: export OMNIBIOAI_MODEL_REGISTRY_ROOT=~/Desktop/machine/local_registry/model_registry"
        )
    backend = os.getenv("OMNIBIOAI_MODEL_REGISTRY_BACKEND", "localfs").strip() or "localfs"
    strict_verify = os.getenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "1").strip() not in {"0", "false", "False"}
    return RegistryConfig(root=root, backend=backend, strict_verify=strict_verify)
