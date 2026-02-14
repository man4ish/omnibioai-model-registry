# File: omnibioai_model_registry/storage/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """
    Minimal storage backend interface for v0.1.0.

    LocalFS implements this today.
    Future backends: S3, Azure Blob, etc.
    """

    @abstractmethod
    def ensure_dirs(self, path: Path) -> None: ...

    @abstractmethod
    def exists(self, path: Path) -> bool: ...

    @abstractmethod
    def copy_tree(self, src_dir: Path, dst_dir: Path) -> None: ...

    @abstractmethod
    def atomic_write_text(self, path: Path, text: str) -> None: ...
