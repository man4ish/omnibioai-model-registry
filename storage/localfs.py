# File: omnibioai_model_registry/storage/localfs.py
from __future__ import annotations
import os
import shutil
import tempfile
from pathlib import Path
from .base import StorageBackend


class LocalFS(StorageBackend):
    def ensure_dirs(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def exists(self, path: Path) -> bool:
        return path.exists()

    def copy_tree(self, src_dir: Path, dst_dir: Path) -> None:
        # shutil.copytree requires dst doesn't exist
        shutil.copytree(src_dir, dst_dir)

    def atomic_write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        finally:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except Exception:
                pass
