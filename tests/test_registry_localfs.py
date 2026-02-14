from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibioai_model_registry.api import ModelRegistry
from omnibioai_model_registry.errors import VersionAlreadyExists, ModelNotFound, ValidationError
from omnibioai_model_registry.package.layout import REQUIRED_FILES


def _make_minimal_package(dir_path: Path, *, meta: dict | None = None) -> None:
    """
    Create a minimal valid model package directory that satisfies REQUIRED_FILES.
    Your validate_package_files() requires all of them to exist.
    """
    dir_path.mkdir(parents=True, exist_ok=True)

    # Minimal contents; only existence matters for validation
    (dir_path / "model.pt").write_bytes(b"fake")
    (dir_path / "model_genes.txt").write_text("GeneA\nGeneB\n", encoding="utf-8")
    (dir_path / "label_map.json").write_text(json.dumps({"0": "A", "1": "B"}, indent=2) + "\n", encoding="utf-8")
    (dir_path / "metrics.json").write_text(json.dumps({"acc": 0.9}, indent=2) + "\n", encoding="utf-8")
    (dir_path / "feature_schema.json").write_text(json.dumps({"features": ["GeneA", "GeneB"]}, indent=2) + "\n", encoding="utf-8")

    # registry overwrites model_meta.json after copy, but it must exist for validate_package_files()
    # (and sha256sums.txt is written by registry, but validate checks it too, so we precreate it)
    (dir_path / "model_meta.json").write_text(json.dumps(meta or {}, indent=2) + "\n", encoding="utf-8")

    # validate requires sha256sums.txt exists; registry writes it after copy,
    # but validation happens AFTER write in register_model(). Still, validate checks presence,
    # so in destination it will exist. In source, it's optional. We'll create it anyway.
    (dir_path / "sha256sums.txt").write_text("", encoding="utf-8")


@pytest.fixture
def env_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "registry_root"
    monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", str(root))
    monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_BACKEND", "localfs")
    monkeypatch.setenv("OMNIBIOAI_MODEL_REGISTRY_STRICT_VERIFY", "1")
    return root


def test_register_creates_version_dir_and_metadata(env_root: Path, tmp_path: Path):
    src = tmp_path / "pkg_src"
    _make_minimal_package(src, meta={"framework": "test"})

    reg = ModelRegistry.from_env()
    out = reg.register_model(
        task="celltype_sc",
        model_name="human_pbmc",
        version="2026-02-14_001",
        artifacts_dir=src,
        metadata={"framework": "sklearn", "model_type": "lr"},
        set_alias=None,
    )

    assert out["ok"] is True
    vdir = Path(out["package_path"])
    assert vdir.exists()

    # Required files exist in destination
    for f in REQUIRED_FILES:
        assert (vdir / f).exists()

    # Metadata should have required defaults
    meta = json.loads((vdir / "model_meta.json").read_text())
    assert meta["task"] == "celltype_sc"
    assert meta["model_name"] == "human_pbmc"
    assert meta["version"] == "2026-02-14_001"
    assert "created_at" in meta
    assert meta["framework"] == "sklearn"
    assert meta["model_type"] == "lr"

    # Manifest should contain hashes for required files
    txt = (vdir / "sha256sums.txt").read_text().strip().splitlines()
    assert len(txt) > 0


def test_register_is_immutable(env_root: Path, tmp_path: Path):
    src = tmp_path / "pkg_src"
    _make_minimal_package(src)

    reg = ModelRegistry.from_env()
    reg.register_model(
        task="t",
        model_name="m",
        version="v1",
        artifacts_dir=src,
        metadata={},
        set_alias=None,
    )

    with pytest.raises(VersionAlreadyExists):
        reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=src,
            metadata={},
            set_alias=None,
        )


def test_resolve_by_version_and_alias(env_root: Path, tmp_path: Path):
    src = tmp_path / "pkg_src"
    _make_minimal_package(src)

    reg = ModelRegistry.from_env()
    reg.register_model(
        task="t",
        model_name="m",
        version="v1",
        artifacts_dir=src,
        metadata={},
        set_alias="latest",
        actor="manish",
        reason="unit test",
    )

    # resolve via explicit version selector
    vdir1 = reg.resolve_model(task="t", model_ref="m@v1", verify=True)
    assert Path(vdir1).exists()

    # resolve via alias
    vdir2 = reg.resolve_model(task="t", model_ref="m@latest", verify=True)
    assert str(vdir2) == str(vdir1)

    # alias file exists
    alias_file = env_root / "tasks" / "t" / "models" / "m" / "aliases" / "latest.json"
    assert alias_file.exists()
    alias = json.loads(alias_file.read_text())
    assert alias["version"] == "v1"
    assert alias["actor"] == "manish"


def test_promote_writes_audit_log(env_root: Path, tmp_path: Path):
    src = tmp_path / "pkg_src"
    _make_minimal_package(src)

    reg = ModelRegistry.from_env()
    reg.register_model(
        task="t",
        model_name="m",
        version="v1",
        artifacts_dir=src,
        metadata={},
        set_alias=None,
    )

    reg.promote_model(task="t", model_name="m", alias="production", version="v1", actor="x", reason="y")

    # audit log appended
    log_path = env_root / "tasks" / "t" / "models" / "m" / "audit" / "promotions.jsonl"
    assert log_path.exists()
    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 1
    ev = json.loads(lines[-1])
    assert ev["alias"] == "production"
    assert ev["version"] == "v1"
    assert ev["actor"] == "x"


def test_resolve_missing_raises(env_root: Path):
    reg = ModelRegistry.from_env()
    with pytest.raises(ModelNotFound):
        reg.resolve_model(task="t", model_ref="m@latest", verify=False)


def test_register_fails_if_artifacts_dir_missing(env_root: Path, tmp_path: Path):
    reg = ModelRegistry.from_env()
    missing = tmp_path / "does_not_exist"

    with pytest.raises(ValidationError):
        reg.register_model(
            task="t",
            model_name="m",
            version="v1",
            artifacts_dir=missing,
            metadata={},
        )
