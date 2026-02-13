from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from omnibioai_model_registry import (
    ModelRegistry,
    register_model,
    resolve_model,
    promote_model,
    verify_model_ref,
)
from omnibioai_model_registry.errors import ModelRegistryError

APP_VERSION = "0.1.0"
DEFAULT_PREFIX = "/v1"

app = FastAPI(title="OmniBioAI Model Registry Service", version=APP_VERSION)


# -------------------------
# Request/Response models
# -------------------------

class RegisterRequest(BaseModel):
    task: str
    model_name: str
    version: str
    artifacts_dir: str = Field(..., description="Directory path accessible to the service container")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    set_alias: Optional[str] = Field(default="latest", description="Alias to set after register. Use null to skip.")
    actor: Optional[str] = None
    reason: Optional[str] = "api register"


class RegisterResponse(BaseModel):
    ok: bool
    task: str
    model_name: str
    version: str
    package_path: str
    alias_set: Optional[str] = None


class PromoteRequest(BaseModel):
    task: str
    model_name: str
    alias: str
    version: str
    actor: Optional[str] = None
    reason: Optional[str] = None


class VerifyRequest(BaseModel):
    task: str
    ref: str = Field(..., description="model@alias_or_version")


class VerifyResponse(BaseModel):
    ok: bool


class ResolveResponse(BaseModel):
    ok: bool
    path: str


class ShowResponse(BaseModel):
    ok: bool
    meta: Dict[str, Any]
    package_dir: str


# -------------------------
# Helpers
# -------------------------

def _registry_root() -> str:
    return os.getenv("OMNIBIOAI_MODEL_REGISTRY_ROOT", "").strip()


def _http_error(status: int, msg: str):
    raise HTTPException(status_code=status, detail=msg)


def _handle_registry_error(e: Exception):
    if isinstance(e, ModelRegistryError):
        _http_error(400, str(e))
    _http_error(500, str(e))


# -------------------------
# Endpoints
# -------------------------

@app.get("/health")
def health():
    root = _registry_root()
    return {
        "ok": True,
        "service": "omnibioai-model-registry",
        "version": APP_VERSION,
        "registry_root_configured": bool(root),
        "registry_root": root or None,
    }


@app.post(f"{DEFAULT_PREFIX}/register", response_model=RegisterResponse)
def api_register(req: RegisterRequest):
    try:
        out = register_model(
            task=req.task,
            model_name=req.model_name,
            version=req.version,
            artifacts_dir=req.artifacts_dir,
            metadata=req.metadata,
            set_alias=req.set_alias,
            actor=req.actor,
            reason=req.reason,
        )
        return RegisterResponse(
            ok=True,
            task=out["task"],
            model_name=out["model_name"],
            version=out["version"],
            package_path=out["package_path"],
            alias_set=out.get("alias_set"),
        )
    except Exception as e:
        _handle_registry_error(e)


@app.get(f"{DEFAULT_PREFIX}/resolve", response_model=ResolveResponse)
def api_resolve(
    task: str = Query(...),
    ref: str = Query(..., description="model@alias_or_version"),
    verify: bool = Query(True),
):
    try:
        path = resolve_model(task=task, model_ref=ref, verify=verify)
        return ResolveResponse(ok=True, path=path)
    except Exception as e:
        _handle_registry_error(e)


@app.post(f"{DEFAULT_PREFIX}/promote")
def api_promote(req: PromoteRequest):
    try:
        promote_model(
            task=req.task,
            model_name=req.model_name,
            alias=req.alias,
            version=req.version,
            actor=req.actor,
            reason=req.reason,
        )
        return {"ok": True}
    except Exception as e:
        _handle_registry_error(e)


@app.post(f"{DEFAULT_PREFIX}/verify", response_model=VerifyResponse)
def api_verify(req: VerifyRequest):
    try:
        verify_model_ref(task=req.task, model_ref=req.ref)
        return VerifyResponse(ok=True)
    except Exception as e:
        _handle_registry_error(e)


@app.get(f"{DEFAULT_PREFIX}/show", response_model=ShowResponse)
def api_show(
    task: str = Query(...),
    ref: str = Query(..., description="model@alias_or_version"),
    verify: bool = Query(False),
):
    try:
        registry = ModelRegistry.from_env()
        vdir = registry.resolve_model(task=task, model_ref=ref, verify=verify)
        meta_path = Path(vdir) / "model_meta.json"
        if not meta_path.exists():
            _http_error(404, f"model_meta.json not found in {vdir}")

        meta = json.loads(meta_path.read_text())
        return ShowResponse(ok=True, meta=meta, package_dir=str(vdir))
    except Exception as e:
        _handle_registry_error(e)
