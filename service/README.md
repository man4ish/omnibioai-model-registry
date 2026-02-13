# OmniBioAI Model Registry – REST Service

This directory contains the **minimal REST wrapper** for the OmniBioAI Model Registry.

It exposes the registry over HTTP while preserving:

* Immutable version storage
* Integrity verification (SHA256)
* Alias promotion workflow
* Storage abstraction
* Scientific provenance tracking

The REST service is a **thin wrapper** over the core `omnibioai_model_registry` Python library.

> No business logic exists in this layer.
> All lifecycle logic lives in the core library.

---

# Architecture

```
Client (Plugin / TES / UI)
        │
        ▼
FastAPI REST Service
        │
        ▼
omnibioai_model_registry (Core Library)
        │
        ▼
Filesystem / Object Store
```

This ensures:

* CLI and REST share identical behavior
* No duplication of logic
* Deterministic model resolution

---

# Environment Variable

The service requires:

```
OMNIBIOAI_MODEL_REGISTRY_ROOT
```

Example (local):

```bash
export OMNIBIOAI_MODEL_REGISTRY_ROOT=~/Desktop/machine/local_registry/model_registry
```

Inside Docker, this must point to a mounted path.

---

# 5) Run Locally (No Docker, No Compose)

```bash
export OMNIBIOAI_MODEL_REGISTRY_ROOT=~/Desktop/machine/local_registry/model_registry

cd ~/Desktop/machine/omnibioai-model-registry
pip install -e .
pip install -r service/requirements.txt

uvicorn service.app.main:app --host 0.0.0.0 --port 8095
```

### Test

```bash
curl -s http://127.0.0.1:8095/health | python -m json.tool
```

You should see:

```json
{
  "ok": true,
  "service": "omnibioai-model-registry",
  "version": "0.1.0",
  "registry_root_configured": true
}
```

---

# 6) Run With Docker (Recommended)

### Important Concept

`artifacts_dir` must be visible **inside the container**.

We use two mounts:

1. Registry root (persistent storage)
2. A shared “staging” directory where training outputs are written

---

## Prepare Directories

```bash
mkdir -p ~/Desktop/machine/local_registry/model_registry
mkdir -p ~/Desktop/machine/local_registry/staging
```

---

## Build

```bash
docker build -f service/Dockerfile -t omnibioai-model-registry-svc .
```

---

## Run

```bash
docker run --rm -p 8095:8095 \
  -e OMNIBIOAI_MODEL_REGISTRY_ROOT=/data/model_registry \
  -v ~/Desktop/machine/local_registry/model_registry:/data/model_registry \
  -v ~/Desktop/machine/local_registry/staging:/shared \
  omnibioai-model-registry-svc
```

Now:

* `/data/model_registry` → registry storage
* `/shared` → staging directory

Your API `artifacts_dir` must reference something like:

```
"/shared/model_pkg_001"
```

---

# 7) Example REST Calls

---

## Register

(Assumes you created `/shared/model_pkg` on host under staging mount)

```bash
curl -s -X POST http://127.0.0.1:8095/v1/register \
  -H "Content-Type: application/json" \
  -d '{
    "task":"celltype_classification_sc",
    "model_name":"human_pbmc",
    "version":"2026-02-13_001",
    "artifacts_dir":"/shared/model_pkg",
    "metadata":{"framework":"pytorch","model_type":"mlp"},
    "set_alias":"latest",
    "actor":"manish",
    "reason":"api smoke test"
  }' | python -m json.tool
```

---

## Resolve

```bash
curl -s "http://127.0.0.1:8095/v1/resolve?task=celltype_classification_sc&ref=human_pbmc@latest&verify=true" \
| python -m json.tool
```

---

## Show Metadata

```bash
curl -s "http://127.0.0.1:8095/v1/show?task=celltype_classification_sc&ref=human_pbmc@latest&verify=true" \
| python -m json.tool
```

---

## Promote

```bash
curl -s -X POST http://127.0.0.1:8095/v1/promote \
  -H "Content-Type: application/json" \
  -d '{
    "task":"celltype_classification_sc",
    "model_name":"human_pbmc",
    "alias":"production",
    "version":"2026-02-13_001",
    "actor":"manish",
    "reason":"validated"
  }' | python -m json.tool
```

---

## Verify

```bash
curl -s -X POST http://127.0.0.1:8095/v1/verify \
  -H "Content-Type: application/json" \
  -d '{"task":"celltype_classification_sc","ref":"human_pbmc@production"}' \
| python -m json.tool
```

---

# Security Model (v0.1)

This minimal service:

* Has no authentication
* Has no RBAC
* Assumes trusted internal network
* Is intended for controlled environments

Authentication and RBAC will be introduced in later versions.

