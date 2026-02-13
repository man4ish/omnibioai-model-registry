# OmniBioAI Model Registry

**OmniBioAI Model Registry** is a **production-grade lifecycle management system for AI/ML models** within the **OmniBioAI ecosystem**.

It provides:

* **Immutable model versioning**
* **Cryptographic integrity verification (SHA256)**
* **Full scientific provenance capture**
* **Staged promotion workflows (staging → production)**
* **Reproducible deployment across local, HPC, and cloud environments**
* **Storage-agnostic architecture (local → S3 → Azure → on-prem)**

The registry is implemented as a **standalone Python library** today and is designed to evolve into an **enterprise-scale model registry service** for biomedical AI.

---

# Why This Exists

Biomedical AI requires:

* **Reproducibility**
* **Auditability**
* **Governance**
* **Offline / air-gapped deployment**
* **Cross-infrastructure execution parity**

Traditional ML tooling often assumes:

* Cloud-first infrastructure
* Mutable artifacts
* Weak provenance guarantees

**OmniBioAI Model Registry is designed differently.**

> It treats AI models as **scientific artifacts** that must be
> **immutable, verifiable, and reproducible across environments**.

---

# Role in the OmniBioAI Architecture

OmniBioAI follows a **four-plane architecture**:

| Plane             | Responsibility                         |
| ----------------- | -------------------------------------- |
| **Control Plane** | UI, registries, metadata, governance   |
| **Compute Plane** | Workflow execution, HPC/cloud adapters |
| **Data Plane**    | Artifacts, datasets, outputs           |
| **AI Plane**      | Reasoning, RAG, agents, interpretation |

The **Model Registry** belongs to the **Control Plane** and provides:

* AI artifact governance
* Deterministic inference references
* Promotion and audit workflows
* Infrastructure-independent model resolution

---

# Core Design Principles

## 1. Immutability

Each model version is **write-once**:

* No overwrites
* No silent mutation
* Full historical trace

This guarantees **scientific reproducibility**.

---

## 2. Integrity Verification

Every model package includes:

```
sha256sums.txt
```

This enables:

* Bit-level reproducibility
* Tamper detection
* Trustworthy deployment in regulated environments

---

## 3. Provenance Tracking

Each model records:

* Training code version (git commit)
* Dataset reference (e.g., DVC / object store)
* Hyperparameters and preprocessing
* Creator and timestamp

This provides **audit-ready scientific lineage**.

---

## 4. Promotion Workflow

Models move through controlled stages:

```
latest → staging → production
```

All promotions are:

* **Explicit**
* **Append-only**
* **Audited**

This mirrors **regulated biomedical deployment pipelines**.

---

## 5. Storage Abstraction

The registry supports:

* Local filesystem (v1)
* HPC shared storage
* S3 / Azure Blob (planned)
* Offline enterprise environments

**No redesign required** when switching backends.

---

# Repository Structure

```
omnibioai-model-registry/
├── omnibioai_model_registry/
│   ├── api.py
│   ├── config.py
│   ├── refs.py
│   ├── storage/
│   ├── package/
│   ├── audit/
│   └── cli/
├── tests/
├── pyproject.toml
└── README.md
```

---

# Canonical Model Package Layout

Registered models follow a strict, portable structure:

```
model_registry/
  tasks/<task>/models/<model_name>/
    versions/<version>/
      model.pt
      model_genes.txt
      label_map.json
      model_meta.json
      metrics.json
      feature_schema.json
      sha256sums.txt
    aliases/
      latest.json
      staging.json
      production.json
    audit/
      promotions.jsonl
```

This guarantees:

* Deterministic loading
* Integrity validation
* Cross-environment portability

---

# Core API Concepts

OmniBioAI components interact with the registry through **three operations**:

### Register

Stores a trained model **immutably** with metadata and integrity hashes.

```
register_model(...)
```

---

### Resolve

Retrieves a model by **stable reference**:

```
human_pbmc@production
```

```
resolve_model(...)
```

---

### Promote

Moves a version into **staging or production** with audit logging.

```
promote_model(...)
```

---

# Local Development Setup

## 1. Clone ecosystem

```
cd ~/Desktop/machine
git clone https://github.com/man4ish/omnibioai-model-registry
```

## 2. Install in editable mode

```
pip install -e ./omnibioai-model-registry
```

## 3. Configure registry root

```
export OMNIBIOAI_MODEL_REGISTRY_ROOT=~/Desktop/machine/local_registry/model_registry
```

---

# Minimal Usage Example

```python
from omnibioai_model_registry import register_model, resolve_model

register_model(
    task="celltype_classification_sc",
    model_name="human_pbmc",
    version="2026-02-13_001",
    artifacts_dir="/tmp/model_pkg",
    metadata={"framework": "pytorch"},
)

path = resolve_model("celltype_classification_sc", "human_pbmc@latest")
print(path)
```

---

# Relationship to OmniBioAI Ecosystem

This registry is a **core control-plane component** of OmniBioAI:

[https://github.com/man4ish](https://github.com/man4ish)

Key companion repositories:

* **omnibioai** → AI-powered bioinformatics workbench
* **omnibioai-tes** → execution orchestration across local/HPC/cloud
* **omnibioai-rag** → reasoning and literature intelligence
* **omnibioai-lims** → laboratory data management
* **omnibioai-workflow-bundles** → reproducible pipelines
* **omnibioai-sdk** → Python client access

The **Model Registry** provides the **AI artifact governance layer** shared by all.

---

# Roadmap

## Near Term

* S3 / Azure storage backends
* CLI tooling (`omr`)
* Extended metadata validation

## Mid Term

* REST API registry service
* RBAC and governance controls
* Model search and comparison

## Long Term

* Enterprise biomedical AI governance platform
* Regulatory-ready audit and lineage
* Integration with experiment tracking and clinical pipelines

---

# Status

* Production-oriented architecture
* Immutable and verifiable model storage
* Audit-ready promotion workflow
* Local-first, cloud-ready design

**OmniBioAI Model Registry establishes the foundation for
trustworthy, reproducible biomedical AI deployment.**
