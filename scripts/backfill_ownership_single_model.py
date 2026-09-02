#!/usr/bin/env python3
"""
Single-model equivalent of ownership.backfill_legacy_ownership().

WHY THIS EXISTS
----------------
backfill_legacy_ownership(registry_root) (omnibioai_model_registry/ownership.py)
has no per-model scope: it always walks every task/model_name directory
under the registry root and writes a status="legacy_unowned"
ownership.json for each one that doesn't already have a record. There is
also no --task/--model flag on the `omr migrate-ownership` CLI command
that wraps it (cli/main.py:103-124) -- it's a whole-registry, one-time
migration by design.

This script performs the IDENTICAL write that backfill_legacy_ownership()
performs for ONE model -- same function (ensure_model_ownership with
model_pre_existing=True), same resulting ownership.json shape -- without
touching any other model directory. It is the single-model canary this
codebase's real migration tooling doesn't offer.

It does NOT reimplement or bypass ownership logic: it imports and calls
the same ensure_model_ownership() that both backfill_legacy_ownership()
and the normal register_model() path use.

After this script runs (status becomes "legacy_unowned"), assigning real
ownership is a SEPARATE, already-scoped step -- use the existing CLI:

    omr resolve-ownership --task <task> --model <model> --org-id <id> --actor <actor>

(or POST /v1/ownership/resolve over HTTP, which requires the caller's own
verified org_id and the model.resolve_ownership permission -- see
ownership.py's resolve_legacy_ownership() docstring). This script never
calls resolve_legacy_ownership() itself.

SAFETY
------
- Read-only by default. Nothing is written unless --yes is passed.
- Refuses to run against a task/model that does not exist on disk.
- Refuses to run (even with --yes) if an ownership.json already exists
  for this model -- ensure_model_ownership() would itself no-op and
  return the existing record (write-once), but this script fails loudly
  instead so "nothing happened, as expected" isn't mistaken for success
  the first time it's run against an already-migrated model.
- Uses ModelRegistry.from_env() -- the exact same OMNIBIOAI_MODEL_REGISTRY_ROOT
  resolution the `omr` CLI and the running service use. It does not accept
  a registry-root override, so it always targets whatever registry the
  environment is actually configured for.

USAGE
-----
    # Preview only -- reads and reports current state, writes nothing:
    python scripts/backfill_ownership_single_model.py --task <task> --model <model>

    # Perform the write (creates ownership.json, status=legacy_unowned):
    python scripts/backfill_ownership_single_model.py --task <task> --model <model> --yes

Exit codes: 0 = success (or clean preview), 1 = refused/error.
"""
from __future__ import annotations

import argparse
import json
import sys

from omnibioai_model_registry.api import ModelRegistry
from omnibioai_model_registry.ownership import ensure_model_ownership, read_ownership
from omnibioai_model_registry.package import layout as L


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Task name (e.g. protein_stability_ddg)")
    parser.add_argument("--model", required=True, help="Model name (e.g. ddg_mlp)")
    parser.add_argument("--yes", action="store_true", help="Actually write ownership.json. Without this, preview only.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary")
    args = parser.parse_args()

    registry = ModelRegistry.from_env()
    root = registry.root

    model_dir = L.task_root(root, args.task) / "models" / args.model
    if not model_dir.is_dir():
        print(f"REFUSED: no such model directory: {model_dir}", file=sys.stderr)
        return 1

    existing = read_ownership(root, args.task, args.model)
    if existing is not None:
        print(
            f"REFUSED: {args.task}/{args.model} already has an ownership record "
            f"(status={existing.status!r}, organization_id={existing.organization_id!r}). "
            "Nothing to backfill -- ownership.json is write-once.",
            file=sys.stderr,
        )
        return 1

    if not args.yes:
        preview = {
            "task": args.task,
            "model_name": args.model,
            "action": "would_create_legacy_unowned",
            "model_dir": str(model_dir),
        }
        if args.json:
            print(json.dumps(preview, indent=2))
        else:
            print(f"DRY RUN (pass --yes to execute): {args.task}/{args.model}")
            print(f"  model_dir: {model_dir}")
            print("  would create ownership.json with status=legacy_unowned")
        return 0

    record = ensure_model_ownership(
        registry.backend,
        root,
        args.task,
        args.model,
        organization_id=None,
        actor=None,
        model_pre_existing=True,
    )

    result = {
        "task": args.task,
        "model_name": args.model,
        "status": record.status,
        "organization_id": record.organization_id,
        "discovered_at": record.discovered_at,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Migrated: {args.task}/{args.model}")
        print(f"Ownership: status={record.status} organization_id={record.organization_id}")
        print(
            "\nNext step (separate, already-scoped command): "
            f"omr resolve-ownership --task {args.task} --model {args.model} --org-id <id> --actor <you>"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
