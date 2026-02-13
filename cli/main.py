# File: omnibioai_model_registry/cli/main.py
import argparse
import json
import sys
from pathlib import Path

from omnibioai_model_registry import (
    ModelRegistry,
    register_model,
    resolve_model,
    promote_model,
    verify_model_ref,
)
from omnibioai_model_registry.errors import ModelRegistryError


def cmd_list(args):
    registry = ModelRegistry.from_env()
    root = registry.root

    task_root = root / "tasks" / args.task / "models"
    if not task_root.exists():
        print(f"No models found for task '{args.task}'")
        return

    for model_dir in sorted(task_root.iterdir()):
        if model_dir.is_dir():
            print(model_dir.name)


def cmd_resolve(args):
    path = resolve_model(args.task, args.model_ref)
    print(path)


def cmd_promote(args):
    promote_model(
        task=args.task,
        model_name=args.model,
        alias=args.alias,
        version=args.version,
        actor=args.actor,
        reason=args.reason,
    )
    print(f"Promoted {args.model}@{args.version} → {args.alias}")


def cmd_verify(args):
    verify_model_ref(args.task, args.model_ref)
    print("Integrity verification passed.")


def _read_json_file(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    return json.loads(p.read_text())


def cmd_register(args):
    meta = {}
    if args.metadata_json:
        meta.update(_read_json_file(args.metadata_json))
    if args.metadata_inline:
        meta.update(json.loads(args.metadata_inline))

    out = register_model(
        task=args.task,
        model_name=args.model,
        version=args.version,
        artifacts_dir=args.artifacts,
        metadata=meta,
        set_alias=args.set_alias,
        actor=args.actor,
        reason=args.reason,
    )
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"Registered: {out['task']}/{out['model_name']}/{out['version']}")
        print(f"Path: {out['package_path']}")
        if out.get("alias_set"):
            print(f"Alias set: {out['alias_set']}")


def cmd_show(args):
    registry = ModelRegistry.from_env()
    vdir = registry.resolve_model(task=args.task, model_ref=args.model_ref, verify=args.verify)

    meta_path = vdir / "model_meta.json"
    if not meta_path.exists():
        print(f"model_meta.json not found in: {vdir}", file=sys.stderr)
        sys.exit(1)

    txt = meta_path.read_text()
    meta = json.loads(txt)

    if args.raw:
        print(txt, end="" if txt.endswith("\n") else "\n")
        return

    if args.json:
        print(json.dumps(meta, indent=2))
        return

    # Pretty minimal human view
    print(f"Task:       {meta.get('task', args.task)}")
    print(f"Model:      {meta.get('model_name')}")
    print(f"Version:    {meta.get('version')}")
    print(f"Created:    {meta.get('created_at')}")
    print(f"Framework:  {meta.get('framework')}")
    print(f"Model type: {meta.get('model_type')}")

    prov = meta.get("provenance") or {}
    if prov:
        print("Provenance:")
        if prov.get("git_commit"):
            print(f"  git_commit: {prov.get('git_commit')}")
        if prov.get("training_data_ref"):
            print(f"  training_data_ref: {prov.get('training_data_ref')}")
        if prov.get("trainer_version"):
            print(f"  trainer_version: {prov.get('trainer_version')}")

    print(f"Package dir: {vdir}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="omr",
        description="OmniBioAI Model Registry CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List models for a task")
    p_list.add_argument("--task", required=True, help="Task name")
    p_list.set_defaults(func=cmd_list)

    # resolve
    p_resolve = subparsers.add_parser("resolve", help="Resolve model reference to a local path")
    p_resolve.add_argument("--task", required=True, help="Task name")
    p_resolve.add_argument("--ref", dest="model_ref", required=True, help="Model reference (model@alias_or_version)")
    p_resolve.set_defaults(func=cmd_resolve)

    # show
    p_show = subparsers.add_parser("show", help="Show model metadata (model_meta.json)")
    p_show.add_argument("--task", required=True, help="Task name")
    p_show.add_argument("--ref", dest="model_ref", required=True, help="Model reference (model@alias_or_version)")
    p_show.add_argument("--verify", action="store_true", help="Verify package integrity before showing")
    p_show.add_argument("--json", action="store_true", help="Print formatted JSON metadata")
    p_show.add_argument("--raw", action="store_true", help="Print raw model_meta.json")
    p_show.set_defaults(func=cmd_show)

    # register
    p_register = subparsers.add_parser("register", help="Register a model package into the registry")
    p_register.add_argument("--task", required=True, help="Task name")
    p_register.add_argument("--model", required=True, help="Model name")
    p_register.add_argument("--version", required=True, help="Version string (immutable)")
    p_register.add_argument("--artifacts", required=True, help="Directory containing model package artifacts")
    p_register.add_argument(
        "--set-alias",
        dest="set_alias",
        default="latest",
        help="Alias to set after register (default: latest). Use '' to skip.",
    )
    p_register.add_argument("--actor", default=None, help="Actor registering the model")
    p_register.add_argument("--reason", default="cli register", help="Reason for registration")
    p_register.add_argument("--metadata-json", default=None, help="Path to JSON file with metadata to merge")
    p_register.add_argument("--metadata-inline", default=None, help="Inline JSON string metadata to merge")
    p_register.add_argument("--json", action="store_true", help="Print JSON result")
    p_register.set_defaults(func=cmd_register)

    # promote
    p_promote = subparsers.add_parser("promote", help="Promote model version to an alias")
    p_promote.add_argument("--task", required=True, help="Task name")
    p_promote.add_argument("--model", required=True, help="Model name")
    p_promote.add_argument("--version", required=True, help="Version string")
    p_promote.add_argument("--alias", required=True, help="Alias name (e.g. production)")
    p_promote.add_argument("--actor", default=None, help="Actor performing promotion")
    p_promote.add_argument("--reason", default=None, help="Reason for promotion")
    p_promote.set_defaults(func=cmd_promote)

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify model integrity")
    p_verify.add_argument("--task", required=True, help="Task name")
    p_verify.add_argument("--ref", dest="model_ref", required=True, help="Model reference")
    p_verify.set_defaults(func=cmd_verify)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Handle special case: user wants no alias set
    if getattr(args, "set_alias", None) == "":
        args.set_alias = None

    try:
        args.func(args)
    except ModelRegistryError as e:
        print(f"[Registry Error] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Unexpected Error] {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
