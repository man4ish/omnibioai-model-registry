import argparse
import json
import sys
from pathlib import Path

from omnibioai_model_registry import (
    ModelRegistry,
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
    p_resolve = subparsers.add_parser("resolve", help="Resolve model reference")
    p_resolve.add_argument("--task", required=True, help="Task name")
    p_resolve.add_argument("--ref", dest="model_ref", required=True, help="Model reference (model@alias_or_version)")
    p_resolve.set_defaults(func=cmd_resolve)

    # promote
    p_promote = subparsers.add_parser("promote", help="Promote model version to alias")
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
