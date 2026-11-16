"""Deployment CLI; importing the worker/model stack is unnecessary."""
import argparse
import json
import subprocess
import sys
import time

import httpx

from nanoxpu.deployment.planner import plan_file
from nanoxpu.deployment.controller import Controller


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="nano-xpu serve",
        description="Start router and workers from -f CONFIG. Direct model paths are no longer accepted.",
    )
    parser.add_argument("command", nargs="?", choices=("plan", "status", "logs", "down"),
                        help="Omit to start; plan validates without starting processes")
    parser.add_argument("-f", "--file", help="Cluster YAML; required for start and plan")
    parser.add_argument("--detach", action="store_true", help="Start in background after readiness")
    parser.add_argument("--name", help="Existing deployment name for status/logs/down")
    parser.add_argument("--state-dir", help="Ledger directory for start/status/logs/down")
    parser.add_argument("--role", help="Filter logs by role type or worker ID")
    argv = sys.argv[1:] if argv is None else argv
    if argv and not argv[0].startswith("-") and argv[0] not in ("plan", "status", "logs", "down"):
        parser.error("direct model paths and old start commands are unsupported; use nano-xpu serve -f cluster.yaml")
    args = parser.parse_args(argv)
    args.command = args.command or "up"  # Controller method name, not a second public start command.
    if args.command in ("up", "plan"):
        if not args.file:
            parser.error("-f/--file is required; use nano-xpu serve -f cluster.yaml")
        if args.name or args.role:
            parser.error("--name and --role are not valid for start or plan")
        if args.command == "plan" and (args.detach or args.state_dir):
            parser.error("plan does not accept --detach or --state-dir")
    else:
        if not args.name:
            parser.error("--name is required for status/logs/down")
        if args.file or args.detach or (args.role and args.command != "logs"):
            parser.error("management commands do not accept -f/--detach; --role is only valid for logs")
    controller = None
    try:
        if args.command == "plan":
            result = plan_file(args.file).public_dict()
        elif args.command == "up":
            controller = Controller(plan_file(args.file), root=args.state_dir)
            result = controller.up()
        else:
            controller = Controller.load(args.name, root=args.state_dir)
            result = controller.logs(args.role) if args.command == "logs" else getattr(controller, args.command)()
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        if args.command == "status" and result["phase"] not in ("RUNNING", "STOPPED"):
            raise SystemExit(1)
        if args.command == "up" and not args.detach:
            try:
                while True:
                    time.sleep(1)
                    status = controller.status()
                    if not status["ready"]:
                        controller.down()
                        raise RuntimeError("cluster became degraded; recorded roles stopped")
            except KeyboardInterrupt:
                print(json.dumps(controller.down(), indent=2), flush=True)
    except ValueError as exc:
        parser.error(str(exc))
    except OSError as exc:
        if args.command == "plan":
            parser.error(str(exc))
        parser.exit(1, f"deployment failed: {exc}\n")
    except (RuntimeError, httpx.HTTPError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"deployment failed: {exc}\n")
    finally:
        if controller is not None:
            controller.close()
