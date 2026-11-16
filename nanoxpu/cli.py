"""Public configuration-driven CLI; model imports stay inside the worker entry point."""
import argparse
import json
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv == ["runtime-info"]:
        from nanoxpu.runtime_identity import runtime_identity
        print(json.dumps(runtime_identity(), sort_keys=True))
        return
    if argv and argv[0] == "_worker":
        from nanoxpu.server.cli import main as worker_main
        return worker_main(argv[1:])
    if argv and argv[0] == "serve":
        try:
            from nanoxpu.deployment.cli import main as serve_main
        except ImportError as exc:
            raise SystemExit("Serving control dependencies are missing; install nano-xpu[deploy].") from exc
        return serve_main(argv[1:])
    parser = argparse.ArgumentParser(prog="nano-xpu")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="Start from -f cluster.yaml, or plan/status/logs/down")
    commands.add_parser("runtime-info", help="Report actual imported package provenance without loading a model")
    if argv and argv[0] == "deploy":
        parser.error("deploy was removed: use nano-xpu serve -f CONFIG to start, or nano-xpu serve plan/status/logs/down")
    parser.parse_args(argv)
