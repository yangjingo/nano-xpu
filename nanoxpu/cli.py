import argparse

from nanoxpu.server.config import ServerConfig


def parse_config(argv=None):
    parser = argparse.ArgumentParser(prog="nano-xpu")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve a local Qwen model using the CUDA engine")
    serve.add_argument("model", help="Local model directory")
    serve.add_argument("--served-model-name")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--tensor-parallel-size", type=int, default=1)
    serve.add_argument("--distributed-port", type=int, default=2333)
    serve.add_argument("--max-model-len", type=int, default=4096)
    serve.add_argument("--max-num-batched-tokens", type=int, default=16384)
    serve.add_argument("--max-num-seqs", type=int, default=512)
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    serve.add_argument("--enforce-eager", action="store_true")
    values = vars(parser.parse_args(argv))
    values.pop("command")
    try:
        return ServerConfig(**values)
    except ValueError as exc:
        parser.error(str(exc))


def main(argv=None):
    config = parse_config(argv)
    try:
        from nanoxpu.server.app import create_app
        from nanoxpu.server.runtime import run_server
    except ImportError as exc:
        raise SystemExit("Server dependencies are missing; install nano-xpu[server].") from exc
    from nanoxpu.server.engine import NanoEngine

    # A single web worker owns the engine; no reload or process pool.
    app = create_app(config, NanoEngine)
    run_server(config, app)
