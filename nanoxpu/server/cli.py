"""Internal worker entry point, invoked by the deployment supervisor."""
import argparse

from nanoxpu.server.config import ServerConfig


def parse_config(argv=None):
    parser = argparse.ArgumentParser(prog="nano-xpu _worker", description="Internal worker; use nano-xpu serve -f for deployment.")
    parser.add_argument("model", help="Local model directory")
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--distributed-port", type=int, default=2333)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--role", choices=("colocated", "prefill", "decode"), default="colocated")
    parser.add_argument("--instance-id", default=argparse.SUPPRESS)
    parser.add_argument("--engine-generation", default=argparse.SUPPRESS)
    parser.add_argument("--advertise-ip")
    parser.add_argument("--kv-port", type=int)
    parser.add_argument("--kv-connector", choices=("nixl-pull",))
    parser.add_argument("--cluster-id")
    parser.add_argument("--prefill-peer", dest="prefill_peers", action="append", default=[])
    values = vars(parser.parse_args(argv))
    values["prefill_peers"] = tuple(values["prefill_peers"])
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
