"""Start an explicitly test-only HTTP server for CPU smoke verification."""
from nanoxpu.server.cli import parse_config
from nanoxpu.server.app import create_app
from tests.helpers.support import TestEngine
from nanoxpu.server.runtime import ManagedServer
import sys


def main():
    if sys.argv[1:] == ["runtime-info"]:
        from nanoxpu.cli import main as cli_main
        return cli_main(["runtime-info"])
    if sys.argv[1:2] != ["_worker"]:
        raise SystemExit("test harness requires the internal _worker entry")
    config = parse_config(sys.argv[2:])
    app = create_app(config, TestEngine)
    server = ManagedServer(app, config)

    @app.post("/__test__/shutdown")
    async def shutdown():
        server.request_shutdown()
        return {"status": "stopping_test_server"}

    print("CPU TEST SERVER: synthetic engine; no Qwen model loaded", flush=True)
    server.run()
    if not server.started or app.state.service.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
