"""Synthetic HTTP worker for real upstream-router contracts; no model or KV transport claim."""
import json
from pathlib import Path
import sys
import time
import uuid

from nanoxpu.server.cli import parse_config
from nanoxpu.kv.leases import LeaseCapacityError
from nanoxpu.server.app import create_app
from nanoxpu.server.engine import Completion
from nanoxpu.server.metadata import Capabilities, Fingerprints, model_fingerprints
from nanoxpu.server.runtime import ManagedServer
from nanoxpu.server.schema import PrefillRequest
from tests.helpers.support import TestEngine


class RouterContractEngine(TestEngine):
    def start(self):
        super().start()
        config = self.config
        pd = config.role != "colocated"
        self.capabilities = Capabilities(fingerprints=Fingerprints(**model_fingerprints(config.model)),
                                         transport="cpu-test" if pd else "none", transport_version="cpu-contract-v1",
                                         connector_ready=pd, layout_digest="b" * 64 if pd else None)

    def complete(self, request):
        # Capture only synthetic test payloads. Credentials/headers are deliberately excluded.
        record = {"role": self.config.role, "request_id": getattr(request, "request_id", ""), "body": request.model_dump()}
        with (Path(self.config.model) / f"capture-{self.config.instance_id}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        content = request.messages[0].content
        if content == "BUSY" or (content == "BUSY_DECODE" and self.config.role == "decode"):
            raise LeaseCapacityError("injected busy worker")
        if content == "FAIL_PREFILL" and self.config.role == "prefill":
            raise RuntimeError("injected prefill failure")
        if content == "FAIL_DECODE" and self.config.role == "decode":
            raise RuntimeError("injected decode failure")
        if content == "SLOW":
            time.sleep(0.4)
        if isinstance(request, PrefillRequest):
            params = {"do_remote_decode": False, "do_remote_prefill": True, "remote_engine_id": self.config.instance_id,
                      "remote_block_ids": [5] if len(content) > 1 else [], "remote_host": self.config.advertise_ip,
                      "remote_port": self.config.kv_port,
                      "nano": {"protocol_version": 1, "lease_id": uuid.uuid4().hex, "generation": self.config.engine_generation,
                               "expires_at": time.time() + 60, "cached_tokens": len(content) - 1, "descriptor_digest": "a" * 64}}
            # A capture of exactly what P exports lets the real-router test compare D's input.
            (Path(self.config.model) / "exported.json").write_text(json.dumps(params), encoding="utf-8")
            return Completion("INTERNAL PREFILL OUTPUT", len(content), 1, "length", params)
        return Completion(f"CPU CONTRACT {self.config.instance_id}", len(content), request.max_tokens, "length")


def main():
    if sys.argv[1:] == ["runtime-info"]:
        from nanoxpu.cli import main as cli_main
        return cli_main(["runtime-info"])
    if sys.argv[1:2] != ["_worker"]:
        raise SystemExit("test harness requires the internal _worker entry")
    config = parse_config(sys.argv[2:])
    app = create_app(config, RouterContractEngine)
    server = ManagedServer(app, config)
    print("CPU CONTRACT WORKER: synthetic responses, no Qwen or NIXL", flush=True)
    server.run()
    if not server.started or app.state.service.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
