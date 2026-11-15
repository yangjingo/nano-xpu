from multiprocessing.shared_memory import SharedMemory

import pytest

from nanoxpu.server.cli import parse_config
from nanoxpu.engine.identity import new_identity, shared_memory_name
from nanoxpu.server.config import ServerConfig
from nanoxpu.server.engine import NanoEngine


def test_two_instances_and_restarts_have_independent_shared_memory():
    names = [shared_memory_name("cluster-a", "generation1"),
             shared_memory_name("cluster-b", "generation1"),
             shared_memory_name("cluster-a", "generation2")]
    assert len(set(names)) == 3
    regions = []
    try:
        for index, name in enumerate(names):
            region = SharedMemory(name=name, create=True, size=16)
            regions.append(region)
            region.buf[0] = index
        for index, name in enumerate(names):
            attached = SharedMemory(name=name)
            try:
                assert attached.buf[0] == index
            finally:
                attached.close()
        regions[0].close()
        regions[0].unlink()
        regions.pop(0)
        assert regions[0].buf[0] == 1
    finally:
        for region in regions:
            region.close()
            region.unlink()


def test_default_instance_and_generation_are_new_per_config(tmp_path):
    first, second = ServerConfig(str(tmp_path)), ServerConfig(str(tmp_path))
    assert first.instance_id != second.instance_id
    assert first.engine_generation != second.engine_generation


def test_cli_forwards_explicit_identity_and_advertise_address(tmp_path):
    config = parse_config([str(tmp_path), "--instance-id", "worker-a",
                           "--engine-generation", "generation-1", "--advertise-ip", "127.0.0.1"])
    assert config.instance_id == "worker-a"
    assert config.engine_generation == "generation-1"
    assert config.advertise_ip == "127.0.0.1"


@pytest.mark.parametrize("value", ["", "../escape", "space bad", "a" * 161])
def test_invalid_identity_rejected(value):
    with pytest.raises(ValueError):
        shared_memory_name(value, new_identity())


def test_production_pd_cannot_fall_back_to_full_generation(tmp_path):
    config = ServerConfig(str(tmp_path), role="prefill", kv_port=24000, kv_connector="nixl-pull")
    with pytest.raises(RuntimeError, match="validated device connector"):
        NanoEngine(config).start()


def test_colliding_worker_ports_rejected(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        ServerConfig(str(tmp_path), port=2333)
