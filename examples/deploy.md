# 部署配置示例

脚本与配置保存在以下命名代码块中，按 filename 保存后使用。两类拓扑都通过 `nano-xpu serve -f` 启动；单 worker 也经 router。CPU harness 只验证接口，生产 NIXL/P/D 尚未完成。运行要求见[部署说明](../docs/deployment.md)。

```yaml filename=qwen3-cuda-profile.yaml
# Deployment profile example. Prepare runtime/model/SSH and pin real revisions before starting.
schema_version: 1
model_path: /models/Qwen3-0.6B
served_model_name: qwen3-0.6b
runtime:
  nano_xpu_bin: /opt/nanoxpu/bin/nano-xpu
  workdir: /opt/nanoxpu
  expected_nano_xpu_revision: "<implementation-commit-to-be-pinned>"
  helper_python: /opt/nanoxpu/bin/python
ssh:
  user: inference
  port: 22
  identity_file: ~/.ssh/id_ed25519
  # Optional controller-local file; strict host-key checking is always enabled.
  known_hosts_file: ~/.ssh/known_hosts
router:
  executable: /opt/nanoxpu/bin/vllm-router
  source_revision: 07c20da66c255ba65a97ab531f9fb5f60413a7d4
  policy: round_robin
  port: 8000
engine:
  max_model_len: 4096
  max_num_batched_tokens: 2048
  max_num_seqs: 1
  gpu_memory_utilization: 0.9
  enforce_eager: true
kv:
  connector: nixl-pull
  expected_version: "<M2-spike-must-pin-nixl-and-network-backend>"
  lease_seconds: 180
timeouts:
  startup_seconds: 600
  request_seconds: 120
  drain_seconds: 30
```

```yaml filename=cluster-colocated.yaml
# serve plan can validate this example; replace profile placeholders before starting.
schema_version: 1
name: qwen-colocated
profile: ./qwen3-cuda-profile.yaml
mode: colocated
router:
  ip: 10.0.0.10
workers:
  - {id: c0, role: colocated, ip: 10.0.0.11, gpus: [0]}
  - {id: c1, role: colocated, ip: 10.0.0.12, gpus: [0]}
```

```yaml filename=cluster-pd.yaml
# serve plan validates this topology; production P/D backend remains unimplemented.
# Same host is also valid with distinct physical GPUs, e.g. P=[0], D=[1].
schema_version: 1
name: qwen-pd
profile: ./qwen3-cuda-profile.yaml
mode: disaggregated
router:
  ip: 10.0.0.10
workers:
  - {id: p0, role: prefill, ip: 10.0.0.11, gpus: [0]}
  - {id: d0, role: decode, ip: 10.0.0.12, gpus: [0]}
```
