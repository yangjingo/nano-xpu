#!/usr/bin/env bash
# All model/device settings belong to the cluster YAML and its runtime profile.
# Real CUDA/router verification is still required.
set -euo pipefail
: "${CLUSTER_CONFIG:?Set CLUSTER_CONFIG to a cluster YAML file}"
if [[ ! -f "$CLUSTER_CONFIG" ]]; then
  echo "Cluster configuration does not exist: $CLUSTER_CONFIG" >&2
  exit 1
fi
exec nano-xpu serve -f "$CLUSTER_CONFIG" "$@"
