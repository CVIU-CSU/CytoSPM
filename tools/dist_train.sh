#!/usr/bin/env bash
# Usage: bash tools/dist_train.sh CONFIG GPUS [train.py args...]
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 CONFIG GPUS [train.py args...]" >&2
    exit 2
fi

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
exec torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    tools/train.py "$CONFIG" --launcher pytorch "${@:3}"
