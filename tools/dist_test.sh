#!/usr/bin/env bash
# Usage: bash tools/dist_test.sh CONFIG CHECKPOINT GPUS [test.py args...]
# Set SCRIPT=tools/test_novel.py to evaluate the novel split.
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 CONFIG CHECKPOINT GPUS [test.py args...]" >&2
    exit 2
fi

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
SCRIPT=${SCRIPT:-tools/test.py}
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
    "$SCRIPT" "$CONFIG" "$CHECKPOINT" --launcher pytorch "${@:4}"
