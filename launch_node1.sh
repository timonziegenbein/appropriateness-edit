#!/bin/bash

# Change to the script's directory (project root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Add project root to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Job Isolation for Node 1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29600

# Set unique job identifier to prevent any cross-job confusion
export TORCHELASTIC_RUN_ID="node1_$(date +%s)"
export NCCL_ASYNC_ERROR_HANDLING=1  # Enable better error reporting

# NCCL isolation - prevent cross-job discovery and shared files
export NCCL_SOCKET_IFNAME=lo  # Use loopback interface only
export NCCL_IB_DISABLE=1       # Disable InfiniBand to prevent cross-node communication
export NCCL_P2P_DISABLE=1      # Disable peer-to-peer to prevent cross-job discovery
export NCCL_DEBUG=INFO         # Enable debug logging to see what NCCL is doing
export NCCL_SHM_DISABLE=1      # Disable shared memory to prevent file conflicts on NFS

# Cache directories (local temp to avoid NFS conflicts)
# Only override write-heavy caches that cause conflicts
export TMPDIR=/tmp/job_node1  # Force ALL temp files to local storage
export TRITON_CACHE_DIR=/tmp/triton_node1
export VLLM_CACHE_DIR=/tmp/vllm_cache_node1
export TORCH_COMPILE_CACHE_DIR=/tmp/torch_compile_cache_node1
export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_compile_cache_node1  # Another torch.compile variable
export WANDB_DIR=/tmp/wandb_node1  # WandB directory to avoid NFS conflicts

# Override XDG_CACHE_HOME to force vLLM to use local cache
export XDG_CACHE_HOME=/tmp/xdg_cache_node1

# HuggingFace authentication - use token from environment
# IMPORTANT: Export HF_TOKEN in your shell BEFORE running this script to avoid NFS conflicts
# Example: export HF_TOKEN=$(cat ~/.cache/huggingface/token)
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set! Please run: export HF_TOKEN=\$(cat ~/.cache/huggingface/token)"
    echo "Attempting to read token file (may cause conflicts if both jobs start simultaneously)..."
    if [ -f "$HOME/.cache/huggingface/token" ]; then
        export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
        echo "Loaded HuggingFace token from ~/.cache/huggingface/token"
    elif [ -f "$HOME/.huggingface/token" ]; then
        export HF_TOKEN=$(cat "$HOME/.huggingface/token")
        echo "Loaded HuggingFace token from ~/.huggingface/token"
    fi
else
    echo "Using HF_TOKEN from environment"
fi

# CRITICAL: HF_HOME must be on LOCAL storage to avoid NFS lock file conflicts
# This means models will be downloaded to each node separately, but prevents SIGBUS crashes
# You can symlink the hub directory after first download to save space if needed
export HF_HOME=/tmp/huggingface_node1

# Create temp directories
mkdir -p $TMPDIR $XDG_CACHE_HOME $HF_HOME

# Ray isolation
export RAY_ENABLE_AUTO_CONNECT=0
export RAY_PORT=0
export RAY_ADDRESS=""
export RAY_USAGE_STATS_ENABLED=0

# vLLM isolation
export VLLM_INSTANCE_ID=node1
export VLLM_RAY_SESSION_DIR=/tmp/vllm_node1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=========================================="
echo "Launching Node 1 Training Job"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Cache dirs: $TRITON_CACHE_DIR"
echo "Log file: node1_training.log"
echo "=========================================="

# Redirect all output to log file (both stdout and stderr)
# Use config file but override main_process_port to ensure isolation
accelerate launch \
  --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
  --main_process_port $MASTER_PORT \
  models/grpo.py \
  --output_dir models/trained/grpo_global_sentence_v11 \
  --use_fluency \
  --use_human_like \
  --use_semantic_similarity \
  --disable_eval_on_start
