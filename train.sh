#!/usr/bin/env bash

set -e

# Dataset configuration
DATASET_REPO_ID="local/so101_pick_cube"
DATASET_ROOT="./data/so101_pick_cube"

# Training configuration
OUTPUT_DIR="./outputs/train/act_so101_pick_cube"
JOB_NAME="act_so101_pick_cube"

# Change this depending on your hardware:
# cuda = NVIDIA GPU
# mps  = Apple Silicon
# cpu  = CPU (very slow)
DEVICE="${DEVICE:-cpu}"

uv run lerobot-train \
    --dataset.repo_id local/so101_pick_cube \
    --dataset.root ./data/so101_pick_cube \
    --policy.type act \
    --policy.device mps \
    --policy.push_to_hub false \
    --output_dir ./outputs/train/act_so101_pick_cube \
    --job_name act_so101_pick_cube \
    --batch_size 8 \
    --steps 20000 \
    --num_workers 0

echo "Training finished!"
echo "Model checkpoints: ${OUTPUT_DIR}/checkpoints"
