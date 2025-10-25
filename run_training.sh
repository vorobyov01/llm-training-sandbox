#!/bin/bash

# Simple training script for DDP demo
# Usage: ./run_training.sh [num_gpus]

NUM_GPUS=${1:-2}

echo "Starting DDP training on $NUM_GPUS GPUs..."

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    train.py \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_epochs 1 \
    --max_length 512 \
    --profile \
    --memory_profile

echo "Training completed!"
