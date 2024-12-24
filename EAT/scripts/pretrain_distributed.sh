#!/bin/bash

# Get the arguments for pretrain.sh
ARGS="$@"

# Extract CUDA_DEVICES from arguments
CUDA_DEVICES="0"
for arg in "$@"; do
    if [[ $prev_arg == "--cuda" ]]; then
        CUDA_DEVICES=$arg
    fi
    prev_arg=$arg
done

# Count GPUs
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}

# Launch distributed training
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=12345 \
    $(dirname "$0")/pretrain.sh $ARGS
